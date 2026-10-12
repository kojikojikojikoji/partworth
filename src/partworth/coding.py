"""Attribute specifications and design-matrix coding.

An attribute is either :class:`Categorical` (a fixed set of levels) or :class:`Linear`
(a numeric value such as price). A :class:`Codebook` is the fitted version of a mapping
from attribute names to specifications: it knows every coded column and can map
coefficients back to one value per level for reporting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .errors import CodingError, SchemaError

NONE_COLUMN = "NONE"


@dataclass(frozen=True)
class Categorical:
    """A categorical attribute.

    Parameters
    ----------
    levels
        Level labels, in reporting order.
    coding
        ``"effects"`` (levels sum to zero; the last level is minus the sum of the others)
        or ``"dummy"`` (the reference level is fixed at zero).
    reference
        Reference level for dummy coding. Defaults to the first level.
    """

    levels: Sequence[str]
    coding: Literal["effects", "dummy"] = "effects"
    reference: str | None = None

    def __post_init__(self) -> None:
        levels = tuple(str(v) for v in self.levels)
        if len(levels) < 2:
            raise SchemaError("a Categorical attribute needs at least two levels")
        if len(set(levels)) != len(levels):
            raise SchemaError(f"duplicated levels: {levels}")
        if self.coding not in ("effects", "dummy"):
            raise SchemaError(f"unknown coding {self.coding!r}")
        if self.reference is not None and str(self.reference) not in levels:
            raise SchemaError(f"reference {self.reference!r} is not one of {levels}")
        object.__setattr__(self, "levels", levels)


@dataclass(frozen=True)
class Linear:
    """A numeric attribute entered as a single linear term.

    The value is transformed to ``(x - center) / scale`` before estimation. When
    ``center`` or ``scale`` is omitted it is taken from the data (mean and standard
    deviation of the values shown). ``sign`` constrains the coefficient's sign, which is
    the usual choice for price.
    """

    center: float | None = None
    scale: float | None = None
    sign: Literal["negative", "positive"] | None = None

    def __post_init__(self) -> None:
        if self.scale is not None and not self.scale > 0:
            raise SchemaError("scale must be positive")
        if self.sign not in (None, "negative", "positive"):
            raise SchemaError(f"unknown sign {self.sign!r}")


AttributeSpec = Categorical | Linear


@dataclass(frozen=True)
class _Fitted:
    name: str
    spec: AttributeSpec
    columns: tuple[str, ...]
    center: float = 0.0
    scale: float = 1.0


class Codebook:
    """The fitted coding of a set of attributes, plus an optional none alternative."""

    def __init__(self, fitted: Sequence[_Fitted], has_none: bool) -> None:
        self._fitted = tuple(fitted)
        self.has_none = has_none
        cols = [c for f in self._fitted for c in f.columns]
        if has_none:
            cols.append(NONE_COLUMN)
        self.columns: tuple[str, ...] = tuple(cols)

    @classmethod
    def fit(
        cls, df: pd.DataFrame, attributes: Mapping[str, AttributeSpec], has_none: bool
    ) -> Codebook:
        """Resolve data-dependent settings (centre and scale of linear terms)."""
        fitted = []
        for name, spec in attributes.items():
            if name not in df.columns:
                raise SchemaError(f"attribute column {name!r} not found")
            if isinstance(spec, Categorical):
                kept = _kept_levels(spec)
                fitted.append(_Fitted(name, spec, tuple(f"{name}[{lv}]" for lv in kept)))
            elif isinstance(spec, Linear):
                values = pd.to_numeric(df[name], errors="coerce").dropna()
                if values.empty:
                    raise SchemaError(f"linear attribute {name!r} has no numeric values")
                center = float(values.mean()) if spec.center is None else float(spec.center)
                scale = spec.scale
                if scale is None:
                    sd = float(values.std(ddof=0))
                    scale = sd if sd > 0 else 1.0
                fitted.append(_Fitted(name, spec, (name,), center, float(scale)))
            else:
                raise SchemaError(f"attribute {name!r}: expected Categorical or Linear")
        return cls(fitted, has_none)

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(f.name for f in self._fitted)

    def spec(self, name: str) -> AttributeSpec:
        return self._get(name).spec

    def sign_constraints(self) -> dict[int, str]:
        """Column index -> ``"negative"``/``"positive"`` for sign-constrained terms."""
        out = {}
        for f in self._fitted:
            if isinstance(f.spec, Linear) and f.spec.sign is not None:
                out[self.columns.index(f.columns[0])] = f.spec.sign
        return out

    def encode(self, df: pd.DataFrame, none: pd.Series | None = None) -> np.ndarray:
        """Return the coded design matrix, one row per row of ``df``.

        Rows flagged in ``none`` get zeros on every attribute and 1 on ``NONE``.
        """
        n = len(df)
        is_none = (
            np.zeros(n, dtype=bool) if none is None else np.asarray(none, dtype=bool)
        )
        blocks = []
        for f in self._fitted:
            col = df[f.name]
            if isinstance(f.spec, Categorical):
                blocks.append(_encode_categorical(f, col, is_none))
            else:
                x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
                bad = np.isnan(x) & ~is_none
                if bad.any():
                    raise CodingError(f"{f.name!r}: non-numeric value in row {np.argmax(bad)}")
                z = (x - f.center) / f.scale
                z[is_none] = 0.0
                blocks.append(z[:, None])
        if self.has_none:
            blocks.append(is_none.astype(float)[:, None])
        elif is_none.any():
            raise SchemaError("none rows given but the codebook has no none alternative")
        return np.hstack(blocks) if blocks else np.zeros((n, 0))

    def expand(self, coef: np.ndarray) -> pd.DataFrame:
        """Map coefficients (last axis = coded columns) to one value per level.

        ``coef`` has shape ``(..., K)``. The result has one row per leading index and a
        two-level column index ``(attribute, level)``. Linear attributes are reported
        per original unit (the coefficient divided by ``scale``) under level ``"slope"``.
        """
        coef = np.asarray(coef, dtype=float)
        if coef.shape[-1] != len(self.columns):
            raise SchemaError(f"expected last axis of length {len(self.columns)}")
        flat = coef.reshape(-1, coef.shape[-1])
        out: dict[tuple[str, str], np.ndarray] = {}
        for f in self._fitted:
            idx = [self.columns.index(c) for c in f.columns]
            block = flat[:, idx]
            if isinstance(f.spec, Categorical):
                for lv, vals in zip(f.spec.levels, _expand_categorical(f.spec, block).T,
                                    strict=True):
                    out[(f.name, lv)] = vals
            else:
                out[(f.name, "slope")] = block[:, 0] / f.scale
        if self.has_none:
            out[(NONE_COLUMN, NONE_COLUMN)] = flat[:, self.columns.index(NONE_COLUMN)]
        frame = pd.DataFrame(out)
        frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["attribute", "level"])
        return frame

    def _get(self, name: str) -> _Fitted:
        for f in self._fitted:
            if f.name == name:
                return f
        raise KeyError(name)

    def __repr__(self) -> str:
        return f"Codebook(columns={list(self.columns)})"


def _kept_levels(spec: Categorical) -> tuple[str, ...]:
    if spec.coding == "effects":
        return tuple(spec.levels[:-1])
    ref = spec.reference if spec.reference is not None else spec.levels[0]
    return tuple(lv for lv in spec.levels if lv != str(ref))


def _encode_categorical(f: _Fitted, col: pd.Series, is_none: np.ndarray) -> np.ndarray:
    spec = f.spec
    assert isinstance(spec, Categorical)
    values = np.array([None if pd.isna(v) else str(v) for v in col], dtype=object)
    known = set(spec.levels)
    for i, v in enumerate(values):
        if not is_none[i] and v not in known:
            raise CodingError(f"{f.name!r}: value {v!r} in row {i} is not a declared level")
    kept = _kept_levels(spec)
    out = np.zeros((len(values), len(kept)))
    for j, lv in enumerate(kept):
        out[:, j] = values == lv
    if spec.coding == "effects":
        last = (values == spec.levels[-1]) & ~is_none
        out[last, :] = -1.0
    out[is_none, :] = 0.0
    return out


def _expand_categorical(spec: Categorical, block: np.ndarray) -> np.ndarray:
    """Columns in ``spec.levels`` order."""
    n = block.shape[0]
    if spec.coding == "effects":
        return np.hstack([block, -block.sum(axis=1, keepdims=True)])
    kept = _kept_levels(spec)
    out = np.zeros((n, len(spec.levels)))
    for j, lv in enumerate(kept):
        out[:, spec.levels.index(lv)] = block[:, j]
    return out
