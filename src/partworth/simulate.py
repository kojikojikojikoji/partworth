"""Market simulation and willingness to pay from posterior draws.

Shares are computed for every posterior draw (respondent-level betas), so each share
comes with a credible interval that reflects estimation uncertainty.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .coding import NONE_COLUMN, Categorical, Linear
from .errors import SchemaError
from .results import HBResult


class MarketSimulator:
    """Preference-share simulator.

    Parameters
    ----------
    result
        A fitted :class:`~partworth.HBResult`.
    rule
        ``"logit"`` (share of preference) or ``"first_choice"`` (each respondent picks
        the highest-utility product).
    exponent
        Scale factor applied to utilities before the logit rule (1 = as estimated).
    max_draws
        Number of posterior draws used (thinned evenly) to keep simulations fast.
    """

    def __init__(self, result: HBResult, *, rule: Literal["logit", "first_choice"] = "logit",
                 exponent: float = 1.0, max_draws: int = 500) -> None:
        if rule not in ("logit", "first_choice"):
            raise ValueError("rule must be 'logit' or 'first_choice'")
        self.result = result
        self.rule = rule
        self.exponent = exponent
        draws = result.beta_draws()
        idx = np.linspace(0, draws.shape[0] - 1, min(draws.shape[0], max_draws)).astype(int)
        self._beta = draws[idx]                        # (S, N, K)

    def _design(self, scenario: pd.DataFrame, include_none: bool) -> tuple[np.ndarray, list]:
        cb = self.result.data.codebook
        missing = [a for a in cb.attributes if a not in scenario.columns]
        if missing:
            raise SchemaError(f"scenario is missing attributes: {missing}")
        names = (list(scenario["product"]) if "product" in scenario.columns
                 else [f"product {i + 1}" for i in range(len(scenario))])
        X = cb.encode(scenario.reset_index(drop=True))
        if include_none:
            if not cb.has_none:
                raise SchemaError("the model has no none alternative")
            none_row = np.zeros((1, X.shape[1]))
            none_row[0, cb.columns.index(NONE_COLUMN)] = 1.0
            X = np.vstack([X, none_row])
            names.append(NONE_COLUMN)
        return X, names

    def _shares_by_draw(self, X: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
        util = np.einsum("snk,pk->snp", self._beta, X)        # (S, N, P)
        if self.rule == "logit":
            u = self.exponent * util
            u -= u.max(axis=2, keepdims=True)
            p = np.exp(u)
            p /= p.sum(axis=2, keepdims=True)
        else:
            p = np.zeros_like(util)
            np.put_along_axis(p, util.argmax(axis=2)[..., None], 1.0, axis=2)
        w = np.ones(p.shape[1]) if weights is None else np.asarray(weights, dtype=float)
        return np.einsum("snp,n->sp", p, w / w.sum())          # (S, P)

    def shares(self, scenario: pd.DataFrame, *, include_none: bool = True,
               interval: float = 0.9, weights: pd.Series | None = None) -> pd.DataFrame:
        """Shares of preference for the products in ``scenario`` (one row per product).

        ``scenario`` has one column per attribute and optionally a ``product`` column.
        ``weights`` (indexed by respondent) reweights respondents.
        """
        X, names = self._design(scenario, include_none)
        w = None
        if weights is not None:
            w = weights.reindex(self.result.data.respondents).to_numpy(dtype=float)
            if np.isnan(w).any():
                raise SchemaError("weights are missing for some respondents")
            if (w < 0).any() or w.sum() <= 0:
                raise SchemaError("weights must be non-negative with a positive sum")
        s = self._shares_by_draw(X, w)
        lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
        return pd.DataFrame({
            "product": names,
            "share": s.mean(axis=0),
            "lower": np.quantile(s, lo, axis=0),
            "upper": np.quantile(s, hi, axis=0),
        })

    def price_curve(self, scenario: pd.DataFrame, *, product: int | str,
                    attribute: str = "price", grid: Sequence[float],
                    include_none: bool = True, interval: float = 0.9) -> pd.DataFrame:
        """Share of one product as one numeric attribute moves along ``grid``."""
        if not isinstance(self.result.data.codebook.spec(attribute), Linear):
            raise SchemaError(f"{attribute!r} is not a Linear attribute")
        scenario = scenario.reset_index(drop=True)
        pos = (int(product) if isinstance(product, int | np.integer)
               else int(np.flatnonzero(scenario["product"].to_numpy() == product)[0]))
        rows = []
        for value in grid:
            sc = scenario.copy()
            sc.loc[pos, attribute] = value
            out = self.shares(sc, include_none=include_none, interval=interval)
            rows.append({attribute: value, **out.iloc[pos][["share", "lower", "upper"]]})
        return pd.DataFrame(rows)


def wtp(result: HBResult, *, attribute: str, from_level: str, to_level: str,
        price: str = "price", interval: float = 0.9) -> pd.DataFrame:
    """Willingness to pay for moving ``attribute`` from ``from_level`` to ``to_level``.

    For every posterior draw and respondent, WTP = (u_to - u_from) / (-price slope),
    in the units of the price attribute. Because individual ratios are heavy-tailed,
    the population summary is the median over respondents; its credible interval comes
    from the spread of that median across draws. The mean is reported for reference.
    """
    cb = result.data.codebook
    if not isinstance(cb.spec(attribute), Categorical):
        raise SchemaError(f"{attribute!r} is not a Categorical attribute")
    if not isinstance(cb.spec(price), Linear):
        raise SchemaError(f"{price!r} is not a Linear attribute")
    draws = result.beta_draws()                       # (S, N, K)
    s, n, _ = draws.shape
    levels = cb.expand(draws)
    gain = (levels[(attribute, to_level)] - levels[(attribute, from_level)]).to_numpy()
    slope = levels[(price, "slope")].to_numpy()
    with np.errstate(divide="ignore"):
        value = (gain / -slope).reshape(s, n)
    median_by_draw = np.median(value, axis=1)
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    positive = (slope < 0).mean()
    return pd.DataFrame({
        "attribute": [attribute], "from": [from_level], "to": [to_level],
        "median": [float(np.median(median_by_draw))],
        "lower": [float(np.quantile(median_by_draw, lo))],
        "upper": [float(np.quantile(median_by_draw, hi))],
        "mean": [float(np.mean(value[np.isfinite(value)]))],
        "share_negative_price_slope": [float(positive)],
    })
