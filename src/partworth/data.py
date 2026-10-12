"""Choice data: validation of the long table and the padded arrays used by the models."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .coding import AttributeSpec, Categorical, Codebook, level_label
from .errors import ChoiceError, SchemaError


@dataclass(frozen=True, eq=False)
class ChoiceData:
    """Choice-based conjoint responses in a form ready for estimation.

    Build it with :meth:`from_long`. Arrays are padded to the largest number of tasks
    (``T``) and alternatives (``J``); ``mask`` marks the cells that exist and
    ``task_mask`` marks the tasks that enter the likelihood.

    Attributes
    ----------
    X : ndarray, shape (N, T, J, K)
    y : ndarray, shape (N, T) -- position of the chosen alternative
    mask : ndarray of bool, shape (N, T, J)
    task_mask : ndarray of bool, shape (N, T)
    Z : ndarray, shape (N, D) -- respondent covariates; column 0 is the constant
    respondents : Index -- respondent ids in array order
    tasks : DataFrame -- (respondent, task) labels with their (i, t) positions
    codebook : Codebook
    """

    X: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    task_mask: np.ndarray
    Z: np.ndarray
    respondents: pd.Index
    tasks: pd.DataFrame
    codebook: Codebook
    covariate_names: tuple[str, ...] = ("Intercept",)
    long: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        attributes: Mapping[str, AttributeSpec],
        respondent: str = "respondent",
        task: str = "task",
        alternative: str = "alternative",
        chosen: str = "chosen",
        none: str | None = None,
        covariates: pd.DataFrame | None = None,
        codebook: Codebook | None = None,
    ) -> ChoiceData:
        """Build from a long table with one row per respondent x task x alternative.

        Parameters
        ----------
        df
            The responses.
        attributes
            Attribute column -> :class:`~partworth.Categorical` or :class:`~partworth.Linear`.
        respondent, task, alternative, chosen
            Column names. ``chosen`` holds 1 for the chosen alternative and 0 otherwise.
        none
            Optional boolean column flagging the none ("no purchase") alternative.
            Attribute values on those rows are ignored.
        covariates
            Optional table indexed by respondent id, one row per respondent. Numeric
            columns are centred; categorical columns are dummy coded (first level dropped).
        codebook
            Reuse the coding of another data set (e.g. a hold-out sample).
        """
        required = [respondent, task, alternative, chosen, *attributes]
        if none is not None:
            required.append(none)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SchemaError(f"missing columns: {missing}")
        work = df[required].copy()
        if work[[respondent, task, alternative, chosen]].isna().any().any():
            raise SchemaError("respondent, task, alternative and chosen must not be missing")
        if none is None:
            is_none = pd.Series(False, index=work.index)
        else:
            flag = work[none]
            if flag.isna().any() or not flag.isin([0, 1, True, False]).all():
                raise SchemaError(f"{none!r} must contain only True/False (or 1/0), "
                                  "with no missing values")
            is_none = flag.astype(bool)
        ch = pd.to_numeric(work[chosen], errors="coerce")
        if ch.isna().any() or not ch.isin([0, 1]).all():
            raise SchemaError(f"{chosen!r} must contain only 0 and 1")
        work[chosen] = ch.astype(int)

        keys = [respondent, task, alternative]
        if work.duplicated(keys).any():
            dup = work.loc[work.duplicated(keys, keep=False), keys].head(3)
            raise SchemaError(f"duplicated respondent/task/alternative rows, e.g.\n{dup}")
        n_chosen = work.groupby([respondent, task], sort=False)[chosen].sum()
        bad = n_chosen[n_chosen != 1]
        if len(bad):
            raise ChoiceError(
                f"{len(bad)} task(s) do not have exactly one chosen alternative, "
                f"e.g. {bad.index[0]} has {bad.iloc[0]}"
            )

        if codebook is None:
            codebook = Codebook.fit(work.loc[~is_none], attributes, has_none=none is not None)
        coded = codebook.encode(work, is_none)

        resp_codes, resp_index = pd.factorize(work[respondent], sort=False)
        t_pos = (
            work.groupby(respondent, sort=False)[task].rank(method="dense").astype(int) - 1
        ).to_numpy()
        j_pos = (
            work.groupby([respondent, task], sort=False)[alternative]
            .rank(method="dense")
            .astype(int)
            - 1
        ).to_numpy()
        n, t_max, j_max, k = len(resp_index), t_pos.max() + 1, j_pos.max() + 1, coded.shape[1]

        X = np.zeros((n, t_max, j_max, k))
        mask = np.zeros((n, t_max, j_max), dtype=bool)
        y = np.zeros((n, t_max), dtype=np.int64)
        X[resp_codes, t_pos, j_pos] = coded
        mask[resp_codes, t_pos, j_pos] = True
        sel = work[chosen].to_numpy() == 1
        y[resp_codes[sel], t_pos[sel]] = j_pos[sel]
        task_mask = mask.any(axis=2)

        tasks = (
            pd.DataFrame({"respondent": work[respondent].to_numpy(),
                          "task": work[task].to_numpy(), "i": resp_codes, "t": t_pos})
            .drop_duplicates(["i", "t"])
            .sort_values(["i", "t"])
            .reset_index(drop=True)
        )
        Z, z_names = _covariates(covariates, resp_index)
        long = work.rename(columns={respondent: "respondent", task: "task",
                                    alternative: "alternative", chosen: "chosen"})
        long["none"] = is_none.to_numpy()
        return cls(X, y, mask, task_mask, Z, pd.Index(resp_index, name="respondent"),
                   tasks, codebook, z_names, long.reset_index(drop=True))

    # ------------------------------------------------------------------ shape helpers
    @property
    def n_respondents(self) -> int:
        return self.X.shape[0]

    @property
    def n_params(self) -> int:
        return self.X.shape[3]

    @property
    def n_tasks(self) -> np.ndarray:
        """Number of tasks entering the likelihood, per respondent."""
        return self.task_mask.sum(axis=1)

    @property
    def n_alternatives(self) -> np.ndarray:
        """Number of alternatives per task, shape (N, T)."""
        return self.mask.sum(axis=2)

    @property
    def coef_names(self) -> tuple[str, ...]:
        return self.codebook.columns

    # ------------------------------------------------------------------ subsetting
    def subset(self, respondents: Sequence) -> ChoiceData:
        """Keep the given respondents, in the given order."""
        if pd.Index(respondents).has_duplicates:
            raise SchemaError("respondents passed to subset() must be unique")
        idx = self.respondents.get_indexer(pd.Index(respondents))
        if (idx < 0).any():
            raise KeyError(f"unknown respondents: {list(pd.Index(respondents)[idx < 0])}")
        tasks = self.tasks[self.tasks["i"].isin(idx)].copy()
        tasks["i"] = pd.Index(idx).get_indexer(tasks["i"])
        tasks = tasks.sort_values(["i", "t"]).reset_index(drop=True)
        keep = set(pd.Index(respondents))
        long = self.long[self.long["respondent"].isin(keep)].reset_index(drop=True)
        return replace(self, X=self.X[idx], y=self.y[idx], mask=self.mask[idx],
                       task_mask=self.task_mask[idx], Z=self.Z[idx],
                       respondents=self.respondents[idx], tasks=tasks, long=long)

    def holdout(self, tasks: Sequence) -> tuple[ChoiceData, ChoiceData]:
        """Split by task label into (estimation, hold-out) data sets of the same shape."""
        is_out = np.zeros_like(self.task_mask)
        sel = self.tasks[self.tasks["task"].isin(list(tasks))]
        if sel.empty:
            raise KeyError(f"none of the tasks {list(tasks)} were found")
        is_out[sel["i"].to_numpy(), sel["t"].to_numpy()] = True
        out_long = self.long["task"].isin(list(tasks))
        empty = ~(self.task_mask & ~is_out).any(axis=1)
        if empty.any():
            warnings.warn(f"{int(empty.sum())} respondent(s) have no estimation task left "
                          "after holdout(); their part-worths come from the prior only",
                          stacklevel=2)
        train = replace(self, task_mask=self.task_mask & ~is_out,
                        long=self.long[~out_long].reset_index(drop=True))
        test = replace(self, task_mask=self.task_mask & is_out,
                       long=self.long[out_long].reset_index(drop=True))
        return train, test

    # ------------------------------------------------------------------ summaries
    def describe(self) -> pd.DataFrame:
        """Times shown, times chosen and choice rate for every attribute level."""
        rows = []
        inside = self.long[~self.long["none"]]
        for name in self.codebook.attributes:
            spec = self.codebook.spec(name)
            if isinstance(spec, Categorical):
                g = inside.groupby(inside[name].map(level_label))["chosen"]
                stats = pd.DataFrame({"shown": g.size(), "chosen": g.sum()})
                stats = stats.reindex(list(spec.levels), fill_value=0)
            else:
                g = inside.groupby(name)["chosen"]
                stats = pd.DataFrame({"shown": g.size(), "chosen": g.sum()})
                stats.index = stats.index.astype(str)
            stats.index.name = "level"
            stats.insert(0, "attribute", name)
            rows.append(stats.reset_index())
        if self.codebook.has_none:
            nn = self.long[self.long["none"]]
            rows.append(pd.DataFrame({"attribute": ["NONE"], "level": ["NONE"],
                                      "shown": [len(nn)], "chosen": [nn["chosen"].sum()]}))
        out = pd.concat(rows, ignore_index=True)[["attribute", "level", "shown", "chosen"]]
        out["rate"] = out["chosen"] / out["shown"].where(out["shown"] > 0)
        return out

    def __repr__(self) -> str:
        n, t, j, k = self.X.shape
        return (f"ChoiceData(respondents={n}, max_tasks={t}, max_alternatives={j}, "
                f"params={k}, covariates={len(self.covariate_names) - 1})")


def _covariates(cov: pd.DataFrame | None, resp_index) -> tuple[np.ndarray, tuple[str, ...]]:
    n = len(resp_index)
    if cov is None:
        return np.ones((n, 1)), ("Intercept",)
    if cov.index.has_duplicates:
        raise SchemaError("covariates must have one row per respondent")
    missing = pd.Index(resp_index).difference(cov.index)
    if len(missing):
        raise SchemaError(f"covariates missing for respondents: {list(missing[:5])}")
    sub = cov.loc[list(resp_index)]
    if sub.isna().any().any():
        raise SchemaError("covariates must not contain missing values")
    sub = pd.get_dummies(sub, drop_first=True, dtype=float)
    vals = sub.to_numpy(dtype=float)
    vals = vals - vals.mean(axis=0, keepdims=True)
    return (np.hstack([np.ones((n, 1)), vals]),
            ("Intercept", *(str(c) for c in sub.columns)))
