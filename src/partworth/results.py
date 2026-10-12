"""Summaries of a fitted hierarchical MNL: part-worths, importances, RLH, diagnostics.

Zero-centred diffs, importances and RLH follow the definitions in Sawtooth Software's
public CBC/HB documentation (see ADR-0005): within each respondent, categorical
part-worths are centred within attribute; the utility ranges of all attributes are
scaled to sum to 100 x (number of attributes); importance is an attribute's range as a
percentage of the sum of ranges; RLH is the geometric mean of the predicted
probabilities of the chosen alternatives.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

from .coding import NONE_COLUMN, Categorical, Linear
from .data import ChoiceData
from .errors import ConvergenceWarning

RHAT_MAX = 1.01
_PAD = -1e9  # utility for alternatives that were not shown
ESS_MIN = 400


@dataclass(frozen=True)
class Diagnostics:
    """Convergence summary over the population parameters and respondent betas."""

    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int
    worst: str

    @property
    def ok(self) -> bool:
        return (self.rhat_max < RHAT_MAX and self.ess_bulk_min > ESS_MIN
                and self.divergences == 0)

    def report(self) -> str:
        lines = [
            f"R-hat max      {self.rhat_max:.3f}  (target < {RHAT_MAX})",
            f"ESS bulk min   {self.ess_bulk_min:.0f}  (target > {ESS_MIN})",
            f"ESS tail min   {self.ess_tail_min:.0f}  (target > {ESS_MIN})",
            f"divergences    {self.divergences}  (target 0)",
            f"worst variable {self.worst}",
        ]
        if self.ok:
            return "\n".join(["converged", *lines])
        hints = []
        if self.divergences:
            hints.append("- divergences: raise target_accept (0.95-0.99) or tighten tau_sd")
        if self.rhat_max >= RHAT_MAX or self.ess_bulk_min <= ESS_MIN:
            hints.append("- low ESS / high R-hat: run more draws and tune steps; check that "
                         "every level was shown often enough (ChoiceData.describe())")
            hints.append("- many sparse levels: try covariance='diagonal' first")
        return "\n".join(["NOT converged", *lines, "what to try:", *hints])


class HBResult:
    """Posterior draws and the data they were fitted to."""

    def __init__(self, idata, data: ChoiceData) -> None:
        self.idata = idata
        self.data = data

    # ------------------------------------------------------------------ draws
    def beta_draws(self) -> np.ndarray:
        """Respondent coefficients in the coded space, shape (draws, N, K)."""
        b = self.idata["posterior"]["beta"]
        return b.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()

    def beta_mean(self) -> np.ndarray:
        return self.idata["posterior"]["beta"].mean(("chain", "draw")).to_numpy()

    # ------------------------------------------------------------------ part-worths
    def part_worths(self, kind: str = "raw") -> pd.DataFrame:
        """Posterior-mean part-worths per respondent, one column per level.

        ``kind="raw"`` gives logit-scale utilities (effects coding sums to zero within
        attribute; dummy coding has the reference at zero; linear terms per original
        unit). ``kind="zc_diffs"`` gives zero-centred diffs.
        """
        beta = self.beta_mean()
        if kind == "raw":
            out = self.data.codebook.expand(beta)
        elif kind == "zc_diffs":
            out = self._zc_diffs(beta)
        else:
            raise ValueError("kind must be 'raw' or 'zc_diffs'")
        out.index = self.data.respondents
        return out

    def importances(self) -> pd.DataFrame:
        """Attribute importances (%) per respondent from posterior-mean part-worths."""
        ranges = self._ranges(self.data.codebook.expand(self.beta_mean()))
        out = ranges.div(ranges.sum(axis=1), axis=0) * 100
        out.index = self.data.respondents
        return out

    def rlh(self) -> pd.Series:
        """Root likelihood per respondent (posterior-mean betas, estimation tasks)."""
        p = self._chosen_prob(self.beta_mean())
        logp = np.where(self.data.task_mask, np.log(p), 0.0)
        out = np.exp(logp.sum(axis=1) / self.data.task_mask.sum(axis=1))
        return pd.Series(out, index=self.data.respondents, name="RLH")

    def population(self) -> pd.DataFrame:
        """Posterior summary of the population mean and standard deviation per coefficient.

        For sign-constrained terms (e.g. price) these refer to the log scale shown in the
        ``scale`` column: the coefficient is ``-exp(x)`` with ``x`` normal.
        """
        post = self.idata["posterior"]
        delta = post["Delta"].isel(covariate=0)
        tau = post["tau"]
        rows = {}
        for name, arr in (("mean", delta), ("sd", tau)):
            q = arr.quantile([0.05, 0.95], dim=("chain", "draw"))
            rows[name] = pd.DataFrame({
                "estimate": arr.mean(("chain", "draw")).to_numpy(),
                "q05": q.sel(quantile=0.05).to_numpy(),
                "q95": q.sel(quantile=0.95).to_numpy(),
            }, index=list(self.data.coef_names))
        out = pd.concat(rows, names=["parameter", "coef"])
        signs = self.data.codebook.sign_constraints()
        out["scale"] = "coef"
        for col, sign in signs.items():
            name = self.data.coef_names[col]
            out.loc[(slice(None), name), "scale"] = (
                "log(-coef)" if sign == "negative" else "log(coef)")
        return out

    def to_sawtooth_csv(self, path, kind: str = "raw") -> None:
        """Write respondent utilities in Sawtooth-style CSV layout (see ``partworth.io``)."""
        from .io.sawtooth import write_utilities

        write_utilities(self.part_worths(kind), path, rlh=self.rlh())

    def save(self, path) -> None:
        """Save posterior draws to a NetCDF file (reload with :meth:`load` and the data)."""
        self.idata.to_netcdf(path, engine="h5netcdf")

    @classmethod
    def load(cls, path, data: ChoiceData) -> HBResult:
        import xarray as xr

        return cls(xr.open_datatree(path, engine="h5netcdf").load(), data)

    def predict(self, data: ChoiceData | None = None) -> pd.DataFrame:
        """Predicted probability of the chosen alternative and hit, per active task.

        Pass the hold-out half of :meth:`ChoiceData.holdout` to score hold-out tasks.
        """
        data = self.data if data is None else data
        beta = self.beta_mean()
        util = np.einsum("ntjk,nk->ntj", data.X, beta)
        util = np.where(data.mask, util, _PAD)
        prob = np.exp(util - util.max(axis=2, keepdims=True))
        prob /= prob.sum(axis=2, keepdims=True)
        ii, tt = np.nonzero(data.task_mask)
        chosen = data.y[ii, tt]
        return pd.DataFrame({
            "respondent": data.respondents[ii],
            "t": tt,
            "p_chosen": prob[ii, tt, chosen],
            "hit": prob[ii, tt].argmax(axis=1) == chosen,
        })

    # ------------------------------------------------------------------ diagnostics
    def diagnostics(self) -> Diagnostics:
        post = self.idata["posterior"]
        names = [v for v in ("Delta", "tau", "corr", "beta") if v in post.data_vars]
        rhat = az.rhat(self.idata, var_names=names)
        ess_b = az.ess(self.idata, var_names=names, method="bulk")
        ess_t = az.ess(self.idata, var_names=names, method="tail")

        def extreme(ds, fn):
            vals = {}
            for v in names:
                arr = ds[v].to_numpy().astype(float)
                if v == "corr":  # the diagonal is constant 1
                    arr = arr[~np.eye(arr.shape[0], dtype=bool)]
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    vals[v] = fn(arr)
            return vals

        r = extreme(rhat, np.max)
        eb, et = extreme(ess_b, np.min), extreme(ess_t, np.min)
        div = self.idata["sample_stats"]["diverging"]
        return Diagnostics(  # nan when not computable (e.g. a single chain)
            rhat_max=float(max(r.values())) if r else float("nan"),
            ess_bulk_min=float(min(eb.values())) if eb else float("nan"),
            ess_tail_min=float(min(et.values())) if et else float("nan"),
            divergences=int(div.sum()),
            worst=max(r, key=r.get) if r else "",
        )

    def _warn_if_not_converged(self) -> None:
        d = self.diagnostics()
        if not d.ok:
            warnings.warn(d.report(), ConvergenceWarning, stacklevel=3)

    # ------------------------------------------------------------------ helpers
    def _chosen_prob(self, beta: np.ndarray) -> np.ndarray:
        d = self.data
        util = np.einsum("ntjk,nk->ntj", d.X, beta)
        util = np.where(d.mask, util, _PAD)
        util -= util.max(axis=2, keepdims=True)
        prob = np.exp(util)
        prob /= prob.sum(axis=2, keepdims=True)
        return np.take_along_axis(prob, d.y[..., None], axis=2)[..., 0]

    def _linear_span(self, name: str) -> float:
        vals = pd.to_numeric(self.data.long.loc[~self.data.long["none"], name])
        return float(vals.max() - vals.min())

    def _ranges(self, levels: pd.DataFrame) -> pd.DataFrame:
        cb = self.data.codebook
        out = {}
        for name in cb.attributes:
            spec = cb.spec(name)
            if isinstance(spec, Categorical):
                block = levels[name]
                out[name] = block.max(axis=1) - block.min(axis=1)
            elif isinstance(spec, Linear):
                out[name] = levels[(name, "slope")].abs() * self._linear_span(name)
        return pd.DataFrame(out)

    def _zc_diffs(self, beta: np.ndarray) -> pd.DataFrame:
        cb = self.data.codebook
        levels = cb.expand(beta)
        ranges = self._ranges(levels)
        factor = 100.0 * len(cb.attributes) / ranges.sum(axis=1)
        out = levels.copy()
        for name in cb.attributes:
            if isinstance(cb.spec(name), Categorical):
                block = levels[name]
                out[name] = block.sub(block.mean(axis=1), axis=0).to_numpy()
        out = out.mul(factor, axis=0)
        if cb.has_none:
            out[(NONE_COLUMN, NONE_COLUMN)] = levels[(NONE_COLUMN, NONE_COLUMN)] * factor
        return out
