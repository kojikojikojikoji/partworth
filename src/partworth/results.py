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
from .errors import ConvergenceWarning, SchemaError

RHAT_MAX = 1.01
_PAD = -1e9  # utility for alternatives that were not shown
ESS_MIN = 400


@dataclass(frozen=True)
class Diagnostics:
    """Convergence summary.

    ``rhat_max``, ``ess_bulk_min`` and ``ess_tail_min`` cover the parameters that
    part-worths, shares and WTP depend on: population means (Delta), between-respondent
    SDs (tau) and respondent coefficients (beta). Correlations mix more slowly and are
    reported separately in ``corr_rhat_max`` / ``corr_ess_bulk_min``; they only need to
    meet the targets if you report correlations.
    """

    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergences: int
    worst: str
    corr_rhat_max: float = float("nan")
    corr_ess_bulk_min: float = float("nan")

    @property
    def ok(self) -> bool:
        return (self.rhat_max < RHAT_MAX and self.ess_bulk_min > ESS_MIN
                and self.divergences == 0)

    @property
    def corr_ok(self) -> bool:
        if np.isnan(self.corr_rhat_max):
            return True
        return self.corr_rhat_max < RHAT_MAX and self.corr_ess_bulk_min > ESS_MIN

    def report(self) -> str:
        lines = [
            f"R-hat max      {self.rhat_max:.3f}  (target < {RHAT_MAX})",
            f"ESS bulk min   {self.ess_bulk_min:.0f}  (target > {ESS_MIN})",
            f"ESS tail min   {self.ess_tail_min:.0f}  (target > {ESS_MIN})",
            f"divergences    {self.divergences}  (target 0)",
            f"worst variable {self.worst}",
        ]
        if not np.isnan(self.corr_rhat_max):
            lines.append(f"correlations   R-hat max {self.corr_rhat_max:.3f}, "
                         f"ESS bulk min {self.corr_ess_bulk_min:.0f}")
        head = "converged" if self.ok else "NOT converged"
        hints = []
        if self.divergences:
            hints.append("- divergences: raise target_accept (0.95-0.99) or tighten tau_sd")
        if self.rhat_max >= RHAT_MAX or self.ess_bulk_min <= ESS_MIN:
            hints.append("- low ESS / high R-hat: run more draws and tune steps; check that "
                         "every level was shown often enough (ChoiceData.describe())")
            hints.append("- many sparse levels: try priors=pw.Priors(covariance='diagonal')")
        if not self.corr_ok:
            hints.append("- correlations have not reached the targets: part-worths, shares "
                         "and WTP are unaffected, but rerun with more draws (e.g. 3000) "
                         "before reporting correlations")
        out = [head, *lines]
        if hints:
            out += ["what to try:", *hints]
        return "\n".join(out)


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
        n_tasks = self.data.task_mask.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            logp = np.where(self.data.task_mask, np.log(p), 0.0)
            # NaN for respondents with no estimation task (all tasks held out)
            out = np.where(n_tasks > 0, np.exp(logp.sum(axis=1) / n_tasks), np.nan)
        return pd.Series(out, index=self.data.respondents, name="RLH")

    def population(self) -> pd.DataFrame:
        """Posterior summary of the population mean and standard deviation per coefficient.

        Values are in the coded space used for estimation. For linear attributes that is
        per unit of ``(x - center) / scale`` (see ``ChoiceData.codebook``), not per original
        unit. For sign-constrained terms (e.g. price) they also refer to the log scale
        shown in the ``scale`` column: the coefficient is ``-exp(z)`` with ``z`` normal.
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
        for name in self.data.codebook.attributes:
            if isinstance(self.data.codebook.spec(name), Linear):
                out.loc[(slice(None), name), "scale"] = "coef per standardised unit"
        for col, sign in signs.items():
            name = self.data.coef_names[col]
            out.loc[(slice(None), name), "scale"] = (
                "log(-coef per standardised unit)" if sign == "negative"
                else "log(coef per standardised unit)")
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
        if data.codebook != self.data.codebook:
            raise SchemaError("data was coded differently from the estimation data; "
                              "build it with codebook=result.data.codebook")
        pos = self.data.respondents.get_indexer(data.respondents)
        if (pos < 0).any():
            raise SchemaError("data contains respondents that were not in the estimation "
                              f"data: {list(data.respondents[pos < 0][:5])}")
        beta = self.beta_mean()[pos]
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
        core = [v for v in ("Delta", "tau", "beta") if v in post.data_vars]
        names = core + (["corr"] if "corr" in post.data_vars else [])
        with warnings.catch_warnings():
            # constant entries (the diagonal of corr) divide by zero inside ArviZ
            warnings.simplefilter("ignore", RuntimeWarning)
            rhat = az.rhat(self.idata, var_names=names)
            ess_b = az.ess(self.idata, var_names=names, method="bulk")
            ess_t = az.ess(self.idata, var_names=names, method="tail")

        def values(ds, v):
            arr = ds[v].to_numpy().astype(float)
            if v == "corr":  # the diagonal is constant 1
                arr = arr[~np.eye(arr.shape[0], dtype=bool)]
            return arr[np.isfinite(arr)]

        def extreme(ds, fn, variables):
            out = {v: fn(a) for v in variables if (a := values(ds, v)).size}
            return out

        r, eb, et = (extreme(rhat, np.max, core), extreme(ess_b, np.min, core),
                     extreme(ess_t, np.min, core))
        nan = float("nan")
        corr_r = extreme(rhat, np.max, ["corr"]) if "corr" in names else {}
        corr_e = extreme(ess_b, np.min, ["corr"]) if "corr" in names else {}
        div = self.idata["sample_stats"]["diverging"]
        return Diagnostics(  # nan when not computable (e.g. a single chain)
            rhat_max=float(max(r.values())) if r else nan,
            ess_bulk_min=float(min(eb.values())) if eb else nan,
            ess_tail_min=float(min(et.values())) if et else nan,
            divergences=int(div.sum()),
            worst=max(r, key=r.get) if r else "",
            corr_rhat_max=float(corr_r["corr"]) if corr_r else nan,
            corr_ess_bulk_min=float(corr_e["corr"]) if corr_e else nan,
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
        lo, hi = self.data.codebook.linear_span(name)
        return hi - lo

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
        shift = np.zeros(len(levels))
        for name in cb.attributes:
            if isinstance(cb.spec(name), Categorical):
                block = levels[name]
                means = block.mean(axis=1)
                out[name] = block.sub(means, axis=0).to_numpy()
                shift += means.to_numpy()
        if cb.has_none:
            # centring lowers every product's utility by `shift`; move NONE with it so
            # product-versus-none comparisons are unchanged
            out[(NONE_COLUMN, NONE_COLUMN)] = levels[(NONE_COLUMN, NONE_COLUMN)] - shift
        return out.mul(factor, axis=0)
