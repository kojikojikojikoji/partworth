"""Parameter recovery on simulated data, with the numbers written to a report.

Same design as tests/statistical/test_recovery.py: 300 respondents x 12 tasks x
3 alternatives + none; brand (3 levels), size (3 levels), price (4 values, negative).

    python validation/recovery.py
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np

import partworth as pw
from partworth.datasets import simulate_cbc

MU = [0.6, -0.4, 0.4, 0.0, 0.3, -0.5]


def covered(draws: np.ndarray, true: np.ndarray, level: float = 0.9) -> np.ndarray:
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2], axis=0)
    return (lo <= true) & (true <= hi)


def main() -> None:
    data, truth = simulate_cbc(
        300, 12, 3, {"brand": ["x", "y", "z"], "size": ["s", "m", "l"], "price": [1, 2, 3, 4]},
        price="price", mu=MU, sd=0.6, corr=0.3, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pw.ConvergenceWarning)
        res = pw.HierarchicalMNL(data).fit(chains=4, random_seed=3)
    names = list(data.coef_names)
    post = res.idata["posterior"]
    delta = post["Delta"].isel(covariate=0).stack(s=("chain", "draw")).transpose("s", ...)
    tau = post["tau"].stack(s=("chain", "draw")).transpose("s", ...)
    beta_draws = res.beta_draws()
    beta_mean = res.beta_mean()
    ind_cov = covered(beta_draws, truth["beta"])
    corr = [np.corrcoef(beta_mean[:, k], truth["beta"][:, k])[0, 1] for k in range(len(names))]
    d = res.diagnostics()

    lines = [
        "# Parameter recovery on simulated data",
        "",
        f"Run {dt.date.today().isoformat()} with partworth {pw.__version__}. "
        "300 respondents x 12 tasks x 3 alternatives + none; true population means "
        f"{MU}, SD 0.6, common correlation 0.3; price lognormal (negative). "
        "4 chains x 1000 draws after 2000 tuning steps (nutpie), seed 3.",
        "",
        "Price's population mean and SD are on the log scale (see ADR-0004).",
        "",
        "| coefficient | true mean | 90% interval of mean | covered | true SD "
        "| 90% interval of SD | covered | corr(respondent means, truth) "
        "| respondent 90% coverage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    sd_true = np.sqrt(np.diag(truth["cov"]))
    for k, name in enumerate(names):
        mlo, mhi = np.quantile(delta.to_numpy()[:, k], [0.05, 0.95])
        slo, shi = np.quantile(tau.to_numpy()[:, k], [0.05, 0.95])
        lines.append(
            f"| {name} | {MU[k]:.2f} | [{mlo:.2f}, {mhi:.2f}] | {mlo <= MU[k] <= mhi} "
            f"| {sd_true[k]:.2f} | [{slo:.2f}, {shi:.2f}] | {slo <= sd_true[k] <= shi} "
            f"| {corr[k]:.3f} | {ind_cov[:, k].mean():.3f} |")
    lines += [
        "",
        f"Overall respondent-level 90% interval coverage: {ind_cov.mean():.3f} "
        f"({ind_cov.size} respondent x coefficient pairs).",
        f"Diagnostics: R-hat max {d.rhat_max:.3f}, ESS bulk min {d.ess_bulk_min:.0f}, "
        f"divergences {d.divergences}.",
        "",
    ]
    out = Path("validation/results/recovery_report.md")
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
