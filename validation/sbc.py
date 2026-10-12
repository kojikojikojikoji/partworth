"""Simulation-based calibration (Talts et al. 2018) for HierarchicalMNL.

For each replicate: draw parameters and choices from the prior predictive of the model
on a fixed design, fit the model to those choices, and record the rank of each true
population parameter among thinned posterior draws. If the sampler and model are
correct, ranks are uniform. Results go to validation/results/sbc_report.md.

    python validation/sbc.py --reps 100
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import partworth as pw
from partworth.datasets import simulate_cbc

N_RANK = 99  # posterior draws kept per replicate -> ranks 0..99


def thinned(arr: np.ndarray, n: int) -> np.ndarray:
    idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
    return arr[idx]


def one_replicate(base: pw.ChoiceData, priors: pw.Priors, seed: int, draws: int):
    model = pw.HierarchicalMNL(base, priors=priors)
    prior = model.prior_predictive(samples=1, random_seed=seed)
    truth_delta = prior["prior"]["Delta"].isel(chain=0, draw=0, covariate=0).to_numpy()
    truth_tau = prior["prior"]["tau"].isel(chain=0, draw=0).to_numpy()
    y_obs = prior["prior_predictive"]["y"].isel(chain=0, draw=0).to_numpy()

    ii, tt = np.nonzero(base.task_mask)
    y = base.y.copy()
    y[ii, tt] = y_obs
    data = dataclasses.replace(base, y=y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pw.HierarchicalMNL(data, priors=priors).fit(
            draws=draws, tune=draws, chains=2, random_seed=seed)
    post = res.idata["posterior"]
    d = post["Delta"].isel(covariate=0).stack(s=("chain", "draw")).transpose("s", ...)
    t = post["tau"].stack(s=("chain", "draw")).transpose("s", ...)
    d, t = thinned(d.to_numpy(), N_RANK), thinned(t.to_numpy(), N_RANK)
    diag = res.diagnostics()
    return ((d < truth_delta).sum(axis=0), (t < truth_tau).sum(axis=0),
            diag.divergences, diag.rhat_max)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--out", default="validation/results")
    args = ap.parse_args()

    base, _ = simulate_cbc(60, 10, 3, {"brand": ["x", "y", "z"], "price": [1, 2, 3]},
                           price="price", seed=0)
    priors = pw.Priors(mu_sd=1.0, tau_sd=0.5)
    names = list(base.coef_names)
    rows, t0 = [], time.time()
    for rep in range(args.reps):
        rd, rt, div, rhat = one_replicate(base, priors, 1000 + rep, args.draws)
        rows.append({"rep": rep, "divergences": div, "rhat_max": rhat,
                     **{f"mean[{n}]": r for n, r in zip(names, rd, strict=True)},
                     **{f"sd[{n}]": r for n, r in zip(names, rt, strict=True)}})
        print(f"rep {rep + 1}/{args.reps} done ({time.time() - t0:.0f}s)", flush=True)
    ranks = pd.DataFrame(rows)
    out = Path(args.out)
    ranks.to_csv(out / "sbc_ranks.csv", index=False)
    write_report(out / "sbc_report.md", ranks, names, args, priors, time.time() - t0)


def write_report(path, ranks, names, args, priors, seconds) -> None:
    bins = 10
    lines = [
        "# Simulation-based calibration",
        "",
        f"Run {dt.date.today().isoformat()} with partworth {pw.__version__}: "
        f"{args.reps} replicates, 60 respondents x 10 tasks x 3 alternatives + none, "
        f"coefficients {names}, priors mu_sd={priors.mu_sd}, tau_sd={priors.tau_sd}, "
        f"lkj_eta={priors.lkj_eta}; 2 chains x {args.draws} draws; ranks among "
        f"{N_RANK} thinned draws; {seconds / 60:.0f} minutes.",
        "",
        "Ranks of the true value should be uniform on 0..99. The p-value is a chi-square "
        f"test of uniformity over {bins} bins; with many parameters, a few p-values below "
        "0.05 are expected by chance.",
        "",
        "| parameter | chi-square p | mean rank (expected 49.5) | counts per bin |",
        "|---|---|---|---|",
    ]
    for col in [c for c in ranks.columns if c.startswith(("mean[", "sd["))]:
        counts = np.histogram(ranks[col], bins=bins, range=(0, N_RANK + 1))[0]
        p = stats.chisquare(counts).pvalue
        lines.append(f"| {col} | {p:.3f} | {ranks[col].mean():.1f} | "
                     f"{' '.join(str(c) for c in counts)} |")
    lines += [
        "",
        f"Replicates with divergences: {(ranks['divergences'] > 0).sum()} of {len(ranks)}. "
        f"Max R-hat over replicates: {ranks['rhat_max'].max():.3f}.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
