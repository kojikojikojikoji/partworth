"""Compare partworth with R bayesm on the public camera data (V-01 to V-04).

Requires R with the bayesm package (``install.packages("bayesm")``). bayesm is run in a
separate process; its code is not used by partworth.

    python validation/compare_camera.py --out validation/results

Coefficients are put on bayesm's parameterisation before comparing: bayesm has one
dummy per brand against the outside option, partworth has brand dummies against canon
plus NONE, so  NONE = -canon  and  brand[b] = b - canon.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

import partworth as pw
from partworth.datasets import load_camera

HOLDOUT = [15, 16]
BRANDS = ["canon", "sony", "nikon", "panasonic"]
FEATURES = ["pixels", "zoom", "video", "swivel", "wifi"]


def to_bayesm_space(beta: np.ndarray, names: list[str]) -> pd.DataFrame:
    """partworth coded coefficients (..., K) -> bayesm columns."""
    b = pd.DataFrame(beta.reshape(-1, beta.shape[-1]), columns=names)
    out = pd.DataFrame(index=b.index)
    canon = -b["NONE"]
    out["canon"] = canon
    for br in BRANDS[1:]:
        out[br] = b[f"brand[{br}]"] + canon
    for f in FEATURES:
        out[f] = b[f"{f}[yes]"]
    out["price"] = b["price"]
    return out


def hit_rate(beta: np.ndarray, data: pw.ChoiceData) -> float:
    util = np.einsum("ntjk,nk->ntj", data.X, beta)
    util = np.where(data.mask, util, -np.inf)
    ii, tt = np.nonzero(data.task_mask)
    return float((util[ii, tt].argmax(axis=1) == data.y[ii, tt]).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/results")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--bayesm-R", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_camera(price_sign=None)
    train, test = data.holdout(HOLDOUT)

    t0 = time.time()
    subprocess.run(["Rscript", str(Path(__file__).with_name("run_bayesm.R")), str(out),
                    str(16 - len(HOLDOUT)), str(args.bayesm_R), "10", str(args.seed)],
                   check=True)
    t_bayesm = time.time() - t0
    bm_beta = pd.read_csv(out / "bayesm_beta_mean.csv")
    bm_mu = pd.read_csv(out / "bayesm_mu_draws.csv")

    results = {}
    settings = {
        "default priors": pw.Priors(),
        "wide priors (mu_sd=10, tau_sd=2, lkj_eta=1)": pw.Priors(mu_sd=10.0, tau_sd=2.0,
                                                                lkj_eta=1.0),
    }
    for label, priors in settings.items():
        t0 = time.time()
        res = pw.HierarchicalMNL(train, priors=priors).fit(
            draws=args.draws, tune=args.tune, chains=4, random_seed=args.seed)
        elapsed = time.time() - t0
        names = list(data.coef_names)
        delta = res.idata["posterior"]["Delta"].isel(covariate=0)
        mu_draws = to_bayesm_space(
            delta.stack(s=("chain", "draw")).transpose("s", ...).to_numpy(), names)
        pw_beta = to_bayesm_space(res.beta_mean(), names)
        cols = list(bm_beta.columns)
        table = pd.DataFrame({
            "partworth_mean": mu_draws[cols].mean(),
            "partworth_sd": mu_draws[cols].std(),
            "bayesm_mean": bm_mu[cols].mean(),
            "bayesm_sd": bm_mu[cols].std(),
        })
        table["diff_in_sd"] = ((table["partworth_mean"] - table["bayesm_mean"])
                               / table["bayesm_sd"])
        table["respondent_corr"] = [np.corrcoef(pw_beta[c], bm_beta[c])[0, 1] for c in cols]
        d = res.diagnostics()
        bm_coded = np.column_stack([
            bm_beta[[f"{b}" for b in BRANDS[1:]]].to_numpy()
            - bm_beta[["canon"]].to_numpy(),
            bm_beta[FEATURES].to_numpy(), bm_beta["price"].to_numpy(),
            -bm_beta["canon"].to_numpy()])
        results[label] = {
            "table": table,
            "hit_partworth": hit_rate(res.beta_mean(), test),
            "hit_bayesm": hit_rate(bm_coded, test),
            "seconds": elapsed,
            "diag": d,
        }
        slug = label.split(" (")[0].replace(" ", "_")
        table.to_csv(out / f"camera_{slug}.csv")

    write_report(out / "camera_report.md", results, t_bayesm, args)


def write_report(path: Path, results: dict, t_bayesm: float, args) -> None:
    version = (path.parent / "bayesm_version.txt").read_text().strip()
    lines = [
        "# partworth vs bayesm on the camera data",
        "",
        f"Run {dt.date.today().isoformat()} with partworth {pw.__version__} and {version}.",
        f"Estimation tasks 1-14, hold-out tasks {HOLDOUT}. bayesm: R={args.bayesm_R}, "
        f"keep=10, first half discarded ({t_bayesm:.0f}s). partworth: 4 chains x "
        f"{args.draws} draws after {args.tune} tuning steps (nutpie).",
        "",
        "The two packages use different algorithms and priors. bayesm uses random-walk "
        "Metropolis within Gibbs and an inverse-Wishart prior on the covariance. "
        "partworth uses NUTS and an LKJ prior. We use the comparison to check that the "
        "model is implemented correctly. The second setting widens partworth's priors. "
        "It does not copy bayesm's prior, so its larger differences show how much the "
        "priors matter.",
        "",
    ]
    for label, r in results.items():
        t = r["table"]
        lines += [
            f"## {label}",
            "",
            f"- max |difference in population means| / bayesm posterior sd: "
            f"{t['diff_in_sd'].abs().max():.2f} "
            "(target 0.1; reported, not enforced: the priors differ)",
            f"- min correlation of respondent posterior means: "
            f"{t['respondent_corr'].min():.3f} (target 0.9)",
            f"- hold-out hit rate: partworth {r['hit_partworth']:.3f}, "
            f"bayesm {r['hit_bayesm']:.3f} (target difference within 0.02)",
            f"- partworth time {r['seconds']:.0f}s; diagnostics: R-hat max "
            f"{r['diag'].rhat_max:.3f}, ESS bulk min {r['diag'].ess_bulk_min:.0f}, "
            f"divergences {r['diag'].divergences}",
            "",
            t.round(3).to_markdown(),
            "",
        ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
