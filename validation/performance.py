"""Record sampling time and peak memory for typical study sizes (P-01, P-02).

    python validation/performance.py
"""

from __future__ import annotations

import datetime as dt
import platform
import resource
import time
import warnings
from pathlib import Path

import partworth as pw
from partworth.datasets import simulate_cbc

ATTRS = {"brand": ["a", "b", "c", "d"], "size": ["s", "m", "l"], "colour": ["x", "y", "z"],
         "feature": ["no", "yes"], "price": [1, 2, 3, 4, 5]}


def main() -> None:
    rows = []
    for n in (300, 1000):
        data, _ = simulate_cbc(n, 12, 4, ATTRS, price="price", seed=n)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pw.HierarchicalMNL(data).fit(draws=1000, tune=1000, chains=4, random_seed=1)
        secs = time.time() - t0
        d = res.diagnostics()
        peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
        rows.append(f"| {n} | 12 | 4 + none | {data.n_params} | {secs:.0f} | {peak_gb:.1f} | "
                    f"{d.rhat_max:.3f} | {d.ess_bulk_min:.0f} | {d.divergences} |")
        print(rows[-1], flush=True)
    out = Path("validation/results/performance.md")
    out.write_text("\n".join([
        "# Performance",
        "",
        f"Run {dt.date.today().isoformat()}, partworth {pw.__version__}, "
        f"{platform.platform()}, {platform.processor() or platform.machine()}. "
        "4 chains x 1000 draws after 1000 tuning steps, nutpie, full covariance. "
        "Peak memory is the process maximum so far (cumulative across rows).",
        "",
        "| respondents | tasks | alternatives | coefficients | seconds | peak memory (GB) "
        "| R-hat max | ESS bulk min | divergences |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]))


if __name__ == "__main__":
    main()
