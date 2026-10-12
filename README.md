# partworth

Hierarchical Bayes (HB) estimation for choice-based conjoint (CBC) studies in Python,
built on [PyMC](https://www.pymc.io/). partworth is aimed at analysts who run CBC studies
and want the whole workflow — data checks, HB estimation, respondent-level part-worths,
Sawtooth-style output files, a market simulator and willingness to pay — in one Python
package, with the validation results published alongside the code.

> **Status: pre-alpha.** The API is being built in the open. v0.1 is planned for the end
> of November 2026 and v0.2 for the end of December 2026. Nothing here is stable yet.

## What it does (planned for v0.1 / v0.2)

| | v0.1 | v0.2 |
|---|---|---|
| Long-format choice data with validation, ragged tasks, a none option | ✓ | |
| Effects, dummy and linear coding; sign-constrained price | ✓ | |
| Hierarchical MNL: full covariance (LKJ), non-centred, respondent covariates, nutpie sampler | ✓ | |
| Part-worths, zero-centred diffs, importances, RLH, convergence diagnostics | ✓ | |
| Sawtooth-style utilities CSV (read and write) | ✓ | |
| Market simulator with posterior intervals, price curves, WTP | | ✓ |
| Validation report against R `bayesm` on public data, recovery tests, SBC | | ✓ |

Out of scope: experimental design (use [ChoiceDesign](https://choicedesign.readthedocs.io/)),
latent class, MaxDiff, ACBC.

## Quick look

```python
import pandas as pd
import partworth as pw

# one row per respondent x task x alternative
data = pw.ChoiceData.from_long(
    df,
    attributes={
        "brand": pw.Categorical(["A", "B", "C"]),
        "size": pw.Categorical(["S", "M", "L"]),
        "price": pw.Linear(sign="negative"),   # lognormal: negative for everyone
    },
    none="is_none",                            # optional "no purchase" alternative
    covariates=respondents,                    # optional, one row per respondent
)
data.describe()                                # level counts and choice rates

result = pw.HierarchicalMNL(data).fit(draws=1000, tune=1000, chains=4, random_seed=1)
print(result.diagnostics().report())          # R-hat, ESS, divergences, what to try

result.part_worths()                           # raw utilities, one column per level
result.part_worths("zc_diffs")                 # zero-centred diffs
result.importances()
result.to_sawtooth_csv("utilities.csv")

sim = pw.MarketSimulator(result)
scenario = pd.DataFrame({"product": ["Ours", "Theirs"], "brand": ["A", "B"],
                         "size": ["M", "M"], "price": [3.0, 2.5]})
sim.shares(scenario)                           # shares with 90% credible intervals
sim.price_curve(scenario, product="Ours", grid=[2.0, 2.5, 3.0, 3.5])
pw.wtp(result, attribute="size", from_level="S", to_level="L")
```

A runnable example on public data is in `examples/camera.py`.

## Validation

- Parameter recovery on simulated data (`tests/statistical`, run with `--run-slow`):
  population means and SDs inside 90% intervals, respondent-level 90% intervals cover the
  true values 85-95% of the time, no divergences.
- Comparison with R `bayesm::rhierMnlRwMixture` on the camera data
  (`validation/compare_camera.py`, report in `validation/results/camera_report.md`).
  With default priors (2026-10-12): respondent-level posterior means correlate at
  0.997 or more for every coefficient, population means differ by at most 0.15 posterior
  SDs, and hold-out hit rates are 0.726 (partworth) vs 0.730 (bayesm).

## Installation

```
pip install "partworth[nutpie,datasets]"
```

Requires Python 3.12+. On macOS 26 or later with PyTensor 3.3.3 or earlier, set
`PYTENSOR_FLAGS=cxx=` if you see `ld: library 'd64' not found` (fixed upstream in
pymc-devs/pytensor#2432, not yet released).

## Data and licences

partworth is Apache-2.0. The `camera` example data belongs to the R package
[bayesm](https://cran.r-project.org/package=bayesm) (GPL); it is **downloaded from CRAN
on first use** and is never bundled. bayesm code is not ported; comparisons run R in a
separate process (`validation/`).

All examples use public or simulated data only.

## How this is built

Design decisions are recorded in `docs/adr/`. Parts of the code are written with the help
of an AI coding assistant; the model, priors and validation criteria are the author's
decisions and are documented in the ADRs.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
