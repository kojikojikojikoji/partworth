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
import partworth as pw

data = pw.ChoiceData.from_long(
    df,
    attributes={
        "brand": pw.Categorical(["A", "B", "C"]),
        "price": pw.Linear(sign="negative"),
    },
    none="is_none",
)
data.describe()
```

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
