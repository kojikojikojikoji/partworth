# Changelog

## Unreleased

- Example notebooks on the public camera data, in English and Japanese

## 0.0.1 (2026-10-14)

First release on PyPI.

- `ChoiceData.from_long`: validated long-format input, ragged tasks/alternatives, none
  option, respondent covariates, `subset`, `holdout`, `describe`
- Effects, dummy and linear coding with sign constraints
- `HierarchicalMNL`: non-centred hierarchical MNL with LKJ covariance, covariate
  regression of the population mean, lognormal sign constraints, nutpie or PyMC NUTS
- `HBResult`: part-worths, zero-centred diffs, importances, RLH, population summary,
  diagnostics with guidance, hold-out prediction, Sawtooth-style CSV, save/load
- `MarketSimulator` (logit and first choice, credible intervals, weights, price curves,
  shares for each posterior draw with `share_draws`) and `wtp`
- `datasets.load_camera` (fetched from CRAN at run time) and `datasets.simulate_cbc`
- Validation against R bayesm (`validation/`)
- Example notebooks on a dummy retail survey, in English and Japanese (`examples/`)
