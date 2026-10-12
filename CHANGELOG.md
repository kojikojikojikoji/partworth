# Changelog

## Unreleased

- `ChoiceData.from_long`: validated long-format input, ragged tasks/alternatives, none
  option, respondent covariates, `subset`, `holdout`, `describe`
- Effects, dummy and linear coding with sign constraints
- `HierarchicalMNL`: non-centred hierarchical MNL with LKJ covariance, covariate
  regression of the population mean, lognormal sign constraints, nutpie or PyMC NUTS
- `HBResult`: part-worths, zero-centred diffs, importances, RLH, population summary,
  diagnostics with guidance, hold-out prediction, Sawtooth-style CSV, save/load
- `MarketSimulator` (logit and first choice, credible intervals, weights, price curves)
  and `wtp`
- `datasets.load_camera` (fetched from CRAN at run time) and `datasets.simulate_cbc`
- Validation against R bayesm (`validation/`)
