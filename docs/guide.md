# partworth user guide

This guide covers the decisions an analyst makes between "I have CBC responses" and
"I have shares and WTP I can defend". Each section says what partworth does by default
and when to change it.

## 1. Preparing the data

partworth takes a **long table**: one row per respondent x task x alternative.

| column | meaning |
|---|---|
| `respondent` | respondent id (string or integer) |
| `task` | task id, unique within a respondent |
| `alternative` | alternative id, unique within a task |
| `chosen` | 1 for the chosen alternative, else 0 (exactly one per task) |
| `none` | optional; True on the "no purchase" row (its attribute cells may be empty) |
| attribute columns | one column per attribute |

Tasks may have different numbers of alternatives, and respondents different numbers of
tasks; partworth pads internally and masks the padding.

If your design is in wide form (one row per task, columns such as `alt1_price`), as
produced by [ChoiceDesign](https://choicedesign.readthedocs.io/), convert it:

```python
from partworth.io import wide_design_to_long, attach_responses

long_design = wide_design_to_long(design, pattern="alt{alt}_{attr}", none=True,
                                  levels={"colour": {0: "red", 1: "green", 2: "blue"}})
long = attach_responses(long_design, answers, block="block")   # answers: respondent, task, choice
```

Always look at `data.describe()` before estimating. A level that was shown rarely, or
almost never chosen, is the most common cause of poor convergence.

## 2. Coding attributes

- **Categorical, effects coding (default).** Levels of an attribute sum to zero, so
  every level has an estimate and the attribute's mean utility is zero. This is the usual
  choice for reporting and matches Sawtooth's raw utilities.
- **Categorical, dummy coding.** One reference level is fixed at zero. Use it when a
  natural baseline exists (feature absent / present) or to match another tool.
- **Linear.** One coefficient per unit, for numeric attributes such as price. partworth
  centres and scales the values for estimation and reports the slope per original unit.
- **Sign constraints.** `Linear(sign="negative")` makes the coefficient negative for every
  respondent (lognormal). Use it for price when a positive price effect would not be
  credible; it keeps WTP finite. The population mean for such a term is reported on the
  log scale (`HBResult.population()` marks it).

Do not use a sign constraint to hide a design problem: if many respondents genuinely
prefer the higher price (prestige goods, price as a quality signal), estimate without it
first and look at the respondent-level slopes.

## 3. Priors

Defaults (`pw.Priors()`): population mean N(0, 2), covariate effects N(0, 1),
between-respondent SDs HalfNormal(1), LKJ(2) on correlations, full covariance.

- These are weakly informative on the logit scale. A prior predictive check shows what
  they imply: `HierarchicalMNL(data).prior_predictive()`.
- With many attributes and few tasks per respondent, try `covariance="diagonal"`: it has
  far fewer parameters and is often enough for simulation.
- Results are not sensitive to these choices on typical CBC data (see
  `validation/results/camera_report.md`, where a much wider setting changed population
  means by under one posterior SD and left respondent-level estimates almost unchanged).

## 4. Estimation and convergence

```python
result = pw.HierarchicalMNL(data, covariates=True).fit(draws=1000, tune=2000, chains=4)
print(result.diagnostics().report())
```

partworth warns if R-hat >= 1.01, bulk ESS <= 400 or any divergence occurs, and the report
lists what to try. In order:

1. **Divergences:** `fit(target_accept=0.95)` (up to 0.99). If they persist, tighten
   `tau_sd` (e.g. 0.5) or switch to `covariance="diagonal"`.
2. **Low ESS / high R-hat:** more draws and tuning (2000/2000). Correlations are usually
   the slowest parameters; respondent-level part-worths converge first.
3. **Sparse levels:** merge levels that were rarely shown or chosen, or drop them.

Correlations between coefficients mix more slowly than everything else. They are
reported on their own line (`corr_rhat_max`, `corr_ess_bulk_min`) and do not trigger the
warning, because part-worths, shares and WTP do not depend on them directly. Before you
report correlations, rerun with more draws (e.g. `draws=3000`) until that line also meets
the targets. In our 300-respondent benchmark the default run gave correlation ESS of
about 220 while all other parameters exceeded 400.

## 5. Outputs

| call | what it is | use it for |
|---|---|---|
| `part_worths()` | raw logit-scale utilities per respondent | simulators |
| `part_worths("zc_diffs")` | zero-centred diffs (ranges average 100 per attribute) | comparing respondents, segmentation |
| `importances()` | attribute range as % of the sum of ranges | reporting |
| `rlh()` | root likelihood (geometric mean probability of choices) | spotting random responders (RLH near 1 / number of alternatives) |
| `population()` | population means and SDs with 90% intervals | describing the market |
| `predict(holdout)` | probability of the chosen alternative and hit per hold-out task | validation |
| `to_sawtooth_csv(path)` | Sawtooth-style utilities file | handing results to existing tools |

## 6. Simulation and WTP

```python
sim = pw.MarketSimulator(result)            # share of preference (logit)
sim.shares(scenario)                         # share, lower, upper (90% credible interval)
sim.price_curve(scenario, product="Ours", grid=[2.0, 2.5, 3.0])
pw.wtp(result, attribute="size", from_level="S", to_level="L")
```

- Intervals come from the posterior draws, so they show estimation uncertainty; they do
  not include uncertainty about how the market differs from the survey.
- The logit rule has the IIA property within a respondent: adding a near-copy of a product
  takes share proportionally from all others. Heterogeneity across respondents softens
  this. `rule="first_choice"` avoids it but exaggerates differences.
- `exponent` rescales utilities before the logit rule (the usual "scale factor" tuning
  against known market shares).
- WTP is reported as the median over respondents. Means of ratios are unstable when some
  price slopes are close to zero; `share_negative_price_slope` shows how many respondents
  have the expected sign.

## 7. Reproducibility

Pass `random_seed` to `fit()`. `HBResult.save()` writes the posterior to NetCDF;
`HBResult.load(path, data)` restores it.
