# Parameter recovery on simulated data

Run 2026-10-13 with partworth 0.0.1.dev0. 300 respondents x 12 tasks x 3 alternatives + none; true population means [0.6, -0.4, 0.4, 0.0, 0.3, -0.5], SD 0.6, common correlation 0.3; price lognormal (negative). 4 chains x 1000 draws after 2000 tuning steps (nutpie), seed 3.

Price's population mean and SD are on the log scale (see ADR-0004).

| coefficient | true mean | 90% interval of mean | covered | true SD | 90% interval of SD | covered | corr(respondent means, truth) | respondent 90% coverage |
|---|---|---|---|---|---|---|---|---|
| brand[x] | 0.60 | [0.56, 0.74] | True | 0.60 | [0.46, 0.67] | True | 0.700 | 0.897 |
| brand[y] | -0.40 | [-0.46, -0.27] | True | 0.60 | [0.52, 0.73] | True | 0.638 | 0.897 |
| size[s] | 0.40 | [0.30, 0.48] | True | 0.60 | [0.42, 0.65] | True | 0.677 | 0.883 |
| size[m] | 0.00 | [-0.10, 0.09] | True | 0.60 | [0.51, 0.72] | True | 0.633 | 0.873 |
| price | 0.30 | [0.13, 0.29] | False | 0.60 | [0.57, 0.74] | True | 0.790 | 0.890 |
| NONE | -0.50 | [-0.75, -0.49] | True | 0.60 | [0.47, 0.80] | True | 0.546 | 0.880 |

Overall respondent-level 90% interval coverage: 0.887 (1800 respondent x coefficient pairs).
Diagnostics: R-hat max 1.013, ESS bulk min 399, divergences 0.
