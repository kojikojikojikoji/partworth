# Simulation-based calibration

Run 2026-10-13 with partworth 0.0.1.dev0: 200 replicates, 60 respondents x 10 tasks x 3 alternatives + none, coefficients ['brand[x]', 'brand[y]', 'price', 'NONE'], priors mu_sd=1.0, tau_sd=0.5, lkj_eta=2.0; seeds 5000.. ; diagonal mass matrix; 4 chains x 500 draws after 1000 tuning steps; ranks among 99 thinned draws; 22 minutes.

Ranks of the true value should be uniform on 0..99. The p-value is a chi-square test of uniformity over 10 bins; with many parameters, a few p-values below 0.05 are expected by chance.

| parameter | chi-square p | mean rank (expected 49.5) | counts per bin |
|---|---|---|---|
| mean[brand[x]] | 0.534 | 51.3 | 18 17 14 22 23 16 29 21 20 20 |
| mean[brand[y]] | 0.350 | 46.9 | 23 24 23 20 14 21 15 22 26 12 |
| mean[price] | 0.015 | 48.5 | 27 15 33 10 20 16 14 21 20 24 |
| mean[NONE] | 0.668 | 47.6 | 17 21 25 20 22 20 22 23 11 19 |
| sd[brand[x]] | 0.312 | 48.5 | 20 13 29 21 26 15 19 22 18 17 |
| sd[brand[y]] | 0.027 | 51.0 | 21 25 10 18 12 24 31 18 16 25 |
| sd[price] | 0.350 | 50.8 | 21 18 15 13 22 29 24 22 16 20 |
| sd[NONE] | 0.905 | 49.3 | 16 24 20 18 20 25 18 23 18 18 |

Replicates with divergences: 0 of 200. R-hat max per replicate (over all parameters, including respondent-level coefficients and correlations): median 1.024, above 1.05 in 24 of 200, maximum 1.538. Minimum bulk ESS of the population parameters per replicate: median 195, below 99 in 33 of 200 (ranks from autocorrelated draws are not uniform even for a correct sampler).
