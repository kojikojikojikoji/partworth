# Simulation-based calibration

Run 2026-10-13 with partworth 0.0.1.dev0: 200 replicates, 60 respondents x 10 tasks x 3 alternatives + none, coefficients ['brand[x]', 'brand[y]', 'price', 'NONE'], priors mu_sd=1.0, tau_sd=0.5, lkj_eta=2.0; seeds 9000.. ; diagonal mass matrix; 4 chains x 2000 draws after 1000 tuning steps; ranks among 99 thinned draws; 29 minutes.

Ranks of the true value should be uniform on 0..99. The p-value is a chi-square test of uniformity over 10 bins; with many parameters, a few p-values below 0.05 are expected by chance.

| parameter | chi-square p | mean rank (expected 49.5) | counts per bin |
|---|---|---|---|
| mean[brand[x]] | 0.017 | 47.8 | 35 19 8 22 17 17 20 20 23 19 |
| mean[brand[y]] | 0.342 | 46.5 | 24 18 24 25 24 14 20 18 11 22 |
| mean[price] | 0.596 | 51.1 | 16 19 20 23 17 25 12 22 24 22 |
| mean[NONE] | 0.891 | 50.3 | 17 20 17 23 26 20 17 18 19 23 |
| sd[brand[x]] | 0.304 | 50.8 | 22 17 17 17 16 28 24 13 26 20 |
| sd[brand[y]] | 0.585 | 49.5 | 16 25 19 18 21 22 17 19 28 15 |
| sd[price] | 0.149 | 49.6 | 17 21 25 19 14 20 32 19 13 20 |
| sd[NONE] | 0.534 | 50.1 | 24 20 19 17 13 25 16 22 18 26 |

Replicates with divergences: 3 of 200. R-hat max per replicate (over all parameters, including respondent-level coefficients and correlations): median 1.007, above 1.05 in 15 of 200, maximum 1.456. Minimum bulk ESS of the population parameters per replicate: median 664, below 99 in 15 of 200 (ranks from autocorrelated draws are not uniform even for a correct sampler).
