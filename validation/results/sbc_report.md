# Simulation-based calibration

Run 2026-10-13 with partworth 0.0.1.dev0: 200 replicates, 60 respondents x 10 tasks x 3 alternatives + none, coefficients ['brand[x]', 'brand[y]', 'price', 'NONE'], priors mu_sd=1.0, tau_sd=0.5, lkj_eta=2.0; 4 chains x 500 draws after 1000 tuning steps; ranks among 99 thinned draws; 18 minutes.

Ranks of the true value should be uniform on 0..99. The p-value is a chi-square test of uniformity over 10 bins; with many parameters, a few p-values below 0.05 are expected by chance.

| parameter | chi-square p | mean rank (expected 49.5) | counts per bin |
|---|---|---|---|
| mean[brand[x]] | 0.941 | 50.2 | 16 23 19 17 20 24 23 21 20 17 |
| mean[brand[y]] | 0.843 | 49.6 | 25 19 16 18 23 21 15 18 23 22 |
| mean[price] | 0.930 | 49.0 | 21 21 17 21 22 18 24 18 15 23 |
| mean[NONE] | 0.208 | 54.1 | 18 14 17 17 24 17 22 15 27 29 |
| sd[brand[x]] | 0.911 | 50.1 | 17 24 17 18 22 19 22 21 24 16 |
| sd[brand[y]] | 0.002 | 49.5 | 33 22 10 12 22 15 15 26 16 29 |
| sd[price] | 0.616 | 49.2 | 20 26 18 24 16 13 18 19 23 23 |
| sd[NONE] | 0.534 | 46.0 | 27 21 16 20 22 24 20 21 17 12 |

Replicates with divergences: 1 of 200. R-hat max per replicate (over all parameters, including respondent-level coefficients and correlations): median 1.027, above 1.05 in 36 of 200, maximum 1.562.
