# partworth vs bayesm on the camera data

Run 2026-10-12 with partworth 0.0.1.dev0 and bayesm 3.1.7.
Estimation tasks 1-14, hold-out tasks [15, 16]. bayesm: R=20000, keep=10, first half discarded (33s). partworth: 4 chains x 1000 draws after 1000 tuning steps (nutpie).

The algorithms differ (bayesm: random-walk Metropolis within Gibbs, inverse-Wishart prior on the covariance; partworth: NUTS, LKJ prior). Agreement is a check that the model is implemented correctly, not a claim that the two are identical. The second setting widens partworth's priors; it does not reproduce bayesm's inverse-Wishart prior, so larger differences there show prior sensitivity.

## default priors

- max |difference in population means| / bayesm posterior sd: 0.15 (target 0.1; reported, not enforced: the priors differ)
- min correlation of respondent posterior means: 0.997 (target 0.9)
- hold-out hit rate: partworth 0.726, bayesm 0.730 (target difference within 0.02)
- partworth time 54s; diagnostics: R-hat max 1.018, ESS bulk min 307, divergences 0

|           |   partworth_mean |   partworth_sd |   bayesm_mean |   bayesm_sd |   diff_in_sd |   respondent_corr |
|:----------|-----------------:|---------------:|--------------:|------------:|-------------:|------------------:|
| canon     |            1.932 |          0.325 |         1.955 |       0.356 |       -0.065 |             0.999 |
| sony      |            1.511 |          0.341 |         1.535 |       0.363 |       -0.066 |             0.999 |
| nikon     |            1.667 |          0.338 |         1.682 |       0.357 |       -0.042 |             0.999 |
| panasonic |            1.129 |          0.333 |         1.158 |       0.367 |       -0.081 |             0.999 |
| pixels    |            1.396 |          0.134 |         1.383 |       0.132 |        0.1   |             0.998 |
| zoom      |            1.754 |          0.138 |         1.74  |       0.14  |        0.098 |             0.998 |
| video     |            1.302 |          0.111 |         1.291 |       0.112 |        0.099 |             0.997 |
| swivel    |            0.684 |          0.112 |         0.668 |       0.113 |        0.147 |             0.998 |
| wifi      |            1.2   |          0.124 |         1.202 |       0.119 |       -0.02  |             0.997 |
| price     |           -3.399 |          0.177 |        -3.404 |       0.178 |        0.028 |             0.999 |

## wide priors (mu_sd=10, tau_sd=2, lkj_eta=1)

- max |difference in population means| / bayesm posterior sd: 0.88 (target 0.1; reported, not enforced: the priors differ)
- min correlation of respondent posterior means: 0.997 (target 0.9)
- hold-out hit rate: partworth 0.724, bayesm 0.730 (target difference within 0.02)
- partworth time 58s; diagnostics: R-hat max 1.011, ESS bulk min 229, divergences 0

|           |   partworth_mean |   partworth_sd |   bayesm_mean |   bayesm_sd |   diff_in_sd |   respondent_corr |
|:----------|-----------------:|---------------:|--------------:|------------:|-------------:|------------------:|
| canon     |            2.052 |          0.371 |         1.955 |       0.356 |        0.274 |             0.999 |
| sony      |            1.62  |          0.379 |         1.535 |       0.363 |        0.234 |             0.999 |
| nikon     |            1.768 |          0.372 |         1.682 |       0.357 |        0.239 |             0.999 |
| panasonic |            1.216 |          0.371 |         1.158 |       0.367 |        0.157 |             0.999 |
| pixels    |            1.443 |          0.14  |         1.383 |       0.132 |        0.456 |             0.998 |
| zoom      |            1.821 |          0.145 |         1.74  |       0.14  |        0.576 |             0.998 |
| video     |            1.348 |          0.12  |         1.291 |       0.112 |        0.505 |             0.998 |
| swivel    |            0.714 |          0.116 |         0.668 |       0.113 |        0.408 |             0.998 |
| wifi      |            1.241 |          0.129 |         1.202 |       0.119 |        0.328 |             0.997 |
| price     |           -3.56  |          0.195 |        -3.404 |       0.178 |       -0.88  |             0.999 |
