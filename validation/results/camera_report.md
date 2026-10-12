# partworth vs bayesm on the camera data

Run 2026-10-13 with partworth 0.0.1.dev0 and bayesm 3.1.7.
Estimation tasks 1-14, hold-out tasks [15, 16]. bayesm: R=20000, keep=10, first half discarded (33s). partworth: 4 chains x 1000 draws after 2000 tuning steps (nutpie).

The two packages use different algorithms and priors. bayesm uses random-walk Metropolis within Gibbs and an inverse-Wishart prior on the covariance. partworth uses NUTS and an LKJ prior. We use the comparison to check that the model is implemented correctly. The second setting widens partworth's priors. It does not copy bayesm's prior, so its larger differences show how much the priors matter.

## default priors

- max |difference in population means| / bayesm posterior sd: 0.21 (target 0.1; reported, not enforced: the priors differ)
- min correlation of respondent posterior means: 0.997 (target 0.9)
- hold-out hit rate: partworth 0.729, bayesm 0.730 (target difference within 0.02)
- partworth time 74s; diagnostics: R-hat max 1.011, ESS bulk min 897, divergences 0

|           |   partworth_mean |   partworth_sd |   bayesm_mean |   bayesm_sd |   diff_in_sd |   respondent_corr |
|:----------|-----------------:|---------------:|--------------:|------------:|-------------:|------------------:|
| canon     |            1.916 |          0.329 |         1.955 |       0.356 |       -0.108 |             0.999 |
| sony      |            1.492 |          0.342 |         1.535 |       0.363 |       -0.117 |             0.999 |
| nikon     |            1.643 |          0.338 |         1.682 |       0.357 |       -0.11  |             0.999 |
| panasonic |            1.107 |          0.337 |         1.158 |       0.367 |       -0.141 |             0.999 |
| pixels    |            1.393 |          0.133 |         1.383 |       0.132 |        0.081 |             0.998 |
| zoom      |            1.749 |          0.138 |         1.74  |       0.14  |        0.063 |             0.998 |
| video     |            1.295 |          0.11  |         1.291 |       0.112 |        0.036 |             0.997 |
| swivel    |            0.692 |          0.113 |         0.668 |       0.113 |        0.213 |             0.998 |
| wifi      |            1.199 |          0.123 |         1.202 |       0.119 |       -0.029 |             0.997 |
| price     |           -3.382 |          0.174 |        -3.404 |       0.178 |        0.121 |             0.999 |

## wide priors (mu_sd=10, tau_sd=2, lkj_eta=1)

- max |difference in population means| / bayesm posterior sd: 0.92 (target 0.1; reported, not enforced: the priors differ)
- min correlation of respondent posterior means: 0.998 (target 0.9)
- hold-out hit rate: partworth 0.724, bayesm 0.730 (target difference within 0.02)
- partworth time 74s; diagnostics: R-hat max 1.005, ESS bulk min 820, divergences 0

|           |   partworth_mean |   partworth_sd |   bayesm_mean |   bayesm_sd |   diff_in_sd |   respondent_corr |
|:----------|-----------------:|---------------:|--------------:|------------:|-------------:|------------------:|
| canon     |            2.035 |          0.355 |         1.955 |       0.356 |        0.224 |             0.999 |
| sony      |            1.603 |          0.367 |         1.535 |       0.363 |        0.187 |             0.999 |
| nikon     |            1.755 |          0.363 |         1.682 |       0.357 |        0.202 |             0.999 |
| panasonic |            1.199 |          0.364 |         1.158 |       0.367 |        0.111 |             0.999 |
| pixels    |            1.446 |          0.143 |         1.383 |       0.132 |        0.481 |             0.998 |
| zoom      |            1.829 |          0.142 |         1.74  |       0.14  |        0.634 |             0.998 |
| video     |            1.347 |          0.119 |         1.291 |       0.112 |        0.495 |             0.998 |
| swivel    |            0.712 |          0.119 |         0.668 |       0.113 |        0.393 |             0.998 |
| wifi      |            1.25  |          0.132 |         1.202 |       0.119 |        0.399 |             0.998 |
| price     |           -3.567 |          0.19  |        -3.404 |       0.178 |       -0.915 |             0.999 |
