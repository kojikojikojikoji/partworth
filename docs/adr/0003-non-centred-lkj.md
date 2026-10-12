# ADR-0003: Non-centred parameterisation and LKJ prior

- Status: accepted
- Date: 2026-10-12

## Decision

`beta_i = Delta' z_i + diag(tau) L eta_i`, `eta_i ~ N(0, I)`, `L ~ LKJCholesky(eta=2)`,
`tau ~ HalfNormal(1)`, population mean `~ N(0, 2)`, covariate effects `~ N(0, 1)` with
covariates centred.

## Reasons

- CBC data give each respondent 8-20 tasks, so individual likelihoods are weak and the
  centred parameterisation produces funnels and divergences under NUTS
  (Betancourt and Girolami 2015). The non-centred form samples cleanly.
- The LKJ prior separates scale (tau) from correlation, so each can be given a weakly
  informative prior. eta = 2 mildly favours small correlations; `lkj_eta=1` is uniform.
- Priors are on the logit scale of coded variables. With effects coding and linear terms
  standardised, a N(0, 2) mean prior keeps prior choice shares away from 0/1.
- bayesm's inverse-Wishart prior is not reproduced; validation compares results under a
  "bayesm-like" setting (wide mean prior, lkj_eta = 1) and reports the difference.

`covariance="diagonal"` is offered for studies with many sparse levels.
