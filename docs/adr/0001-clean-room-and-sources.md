# ADR-0001: Clean-room development and the sources it relies on

- Status: accepted
- Date: 2026-10-12

## Context

partworth must contain only public or simulated material. It must not contain code,
data, attribute sets or results from non-public studies. It must also stay compatible
with Apache-2.0, so GPL code (R `bayesm`, `apollo`) cannot be ported.

## Decision

1. partworth is developed in its own directory, away from other work. A local settings
   file, which is not committed, stops the AI coding assistant from reading the author's
   private notes on conjoint analysis.
2. Only the public sources listed below are used. Anything else must be added to this
   list in the same pull request that relies on it.
3. GPL software is used only as an external program for validation. R runs in a
   separate process from `validation/`. Its data is downloaded at run time. It is never
   committed or distributed.
4. Examples and tests use only public data (`bayesm::camera`) and simulated data.

## Public sources

Models and definitions
- Rossi, Allenby and McCulloch (2005), *Bayesian Statistics and Marketing*, Wiley.
- Train (2009), *Discrete Choice Methods with Simulation*, 2nd ed., Cambridge University Press.
- Lewandowski, Kurowicka and Joe (2009), "Generating random correlation matrices based on
  vines and extended onion method", *Journal of Multivariate Analysis* 100(9) (LKJ prior).
- Betancourt and Girolami (2015), "Hamiltonian Monte Carlo for hierarchical models"
  (non-centred parameterisation).
- Talts, Betancourt, Simpson, Vehtari and Gelman (2018), "Validating Bayesian inference
  algorithms with simulation-based calibration", arXiv:1804.06788.
- Sawtooth Software, CBC/HB technical paper and public online manuals (file formats,
  zero-centred diffs, RLH). Cited by section in ADR-0005.

Software documentation (APIs only; no code copied)
- PyMC, nutpie, ArviZ, pandas, xarray documentation.
- pymc-marketing `customer_choice` module and issues #2627, PRs #2490 and #2514
  (Apache-2.0), read for interoperability. See ADR-0002.
- bayesm reference manual, `rhierMnlRwMixture` and `camera` help pages (documentation
  only; the GPL source is not read for porting).
- ChoiceDesign documentation (for the design import in a later release).

Data
- `camera`: Allenby, Brazell, Howell and Rossi (2014), "Economic valuation of product
  features", *Quantitative Marketing and Economics* 12(4); distributed in R `bayesm`.

## Consequences

Contributors must not add examples derived from non-public studies. Reviews check new
examples and fixtures against this rule.
