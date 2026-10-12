# ADR-0004: Sign constraints by a lognormal transform

- Status: accepted
- Date: 2026-10-12

## Decision

A `Linear(sign="negative")` coefficient is `-exp(x_i)` with `x_i` from the multivariate
normal; `"positive"` is `exp(x_i)`. The population mean and SD reported for such terms are
on the log scale (`HBResult.population()` labels them).

## Reasons

- Positive price coefficients for some respondents make WTP and price curves meaningless;
  the lognormal is the standard fix (Train 2009, ch. 6) and keeps the term in the same
  joint normal, so its correlation with other coefficients is still estimated.
- Truncated normals or post-hoc constraints break the joint normal or bias other terms.

## Consequences

`mu` and `tau` are on the log scale of the coefficient per standardised unit of the
attribute, `(x - center) / scale`. The mean coefficient per standardised unit is
`-exp(mu + tau^2 / 2)`, not `-exp(mu)`; divide by `scale` for the coefficient per
original unit (what `HBResult.part_worths()` reports). WTP uses respondent
draws and reports medians, because ratios with a lognormal denominator are heavy-tailed.
