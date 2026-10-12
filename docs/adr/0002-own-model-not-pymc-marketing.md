# ADR-0002: Estimate with partworth's own PyMC model, not pymc-marketing's MixedLogit

- Status: accepted
- Date: 2026-10-12

## Context

pymc-marketing 1.2 ships `customer_choice.MixedLogit`. As of October 2026 it takes a wide
table (one column per alternative), has diagonal random coefficients (no correlation, no
LKJ prior), an intercept-only population mean, and no Sawtooth-style input/output or
WTP. Its roadmap is in pymc-labs/pymc-marketing#2627.

## Decision

partworth builds its own hierarchical MNL directly in PyMC (`models.py`) and keeps
pymc-marketing as an optional extra (`partworth[pmm]`) for comparison only.

## Consequences

- partworth is not broken by API changes in pymc-marketing's experimental module.
- Features that land upstream can replace partworth's implementation later; the public
  partworth API (ChoiceData, HierarchicalMNL, HBResult) does not depend on that choice.
- Python >= 3.12 is required anyway, to stay installable next to pymc-marketing.
