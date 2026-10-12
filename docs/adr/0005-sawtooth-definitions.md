# ADR-0005: Definitions of zero-centred diffs, importances, RLH and the utilities CSV

- Status: accepted
- Date: 2026-10-12

## Sources

- Sawtooth Software, Lighthouse Studio manual, "Export Settings (CBC, ACBC, ACA, CVA)":
  zero-centred diffs are normalised so that, per respondent, utilities sum to zero within
  each attribute and the best-minus-worst differences average 100 across attributes.
  https://legacy.sawtoothsoftware.com/help/lighthouse-studio/manual/analysis_manager_export_settings_conjoint.html
  (retrieved 2026-10-12; the legacy site can be slow or offline)
- Sawtooth Software, Discover help, "CBC utilities": raw utilities are the ones to use in
  market simulators. https://sawtoothsoftware.com/help/discover/analysis/cbc/cbc-utilities
  (retrieved 2026-10-12)

## Decision

- Zero-centred diffs: per respondent, centre categorical levels within attribute; for a
  linear term the range is |slope| x (max - min of the values shown); multiply everything
  (including NONE) by 100 x A / sum of ranges, where A is the number of attributes.
- Importance: an attribute's range as a percentage of the sum of ranges.
- RLH: geometric mean, over a respondent's estimation tasks, of the predicted probability of
  the chosen alternative, using posterior-mean betas. Written x1000 in the CSV.
- Utilities CSV: header row; `Respondent`, `RLH`, then `"<attribute>: <level>"`,
  `"<attribute>"` for linear terms, `None`.

## Limits

partworth calls this "Sawtooth-style". It has not been compared with files produced by
Sawtooth software; users who can are invited to report differences in an issue.
