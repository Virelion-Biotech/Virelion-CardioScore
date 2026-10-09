# Scientific audit changes — 2026-10-09

## Behavior

Label serialized weighted outputs as research scores and report categories as heuristic. Stop counting technical wells as independent units when no unit column is supplied. Reject nonfinite score weights and effect thresholds.

## Scope and remaining evidence

Legacy risk_class keys remain for compatibility and are explicitly labeled as heuristic. Reference-panel locking choices and external calibration/transportability require scientific decisions and datasets; no qualified drug-risk probability is introduced.

## Implementation

- `tests/test_direct_scoring_units.py`
- `virelion_cardioscore/analysis/cipa_scoring.py`
- `virelion_cardioscore/reporting/report_generator.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
