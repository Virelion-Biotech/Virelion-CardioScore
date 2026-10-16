# Blinova 2018 workbook audit

## Scope

This audit covers `Blinova_etal_2018_data.xlsx` as a **processed workbook**, now verified byte for byte against the official CiPA download. The workbook is not archived in the repository and is not treated as the immutable CiPA raw archive.

Reference: Blinova et al. 2018, *Cell Reports*, DOI 10.1016/j.celrep.2018.08.079.

## File identity

- Filename: `Blinova_etal_2018_data.xlsx`
- SHA-256: `5fd91c2fa1c9b7fbb2f12cf836a061853e0c6a7af22b62965973198d9066459b`
- Rows: 8,859
- Columns: `Drug_Name`, `Cell_type`, `risk`, `Platform`, `Type_of_EADs`, `conc`, `EAD`, `ddFPDc`, `site`

## Coverage observed

- 29 raw drug strings; 28 after treating `D,l,Sotalol` and `D,l Sotalol` as a known spelling alias for audit purposes only.
- Sites 1–10.
- Cell types `CDI` and `AXG`.
- Concentrations 1–4.
- Observed platform codes: `ACA`, `AMD`, `AXN`, `CLY`, `ECR`, `MCS`.
- Risk codes observed: `H`, `M`, `L`, plus one missing value.
- 1,887 rows have `EAD=1`.
- 2,067 rows have missing `ddFPDc`.

## Findings

### 1. Drug spelling alias

One row uses `D,l,Sotalol` while the remaining sotalol rows use `D,l Sotalol`. This is reported but not silently rewritten in the source asset.

### 2. Missing risk label

One row is missing `risk`:

`D,l,Sotalol / CDI / site 6 / concentration 1 / EAD 0 / ddFPDc 28.2979749885`

The published panel identifies D,L-sotalol as high TdP risk. The missing cell is therefore a data-quality issue, but the reference label must not be injected into the source record until provenance is established.

### 3. Platform-code mismatch

The workbook contains platform code `ACA`. The paper's documented platform abbreviations are AXN, CLY, ECR, AMD, and MCS. No silent mapping from `ACA` to another platform is permitted.

### 4. Terfenadine: apparent event discrepancy resolved

The verified workbook has 46 EAD=1 rows across 10 of 15 site/cell datasets, all labeled Q (quiescence). It has no resolved A–D arrhythmia labels for terfenadine. Since the paper distinguishes quiescence from arrhythmia-like events, these rows do not contradict its no-arrhythmia statement. EAD is an any-event-label indicator in this workbook, not an arrhythmia-specific truth label.

### 5. Verapamil: apparent arrhythmia discrepancy resolved

All 34 EAD=1 rows across six of 15 site/cell datasets are Q-only, with no resolved A–D labels. This resolves the apparent arrhythmia discrepancy; it does not independently establish the paper's separate repolarization claim or validate a CardioScore prediction.

### 6. Dofetilide should not be reduced to EAD count

The paper reports statistically significant repolarization prolongation **or** arrhythmia-like events in 14 of 15 datasets for dofetilide. The workbook has `EAD=1` in 12 of 15 datasets. These are not directly contradictory because the published claim is an OR of two endpoint families, whereas this workbook exposes a single `EAD` flag plus `ddFPDc`.

### 7. Missing ddFPDc

The paper specifies that ddFPDc/APD90c was not calculated and was designated missing when at least 50% of included wells were arrhythmic after dosing. The workbook contains 2,067 missing ddFPDc values, including every row with `EAD=1`. The workbook does not expose enough information in a dedicated denominator field to independently prove the 50% rule for every missing record, so missingness is preserved rather than reconstructed by assumption.

## Decision

**Status: readiness audit complete; predictive/raw-waveform qualification remains blocked.**

The official processed workbook is suitable for the documented semantic/reference audit. It is not the raw MEA archive and cannot supply all six CardioScore endpoints or independent waveform truth. The official URL and hash are recorded in the [Colab review](COLAB_CIPA_REVIEW_2026-10-17.md).

The 2026-10-17 correction keeps all 52 unresolved labels (`T`, `1AP`, `BCT`) as null typed-event indicators. Only fully resolved tokens contribute to typed counts: 1,286 A–D rows and 550 Q rows, with one AQ overlap. The old 1,287 count included an unjustified partial interpretation of BCT. All original rows and outputs remain archived separately.

Still required: authoritative meanings for unresolved codes and ACA, complete endpoint/exposure/replicate metadata, independent raw-waveform truth, frozen score/calibration, and an untouched compound/site evaluation with uncertainty. Missing ddFPDc and source risk cells remain missing. Do not tune CardioScore from these audit observations or treat the publication's performance as validation of this implementation.
