# Blinova 2018 workbook audit

## Scope

This audit covers the user-supplied `Blinova_etal_2018_data.xlsx` as a **processed-workbook candidate**. The workbook is not archived in the repository and is not treated as the immutable CiPA raw archive.

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

### 4. Published directional-claim mismatch: terfenadine

The paper states that none of the 15 site/cell datasets observed terfenadine-induced arrhythmia-like events, even at concentrations as high as 350-fold Cmax. In this workbook, `EAD=1` occurs in 10 of the 15 reconstructed site/cell datasets for terfenadine.

This is a **material discrepancy**. It may reflect a different meaning of the `EAD` field, a different preprocessing definition, or a non-identical source dataset. It must be resolved before treating this workbook as the validation truth set.

### 5. Published directional-claim mismatch: verapamil

The paper states that no verapamil-induced repolarization prolongation or arrhythmias were observed at any studied concentration. In this workbook, `EAD=1` occurs in 6 of the 15 reconstructed site/cell datasets for verapamil.

This is another **material discrepancy** requiring field-semantic/provenance resolution.

### 6. Dofetilide should not be reduced to EAD count

The paper reports statistically significant repolarization prolongation **or** arrhythmia-like events in 14 of 15 datasets for dofetilide. The workbook has `EAD=1` in 12 of 15 datasets. These are not directly contradictory because the published claim is an OR of two endpoint families, whereas this workbook exposes a single `EAD` flag plus `ddFPDc`.

### 7. Missing ddFPDc

The paper specifies that ddFPDc/APD90c was not calculated and was designated missing when at least 50% of included wells were arrhythmic after dosing. The workbook contains 2,067 missing ddFPDc values, including every row with `EAD=1`. The workbook does not expose enough information in a dedicated denominator field to independently prove the 50% rule for every missing record, so missingness is preserved rather than reconstructed by assumption.

## Decision

**Status: audit complete, validation blocked pending provenance resolution.**

The workbook is useful for building and testing the ingestion/standardization layer, but it is **not yet acceptable as the definitive external validation dataset**.

Before CardioScore performance is interpreted against it, we need to determine:

1. the source/provenance of the workbook;
2. the exact semantic definition of `EAD`, `Type_of_EADs`, and `ddFPDc`;
3. why `ACA` appears as a platform code;
4. why the terfenadine and verapamil directional claims disagree with the publication;
5. whether the workbook is an extracted/derived table from one of the paper's supplemental data files or a separately processed reconstruction.

The official CiPA archive remains the required reference for raw-archive integrity and independent parser validation.
