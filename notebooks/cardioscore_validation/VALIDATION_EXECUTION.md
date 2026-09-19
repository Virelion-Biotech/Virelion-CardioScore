# CardioScore validation notebooks

Pinned CardioScore revision: `869150cd5fb5ccf155fb066258404bd4df163ade`

Run in order:

1. `00_validation_intake_colab.ipynb` — immutable source receipt, SHA-256, archive inventory, and schema inspection.
2. `01_blinova_cipa_summary_colab.ipynb` — official CiPA workbook semantic/component audit; Q is kept separate from A–D.
3. `02_raw_mea_validation_colab.ipynb` — raw MEA CSV through the repository's real trace ingestion and feature extraction.
4. `03_locked_external_validation_colab.ipynb` — headline external risk-class validation; it requires an explicitly verified source-to-feature rebuild.
5. `04_robustness_and_report_colab.ipynb` — secondary endpoint/threshold sensitivity and QC/dropout reporting.

## GIGO gates

The suite fails closed on missing schema, unexpected platform labels, unverified source-to-feature lineage, leakage-prone reference columns, missing reference compounds, and unsupported raw source formats. Technical wells are not treated as independent drugs. Missing endpoints are not fabricated.

The released CiPA workbook is not used as a five-endpoint CardioScore gold standard. A complete external CardioScore claim requires a genuinely verified feature table containing all five endpoints, explicit vehicle controls, and a locked reference table.
