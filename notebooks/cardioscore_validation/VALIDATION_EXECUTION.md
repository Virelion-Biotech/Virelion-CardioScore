# CardioScore validation notebooks

Pinned CardioScore revision: `869150cd5fb5ccf155fb066258404bd4df163ade`

Run in order:

1. `00_validation_intake_colab.ipynb` — immutable source receipt, SHA-256, archive inventory, and schema inspection.
2. `01_blinova_cipa_summary_colab.ipynb` — official CiPA workbook semantic/component audit; Q is kept separate from A–D.
3. `02_raw_mea_validation_colab.ipynb` — raw MEA CSV through the repository's real trace ingestion and feature extraction.
4. `03_locked_external_validation_colab.ipynb` — headline external risk-class validation; it requires an explicitly verified source-to-feature rebuild.
5. `04_robustness_and_report_colab.ipynb` — secondary endpoint/threshold sensitivity and QC/dropout reporting.
6. `05_publish_validation_results_colab.ipynb` — safety-gated Colab → GitHub `main` publication of derived validation artifacts.

## GIGO gates

The suite fails closed on missing schema, unexpected platform labels, unverified source-to-feature lineage, leakage-prone reference columns, missing reference compounds, and unsupported raw source formats. Technical wells are not treated as independent drugs. Missing endpoints are not fabricated.

The released CiPA workbook is not used as a five-endpoint CardioScore gold standard. A complete external CardioScore claim requires a genuinely verified feature table containing all five endpoints, explicit vehicle controls, and a locked reference table.

## Publishing results

Notebook 05 requires a Colab Secret named `GITHUB_TOKEN`. It clones `main`, copies only an allowlisted set of derived JSON/CSV/Markdown artifacts into `validation_results/<run_label>/`, creates a hash manifest, checks that no other repository paths changed, commits, pushes to `main`, and verifies the remote commit SHA.

Never paste a token directly into notebook code. Never publish the raw external dataset.
