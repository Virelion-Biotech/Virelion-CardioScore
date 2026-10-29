# CardioScore execution gates (2026-10)

This document distinguishes executable software checks from empirical/scientific qualification. Do not report a completed software run as validation of drug-risk predictions.

## 1. Browser workbench

- \`web/app.js\` now waits for the scoring contract to load before automatically running the demo.
- \`tests/browser_smoke.cjs\` exercises the real JavaScript startup, score rendering, QC logging, and failed-contract path with a small dependency-free DOM shim.
- The CI workflow runs that smoke test in addition to syntax validation.
- The Pages workflow asks \`actions/configure-pages\` to enable Pages on first deployment. The deployment job is the confirmation gate.

Open the deployed site only after the Pages workflow succeeds. The browser scorer intentionally remains a demonstration and a lightweight preview, not a substitute for Python's full normalization/inference/validation pipeline.

## 2. Reproduce the public SiMulTox component challenge

Dataset: [Zenodo record 13121249](https://zenodo.org/records/13121249), DOI 10.5281/zenodo.13121249.

The standalone runner downloads and verifies exactly three source files using the publisher-listed MD5 checksums; raw files are not committed or included in the validation-result artifact.

\`\`\`bash
python -m pip install -e '.[dev,hdf5]'
python scripts/validation/run_simultox_component_validation.py
\`\`\`

Defaults:
- Inputs: \`data/external/simultox_2024/\`
- Outputs: \`outputs/simultox_component_validation/\`

The runner records source MD5 and SHA-256 checksums, manufacturer-defined MCS HDF5 voltage scaling, sampling rate, feature outputs, and a comparison of one control recording with one E-4031 recording. It also summarizes channel-level extraction from the six-well control file. An Actions workflow can run the challenge and retain the compact JSON/CSV results as an artifact.

**Evidence limit:** this is raw-file ingestion and signal-feature component validation. There is one source recording per single-well condition, so the comparison is descriptive, not an independent replicated experiment. No composite drug-risk class, AUC, or claim of predictive validity is produced from it.

## 3. Independent beat-detection check

Use the existing notebook \`notebooks/cardioscore_validation/CardioScore_PyMEA_AutoDownload_Heldout_Validation_Colab.ipynb\`. It acquires Cardio PyMEA source files and prepares blinded annotation windows. Annotation is an independent, human task; software cannot honestly synthesize the reference labels it is supposed to be evaluated against.

Record the source checksums, the frozen protocol/code revisions, annotator identities or pseudonyms, completed annotation sheets, exclusions, and all beat-detection metrics. Do not call this gate complete until annotations exist and the scoring stage is run.

## 4. Locked drug-risk qualification

The current \`validation/preregistration.yaml\` remains a draft. Required declarations and primary data provenance fields are intentionally unresolved. Do not fill those fields with inferred values to make the freeze command pass.

Before the locked run:
1. Select a dataset with authoritative provenance, an appropriate independent-unit structure, and enough documented endpoints for the declared analysis.
2. Verify the source archive and metadata; retain original data unchanged and store hashes/receipts.
3. Resolve all required declarations and numeric criteria, have the study owner attest them, and freeze code/configuration before comparing any score with reference labels.
4. Use the existing locked runner and report the pre-registered metric, compound-level uncertainty, all unscored compounds, informative signal loss, and sensitivity analysis.

The Blinova 2018 processed workbook audit remains useful for semantic/reference readiness, but the workbook has missing endpoint values and no raw waveforms. It cannot be upgraded into full raw-waveform or composite-score validation by relabeling its scope.

## 5. HeartTwin output contract

The adapter now emits \`virelion.cardioscore.hearttwin-result.v1\`, with formal JSON Schemas in \`schemas/hearttwin/\` for the request and result envelopes. It keeps the existing top-level output keys for compatibility and adds:
- software version and score semantics;
- input filename, row count, SHA-256 fingerprint;
- hashes of the active pipeline and endpoint configuration;
- QC rejections, dropout and exclusion tables, and run provenance.

The tests exercise this boundary against the synthetic example file, enforce strict JSON serialization, and check that the local absolute input path is not echoed back in the result. This contract is a research-score interface, not a patient-risk or clinical-diagnosis API.

## What counts as completion

| Gate | Completion evidence |
|---|---|
| Browser software | Behavior smoke test passes and deployment workflow succeeds |
| HDF5 component | Source hashes match; feature outputs and source/config receipts are retained |
| Beat detector | Held-out human annotations scored under the frozen protocol |
| Composite score | Locked external evaluation produces the pre-registered report without leakage |
| HeartTwin boundary | Contract tests pass and an end-to-end caller integration test is exercised |

Do not merge conclusions across rows: passing a component gate does not imply the later gates passed.
