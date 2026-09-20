# External validation engineering runbook

## Engineering state
The repository contains a validation-only execution path. It does not train, tune, or modify CardioScore from the external validation set.

## Immutable source handling
Keep the downloaded source archive outside Git under `data/external/`. Build an `AssetManifest` with source URL, acquisition date, exact filename, byte size, SHA-256, and archive member inventory. ZIP/TAR members are checked for path traversal and hashed without extracting or modifying the original archive.

## Standardized data contract
The validation layer requires explicit identifiers for compound, site, cell type, concentration, and well, plus the electrophysiology fields consumed by CardioScore. Reference risk labels are maintained in a separate table. Raw source anomalies are never silently repaired; explicit aliases belong in provenance/standardization code.

## Locked run
Use `scripts/validate_external_asset.py --manifest <manifest.yaml>`. The command verifies the source checksum, validates standardized/reference schemas, invokes the existing `CardioScorePipeline`, and writes deterministic JSON/CSV validation artifacts.

## Metrics
The locked evaluator reports 3-class confusion matrix, accuracy, balanced accuracy, macro precision/recall/F1, Cohen kappa, ordinal MAE, Spearman rho, Kendall tau, and deterministic stratified failure summaries.

## CI contract
CI always validates the checked-in manifest template. Numerical external validation is intentionally not run until an authoritative dataset and a checksum-pinned real manifest are supplied. Raw external datasets are ignored by Git.

## Scientific validation gate
This engineering layer does not establish that an external dataset is scientifically authoritative. Issue #1 remains open until the official CiPA source archive is independently obtained, verified, inventoried, parsed, reconciled with the publication, and then evaluated without tuning on the validation set.

## Informative dropout (signal loss is not "no effect")
QC used to remove wells that lost signal (cells stopped beating, no usable electrodes, endpoints not computable) without any
concentration-level accounting, so a genuinely proarrhythmic compound could drop from High to Low, or vanish, when its top
concentrations collapsed. The pipeline now records every rejection with reason codes (`qc_rejections`), reports per
compound x concentration signal loss (`dropout_table`), flags `informative_dropout` when at least
`quality_control.informative_dropout_min_fraction` of treated wells lost signal and that exceeds the compound's vehicle loss by
`informative_dropout_min_excess`, and lists every compound it could not score with a reason (`exclusion_table`). Missing
endpoints are never aggregated to zero. In the locked feature table a well may carry missing endpoints only if it has no
usable signal (`n_electrodes == 0` or `beat_detection_rate == 0`).

## Freeze before scoring
1. Commit the CardioScore code to be validated; note its commit SHA `A`.
2. Set `PIN = A` in `scripts/validation/run_colab_validation.py`.
3. Resolve every `null` in `validation/preregistration.yaml` (owner attestations, dataset DOI/URL, raw-data availability),
   review the proposed numeric criteria, and set `status: frozen`.
4. Run `python scripts/validation/make_freeze_manifest.py --package-sha A`. It refuses while anything is unresolved or
   disagrees with the shipped configuration.
5. Commit the pre-registration, the manifest and the `PIN` change as commit `B`, and run the Colab launcher with `RUNNER_SHA = B`.

The locked stage reads both files from commit `B`, verifies their hashes against the installed package configuration, and
produces no headline result if anything differs. The primary result is the pre-registered AUROC (positive class
intermediate + high) with a compound-level bootstrap interval and a three-outcome rule; the report also contains the
three-class metrics, the informative-dropout list, and a sensitivity analysis that reclassifies those compounds as High.
The git history, not this code, is the evidence of when the plan was frozen.
