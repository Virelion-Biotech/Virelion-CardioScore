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
