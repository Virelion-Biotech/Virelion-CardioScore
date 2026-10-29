# SiMulTox component-validation record — 2026-10-29

## Final QC-gated run and source

- Final GitHub Actions run: [40875172929](https://github.com/Virelion-Biotech/Virelion-CardioScore/actions/runs/40875172929) — success
- Code revision: `12b3e1da62c8ec84729669214af987bf6b36fee0`
- Dataset: [Zenodo record 13121249](https://zenodo.org/records/13121249), DOI `10.5281/zenodo.13121249`
- Three source files were acquired and all publisher-listed MD5 values matched. SHA-256 fingerprints are in the workflow artifact.

| Recording | Bytes | Beat rate | Detected beats | FPD candidates | Candidate FPD | Detection rate | FPD eligibility |
|---|---:|---:|---:|---:|---:|---:|---|
| Control, single-well | 9,400,016 | 49.32 bpm | 50 | 21 | 486.98 ms | 0.973 | Eligible for descriptive FPD summary |
| E-4031 1 µM, single-well | 7,496,856 | 0.00 bpm | 1 | 1 | 236.90 ms | 0.024 | **Blocked by signal QC** |

Both recordings were sampled at 10 kHz and evaluated in centered 60-second windows using the MCS physical conversion formula encoded in the channel metadata. The single-well amplitude outputs were 13,941.98 µV for control and 28,214.39 µV for E-4031; because these are much larger than most channels in the multi-channel control file, raw scale, channel metadata, and feature semantics should be audited before quantitative amplitude comparison.

## QC findings

The configured CardioScore beat-detection threshold is 0.70. E-4031 fails this screen (0.024), with only one detected beat and one FPD candidate. The final JSON marks:

- `fpd_comparison_qc.eligible = false`
- `fpd_comparison_qc.status = blocked_by_signal_qc`
- failed reasons: beat-detection rate below threshold, fewer than two detected beats, and fewer than two FPD candidates.

The raw FPD difference of −250.08 ms remains in the output for auditability only. **Do not interpret it as a drug effect.**

The six-well control asset yielded 54 channel feature rows. Median beat-detection rate was 0.366; **20/54** passed the configured detection-rate screen and were eligible for the FPD-summary gate; 34/54 had at least two detected beats, and 35/54 yielded at least one FPD candidate. Channel-level CSV rows contain `passes_beat_detection_qc`, `fpd_estimate_eligible`, and `qc_failures` columns so ineligible features are not silently promoted as valid.

## Scope

**Passed:** source download/checksum verification, SHA-256 provenance, MCS HDF5 reading, physical-scaling execution, and generation of CSV/JSON features with explicit signal-QC eligibility flags.

**Blocked:** a reliable paired FPD comparison for the single-well Control/E-4031 challenge because E-4031 fails signal QC.

**Not tested:** CardioScore composite-risk performance, calibrated probability, or clinical/regulatory validity. There is one source recording per condition and no biological replication; no risk class or AUC was calculated from this dataset.

## Artifact

The final artifact is [`cardioscore-simultox-component-validation`](https://github.com/Virelion-Biotech/Virelion-CardioScore/actions/runs/40875172929). It contains the machine-readable source receipt/summary, single-well feature table, and six-well channel feature table—not raw HDF5 recordings. GitHub Actions retains it through 2026-11-28.

The next scientific step is a raw voltage/channel-metadata review followed by independent blinded human beat annotation using the existing frozen Cardio PyMEA workflow. Composite-risk qualification still requires a separate eligible dataset and an owner-attested locked preregistration.
