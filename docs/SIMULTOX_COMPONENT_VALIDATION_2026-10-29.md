# SiMulTox component-validation record — 2026-10-29

## Run and source

- GitHub Actions run: [40873522938](https://github.com/Virelion-Biotech/Virelion-CardioScore/actions/runs/40873522938) — success
- Runner revision: `afb03a8958c71cd38b6f270f0fcd23488f7b28d7`
- Dataset: [Zenodo record 13121249](https://zenodo.org/records/13121249), DOI `10.5281/zenodo.13121249`
- Three agreed source files were acquired and the publisher-listed MD5 values all matched. SHA-256 fingerprints are in the [run artifact](https://github.com/Virelion-Biotech/Virelion-CardioScore/actions/runs/40873522938).

| Recording | Bytes | Beat rate | Detected beats | Beats with an FPD candidate | FPD candidate | Beat-detection rate |
|---|---:|---:|---:|---:|---:|---:|
| Control, single-well | 9,400,016 | 49.32 bpm | 50 | 21 | 486.98 ms | 0.973 |
| E-4031 1 µM, single-well | 7,496,856 | 0.00 bpm | 1 | 1 | 236.90 ms | 0.024 |

Both traces were sampled at 10 kHz; the analysis used a centered 60-second window and the MCS physical conversion formula documented by the channel metadata. The control's extracted amplitude was 13,941.98 µV and E-4031's was 28,214.39 µV, far higher than most values in the multi-channel control asset; investigate raw-scale/channel semantics before treating amplitude as quantitatively comparable.

## QC interpretation

The configured CardioScore beat-detection screen is 0.70. The E-4031 trace's detection rate (0.024) is below that threshold and contains only one beat; its FPD is therefore a single candidate, not a reliable condition estimate. Do **not** interpret the raw difference of -250.08 ms in FPD as a drug effect.

The six-well control asset yielded 54 channel feature rows. Median beat-detection rate was 0.366; 20/54 channels met the 0.70 screen, 34/54 had at least two detected beats, and 35/54 produced an FPD candidate. These results show that the data reader and feature extractor execute, but also expose signal-quality/endpoint-eligibility concerns.

## Scope

**Passed:** source download, publisher MD5 verification, SHA-256 provenance, MCS HDF5 read, physical conversion execution, and generation of CSV/JSON features.

**Not passed:** reliable paired FPD comparison for the single-well Control/E-4031 challenge. The comparison has only one source recording per condition and is not replicated biological evidence.

**Not tested:** CardioScore composite-risk performance, calibrated probability, or clinical/regulatory validity. No risk class or AUC was calculated from this dataset.

The initial artifact was generated before endpoint eligibility fields were added to the runner. The follow-up runner change adds per-condition and per-channel QC flags so low-quality candidate features are explicitly marked rather than silently presented as eligible comparisons.
