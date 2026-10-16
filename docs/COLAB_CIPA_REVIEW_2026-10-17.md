# CardioScore CiPA Colab review — 2026-10-17

The uploaded notebook **worked**, completing its declared semantic/reference-readiness audit in 2.20 seconds. Its status is **READINESS_ONLY**, not predictive score qualification. All 17 artifact hashes and 161 pinned source files verify against `40b1ef02cf4cd38b652d0448dce8c2290361aabc`. No uploaded launcher was executed. The original ZIP is retained byte for byte in [colab_cipa_2026_10_17](../validation/colab_cipa_2026_10_17/), separate from corrected replay outputs.

## Reproduction and data fitness

The processed workbook downloaded from the [official CiPA URL](https://cipaproject.org/wp-content/uploads/2018/09/Blinova_etal_2018_data.xlsx) has SHA256 `5fd91c2fa1c9b7fbb2f12cf836a061853e0c6a7af22b62965973198d9066459b`, matching the uploaded receipt and prior intake. This verifies byte identity at that URL; it is not an independently published raw-archive checksum. The workbook itself is not committed. All three original derived artifacts reproduce byte for byte locally; the audit JSON matches after allowing for output paths.

The workbook covers 8,859 rows, 28 normalized compounds and ten sites. Its compound reference contains eight high-, eleven intermediate- and nine low-risk compounds. One missing source risk cell remains missing; the sotalol compound reference comes from other labeled sotalol rows. The source alias is preserved. The undocumented ACA platform appears in 560 rows and is not mapped. There are 2,067 missing ddFPDc values; none is filled or treated as zero. Concentration codes 1–4 are not substituted for physical exposure or Cmax metadata.

This source is fit for the restricted audit. It has no raw MEA waveforms and cannot independently supply CardioScore's complete six-endpoint inputs, a frozen score-to-probability calibration or raw-waveform endpoint truth. The publication's own models/AUC values are not CardioScore validation results.

## Verified correction

The runner flagged 52 event labels as unresolved (`T`: 49, `1AP`: 2, `BCT`: 1), but its substring logic nevertheless counted BCT as an A–D event. An unresolved token cannot establish a typed event or a directional disagreement with the paper. The correction leaves **both typed indicators null** on all 52 rows; they are not confirmed negatives. Counts and claim comparisons use fully resolved labels only. Original source labels, risk reference, missing values and platform codes remain unchanged.

| Audit quantity | Original uploaded result | Corrected replay |
|---|---|---|
| Resolved A–D event rows | 1,287 (included unresolved BCT) | **1,286** |
| Rows with resolved Q | 550 | 550 |
| Unresolved event-label rows | 52 | 52; typed indicators null |
| EAD = 1 | 1,887 | 1,887 |

The resolved A–D and Q counts overlap in one AQ row. They are observation counts, not independent-compound sample sizes or validated event-incidence estimates. Unknown rows must remain distinguishable from true event-free rows in downstream analysis. The corrected semantic CSV is compressed losslessly and accompanied by the unresolved-row table and fresh audit outputs.

The paper explicitly distinguishes A–D arrhythmia-like events from quiescence (Q) and allows combinations. In the verified workbook, all EAD=1 rows for terfenadine (46) and verapamil (34) are Q-only. The old documentation's apparent arrhythmia disagreement was therefore a field-semantic error and is corrected. This does not independently validate every other published directional claim or establish CardioScore performance.

Primary reference: [Blinova et al., Cell Reports 2018](https://doi.org/10.1016/j.celrep.2018.08.079), particularly Experimental Procedures and Figures 1–3. The primary full text was checked through [Europe PMC](https://europepmc.org/articles/PMC6226030); the XML and raw workbook are not redistributed here.

## Remaining work

Unknown T/1AP/BCT definitions, ACA provenance, complete endpoint/exposure/replicate mapping, independent waveform truth, score/probability calibration, and an untouched compound/site evaluation remain blocked. No source label or default score threshold is guessed or tuned to this workbook. Completion of this audit is not empirical or clinical validation. There was no notebook execution failure, so a replacement Colab notebook is unnecessary.

## Software verification

Three regression cases fail before the semantic correction and pass after it. All 32 targeted audit tests pass; the complete local suite reports **404 passed, one optional skip**. Ruff, the validation contract, browser JavaScript syntax and the synthetic demo pass. Hosted CI is required before merging. These software checks are not new empirical score-performance evidence.
