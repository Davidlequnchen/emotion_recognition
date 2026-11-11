# Bootstrap: 95% confidence intervals for scores and model differences

Saved predictions only; no models were rerun. Models currently included: CREMA-D 7; RAVDESS 7; MELD 10.
Resample whole clusters with replacement 10,000 times; seed `20261111`. Percentile intervals use the 2.5th and 97.5th percentiles.

Scores on all saved samples remain the point estimates. Bootstrap changes the mixture of speakers or dialogues and measures how scores and model differences vary. Narrower intervals indicate greater stability under that resampling scheme. The 95% level is nominal coverage of the procedure, not a 95% probability that a fixed parameter lies inside this particular interval.

## Resampling scope

| Dataset | Cluster unit | Current clips | Current clusters | Models |
|---|---|---:|---:|---:|
| CREMA-D | Speaker | 7,442 | 91 | 7 |
| RAVDESS | Speaker | 1,440 | 24 | 7 |
| MELD | Dialogue | 2,608 | 280 | 10 |

Each draw samples the original number of clusters and includes every clip of each selected cluster; repeated clusters repeat all their clips. Metrics are pooled over clips, not averaged over speaker scores.

## Individual model scores

Cells are score [95% CI], in percent. N / G means clips / clusters. Each model uses all its saved predictions; * marks a partial run relative to its declared evaluation scope.

### CREMA-D

| Model | N / G | Accuracy | UAR | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| emotion2vec+ | 7,442 / 91 | 86.46 [85.08, 87.79] | 86.54 [85.17, 87.88] | 86.57 [85.18, 87.90] | 86.48 [85.10, 87.81] |
| MERaLiON | 7,442 / 91 | 84.68 [83.41, 85.93] | 84.87 [83.60, 86.11] | 84.74 [83.46, 85.99] | 84.58 [83.30, 85.83] |
| Melati | 7,442 / 91 | 51.37 [49.41, 53.37] | 52.45 [50.53, 54.39] | 46.73 [44.27, 49.17] | 46.61 [44.13, 49.08] |
| Vox-Profile Whisper | 7,442 / 91 | 33.66 [32.15, 35.26] | 35.20 [33.73, 36.76] | 29.93 [28.31, 31.58] | 29.78 [28.15, 31.45] |
| Vox-Profile WavLM | 7,442 / 91 | 30.74 [29.52, 32.07] | 32.09 [30.90, 33.39] | 27.41 [26.08, 28.81] | 27.30 [25.95, 28.72] |
| GPT Realtime | 7,442 / 91 | 16.00 [15.70, 16.34] | 18.02 [17.73, 18.34] | 7.00 [6.44, 7.60] | 6.53 [5.96, 7.14] |
| Gemini 3.8 Flash | 7,442 / 91 | 32.68 [31.07, 34.42] | 33.94 [32.37, 35.64] | 30.24 [28.45, 32.10] | 30.05 [28.25, 31.93] |

No current-protocol results: Qwen2.5-Omni, Qwen2-Audio, Gemma 4.

### RAVDESS

| Model | N / G | Accuracy | UAR | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| emotion2vec+ | 1,440 / 24 | 92.08 [88.19, 95.42] | 92.29 [88.44, 95.59] | 92.20 [88.33, 95.53] | 92.10 [88.23, 95.42] |
| MERaLiON | 1,440 / 24 | 59.31 [55.97, 62.71] | 61.26 [57.64, 64.86] | 59.68 [55.80, 63.37] | 58.47 [54.80, 61.97] |
| Melati | 1,440 / 24 | 48.40 [44.51, 52.22] | 49.33 [45.31, 53.32] | 46.61 [42.10, 50.93] | 46.65 [42.27, 50.83] |
| Vox-Profile Whisper | 1,440 / 24 | 50.76 [48.19, 53.61] | 49.33 [46.70, 52.16] | 44.50 [41.46, 47.71] | 45.15 [42.29, 48.18] |
| Vox-Profile WavLM | 1,440 / 24 | 44.65 [42.08, 47.29] | 42.41 [39.86, 45.01] | 38.44 [35.58, 41.14] | 39.65 [36.94, 42.26] |
| GPT Realtime | 1,440 / 24 | 27.01 [24.72, 29.44] | 21.83 [19.35, 24.40] | 17.76 [13.89, 21.50] | 19.06 [15.35, 22.62] |
| Gemini 3.8 Flash | 1,440 / 24 | 46.94 [43.68, 50.35] | 45.76 [42.34, 49.23] | 46.13 [42.52, 49.54] | 46.51 [43.06, 49.83] |

No current-protocol results: Qwen2.5-Omni, Qwen2-Audio, Gemma 4.

### MELD

| Model | N / G | Accuracy | UAR | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| emotion2vec+ | 2,608 / 280 | 45.59 [43.41, 47.71] | 29.84 [28.11, 31.49] | 28.49 [26.71, 30.13] | 45.58 [43.46, 47.68] |
| MERaLiON | 2,608 / 280 | 30.06 [27.97, 32.20] | 38.38 [35.39, 41.41] | 28.00 [25.70, 30.18] | 24.31 [22.14, 26.58] |
| Melati | 2,608 / 280 | 28.76 [26.73, 30.87] | 23.00 [21.18, 24.92] | 18.68 [17.03, 20.31] | 28.13 [25.90, 30.42] |
| Vox-Profile Whisper | 2,608 / 280 | 33.82 [31.56, 36.05] | 29.60 [27.27, 31.88] | 26.11 [23.81, 28.30] | 34.82 [32.34, 37.24] |
| Vox-Profile WavLM | 2,608 / 280 | 34.85 [32.79, 36.92] | 29.42 [27.11, 31.73] | 26.80 [24.78, 28.67] | 36.79 [34.60, 38.95] |
| Qwen2.5-Omni | 2,608 / 280 | 51.27 [49.03, 53.45] | 43.45 [40.14, 46.81] | 40.43 [37.59, 43.05] | 52.39 [50.14, 54.59] |
| Qwen2-Audio | 2,608 / 280 | 30.71 [28.62, 32.79] | 39.55 [36.26, 42.75] | 29.35 [27.28, 31.31] | 32.23 [30.08, 34.38] |
| Gemma 4 | 2,608 / 280 | 47.81 [45.58, 50.02] | 29.66 [27.57, 31.88] | 30.86 [28.28, 33.39] | 46.09 [43.76, 48.35] |
| GPT Realtime | 2,608 / 280 | 46.63 [44.40, 48.88] | 34.94 [32.33, 37.59] | 36.36 [33.21, 39.28] | 47.21 [44.95, 49.49] |
| Gemini 3.8 Flash | 2,608 / 280 | 40.03 [37.94, 42.09] | 41.59 [38.01, 45.07] | 35.16 [32.45, 37.61] | 41.23 [39.03, 43.41] |

## Paired differences for the selected models

These comparisons use the existing selected-model set: emotion2vec+, Gemini and Qwen2.5-Omni. They were not selected by the scores in this analysis. The CSV includes every available model pair and all four metrics. Values are A−B in percentage points. An interval crossing zero does not resolve the direction of the difference; excluding zero is not a multiplicity-adjusted significance test.

| Dataset | A − B | Common N / G | Δ UAR | Δ Macro-F1 |
|---|---|---:|---:|---:|
| CREMA-D | emotion2vec+ − Gemini 3.8 Flash | 7,442 / 91 | +52.60 [+50.90, +54.21] | +56.33 [+54.49, +58.11] |
| RAVDESS | emotion2vec+ − Gemini 3.8 Flash | 1,440 / 24 | +46.53 [+41.42, +51.34] | +46.08 [+40.95, +51.08] |
| MELD | emotion2vec+ − Gemini 3.8 Flash | 2,608 / 280 | -11.75 [-15.15, -8.32] | -6.67 [-9.19, -4.07] |
| MELD | emotion2vec+ − Qwen2.5-Omni | 2,608 / 280 | -13.61 [-17.09, -10.27] | -11.94 [-14.80, -9.10] |
| MELD | Gemini 3.8 Flash − Qwen2.5-Omni | 2,608 / 280 | -1.86 [-5.48, +1.62] | -5.27 [-7.99, -2.60] |

Current findings:

- CREMA-D: The interval supports higher Macro-F1 for emotion2vec+ than Gemini 3.8 Flash.
- RAVDESS: The interval supports higher Macro-F1 for emotion2vec+ than Gemini 3.8 Flash.
- MELD: The interval supports higher Macro-F1 for Gemini 3.8 Flash than emotion2vec+.
- MELD: The interval supports higher Macro-F1 for Qwen2.5-Omni than emotion2vec+.
- MELD: The interval supports higher Macro-F1 for Qwen2.5-Omni than Gemini 3.8 Flash.
- MELD: The UAR difference between Gemini 3.8 Flash and Qwen2.5-Omni includes zero; neither has a resolved advantage.

## Interpretation and limitations

- Each comparison uses the two models' common audio paths, identical gold/candidate rules and the same resampled clusters. Its interval comes from replicate-wise A−B, not subtraction of marginal interval endpoints. Overlapping individual-model intervals do not determine the paired result.
- The class set is fixed. Invalid answers remain errors, count in N/false negatives, and are not an extra macro-average class. Recall/F1 is zero for a class absent in a draw. Weighted-F1 uses each draw's gold supports. Resamples missing a class: 0, counted over distinct cohorts.
- These approximate intervals concern fixed predictions under changes in the observed cluster mixture; they exclude training/API rerun variability. Possible training exposure prevents claims of unseen-speaker generalization. RAVDESS has only 24 speakers, so interpret its intervals cautiously.
- MELD clusters dialogues. Recurring actors/episodes can create dependence across dialogues. This is neither unseen-actor nor unseen-episode inference. Episode sensitivity and multiplicity corrections were not performed. The intervals do not establish accuracy on company calls.

## Reproduce and update

From `research_paper/`:

```bash
python scripts/bootstrap.py --resamples 10000 --seed 20261111
```

The default discovers all available current-protocol models and selects the largest finished saved run per model/dataset. Rerun after new predictions arrive. Identical inputs, parameters and NumPy version reproduce the output. Streams depend on the seed and cohort, so adding another model does not change existing comparisons. All results are in [bootstrap.csv](../output/bootstrap.csv); parameters, source SHA256s, grouping provenance and diagnostics are in the [manifest](../output/bootstrap_manifest.json). CSV estimates and bounds use the raw score scale; this report multiplies them by 100 to show percentages or percentage points.

Method references: [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for percentile/paired definitions; [Cameron & Miller, 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf) for whole-cluster resampling and few-cluster cautions. That paper concerns regression; this analysis uses its cluster-resampling principle only.
