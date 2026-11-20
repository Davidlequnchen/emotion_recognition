# Vox-Profile WavLM on JL-Corpus

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,200 |
| Accuracy: acted emotion | 48.33% |
| UAR: mean ground-truth-row recall | 48.33% |
| Macro F1 | 50.50% |
| Weighted F1: true-support weighted | 47.94% |
| Accuracy: all model classes | 46.00% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_wavlm/jl_corpus_predictions.csv) and [metadata JSON](../../outputs/vox_profile_wavlm/jl_corpus_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-10-29`.
- Dataset scope recorded in JSON: **JL-Corpus**, English (New Zealand); 1,200 / 1,200 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 53.8, "ms_per_clip": 44.9, "on_cpu": 0}`.
- Composite labels follow the existing accepted-label rule: excited accepts happy or surprised. F1 resolves a correct composite to the predicted class, otherwise to its first accepted class. Weighted F1 uses the resulting resolved-truth support, which can depend on the predictions.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 240 | 51.25% | — |
| happy | 240 | 27.08% | — |
| neutral | 240 | 67.92% | — |
| sad | 240 | 35.83% | — |
| excited (happy or surprised) | 240 | 59.58% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 240 | 55.41% | 51.25% | 53.25% |
| happy | 420 | 75.90% | 35.24% | 48.13% |
| neutral | 240 | 34.98% | 67.92% | 46.18% |
| sad | 240 | 44.56% | 35.83% | 39.72% |
| surprised | 60 | 48.39% | 100.00% | 65.22% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/jl_corpus_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad | surprised |
|---|---|---|---|---|---|
| angry (240) | **51.2%** | 14.6% | 22.1% | 2.9% | 9.2% |
| happy (240) | 9.2% | **27.1%** | 37.5% | 10.0% | 16.2% |
| neutral (240) | 2.9% | 1.7% | **67.9%** | 27.5% | 0.0% |
| sad (240) | 4.2% | 3.3% | 55.4% | **35.8%** | 1.2% |
| excited (happy or surprised) (240) | 25.0% | **34.6%** | 11.2% | 4.2% | **25.0%** |

## Caveats

- **Training involvement for Vox-Profile WavLM: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; WavLM-large backbone trained on Libri-Light, GigaSpeech and VoxPopuli; no clip-level overlap audit. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](jl_corpus_offline_audit.json)
