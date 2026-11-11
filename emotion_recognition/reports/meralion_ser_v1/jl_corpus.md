# MERaLiON-SER-v1 on JL-Corpus

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,200 |
| Accuracy: acted emotion | 75.92% |
| UAR: mean ground-truth-row recall | 75.92% |
| Macro F1 | 77.28% |
| Weighted F1: true-support weighted | 76.05% |
| Accuracy: all model classes | 75.92% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/jl_corpus_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/jl_corpus_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-10-29`.
- Dataset scope recorded in JSON: **JL-Corpus**, English (New Zealand); 1,200 / 1,200 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 308.2, "ms_per_clip": 256.8, "on_cpu": 0}`.
- Composite labels follow the existing accepted-label rule: excited accepts happy or surprised. F1 resolves a correct composite to the predicted class, otherwise to its first accepted class. Weighted F1 uses the resulting resolved-truth support, which can depend on the predictions.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 240 | 80.42% | — |
| happy | 240 | 70.00% | — |
| neutral | 240 | 56.67% | — |
| sad | 240 | 89.17% | — |
| excited (happy or surprised) | 240 | 83.33% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 240 | 84.28% | 80.42% | 82.30% |
| happy | 447 | 90.54% | 74.94% | 82.01% |
| neutral | 240 | 61.54% | 56.67% | 59.00% |
| sad | 240 | 63.13% | 89.17% | 73.92% |
| surprised | 33 | 80.49% | 100.00% | 89.19% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/jl_corpus_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad | surprised |
|---|---|---|---|---|---|
| angry (240) | **80.4%** | 14.2% | 2.5% | 1.2% | 1.7% |
| happy (240) | 0.0% | **70.0%** | 20.8% | 7.5% | 1.7% |
| neutral (240) | 0.0% | 0.4% | **56.7%** | 42.9% | 0.0% |
| sad (240) | 0.0% | 0.0% | 10.8% | **89.2%** | 0.0% |
| excited (happy or surprised) (240) | 15.0% | **69.6%** | 1.2% | 0.4% | **13.8%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: Used via upstream training**. Evidence scope: upstream teacher lineage. SER report documents emotion2vec seed/base teachers for SG-ECMT labels; their EmoBox corpus subset is unresolved. This counts as upstream training use; it does not assert direct reuse of those raw clips by MERaLiON. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](jl_corpus_offline_audit.json)
