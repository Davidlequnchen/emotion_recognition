# MERaLiON-SER-v1 on TESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,800 |
| Accuracy: acted emotion | 49.25% |
| UAR: mean ground-truth-row recall | 49.25% |
| Macro F1 | 37.86% |
| Weighted F1: true-support weighted | 37.86% |
| Accuracy: all model classes | 49.25% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/tess_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/tess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-10-29`.
- Dataset scope recorded in JSON: **TESS**, English; 2,800 / 2,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, surprised, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 720.6, "ms_per_clip": 257.3, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 400 | 96.25% | — |
| disgusted | 400 | 1.50% | — |
| fearful | 400 | 0.00% | — |
| happy | 400 | 57.50% | — |
| neutral | 400 | 0.00% | — |
| surprised | 400 | 89.50% | — |
| sad | 400 | 100.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 400 | 86.32% | 96.25% | 91.02% |
| disgusted | 400 | 100.00% | 1.50% | 2.96% |
| fearful | 400 | 0.00% | 0.00% | 0.00% |
| happy | 400 | 35.88% | 57.50% | 44.19% |
| neutral | 400 | 0.00% | 0.00% | 0.00% |
| surprised | 400 | 65.45% | 89.50% | 75.61% |
| sad | 400 | 34.48% | 100.00% | 51.28% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/tess_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | surprised | sad |
|---|---|---|---|---|---|---|---|
| angry (400) | **96.2%** | 0.0% | 0.0% | 3.8% | 0.0% | 0.0% | 0.0% |
| disgusted (400) | 12.8% | **1.5%** | 0.0% | 59.5% | 0.0% | 3.2% | 23.0% |
| fearful (400) | 2.5% | 0.0% | **0.0%** | 29.0% | 0.0% | 2.2% | 66.2% |
| happy (400) | 0.0% | 0.0% | 0.0% | **57.5%** | 0.0% | 41.8% | 0.8% |
| neutral (400) | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 100.0% |
| surprised (400) | 0.0% | 0.0% | 0.0% | 10.5% | 0.0% | **89.5%** | 0.0% |
| sad (400) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **100.0%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: Used via upstream training**. Evidence scope: upstream teacher lineage. SER report documents emotion2vec seed/base teachers for SG-ECMT labels; their EmoBox corpus subset is unresolved. This counts as upstream training use; it does not assert direct reuse of those raw clips by MERaLiON. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](tess_offline_audit.json)
