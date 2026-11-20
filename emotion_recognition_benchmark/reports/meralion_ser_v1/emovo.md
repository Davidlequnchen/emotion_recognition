# MERaLiON-SER-v1 on EMOVO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 588 |
| Accuracy: acted emotion | 52.89% |
| UAR: mean ground-truth-row recall | 52.89% |
| Macro F1 | 46.87% |
| Weighted F1: true-support weighted | 46.87% |
| Accuracy: all model classes | 52.89% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/emovo_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/emovo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **EMOVO**, Italian; 588 / 588 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 167.9, "ms_per_clip": 285.6, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 84 | 92.86% | — |
| disgusted | 84 | 1.19% | — |
| fearful | 84 | 45.24% | — |
| happy | 84 | 55.95% | — |
| neutral | 84 | 92.86% | — |
| sad | 84 | 63.10% | — |
| surprised | 84 | 19.05% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 84 | 61.90% | 92.86% | 74.29% |
| disgusted | 84 | 100.00% | 1.19% | 2.35% |
| fearful | 84 | 73.08% | 45.24% | 55.88% |
| happy | 84 | 44.34% | 55.95% | 49.47% |
| neutral | 84 | 45.09% | 92.86% | 60.70% |
| sad | 84 | 48.18% | 63.10% | 54.64% |
| surprised | 84 | 80.00% | 19.05% | 30.77% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emovo_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (84) | **92.9%** | 0.0% | 0.0% | 1.2% | 6.0% | 0.0% | 0.0% |
| disgusted (84) | 10.7% | **1.2%** | 1.2% | 25.0% | 28.6% | 32.1% | 1.2% |
| fearful (84) | 8.3% | 0.0% | **45.2%** | 4.8% | 15.5% | 26.2% | 0.0% |
| happy (84) | 21.4% | 0.0% | 2.4% | **56.0%** | 13.1% | 3.6% | 3.6% |
| neutral (84) | 4.8% | 0.0% | 0.0% | 2.4% | **92.9%** | 0.0% | 0.0% |
| sad (84) | 0.0% | 0.0% | 0.0% | 0.0% | 36.9% | **63.1%** | 0.0% |
| surprised (84) | 11.9% | 0.0% | 13.1% | 36.9% | 13.1% | 6.0% | **19.0%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: Used via upstream training**. Evidence scope: upstream teacher lineage. SER report documents emotion2vec seed/base teachers for SG-ECMT labels; their EmoBox corpus subset is unresolved. This counts as upstream training use; it does not assert direct reuse of those raw clips by MERaLiON. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emovo_offline_audit.json)
