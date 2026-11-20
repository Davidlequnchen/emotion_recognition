# MERaLiON-SER-v1 on MEAD (8,000-clip subset)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 8,000 |
| Accuracy: acted emotion | 28.32% |
| UAR: mean ground-truth-row recall | 32.28% |
| Macro F1 | 25.03% |
| Weighted F1: true-support weighted | 25.29% |
| Accuracy: all model classes | 28.32% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/mead_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/mead_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **MEAD (8,000-clip subset)**, English; 8,000 / 8,000 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 2286.8, "ms_per_clip": 285.8, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 1,223 | 44.56% | — |
| disgusted | 1,248 | 5.29% | — |
| fearful | 1,233 | 5.43% | — |
| happy | 1,247 | 39.53% | — |
| neutral | 561 | 78.79% | — |
| sad | 1,249 | 46.92% | — |
| surprised | 1,239 | 5.41% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 1,223 | 31.30% | 44.56% | 36.77% |
| disgusted | 1,248 | 56.41% | 5.29% | 9.67% |
| fearful | 1,233 | 52.34% | 5.43% | 9.85% |
| happy | 1,247 | 43.82% | 39.53% | 41.57% |
| neutral | 561 | 12.93% | 78.79% | 22.21% |
| sad | 1,249 | 44.09% | 46.92% | 45.46% |
| surprised | 1,239 | 47.52% | 5.41% | 9.71% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/mead_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (1,223) | **44.6%** | 1.4% | 0.3% | 4.7% | 42.7% | 5.1% | 1.3% |
| disgusted (1,248) | 20.3% | **5.3%** | 0.5% | 14.3% | 45.6% | 12.3% | 1.8% |
| fearful (1,233) | 26.5% | 0.6% | **5.4%** | 10.5% | 32.3% | 23.1% | 1.5% |
| happy (1,247) | 8.5% | 0.8% | 0.5% | **39.5%** | 46.5% | 3.4% | 0.8% |
| neutral (561) | 6.6% | 0.4% | 0.0% | 2.1% | **78.8%** | 11.9% | 0.2% |
| sad (1,249) | 13.1% | 0.5% | 0.8% | 9.3% | 29.0% | **46.9%** | 0.4% |
| surprised (1,239) | 24.9% | 0.7% | 2.8% | 11.2% | 44.1% | 10.8% | **5.4%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: Used via upstream training**. Evidence scope: upstream teacher lineage. SER report documents emotion2vec seed/base teachers for SG-ECMT labels; their EmoBox corpus subset is unresolved. This counts as upstream training use; it does not assert direct reuse of those raw clips by MERaLiON. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](mead_offline_audit.json)
