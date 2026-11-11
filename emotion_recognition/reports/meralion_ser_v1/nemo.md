# MERaLiON-SER-v1 on nEMO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 4,481 |
| Accuracy: acted emotion | 55.14% |
| UAR: mean ground-truth-row recall | 53.77% |
| Macro F1 | 48.06% |
| Weighted F1: true-support weighted | 48.79% |
| Accuracy: all model classes | 55.01% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/nemo_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/nemo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **nEMO**, Polish; 4,481 / 4,481 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 1172.6, "ms_per_clip": 261.7, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 749 | 58.08% | — |
| fearful | 736 | 0.82% | — |
| happy | 749 | 79.84% | — |
| neutral | 809 | 89.99% | — |
| sad | 769 | 75.81% | — |
| surprised | 669 | 18.09% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 749 | 87.17% | 58.08% | 69.71% |
| fearful | 736 | 25.00% | 0.82% | 1.58% |
| happy | 749 | 57.83% | 79.84% | 67.08% |
| neutral | 809 | 52.60% | 89.99% | 66.39% |
| sad | 769 | 41.88% | 75.81% | 53.96% |
| surprised | 669 | 81.76% | 18.09% | 29.62% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/nemo_confusion.png)

| Ground truth \ Predicted | angry | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|
| angry (749) | **58.1%** | 1.5% | 7.6% | 22.6% | 9.7% | 0.5% |
| fearful (736) | 1.8% | **0.8%** | 6.4% | 8.8% | 81.8% | 0.4% |
| happy (749) | 1.1% | 0.0% | **79.8%** | 14.6% | 1.9% | 2.7% |
| neutral (809) | 0.1% | 0.0% | 1.0% | **90.0%** | 8.9% | 0.0% |
| sad (769) | 0.1% | 0.0% | 2.0% | 22.1% | **75.8%** | 0.0% |
| surprised (669) | 6.1% | 1.0% | 46.2% | 21.4% | 7.2% | **18.1%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: No corpus-specific evidence**. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](nemo_offline_audit.json)
