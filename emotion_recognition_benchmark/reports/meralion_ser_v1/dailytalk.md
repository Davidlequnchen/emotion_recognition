# MERaLiON-SER-v1 on DailyTalk (validation)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,197 |
| Accuracy: acted emotion | 60.07% |
| UAR: mean ground-truth-row recall | 48.65% |
| Macro F1 | 29.82% |
| Weighted F1: true-support weighted | 66.18% |
| Accuracy: all model classes | 60.07% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/dailytalk_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/dailytalk_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-10`.
- Dataset scope recorded in JSON: **DailyTalk (validation)**, English; 1,197 / 1,197 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M3 Pro", "seconds": 550.7, "ms_per_clip": 460.1, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 4 | 25.00% | — |
| disgusted | 3 | 66.67% | — |
| fearful | 1 | 0.00% | — |
| happy | 217 | 53.92% | — |
| neutral | 931 | 61.44% | — |
| sad | 22 | 54.55% | — |
| surprised | 19 | 78.95% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 4 | 0.92% | 25.00% | 1.77% |
| disgusted | 3 | 20.00% | 66.67% | 30.77% |
| fearful | 1 | 0.00% | 0.00% | 0.00% |
| happy | 217 | 46.25% | 53.92% | 49.79% |
| neutral | 931 | 87.46% | 61.44% | 72.18% |
| sad | 22 | 20.34% | 54.55% | 29.63% |
| surprised | 19 | 14.56% | 78.95% | 24.59% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/dailytalk_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (4) | **25.0%** | 25.0% | 0.0% | 0.0% | 25.0% | 0.0% | 25.0% |
| disgusted (3) | 0.0% | **66.7%** | 0.0% | 0.0% | 0.0% | 0.0% | 33.3% |
| fearful (1) | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | 100.0% | 0.0% |
| happy (217) | 2.3% | 0.5% | 0.5% | **53.9%** | 33.6% | 1.8% | 7.4% |
| neutral (931) | 10.7% | 0.6% | 0.9% | 14.4% | **61.4%** | 4.4% | 7.5% |
| sad (22) | 13.6% | 0.0% | 0.0% | 0.0% | 31.8% | **54.5%** | 0.0% |
| surprised (19) | 0.0% | 0.0% | 0.0% | 10.5% | 5.3% | 5.3% | **78.9%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: No corpus-specific evidence**. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](dailytalk_offline_audit.json)
