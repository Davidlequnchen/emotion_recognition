# Vox-Profile Whisper on DailyTalk (validation)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,197 |
| Accuracy: acted emotion | 58.65% |
| UAR: mean ground-truth-row recall | 58.72% |
| Macro F1 | 27.77% |
| Weighted F1: true-support weighted | 65.86% |
| Accuracy: all model classes | 55.72% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_whisper/dailytalk_predictions.csv) and [metadata JSON](../../outputs/vox_profile_whisper/dailytalk_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-11`.
- Dataset scope recorded in JSON: **DailyTalk (validation)**, English; 1,197 / 1,197 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M3 Pro", "seconds": 613.2, "ms_per_clip": 512.3, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 4 | 75.00% | — |
| disgusted | 3 | 0.00% | — |
| fearful | 1 | 100.00% | — |
| happy | 217 | 66.82% | — |
| neutral | 931 | 57.04% | — |
| sad | 22 | 22.73% | — |
| surprised | 19 | 89.47% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 4 | 1.82% | 75.00% | 3.55% |
| disgusted | 3 | 0.00% | 0.00% | 0.00% |
| fearful | 1 | 9.09% | 100.00% | 16.67% |
| happy | 217 | 49.32% | 66.82% | 56.75% |
| neutral | 931 | 91.87% | 57.04% | 70.38% |
| sad | 22 | 22.73% | 22.73% | 22.73% |
| surprised | 19 | 14.05% | 89.47% | 24.29% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/dailytalk_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (4) | **75.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 25.0% |
| disgusted (3) | 33.3% | **0.0%** | 0.0% | 0.0% | 33.3% | 0.0% | 33.3% |
| fearful (1) | 0.0% | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% |
| happy (217) | 6.5% | 0.0% | 0.5% | **66.8%** | 18.0% | 0.0% | 8.3% |
| neutral (931) | 15.3% | 0.6% | 0.6% | 15.8% | **57.0%** | 1.8% | 8.8% |
| sad (22) | 18.2% | 0.0% | 13.6% | 4.5% | 31.8% | **22.7%** | 9.1% |
| surprised (19) | 5.3% | 0.0% | 0.0% | 5.3% | 0.0% | 0.0% | **89.5%** |

## Caveats

- **Training involvement for Vox-Profile Whisper: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; retained Whisper-large-v3 pretraining sources incompletely disclosed. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](dailytalk_offline_audit.json)
