# Vox-Profile Whisper on MELD

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,608 |
| Accuracy: acted emotion | 33.82% |
| UAR: mean ground-truth-row recall | 29.60% |
| Macro F1 | 26.11% |
| Weighted F1: true-support weighted | 34.82% |
| Accuracy: all model classes | 32.55% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_whisper/meld_predictions.csv) and [metadata JSON](../../outputs/vox_profile_whisper/meld_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **MELD**, English; 2,608 / 2,608 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 570.4, "ms_per_clip": 218.7, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 345 | 45.51% | — |
| disgusted | 67 | 8.96% | — |
| fearful | 50 | 12.00% | — |
| happy | 401 | 52.62% | — |
| neutral | 1,256 | 26.99% | — |
| sad | 208 | 12.02% | — |
| surprised | 281 | 49.11% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 345 | 38.48% | 45.51% | 41.70% |
| disgusted | 67 | 11.76% | 8.96% | 10.17% |
| fearful | 50 | 4.72% | 12.00% | 6.78% |
| happy | 401 | 20.85% | 52.62% | 29.87% |
| neutral | 1,256 | 72.28% | 26.99% | 39.30% |
| sad | 208 | 43.86% | 12.02% | 18.87% |
| surprised | 281 | 28.51% | 49.11% | 36.08% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/meld_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (345) | **45.5%** | 3.8% | 3.8% | 27.5% | 4.6% | 0.0% | 14.8% |
| disgusted (67) | 19.4% | **9.0%** | 6.0% | 44.8% | 6.0% | 0.0% | 14.9% |
| fearful (50) | 14.0% | 2.0% | **12.0%** | 42.0% | 10.0% | 4.0% | 16.0% |
| happy (401) | 8.5% | 1.0% | 4.7% | **52.6%** | 10.0% | 0.7% | 22.4% |
| neutral (1,256) | 10.1% | 1.2% | 4.3% | 43.6% | **27.0%** | 2.0% | 11.9% |
| sad (208) | 18.3% | 3.4% | 8.7% | 21.6% | 17.8% | **12.0%** | 18.3% |
| surprised (281) | 11.4% | 1.8% | 4.6% | 22.4% | 10.0% | 0.7% | **49.1%** |

## Caveats

- **Training involvement for Vox-Profile Whisper: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; retained Whisper-large-v3 pretraining sources incompletely disclosed. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](meld_offline_audit.json)
