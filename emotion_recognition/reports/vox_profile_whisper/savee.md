# Vox-Profile Whisper on SAVEE

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 480 |
| Accuracy: acted emotion | 37.29% |
| UAR: mean ground-truth-row recall | 31.55% |
| Macro F1 | 29.36% |
| Weighted F1: true-support weighted | 32.39% |
| Accuracy: all model classes | 36.25% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_whisper/savee_predictions.csv) and [metadata JSON](../../outputs/vox_profile_whisper/savee_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **SAVEE**, English (British); 480 / 480 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 112.4, "ms_per_clip": 234.1, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 60 | 48.33% | — |
| disgusted | 60 | 10.00% | — |
| fearful | 60 | 5.00% | — |
| happy | 60 | 13.33% | — |
| neutral | 120 | 77.50% | — |
| sad | 60 | 30.00% | — |
| surprised | 60 | 36.67% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 60 | 40.28% | 48.33% | 43.94% |
| disgusted | 60 | 24.00% | 10.00% | 14.12% |
| fearful | 60 | 75.00% | 5.00% | 9.38% |
| happy | 60 | 21.62% | 13.33% | 16.49% |
| neutral | 120 | 40.97% | 77.50% | 53.60% |
| sad | 60 | 32.73% | 30.00% | 31.30% |
| surprised | 60 | 36.67% | 36.67% | 36.67% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/savee_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (60) | **48.3%** | 6.7% | 0.0% | 6.7% | 26.7% | 3.3% | 8.3% |
| disgusted (60) | 10.0% | **10.0%** | 0.0% | 6.7% | 58.3% | 1.7% | 13.3% |
| fearful (60) | 15.0% | 5.0% | **5.0%** | 8.3% | 30.0% | 18.3% | 18.3% |
| happy (60) | 28.3% | 8.3% | 0.0% | **13.3%** | 28.3% | 0.0% | 21.7% |
| neutral (120) | 0.8% | 0.8% | 0.0% | 2.5% | **77.5%** | 18.3% | 0.0% |
| sad (60) | 0.0% | 5.0% | 1.7% | 6.7% | 55.0% | **30.0%** | 1.7% |
| surprised (60) | 16.7% | 5.0% | 0.0% | 15.0% | 25.0% | 1.7% | **36.7%** |

## Caveats

- **Training involvement for Vox-Profile Whisper: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; retained Whisper-large-v3 pretraining sources incompletely disclosed. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](savee_offline_audit.json)
