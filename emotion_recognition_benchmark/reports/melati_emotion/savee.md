# Melati on SAVEE

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 480 |
| Accuracy: acted emotion | 61.25% |
| UAR: mean ground-truth-row recall | 56.43% |
| Macro F1 | 53.21% |
| Weighted F1: true-support weighted | 58.05% |
| Accuracy: all model classes | 61.25% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/melati_emotion/savee_predictions.csv) and [metadata JSON](../../outputs/melati_emotion/savee_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **SAVEE**, English (British); 480 / 480 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 195.2, "ms_per_clip": 406.7, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 60 | 78.33% | — |
| disgusted | 60 | 30.00% | — |
| fearful | 60 | 16.67% | — |
| happy | 60 | 91.67% | — |
| neutral | 120 | 95.00% | — |
| sad | 60 | 65.00% | — |
| surprised | 60 | 18.33% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 60 | 68.12% | 78.33% | 72.87% |
| disgusted | 60 | 90.00% | 30.00% | 45.00% |
| fearful | 60 | 37.04% | 16.67% | 22.99% |
| happy | 60 | 39.29% | 91.67% | 55.00% |
| neutral | 120 | 89.06% | 95.00% | 91.94% |
| sad | 60 | 49.37% | 65.00% | 56.12% |
| surprised | 60 | 64.71% | 18.33% | 28.57% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/savee_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (60) | **78.3%** | 3.3% | 1.7% | 8.3% | 3.3% | 5.0% | 0.0% |
| disgusted (60) | 15.0% | **30.0%** | 5.0% | 20.0% | 8.3% | 18.3% | 3.3% |
| fearful (60) | 3.3% | 0.0% | **16.7%** | 53.3% | 1.7% | 20.0% | 5.0% |
| happy (60) | 1.7% | 0.0% | 0.0% | **91.7%** | 0.0% | 6.7% | 0.0% |
| neutral (120) | 0.0% | 0.0% | 0.0% | 0.0% | **95.0%** | 5.0% | 0.0% |
| sad (60) | 0.0% | 0.0% | 15.0% | 8.3% | 10.0% | **65.0%** | 1.7% |
| surprised (60) | 16.7% | 0.0% | 6.7% | 51.7% | 0.0% | 6.7% | **18.3%** |

## Caveats

- **Training involvement for Melati: No corpus-specific evidence**. Source: Public SER head trained on nine named corpora including JL-Corpus and Emozionalmente; Whisper-v3 pretraining incompletely disclosed; author nEMO threshold calibration is counted as seen development use, although its thresholds are not applied here. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](savee_offline_audit.json)
