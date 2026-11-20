# Melati on MELD

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,608 |
| Accuracy: acted emotion | 28.76% |
| UAR: mean ground-truth-row recall | 23.00% |
| Macro F1 | 18.68% |
| Weighted F1: true-support weighted | 28.13% |
| Accuracy: all model classes | 28.76% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/melati_emotion/meld_predictions.csv) and [metadata JSON](../../outputs/melati_emotion/meld_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **MELD**, English; 2,608 / 2,608 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 774.2, "ms_per_clip": 296.9, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 345 | 42.32% | — |
| disgusted | 67 | 1.49% | — |
| fearful | 50 | 10.00% | — |
| happy | 401 | 66.83% | — |
| neutral | 1,256 | 23.33% | — |
| sad | 208 | 14.90% | — |
| surprised | 281 | 2.14% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 345 | 33.95% | 42.32% | 37.68% |
| disgusted | 67 | 2.56% | 1.49% | 1.89% |
| fearful | 50 | 3.38% | 10.00% | 5.05% |
| happy | 401 | 19.35% | 66.83% | 30.01% |
| neutral | 1,256 | 64.97% | 23.33% | 34.33% |
| sad | 208 | 21.99% | 14.90% | 17.77% |
| surprised | 281 | 42.86% | 2.14% | 4.07% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/meld_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (345) | **42.3%** | 0.6% | 2.6% | 43.2% | 9.0% | 2.3% | 0.0% |
| disgusted (67) | 26.9% | **1.5%** | 0.0% | 59.7% | 9.0% | 3.0% | 0.0% |
| fearful (50) | 16.0% | 4.0% | **10.0%** | 56.0% | 8.0% | 6.0% | 0.0% |
| happy (401) | 12.5% | 1.0% | 6.0% | **66.8%** | 8.0% | 5.7% | 0.0% |
| neutral (1,256) | 12.6% | 1.5% | 5.7% | 51.4% | **23.3%** | 4.9% | 0.6% |
| sad (208) | 7.2% | 3.8% | 10.1% | 43.8% | 19.7% | **14.9%** | 0.5% |
| surprised (281) | 12.5% | 1.1% | 6.4% | 58.0% | 15.7% | 4.3% | **2.1%** |

## Caveats

- **Training involvement for Melati: No corpus-specific evidence**. Source: Public SER head trained on nine named corpora including JL-Corpus and Emozionalmente; Whisper-v3 pretraining incompletely disclosed; author nEMO threshold calibration is counted as seen development use, although its thresholds are not applied here. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](meld_offline_audit.json)
