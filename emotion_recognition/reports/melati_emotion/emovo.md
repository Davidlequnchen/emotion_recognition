# Melati on EMOVO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 588 |
| Accuracy: acted emotion | 55.95% |
| UAR: mean ground-truth-row recall | 55.95% |
| Macro F1 | 53.76% |
| Weighted F1: true-support weighted | 53.76% |
| Accuracy: all model classes | 55.95% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/melati_emotion/emovo_predictions.csv) and [metadata JSON](../../outputs/melati_emotion/emovo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **EMOVO**, Italian; 588 / 588 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 174.5, "ms_per_clip": 296.8, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 84 | 96.43% | — |
| disgusted | 84 | 35.71% | — |
| fearful | 84 | 64.29% | — |
| happy | 84 | 63.10% | — |
| neutral | 84 | 73.81% | — |
| sad | 84 | 19.05% | — |
| surprised | 84 | 39.29% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 84 | 42.19% | 96.43% | 58.70% |
| disgusted | 84 | 85.71% | 35.71% | 50.42% |
| fearful | 84 | 54.00% | 64.29% | 58.70% |
| happy | 84 | 63.10% | 63.10% | 63.10% |
| neutral | 84 | 52.99% | 73.81% | 61.69% |
| sad | 84 | 88.89% | 19.05% | 31.37% |
| surprised | 84 | 78.57% | 39.29% | 52.38% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emovo_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (84) | **96.4%** | 0.0% | 1.2% | 1.2% | 1.2% | 0.0% | 0.0% |
| disgusted (84) | 28.6% | **35.7%** | 16.7% | 8.3% | 3.6% | 2.4% | 4.8% |
| fearful (84) | 22.6% | 0.0% | **64.3%** | 3.6% | 8.3% | 0.0% | 1.2% |
| happy (84) | 26.2% | 0.0% | 0.0% | **63.1%** | 6.0% | 0.0% | 4.8% |
| neutral (84) | 25.0% | 0.0% | 0.0% | 1.2% | **73.8%** | 0.0% | 0.0% |
| sad (84) | 8.3% | 6.0% | 21.4% | 2.4% | 42.9% | **19.0%** | 0.0% |
| surprised (84) | 21.4% | 0.0% | 15.5% | 20.2% | 3.6% | 0.0% | **39.3%** |

## Caveats

- **Training involvement for Melati: No corpus-specific evidence**. Source: Public SER head trained on nine named corpora including JL-Corpus and Emozionalmente; Whisper-v3 pretraining incompletely disclosed; author nEMO threshold calibration is counted as seen development use, although its thresholds are not applied here. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emovo_offline_audit.json)
