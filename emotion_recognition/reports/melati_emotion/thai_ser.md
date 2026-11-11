# Melati on THAI-SER (high-agreement subset)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 11,848 |
| Accuracy: acted emotion | 72.70% |
| UAR: mean ground-truth-row recall | 71.76% |
| Macro F1 | 72.15% |
| Weighted F1: true-support weighted | 72.44% |
| Accuracy: all model classes | 70.25% |
| Accuracy: acted emotion or listener vote | 75.73% |
| Single listener vote vs acted emotion | 77.23% |
| Model vs single listener vote | 63.38% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/melati_emotion/thai_ser_predictions.csv) and [metadata JSON](../../outputs/melati_emotion/thai_ser_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **THAI-SER (high-agreement subset)**, Thai; 11,848 / 11,848 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: neutral, angry, happy, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 3588.9, "ms_per_clip": 302.9, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| neutral | 2,999 | 72.42% | 72.82% |
| angry | 2,883 | 86.61% | 88.10% |
| happy | 3,665 | 73.32% | 78.14% |
| sad | 2,301 | 54.67% | 60.19% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 2,999 | 74.49% | 72.42% | 73.44% |
| angry | 2,883 | 65.83% | 86.61% | 74.81% |
| happy | 3,665 | 72.31% | 73.32% | 72.81% |
| sad | 2,301 | 88.40% | 54.67% | 67.56% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/thai_ser_confusion.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (2,999) | **72.4%** | 11.9% | 13.4% | 2.3% |
| angry (2,883) | 4.1% | **86.6%** | 8.9% | 0.5% |
| happy (3,665) | 5.0% | 19.4% | **73.3%** | 2.3% |
| sad (2,301) | 19.3% | 9.9% | 16.2% | **54.7%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/thai_ser_confusion_acted_or_listener.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (3,305) | **75.3%** | 10.8% | 11.9% | 2.0% |
| angry (2,859) | 2.7% | **88.0%** | 8.9% | 0.4% |
| happy (3,506) | 0.7% | 19.8% | **77.2%** | 2.3% |
| sad (2,178) | 14.9% | 10.4% | 16.8% | **57.9%** |

## Caveats

- **Training involvement for Melati: No corpus-specific evidence**. Source: Public SER head trained on nine named corpora including JL-Corpus and Emozionalmente; Whisper-v3 pretraining incompletely disclosed; author nEMO threshold calibration is counted as seen development use, although its thresholds are not applied here. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](thai_ser_offline_audit.json)
