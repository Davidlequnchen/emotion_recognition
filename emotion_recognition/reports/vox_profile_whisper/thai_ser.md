# Vox-Profile Whisper on THAI-SER (high-agreement subset)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 11,848 |
| Accuracy: acted emotion | 69.29% |
| UAR: mean ground-truth-row recall | 70.06% |
| Macro F1 | 69.30% |
| Weighted F1: true-support weighted | 68.82% |
| Accuracy: all model classes | 61.23% |
| Accuracy: acted emotion or listener vote | 75.18% |
| Single listener vote vs acted emotion | 77.23% |
| Model vs single listener vote | 65.27% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_whisper/thai_ser_predictions.csv) and [metadata JSON](../../outputs/vox_profile_whisper/thai_ser_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **THAI-SER (high-agreement subset)**, Thai; 11,848 / 11,848 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: neutral, angry, happy, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 2566.4, "ms_per_clip": 216.6, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| neutral | 2,999 | 85.56% | 86.26% |
| angry | 2,883 | 80.65% | 83.84% |
| happy | 3,665 | 50.89% | 62.05% |
| sad | 2,301 | 63.15% | 70.80% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 2,999 | 58.76% | 85.56% | 69.67% |
| angry | 2,883 | 67.84% | 80.65% | 73.69% |
| happy | 3,665 | 82.93% | 50.89% | 63.07% |
| sad | 2,301 | 80.50% | 63.15% | 70.77% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/thai_ser_confusion.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (2,999) | **85.6%** | 2.1% | 5.0% | 7.3% |
| angry (2,883) | 15.3% | **80.6%** | 2.2% | 1.8% |
| happy (3,665) | 22.5% | 24.5% | **50.9%** | 2.2% |
| sad (2,301) | 23.3% | 6.2% | 7.3% | **63.1%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/thai_ser_confusion_acted_or_listener.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (3,621) | **88.6%** | 1.7% | 3.9% | 5.7% |
| angry (2,810) | 12.5% | **83.4%** | 2.2% | 1.9% |
| happy (3,275) | 13.4% | 26.8% | **57.5%** | 2.3% |
| sad (2,142) | 17.2% | 6.6% | 7.5% | **68.6%** |

## Caveats

- **Training involvement for Vox-Profile Whisper: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; retained Whisper-large-v3 pretraining sources incompletely disclosed. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](thai_ser_offline_audit.json)
