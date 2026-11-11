# MERaLiON-SER-v1 on THAI-SER (high-agreement subset)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 11,848 |
| Accuracy: acted emotion | 47.05% |
| UAR: mean ground-truth-row recall | 47.47% |
| Macro F1 | 47.41% |
| Weighted F1: true-support weighted | 47.05% |
| Accuracy: all model classes | 47.02% |
| Accuracy: acted emotion or listener vote | 48.45% |
| Single listener vote vs acted emotion | 77.23% |
| Model vs single listener vote | 39.70% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/thai_ser_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/thai_ser_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **THAI-SER (high-agreement subset)**, Thai; 11,848 / 11,848 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: neutral, angry, happy, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 3385.8, "ms_per_clip": 285.8, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| neutral | 2,999 | 91.03% | 91.13% |
| angry | 2,883 | 94.66% | 95.11% |
| happy | 3,665 | 1.34% | 4.12% |
| sad | 2,301 | 2.87% | 4.95% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 2,999 | 93.56% | 91.03% | 92.28% |
| angry | 2,883 | 92.38% | 94.66% | 93.51% |
| happy | 3,665 | 2.19% | 1.34% | 1.66% |
| sad | 2,301 | 1.77% | 2.87% | 2.19% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/thai_ser_confusion.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (2,999) | **91.0%** | 1.7% | 1.7% | 5.5% |
| angry (2,883) | 1.1% | **94.7%** | 0.6% | 3.7% |
| happy (3,665) | 2.0% | 3.9% | **1.3%** | 92.7% |
| sad (2,301) | 3.7% | 1.3% | 92.2% | **2.9%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/thai_ser_confusion_acted_or_listener.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (3,119) | **91.5%** | 1.6% | 1.7% | 5.3% |
| angry (2,887) | 0.6% | **95.1%** | 0.6% | 3.7% |
| happy (3,564) | 0.3% | 3.7% | **1.4%** | 94.7% |
| sad (2,278) | 1.7% | 1.2% | 93.1% | **4.0%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: No corpus-specific evidence**. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](thai_ser_offline_audit.json)
