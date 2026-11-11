# Vox-Profile WavLM on MESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,800 |
| Accuracy: acted emotion | 36.33% |
| UAR: mean ground-truth-row recall | 36.33% |
| Macro F1 | 28.72% |
| Weighted F1: true-support weighted | 28.72% |
| Accuracy: all model classes | 33.00% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_wavlm/mess_predictions.csv) and [metadata JSON](../../outputs/vox_profile_wavlm/mess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-10-29`.
- Dataset scope recorded in JSON: **MESS**, English; 1,800 / 1,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 459.6, "ms_per_clip": 255.3, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 450 | 13.11% | — |
| happy | 450 | 32.00% | — |
| neutral | 450 | 99.11% | — |
| sad | 450 | 1.11% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 450 | 89.39% | 13.11% | 22.87% |
| happy | 450 | 75.00% | 32.00% | 44.86% |
| neutral | 450 | 29.09% | 99.11% | 44.98% |
| sad | 450 | 55.56% | 1.11% | 2.18% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/mess_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad |
|---|---|---|---|---|
| angry (450) | **13.1%** | 9.1% | 77.8% | 0.0% |
| happy (450) | 0.9% | **32.0%** | 66.9% | 0.2% |
| neutral (450) | 0.0% | 0.2% | **99.1%** | 0.7% |
| sad (450) | 0.7% | 1.3% | 96.9% | **1.1%** |

## Caveats

- **Training involvement for Vox-Profile WavLM: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; WavLM-large backbone trained on Libri-Light, GigaSpeech and VoxPopuli; no clip-level overlap audit. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](mess_offline_audit.json)
