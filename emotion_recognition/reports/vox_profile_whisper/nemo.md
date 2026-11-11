# Vox-Profile Whisper on nEMO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 4,481 |
| Accuracy: acted emotion | 32.78% |
| UAR: mean ground-truth-row recall | 32.00% |
| Macro F1 | 28.79% |
| Weighted F1: true-support weighted | 29.07% |
| Accuracy: all model classes | 32.56% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_whisper/nemo_predictions.csv) and [metadata JSON](../../outputs/vox_profile_whisper/nemo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **nEMO**, Polish; 4,481 / 4,481 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 909.7, "ms_per_clip": 203.0, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 749 | 7.88% | — |
| fearful | 736 | 19.57% | — |
| happy | 749 | 53.00% | — |
| neutral | 809 | 71.69% | — |
| sad | 769 | 22.37% | — |
| surprised | 669 | 17.49% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 749 | 71.95% | 7.88% | 14.20% |
| fearful | 736 | 26.18% | 19.57% | 22.40% |
| happy | 749 | 29.78% | 53.00% | 38.14% |
| neutral | 809 | 31.56% | 71.69% | 43.82% |
| sad | 769 | 41.35% | 22.37% | 29.03% |
| surprised | 669 | 44.66% | 17.49% | 25.13% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/nemo_confusion.png)

| Ground truth \ Predicted | angry | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|
| angry (749) | **7.9%** | 26.2% | 29.4% | 27.5% | 2.4% | 6.7% |
| fearful (736) | 0.3% | **19.6%** | 28.1% | 29.5% | 18.2% | 4.3% |
| happy (749) | 0.8% | 8.1% | **53.0%** | 29.0% | 2.5% | 6.5% |
| neutral (809) | 0.5% | 1.4% | 18.9% | **71.7%** | 6.6% | 1.0% |
| sad (769) | 0.0% | 1.0% | 11.2% | 64.6% | **22.4%** | 0.8% |
| surprised (669) | 1.6% | 19.4% | 40.4% | 18.1% | 3.0% | **17.5%** |

## Caveats

- **Training involvement for Vox-Profile Whisper: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; retained Whisper-large-v3 pretraining sources incompletely disclosed. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](nemo_offline_audit.json)
