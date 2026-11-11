# Vox-Profile WavLM on EMOVO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 588 |
| Accuracy: acted emotion | 29.25% |
| UAR: mean ground-truth-row recall | 29.25% |
| Macro F1 | 22.50% |
| Weighted F1: true-support weighted | 22.50% |
| Accuracy: all model classes | 29.25% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_wavlm/emovo_predictions.csv) and [metadata JSON](../../outputs/vox_profile_wavlm/emovo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-07`.
- Dataset scope recorded in JSON: **EMOVO**, Italian; 588 / 588 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 74.7, "ms_per_clip": 127.0, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 84 | 75.00% | — |
| disgusted | 84 | 0.00% | — |
| fearful | 84 | 9.52% | — |
| happy | 84 | 10.71% | — |
| neutral | 84 | 98.81% | — |
| sad | 84 | 4.76% | — |
| surprised | 84 | 5.95% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 84 | 70.79% | 75.00% | 72.83% |
| disgusted | 84 | 0.00% | 0.00% | 0.00% |
| fearful | 84 | 50.00% | 9.52% | 16.00% |
| happy | 84 | 52.94% | 10.71% | 17.82% |
| neutral | 84 | 18.44% | 98.81% | 31.09% |
| sad | 84 | 57.14% | 4.76% | 8.79% |
| surprised | 84 | 71.43% | 5.95% | 10.99% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emovo_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (84) | **75.0%** | 0.0% | 0.0% | 2.4% | 22.6% | 0.0% | 0.0% |
| disgusted (84) | 8.3% | **0.0%** | 1.2% | 3.6% | 85.7% | 0.0% | 1.2% |
| fearful (84) | 1.2% | 2.4% | **9.5%** | 2.4% | 81.0% | 2.4% | 1.2% |
| happy (84) | 13.1% | 0.0% | 4.8% | **10.7%** | 71.4% | 0.0% | 0.0% |
| neutral (84) | 1.2% | 0.0% | 0.0% | 0.0% | **98.8%** | 0.0% | 0.0% |
| sad (84) | 0.0% | 0.0% | 0.0% | 0.0% | 95.2% | **4.8%** | 0.0% |
| surprised (84) | 7.1% | 0.0% | 3.6% | 1.2% | 81.0% | 1.2% | **6.0%** |

## Caveats

- **Training involvement for Vox-Profile WavLM: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; WavLM-large backbone trained on Libri-Light, GigaSpeech and VoxPopuli; no clip-level overlap audit. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emovo_offline_audit.json)
