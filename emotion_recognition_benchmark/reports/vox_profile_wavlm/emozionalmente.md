# Vox-Profile WavLM on Emozionalmente

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 6,902 |
| Accuracy: acted emotion | 27.83% |
| UAR: mean ground-truth-row recall | 27.83% |
| Macro F1 | 23.72% |
| Weighted F1: true-support weighted | 23.72% |
| Accuracy: all model classes | 27.72% |
| Accuracy: acted emotion or listener vote | 37.47% |
| Single listener vote vs acted emotion | 81.22% |
| Model vs single listener vote | 33.59% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_wavlm/emozionalmente_predictions.csv) and [metadata JSON](../../outputs/vox_profile_wavlm/emozionalmente_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-07`.
- Dataset scope recorded in JSON: **Emozionalmente**, Italian; 6,902 / 6,902 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 817.2, "ms_per_clip": 118.4, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 986 | 26.06% | 38.34% |
| disgusted | 986 | 1.12% | 15.21% |
| fearful | 986 | 8.52% | 22.72% |
| happy | 986 | 29.41% | 40.16% |
| neutral | 986 | 98.48% | 98.99% |
| sad | 986 | 19.98% | 28.09% |
| surprised | 986 | 11.26% | 18.76% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 986 | 50.99% | 26.06% | 34.50% |
| disgusted | 986 | 31.43% | 1.12% | 2.15% |
| fearful | 986 | 35.90% | 8.52% | 13.77% |
| happy | 986 | 51.97% | 29.41% | 37.56% |
| neutral | 986 | 20.05% | 98.48% | 33.32% |
| sad | 986 | 35.69% | 19.98% | 25.62% |
| surprised | 986 | 63.07% | 11.26% | 19.10% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emozionalmente_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (986) | **26.1%** | 1.3% | 4.1% | 4.0% | 63.2% | 0.9% | 0.5% |
| disgusted (986) | 2.1% | **1.1%** | 1.1% | 4.5% | 83.7% | 6.9% | 0.6% |
| fearful (986) | 2.6% | 0.6% | **8.5%** | 5.7% | 58.7% | 22.6% | 1.2% |
| happy (986) | 11.5% | 0.0% | 3.1% | **29.4%** | 49.8% | 2.0% | 4.2% |
| neutral (986) | 0.7% | 0.0% | 0.0% | 0.2% | **98.5%** | 0.6% | 0.0% |
| sad (986) | 0.4% | 0.1% | 0.1% | 3.3% | 76.0% | **20.0%** | 0.1% |
| surprised (986) | 7.7% | 0.4% | 6.8% | 9.5% | 61.4% | 2.9% | **11.3%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/emozionalmente_confusion_acted_or_listener.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (889) | **31.6%** | 1.5% | 4.2% | 3.9% | 57.4% | 1.0% | 0.4% |
| disgusted (847) | 2.1% | **1.3%** | 0.9% | 4.8% | 84.4% | 5.7% | 0.7% |
| fearful (858) | 2.7% | 0.7% | **11.2%** | 5.8% | 60.7% | 18.2% | 0.7% |
| happy (913) | 10.8% | 0.0% | 3.0% | **35.4%** | 45.7% | 1.9% | 3.3% |
| neutral (1,465) | 0.5% | 0.0% | 0.0% | 0.1% | **99.3%** | 0.1% | 0.0% |
| sad (1,000) | 0.4% | 0.1% | 0.0% | 3.2% | 67.1% | **29.1%** | 0.1% |
| surprised (930) | 7.7% | 0.4% | 7.1% | 8.2% | 59.6% | 3.1% | **13.9%** |

## Caveats

- **Training involvement for Vox-Profile WavLM: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; WavLM-large backbone trained on Libri-Light, GigaSpeech and VoxPopuli; no clip-level overlap audit. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emozionalmente_offline_audit.json)
