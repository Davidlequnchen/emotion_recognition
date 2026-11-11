# Vox-Profile WavLM on MELD

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,608 |
| Accuracy: acted emotion | 34.85% |
| UAR: mean ground-truth-row recall | 29.42% |
| Macro F1 | 26.80% |
| Weighted F1: true-support weighted | 36.79% |
| Accuracy: all model classes | 33.90% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/vox_profile_wavlm/meld_predictions.csv) and [metadata JSON](../../outputs/vox_profile_wavlm/meld_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-07`.
- Dataset scope recorded in JSON: **MELD**, English; 2,608 / 2,608 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 386.9, "ms_per_clip": 148.3, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 345 | 44.35% | — |
| disgusted | 67 | 16.42% | — |
| fearful | 50 | 2.00% | — |
| happy | 401 | 54.11% | — |
| neutral | 1,256 | 30.25% | — |
| sad | 208 | 25.00% | — |
| surprised | 281 | 33.81% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 345 | 36.52% | 44.35% | 40.05% |
| disgusted | 67 | 6.63% | 16.42% | 9.44% |
| fearful | 50 | 0.74% | 2.00% | 1.08% |
| happy | 401 | 25.59% | 54.11% | 34.75% |
| neutral | 1,256 | 68.72% | 30.25% | 42.01% |
| sad | 208 | 37.68% | 25.00% | 30.06% |
| surprised | 281 | 27.30% | 33.81% | 30.21% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/meld_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (345) | **44.3%** | 9.3% | 4.3% | 23.8% | 9.0% | 1.7% | 7.5% |
| disgusted (67) | 23.9% | **16.4%** | 7.5% | 29.9% | 10.4% | 0.0% | 11.9% |
| fearful (50) | 20.0% | 8.0% | **2.0%** | 30.0% | 20.0% | 6.0% | 14.0% |
| happy (401) | 8.5% | 3.5% | 4.2% | **54.1%** | 14.0% | 2.7% | 13.0% |
| neutral (1,256) | 10.0% | 5.8% | 5.3% | 32.7% | **30.3%** | 4.8% | 11.1% |
| sad (208) | 14.9% | 8.7% | 6.7% | 18.8% | 15.9% | **25.0%** | 10.1% |
| surprised (281) | 17.4% | 5.0% | 6.0% | 22.8% | 12.8% | 2.1% | **33.8%** |

## Caveats

- **Training involvement for Vox-Profile WavLM: No corpus-specific evidence**. Source: SER adaptation on MSP-Podcast v1.12; WavLM-large backbone trained on Libri-Light, GigaSpeech and VoxPopuli; no clip-level overlap audit. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](meld_offline_audit.json)
