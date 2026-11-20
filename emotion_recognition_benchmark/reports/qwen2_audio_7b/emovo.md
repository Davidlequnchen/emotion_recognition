# Qwen2-Audio 7B Instruct on EMOVO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 588 |
| Accuracy: acted emotion | 26.87% |
| UAR: mean ground-truth-row recall | 26.87% |
| Macro F1 | 21.78% |
| Weighted F1: true-support weighted | 21.78% |
| Accuracy: all model classes | 26.87% |
| Invalid / outside-candidate answers | 0 (0.00%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/emovo_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/emovo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **EMOVO**, Italian; 588 / 588 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 826.6, "ms_per_clip": 1405.8, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 425 | 72.28% |
| disgusted | 26 | 4.42% |
| fearful | 70 | 11.90% |
| happy | 6 | 1.02% |
| neutral | 22 | 3.74% |
| sad | 36 | 6.12% |
| surprised | 3 | 0.51% |
| invalid / outside candidates | 0 | 0.00% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 84 | 98.81% | — |
| disgusted | 84 | 11.90% | — |
| fearful | 84 | 35.71% | — |
| happy | 84 | 2.38% | — |
| neutral | 84 | 10.71% | — |
| sad | 84 | 26.19% | — |
| surprised | 84 | 2.38% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 84 | 19.53% | 98.81% | 32.61% |
| disgusted | 84 | 38.46% | 11.90% | 18.18% |
| fearful | 84 | 42.86% | 35.71% | 38.96% |
| happy | 84 | 33.33% | 2.38% | 4.44% |
| neutral | 84 | 40.91% | 10.71% | 16.98% |
| sad | 84 | 61.11% | 26.19% | 36.67% |
| surprised | 84 | 66.67% | 2.38% | 4.60% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emovo_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (84) | **98.8%** | 0.0% | 0.0% | 1.2% | 0.0% | 0.0% | 0.0% |
| disgusted (84) | 67.9% | **11.9%** | 9.5% | 1.2% | 1.2% | 8.3% | 0.0% |
| fearful (84) | 60.7% | 1.2% | **35.7%** | 0.0% | 1.2% | 1.2% | 0.0% |
| happy (84) | 88.1% | 0.0% | 3.6% | **2.4%** | 3.6% | 1.2% | 1.2% |
| neutral (84) | 67.9% | 3.6% | 11.9% | 2.4% | **10.7%** | 3.6% | 0.0% |
| sad (84) | 39.3% | 13.1% | 13.1% | 0.0% | 8.3% | **26.2%** | 0.0% |
| surprised (84) | 83.3% | 1.2% | 9.5% | 0.0% | 1.2% | 2.4% | **2.4%** |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emovo_offline_audit.json)
