# Qwen2-Audio 7B Instruct on RAVDESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,440 |
| Accuracy: acted emotion | 82.29% |
| UAR: mean ground-truth-row recall | 84.57% |
| Macro F1 | 83.12% |
| Weighted F1: true-support weighted | 82.05% |
| Accuracy: all model classes | 82.29% |
| Invalid / outside-candidate answers | 0 (0.00%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/ravdess_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/ravdess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **RAVDESS**, English; 1,440 / 1,440 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: neutral, happy, sad, angry, fearful, disgusted, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 2245.1, "ms_per_clip": 1559.1, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| neutral | 145 | 10.07% |
| happy | 161 | 11.18% |
| sad | 271 | 18.82% |
| angry | 166 | 11.53% |
| fearful | 212 | 14.72% |
| disgusted | 283 | 19.65% |
| surprised | 202 | 14.03% |
| invalid / outside candidates | 0 | 0.00% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| neutral | 288 | 50.35% | — |
| happy | 192 | 82.81% | — |
| sad | 192 | 81.25% | — |
| angry | 192 | 85.42% | — |
| fearful | 192 | 97.40% | — |
| disgusted | 192 | 100.00% | — |
| surprised | 192 | 94.79% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 288 | 100.00% | 50.35% | 66.97% |
| happy | 192 | 98.76% | 82.81% | 90.08% |
| sad | 192 | 57.56% | 81.25% | 67.39% |
| angry | 192 | 98.80% | 85.42% | 91.62% |
| fearful | 192 | 88.21% | 97.40% | 92.57% |
| disgusted | 192 | 67.84% | 100.00% | 80.84% |
| surprised | 192 | 90.10% | 94.79% | 92.39% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/ravdess_confusion.png)

| Ground truth \ Predicted | neutral | happy | sad | angry | fearful | disgusted | surprised |
|---|---|---|---|---|---|---|---|
| neutral (288) | **50.3%** | 0.3% | 38.2% | 0.0% | 0.0% | 10.8% | 0.3% |
| happy (192) | 0.0% | **82.8%** | 1.0% | 1.0% | 0.0% | 5.2% | 9.9% |
| sad (192) | 0.0% | 0.5% | **81.2%** | 0.0% | 8.9% | 9.4% | 0.0% |
| angry (192) | 0.0% | 0.0% | 0.0% | **85.4%** | 1.0% | 13.5% | 0.0% |
| fearful (192) | 0.0% | 0.0% | 1.6% | 0.0% | **97.4%** | 1.0% | 0.0% |
| disgusted (192) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | **100.0%** | 0.0% |
| surprised (192) | 0.0% | 0.0% | 0.0% | 0.0% | 3.1% | 2.1% | **94.8%** |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](ravdess_offline_audit.json)
