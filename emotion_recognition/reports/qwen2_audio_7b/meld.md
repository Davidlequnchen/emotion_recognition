# Qwen2-Audio 7B Instruct on MELD

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,608 |
| Accuracy: acted emotion | 30.71% |
| UAR: mean ground-truth-row recall | 39.55% |
| Macro F1 | 29.35% |
| Weighted F1: true-support weighted | 32.23% |
| Accuracy: all model classes | 30.71% |
| Invalid / outside-candidate answers | 16 (0.61%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/meld_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/meld_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-09`.
- Dataset scope recorded in JSON: **MELD**, English; 2,608 / 2,608 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 2625.6, "ms_per_clip": 1006.8, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 871 | 33.40% |
| disgusted | 304 | 11.66% |
| fearful | 364 | 13.96% |
| happy | 245 | 9.39% |
| neutral | 276 | 10.58% |
| sad | 277 | 10.62% |
| surprised | 255 | 9.78% |
| invalid / outside candidates | 16 | 0.61% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 345 | 59.71% | — |
| disgusted | 67 | 40.30% | — |
| fearful | 50 | 46.00% | — |
| happy | 401 | 32.67% | — |
| neutral | 1,256 | 17.20% | — |
| sad | 208 | 40.38% | — |
| surprised | 281 | 40.57% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 345 | 23.65% | 59.71% | 33.88% |
| disgusted | 67 | 8.88% | 40.30% | 14.56% |
| fearful | 50 | 6.32% | 46.00% | 11.11% |
| happy | 401 | 53.47% | 32.67% | 40.56% |
| neutral | 1,256 | 78.26% | 17.20% | 28.20% |
| sad | 208 | 30.32% | 40.38% | 34.64% |
| surprised | 281 | 44.71% | 40.57% | 42.54% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/meld_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised | other answer |
|---|---|---|---|---|---|---|---|---|
| angry (345) | **59.7%** | 14.8% | 11.0% | 2.3% | 2.3% | 5.5% | 4.3% | 0.0% |
| disgusted (67) | 43.3% | **40.3%** | 6.0% | 0.0% | 1.5% | 4.5% | 4.5% | 0.0% |
| fearful (50) | 24.0% | 8.0% | **46.0%** | 4.0% | 4.0% | 10.0% | 4.0% | 0.0% |
| happy (401) | 29.9% | 8.7% | 6.2% | **32.7%** | 7.7% | 5.0% | 8.2% | 1.5% |
| neutral (1,256) | 29.5% | 10.8% | 18.0% | 7.1% | **17.2%** | 10.4% | 6.3% | 0.6% |
| sad (208) | 22.6% | 11.5% | 13.0% | 3.4% | 4.3% | **40.4%** | 4.3% | 0.5% |
| surprised (281) | 30.6% | 9.6% | 7.5% | 2.8% | 3.2% | 5.3% | **40.6%** | 0.4% |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](meld_offline_audit.json)
