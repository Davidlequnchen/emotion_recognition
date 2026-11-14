# Qwen2-Audio 7B Instruct on TESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,800 |
| Accuracy: acted emotion | 92.64% |
| UAR: mean ground-truth-row recall | 92.64% |
| Macro F1 | 92.34% |
| Weighted F1: true-support weighted | 92.34% |
| Accuracy: all model classes | 92.64% |
| Invalid / outside-candidate answers | 0 (0.00%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/tess_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/tess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **TESS**, English; 2,800 / 2,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, surprised, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 4298.8, "ms_per_clip": 1535.3, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 423 | 15.11% |
| disgusted | 412 | 14.71% |
| fearful | 423 | 15.11% |
| happy | 374 | 13.36% |
| neutral | 243 | 8.68% |
| surprised | 387 | 13.82% |
| sad | 538 | 19.21% |
| invalid / outside candidates | 0 | 0.00% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 400 | 99.00% | — |
| disgusted | 400 | 100.00% | — |
| fearful | 400 | 99.75% | — |
| happy | 400 | 93.50% | — |
| neutral | 400 | 60.50% | — |
| surprised | 400 | 96.75% | — |
| sad | 400 | 99.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 400 | 93.62% | 99.00% | 96.23% |
| disgusted | 400 | 97.09% | 100.00% | 98.52% |
| fearful | 400 | 94.33% | 99.75% | 96.96% |
| happy | 400 | 100.00% | 93.50% | 96.64% |
| neutral | 400 | 99.59% | 60.50% | 75.27% |
| surprised | 400 | 100.00% | 96.75% | 98.35% |
| sad | 400 | 73.61% | 99.00% | 84.43% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/tess_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | surprised | sad |
|---|---|---|---|---|---|---|---|
| angry (400) | **99.0%** | 0.2% | 0.5% | 0.0% | 0.2% | 0.0% | 0.0% |
| disgusted (400) | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fearful (400) | 0.2% | 0.0% | **99.8%** | 0.0% | 0.0% | 0.0% | 0.0% |
| happy (400) | 3.2% | 0.8% | 2.5% | **93.5%** | 0.0% | 0.0% | 0.0% |
| neutral (400) | 3.0% | 0.8% | 0.2% | 0.0% | **60.5%** | 0.0% | 35.5% |
| surprised (400) | 0.0% | 0.5% | 2.8% | 0.0% | 0.0% | **96.8%** | 0.0% |
| sad (400) | 0.2% | 0.8% | 0.0% | 0.0% | 0.0% | 0.0% | **99.0%** |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](tess_offline_audit.json)
