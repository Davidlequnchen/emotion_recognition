# Qwen2-Audio 7B Instruct on MESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,800 |
| Accuracy: acted emotion | 30.50% |
| UAR: mean ground-truth-row recall | 30.50% |
| Macro F1 | 21.26% |
| Weighted F1: true-support weighted | 21.26% |
| Accuracy: all model classes | 30.50% |
| Invalid / outside-candidate answers | 6 (0.33%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/mess_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/mess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **MESS**, English; 1,800 / 1,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `e683d765d4ac`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 2483.6, "ms_per_clip": 1379.8, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 1,143 | 63.50% |
| happy | 7 | 0.39% |
| neutral | 636 | 35.33% |
| sad | 8 | 0.44% |
| invalid / outside candidates | 6 | 0.33% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 450 | 72.44% | — |
| happy | 450 | 1.56% | — |
| neutral | 450 | 46.89% | — |
| sad | 450 | 1.11% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 450 | 28.52% | 72.44% | 40.93% |
| happy | 450 | 100.00% | 1.56% | 3.06% |
| neutral | 450 | 33.18% | 46.89% | 38.86% |
| sad | 450 | 62.50% | 1.11% | 2.18% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/mess_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad | other answer |
|---|---|---|---|---|---|
| angry (450) | **72.4%** | 0.0% | 27.3% | 0.2% | 0.0% |
| happy (450) | 78.4% | **1.6%** | 19.6% | 0.4% | 0.0% |
| neutral (450) | 52.9% | 0.0% | **46.9%** | 0.0% | 0.2% |
| sad (450) | 50.2% | 0.0% | 47.6% | **1.1%** | 1.1% |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](mess_offline_audit.json)
