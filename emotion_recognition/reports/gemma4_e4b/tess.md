# Gemma 4 E4B on TESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,800 |
| Accuracy: acted emotion | 14.50% |
| UAR: mean ground-truth-row recall | 14.50% |
| Macro F1 | 4.07% |
| Weighted F1: true-support weighted | 4.07% |
| Accuracy: all model classes | 14.50% |
| Invalid / outside-candidate answers | 0 (0.00%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/gemma4_e4b/tess_predictions.csv) and [metadata JSON](../../outputs/gemma4_e4b/tess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-14`.
- Dataset scope recorded in JSON: **TESS**, English; 2,800 / 2,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, surprised, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `google/gemma-4-E4B-it`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cuda:0", "hardware": "NVIDIA L4", "seconds": 3227.0, "ms_per_clip": 1152.5, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 2 | 0.07% |
| disgusted | 21 | 0.75% |
| fearful | 0 | 0.00% |
| happy | 2 | 0.07% |
| neutral | 2,772 | 99.00% |
| surprised | 3 | 0.11% |
| sad | 0 | 0.00% |
| invalid / outside candidates | 0 | 0.00% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 400 | 0.00% | — |
| disgusted | 400 | 1.50% | — |
| fearful | 400 | 0.00% | — |
| happy | 400 | 0.00% | — |
| neutral | 400 | 99.75% | — |
| surprised | 400 | 0.25% | — |
| sad | 400 | 0.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 400 | 0.00% | 0.00% | 0.00% |
| disgusted | 400 | 28.57% | 1.50% | 2.85% |
| fearful | 400 | 0.00% | 0.00% | 0.00% |
| happy | 400 | 0.00% | 0.00% | 0.00% |
| neutral | 400 | 14.39% | 99.75% | 25.16% |
| surprised | 400 | 33.33% | 0.25% | 0.50% |
| sad | 400 | 0.00% | 0.00% | 0.00% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/tess_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | surprised | sad |
|---|---|---|---|---|---|---|---|
| angry (400) | **0.0%** | 0.5% | 0.0% | 0.0% | 99.5% | 0.0% | 0.0% |
| disgusted (400) | 0.2% | **1.5%** | 0.0% | 0.2% | 97.8% | 0.2% | 0.0% |
| fearful (400) | 0.0% | 0.0% | **0.0%** | 0.0% | 100.0% | 0.0% | 0.0% |
| happy (400) | 0.2% | 0.8% | 0.0% | **0.0%** | 98.8% | 0.2% | 0.0% |
| neutral (400) | 0.0% | 0.2% | 0.0% | 0.0% | **99.8%** | 0.0% | 0.0% |
| surprised (400) | 0.0% | 2.0% | 0.0% | 0.2% | 97.5% | **0.2%** | 0.0% |
| sad (400) | 0.0% | 0.2% | 0.0% | 0.0% | 99.8% | 0.0% | **0.0%** |

## Caveats

- **Training involvement for Gemma 4 E4B: No corpus-specific evidence**. Source: Broad multimodal pretraining and instruction tuning; complete corpus-level provenance not published (Google). Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](tess_offline_audit.json)
