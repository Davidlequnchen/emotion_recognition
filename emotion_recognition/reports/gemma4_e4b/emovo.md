# Gemma 4 E4B on EMOVO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 588 |
| Accuracy: acted emotion | 14.29% |
| UAR: mean ground-truth-row recall | 14.29% |
| Macro F1 | 4.74% |
| Weighted F1: true-support weighted | 4.74% |
| Accuracy: all model classes | 14.29% |
| Invalid / outside-candidate answers | 0 (0.00%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/gemma4_e4b/emovo_predictions.csv) and [metadata JSON](../../outputs/gemma4_e4b/emovo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-14`.
- Dataset scope recorded in JSON: **EMOVO**, Italian; 588 / 588 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `google/gemma-4-E4B-it`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M3 Pro", "seconds": 415.2, "ms_per_clip": 706.1, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 2 | 0.34% |
| disgusted | 4 | 0.68% |
| fearful | 8 | 1.36% |
| happy | 21 | 3.57% |
| neutral | 552 | 93.88% |
| sad | 0 | 0.00% |
| surprised | 1 | 0.17% |
| invalid / outside candidates | 0 | 0.00% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 84 | 1.19% | — |
| disgusted | 84 | 0.00% | — |
| fearful | 84 | 0.00% | — |
| happy | 84 | 3.57% | — |
| neutral | 84 | 95.24% | — |
| sad | 84 | 0.00% | — |
| surprised | 84 | 0.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 84 | 50.00% | 1.19% | 2.33% |
| disgusted | 84 | 0.00% | 0.00% | 0.00% |
| fearful | 84 | 0.00% | 0.00% | 0.00% |
| happy | 84 | 14.29% | 3.57% | 5.71% |
| neutral | 84 | 14.49% | 95.24% | 25.16% |
| sad | 84 | 0.00% | 0.00% | 0.00% |
| surprised | 84 | 0.00% | 0.00% | 0.00% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emovo_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (84) | **1.2%** | 2.4% | 2.4% | 7.1% | 85.7% | 0.0% | 1.2% |
| disgusted (84) | 0.0% | **0.0%** | 2.4% | 3.6% | 94.0% | 0.0% | 0.0% |
| fearful (84) | 0.0% | 0.0% | **0.0%** | 2.4% | 97.6% | 0.0% | 0.0% |
| happy (84) | 0.0% | 1.2% | 1.2% | **3.6%** | 94.0% | 0.0% | 0.0% |
| neutral (84) | 0.0% | 0.0% | 0.0% | 4.8% | **95.2%** | 0.0% | 0.0% |
| sad (84) | 0.0% | 1.2% | 1.2% | 1.2% | 96.4% | **0.0%** | 0.0% |
| surprised (84) | 1.2% | 0.0% | 2.4% | 2.4% | 94.0% | 0.0% | **0.0%** |

## Caveats

- **Training involvement for Gemma 4 E4B: No corpus-specific evidence**. Source: Broad multimodal pretraining and instruction tuning; complete corpus-level provenance not published (Google). Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emovo_offline_audit.json)
