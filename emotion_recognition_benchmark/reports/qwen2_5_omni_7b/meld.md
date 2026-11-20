# Qwen2.5-Omni 7B on MELD

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,608 |
| Accuracy: acted emotion | 51.27% |
| UAR: mean ground-truth-row recall | 43.45% |
| Macro F1 | 40.43% |
| Weighted F1: true-support weighted | 52.39% |
| Accuracy: all model classes | 51.27% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_5_omni_7b/meld_predictions.csv) and [metadata JSON](../../outputs/qwen2_5_omni_7b/meld_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-09`.
- Dataset scope recorded in JSON: **MELD**, English; 2,608 / 2,608 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2.5-Omni-7B`, prompt hash `aab2967fd07a`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 1858.7, "ms_per_clip": 712.7, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 345 | 37.68% | — |
| disgusted | 67 | 37.31% | — |
| fearful | 50 | 36.00% | — |
| happy | 401 | 57.61% | — |
| neutral | 1,256 | 59.00% | — |
| sad | 208 | 31.73% | — |
| surprised | 281 | 44.84% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 345 | 44.37% | 37.68% | 40.75% |
| disgusted | 67 | 14.97% | 37.31% | 21.37% |
| fearful | 50 | 23.08% | 36.00% | 28.12% |
| happy | 401 | 40.24% | 57.61% | 47.38% |
| neutral | 1,256 | 70.04% | 59.00% | 64.04% |
| sad | 208 | 39.76% | 31.73% | 35.29% |
| surprised | 281 | 47.37% | 44.84% | 46.07% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/meld_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised | other answer |
|---|---|---|---|---|---|---|---|---|
| angry (345) | **37.7%** | 13.3% | 3.5% | 12.8% | 21.7% | 4.3% | 6.7% | 0.0% |
| disgusted (67) | 22.4% | **37.3%** | 4.5% | 11.9% | 17.9% | 0.0% | 3.0% | 3.0% |
| fearful (50) | 8.0% | 4.0% | **36.0%** | 10.0% | 24.0% | 6.0% | 10.0% | 2.0% |
| happy (401) | 3.2% | 2.2% | 2.0% | **57.6%** | 25.2% | 2.5% | 7.2% | 0.0% |
| neutral (1,256) | 6.7% | 4.0% | 1.4% | 18.4% | **59.0%** | 5.0% | 5.3% | 0.2% |
| sad (208) | 9.1% | 7.2% | 5.3% | 9.6% | 30.3% | **31.7%** | 6.7% | 0.0% |
| surprised (281) | 10.0% | 7.1% | 2.8% | 12.5% | 19.2% | 3.2% | **44.8%** | 0.4% |

## Caveats

- **Training involvement for Qwen2.5-Omni 7B: No corpus-specific evidence**. Source: Qwen2.5 and Whisper-large-v3 initialization plus multimodal pre/post-training; complete corpus-level provenance not published. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](meld_offline_audit.json)
