# Qwen2-Audio 7B Instruct on JL-Corpus

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,200 |
| Accuracy: acted emotion | 48.58% |
| UAR: mean ground-truth-row recall | 48.58% |
| Macro F1 | 39.57% |
| Weighted F1: true-support weighted | 47.90% |
| Accuracy: all model classes | 48.58% |
| Invalid / outside-candidate answers | 63 (5.25%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/jl_corpus_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/jl_corpus_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **JL-Corpus**, English (New Zealand); 1,200 / 1,200 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `1add00540a85`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 1766.1, "ms_per_clip": 1471.7, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.
- Composite labels follow the existing accepted-label rule: excited accepts happy or surprised. F1 resolves a correct composite to the predicted class, otherwise to its first accepted class. Weighted F1 uses the resulting resolved-truth support, which can depend on the predictions.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 582 | 48.50% |
| happy | 130 | 10.83% |
| neutral | 227 | 18.92% |
| sad | 192 | 16.00% |
| surprised | 6 | 0.50% |
| invalid / outside candidates | 63 | 5.25% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 240 | 97.08% | — |
| happy | 240 | 22.92% | — |
| neutral | 240 | 44.58% | — |
| sad | 240 | 48.33% | — |
| excited (happy or surprised) | 240 | 30.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 240 | 40.03% | 97.08% | 56.69% |
| happy | 480 | 97.69% | 26.46% | 41.64% |
| neutral | 240 | 47.14% | 44.58% | 45.82% |
| sad | 240 | 60.42% | 48.33% | 53.70% |
| surprised | 0 | 0.00% | 0.00% | 0.00% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/jl_corpus_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad | surprised | other answer |
|---|---|---|---|---|---|---|
| angry (240) | **97.1%** | 1.2% | 0.4% | 0.0% | 0.0% | 1.2% |
| happy (240) | 48.8% | **22.9%** | 12.5% | 5.0% | 2.5% | 8.3% |
| neutral (240) | 24.6% | 0.0% | **44.6%** | 26.7% | 0.0% | 4.2% |
| sad (240) | 2.9% | 0.0% | 37.1% | **48.3%** | 0.0% | 11.7% |
| excited (happy or surprised) (240) | 69.2% | **30.0%** | 0.0% | 0.0% | **0.0%** | 0.8% |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](jl_corpus_offline_audit.json)
