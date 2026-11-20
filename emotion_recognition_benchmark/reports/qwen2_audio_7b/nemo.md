# Qwen2-Audio 7B Instruct on nEMO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 4,481 |
| Accuracy: acted emotion | 26.27% |
| UAR: mean ground-truth-row recall | 26.21% |
| Macro F1 | 18.88% |
| Weighted F1: true-support weighted | 19.06% |
| Accuracy: all model classes | 26.27% |
| Invalid / outside-candidate answers | 35 (0.78%) |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/nemo_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/nemo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **nEMO**, Polish; 4,481 / 4,481 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `6c3ec6823d3d`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 6053.2, "ms_per_clip": 1350.9, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| angry | 2,198 | 49.05% |
| fearful | 1,802 | 40.21% |
| happy | 44 | 0.98% |
| neutral | 83 | 1.85% |
| sad | 318 | 7.10% |
| surprised | 1 | 0.02% |
| invalid / outside candidates | 35 | 0.78% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 749 | 68.49% | — |
| fearful | 736 | 58.70% | — |
| happy | 749 | 1.74% | — |
| neutral | 809 | 2.60% | — |
| sad | 769 | 25.75% | — |
| surprised | 669 | 0.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 749 | 23.34% | 68.49% | 34.82% |
| fearful | 736 | 23.97% | 58.70% | 34.04% |
| happy | 749 | 29.55% | 1.74% | 3.28% |
| neutral | 809 | 25.30% | 2.60% | 4.71% |
| sad | 769 | 62.26% | 25.75% | 36.43% |
| surprised | 669 | 0.00% | 0.00% | 0.00% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/nemo_confusion.png)

| Ground truth \ Predicted | angry | fearful | happy | neutral | sad | surprised | other answer |
|---|---|---|---|---|---|---|---|
| angry (749) | **68.5%** | 28.6% | 0.8% | 0.5% | 0.7% | 0.0% | 0.9% |
| fearful (736) | 34.4% | **58.7%** | 0.4% | 0.7% | 5.3% | 0.0% | 0.5% |
| happy (749) | 63.8% | 30.7% | **1.7%** | 1.7% | 0.9% | 0.1% | 0.9% |
| neutral (809) | 41.8% | 46.7% | 1.1% | **2.6%** | 7.3% | 0.0% | 0.5% |
| sad (769) | 24.8% | 42.4% | 0.9% | 4.7% | **25.7%** | 0.0% | 1.4% |
| surprised (669) | 63.5% | 33.2% | 0.9% | 0.6% | 1.5% | **0.0%** | 0.3% |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](nemo_offline_audit.json)
