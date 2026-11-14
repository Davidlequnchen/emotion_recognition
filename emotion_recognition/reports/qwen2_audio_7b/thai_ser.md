# Qwen2-Audio 7B Instruct on THAI-SER (high-agreement subset)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 11,848 |
| Accuracy: acted emotion | 25.27% |
| UAR: mean ground-truth-row recall | 26.40% |
| Macro F1 | 25.75% |
| Weighted F1: true-support weighted | 24.80% |
| Accuracy: all model classes | 25.27% |
| Invalid / outside-candidate answers | 4,604 (38.86%) |
| Accuracy: acted emotion or listener vote | 25.84% |
| Single listener vote vs acted emotion | 77.23% |
| Model vs single listener vote | 20.35% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/qwen2_audio_7b/thai_ser_predictions.csv) and [metadata JSON](../../outputs/qwen2_audio_7b/thai_ser_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-13`.
- Dataset scope recorded in JSON: **THAI-SER (high-agreement subset)**, Thai; 11,848 / 11,848 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: neutral, angry, happy, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `Qwen/Qwen2-Audio-7B-Instruct`, prompt hash `e683d765d4ac`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 4150.1, "ms_per_clip": 350.3, "on_cpu": 0}`.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Prediction distribution

Counts and shares use all saved samples. The invalid bucket combines `invalid` and `other answer`: answers outside the dataset's candidate classes remain errors.

| Predicted label | Clips | Share of all clips |
|---|---:|---:|
| neutral | 252 | 2.13% |
| angry | 5,452 | 46.02% |
| happy | 415 | 3.50% |
| sad | 1,125 | 9.50% |
| invalid / outside candidates | 4,604 | 38.86% |

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| neutral | 2,999 | 4.00% | 4.13% |
| angry | 2,883 | 69.44% | 69.61% |
| happy | 3,665 | 9.66% | 11.00% |
| sad | 2,301 | 22.51% | 22.95% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 2,999 | 47.62% | 4.00% | 7.38% |
| angry | 2,883 | 36.72% | 69.44% | 48.04% |
| happy | 3,665 | 85.30% | 9.66% | 17.35% |
| sad | 2,301 | 46.04% | 22.51% | 30.24% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/thai_ser_confusion.png)

| Ground truth \ Predicted | neutral | angry | happy | sad | other answer |
|---|---|---|---|---|---|
| neutral (2,999) | **4.0%** | 33.4% | 0.8% | 9.7% | 52.1% |
| angry (2,883) | 0.8% | **69.4%** | 0.5% | 4.4% | 24.8% |
| happy (3,665) | 1.7% | 54.7% | **9.7%** | 5.2% | 28.8% |
| sad (2,301) | 2.0% | 19.3% | 1.0% | **22.5%** | 55.2% |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/thai_ser_confusion_acted_or_listener.png)

| Ground truth \ Predicted | neutral | angry | happy | sad | other answer |
|---|---|---|---|---|---|
| neutral (3,040) | **5.4%** | 32.9% | 0.8% | 9.5% | 51.4% |
| angry (2,897) | 0.7% | **69.8%** | 0.5% | 4.3% | 24.7% |
| happy (3,617) | 0.9% | 55.0% | **9.8%** | 5.2% | 29.1% |
| sad (2,294) | 1.6% | 19.4% | 1.0% | **22.7%** | 55.4% |

## Caveats

- **Training involvement for Qwen2-Audio 7B Instruct: No corpus-specific evidence**. Source: Qwen-7B and Whisper-large-v3 initialization; report states evaluation-data exclusion, but full backbone and pre/post-training corpus provenance is unavailable. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](thai_ser_offline_audit.json)
