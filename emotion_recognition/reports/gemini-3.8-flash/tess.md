# Gemini 3.8 Flash on TESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 2,800 |
| Accuracy: acted emotion | 29.89% |
| UAR: mean ground-truth-row recall | 29.89% |
| Macro F1 | 24.13% |
| Accuracy: all model classes | 29.89% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/gemini-3.8-flash/tess_predictions.csv) and [metadata JSON](../../outputs/gemini-3.8-flash/tess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-10`.
- Dataset scope recorded in JSON: **TESS**, English; 2,800 / 2,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, surprised, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `gemini-3.8-flash`, prompt hash `dc84c979c561`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M3 Pro", "seconds": 480.0, "ms_per_clip": 171.4, "on_cpu": 0}`.
- The original notes report one 24 kHz audio clip per request.
- The original notes report up to 30 concurrent clip evaluations.
- The recorded ms_per_clip is total wall-clock time divided by the number of clips, not individual request latency; it is not directly comparable with local GPU inference timing.
- The original notes describe schema-constrained answers validated as literal JSON strings.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 400 | 7.50% | — |
| disgusted | 400 | 16.00% | — |
| fearful | 400 | 8.25% | — |
| happy | 400 | 41.00% | — |
| neutral | 400 | 52.00% | — |
| surprised | 400 | 2.75% | — |
| sad | 400 | 81.75% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 400 | 33.33% | 7.50% | 12.24% |
| disgusted | 400 | 47.41% | 16.00% | 23.93% |
| fearful | 400 | 37.50% | 8.25% | 13.52% |
| happy | 400 | 36.12% | 41.00% | 38.41% |
| neutral | 400 | 21.62% | 52.00% | 30.54% |
| surprised | 400 | 64.71% | 2.75% | 5.28% |
| sad | 400 | 31.02% | 81.75% | 44.98% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/tess_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | surprised | sad |
|---|---|---|---|---|---|---|---|
| angry (400) | **7.5%** | 2.5% | 3.2% | 6.2% | 67.8% | 0.5% | 12.2% |
| disgusted (400) | 4.0% | **16.0%** | 1.8% | 4.8% | 25.2% | 0.2% | 48.0% |
| fearful (400) | 1.8% | 3.2% | **8.2%** | 15.8% | 26.5% | 0.8% | 43.8% |
| happy (400) | 3.2% | 4.5% | 4.8% | **41.0%** | 27.0% | 0.0% | 19.5% |
| neutral (400) | 0.0% | 1.0% | 0.2% | 0.5% | **52.0%** | 0.0% | 46.2% |
| surprised (400) | 6.0% | 5.5% | 3.0% | 45.2% | 25.5% | **2.8%** | 12.0% |
| sad (400) | 0.0% | 1.0% | 0.8% | 0.0% | 16.5% | 0.0% | **81.8%** |

## Caveats

- **Training involvement for Gemini 3.8 Flash: No corpus-specific evidence**. Source: not published (Google). Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](tess_offline_audit.json)
