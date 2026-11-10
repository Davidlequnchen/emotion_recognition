# Gemini 3.8 Flash on SAVEE

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 480 |
| Accuracy: acted emotion | 33.12% |
| UAR: mean ground-truth-row recall | 26.07% |
| Macro F1 | 24.99% |
| Accuracy: all model classes | 33.12% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/gemini-3.8-flash/savee_predictions.csv) and [metadata JSON](../../outputs/gemini-3.8-flash/savee_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-10`.
- Dataset scope recorded in JSON: **SAVEE**, English (British); 480 / 480 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `gemini-3.8-flash`, prompt hash `dc84c979c561`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M3 Pro", "seconds": 79.8, "ms_per_clip": 166.3, "on_cpu": 0}`.
- The original notes report one 24 kHz audio clip per request.
- The original notes report up to 30 concurrent clip evaluations.
- The recorded ms_per_clip is total wall-clock time divided by the number of clips, not individual request latency; it is not directly comparable with local GPU inference timing.
- The original notes describe schema-constrained answers validated as literal JSON strings.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 60 | 23.33% | — |
| disgusted | 60 | 16.67% | — |
| fearful | 60 | 10.00% | — |
| happy | 60 | 18.33% | — |
| neutral | 120 | 82.50% | — |
| sad | 60 | 21.67% | — |
| surprised | 60 | 10.00% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 60 | 37.84% | 23.33% | 28.87% |
| disgusted | 60 | 21.28% | 16.67% | 18.69% |
| fearful | 60 | 42.86% | 10.00% | 16.22% |
| happy | 60 | 22.45% | 18.33% | 20.18% |
| neutral | 120 | 36.53% | 82.50% | 50.64% |
| sad | 60 | 28.89% | 21.67% | 24.76% |
| surprised | 60 | 35.29% | 10.00% | 15.58% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/savee_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (60) | **23.3%** | 8.3% | 0.0% | 18.3% | 45.0% | 3.3% | 1.7% |
| disgusted (60) | 5.0% | **16.7%** | 0.0% | 6.7% | 60.0% | 6.7% | 5.0% |
| fearful (60) | 6.7% | 8.3% | **10.0%** | 15.0% | 38.3% | 15.0% | 6.7% |
| happy (60) | 11.7% | 11.7% | 0.0% | **18.3%** | 41.7% | 11.7% | 5.0% |
| neutral (120) | 3.3% | 5.8% | 1.7% | 2.5% | **82.5%** | 4.2% | 0.0% |
| sad (60) | 1.7% | 11.7% | 1.7% | 0.0% | 63.3% | **21.7%** | 0.0% |
| surprised (60) | 6.7% | 10.0% | 8.3% | 18.3% | 38.3% | 8.3% | **10.0%** |

## Caveats

- **Training involvement for Gemini 3.8 Flash: No corpus-specific evidence**. Source: not published (Google). Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](savee_offline_audit.json)
