# Gemini 3.8 Flash on MultiDialog (gold_both)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 7,672 |
| Accuracy: acted emotion | 56.63% |
| UAR: mean ground-truth-row recall | 29.86% |
| Macro F1 | 27.30% |
| Accuracy: all model classes | 56.63% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/gemini-3.8-flash/multidialog_predictions.csv) and [metadata JSON](../../outputs/gemini-3.8-flash/multidialog_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-10`.
- Dataset scope recorded in JSON: **MultiDialog (gold_both)**, English; 7,672 / 7,672 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `gemini-3.8-flash`, prompt hash `dc84c979c561`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M3 Pro", "seconds": 1407.4, "ms_per_clip": 183.4, "on_cpu": 0}`.
- The original notes report one 24 kHz audio clip per request.
- The original notes report up to 30 concurrent clip evaluations.
- The recorded ms_per_clip is total wall-clock time divided by the number of clips, not individual request latency; it is not directly comparable with local GPU inference timing.
- The original notes describe schema-constrained answers validated as literal JSON strings.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 35 | 25.71% | — |
| disgusted | 67 | 19.40% | — |
| fearful | 26 | 11.54% | — |
| happy | 1,064 | 46.15% | — |
| neutral | 5,284 | 68.81% | — |
| sad | 83 | 21.69% | — |
| surprised | 1,113 | 15.72% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 35 | 15.25% | 25.71% | 19.15% |
| disgusted | 67 | 14.13% | 19.40% | 16.35% |
| fearful | 26 | 17.65% | 11.54% | 13.95% |
| happy | 1,064 | 24.08% | 46.15% | 31.65% |
| neutral | 5,284 | 75.09% | 68.81% | 71.82% |
| sad | 83 | 12.86% | 21.69% | 16.14% |
| surprised | 1,113 | 36.92% | 15.72% | 22.05% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/multidialog_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised | other answer |
|---|---|---|---|---|---|---|---|---|
| angry (35) | **25.7%** | 14.3% | 0.0% | 11.4% | 40.0% | 5.7% | 2.9% | 0.0% |
| disgusted (67) | 10.4% | **19.4%** | 1.5% | 6.0% | 55.2% | 6.0% | 1.5% | 0.0% |
| fearful (26) | 0.0% | 7.7% | **11.5%** | 19.2% | 46.2% | 15.4% | 0.0% | 0.0% |
| happy (1,064) | 0.4% | 0.6% | 0.3% | **46.1%** | 45.9% | 2.1% | 4.6% | 0.1% |
| neutral (5,284) | 0.6% | 0.9% | 0.2% | 23.2% | **68.8%** | 1.5% | 4.7% | 0.1% |
| sad (83) | 1.2% | 1.2% | 0.0% | 4.8% | 71.1% | **21.7%** | 0.0% | 0.0% |
| surprised (1,113) | 0.4% | 1.5% | 0.0% | 27.5% | 53.5% | 1.0% | **15.7%** | 0.3% |

## Caveats

- **Training involvement for Gemini 3.8 Flash: No corpus-specific evidence**. Source: not published (Google). Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](multidialog_offline_audit.json)
