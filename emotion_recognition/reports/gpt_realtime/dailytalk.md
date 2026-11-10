# GPT Realtime on DailyTalk (validation)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,197 |
| Accuracy: acted emotion | 74.52% |
| UAR: mean ground-truth-row recall | 48.43% |
| Macro F1 | 35.08% |
| Accuracy: all model classes | 74.52% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/gpt_realtime/dailytalk_predictions.csv) and [metadata JSON](../../outputs/gpt_realtime/dailytalk_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-10`.
- Dataset scope recorded in JSON: **DailyTalk (validation)**, English; 1,197 / 1,197 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Saved candidate/prompt/model protocol matches current configuration: scope `dataset`, model `gpt-realtime`, prompt hash `dc84c979c561`.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M3 Pro", "seconds": 150.6, "ms_per_clip": 125.8, "on_cpu": 0}`.
- The original notes report up to 30 concurrent clip evaluations.
- The recorded ms_per_clip is total wall-clock time divided by the number of clips, not individual request latency; it is not directly comparable with local GPU inference timing.
- Saved p_* columns encode the returned answer as one-hot values; they are not calibrated API class probabilities, and no confidence plot is inferred from them.
- A raw responses journal is present; this refresh validates CSV/JSON only.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 4 | 0.00% | — |
| disgusted | 3 | 33.33% | — |
| fearful | 1 | 100.00% | — |
| happy | 217 | 59.91% | — |
| neutral | 931 | 80.24% | — |
| sad | 22 | 18.18% | — |
| surprised | 19 | 47.37% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 4 | 0.00% | 0.00% | 0.00% |
| disgusted | 3 | 20.00% | 33.33% | 25.00% |
| fearful | 1 | 11.11% | 100.00% | 20.00% |
| happy | 217 | 48.87% | 59.91% | 53.83% |
| neutral | 931 | 88.09% | 80.24% | 83.98% |
| sad | 22 | 13.33% | 18.18% | 15.38% |
| surprised | 19 | 47.37% | 47.37% | 47.37% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/dailytalk_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised | other answer |
|---|---|---|---|---|---|---|---|---|
| angry (4) | **0.0%** | 0.0% | 0.0% | 0.0% | 25.0% | 0.0% | 75.0% | 0.0% |
| disgusted (3) | 0.0% | **33.3%** | 0.0% | 0.0% | 33.3% | 0.0% | 33.3% | 0.0% |
| fearful (1) | 0.0% | 0.0% | **100.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| happy (217) | 0.0% | 0.0% | 0.0% | **59.9%** | 38.2% | 0.5% | 1.4% | 0.0% |
| neutral (931) | 1.4% | 0.3% | 0.6% | 14.0% | **80.2%** | 2.6% | 0.3% | 0.5% |
| sad (22) | 4.5% | 4.5% | 9.1% | 0.0% | 63.6% | **18.2%** | 0.0% | 0.0% |
| surprised (19) | 0.0% | 0.0% | 0.0% | 31.6% | 10.5% | 5.3% | **47.4%** | 5.3% |

## Caveats

- **Training involvement for GPT Realtime: No corpus-specific evidence**. Source: not published (OpenAI). Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](dailytalk_offline_audit.json)
