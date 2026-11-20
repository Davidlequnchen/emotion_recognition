# emotion2vec+ large on MESS

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,800 |
| Accuracy: acted emotion | 40.78% |
| UAR: mean ground-truth-row recall | 40.78% |
| Macro F1 | 35.22% |
| Weighted F1: true-support weighted | 35.22% |
| Accuracy: all model classes | 38.67% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/emotion2vec_plus_large/mess_predictions.csv) and [metadata JSON](../../outputs/emotion2vec_plus_large/mess_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-10-28`.
- Dataset scope recorded in JSON: **MESS**, English; 1,800 / 1,800 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M4 Max", "seconds": 1254.8, "ms_per_clip": 697.1, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 450 | 10.44% | — |
| happy | 450 | 49.11% | — |
| neutral | 450 | 96.00% | — |
| sad | 450 | 7.56% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 450 | 100.00% | 10.44% | 18.91% |
| happy | 450 | 87.35% | 49.11% | 62.87% |
| neutral | 450 | 29.61% | 96.00% | 45.26% |
| sad | 450 | 82.93% | 7.56% | 13.85% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/mess_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad |
|---|---|---|---|---|
| angry (450) | **10.4%** | 4.4% | 85.1% | 0.0% |
| happy (450) | 0.0% | **49.1%** | 50.9% | 0.0% |
| neutral (450) | 0.0% | 2.4% | **96.0%** | 1.6% |
| sad (450) | 0.0% | 0.2% | 92.2% | **7.6%** |

## Caveats

- **Training involvement for emotion2vec+ large: No corpus-specific evidence**. Source: EmoBox academic data is used for seed fine-tuning, followed by teacher pseudo-labelling and data iteration to large; this project counts those corpora as used through upstream training. Large also uses 42,526 h of industrial pseudo-labelled audio. This is distinct from a claim that every EmoBox corpus directly entered large encoder self-supervised pretraining. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](mess_offline_audit.json)
