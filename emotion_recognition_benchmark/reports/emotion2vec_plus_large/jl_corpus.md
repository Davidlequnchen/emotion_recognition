# emotion2vec+ large on JL-Corpus

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 1,200 |
| Accuracy: acted emotion | 92.50% |
| UAR: mean ground-truth-row recall | 92.50% |
| Macro F1 | 92.27% |
| Weighted F1: true-support weighted | 92.53% |
| Accuracy: all model classes | 92.50% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/emotion2vec_plus_large/jl_corpus_predictions.csv) and [metadata JSON](../../outputs/emotion2vec_plus_large/jl_corpus_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-10-28`.
- Dataset scope recorded in JSON: **JL-Corpus**, English (New Zealand); 1,200 / 1,200 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M4 Max", "seconds": 108.7, "ms_per_clip": 90.6, "on_cpu": 0}`.
- Composite labels follow the existing accepted-label rule: excited accepts happy or surprised. F1 resolves a correct composite to the predicted class, otherwise to its first accepted class. Weighted F1 uses the resulting resolved-truth support, which can depend on the predictions.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 240 | 99.17% | — |
| happy | 240 | 92.08% | — |
| neutral | 240 | 100.00% | — |
| sad | 240 | 87.50% | — |
| excited (happy or surprised) | 240 | 83.75% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 240 | 88.15% | 99.17% | 93.33% |
| happy | 480 | 100.00% | 87.92% | 93.57% |
| neutral | 240 | 83.92% | 100.00% | 91.25% |
| sad | 240 | 94.59% | 87.50% | 90.91% |
| surprised | 0 | — | — | — |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/jl_corpus_confusion.png)

| Ground truth \ Predicted | angry | happy | neutral | sad | surprised |
|---|---|---|---|---|---|
| angry (240) | **99.2%** | 0.0% | 0.0% | 0.8% | 0.0% |
| happy (240) | 0.0% | **92.1%** | 6.7% | 1.2% | 0.0% |
| neutral (240) | 0.0% | 0.0% | **100.0%** | 0.0% | 0.0% |
| sad (240) | 0.0% | 0.0% | 12.5% | **87.5%** | 0.0% |
| excited (happy or surprised) (240) | 13.3% | **83.8%** | 0.0% | 2.9% | **0.0%** |

## Caveats

- **Training involvement for emotion2vec+ large: Used via upstream training**. Evidence scope: upstream teacher lineage. Author documents EmoBox seed training and iterative pseudo-label lineage into large; the exact seed corpus subset and raw-audio/weight inheritance into large are not enumerated. Source: EmoBox academic data is used for seed fine-tuning, followed by teacher pseudo-labelling and data iteration to large; this project counts those corpora as used through upstream training. Large also uses 42,526 h of industrial pseudo-labelled audio. This is distinct from a claim that every EmoBox corpus directly entered large encoder self-supervised pretraining. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](jl_corpus_offline_audit.json)
