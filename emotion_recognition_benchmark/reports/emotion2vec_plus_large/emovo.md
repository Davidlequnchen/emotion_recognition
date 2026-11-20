# emotion2vec+ large on EMOVO

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 588 |
| Accuracy: acted emotion | 87.59% |
| UAR: mean ground-truth-row recall | 87.59% |
| Macro F1 | 87.62% |
| Weighted F1: true-support weighted | 87.62% |
| Accuracy: all model classes | 87.59% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/emotion2vec_plus_large/emovo_predictions.csv) and [metadata JSON](../../outputs/emotion2vec_plus_large/emovo_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **EMOVO**, Italian; 588 / 588 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M4 Max", "seconds": 63.0, "ms_per_clip": 107.2, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 84 | 96.43% | — |
| disgusted | 84 | 76.19% | — |
| fearful | 84 | 94.05% | — |
| happy | 84 | 86.90% | — |
| neutral | 84 | 96.43% | — |
| sad | 84 | 78.57% | — |
| surprised | 84 | 84.52% | — |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 84 | 91.01% | 96.43% | 93.64% |
| disgusted | 84 | 98.46% | 76.19% | 85.91% |
| fearful | 84 | 77.45% | 94.05% | 84.95% |
| happy | 84 | 82.95% | 86.90% | 84.88% |
| neutral | 84 | 94.19% | 96.43% | 95.29% |
| sad | 84 | 100.00% | 78.57% | 88.00% |
| surprised | 84 | 77.17% | 84.52% | 80.68% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emovo_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (84) | **96.4%** | 0.0% | 1.2% | 1.2% | 0.0% | 0.0% | 1.2% |
| disgusted (84) | 1.2% | **76.2%** | 10.7% | 3.6% | 1.2% | 0.0% | 7.1% |
| fearful (84) | 1.2% | 0.0% | **94.0%** | 1.2% | 0.0% | 0.0% | 3.6% |
| happy (84) | 1.2% | 0.0% | 0.0% | **86.9%** | 0.0% | 0.0% | 11.9% |
| neutral (84) | 0.0% | 0.0% | 0.0% | 2.4% | **96.4%** | 0.0% | 1.2% |
| sad (84) | 0.0% | 1.2% | 13.1% | 2.4% | 4.8% | **78.6%** | 0.0% |
| surprised (84) | 6.0% | 0.0% | 2.4% | 7.1% | 0.0% | 0.0% | **84.5%** |

## Caveats

- **Training involvement for emotion2vec+ large: Used via upstream training**. Evidence scope: upstream teacher lineage. Author documents EmoBox seed training and iterative pseudo-label lineage into large; the exact seed corpus subset and raw-audio/weight inheritance into large are not enumerated. Source: EmoBox academic data is used for seed fine-tuning, followed by teacher pseudo-labelling and data iteration to large; this project counts those corpora as used through upstream training. Large also uses 42,526 h of industrial pseudo-labelled audio. This is distinct from a claim that every EmoBox corpus directly entered large encoder self-supervised pretraining. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emovo_offline_audit.json)
