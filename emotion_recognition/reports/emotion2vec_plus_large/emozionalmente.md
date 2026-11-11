# emotion2vec+ large on Emozionalmente

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 6,902 |
| Accuracy: acted emotion | 86.95% |
| UAR: mean ground-truth-row recall | 86.95% |
| Macro F1 | 86.93% |
| Weighted F1: true-support weighted | 86.93% |
| Accuracy: all model classes | 86.93% |
| Accuracy: acted emotion or listener vote | 90.99% |
| Single listener vote vs acted emotion | 81.22% |
| Model vs single listener vote | 77.49% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/emotion2vec_plus_large/emozionalmente_predictions.csv) and [metadata JSON](../../outputs/emotion2vec_plus_large/emozionalmente_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **Emozionalmente**, Italian; 6,902 / 6,902 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "cpu", "hardware": "Apple M4 Max", "seconds": 748.2, "ms_per_clip": 108.4, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 986 | 76.17% | 81.95% |
| disgusted | 986 | 89.66% | 93.10% |
| fearful | 986 | 81.85% | 89.96% |
| happy | 986 | 89.05% | 92.39% |
| neutral | 986 | 92.29% | 94.12% |
| sad | 986 | 90.37% | 92.60% |
| surprised | 986 | 89.25% | 92.80% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 986 | 96.78% | 76.17% | 85.24% |
| disgusted | 986 | 89.84% | 89.66% | 89.75% |
| fearful | 986 | 87.06% | 81.85% | 84.37% |
| happy | 986 | 85.24% | 89.05% | 87.10% |
| neutral | 986 | 90.19% | 92.29% | 91.23% |
| sad | 986 | 82.35% | 90.37% | 86.17% |
| surprised | 986 | 80.44% | 89.25% | 84.62% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emozionalmente_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (986) | **76.2%** | 3.1% | 1.7% | 4.4% | 4.3% | 2.0% | 8.3% |
| disgusted (986) | 1.0% | **89.7%** | 1.5% | 0.9% | 1.3% | 4.7% | 0.9% |
| fearful (986) | 0.2% | 3.1% | **81.8%** | 1.6% | 0.8% | 8.3% | 4.1% |
| happy (986) | 0.4% | 0.7% | 2.0% | **89.0%** | 0.8% | 0.7% | 6.3% |
| neutral (986) | 0.3% | 0.4% | 0.1% | 1.6% | **92.3%** | 3.2% | 2.0% |
| sad (986) | 0.0% | 2.3% | 4.9% | 0.7% | 1.6% | **90.4%** | 0.1% |
| surprised (986) | 0.6% | 0.4% | 1.9% | 6.2% | 1.2% | 0.4% | **89.2%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/emozionalmente_confusion_acted_or_listener.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (937) | **81.0%** | 2.5% | 1.6% | 3.9% | 2.5% | 1.8% | 6.7% |
| disgusted (970) | 0.6% | **93.0%** | 1.4% | 0.8% | 0.8% | 2.7% | 0.6% |
| fearful (929) | 0.2% | 2.8% | **89.3%** | 1.5% | 0.3% | 3.7% | 2.2% |
| happy (987) | 0.1% | 0.7% | 1.7% | **92.4%** | 0.5% | 0.6% | 4.0% |
| neutral (1,009) | 0.3% | 0.4% | 0.1% | 1.4% | **94.3%** | 2.1% | 1.5% |
| sad (1,048) | 0.0% | 1.7% | 3.5% | 0.7% | 1.0% | **93.0%** | 0.0% |
| surprised (1,022) | 0.5% | 0.4% | 1.3% | 3.7% | 0.8% | 0.3% | **93.1%** |

## Caveats

- **Training involvement for emotion2vec+ large: Used via upstream training**. Evidence scope: upstream teacher lineage. Author documents EmoBox seed training and iterative pseudo-label lineage into large; the exact seed corpus subset and raw-audio/weight inheritance into large are not enumerated. Source: EmoBox academic data is used for seed fine-tuning, followed by teacher pseudo-labelling and data iteration to large; this project counts those corpora as used through upstream training. Large also uses 42,526 h of industrial pseudo-labelled audio. This is distinct from a claim that every EmoBox corpus directly entered large encoder self-supervised pretraining. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emozionalmente_offline_audit.json)
