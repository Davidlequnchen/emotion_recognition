# emotion2vec+ large on THAI-SER (high-agreement subset)

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 11,848 |
| Accuracy: acted emotion | 70.59% |
| UAR: mean ground-truth-row recall | 68.79% |
| Macro F1 | 69.66% |
| Weighted F1: true-support weighted | 70.21% |
| Accuracy: all model classes | 64.90% |
| Accuracy: acted emotion or listener vote | 75.04% |
| Single listener vote vs acted emotion | 77.23% |
| Model vs single listener vote | 65.92% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/emotion2vec_plus_large/thai_ser_predictions.csv) and [metadata JSON](../../outputs/emotion2vec_plus_large/thai_ser_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **THAI-SER (high-agreement subset)**, Thai; 11,848 / 11,848 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: neutral, angry, happy, sad.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| neutral | 2,999 | 82.69% | 82.96% |
| angry | 2,883 | 65.24% | 68.23% |
| happy | 3,665 | 78.85% | 86.08% |
| sad | 2,301 | 48.37% | 55.67% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| neutral | 2,999 | 64.87% | 82.69% | 72.71% |
| angry | 2,883 | 84.77% | 65.24% | 73.74% |
| happy | 3,665 | 64.73% | 78.85% | 71.09% |
| sad | 2,301 | 83.00% | 48.37% | 61.12% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/thai_ser_confusion.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (2,999) | **82.7%** | 0.7% | 14.7% | 1.9% |
| angry (2,883) | 11.4% | **65.2%** | 21.3% | 2.0% |
| happy (3,665) | 10.6% | 7.5% | **78.9%** | 3.0% |
| sad (2,301) | 27.2% | 1.8% | 22.6% | **48.4%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/thai_ser_confusion_acted_or_listener.png)

| Ground truth \ Predicted | neutral | angry | happy | sad |
|---|---|---|---|---|
| neutral (3,477) | **85.3%** | 0.6% | 12.5% | 1.7% |
| angry (2,810) | 8.9% | **67.4%** | 21.6% | 2.1% |
| happy (3,419) | 4.2% | 7.7% | **85.1%** | 3.0% |
| sad (2,142) | 21.7% | 1.9% | 24.0% | **52.4%** |

## Caveats

- **Training involvement for emotion2vec+ large: No corpus-specific evidence**. Source: EmoBox academic data is used for seed fine-tuning, followed by teacher pseudo-labelling and data iteration to large; this project counts those corpora as used through upstream training. Large also uses 42,526 h of industrial pseudo-labelled audio. This is distinct from a claim that every EmoBox corpus directly entered large encoder self-supervised pretraining. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](thai_ser_offline_audit.json)
