# MERaLiON-SER-v1 on Emozionalmente

Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. This refresh performs no model inference or API requests.

## Summary

| Metric | Recomputed value |
|---|---:|
| Clips | 6,902 |
| Accuracy: acted emotion | 47.44% |
| UAR: mean ground-truth-row recall | 47.44% |
| Macro F1 | 41.75% |
| Weighted F1: true-support weighted | 41.75% |
| Accuracy: all model classes | 47.44% |
| Accuracy: acted emotion or listener vote | 58.04% |
| Single listener vote vs acted emotion | 81.22% |
| Model vs single listener vote | 52.58% |

## Imported run and verification

- Source: [prediction CSV](../../outputs/meralion_ser_v1/emozionalmente_predictions.csv) and [metadata JSON](../../outputs/meralion_ser_v1/emozionalmente_predictions.json); both remain byte-for-byte unchanged.
- Original execution date recorded in JSON: `2026-11-06`.
- Dataset scope recorded in JSON: **Emozionalmente**, Italian; 6,902 / 6,902 clips.
- Ground-truth column: `acted_emotion`. Candidate classes: angry, disgusted, fearful, happy, neutral, sad, surprised.
- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. Metric agreement tolerance is 0.000001 (six-decimal metadata).
- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline audit and summaries; the source JSON is preserved.
- Original execution/client runtime metadata (reported by the imported run, not measured by this refresh): `{"device": "mps", "hardware": "Apple M4 Max", "seconds": 1504.7, "ms_per_clip": 218.0, "on_cpu": 0}`.

## Results per emotion

| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |
|---|---:|---:|---:|
| angry | 986 | 54.16% | 61.97% |
| disgusted | 986 | 4.26% | 24.44% |
| fearful | 986 | 11.97% | 37.63% |
| happy | 986 | 80.73% | 84.58% |
| neutral | 986 | 73.73% | 76.67% |
| sad | 986 | 87.53% | 90.77% |
| surprised | 986 | 19.68% | 30.22% |

Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses the resolved-truth clip counts below as weights.

| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| angry | 986 | 73.35% | 54.16% | 62.31% |
| disgusted | 986 | 73.68% | 4.26% | 8.05% |
| fearful | 986 | 65.56% | 11.97% | 20.24% |
| happy | 986 | 45.56% | 80.73% | 58.25% |
| neutral | 986 | 57.47% | 73.73% | 64.59% |
| sad | 986 | 32.03% | 87.53% | 46.90% |
| surprised | 986 | 83.98% | 19.68% | 31.88% |

## Confusion matrix: acted emotion

![Acted-emotion confusion matrix](figures/emozionalmente_confusion.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (986) | **54.2%** | 0.7% | 1.1% | 16.8% | 16.5% | 9.6% | 1.0% |
| disgusted (986) | 4.3% | **4.3%** | 0.1% | 22.4% | 9.9% | 58.7% | 0.3% |
| fearful (986) | 1.6% | 0.1% | **12.0%** | 6.7% | 4.5% | 74.4% | 0.7% |
| happy (986) | 4.6% | 0.0% | 0.1% | **80.7%** | 2.6% | 10.3% | 1.6% |
| neutral (986) | 1.2% | 0.2% | 0.0% | 7.5% | **73.7%** | 17.3% | 0.0% |
| sad (986) | 0.7% | 0.0% | 0.3% | 2.2% | 9.1% | **87.5%** | 0.1% |
| surprised (986) | 7.3% | 0.5% | 4.7% | 40.8% | 11.9% | 15.2% | **19.7%** |

## Confusion matrix: acted emotion or listener vote

A prediction accepted only through the listener vote is placed in that predicted emotion's row; other clips keep their original ground-truth row. Row supports can therefore change.

![Acted-or-listener confusion matrix](figures/emozionalmente_confusion_acted_or_listener.png)

| Ground truth \ Predicted | angry | disgusted | fearful | happy | neutral | sad | surprised |
|---|---|---|---|---|---|---|---|
| angry (941) | **60.1%** | 0.7% | 1.2% | 16.8% | 11.1% | 9.4% | 0.7% |
| disgusted (788) | 4.6% | **5.5%** | 0.1% | 26.0% | 8.1% | 55.5% | 0.3% |
| fearful (740) | 1.5% | 0.0% | **16.9%** | 7.8% | 3.8% | 69.9% | 0.1% |
| happy (1,047) | 3.2% | 0.0% | 0.1% | **85.5%** | 1.3% | 9.1% | 0.9% |
| neutral (1,128) | 1.1% | 0.2% | 0.0% | 6.2% | **79.6%** | 12.9% | 0.0% |
| sad (1,359) | 0.5% | 0.0% | 0.0% | 1.5% | 4.6% | **93.3%** | 0.1% |
| surprised (899) | 7.0% | 0.6% | 4.7% | 37.8% | 10.6% | 15.9% | **23.5%** |

## Caveats

- **Training involvement for MERaLiON-SER-v1: Used via upstream training**. Evidence scope: upstream teacher lineage. SER report documents emotion2vec seed/base teachers for SG-ECMT labels; their EmoBox corpus subset is unresolved. This counts as upstream training use; it does not assert direct reuse of those raw clips by MERaLiON. Source: SER training on CREMA-D, ESD, MELD, M3ED and Singaporean speech; retained Whisper-medium pretraining sources are incompletely disclosed; upstream seed/base teacher use is counted as seen. Direct use and upstream teacher use both count as training involvement. No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.
- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.

[Offline validation audit](emozionalmente_offline_audit.json)
