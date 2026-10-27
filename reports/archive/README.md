# Archived reports

Reports of datasets that are no longer evaluated: they are listed in `evaluation.exclude` in
[config/config.yaml](../../config/config.yaml), so `scripts/evaluate.py` skips them and the summary leaves them out.
Their data stays in `data/`, and their per-clip predictions are in `outputs/archive/` (not in git).

| Report | Why it is left out |
|---|---|
| [ESD, 3,500-clip sample](esd_emotion2vec_plus_large_sample3500.md) | Research-only licence that forbids sharing the data, no listener test, and in EmoBox (the model has probably seen it) |
| [THAI-SER](thai_ser_emotion2vec_plus_large.md) | Thai, which matters less for this project |

To evaluate one of them again, remove it from `evaluation.exclude` (and from `viewer.exclude` to show its clips).
