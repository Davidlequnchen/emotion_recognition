# Archived reports

Reports of datasets that are no longer evaluated: they are listed in `exclude` in
[config/evaluation.yaml](../../config/evaluation.yaml), so `scripts/evaluation/evaluate.py` skips them and the
summaries leave them out. Their data stays in `data/`.

The archive has the same layout as the active results, one folder per model (named by its key in the model
registry, `config/models.yaml`):

- `reports/archive/<model>/<dataset>.md`: the report, with its figures in `reports/archive/<model>/figures/`
- `outputs/archive/<model>/<dataset>_predictions.csv` and `.json`: the per-clip predictions (in git, except ESD's: its licence forbids sharing any part of the data)

A sample run has `_sample<N>` after the dataset's name. Every archived result is from emotion2vec+ large; the other
models were never run on these datasets.

| Dataset | Report | Accuracy | Archived | Why it is left out |
|---|---|---|---|---|
| ESD | [emotion2vec+ large, 3,500-clip sample](emotion2vec_plus_large/esd_sample3500.md) | 97.0% | 2026-10-27 | Research-only licence that forbids sharing the data, no listener test, and in EmoBox (the model has probably seen it) |
| THAI-SER | [emotion2vec+ large](emotion2vec_plus_large/thai_ser.md) | 63.2% | 2026-10-27 | Thai, which matters less for this project |
| nEMO | [emotion2vec+ large](emotion2vec_plus_large/nemo.md) | 52.0% | 2026-10-28 | Polish, which matters less for this project |

To evaluate one of them again, remove it from `exclude` in `config/evaluation.yaml` (and in `config/viewer.yaml` to
show its clips).
The new results then go to `outputs/<model>/` and `reports/<model>/`, next to the other datasets.
