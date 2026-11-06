# Archived reports

Reports that the summaries no longer show: `scripts/evaluation/evaluate.py`, its summaries, the plots and the viewer
read only `outputs/<model>/` and `reports/<model>/`, never the archive.

- **Datasets that are no longer evaluated**: they are listed in `exclude` in
  [config/evaluation.yaml](../../config/evaluation.yaml), so `scripts/evaluation/evaluate.py` skips them and the
  summaries leave them out. Their data stays in `data/`. Only ESD is left out now: THAI-SER and nEMO were restored on
  2026-11-05, and their emotion2vec+ large results moved back to `reports/emotion2vec_plus_large/` and
  `outputs/emotion2vec_plus_large/`, next to the other datasets.
- **Runs superseded by a subset**: since 2026-11-06 THAI-SER and MEAD are evaluated on a fixed subset (`subsets` in
  config/evaluation.yaml). A run on the full dataset that was finished before is kept here as `<dataset>_full`;
  `scripts/evaluation/evaluate.py` takes the predictions of the subset's clips from it instead of running the model
  again.

The archive has the same layout as the active results, one folder per model (named by its key in the model
registry, `config/models.yaml`):

- `reports/archive/<model>/<dataset>.md`: the report, with its figures in `reports/archive/<model>/figures/`
- `outputs/archive/<model>/<dataset>_predictions.csv` and `.json`: the per-clip predictions (ESD's are not in git: its licence forbids sharing any part of the data)

A sample run has `_sample<N>` after the dataset's name, a full run superseded by a subset `_full`. The archived
results are from emotion2vec+ large; the other models were never run on ESD or on the full THAI-SER.

| Dataset | Report | Accuracy | Archived | Why it is left out |
|---|---|---|---|---|
| ESD | [emotion2vec+ large, 3,500-clip sample](emotion2vec_plus_large/esd_sample3500.md) | 97.0% | 2026-10-27 | Research-only licence that forbids sharing the data, no listener test, and in EmoBox (the model has probably seen it) |
| THAI-SER, all 22,584 clips | [emotion2vec+ large, full run](emotion2vec_plus_large/thai_ser_full.md) | 63.2% | 2026-11-06 | Full THAI-SER run, superseded by the high-agreement subset on 2026-11-06 (agreement >= 0.71, 11,848 clips); the [subset result](../emotion2vec_plus_large/thai_ser.md) reuses its per-clip predictions |

To evaluate ESD again, remove it from `exclude` in `config/evaluation.yaml` (and in `config/viewer.yaml` to
show its clips).
The new results then go to `outputs/<model>/` and `reports/<model>/`, next to the other datasets.
