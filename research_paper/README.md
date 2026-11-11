# research_paper

Preparation of an academic paper based on the speech emotion recognition benchmark in `emotion_recognition/`:
open SER models vs audio LLMs on public, lexically neutral emotional speech.

Constraints: public data only; no company data, systems or business details; no method that could be commercially
valuable or patentable (the paper is an evaluation and analysis). Publication needs the company's approval.

| File | Content |
|---|---|
| [docs/research_angles.md](docs/research_angles.md) | candidate research questions, novelty against the closest work, two possible papers, what must be added |
| [docs/venues.md](docs/venues.md) | target venues: CCF 2026 classes, 2027 deadlines, precedents, recommended plan |
| [docs/related_work.md](docs/related_work.md) | literature notes with links and numbers (audio-LLM limitations, audio in-context learning, novelty checks) |
| [analysis_report/consensus_zh.md](analysis_report/consensus_zh.md) · [analysis_report/consensus_en.md](analysis_report/consensus_en.md) | concise model-consensus analysis for pseudo-label candidates; Chinese and English |
| [analysis_report/bootstrap_zh.md](analysis_report/bootstrap_zh.md) · [analysis_report/bootstrap_en.md](analysis_report/bootstrap_en.md) | cluster-bootstrap score intervals and paired model differences; Chinese and English |

Refresh both analyses from saved predictions, from this folder in the `emotion_recognition_llm` environment:

```bash
python scripts/consensus.py
python scripts/bootstrap.py
```

Both scripts write bilingual reports under `analysis_report/` and compact results under `output/`; neither runs
models or APIs. They discover available current-protocol predictions automatically and select the largest finished
saved run per model/dataset. Rerun after more model results arrive.

- `output/consensus.csv`: counts, model membership, source hashes and consensus rates.
- `output/bootstrap.csv`: individual-model score intervals and all paired differences for Accuracy, UAR, macro-F1
  and weighted-F1. `kind=single` uses that model's saved samples; `kind=paired_difference` uses the pair's common
  audio paths, with `estimate = score_a - score_b`. Scores and CI bounds use the raw score scale, not percentages.
- `output/bootstrap_manifest.json`: parameters, software versions, input/code SHA256s, selected/excluded runs,
  sample-cohort hashes and grouping diagnostics.

Bootstrap defaults: 10,000 resamples, seed `20261111`, 95% percentile intervals; speaker clusters for CREMA-D and
RAVDESS, dialogue clusters for MELD. Model differences share the exact same cluster draws. Streams depend on the
seed and sample cohort, so newly added models do not change existing comparisons on unchanged samples.
For example, `python scripts/bootstrap.py --resamples 10000 --seed 20261111 --datasets meld` limits the scope to MELD;
`--models emotion2vec_plus_large gemini-3.8-flash qwen2_5_omni_7b` limits the models. Optional episode-level
MELD analysis and other datasets are not implemented. These CIs describe fixed-prediction sampling uncertainty;
they do not include API rerun/training variability and are not multiplicity-adjusted significance tests.

Offline checks: `python -m unittest discover -s tests -p 'test_*.py' -v`.

The experiments themselves reuse `emotion_recognition/` (models, data loaders, evaluation scripts, outputs).
