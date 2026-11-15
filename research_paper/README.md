# research_paper

Preparation of an academic paper based on the speech emotion recognition benchmark in `emotion_recognition/`:
open SER models vs audio LLMs on public acted and dialogue-source emotional speech.

Constraints: public data only; no company data, systems or business details; no method that could be commercially
valuable or patentable (the paper is an evaluation and analysis). Publication needs the company's approval.

Benchmark status, 2026-11-15: all ten models—the five SER models, Qwen2.5-Omni, Qwen2-Audio, Gemma 4,
GPT Realtime and Gemini 3.8 Flash—have current-protocol results on all 16 datasets (67,062 clips each),
for 160 compatible model–dataset runs. See [the benchmark summary](../emotion_recognition/reports/summary.md).

| File | Content |
|---|---|
| [docs/research_angles.md](docs/research_angles.md) | candidate research questions, novelty against the closest work, two possible papers, what must be added |
| [docs/experimental_design.md](docs/experimental_design.md) | the benchmark design of 2026-11-06 (datasets, models, scoring, run plan), with a 2026-11-15 completion update |
| [docs/venues.md](docs/venues.md) | target venues: CCF 2026 classes, 2027 deadlines, precedents, recommended plan |
| [docs/related_work.md](docs/related_work.md) | literature notes with links and numbers (audio-LLM limitations, audio in-context learning, novelty checks) |
| [analysis_report/consensus_zh.md](analysis_report/consensus_zh.md) · [analysis_report/consensus_en.md](analysis_report/consensus_en.md) | model consensus for pseudo-label candidates: coverage, accepted-label accuracy, errors and vote thresholds; Chinese and English |
| [analysis_report/figures/consensus_tradeoff_by_dataset.svg](analysis_report/figures/consensus_tradeoff_by_dataset.svg) | accuracy of agreed labels vs coverage for the four main groups (SER5, LLM3, AudioLLM5, All10), one panel per dataset, every vote threshold a point |
| [analysis_report/figures/consensus_meld_tradeoff.svg](analysis_report/figures/consensus_meld_tradeoff.svg) | MELD, where all four main groups are complete on identical clips: every vote threshold labelled |
| [analysis_report/figures/consensus_unanimous_by_dataset.svg](analysis_report/figures/consensus_unanimous_by_dataset.svg) | heat map of unanimous accuracy and coverage per dataset and main group |
| [analysis_report/figures/consensus_tradeoff_extra_groups.svg](analysis_report/figures/consensus_tradeoff_extra_groups.svg) | the same grid for the appendix groups Selected3, SER+API7 and AudioLLM4 |
| [presentation/consensus/](presentation/consensus/README.md) | slide deck on the consensus analysis (`consensus_analysis.pptx`, 14 slides + 4 hidden appendix slides, speaker notes), rebuilt with `./rebuild.sh` |
| [analysis_report/bootstrap_zh.md](analysis_report/bootstrap_zh.md) · [analysis_report/bootstrap_en.md](analysis_report/bootstrap_en.md) | cluster-bootstrap score intervals and paired model differences; Chinese and English |
| [analysis_report/cross_dataset_zh.md](analysis_report/cross_dataset_zh.md) · [analysis_report/cross_dataset_en.md](analysis_report/cross_dataset_en.md) | language, conversational source, neutral baselines, actual clip-duration distributions and adjusted duration associations; training-exposure hypotheses and limits |
| [analysis_report/intensity_zh.md](analysis_report/intensity_zh.md) · [analysis_report/intensity_en.md](analysis_report/intensity_en.md) | emotion-classification accuracy by native strength, class/duration adjustment and matched recordings; MERaLiON VAD output audit |
| [presentation/cross_dataset/](presentation/cross_dataset/README.md) | slide deck with the complete ten-model language, setting and duration analysis |
| [presentation/intensity/](presentation/intensity/README.md) | slide deck with all ten models on the six native-intensity corpora |
| [presentation/combined/](presentation/combined/README.md) | combined research presentation: consensus, cross-dataset analysis, intensity and the existing venue section |
| [main benchmark presentation](../emotion_recognition/presentation/README.md) | complete 10 × 16 benchmark, selected playable listening examples and the adaptation proposal |

Refresh both analyses from saved predictions, from this folder in the `emotion_recognition_llm` environment:

```bash
python scripts/consensus.py
python scripts/bootstrap.py
```

Both scripts write bilingual reports under `analysis_report/` and compact results under `output/`; neither runs
models or APIs. They discover available current-protocol predictions automatically and select the largest finished
saved run per model/dataset. Rerun after more model results arrive.

- `output/consensus.csv`: counts, model membership, source hashes and consensus rates.
- `output/consensus_thresholds.csv`: every integer vote threshold from strict majority to unanimity, accepted/correct/error
  counts, conditional accuracy, selective risk, coverage on the common cohort and accepted fraction of the available union.
- `output/consensus_by_class.csv`: the same diagnostics by **predicted** class, including its share of all accepted labels.
- `output/consensus_baselines.csv`: each participating model's accuracy on exactly the same within-group common cohort.
- `output/consensus_manifest.json`: refresh time, code/configuration/input hashes, selected/excluded runs, evaluation scopes,
  group membership and cohort hashes. Counts are observed point estimates; consensus confidence intervals are not supplied.
- `output/bootstrap.csv`: individual-model score intervals and all paired differences for Accuracy, UAR, macro-F1
  and weighted-F1. `kind=single` uses that model's saved samples; `kind=paired_difference` uses the pair's common
  audio paths, with `estimate = score_a - score_b`. Scores and CI bounds use the raw score scale, not percentages.
- `output/bootstrap_manifest.json`: parameters, software versions, input/code SHA256s, selected/excluded runs,
  sample-cohort hashes and grouping diagnostics.

Consensus keeps the original `SER5`, `AudioLLM5` and `Selected3` groups (`output/consensus.csv`). The latter is the
legacy group name, not an accuracy ranking; it contains emotion2vec+, Gemini and Qwen2.5-Omni. The figures, report
tables and slides compare four main groups: `SER5`, `LLM3` (the open audio LLMs Qwen2.5-Omni, Qwen2-Audio and
Gemma 4), `AudioLLM5` (LLM3 + GPT Realtime + Gemini) and `All10` (all ten models). `LLM3`, `AudioLLM5` and `All10` vote
with the available members (at least two) when results are incomplete. With the final Gemma 4 import on 2026-11-15,
all four main groups are complete on every dataset: SER5 5/5, LLM3 3/3, AudioLLM5 5/5 and All10 10/10 at unanimity.
There are no missing-member markers in the completed benchmark. Supplementary fixed groups
`AudioLLM4` (GPT, Gemini, Qwen2.5-Omni and Qwen2-Audio) and `SER_API7` (five SER models + GPT + Gemini) wait for every
member and are complete on all 16 datasets; with `Selected3` they form the appendix groups. Groups are defined by the
existing analysis or model families, without searching these evaluation scores for an optimal combination. Compare
groups only on matching cohort hashes. `python scripts/consensus.py --no-plots` skips matplotlib exports while
refreshing reports and CSVs.

After any prediction update, one command refreshes the consensus analysis, reports, paper figures and the slide deck
(saved predictions only, no model runs), in the `emotion_recognition_llm` environment:

```bash
cd presentation/consensus && ./rebuild.sh
```

The paper can use this as a public-data selective-prediction analysis. Public coverage does not establish internal
labelling effort savings: freeze a group/threshold on independent calibration labels, validate target-domain performance,
then account for auditing and correction. See the bilingual report for the detailed interpretation and literature link.

Figure axes: **Coverage (% of clips labelled) = accepted / common clips** and **Accuracy of agreed labels (%) =
correct accepted / accepted**. A group plus a minimum vote count defines one point; the five-SER line contains three
rules, not three repeated measurements. Tighter vote thresholds cannot increase coverage, but accuracy is not guaranteed
to rise. Adding members also does not guarantee higher accuracy: majority votes may change the answer, while unanimity
adds a filter that can reject correct labels as well as wrong ones. The report includes an observed CREMA-D filtering
example and a matched MELD endpoint table explaining this distinction. `scripts/consensus_plot.py` is the single source
of the figures: four paper figures (PNG 300 dpi + SVG, benchmark dataset order, one colour-blind-safe colour and marker
per group, shared axes, no titles or footnotes inside) and, through `plot_slides`, the same figures drawn at slide size
for the deck. Each trade-off point carries both metrics, which replaces the earlier separate accuracy and coverage bar
charts.

Bootstrap defaults: 10,000 resamples, seed `20261111`, 95% percentile intervals; speaker clusters for CREMA-D and
RAVDESS, dialogue clusters for MELD. Model differences share the exact same cluster draws. Streams depend on the
seed and sample cohort, so newly added models do not change existing comparisons on unchanged samples.
For example, `python scripts/bootstrap.py --resamples 10000 --seed 20261111 --datasets meld` limits the scope to MELD;
`--models emotion2vec_plus_large gemini-3.8-flash qwen2_5_omni_7b` limits the models. Optional episode-level
MELD analysis and other datasets are not implemented. These CIs describe fixed-prediction sampling uncertainty;
they do not include API rerun/training variability and are not multiplicity-adjusted significance tests.

Offline checks: `python -m unittest discover -s tests -p 'test_*.py' -v`.

Additional offline diagnostics, from this folder:

```bash
python scripts/cross_dataset_analysis.py
python scripts/intensity_analysis.py
```

Use the same `emotion_recognition_llm` environment as above. These commands reuse current-protocol predictions
and never run models or APIs. The first builds a cached
`output/audio_inventory.csv` from all 67,062 audio headers, with source grouping IDs, and writes `cross_dataset_*.csv`,
the complete `cross_dataset_analysis.json` snapshot and its manifest. The second joins explicitly documented native
strength fields into `output/intensity_inventory.csv`, then writes `intensity_*.csv`, including level/class counts
and common wrong predictions, plus `intensity_analysis.json` and its
manifest. Metadata, code and prediction hashes are recorded; cached inventories reject changed inputs. Use
`--refresh-inventory` or `--refresh-intensity` respectively after a relevant source change. `--no-plots` refreshes
tables/reports only; `--data-only` refreshes statistics only.

The current analyses export 16 PNG/SVG figure pairs: four consensus figures, six cross-dataset figures and six
intensity figures. They cover agreement/coverage, duration distributions, language and recording-source comparisons,
adjusted duration associations, native-strength changes, class-specific errors and matched recordings. Text is black and enlarged, with explicit axes,
units, sample scopes and missing-data rules. All ten models, including Gemma 4, enter cross-dataset means;
Gemma also has results on all six native-intensity corpora.

Duration associations use original file duration including silence, not voiced time or a uniform model input window.
The primary fit balances datasets and original gold classes, controls dataset-by-gold intercepts and reports
dataset-cluster intervals. Native strength is available in six corpora (19,131 known, non-neutral clips); scales
are kept separate, unspecified/neutral samples are excluded, and a duration covariate is added in a sensitivity fit.
MCAE listener ratings remain continuous. Fewer than 20 source clusters receive no strength CI; DailyTalk's sparse
high-strength samples and MEAD's three complete take keys cannot support general conclusions. MERaLiON's native
VAD dimensions were discarded by the existing adapter, so these outputs test classification sensitivity to strength,
not intensity-prediction accuracy. Observed associations and high benchmark scores do not establish causality or
training leakage; neither report certifies unseen test data or internal pseudo-label reliability.

The experiments themselves reuse `emotion_recognition/` (models, data loaders, evaluation scripts, outputs).
