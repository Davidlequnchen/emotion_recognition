# research_paper

Preparation of an academic paper based on the speech emotion recognition benchmark in `emotion_recognition/`:
open SER models vs audio LLMs on public, lexically neutral emotional speech.

Constraints: public data only; no company data, systems or business details; no method that could be commercially
valuable or patentable (the paper is an evaluation and analysis). Publication needs the company's approval.

Benchmark status, 2026-11-14: the five SER models, Qwen2.5-Omni, Qwen2-Audio, GPT Realtime and Gemini 3.8 Flash
have current-protocol results on all 16 datasets (nine of ten models; 67,062 clips each). Gemma 4 has MELD only
(1/16), for 145 compatible model–dataset runs in total. See [the benchmark summary](../emotion_recognition/reports/summary.md).

| File | Content |
|---|---|
| [docs/research_angles.md](docs/research_angles.md) | candidate research questions, novelty against the closest work, two possible papers, what must be added |
| [docs/venues.md](docs/venues.md) | target venues: CCF 2026 classes, 2027 deadlines, precedents, recommended plan |
| [docs/related_work.md](docs/related_work.md) | literature notes with links and numbers (audio-LLM limitations, audio in-context learning, novelty checks) |
| [analysis_report/consensus_zh.md](analysis_report/consensus_zh.md) · [analysis_report/consensus_en.md](analysis_report/consensus_en.md) | model consensus for pseudo-label candidates: coverage, accepted-label accuracy, errors and vote thresholds; Chinese and English |
| [analysis_report/figures/consensus_accuracy_coverage.svg](analysis_report/figures/consensus_accuracy_coverage.svg) | five-SER vote thresholds, with every 3/5, 4/5 and 5/5 point explicitly labelled |
| [analysis_report/figures/consensus_meld_accuracy_coverage.svg](analysis_report/figures/consensus_meld_accuracy_coverage.svg) | detailed MELD vote thresholds, one panel per model group |
| [analysis_report/figures/consensus_unanimous_accuracy_by_dataset.svg](analysis_report/figures/consensus_unanimous_accuracy_by_dataset.svg) · [coverage](analysis_report/figures/consensus_unanimous_coverage_by_dataset.svg) | benchmark-style bars comparing complete fixed groups under unanimity |
| [analysis_report/figures/consensus_ser5_accuracy_by_votes.svg](analysis_report/figures/consensus_ser5_accuracy_by_votes.svg) · [coverage](analysis_report/figures/consensus_ser5_coverage_by_votes.svg) | five-SER bar comparisons of the three minimum vote rules |
| [analysis_report/figures/consensus_audiollm4_accuracy_by_votes.svg](analysis_report/figures/consensus_audiollm4_accuracy_by_votes.svg) · [coverage](analysis_report/figures/consensus_audiollm4_coverage_by_votes.svg) | fixed four-audio-LLM bar comparisons of the 3/4 and 4/4 vote rules |
| [analysis_report/bootstrap_zh.md](analysis_report/bootstrap_zh.md) · [analysis_report/bootstrap_en.md](analysis_report/bootstrap_en.md) | cluster-bootstrap score intervals and paired model differences; Chinese and English |
| [analysis_report/cross_dataset_zh.md](analysis_report/cross_dataset_zh.md) · [analysis_report/cross_dataset_en.md](analysis_report/cross_dataset_en.md) | language, conversational source, neutral baselines, actual clip-duration distributions and adjusted duration associations; training-exposure hypotheses and limits |
| [analysis_report/intensity_zh.md](analysis_report/intensity_zh.md) · [analysis_report/intensity_en.md](analysis_report/intensity_en.md) | emotion-classification accuracy by native strength, class/duration adjustment and matched recordings; MERaLiON VAD output audit |

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

Consensus keeps the original `SER5`, `AudioLLM5` and `Selected3` groups. The latter is the legacy group name, not an
accuracy ranking; it contains emotion2vec+, Gemini and Qwen2.5-Omni. `AudioLLM5` can use available members and therefore
has varying membership across datasets, always reported explicitly. Supplementary fixed groups `AudioLLM4` (GPT,
Gemini, Qwen2.5-Omni and Qwen2-Audio), `SER_API7` (five SER models + GPT + Gemini) and `All10` (all ten models)
wait for every member; `AudioLLM4` is complete on all 16 datasets. Groups are defined by the existing analysis
or model families, without searching these evaluation scores for an optimal combination. Compare groups only on matching
cohort hashes. `python scripts/consensus.py --no-plots` skips matplotlib exports while refreshing reports and CSVs.

The paper can use this as a public-data selective-prediction analysis. Public coverage does not establish internal
labelling effort savings: freeze a group/threshold on independent calibration labels, validate target-domain performance,
then account for auditing and correction. See the bilingual report for the detailed interpretation and literature link.

Figure axes use direct operational wording: **Samples auto-labelled (%) = accepted / common clips** and
**Correct auto-labels (%) = correct accepted / accepted**. A group plus a minimum vote count defines one point;
the five-SER line contains three rules, not three repeated measurements. Tighter vote thresholds cannot increase coverage,
but accuracy is not guaranteed to rise. Adding members also does not guarantee higher accuracy: majority votes may change
the answer, while unanimity adds a filter that can reject correct labels as well as wrong ones. The report includes an
observed CREMA-D filtering example and a matched MELD endpoint table explaining this distinction. All chart text is black
and enlarged; bars follow the benchmark's dataset order and whole-percentage value labels. Eight figure pairs are exported
as SVG and PNG (16 files in total), including the fixed `AudioLLM4` 3/4 versus 4/4 comparisons.

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

The reports include eight new PNG/SVG figure pairs: duration distributions, language/conversation bars, adjusted
duration effects, dataset-specific duration slopes, accuracy by duration, native-strength accuracy, strength
associations and matched higher-versus-lower strength recordings. Text is black and enlarged, with explicit axes,
units, sample scopes and missing-data rules. Gemma remains MELD-only and is excluded from cross-dataset means.

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
