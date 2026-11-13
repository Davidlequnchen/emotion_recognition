# Can model agreement help select pseudo-labels?

Updated: 2026-11-13 16:31 SGT.

This analysis reuses 136 compatible saved prediction runs across 16 public datasets; no model was rerun.
Model completion (datasets with a full run in the current evaluation scope / 16): Gemini 3.8 Flash 16/16; GPT Realtime 16/16; Each of the five SER models 16/16; Qwen2.5-Omni 16/16; Qwen2-Audio 7/16; Gemma 4 1/16.
Datasets with every target member available: five-SER group 16; audio-LLM group 1; selected-model group 16. Group completeness requires results from every target member; available and missing members are listed below.
On the same MELD samples, unanimous accepted-label accuracy / coverage is: Five SER models 54.12% / 10.70%; Audio LLMs (including APIs) 72.87% / 12.58%; Selected models 70.38% / 27.19%.
The selected models retain more samples, while their accepted-label accuracy is not the highest of the three groups.
Public-data results do not establish accuracy on company calls; calibrate the group and threshold on a small manually labelled call sample first.

Coverage is the accepted share of common samples; accepted-label accuracy is the correct share of accepted labels. Majority means strictly more than half of the current model count. Invalid answers abstain without reducing that denominator; top-vote ties are rejected. With two models, majority equals unanimity; with five models, majority requires at least three votes. Accuracy is — when no samples are accepted.

## Reading the figures: one model group plus one vote rule defines a point

The x-axis, Samples auto-labelled (%), is the share of the common audio cohort that passes the vote rule: accepted / common samples. The y-axis, Correct auto-labels (%), is the share of those selected labels matching the dataset reference: correct accepted / accepted. This is neither full-dataset accuracy nor a model's confidence score. Auto-labelling here is a simulated selection on benchmark data.

A point at 20% x and 97% y means that out of 1,000 clips, about 200 receive candidate labels: 194 correct and six wrong, with 800 unselected. Unlabelled internal audio has no known y-value; independent human verification is needed.

A model group alone does not define the result: the minimum vote count must also be fixed. The three five-SER points mean at least 3/5, at least 4/5 and all 5/5 agreeing. They are not repetitions of one rule, and their common-cohort denominator is unchanged. The 3/5 rule includes four- and five-vote cases; 4/5 includes five-vote cases. Raising the threshold cannot increase coverage, but need not improve accuracy. Lines only guide the eye between discrete rules; intermediate coverage is not an evaluated rule.

CREMA-D, always the same 7,442 clips:

| Five-SER rule | Selected | Correct / wrong | x: selected share | y: selected-label accuracy |
|---|---:|---:|---:|---:|
| ≥3/5 | 5,843 | 4,177 / 1,666 | 78.51% | 71.49% |
| ≥4/5 | 2,523 | 2,331 / 192 | 33.90% | 92.39% |
| ≥5/5 | 1,448 | 1,415 / 33 | 19.46% | 97.72% |

Colour identifies members, not a quality ranking: blue SER5 is the five SER models; orange Selected3 is emotion2vec+, Gemini and Qwen2.5-Omni; green SER+API7 adds GPT Realtime and Gemini to SER5; pink All10 includes all ten models. Selected3 is a legacy name, not an optimality claim.

Align the rules before comparing colours. Strict majority means 3/5, 2/3, 4/7 or 6/10: different agreement fractions, potentially different selected samples and labels. If five SER models vote three for the correct label and two for a wrong label, two APIs voting for that wrong label turn the result into three correct votes against four wrong votes. Adding models can change the answer, and majority acceptance sets need not shrink.

On the same common cohort, 7/7 unanimity is a subset of 5/5, and 10/10 is a subset of both 7/7 and 3/3: coverage cannot increase. Selected-label accuracy improves only when added models preferentially reject wrong labels rather than correct ones; shared errors or label preferences can prevent that.

For example, CREMA-D blue 3/5 selects 5,843 clips at 71.49% accuracy; green 4/7 selects 5,481 at 52.51%. This changes the voters and minimum vote count rather than simply applying a tighter version of the same filter.

An observed CREMA-D example: 5/5 selects 1,448 clips (1,415 correct, 33 wrong). Requiring both APIs to agree leaves 758 (738 correct, 20 wrong). The added condition retains 52.16% of the original correct cases but 60.61% of the errors, changing precision from 97.72% to 97.36%.

Matched MELD endpoints: unanimity on the identical common cohort.

| Group | Rule | Selected | Correct / wrong | Selected share | Selected-label accuracy |
|---|---|---:|---:|---:|---:|
| SER5 | 5/5 | 279 | 151 / 128 | 10.70% | 54.12% |
| Selected3 | 3/3 | 709 | 499 / 210 | 27.19% | 70.38% |
| SER_API7 | 7/7 | 184 | 109 / 75 | 7.06% | 59.24% |
| All10 | 10/10 | 79 | 56 / 23 | 3.03% | 70.89% |

The pink curve is not an average of the others: every point recounts all ten model votes. MELD 10/10 selects only 79 clips, including 23 errors. It trades much lower coverage for an observed accuracy close to Selected3 at 3/3, rather than automatically becoming optimal. Its other points are 6/10 through 9/10 and should not be treated as the same rule as another group's unanimous endpoint.

Five-SER unanimity reaches an observed 95% accepted-label accuracy on 8/16 datasets. Agreement can select reliable labels in some domains, but does not guarantee high accuracy across datasets, and adding members need not improve it. The supplement adds 4/5 votes, fixed mixed-family groups, error counts and class distributions; every threshold is retained without selecting a best rule on these evaluation labels.

Results use each dataset's configured evaluation scope, not necessarily the entire source corpus: THAI-SER uses high_agreement and MEAD uses stratified_8000. Member availability does not imply full-source-corpus coverage. AudioLLM5 membership varies across datasets, so it is not pooled as a fixed ensemble. Selected3 is the legacy group name, not a claim that it is optimal in this refresh.

## Five SER models

Target members: emotion2vec+, MERaLiON, Melati, Vox-Profile Whisper, Vox-Profile WavLM.

16 datasets: all target members.

This group waits until every member is available.

| Dataset | Participating / target members | Common samples | Unanimous coverage | Unanimous accuracy | Majority coverage | Majority accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 5/5 | 7,442 | 19.46% | 97.72% | 78.51% | 71.49% |
| MCAE-SPPS | 5/5 | 6,675 | 15.13% | 94.46% | 77.50% | 70.52% |
| THAI-SER | 5/5 | 11,848 | 24.59% | 96.02% | 84.72% | 80.93% |
| nEMO | 5/5 | 4,481 | 13.68% | 83.36% | 78.40% | 60.77% |
| MESS | 5/5 | 1,800 | 17.06% | 68.73% | 94.39% | 41.38% |
| RAVDESS | 5/5 | 1,440 | 20.83% | 95.33% | 81.88% | 74.72% |
| TESS | 5/5 | 2,800 | 11.79% | 100.00% | 58.46% | 87.35% |
| JL-Corpus | 5/5 | 1,200 | 22.67% | 99.63% | 91.50% | 89.25% |
| MEAD | 5/5 | 8,000 | 8.22% | 91.19% | 73.02% | 37.18% |
| MELD | 5/5 | 2,608 | 10.70% | 54.12% | 69.02% | 43.72% |
| EMOVO | 5/5 | 588 | 20.24% | 98.32% | 69.22% | 68.80% |
| Emozionalmente | 5/5 | 6,902 | 15.27% | 96.68% | 69.21% | 78.02% |
| SAVEE | 5/5 | 480 | 15.83% | 98.68% | 69.79% | 81.49% |
| EmotionTalk | 5/5 | 1,929 | 12.13% | 83.33% | 68.12% | 61.04% |
| DailyTalk | 5/5 | 1,197 | 26.40% | 87.66% | 85.46% | 74.10% |
| MultiDialog | 5/5 | 7,672 | 30.03% | 73.57% | 90.50% | 61.17% |

## Audio LLMs (including APIs)

Target members: GPT Realtime, Gemini 3.8 Flash, Qwen2.5-Omni, Qwen2-Audio, Gemma 4.

6 datasets: GPT Realtime, Gemini 3.8 Flash, Qwen2.5-Omni, Qwen2-Audio; awaiting: Gemma 4.
9 datasets: GPT Realtime, Gemini 3.8 Flash, Qwen2.5-Omni; awaiting: Qwen2-Audio, Gemma 4.
MELD: all target members.

* marks a group missing other members; available members are included. Consensus is calculated when at least two members are available.

| Dataset | Participating / target members | Common samples | Unanimous coverage | Unanimous accuracy | Majority coverage | Majority accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 4/5* | 7,442 | 12.66% | 93.63% | 54.64% | 34.19% |
| MCAE-SPPS | 4/5* | 6,675 | 4.69% | 69.97% | 39.36% | 44.04% |
| THAI-SER | 4/5* | 11,848 | 11.20% | 95.40% | 60.53% | 75.46% |
| nEMO | 4/5* | 4,481 | 1.32% | 37.29% | 34.72% | 23.46% |
| MESS | 4/5* | 1,800 | 28.61% | 35.34% | 84.33% | 26.42% |
| RAVDESS | 4/5* | 1,440 | 12.78% | 100.00% | 48.96% | 67.80% |
| TESS | 3/5* | 2,800 | 26.14% | 27.19% | 85.93% | 23.36% |
| JL-Corpus | 3/5* | 1,200 | 43.92% | 34.72% | 92.83% | 32.41% |
| MEAD | 3/5* | 8,000 | 50.81% | 14.76% | 91.00% | 15.44% |
| MELD | 5/5 | 2,608 | 12.58% | 72.87% | 75.73% | 56.71% |
| EMOVO | 3/5* | 588 | 42.35% | 26.91% | 91.16% | 20.71% |
| Emozionalmente | 3/5* | 6,902 | 29.54% | 36.10% | 85.03% | 23.19% |
| SAVEE | 3/5* | 480 | 49.17% | 38.98% | 91.04% | 31.12% |
| EmotionTalk | 3/5* | 1,929 | 45.26% | 72.74% | 87.35% | 62.73% |
| DailyTalk | 3/5* | 1,197 | 53.55% | 81.59% | 94.74% | 73.90% |
| MultiDialog | 3/5* | 7,672 | 57.06% | 67.45% | 96.34% | 60.82% |

## Selected models

Target members: emotion2vec+, Gemini 3.8 Flash, Qwen2.5-Omni.

16 datasets: all target members.

* marks a group missing other members; available members are included. Consensus is calculated when at least two members are available.

| Dataset | Participating / target members | Common samples | Unanimous coverage | Unanimous accuracy | Majority coverage | Majority accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 3/3 | 7,442 | 16.07% | 92.56% | 75.96% | 47.97% |
| MCAE-SPPS | 3/3 | 6,675 | 26.49% | 72.12% | 80.19% | 51.54% |
| THAI-SER | 3/3 | 11,848 | 42.84% | 88.46% | 92.06% | 74.31% |
| nEMO | 3/3 | 4,481 | 17.41% | 46.41% | 78.93% | 35.51% |
| MESS | 3/3 | 1,800 | 65.39% | 31.61% | 98.44% | 27.82% |
| RAVDESS | 3/3 | 1,440 | 30.21% | 94.48% | 81.46% | 74.77% |
| TESS | 3/3 | 2,800 | 13.64% | 98.69% | 72.61% | 67.39% |
| JL-Corpus | 3/3 | 1,200 | 21.50% | 87.21% | 83.58% | 53.94% |
| MEAD | 3/3 | 8,000 | 10.35% | 87.80% | 75.28% | 31.25% |
| MELD | 3/3 | 2,608 | 27.19% | 70.38% | 80.25% | 56.43% |
| EMOVO | 3/3 | 588 | 13.27% | 97.44% | 73.64% | 49.19% |
| Emozionalmente | 3/3 | 6,902 | 14.43% | 92.47% | 66.94% | 54.11% |
| SAVEE | 3/3 | 480 | 21.04% | 98.02% | 79.38% | 49.61% |
| EmotionTalk | 3/3 | 1,929 | 29.76% | 82.58% | 81.75% | 63.92% |
| DailyTalk | 3/3 | 1,197 | 39.77% | 82.35% | 90.64% | 70.69% |
| MultiDialog | 3/3 | 7,672 | 41.76% | 68.26% | 93.17% | 59.28% |

## Vote thresholds, accepted labels and errors

The supplement enumerates every integer threshold from strict majority to unanimity; it does not select an optimal group or threshold on these evaluation labels. This table adds the 4/5 SER threshold and makes the cost of erroneous accepted labels explicit.

| Dataset | ≥4/5 accepted | Errors | Coverage | Accuracy | 5/5 accepted | Errors |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 2,523 | 192 | 33.90% | 92.39% | 1,448 | 33 |
| MCAE-SPPS | 3,083 | 669 | 46.19% | 78.30% | 1,010 | 56 |
| THAI-SER | 6,361 | 682 | 53.69% | 89.28% | 2,914 | 116 |
| nEMO | 1,764 | 494 | 39.37% | 72.00% | 613 | 102 |
| MESS | 1,029 | 589 | 57.17% | 42.76% | 307 | 96 |
| RAVDESS | 661 | 113 | 45.90% | 82.90% | 300 | 14 |
| TESS | 647 | 14 | 23.11% | 97.84% | 330 | 0 |
| JL-Corpus | 555 | 20 | 46.25% | 96.40% | 272 | 1 |
| MEAD | 2,867 | 1,669 | 35.84% | 41.79% | 658 | 58 |
| MELD | 887 | 461 | 34.01% | 48.03% | 279 | 128 |
| EMOVO | 219 | 40 | 37.24% | 81.74% | 119 | 2 |
| Emozionalmente | 2,508 | 217 | 36.34% | 91.35% | 1,054 | 35 |
| SAVEE | 171 | 10 | 35.62% | 94.15% | 76 | 1 |
| EmotionTalk | 630 | 186 | 32.66% | 70.48% | 234 | 39 |
| DailyTalk | 634 | 106 | 52.97% | 83.28% | 316 | 39 |
| MultiDialog | 4,502 | 1,456 | 58.68% | 67.66% | 2,304 | 609 |

## Fixed groups across model families

SER+API7 = five SER models + GPT Realtime + Gemini; All10 = five SER models + all five audio LLM/API models. Both supplemental groups wait for every member. The table reports unanimity; the CSV contains all integer thresholds. Direct group comparisons require identical common audio-path cohorts.

| Dataset | SER+API7 common | 7/7 accepted / errors | Coverage | Accuracy | All10 common | 10/10 accepted / errors | Coverage | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CREMA-D | 7,442 | 758 / 20 | 10.19% | 97.36% | Pending | — | — | — |
| MCAE-SPPS | 6,675 | 181 / 4 | 2.71% | 97.79% | Pending | — | — | — |
| THAI-SER | 11,848 | 1,595 / 43 | 13.46% | 97.30% | Pending | — | — | — |
| nEMO | 4,481 | 360 / 61 | 8.03% | 83.06% | Pending | — | — | — |
| MESS | 1,800 | 265 / 80 | 14.72% | 69.81% | Pending | — | — | — |
| RAVDESS | 1,440 | 112 / 5 | 7.78% | 95.54% | Pending | — | — | — |
| TESS | 2,800 | 0 / 0 | 0.00% | — | Pending | — | — | — |
| JL-Corpus | 1,200 | 104 / 1 | 8.67% | 99.04% | Pending | — | — | — |
| MEAD | 8,000 | 327 / 34 | 4.09% | 89.60% | Pending | — | — | — |
| MELD | 2,608 | 184 / 75 | 7.06% | 59.24% | 2,608 | 79 / 23 | 3.03% | 70.89% |
| EMOVO | 588 | 54 / 1 | 9.18% | 98.15% | Pending | — | — | — |
| Emozionalmente | 6,902 | 463 / 13 | 6.71% | 97.19% | Pending | — | — | — |
| SAVEE | 480 | 57 / 0 | 11.88% | 100.00% | Pending | — | — | — |
| EmotionTalk | 1,929 | 161 / 16 | 8.35% | 90.06% | Pending | — | — | — |
| DailyTalk | 1,197 | 206 / 12 | 17.21% | 94.17% | Pending | — | — | — |
| MultiDialog | 7,672 | 1,745 / 370 | 22.75% | 78.80% | Pending | — | — | — |

## Figures: separate vote thresholds from group membership

The five-SER threshold plot labels each minimum vote count; MELD uses a separate panel for each group. Bar charts follow the benchmark's dataset order, coloured bars and value labels, with larger black text. Unanimity bars compare complete fixed groups, excluding the variable-membership AudioLLM5.

![Five-SER vote thresholds](figures/consensus_accuracy_coverage.png)

![MELD model groups and vote thresholds](figures/consensus_meld_accuracy_coverage.png)

![Accuracy of unanimous auto-labels](figures/consensus_unanimous_accuracy_by_dataset.png)

![Share of clips with unanimous auto-labels](figures/consensus_unanimous_coverage_by_dataset.png)

![Five-SER accuracy by minimum votes](figures/consensus_ser5_accuracy_by_votes.png)

![Five-SER auto-labelling share by minimum votes](figures/consensus_ser5_coverage_by_votes.png)

Bar labels round to whole percentages; tables and CSVs retain full precision. n/a indicates missing group members. With zero selected clips, coverage is 0% but accuracy is undefined, not 0%.


## Interpretation for labelling and the paper

Coverage = accepted / common_samples; accepted-label accuracy = accepted_correct / accepted; selective risk = 1 − accuracy; correct accepted fraction = accepted_correct / common_samples. accepted / union_samples also exposes missing model overlap, but the union is still only the available prediction pool, not the entire intended labelling pool.

The per-predicted-class file reports accepted/error counts, class precision and share of accepted labels to reveal dominance by neutral or another class. Individual-model baselines are recomputed on exactly the same within-group cohort. These are observed fixed-prediction point estimates, without confidence intervals or accuracy guarantees; 100% on a small accepted set does not establish zero risk.

For internal labelling, set a tolerated error rate using independently labelled calibration data, freeze the model group and vote threshold, then validate by class, language and recording condition on held-out samples. Audit accepted labels and route rejected samples to humans. Public-data coverage is potential automatic acceptance, while realised effort savings also depend on internal errors, auditing and correction.

For the paper, this can be a supporting selective-prediction accuracy–coverage / risk–coverage analysis of fixed groups and thresholds across public datasets, without claiming a new algorithm or internal labelling benefit. The framework has precedent: [Geifman & El-Yaniv, NeurIPS 2017](https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf). Formal inference would additionally need speaker/dialogue-cluster uncertainty estimates and training provenance disclosure; those intervals are not supplied here.

Supplementary results: [vote thresholds](../output/consensus_thresholds.csv) · [per predicted class](../output/consensus_by_class.csv) · [matched individual baselines](../output/consensus_baselines.csv) · [reproducibility manifest](../output/consensus_manifest.json)

Excluded 0 runs with old inference protocols. Each group is scored on its common audio-path intersection; composite gold labels keep their existing acceptance rules. Performed and dialogue corpora also differ in language, class distribution and recording source, so their differences cannot be attributed to context alone. Shared training sources and errors mean agreement votes are not independent evidence.

[Full counts, membership and source files](../output/consensus.csv). Rerun the script after more model results arrive to update both reports and the CSV.
