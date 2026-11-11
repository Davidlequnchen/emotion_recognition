# Can model agreement help select pseudo-labels?

This analysis reuses 115 compatible saved prediction runs across 16 public datasets; no model was rerun.
Model completion (datasets with a full run in the current evaluation scope / 16): Gemini 3.8 Flash 16/16; GPT Realtime 16/16; Each of the five SER models 16/16; Each of the three local audio LLMs 1/16.
Datasets with every target member available: five-SER group 16; audio-LLM group 1; selected-model group 1. Group completeness requires results from every target member; available and missing members are listed below.
On the same MELD samples, unanimous accepted-label accuracy / coverage is: Five SER models 54.12% / 10.70%; Audio LLMs (including APIs) 72.87% / 12.58%; Selected models 70.38% / 27.19%.
The selected models retain more samples, while their accepted-label accuracy is not the highest of the three groups.
Public-data results do not establish accuracy on company calls; calibrate the group and threshold on a small manually labelled call sample first.

Coverage is the accepted share of common samples; accepted-label accuracy is the correct share of accepted labels. Majority means strictly more than half of the current model count. Invalid answers abstain without reducing that denominator; top-vote ties are rejected. With two models, majority equals unanimity; with five models, majority requires at least three votes. Accuracy is — when no samples are accepted.

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

the 15 datasets other than MELD: GPT Realtime, Gemini 3.8 Flash; awaiting: Qwen2.5-Omni, Qwen2-Audio, Gemma 4.
MELD: all target members.

* marks a group missing other members; available members are included. Consensus is calculated when at least two members are available.

| Dataset | Participating / target members | Common samples | Unanimous coverage | Unanimous accuracy | Majority coverage | Majority accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 2/5* | 7,442 | 52.31% | 25.69% | 52.31% | 25.69% |
| MCAE-SPPS | 2/5* | 6,675 | 38.32% | 32.60% | 38.32% | 32.60% |
| THAI-SER | 2/5* | 11,848 | 66.26% | 73.15% | 66.26% | 73.15% |
| nEMO | 2/5* | 4,481 | 68.42% | 21.10% | 68.42% | 21.10% |
| MESS | 2/5* | 1,800 | 78.28% | 26.54% | 78.28% | 26.54% |
| RAVDESS | 2/5* | 1,440 | 35.90% | 51.45% | 35.90% | 51.45% |
| TESS | 2/5* | 2,800 | 34.04% | 21.93% | 34.04% | 21.93% |
| JL-Corpus | 2/5* | 1,200 | 56.08% | 35.07% | 56.08% | 35.07% |
| MEAD | 2/5* | 8,000 | 63.11% | 14.30% | 63.11% | 14.30% |
| MELD | 5/5 | 2,608 | 12.58% | 72.87% | 75.73% | 56.71% |
| EMOVO | 2/5* | 588 | 49.15% | 26.99% | 49.15% | 26.99% |
| Emozionalmente | 2/5* | 6,902 | 35.99% | 36.92% | 35.99% | 36.92% |
| SAVEE | 2/5* | 480 | 64.38% | 36.89% | 64.38% | 36.89% |
| EmotionTalk | 2/5* | 1,929 | 56.92% | 69.31% | 56.92% | 69.31% |
| DailyTalk | 2/5* | 1,197 | 62.57% | 77.97% | 62.57% | 77.97% |
| MultiDialog | 2/5* | 7,672 | 70.02% | 65.90% | 70.02% | 65.90% |

## Selected models

Target members: emotion2vec+, Gemini 3.8 Flash, Qwen2.5-Omni.

the 15 datasets other than MELD: emotion2vec+, Gemini 3.8 Flash; awaiting: Qwen2.5-Omni.
MELD: all target members.

* marks a group missing other members; available members are included. Consensus is calculated when at least two members are available.

| Dataset | Participating / target members | Common samples | Unanimous coverage | Unanimous accuracy | Majority coverage | Majority accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CREMA-D | 2/3* | 7,442 | 32.93% | 90.04% | 32.93% | 90.04% |
| MCAE-SPPS | 2/3* | 6,675 | 38.64% | 69.29% | 38.64% | 69.29% |
| THAI-SER | 2/3* | 11,848 | 59.72% | 87.05% | 59.72% | 87.05% |
| nEMO | 2/3* | 4,481 | 37.27% | 46.89% | 37.27% | 46.89% |
| MESS | 2/3* | 1,800 | 68.78% | 33.84% | 68.78% | 33.84% |
| RAVDESS | 2/3* | 1,440 | 48.26% | 93.24% | 48.26% | 93.24% |
| TESS | 2/3* | 2,800 | 30.50% | 96.72% | 30.50% | 96.72% |
| JL-Corpus | 2/3* | 1,200 | 38.33% | 90.22% | 38.33% | 90.22% |
| MEAD | 2/3* | 8,000 | 19.05% | 87.20% | 19.05% | 87.20% |
| MELD | 3/3 | 2,608 | 27.19% | 70.38% | 80.25% | 56.43% |
| EMOVO | 2/3* | 588 | 31.46% | 92.43% | 31.46% | 92.43% |
| Emozionalmente | 2/3* | 6,902 | 34.61% | 90.54% | 34.61% | 90.54% |
| SAVEE | 2/3* | 480 | 32.50% | 96.79% | 32.50% | 96.79% |
| EmotionTalk | 2/3* | 1,929 | 41.21% | 75.97% | 41.21% | 75.97% |
| DailyTalk | 2/3* | 1,197 | 51.63% | 73.30% | 51.63% | 73.30% |
| MultiDialog | 2/3* | 7,672 | 57.73% | 62.68% | 57.73% | 62.68% |

Excluded 12 runs with old inference protocols. Each group is scored on its common audio-path intersection; composite gold labels keep their existing acceptance rules. Performed and dialogue corpora also differ in language, class distribution and recording source, so their differences cannot be attributed to context alone. Shared training sources and errors mean agreement votes are not independent evidence.

[Full counts, membership and source files](../output/consensus.csv). Rerun the script after more model results arrive to update both reports and the CSV.
