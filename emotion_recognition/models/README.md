# models/

One sub-folder per model, registered under `models:` in `config/config.yaml`. Each model is run by its loader,
`scripts/models/<loader>.py`, which builds it from the files in its folder only and never uses the network.

| Folder | Source | Licence | Size | How it is loaded |
|---|---|---|---|---|
| `emotion2vec_plus_large/` (in use) | [emotion2vec/emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) · [ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) | FunASR model licence (attribution, keep the model name) | data2vec-style encoder, 162.1M; `model.pt` 1.95 GB (with optimizer state) | FunASR (`emotion2vec.py`) |
| `meralion_ser_v1/` (new) | [MERaLiON/MERaLiON-SER-v1](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) | MERaLiON Public Licence (MIT with attribution terms) | Whisper-medium encoder + ECAPA-TDNN head, 313.4M in use; `model.safetensors` 3.08 GB (770.0M, with an unused Whisper decoder) | transformers, the model code in the folder (`meralion_ser.py`) |
| `melati_emotion/` (new) | [Scicom-intl/Melati-Emotion-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3) | non-commercial research only | Whisper-large-v3 encoder + small head, 637.3M; `model.safetensors` 1.28 GB (encoder in float16) | transformers, the model code in the folder (`melati.py`) |
| `vox_profile_whisper/` (new) | [tiantiaf/whisper-large-v3-msp-podcast-emotion](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion) | OpenRAIL; the card rules out commercial use | Whisper-large-v3 encoder + head, 639.9M; `model_slim.safetensors` 1.29 GB | Vox-Profile code (`vox_profile.py`) |
| `vox_profile_wavlm/` (new) | [tiantiaf/wavlm-large-categorical-emotion](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion) | OpenRAIL; the card rules out commercial use | WavLM-large + head, 316.2M; `model.safetensors` 1.26 GB | Vox-Profile code (`vox_profile.py`) |

## Getting the models

**On the Mac, or any machine that reaches Hugging Face** (from the emotion_recognition/ folder):

```bash
python scripts/common/download_model.py
```

It downloads every model listed in `download.models` (`all`, or a list such as `[meralion_ser_v1]`), runs the loader's
`prepare()` (extra files for loading offline), lists the files and checks their sizes against Hugging Face. Files that
are already complete are not downloaded again. `download.hub: ms` uses ModelScope instead, for models with an
`ms_repo` (only emotion2vec+ large).

**On a machine without Hugging Face** (the NVIDIA server): the weights come through git, cut into 90 MiB parts.

```bash
git pull
python scripts/common/join_parts.py
```

`join_parts.py` puts each `<file>.part000`, `.part001`, ... back together next to its parts and checks the whole file
against `<file>.sha256`. Nothing else is needed: every file a model loads is in its folder. emotion2vec+ large is not in
git: download it with `download_model.py` (`download.models: [emotion2vec_plus_large]`, and `download.hub: ms` if
only ModelScope is reachable).

## Pushing new weights through git

GitHub refuses files over 100 MB and pushes over 2 GB. On the Mac:

1. Download the model (above).
2. `python scripts/common/split_parts.py` cuts every file listed in `parts.files` (config) into 90 MiB parts and writes
   its `.sha256`. It skips files that are already split and replaces the parts of a file that changed.
3. git ignores the model folders, so add files with `git add -f`, and name them: `git add -f models/<key>` would also
   add the whole weights file (over 100 MB) and `.cache/`. Add the small files first, then the parts.
4. Commit and push in batches of at most about 1.5 GB (15 parts). For example, for Melati:

```bash
# the small files: everything except the weights (*.safetensors), their parts, .cache/ and __pycache__/
git add -f $(find models/melati_emotion -type f ! -name '*.safetensors' ! -name '*.part[0-9][0-9][0-9]' \
             ! -path '*/.cache/*' ! -path '*/__pycache__/*')
git add -f models/melati_emotion/model.safetensors.part00[0-9]   # 10 parts, 0.94 GB
git commit -m "Add the Melati model (1/2)" && git push
git add -f models/melati_emotion/model.safetensors.part01[0-3]   # 4 parts, 0.33 GB
git commit -m "Add the Melati model (2/2)" && git push
```

`download_model.py` skips each repository's `.gitattributes` (and deletes one left by an earlier download): its Git LFS
rules (`*.png`, `*.pdf`, `*.safetensors`, ...) would apply to the folder in this repository and send the licence and
figures to LFS when they are committed.

## emotion2vec_plus_large

| File | Content |
|---|---|
| `model.pt` | weights, 1.95 GB: 162.1M fp32 weights plus the optimizer state saved during training |
| `config.yaml` | network architecture, read by FunASR to build the model |
| `configuration.json` | tells FunASR which file is the config, the weights and the label list |
| `tokens.txt` | the 9 output classes, in output order |
| `README.md`, `example/test.wav` | model card (licence) and an example clip |

The official release has no `config.json`; `configuration.json` plays that role.

## meralion_ser_v1

| File | Content |
|---|---|
| `model.safetensors` | weights, 3.08 GB float32: the Whisper-medium encoder (LoRA merged, 307.2M), the head (6.1M) and the Whisper decoder (456.6M), which is never used. In git: `model.safetensors.part000` to `.part032` and `model.safetensors.sha256` |
| `config.json`, `configuration_ser_whisper_ecapa.py`, `modeling_ser_whisper_ecapa.py` | the model code (remote code); 7 classes: neutral, happy, sad, angry, fearful, disgusted, surprised |
| `preprocessor_config.json` | Whisper feature extractor: 80 mel bins, every clip padded or cut to 30 s |
| `whisper-medium/config.json` | `openai/whisper-medium`'s config (MIT), added by `prepare()`: the model code builds its encoder from it, which would otherwise be a download at every load |
| `MERaLiON-Public-Licence-SER-V1.pdf` | the licence |
| `README.md`, figures (`*.png`, `SG_UAR_4plots.pdf`), tokenizer files, `exporter_*.py` | the rest of the repository, unused here |

Loading: the model code imports `peft` and `omegaconf` (both in `requirements.txt`) but does not use them. The loader
builds the model from its class and then loads the weights, because transformers 5 builds models on the meta device,
where this model's code fails.

Licence obligations: this repository is public, so committing the parts redistributes the weights, which the licence
allows on its terms. Keep the licence PDF next to the weights (it holds the terms and the disclaimer of warranties),
keep the creators' names and the copyright notice, and say whether anything was modified. Nothing is modified here: the
weights travel as parts that `join_parts.py` puts back together byte for byte (SHA-256 checked). One file is added:
`whisper-medium/config.json`.

## melati_emotion

| File | Content |
|---|---|
| `model.safetensors` | weights, 1.28 GB: the Whisper-large-v3 encoder in float16 and the head in float32. In git: `model.safetensors.part000` to `.part013` and `model.safetensors.sha256` |
| `config.json`, `configuration_melati_head.py`, `modeling_melati_head.py` | the model code (remote code, self-contained); 7 classes: neutral, happy, sad, angry, surprised, fearful, disgusted |
| `preprocessor_config.json` | Whisper feature extractor: 128 mel bins, every clip padded or cut to 30 s |
| `melati_emotion.py` | the official inference code, which the loader follows (pooling over the clip's own frames) |
| `LICENSE` | the licence |
| `README.md`, `figures/` | model card and its figures |

The loader uses the raw probabilities (most probable class). It does not apply the thresholds of
`melati_emotion.py`'s `label()`: they were chosen for precise tagging of podcasts and would turn most clips into
neutral. Precision: float32 on the CPU, float16 encoder on MPS and CUDA (as on the model card). On the Mac, MPS
float16 gave the same prediction as CPU float32 for 72 of 72 test clips (largest probability difference 0.002).

Licence obligations: non-commercial research only (research, teaching, publication; no commercial products or
services). This repository is public, so committing the parts redistributes the weights: allowed, because `LICENSE`,
which must travel with them, is committed in the same folder. Nothing is modified.

## vox_profile_whisper, vox_profile_wavlm

The two categorical emotion models of Vox-Profile (Feng et al. 2025, arXiv:2505.14648), trained on MSP-Podcast, with
9 classes: angry, contempt, disgusted, fearful, happy, neutral, sad, surprised, other. `scripts/models/vox_profile.py`
loads both. It re-implements the inference code of the official repository
([vox-profile-release](https://github.com/tiantiaf0627/vox-profile-release)), so it needs no network, loralib or
speechbrain. On 12 test clips its probabilities differ from those of the official code by at most 2.4e-6.

| File | Content |
|---|---|
| `config.json` | the model's settings: base model (`pretrain_model`), head size, LoRA rank |
| `README.md` | model card: OpenRAIL licence and out-of-scope uses |
| `whisper-large-v3/`, `wavlm-large/` | `config.json` and `preprocessor_config.json` of the base model (openai/whisper-large-v3, microsoft/wavlm-large), saved by `prepare()` so that loading needs no network |
| `model_slim.safetensors` (Whisper) | the weights, 1.29 GB (639.9M parameters). `prepare()` writes it from the release's `model.safetensors` (6.19 GB, float32): it removes Whisper's text decoder (772 tensors, 906.5M parameters, never used for emotions) and stores as float16 the 516 tensors whose values are exact in float16 (openai's frozen encoder weights, and LoRA B, which is all zeros); the other 58 stay float32. Its metadata records the source commit and SHA-256. Its predictions are identical to those of the full file |
| `model.safetensors` (WavLM) | the release, 1.26 GB, float32 (316.2M parameters) |

In git, the weights are stored as 90 MiB parts with a `.sha256` file: `model_slim.safetensors.part000`–`part013`
(Whisper) and `model.safetensors.part000`–`part013` (WavLM). `join_parts.py` puts them back together. Whisper's
original `model.safetensors` stays out of git.

The LoRA adapters of the Whisper release have no effect (LoRA B is all zeros), so its encoder is exactly
openai/whisper-large-v3 and only the head on top is its own. The WavLM release has a fully fine-tuned WavLM-large. Both
cards say the models were trained on clips of 3–15 s and that shorter clips give unreliable predictions; most clips in
our datasets are shorter than 3 s.

**Licences.** The weights are under OpenRAIL. Both model cards list as out of scope: clinical or diagnostic use,
surveillance, privacy-invasive uses and any commercial use, so these models are for research only. The loader is
derived from the official code, which is under the Responsible AI Source Code License v1.1 (full text:
`scripts/models/vox_profile_LICENSE.txt`). Its Section 3.2 forbids, among other uses, surveillance (inferring a
person's legally protected characteristics or identity) and predicting crime. Whoever uses or passes on this code must
follow these restrictions and pass them on.

## Adding a model

1. Register it under `models:` in `config/config.yaml` (`name`, `loader`, `dir`, `hf_repo`, `training_data`,
   `training_source`).
2. Write its loader, `scripts/models/<loader>.py`: `load(model_dir, device)`, and `prepare(model_dir)` if it needs
   files that are not in its repository to load offline.
3. Download it: `download.models: [<key>]`, then `python scripts/common/download_model.py`.
4. For a machine without Hugging Face: add its large files to `parts.files`, run `split_parts.py` and commit the parts.
