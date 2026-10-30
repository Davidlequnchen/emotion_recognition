# antideepfake

Deepfake detection: telling real (bona fide) speech from deepfakes (synthetic or converted speech) with
**XLS-R-1B-AntiDeepfake** from NII Yamagishi Lab ([code](https://github.com/nii-yamagishilab/AntiDeepfake); paper:
*Post-training for Deepfake Speech Detection*, [arXiv:2506.21090](https://arxiv.org/abs/2506.21090)). It is the
XLS-R 1B speech model post-trained on 56k hours of real and 18k hours of fake speech, with a 2-class output layer.

This is a side project of the repository, independent of the speech emotion recognition project next to it
([`../emotion_recognition/`](../emotion_recognition/README.md)). The only link between them: by default it scores
the six CREMA-D test clips of that project.

It also makes its own test set of synthetic speech: three voice-cloning models read 200 call-centre lines in the voice
of a consenting speaker (section 7).

How the model is built and trained (architecture diagram, size, training data and method):
[docs/xlsr_1b_antideepfake.md](docs/xlsr_1b_antideepfake.md) (Chinese) and
[docs/xlsr_1b_antideepfake_en.md](docs/xlsr_1b_antideepfake_en.md) (English).

## Quick start

Run every command from this folder, `antideepfake/`: every path in this README is relative to it.

```bash
conda create -n antideepfake python=3.11 -y   # or use the emotion_recognition environment: it has everything
conda activate antideepfake
cd antideepfake                               # this folder of the repository
pip install -r requirements.txt               # NVIDIA server: first the CUDA build of PyTorch, see section 1
python scripts/antideepfake/download_model.py  # xlsr_1b.ckpt (3.86 GB) from Zenodo -> models/antideepfake/
python scripts/antideepfake/inference.py       # NVIDIA server: first set `mode: cuda` in config/antideepfake.yaml
```

The scripts have no command-line options: every setting is in [`config/antideepfake.yaml`](config/antideepfake.yaml).

## Layout

```
antideepfake/
├── config/
│   ├── antideepfake.yaml       # detector: hardware mode (mac | cuda), devices, checkpoint, audio, threshold
│   ├── voice_clone.yaml        # voice cloning: models (pinned sources), devices, reference, texts, seed, output
│   └── record.yaml             # recorder: speaker name, length, microphone, sentences to read
├── requirements.txt            # detector: torch, torchaudio, soundfile, pyyaml (standalone)
├── requirements-voice-clone.txt  # voice cloning: its own conda environment `voice_clone`
├── scripts/
│   ├── antideepfake/
│   │   ├── download_model.py   # downloads the checkpoint from Zenodo and checks its size and MD5
│   │   └── inference.py        # step-by-step inference: P(real) for every audio file
│   └── voice_clone/
│       ├── download_models.py  # CosyVoice code (GitHub), the three models (Hugging Face), wetext FSTs (ModelScope)
│       ├── record_reference.py # a window that records a 5-10 s reference clip of your voice
│       ├── generate.py         # one clip per line of texts.csv, per model, in the reference voice
│       └── cosyvoice_tts.py, chatterbox_tts.py  # the model loaders used by generate.py
├── data/voice_clone/
│   ├── texts.csv               # 200 synthetic call-centre lines (id, scenario, role, text)
│   └── reference/              # your recordings (not in git)
├── docs/                       # the models explained (Chinese and English); figures/ holds the detector diagram
├── models/
│   ├── antideepfake/           # xlsr_1b.ckpt, LICENSE-CHECKPOINT, README.txt (downloaded, not in git)
│   └── voice_clone/            # cosyvoice2_0.5b/, cosyvoice3_0.5b/, chatterbox/, wetext/ (downloaded, or cloned from
│                               # the private repository voice_clone_models; not in this repository)
├── third_party/CosyVoice/      # the official CosyVoice code at a pinned commit (cloned, not in git)
└── outputs/voice_clone/        # generated clips: <speaker>/<model>/<id>.wav + manifest.csv (not in git)
```

## 1. Set up the environment

Python 3.11 with PyTorch 2.5.1 (tested on the M4 Max, MPS and CPU). The packages are torch, torchaudio, soundfile
and pyyaml, pinned in [`requirements.txt`](requirements.txt); the download uses only the Python standard library.

- **With the emotion_recognition environment** (see [its README](../emotion_recognition/README.md), section 1):
  nothing to install. It already has every package at the same version; `pip install -r requirements.txt` installs
  nothing (checked with `pip install --dry-run`).
- **In an environment of its own** (a machine that runs only this project):
  ```bash
  conda create -n antideepfake python=3.11 -y && conda activate antideepfake
  pip install -r requirements.txt   # MacBook (Apple Silicon): PyTorch from PyPI, MPS + CPU
  ```
  On an NVIDIA server (CUDA 12.1), install the CUDA build of PyTorch first; `requirements.txt` then keeps it:
  ```bash
  pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
  The CUDA wheels bundle the CUDA runtime and cuDNN: the server only needs an NVIDIA driver that supports CUDA 12.1
  (`nvidia-smi` shows it).

Check PyTorch: `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"`
prints `2.5.1 False True` on the Mac and `2.5.1+cu121 True False` on the server.

Upstream AntiDeepfake also installs fairseq and speechbrain; neither is used here. fairseq builds the network
upstream, but installing it would downgrade hydra-core and omegaconf, which breaks FunASR in the emotion_recognition
environment, and it has no wheel for Python 3.11; speechbrain is the training framework. Details in
`requirements.txt`.

## 2. Download the model

```bash
python scripts/antideepfake/download_model.py
```

The checkpoint comes from Zenodo only ([record 15580543](https://zenodo.org/records/15580543)), never from Hugging
Face, so the download also works on a machine that cannot reach Hugging Face or ModelScope. The `model` section of
`config/antideepfake.yaml` names the record, the file and the folder (`models/antideepfake/`). The file is streamed to disk and checked against
the size and MD5 that Zenodo publishes; a file that is already correct is skipped. Zenodo can be slow (0.2 to
1.3 MB/s here), can stall a connection without closing it, and sometimes answers "504 Gateway Time-out": the script
then reconnects and resumes where it stopped. If it gives up, run the same command again.

`models/antideepfake/` then holds:

| File | Content |
|---|---|
| `xlsr_1b.ckpt` | weights, 3.86 GB: a PyTorch state dict of 965.5 M fp32 weights with fairseq names (`m_ssl.model.*` for XLS-R, `proj_fc.*` for the output layer); MD5 `61dde44830acfa192a2ff13bd0e8925a` |
| `LICENSE-CHECKPOINT`, `README.txt` | licence and readme of the Zenodo record |

The checkpoint holds no architecture description: `inference.py` defines the network (the `xlsr_1b` configuration of
AntiDeepfake) and loads every weight by name. 3.0 M of the weights (`mask_emb`, `quantizer`, `project_q`,
`final_proj`) belong to wav2vec 2.0 pre-training and are not used for inference. Git ignores these files and keeps
only `models/antideepfake/.gitkeep`.

## 3. Run inference

```bash
python scripts/antideepfake/inference.py
```

Settings in `config/antideepfake.yaml`: `mode` (`mac` or `cuda`) and `devices` choose the hardware (set `devices.mac: cpu` to force
the CPU); in the `inference` section, `audio` (a file, a folder with its sub-folders, or a list of files and folders;
wav, flac, mp3, ogg; default `../emotion_recognition/data/samples`), `threshold` (default 0.5) and `max_seconds`
(default `null`).

The script runs six steps and prints each one: (1) read the configuration, (2) check Python, PyTorch, CUDA and MPS,
(3) activate the device and run a small test computation on it, (4) load the model onto the device, (5) list the
audio files, (6) score every file. For every file it prints P(real), P(fake) and the two logits. If the requested
device is not available, it stops with an error instead of silently falling back to the CPU. It needs no network at
run time: the network is defined in the script and every weight comes from `xlsr_1b.ckpt`.

Output on the M4 Max (`mode: mac`, MPS) for the six CREMA-D test clips in `../emotion_recognition/data/samples/`,
which are real speech (each line is followed by P(fake) and the logits):

```
  [1/6] 1001_DFA_ANG_XX.wav          -> real   P(real) 0.808   53 ms
  [2/6] 1001_DFA_DIS_XX.wav          -> real   P(real) 0.983   147 ms
  [3/6] 1001_DFA_FEA_XX.wav          -> real   P(real) 0.969   140 ms
  [4/6] 1001_DFA_HAP_XX.wav          -> real   P(real) 0.991   143 ms
  [5/6] 1001_DFA_NEU_XX.wav          -> real   P(real) 0.886   134 ms
  [6/6] 1001_DFA_SAD_XX.wav          -> FAKE   P(real) 0.463   131 ms
```

Five of the six real clips count as real; the sad one falls just below 0.5. Two clips synthesised with the macOS
`say` voices score P(real) 0.001 and 0.000, clearly fake, also as a 44.1 kHz stereo file. A clip rebuilt from a
real one with the Griffin-Lim algorithm still scores 0.840: it keeps the speaker's own prosody, and the model was
trained on other kinds of fake speech.

## 4. What the score means

For each file the model gives two logits, in the order [fake, real]. Their softmax gives P(real), the probability
that the speech is real: this is AntiDeepfake's own score (`softmax_score()` in its `evaluation.py`), so a high score
means real, not fake. The deepfake score is P(fake) = 1 − P(real). A file counts as real when P(real) ≥ 0.5,
AntiDeepfake's default threshold. The best threshold depends on the data: on the model card, the threshold at the
equal error rate ranges from 0.46 (FakeOrReal) to 0.99 (Deepfake-Eval-2024), and the equal error rate itself from
1.35% (In-the-Wild) to 26.8% (Deepfake-Eval-2024).

## 5. How it matches AntiDeepfake

The audio is prepared as in its `test_audio_pipeline()`: channels averaged to mono, resampled to 16 kHz, the waveform
normalised to zero mean and unit variance, and each file scored in one pass, whatever its length. AntiDeepfake builds
the network with fairseq, which would downgrade hydra-core and omegaconf and break FunASR, so `inference.py` defines
the same network in plain PyTorch, with fairseq's parameter names: the checkpoint loads unchanged and every name and
shape is checked. Its logits agree with two independent implementations of the same network, Hugging Face
transformers and torchaudio loaded with the same weights, to within 2.5 × 10⁻⁵ on the clips above; MPS and the CPU
give the same scores, and the script also runs with every network connection blocked.

## 6. Speed

On the M4 Max, loading the model takes about 2 s (the checkpoint is memory-mapped) and 3.6 GB of GPU memory; scoring
a 2-second clip takes about 120 ms on MPS and 200 ms on the CPU, and a 30-second file about 3 s. MPS cannot process
files longer than 20.48 s; those run on the CPU automatically, or set `max_seconds: 20`. `max_seconds` splits longer
files into equal segments and averages their logits; it is not part of AntiDeepfake, but keeps the memory use of
long recordings in check.

## 7. Voice cloning: a test set of synthetic speech

Three voice-cloning models with licences that allow commercial use read the 200 lines of `data/voice_clone/texts.csv`
in the voice of a reference recording: CosyVoice2-0.5B, Fun-CosyVoice3-0.5B-2512 (Apache-2.0) and Chatterbox
(English, MIT). With one reference that gives 3 × 200 = 600 clips. The models, their sources, papers, architecture,
input and output: [docs/voice_clone_models.md](docs/voice_clone_models.md) (Chinese) and
[docs/voice_clone_models_en.md](docs/voice_clone_models_en.md) (English).

Only record yourself, or a colleague who has agreed to it. Recordings and generated clips stay on this computer
(git-ignored); never upload them to an online voice service.

```bash
# 1. Environment (once): its own conda environment, the exact commands are at the top of requirements-voice-clone.txt
conda create -n voice_clone python=3.11 -y && conda activate voice_clone
pip install "setuptools<81" wheel
pip install --no-build-isolation openai-whisper==20231117
pip install -r requirements-voice-clone.txt
pip install --no-deps chatterbox-tts==0.1.7

# 2. Models (once): CosyVoice code from GitHub, the models from Hugging Face, wetext FSTs from ModelScope (needs Git LFS)
python scripts/voice_clone/download_models.py      # -> third_party/CosyVoice, models/voice_clone/ (10.7 GB)
#    Company computer (no Hugging Face / ModelScope): get the model files from the private repository first
#    git clone https://github.com/Davidlequnchen/voice_clone_models.git models/voice_clone
#    python models/voice_clone/join_parts.py          # joins the 90 MiB parts, checks every SHA-256
#    then download_models.py only clones the CosyVoice code from GitHub and skips the models

# 3. Record a reference: a window opens; read the sentence shown after the countdown (8 s)
python scripts/voice_clone/record_reference.py     # -> data/voice_clone/reference/<speaker>_<time>.wav + .txt

# 4. Generate: every model, one clip at a time, with the newest recording
python scripts/voice_clone/generate.py             # -> outputs/voice_clone/<speaker>/<model>/<id>.wav + manifest.csv

# 5. Score the clips: set `audio: outputs/voice_clone` in config/antideepfake.yaml, then (in its environment)
python scripts/antideepfake/inference.py
```

- **Recorder** (`config/record.yaml`): set `speaker` to your name first; it names the files and the output folder.
  macOS asks once for microphone access for the terminal app; without it the take is silent (the window warns).
  The sentence in the text box is saved as the transcript: read it exactly, or edit the box to match.
- **Generation** (`config/voice_clone.yaml`): `models` (all, or e.g. `[chatterbox]`), `generate.reference`
  (`latest` or a path), `limit` (first N lines, for a quick test), `seed`, `overwrite`. Models run one after the
  other and are freed in between; nothing is downloaded at run time.
- **Reproducible**: row i uses seed + i, and the models, code and wetext files are pinned to fixed revisions. On the
  same machine and device, a second run gives byte-identical files (checked for all three models); MPS and CPU give
  different clips. `manifest.csv` records the text, reference and its SHA-256, seed, duration, versions and device.
- **Time on the M4 Max**: CosyVoice2 runs at about real time on the CPU, CosyVoice3 slightly slower, Chatterbox on
  MPS at about 1.5 × real time: 200 lines (about 15 minutes of speech) take roughly 15, 20 and 25 minutes.
- A one-line Chatterbox test clip scored P(real) 0.000 with the detector.

## Licence

The checkpoint is CC BY-NC-SA 4.0: research and other non-commercial use only, with attribution, and anything built
from it shared under the same licence. AntiDeepfake's code is BSD-3-Clause; no upstream code is copied into this
repository.

Voice cloning: CosyVoice2-0.5B, Fun-CosyVoice3-0.5B-2512, the CosyVoice code and the wetext FSTs are Apache-2.0;
Chatterbox is MIT. The model cards (README.md) are kept in each model folder.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ERROR: .../models/antideepfake/xlsr_1b.ckpt not found` | Run `python scripts/antideepfake/download_model.py`. |
| `ERROR: ... is incomplete` or `Zenodo does not answer` from `download_model.py` | Zenodo was slow or busy. Run the same command again: it resumes where it stopped. |
| `MPS ... available=False` on the Mac | Python is x86_64 (Rosetta). `python -c "import platform; print(platform.machine())"` must print `arm64`: reinstall Miniconda for Apple Silicon. |
| `CUDA ... available=False` on the server | `torch.__version__` must end in `+cu121`: install the CUDA build as in section 1. Otherwise the driver is too old: check `nvidia-smi`. |
