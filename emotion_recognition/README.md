# emotion_recognition

Exploring speech emotion recognition (SER), from traditional machine learning to deep learning.
Current focus: five open-source SER models, run by one script on the same datasets and scored against the same labels:
**emotion2vec+ large** (162.1M parameters), **MERaLiON-SER-v1** (313.4M), **Melati** (637.3M),
**Vox-Profile Whisper** (639.9M) and **Vox-Profile WavLM** (316.2M); see [Models](#models). They run with PyTorch on
two kinds of machines:

| Mode | Machine | Device |
|---|---|---|
| `mac` | MacBook with Apple Silicon (tested on M4 Max, 48 GB) | Apple GPU via Metal (`mps`) or `cpu` |
| `cuda` | Linux server with NVIDIA GPU(s), CUDA 12.1 | `cuda:0` (or another GPU index) |

## Quick start

This folder is the main project of the repository (the deepfake detection side project is in
[`../antideepfake/`](../antideepfake/README.md)). Run every command from this folder, `emotion_recognition/`: every
path in this README is relative to it.

```bash
conda create -n emotion_recognition python=3.11 -y
conda activate emotion_recognition
cd emotion_recognition                         # this folder of the repository
pip install -r requirements-mac.txt            # NVIDIA server: pip install -r requirements-cuda121.txt
pip install -r requirements.txt
python scripts/common/download_model.py        # the five models -> models/<key>/ (server: see section 2)
python scripts/emotion2vec/inference.py        # NVIDIA server: first set `mode: cuda` in config/config.yaml
# optional (first set `download.dataset: all` in config/config.yaml): the evaluation datasets -> data/,
# every model's accuracy on each -> reports/, and a web page with the results and examples -> outputs/viewer.html
python scripts/common/download_data.py
python scripts/evaluation/evaluate.py
python scripts/viewer/make_viewer.py
```

The scripts have no command-line options: every setting is in [`config/config.yaml`](config/config.yaml), one
section per script. Each step is explained below.

## Layout

```
emotion_recognition/
├── config/config.yaml          # hardware mode (mac | cuda), devices, model registry, one section per script
├── data/
│   ├── samples/                # six CREMA-D test clips (in git, 0.4 MB)
│   ├── crema_d/                # the full CREMA-D dataset (optional download, not in git)
│   ├── mcae_spps/ mess/ ravdess/ tess/ jl_corpus/   # the other evaluated datasets (optional downloads)
│   └── thai_ser/ nemo/ esd/    # optional downloads too, left out of the evaluation for now (section 6)
├── docs/
│   ├── crema_d.md              # CREMA-D explained in Chinese: file names, labels, every column of the label files
│   ├── datasets_survey.md      # emotion recognition public dataset survey (in Chinese)
│   ├── datasets_survey_en.md   # the same survey in English
│   ├── models_survey.md        # open-source emotion recognition model survey (English: models_survey_en.md)
│   ├── emotion2vec_plus_large.md             # the model explained: architecture, size, training (+ _en.md)
│   ├── finetuning_emotion2vec_plus_large.md  # what fine-tuning would need: data, GPUs, which layers (+ _en.md)
│   └── figures/                # the architecture diagram (PowerPoint and PNG)
├── models/
│   ├── README.md               # every model's files, size and licence; how to get, split and add models
│   ├── emotion2vec_plus_large/ # model.pt, config.yaml, configuration.json, tokens.txt (downloaded, not in git)
│   ├── meralion_ser_v1/        # MERaLiON-SER-v1: its Hugging Face repository; weights in git as 90 MiB parts
│   ├── melati_emotion/         # Melati: the same
│   └── vox_profile_whisper/ vox_profile_wavlm/  # the two Vox-Profile models: the same
├── outputs/                    # per-clip predictions (outputs/<model>/) and the page viewer.html (not in git)
├── reports/
│   ├── summary.md              # every model on every dataset (.json: the same numbers)
│   ├── <model>/                # one report per dataset (e.g. emotion2vec_plus_large/crema_d.md) and summary.md
│   │   └── figures/            # confusion matrices used in the reports
│   └── archive/                # reports of the datasets that are no longer evaluated (nEMO, ESD, THAI-SER)
├── scripts/
│   ├── common/                 # shared by the pipelines
│   │   ├── download_data.py    # downloads CREMA-D (test clips or full) and eight more datasets, checks every file
│   │   ├── download_model.py   # downloads the models into models/<key>/, ready to load without the network
│   │   ├── split_parts.py      # cuts large files (model weights) into 90 MiB parts that git can take
│   │   ├── join_parts.py       # joins such parts after git pull (models and datasets), checks their SHA-256
│   │   └── eval_datasets.py    # one loader per evaluation dataset (labels -> the project's classes)
│   ├── models/                 # one loader per model: emotion2vec.py, meralion_ser.py, melati.py, vox_profile.py
│   │                           # (+ vox_profile_LICENSE.txt); audio_io.py reads audio as 16 kHz mono
│   ├── evaluation/
│   │   └── evaluate.py         # runs every chosen model on every chosen dataset, writes the reports and summaries
│   ├── emotion2vec/
│   │   └── inference.py        # step-by-step inference with emotion2vec+ large (a single-model demo)
│   ├── llm_eval/               # audio LLMs (GPT Realtime, Gemini): prompts, data loader, answer parser
│   └── viewer/
│       ├── make_viewer.py      # builds a web page with every model's results and example clips
│       └── templates/          # viewer.html, the page template used by make_viewer.py
├── site/index.html             # the public results page (built by make_viewer.py, published by GitHub Pages)
├── test_scripts/               # informal one-off checks, e.g. of the OpenAI API (see its README)
├── requirements.txt            # common packages (both machines)
├── requirements-mac.txt        # PyTorch for Apple Silicon (MPS + CPU)
└── requirements-cuda121.txt    # PyTorch for NVIDIA GPUs, CUDA 12.1
```

Git tracks the code, configuration, documentation and the six small test clips. Model files and the full datasets
are in the repository's `.gitignore` (`model.pt` alone is 1.9 GB). On every new machine, re-create them with the
two download scripts.
For now there are exceptions, for a machine that cannot reach the hosts of these models and datasets (such as the
NVIDIA server, which cannot reach Hugging Face): the weights of the four new models with their small files (about
6.9 GB, see section 2), the unpacked MCAE-SPPS files (without its zips), the TESS files, the JL-Corpus recordings
(`JL(wav+txt)/`; not the rest of its zip, see `data/jl_corpus/README.md`) and the nEMO Parquet files. Files over
100 MB (GitHub refuses them) are committed in 90 MiB parts; after `git pull`, `python scripts/common/join_parts.py`
joins and checks them.

## 1. Set up the environment

Tested with Python 3.11, PyTorch 2.5.1, FunASR 1.4.16 and transformers 5.18.0 on the M4 Max (MPS and CPU). The
Linux + CUDA 12.1 installation was checked in a Linux x86_64 container without a GPU, before the four new models were
added.

### 1.1 Install conda (once per machine)

Skip this if `conda --version` already works.

- **Mac**: install Miniconda for **Apple Silicon (arm64)**, for example `brew install --cask miniconda`.
  An x86_64 conda running under Rosetta cannot use the Apple GPU.
- **Linux server**:
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh    # then open a new shell
  ```

### 1.2 Create and activate the environment

```bash
conda create -n emotion_recognition python=3.11 -y
conda activate emotion_recognition
python -c "import platform; print(platform.python_version(), platform.machine())"
# Mac: 3.11.x arm64        Linux server: 3.11.x x86_64
```

Activate the environment (`conda activate emotion_recognition`) in every new terminal before running the scripts.

### 1.3 Install the requirements

Run both commands from this folder (`emotion_recognition/`): first PyTorch for your hardware, then the common
packages.

**MacBook (Apple Silicon)**

```bash
pip install -r requirements-mac.txt     # torch 2.5.1 + torchaudio 2.5.1 from PyPI (MPS + CPU)
pip install -r requirements.txt         # FunASR, transformers, peft, hubs, numpy, pyyaml, soundfile
```

**NVIDIA GPU server (CUDA 12.1)**

```bash
nvidia-smi                              # "CUDA Version" (top right) must be 12.1 or higher
pip install -r requirements-cuda121.txt # torch 2.5.1+cu121 + torchaudio 2.5.1+cu121 from download.pytorch.org
pip install -r requirements.txt         # same common packages as on the Mac
```

Notes on the versions:

- **PyTorch 2.5.1** is the newest release with official CUDA 12.1 wheels (2.6 dropped CUDA 12.1). The Mac uses
  the same version, so both machines run identical code.
- The CUDA 12.1 wheels **bundle the CUDA runtime and cuDNN**. The server only needs an NVIDIA driver that supports
  CUDA 12.1 (Linux driver 530 or newer); a separate CUDA toolkit is not needed.
- PyTorch is kept in its own file because the CUDA build comes from the PyTorch package index, while everything
  else comes from PyPI. Each `pip install` therefore uses a single index, and the order of the two commands
  does not matter.
- **Python 3.11** is supported by every pinned package. numpy is pinned to 2.4.6, the newest release for Python 3.11.
- **transformers is pinned to 5.18.0**: it runs MERaLiON-SER-v1, Melati and Vox-Profile, and FunASR imports it. It is
  the newest release that works with PyTorch 2.5: 5.19.0 cannot import any model class there (it calls
  `torch.accelerator`, new in PyTorch 2.6), and 5.20+ requires PyTorch 2.6.
- **peft 0.21.2 and omegaconf 2.3.1** are imported by MERaLiON-SER-v1's model code, which does not use them (its LoRA
  is already merged into the weights). **accelerate 1.15.0** is a peft dependency, pinned so that pip never upgrades
  PyTorch for it.

### 1.4 Check PyTorch

```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

Expected output: on the Mac, `2.5.1 | CUDA: False | MPS: True`. On the server, `2.5.1+cu121 | CUDA: True | MPS: False`.

## 2. Download the models

**On the Mac, or any machine that reaches Hugging Face:**

```bash
python scripts/common/download_model.py   # every model in `download.models` (default: all) -> models/<key>/
```

The script downloads each model's Hugging Face repository into its folder (about 14 GB for all five; the Vox-Profile
Whisper release alone is 6.19 GB). Then it runs the loader's `prepare()`, which saves the few extra files the model
needs to load offline (for Vox-Profile Whisper it also writes a 1.29 GB weights file without the unused text decoder),
and checks the file sizes against Hugging Face. Files that are already complete are skipped, so the script can simply
be re-run. `download.models: [meralion_ser_v1]` gets one model only; `download.hub: ms` uses ModelScope (faster in
mainland China; only emotion2vec+ large is there). The "unauthenticated requests" warning from Hugging Face is
harmless: the models are public and need no token.

**On the NVIDIA server, which cannot reach Hugging Face:** the four new models come through git, in 90 MiB parts
(about 6.9 GB: MERaLiON-SER-v1 33 parts, Melati 14, Vox-Profile Whisper 14, Vox-Profile WavLM 14).

```bash
git pull
python scripts/common/join_parts.py       # joins the parts under data/ and models/, checks each file's SHA-256
```

emotion2vec+ large is not in git: get it with `download_model.py` (`download.models: [emotion2vec_plus_large]`, and
`download.hub: ms` if only ModelScope is reachable).

Every file a model loads is then in its folder; the evaluation never touches the network. On the Mac, after
downloading or changing weights, `python scripts/common/split_parts.py` cuts the files in `parts.files` into parts for
git. [models/README.md](models/README.md) lists every model's files and licence, and how to commit the parts.

## 3. Download the data (CREMA-D)

```bash
python scripts/common/download_data.py
```

The `download` section of `config/config.yaml` chooses what to get. With `dataset: crema_d` (the default),
`crema_d: samples` gets the 6 test clips into `data/samples/` (0.4 MB) and `crema_d: full` the whole dataset into
`data/crema_d/` (606 MB). Both come from **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset), an open dataset for
emotion recognition research: 7,442 clips from 91 actors (48 male, 43 female, aged 20 to 74), each saying one of
12 sentences with one of six emotions (anger, disgust, fear, happy, neutral, sad). Crowd-sourced listeners
(2,443 raters in total) also rated every clip from the audio alone, the video alone and both together.
[docs/crema_d.md](docs/crema_d.md) explains, in Chinese, the file names, the two kinds of labels (acted and
perceived) and every column of the label files such as `summaryTable.csv`.

### 3.1 Test clips (default)

`data/samples/` gets six clips: actor 1001 saying "Don't forget a jacket" (`DFA`) with each emotion, so only the
emotion changes between files. `scripts/emotion2vec/inference.py` uses them by default. Names follow
`<actor>_<sentence>_<emotion>_<level>.wav`, where `XX` means the intensity level was not specified. Each file is
16 kHz mono WAV, about 2 s and 70 kB. The six clips are committed to the repository, so they are already there
after `git clone`; the command above restores any that are missing and checks their SHA-256.

| File | Acted emotion |
|---|---|
| `1001_DFA_ANG_XX.wav` | angry |
| `1001_DFA_DIS_XX.wav` | disgusted |
| `1001_DFA_FEA_XX.wav` | fearful |
| `1001_DFA_HAP_XX.wav` | happy |
| `1001_DFA_NEU_XX.wav` | neutral |
| `1001_DFA_SAD_XX.wav` | sad |

### 3.2 Full dataset (`crema_d: full`)

`data/crema_d/` gets all 7,442 WAV files plus the label and metadata files, at the same paths as in the
CREMA-D repository:

```
data/crema_d/
├── AudioWAV/                            # 7,442 clips, 16 kHz mono: 1,271 per emotion, 1,087 neutral
├── processedResults/summaryTable.csv    # listeners' majority vote per clip (voice, face, both)
├── processedResults/tabulatedVotes.csv  # vote counts per emotion, per clip and presentation
├── VideoDemographics.csv                # actor age, sex, race and ethnicity
├── SentenceFilenames.csv                # stimulus number of every clip
├── README.md                            # the dataset's own documentation
└── LICENSE.txt
```

The first download took about 9 minutes on the MacBook (8 files in parallel; set this with `download.workers`). Most of
that time is GitHub's response time for each file. If the download is interrupted, run the same command again:
files that are already correct are skipped, and checking all 7,442 files takes about 10 seconds. The clips last
1.3 to 5.0 s (5.3 hours in total). To evaluate the models on every clip, see section 6.

### 3.3 How the download works

Both modes use a pinned version of the CREMA-D repository on GitHub,
[CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) at commit `1658cd3`
(2025-03-21), so the data stays the same on every machine and over time. The repository stores its audio with
Git LFS, so each clip is fetched from

```
https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/<commit>/AudioWAV/<file name>
```

and checked against the SHA-256 that the repository publishes for it. For the six test clips, the checksums are
written in the script. For the full dataset, the script first downloads the repository archive (about 25 MB): it
contains the label files and, for every audio file, a small Git LFS pointer that holds the file's SHA-256.

To fetch one clip by hand:

```bash
mkdir -p data/samples
curl -L -o data/samples/1001_DFA_ANG_XX.wav \
  https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/1001_DFA_ANG_XX.wav
```

The script downloads only the WAV audio and the label files. The repository also holds MP3 copies, the videos
and the raw listener responses (about 7.5 GB in total). To get everything, clone it with [git-lfs](https://git-lfs.com):
`git lfs install && git clone https://github.com/CheyneyComputerScience/CREMA-D.git`. A plain ZIP download from
GitHub contains only small LFS pointer files instead of the media.

License: CREMA-D is available under the [Open Database License](http://opendatacommons.org/licenses/odbl/1.0/);
its individual contents are licensed under the [Database Contents License](http://opendatacommons.org/licenses/dbcl/1.0/).
Citation: H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova and R. Verma, "CREMA-D: Crowd-sourced
Emotional Multimodal Actors Dataset", *IEEE Transactions on Affective Computing*, 5(4), 377–390, 2014.

### 3.4 More evaluation datasets

The same script downloads eight more datasets that, like CREMA-D, record the same sentences in different
emotions. They were chosen from the emotion recognition public dataset survey in
[docs/datasets_survey_en.md](docs/datasets_survey_en.md) (Chinese: [docs/datasets_survey.md](docs/datasets_survey.md)),
which has links, sizes and caveats for many more.

```bash
python scripts/common/download_data.py   # with `download.dataset: mcae_spps` (or thai_ser, nemo, mess, esd,
                                         # ravdess, tess, jl_corpus; all = everything)
```

| `download.dataset` | Language | Clips | Download | Format in `data/<name>/` | Labels | Licence |
|---|---|---|---|---|---|---|
| `mcae_spps` | Mandarin | 6,675 (6 actors, 200 sentences) | 0.66 GB | WAV 48 kHz in `audio/<emotion>/`, labels in `Corpus information.xlsx` | 7 emotions; 40 listeners per clip, intensity 1–9 | CC BY-NC 4.0 |
| `thai_ser` | Thai | 27,854 (200 actors) | 12.7 GB | Parquet with the audio inside (up to 4 microphones per clip) | 5 emotions; assigned and crowd majority labels, agreement | CC BY-SA 4.0 |
| `nemo` | Polish | 4,481 (9 speakers, 90 sentences) | 0.53 GB | Parquet with the audio inside | 6 emotions (acted, checked by listeners) | CC BY-NC-SA 4.0 |
| `mess` | English | 1,800 (6 speakers) | 0.19 GB | WAV 24.4 kHz in `audio/`; emotion in the file name (`A`ngry, `C`alm, `H`appy, `S`ad) | 4 emotions (acted); listener ratings only on request | CC BY 4.0 |
| `esd` | English, Mandarin | 35,000 (20 speakers, 350 sentences) | 2.45 GB | WAV 16 kHz in `audio/Emotion Speech Dataset/<speaker>/<emotion>/`; text and emotion per clip in `<speaker>/<speaker>.txt` | 5 emotions (acted; no listener test) | research only, licence form (see below) |
| `ravdess` | English | 1,440 (24 actors, 2 sentences) | 0.21 GB | WAV 48 kHz in `audio/Actor_<nn>/`; emotion, intensity, sentence and actor in the file name | 8 emotions including calm, 2 intensities (acted; listener ratings per file in the paper) | CC BY-NC-SA 4.0 |
| `tess` | English | 2,800 (2 actresses, 200 words) | 0.28 GB | WAV 24.4 kHz, named `<speaker>_<word>_<emotion>.wav` | 7 emotions (acted) | CC BY-NC 4.0 |
| `jl_corpus` | English (New Zealand) | 2,400 (4 actors, 15 sentences) | 1.28 GB | WAV 44.1 kHz, each with its transcript, in `audio/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/`, named `<speaker>_<emotion>_<sentence><session>_<repetition>.wav` | 5 primary + 5 secondary emotions (acted; the perception test results are not published) | CC0 |

Every file is checked against the checksum its host publishes (OSF SHA-256, Zenodo MD5, Hugging Face SHA-256,
Borealis MD5). Google Drive and Kaggle publish none, so the ESD and JL-Corpus zips are checked with their own CRCs and
a SHA-256 written in the script. For RAVDESS only the speech audio is downloaded, not its song audio and videos.
Zip files are unpacked into `audio/` and kept. The Parquet files of THAI-SER and nEMO hold the audio and the labels
together (reading them needs `pyarrow`); their loaders in `scripts/common/eval_datasets.py` extract the audio into
`audio/` the first time the evaluation reads them.
ESD is for research only: before using it, complete the licence form on its
[GitHub page](https://github.com/HLTSingapore/Emotional-Speech-Data) and email it to the authors.
Several licences forbid commercial use or sharing copies, so use the data for research only; `data/` is git-ignored
except the CREMA-D test clips.

## 4. Choose the hardware mode

Edit `config/config.yaml`:

```yaml
mode: mac          # mac  = Apple Silicon MacBook  |  cuda = NVIDIA GPU server

devices:
  mac: mps         # mps (Apple GPU) | cpu
  cuda: cuda:0     # cuda:<gpu index> | cpu
```

Every script reads `mode` from this file. `devices` is the device of the inference script (section 5); the
evaluation has its own setting, `evaluation.devices` (section 6).

## 5. Run inference

```bash
python scripts/emotion2vec/inference.py
```

This script is a step-by-step demo of one model, emotion2vec+ large; the evaluation (section 6) runs all five.
Settings in `config/config.yaml`: `mode` and `devices` choose the hardware (set `devices.mac: cpu` to force the CPU),
`model` the emotion2vec model to run, and `inference.audio` the audio file or folder to run on (wav, flac, mp3, ogg;
default `data/samples`).

The script runs six steps and prints each one: (1) read the configuration, (2) check Python, PyTorch, CUDA and MPS,
(3) activate the device and run a small test computation on it, (4) load the model onto the device,
(5) list the audio files, (6) predict an emotion for every file.
If the requested device is not available, it stops with an error instead of silently falling back to the CPU.

Output on the M4 Max (`mode: mac`, MPS; paths shortened):

```
================================================================
Step 1/6 | Load configuration
================================================================
  config      : .../emotion_recognition/config/config.yaml
  mode        : mac
  device      : mps
  model       : emotion2vec_plus_large (.../emotion_recognition/models/emotion2vec_plus_large)
  audio       : .../emotion_recognition/data/samples

================================================================
Step 2/6 | Check environment
================================================================
  Python      : 3.11.17 (Darwin arm64)
  PyTorch     : 2.5.1
  torchaudio  : 2.5.1
  FunASR      : 1.4.16
  CUDA        : available=False (PyTorch built with CUDA None)
  MPS (Apple) : available=True (built=True)

================================================================
Step 3/6 | Activate device
================================================================
  using       : Apple GPU via Metal (MPS)
  sanity check: 512x512 matrix multiplication on mps ... OK

================================================================
Step 4/6 | Load model
================================================================
  model.pt           : 1,945.79 MB
  config.yaml        : 0.01 MB
  configuration.json : 0.00 MB
  tokens.txt         : 0.00 MB
funasr version: 1.4.16.
  loaded in   : 0.9 s
  parameters  : 164.0 M, on mps:0
  GPU memory  : 0.61 GB allocated
  classes     : 9 -> angry, disgusted, fearful, happy, neutral, other, sad, surprised, <unk>

================================================================
Step 5/6 | Collect audio
================================================================
  1001_DFA_ANG_XX.wav      16000 Hz, 1 ch, 2.28 s
  1001_DFA_DIS_XX.wav      16000 Hz, 1 ch, 2.34 s
  1001_DFA_FEA_XX.wav      16000 Hz, 1 ch, 2.17 s
  1001_DFA_HAP_XX.wav      16000 Hz, 1 ch, 1.87 s
  1001_DFA_NEU_XX.wav      16000 Hz, 1 ch, 2.04 s
  1001_DFA_SAD_XX.wav      16000 Hz, 1 ch, 2.00 s
  (the model works on 16 kHz mono audio; other formats are converted automatically)

================================================================
Step 6/6 | Run inference
================================================================
  warm-up done (the first forward pass initialises the device kernels)

  [1/6] 1001_DFA_ANG_XX.wav      -> angry (1.00)   14 ms
        top-3: angry 1.00 | sad 0.00 | neutral 0.00
  [2/6] 1001_DFA_DIS_XX.wav      -> disgusted (1.00)   91 ms
        top-3: disgusted 1.00 | angry 0.00 | sad 0.00
  [3/6] 1001_DFA_FEA_XX.wav      -> fearful (1.00)   87 ms
        top-3: fearful 1.00 | happy 0.00 | surprised 0.00
  [4/6] 1001_DFA_HAP_XX.wav      -> happy (0.95)   99 ms
        top-3: happy 0.95 | fearful 0.05 | surprised 0.00
  [5/6] 1001_DFA_NEU_XX.wav      -> neutral (1.00)   98 ms
        top-3: neutral 1.00 | sad 0.00 | angry 0.00
  [6/6] 1001_DFA_SAD_XX.wav      -> disgusted (0.80)   98 ms
        top-3: disgusted 0.80 | sad 0.08 | fearful 0.07

Done: 6 file(s) on mps, average 81 ms per file.
```

What to expect:

- **MPS and CPU give the same result.** On the test clips the class probabilities differ by at most 1.2e-5.
- **Speed on the M4 Max:** with MPS, the first clip of each new length takes about 100 ms, because Metal builds
  kernels for every new input shape; clips of an already-seen length take about 11 ms. The CPU takes about 55 ms
  per 2-second clip.
- **5 of the 6 clips match the acted emotion.** The SAD clip comes out as disgusted. The labels are what the actor
  was asked to portray. In CREMA-D's own voice-only listening test, the majority vote for this SAD clip, and for the
  HAP clip, was neutral, so these two clips are ambiguous.
- The `parameters: 164.0 M` line counts the network built from `config.yaml`: 162.1M weights loaded from `model.pt`,
  plus a pre-training decoder of about 2M that inference never uses. The model card says "~300M"; the released
  checkpoint holds 162.1M weights.

## 6. Evaluate on the datasets

```bash
python scripts/evaluation/evaluate.py
```

Settings, in the `evaluation` section of `config/config.yaml`:

```yaml
evaluation:
  models: all       # all registered models, or a list such as [emotion2vec_plus_large, meralion_ser_v1]
  datasets: all     # all downloaded datasets, or a list such as [crema_d, ravdess]
  exclude: [esd, thai_ser, nemo]  # never evaluated, even with `all`, and left out of the summaries (old results:
                                  # reports/archive/); their data stays in data/
  limit: null       # only N clips of each dataset, taken at even intervals within each emotion (quick test or
                    # sample); null = every clip
  overwrite: false  # false: skip a model and dataset whose results (same limit) are already in outputs/<model>/
  devices:          # device per mode; a model's mac_device (in `models`) overrides the one for the Mac
    mac: mps        # emotion2vec+ large has mac_device: cpu (faster for it, see the speeds below)
    cuda: cuda:0
  summary_figure:   # every dataset's confusion matrix in one figure, in each model's summary report
    order: [crema_d, mcae_spps, mess]  # panels row by row; datasets not listed come last
    acted_or_listener: [crema_d]  # these show the matrix where the listeners' vote also counts as correct;
                                  # the others show the acted emotion
```

The models are registered under `models:` in the same file: display name, loader, folder, Hugging Face repository,
and `training_data`, the evaluation datasets that the model was trained on. Each model runs through its loader in
`scripts/models/` and loads from its folder in `models/` only: the script sets `HF_HUB_OFFLINE=1` and never touches
the network. A model whose folder has no weights yet is skipped with a message. To add a model, see
[models/README.md](models/README.md#adding-a-model).

Each dataset has a loader in `scripts/common/eval_datasets.py` that reads its labels and maps them to the project's
classes; to evaluate on a new dataset, add a loader there. For each model and dataset the script writes (`<model>` is
the model's key in the registry, e.g. `emotion2vec_plus_large`):

- `reports/<model>/<dataset>.md`: accuracy, unweighted accuracy, macro F1, results per emotion, confusion matrices
  (figures in `reports/<model>/figures/` and tables), comparison with human listeners where the dataset has their
  labels, and caveats.
- `outputs/<model>/<dataset>_predictions.csv`: one row per clip (not in git) with the audio path, the labels, the
  model's prediction, whether it is correct, and all class probabilities. The JSON file next to it describes the
  columns and holds the metrics, the run time and the ground-truth labels (`labels`, in the order of the
  confusion-matrix rows); [docs/crema_d.md](docs/crema_d.md) explains the CREMA-D columns in Chinese.
- `reports/<model>/summary.md`: every dataset of the model side by side (with its language, its ground-truth labels
  and whether it is in the model's training data), every dataset's confusion matrix in one figure, and a comparison
  on the emotions that every dataset has.
- [`reports/summary.md`](reports/summary.md): every model on every dataset, with accuracy, unweighted accuracy, macro
  F1 and the accuracy on the shared emotions; † marks a dataset that is in the model's training data.

A sample run (`limit`) adds `_sample<N>` to the file names, e.g. `reports/meralion_ser_v1/crema_d_sample14.md`. A model
and dataset whose results for the same `limit` are already in `outputs/<model>/` are skipped, so after adding a
dataset or a model the script evaluates only the new ones; set `overwrite: true` to evaluate again. A model's summary
covers every dataset with results in `outputs/<model>/`, also those not listed in `datasets`, and `reports/summary.md`
every model with results, also those not listed in `models`; for a dataset with a full run and a sample, they show the
run with more clips.

The datasets name their emotions differently. Each loader maps its dataset's labels to the project's seven classes:
angry, disgusted, fearful, happy, neutral, sad and surprised. The model loaders give the models' classes the same
names: MERaLiON-SER-v1 and Melati have exactly these seven, Vox-Profile has nine (plus `contempt` and `other`) and
emotion2vec+ nine (plus `other` and `<unk>`).

| Dataset | Emotions in the dataset (where the label is) | Mapped to the project's classes |
|---|---|---|
| CREMA-D | 6, code in the file name: `ANG` `DIS` `FEA` `HAP` `NEU` `SAD` | ANG → angry, DIS → disgusted, FEA → fearful, HAP → happy, NEU → neutral, SAD → sad |
| MCAE-SPPS | 7, second digit of the file name: 1 Neutral, 2 Happiness, 3 Anger, 4 Fear, 5 Sadness, 6 Disgust, 7 Surprise | Neutral → neutral, Happiness → happy, Anger → angry, Fear → fearful, Sadness → sad, Disgust → disgusted, Surprise → surprised |
| THAI-SER (excluded for now) | 5, column `assigned_emo`: Neutral, Angry, Happy, Sad, Frustrated | Neutral → neutral, Angry → angry, Happy → happy, Sad → sad; **Frustrated has no match, so its 5,270 clips are left out** |
| nEMO (excluded for now) | 6, column `emotion`: neutral, anger, fear, happiness, sadness, surprised | neutral → neutral, anger → angry, fear → fearful, happiness → happy, sadness → sad, surprised → surprised |
| MESS | 4, first letter of the file name: `A`ngry, `C`alm, `H`appy, `S`ad | A → angry, **C (calm) → neutral, the closest class**, H → happy, S → sad |
| ESD (excluded for now) | 5, folder name: Neutral, Angry, Happy, Sad, Surprise (the Mandarin transcripts say 中立, 生气, 快乐, 伤心, 惊喜) | Neutral → neutral, Angry → angry, Happy → happy, Sad → sad, Surprise → surprised |
| RAVDESS | 8, third number of the file name: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised | neutral, happy, sad, angry, fearful and surprised keep their names, disgust → disgusted, **calm → neutral, the closest class** (so neutral has 96 + 192 clips) |
| TESS | 7, last word of the file name: angry, disgust, fear, happy, neutral, ps, sad | angry, happy, neutral and sad keep their names, disgust → disgusted, fear → fearful, **ps (pleasant surprise) → surprised** |
| JL-Corpus | 10, second part of the file name: angry, excited, happy, neutral, sad (primary); anxious, apologetic, assertive, concerned, encouraging (secondary) | angry, happy, neutral and sad keep their names; **excited stays excited, and a prediction of happy or surprised counts as correct** (a composite label, below); **the five secondary emotions have no match, so their 1,200 clips are left out** |

How the probabilities become a prediction:

- **Main result** (*accuracy: acted emotion*): the prediction is the most probable of the dataset's own emotions; the
  model's other classes (for example emotion2vec+'s `other` and `<unk>`, or Vox-Profile's `contempt` and `other`) are
  ignored. On MESS, for example, only angry, happy, neutral and sad compete, so chance level is 25%. Every model's raw
  probabilities are used: Melati's official thresholds are not applied (see [Models](#models)).
- **All model classes** (in each report): the prediction is the most probable of all the model's classes, so choosing
  a class that the dataset does not have (for example surprised on CREMA-D) counts as wrong.
- **Unweighted accuracy** is the mean recall of the acted emotions, so every emotion counts the same; **macro F1** is
  the mean F1 of the predicted emotions.
- **Listeners' votes** (CREMA-D, MCAE-SPPS, THAI-SER) use the same mapping; for a tied vote, any of the tied emotions
  counts.
- **Composite labels**: a label without a matching class can keep its name and accept several classes (`accept` in
  the loader). JL-Corpus's *excited* accepts happy or surprised: an excited clip counts as correct when the prediction
  is either (also with all model classes), surprised is one of JL-Corpus's prediction classes for that reason, and
  chance level counts both (24% instead of 20%). In the report, such a label has its own row in the confusion matrix
  (*excited (happy or surprised)*), whose cells that count as correct are in bold and, in the figure, outlined (bold
  text alone is hard to see in a light cell); for precision, recall and macro F1,
  a correct excited clip counts as a clip of the class predicted and a wrong one as happy (macro F1 leaves out
  surprised, which no clip is acted, as long as the model never predicts it).
- Some labels do not match exactly. MESS's and RAVDESS's *calm* are scored as *neutral*, and JL-Corpus's *excited* as
  happy or surprised. Clips of emotions that the project's classes do not have are left out: THAI-SER's *frustrated*
  (and a listeners' vote of *frustrated* can never match) and JL-Corpus's secondary emotions. TESS's *pleasant
  surprise* and ESD's Mandarin label for Surprise, 惊喜, both mean a pleasant surprise, while emotion2vec+'s class is
  called 吃惊 (surprised).

Results: only emotion2vec+ large has full results so far. The four new models were only tested on 14 clips per
dataset (those test results are not kept). To evaluate them on every clip, keep `evaluation.models: all` (or list them) and `limit: null`, run
`python scripts/evaluation/evaluate.py` (about 4.5 hours on the Mac, see the speeds below), then
`python scripts/viewer/make_viewer.py`. [reports/summary.md](reports/summary.md) compares every model on every
dataset.

emotion2vec+ large (its [summary](reports/emotion2vec_plus_large/summary.md) adds unweighted accuracy and macro F1):

| Dataset | Language | Ground-truth labels | In emotion2vec+ training data (EmoBox) | Clips | Emotions (chance) | Accuracy: acted emotion | Accuracy: acted or listeners' vote |
|---|---|---|---|---|---|---|---|
| [CREMA-D](reports/emotion2vec_plus_large/crema_d.md) | English | angry, disgusted, fearful, happy, neutral, sad | yes | 7,442 | 6 (16.7%) | **86.5%** | 89.6% |
| [MCAE-SPPS](reports/emotion2vec_plus_large/mcae_spps.md) | Mandarin | neutral, happy, angry, fearful, sad, disgusted, surprised | no | 6,675 | 7 (14.3%) | **61.3%** | 61.3% |
| [MESS](reports/emotion2vec_plus_large/mess.md) | English | angry, happy, neutral, sad | no | 1,800 | 4 (25.0%) | **40.8%** | – |
| [RAVDESS](reports/emotion2vec_plus_large/ravdess.md) | English | neutral, happy, sad, angry, fearful, disgusted, surprised | yes | 1,440 | 7 (14.3%) | **92.1%** | – |
| [TESS](reports/emotion2vec_plus_large/tess.md) | English | angry, disgusted, fearful, happy, neutral, surprised, sad | yes | 2,800 | 7 (14.3%) | **85.5%** | – |
| [JL-Corpus](reports/emotion2vec_plus_large/jl_corpus.md) | English (New Zealand) | angry, happy, neutral, sad, excited (happy or surprised) | yes | 1,200 | 5 (24.0%) | **92.5%** | – |

*Ground-truth labels*: the acted emotions under the project's names (*neutral* includes MESS's and RAVDESS's
*calm*, TESS's *surprised* is its *pleasant surprise*, and JL-Corpus's *excited* is a composite label, see above), in
the order of the confusion-matrix rows. *In emotion2vec+ training data (EmoBox)*: *yes* means that the dataset is in
EmoBox, whose data fine-tuned emotion2vec+ seed, the first model in the chain that led to emotion2vec+ large, so the
model has probably seen its clips; *no* means that it is not in EmoBox, although earlier exposure cannot be ruled out
completely (emotion2vec+ large was later trained on pseudo-labelled data whose sources are not published). The
results of CREMA-D, RAVDESS, TESS and JL-Corpus are therefore optimistic; MCAE-SPPS and MESS give a more realistic
picture. *Acted or listeners' vote* also accepts the listeners' majority vote; it exists only for datasets that
publish listeners' labels (for MCAE-SPPS the two agree by construction, because only clips that listeners recognised
were kept). The other models were trained on other data: CREMA-D is in MERaLiON-SER-v1's training data and JL-Corpus
in Melati's, while the Vox-Profile models were trained on MSP-Podcast only; `reports/summary.md` marks such results
with †.

ESD, THAI-SER and nEMO are left out of the evaluation for now (`exclude`): ESD's licence is research-only and forbids
sharing it, it has no listener test, and it is in EmoBox; THAI-SER (Thai) and nEMO (Polish) are in languages that
matter less for this project. Their data stays in `data/`; their earlier emotion2vec+ reports are in
[reports/archive/](reports/archive/README.md) (ESD on a 3,500-clip sample: 97.0%; THAI-SER: 63.2%; nEMO: 52.0%) and
their predictions in `outputs/archive/`. Remove a dataset from `exclude` (and from `viewer.exclude`) to evaluate it
again.

RAVDESS, TESS and JL-Corpus, the three English datasets added last, are all in EmoBox, and their results are high, as
expected. TESS's two speakers reach 99% to 100% on six emotions, but none of its 400 *pleasant surprise* clips comes
out as surprised (70% are predicted as fearful); JL-Corpus counts only its five primary emotions.

The datasets have different emotions, and some emotions are much harder than others. On the four emotions that every
dataset has (angry, happy, neutral and sad; chance 25%), counting only clips of these emotions and taking the most
probable of the four as the prediction (also in the [summary](reports/emotion2vec_plus_large/summary.md)):

| Dataset | Language | Ground-truth labels | In emotion2vec+ training data (EmoBox) | Clips | Accuracy, 4 shared emotions |
|---|---|---|---|---|---|
| [TESS](reports/emotion2vec_plus_large/tess.md) | English | angry, disgusted, fearful, happy, neutral, surprised, sad | yes | 1,600 | **100.0%** |
| [JL-Corpus](reports/emotion2vec_plus_large/jl_corpus.md) | English (New Zealand) | angry, happy, neutral, sad, excited (happy or surprised) | yes | 960 | **94.7%** |
| [CREMA-D](reports/emotion2vec_plus_large/crema_d.md) | English | angry, disgusted, fearful, happy, neutral, sad | yes | 4,900 | **93.2%** |
| [RAVDESS](reports/emotion2vec_plus_large/ravdess.md) | English | neutral, happy, sad, angry, fearful, disgusted, surprised | yes | 864 | **92.5%** |
| [MCAE-SPPS](reports/emotion2vec_plus_large/mcae_spps.md) | Mandarin | neutral, happy, angry, fearful, sad, disgusted, surprised | no | 4,297 | **87.8%** |
| [MESS](reports/emotion2vec_plus_large/mess.md) | English | angry, happy, neutral, sad | no | 1,800 | **40.8%** |

The four datasets in EmoBox come first (92.5% to 100%), but MCAE-SPPS, which is not, is close behind: its lower
overall result comes from its three other emotions (recall: fearful 8.5%, disgusted 8.5%, surprised 28.6%). MESS
stays low: the model takes most of its clips for neutral (85% of the angry clips and 92% of the sad ones), and its
*calm* is scored as *neutral*.

Speed on the M4 Max, and the time of a full run (21,357 clips per model: the six datasets):

| Model | Device on the Mac | Time per clip | Full run |
|---|---|---|---|
| emotion2vec+ large | CPU (`mac_device: cpu`) | about 90 ms; 0.4 to 0.5 s on MESS and TESS | about 50 minutes |
| MERaLiON-SER-v1 | MPS | about 180 ms | about 1 hour |
| Melati | MPS (float16 encoder) | about 280 ms | about 1.5 hours |
| Vox-Profile Whisper | MPS | about 150 ms | about 50 minutes |
| Vox-Profile WavLM | MPS | 130 to 230 ms | about 1 hour |

The times of the four new models come from the 14-clip test, so their full-run times are rough. emotion2vec+ large runs
on the CPU because nearly every clip has a different length, and MPS compiles new kernels for every new length.
FunASR, which runs it, also resamples the 24,414 Hz audio of MESS and TESS slowly; the other models read audio with
`scripts/models/audio_io.py`, which builds each resampler only once. The Whisper-based models pad every clip to the
same length (30 s, or 15 s for Vox-Profile Whisper), so MPS sees a single input shape. An NVIDIA GPU should be much
faster (not measured).

## 7. Results and examples in the browser

```bash
python scripts/viewer/make_viewer.py
```

Settings, in the `viewer` section of `config/config.yaml`: `predictions` (`all` = every
`outputs/<model>/*_predictions.csv`, or a list of such files), `correct` and `wrong` (per dataset and model: how many
clips it got right and wrong to add to the examples; default 10 and 10), `seed` (the same seed always picks the same
clips), `link_audio`, `out` (the page for this machine), `site` (the public page, see 7.1) and `exclude` (datasets
left out of both pages; default ESD, THAI-SER and nEMO).

The script writes one self-contained page, `outputs/viewer.html` (audio embedded as 16 kHz MP3: about 2.5 MB with one
model, and expected to stay under 10 MB with all five; open it in any browser). It has three tabs:

- **Results** (opens first), with a model switch; values are coloured by level (green for 80% or more, amber for 60
  to 80%, red under 60%):
  - **All models**: one row per dataset and one column per model, as in [reports/summary.md](reports/summary.md), with
    a switch between accuracy, unweighted accuracy and macro F1; † marks a dataset in that model's training data.
  - **One model**: its results table, as in `reports/<model>/summary.md` (read from the JSON file next to it), with
    the datasets in its training data first, then the others, each group sorted by accuracy; then its confusion
    matrices (click the figure to enlarge it) and the table of the emotions that every dataset has.
- **Examples**: pick a dataset and a model, and listen to clips that the models got right and wrong (below).
- **Datasets**: the source, licence and credit of each dataset (`SOURCES` in `scripts/common/eval_datasets.py`).

The open tab is part of the page's address (`#results`, `#examples`, `#datasets`), so a link can open a tab directly
(for example `index.html#examples`) and the browser's back button returns to the previous tab. The tabs also work
with the keyboard (arrow keys, Home and End). On a phone, a wide table scrolls sideways inside its card.

The examples are a table with one row per clip: play the audio and compare the acted emotion (ground truth) and the
listeners' vote with the chosen model's prediction and confidence; a ✓ marks the labels the prediction matches (for a
composite label such as JL-Corpus's *excited*, any class it accepts), and each row expands to show the model's class
probabilities and every model's prediction for the clip. The models share the clips: for each dataset, every model
adds up to `correct` clips it got right and `wrong` clips it got wrong (judged on the acted emotion), so switching the
model keeps the same clips. You can switch between the two ways of counting a prediction as correct, and filter by
result (all, correct, wrong) and by ground-truth emotion. The script prints the seed it used, so you can rebuild the
same page later. `link_audio: true` links to the audio files instead of embedding them (a much smaller page that only
works inside this repository).

The page is built from the files of the evaluation, so it grows with the project: every `reports/<model>/summary.json`
and every `outputs/<model>/*_predictions.csv` (with its JSON file) appear in it, and new models and datasets appear
automatically. A new evaluation script only has to write the same kind of files: a CSV with `audio_path`,
`prediction`, `correct`, a ground-truth column and optional `p_<class>` probabilities, plus a JSON file with
`dataset`, `key`, `model`, the name of the ground-truth column and, for composite labels, `accept` (a model without a
`summary.json` gets example clips but no results). The docstring of `scripts/viewer/make_viewer.py` lists the exact
format.

### 7.1 Publish the page with GitHub Pages

The same run writes `site/index.html`, the page for the public: the same results, with embedded audio, but only clips
of datasets whose licence allows putting them on a public web page (`public_audio` in `SOURCES`). ESD's licence forbids
giving the data to anyone, so its clips never go on the public page, even if it is evaluated again; a dataset that
is not in `SOURCES` gets no clips either. The workflow `.github/workflows/pages.yml`, at the repository root, publishes
only this `site/` folder.

To turn it on, once: in the repository's Settings > Pages, set Source to "GitHub Actions". After that, every push to
`main` that changes `emotion_recognition/site/` publishes the page at
https://davidlequnchen.github.io/emotion_recognition/ (the workflow can also be run by hand from the Actions tab).
Everything in `site/` becomes public, so look at the page before pushing it. Each rebuild picks new clips and adds
the whole page (about 2.5 MB with one model) to the git history; set `viewer.seed` to keep the same clips (a
dataset's clips depend only on the seed and on that dataset's results).

## 8. Audio LLMs (in progress)

`scripts/llm_eval/` will evaluate two audio LLMs, GPT Realtime (`gpt-realtime`, `gpt-realtime-2.1`) and Google
Gemini, on the same clips and labels as the speech emotion models: each clip is sent with a prompt that asks for one
of the seven emotions (angry, disgusted, fearful, happy, neutral, sad, surprised), the same list for every dataset.
The parts that call no API are ready: the prompts (`prompts.yaml`, with a system and a user prompt), a data loader
that takes the same clips as the evaluation (`llm.limit`, default 200 per dataset, sampled like `evaluation.limit`)
and converts each clip to the format of the API (24 kHz 16-bit mono PCM for Realtime, a 16 kHz WAV file for Gemini),
and a parser for the answers. `python scripts/llm_eval/data_loader.py` previews the first clips of each dataset
without calling any API; the settings are in the `llm` section of `config/config.yaml`. The LLMs choose from all
seven emotions, so the emotion2vec+ number to compare with is its accuracy on all model classes (in each report), not
the main result.
A dataset whose licence forbids giving its audio to others (ESD) is never sent to an API. Details:
[scripts/llm_eval/README.md](scripts/llm_eval/README.md). Quick checks of the OpenAI key and the Realtime API are
in [test_scripts/](test_scripts/README.md).

## Models

Five models, each run as released (no fine-tuning) by its loader in `scripts/models/`, from its folder in `models/`
only. [models/README.md](models/README.md) lists their files, sizes and licence obligations;
[docs/models_survey.md](docs/models_survey.md) (English: [docs/models_survey_en.md](docs/models_survey_en.md)) compares
them with other open-source emotion recognition models.

### emotion2vec+ large

More detail, in Chinese and English: [docs/emotion2vec_plus_large.md](docs/emotion2vec_plus_large.md) (architecture
diagram, size, how it was trained) and [docs/finetuning_emotion2vec_plus_large.md](docs/finetuning_emotion2vec_plus_large.md)
(how much data and GPU time fine-tuning would need, and which layers to train).

- Paper: *emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation* ([arXiv:2312.15185](https://arxiv.org/abs/2312.15185));
  code: [ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec).
- Fine-tuned on about 42,500 hours of pseudo-labelled speech emotion data.
- 162.1M parameters (counted in `model.pt`; the model card says about 300M). `model.pt` is 1.95 GB because it also
  holds the optimizer state; the weights alone are 0.65 GB.
- 9 output classes (`tokens.txt`): angry, disgusted, fearful, happy, neutral, other, sad, surprised, `<unk>`.
- Input: audio, converted to 16 kHz mono. Output: one probability per class for the whole utterance.
- Runtime: [FunASR](https://github.com/modelscope/FunASR) builds the PyTorch network from `config.yaml` and loads
  `model.pt` (loader `emotion2vec.py`).
- License: see the model card (`models/emotion2vec_plus_large/README.md`, "model-license" from FunASR).

### The four new models (added 2026-10-28)

- **MERaLiON-SER-v1** ([MERaLiON/MERaLiON-SER-v1](https://huggingface.co/MERaLiON/MERaLiON-SER-v1)): Whisper-medium
  encoder with an ECAPA-TDNN head, 313.4M parameters in use; exactly the project's 7 classes. MERaLiON Public Licence
  (MIT with attribution terms). Loader `meralion_ser.py`. It pads every clip to 30 s and pools over the whole window,
  padding included (its official behaviour). CREMA-D is in its training data.
- **Melati** ([Scicom-intl/Melati-Emotion-Whisper-Large-V3](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3)):
  Whisper-large-v3 encoder with a small head, 637.3M; the project's 7 classes. Non-commercial research only. Loader
  `melati.py`. The prediction is the raw most probable class: the official thresholds, set for tagging podcasts, are
  not applied. JL-Corpus is in its training data.
- **Vox-Profile Whisper** ([tiantiaf/whisper-large-v3-msp-podcast-emotion](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion)):
  Whisper-large-v3 encoder with a head, 639.9M; 9 classes (the project's 7, contempt, other); trained on MSP-Podcast
  only. OpenRAIL weights, whose card excludes commercial use and surveillance. Loader `vox_profile.py`, a
  re-implementation of the official code, under that code's licence (`scripts/models/vox_profile_LICENSE.txt`). Its
  LoRA has no effect (LoRA B is all zeros), so the encoder is plain openai/whisper-large-v3 and only the head is
  trained.
- **Vox-Profile WavLM** ([tiantiaf/wavlm-large-categorical-emotion](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion)):
  WavLM-large with a head, 316.2M; the same classes, training data, licence and loader. Both Vox-Profile cards say
  that the models were trained on 3–15 s clips and that shorter clips give unreliable predictions, but 80–100% of the
  clips in CREMA-D, MCAE-SPPS, MESS, TESS and JL-Corpus are shorter than 3 s.

Each new loader was checked against the official code or the model card's example: the largest probability difference
is 0 for Melati, 2.4e-6 for Vox-Profile and 5.4e-4 for MERaLiON-SER-v1 (from resampling only). MPS and CPU give the
same top class on every test clip, and a simulated server (only the committed files, offline, an empty Hugging Face
cache) loads and runs all four.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `MPS ... available=False` on the Mac | Python is x86_64 (Rosetta). `platform.machine()` must print `arm64`: reinstall Miniconda for Apple Silicon. |
| `CUDA ... available=False` on the server | `torch.__version__` must end in `+cu121`: reinstall with `requirements-cuda121.txt`. Otherwise the driver is too old: check `nvidia-smi`. |
| `ERROR: .../model.pt not found` | emotion2vec+ large is not downloaded: run `python scripts/common/download_model.py` (section 2). |
| `ERROR: ...safetensors not found`, or `not ready ... still in parts` from `evaluate.py` | The weights came through git in parts: run `python scripts/common/join_parts.py` (after `git pull`). On a machine that reaches Hugging Face, `download_model.py` also works. |
| `parts are missing` or `SHA-256 does not match` from `join_parts.py` | The parts are incomplete: run `git pull` again, then the script again. |
| `module 'torch' has no attribute 'accelerator'` | transformers is newer than 5.18.0, which is the newest release that works with PyTorch 2.5.1 (5.19.0 calls `torch.accelerator`, new in PyTorch 2.6). Run `pip install -r requirements.txt` again to get 5.18.0. |
| `ERROR: no audio files found` | Run `python scripts/common/download_data.py`. |
| `ERROR: ... file(s) failed` from `download_data.py` | A network error, or GitHub sent an error page instead of the audio (for example a temporary LFS bandwidth limit). Run the same command again: correct files are skipped. |
| Download from Hugging Face is slow or blocked | Set `download.hub: ms` in `config/config.yaml` and run `python scripts/common/download_model.py` again. Only emotion2vec+ large is on ModelScope; the other models also come through git (section 2). |
| `Notice: ffmpeg is not installed ...` | Printed by FunASR on machines without ffmpeg. Harmless: wav, flac, mp3 and ogg are read without it. To silence it: `conda install -c conda-forge ffmpeg`. |
