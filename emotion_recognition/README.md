# emotion_recognition

Exploring speech emotion recognition (SER), from traditional machine learning to deep learning.
Current focus: **emotion2vec+ large** ([`emotion2vec_plus_large`](https://huggingface.co/emotion2vec/emotion2vec_plus_large)),
run with PyTorch on two kinds of machines:

| Mode | Machine | Device |
|---|---|---|
| `mac` | MacBook with Apple Silicon (tested on M4 Max, 48 GB) | Apple GPU via Metal (`mps`) or `cpu` |
| `cuda` | Linux server with NVIDIA GPU(s), CUDA 12.1 | `cuda:0` (or another GPU index) |

## Quick start

```bash
conda create -n emotion_recognition python=3.11 -y
conda activate emotion_recognition
pip install -r requirements-mac.txt        # NVIDIA server: pip install -r requirements-cuda121.txt
pip install -r requirements.txt
python scripts/download_model.py           # model files -> models/emotion2vec_plus_large/
python scripts/inference.py                # NVIDIA server: first set `mode: cuda` in config/config.yaml
# optional (first set `download.dataset: all` in config/config.yaml): the evaluation datasets -> data/,
# the model's accuracy on each -> reports/, and a web page with the results and examples -> outputs/viewer.html
python scripts/download_data.py
python scripts/evaluate.py
python scripts/make_viewer.py
```

The scripts have no command-line options: every setting is in [`config/config.yaml`](config/config.yaml), one
section per script. Each step is explained below.

## Repository layout

```
emotion_recognition/
├── .github/workflows/pages.yml # publishes site/ with GitHub Pages
├── config/config.yaml          # hardware mode (mac | cuda), devices, model registry, audio path, viewer settings
├── data/
│   ├── samples/                # six CREMA-D test clips (in git, 0.4 MB)
│   ├── crema_d/                # the full CREMA-D dataset (optional download, not in git)
│   ├── mcae_spps/ thai_ser/ nemo/ mess/ esd/   # more datasets like CREMA-D (optional downloads, not in git)
│   └── ravdess/ tess/ jl_corpus/               # three more English ones (optional downloads, not in git)
├── docs/
│   ├── crema_d.md              # CREMA-D explained in Chinese: file names, labels, every column of the label files
│   ├── datasets_survey.md      # survey (in Chinese) of datasets like CREMA-D: parallel text, good labels, larger
│   └── datasets_survey_en.md   # the same survey in English
├── models/
│   ├── README.md               # what each model file is, how to add models
│   └── emotion2vec_plus_large/ # model.pt, config.yaml, configuration.json, tokens.txt (downloaded, not in git)
├── outputs/                    # per-clip predictions and the example viewer page (not in git)
├── reports/                    # evaluation reports, e.g. crema_d_emotion2vec_plus_large.md
│   └── figures/                # figures used in the reports
├── scripts/
│   ├── download_model.py       # downloads a model from HuggingFace or ModelScope
│   ├── download_data.py        # downloads CREMA-D (test clips or full) and eight more datasets, checks every file
│   ├── join_parts.py           # joins dataset files committed in 90 MiB parts, checks their SHA-256
│   ├── eval_datasets.py        # one loader per evaluation dataset (labels -> emotion2vec+ classes)
│   ├── evaluate.py             # runs the model on each dataset, writes the reports and a summary
│   ├── make_viewer.py          # builds a web page with the results and random right and wrong examples
│   ├── templates/viewer.html   # the page template used by make_viewer.py
│   └── inference.py            # step-by-step inference
├── site/index.html             # the public results page (built by make_viewer.py, published by GitHub Pages)
├── requirements.txt            # common packages (both machines)
├── requirements-mac.txt        # PyTorch for Apple Silicon (MPS + CPU)
└── requirements-cuda121.txt    # PyTorch for NVIDIA GPUs, CUDA 12.1
```

Git tracks the code, configuration, documentation and the six small test clips. Model files and the full dataset
are in `.gitignore` (`model.pt` alone is 1.9 GB). On every new machine, re-create them with the two download scripts.
For now there are exceptions, for a machine that cannot reach the hosts of these datasets: the unpacked MCAE-SPPS files
(without its zips), the TESS files, the JL-Corpus recordings (`JL(wav+txt)/`; not the rest of its zip, see
`data/jl_corpus/README.md`), and the nEMO Parquet files in 90 MiB parts (GitHub refuses files over 100 MB); after
`git pull`, `python scripts/join_parts.py` joins and checks them.

## 1. Set up the environment

Tested with Python 3.11, PyTorch 2.5.1 and FunASR 1.4.16 on the M4 Max (MPS and CPU). The Linux + CUDA 12.1
installation was checked in a Linux x86_64 container without a GPU.

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

Run both commands from the repository root: first PyTorch for your hardware, then the common packages.

**MacBook (Apple Silicon)**

```bash
pip install -r requirements-mac.txt     # torch 2.5.1 + torchaudio 2.5.1 from PyPI (MPS + CPU)
pip install -r requirements.txt         # FunASR, transformers, hubs, numpy, pyyaml, soundfile
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
- **transformers is pinned to 5.19.0**: FunASR imports it, and 5.20+ requires PyTorch 2.6.

### 1.4 Check PyTorch

```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

Expected output: on the Mac, `2.5.1 | CUDA: False | MPS: True`. On the server, `2.5.1+cu121 | CUDA: True | MPS: False`.

## 2. Download the model

```bash
python scripts/download_model.py   # from HuggingFace; set `download.hub: ms` for ModelScope (faster in mainland China)
```

This fills `models/emotion2vec_plus_large/` with `model.pt` (1.9 GB of weights), `config.yaml`,
`configuration.json`, `tokens.txt`, the model card and an example clip; see [models/README.md](models/README.md).
The "unauthenticated requests" warning from HuggingFace is harmless: the model is public and needs no token.

## 3. Download the data (CREMA-D)

```bash
python scripts/download_data.py
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
emotion changes between files. `scripts/inference.py` uses them by default. Names follow
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
1.3 to 5.0 s (5.3 hours in total). To evaluate the model on every clip, see section 6.

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

### 3.4 More datasets like CREMA-D

The same script downloads eight more datasets that also record the same sentences in different emotions.
They were chosen from the survey in [docs/datasets_survey_en.md](docs/datasets_survey_en.md) (Chinese:
[docs/datasets_survey.md](docs/datasets_survey.md)), which has links, sizes and caveats for many more.

```bash
python scripts/download_data.py   # with `download.dataset: mcae_spps` (or thai_ser, nemo, mess, esd, ravdess, tess,
                                  # jl_corpus; all = everything)
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
together (reading them needs `pyarrow`); `scripts/evaluate.py` extracts the audio it needs into `audio/` on its first run.
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

Every script reads `mode` from this file; scripts that run the model use the device of that mode.

## 5. Run inference

```bash
python scripts/inference.py
```

Settings in `config/config.yaml`: `mode` and `devices` choose the hardware (set `devices.mac: cpu` to force the CPU),
and `inference.audio` the audio file or folder to run on (wav, flac, mp3, ogg; default `data/samples`).

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
- The `parameters: 164.0 M` line counts the network built from `config.yaml`: 162 M weights loaded from `model.pt`,
  plus a 2 M pre-training decoder that inference never uses. The model card says "~300M"; the released checkpoint
  holds 162 M weights.

## 6. Evaluate on the datasets

```bash
python scripts/evaluate.py
```

Settings, in the `evaluation` section of `config/config.yaml`:

```yaml
evaluation:
  datasets: all     # all downloaded datasets, or a list such as [crema_d, nemo]
  exclude: [esd, thai_ser]  # never evaluated, even with `all`, and left out of the summary
  limit: null       # only N clips of each dataset, taken at even intervals within each emotion (quick test or
                    # sample); null = every clip
  overwrite: false  # false: skip a dataset whose results (same model and limit) are already in outputs/
  devices:          # device per mode for evaluation
    mac: cpu        # on the Mac the CPU is faster than MPS here (see below)
    cuda: cuda:0
  summary_figure:   # every dataset's confusion matrix in one figure, in the summary report
    order: [crema_d, mcae_spps, nemo, mess]  # panels row by row; datasets not listed come last
    acted_or_listener: [crema_d]  # these show the matrix where the listeners' vote also counts as correct;
                                  # the others show the acted emotion
```

Each dataset has a loader in `scripts/eval_datasets.py` that reads its labels and maps them to emotion2vec+ classes;
to evaluate on a new dataset, add a loader there. For each dataset the script writes:

- `reports/<dataset>_emotion2vec_plus_large.md`: accuracy, results per emotion, confusion matrices (figures in
  `reports/figures/` and tables), comparison with human listeners where the dataset has their labels, and caveats.
- `outputs/<dataset>_emotion2vec_plus_large_predictions.csv`: one row per clip (not in git) with the audio path, the
  labels, the model's prediction, whether it is correct, and all class probabilities. The JSON file next to it
  describes the columns and holds the metrics; [docs/crema_d.md](docs/crema_d.md) explains the CREMA-D columns in Chinese.
- [`reports/emotion2vec_plus_large_summary.md`](reports/emotion2vec_plus_large_summary.md): every dataset side by side,
  every dataset's confusion matrix in one figure (`reports/figures/emotion2vec_plus_large_summary_confusion.png`), and
  a comparison on the emotions that every dataset has.

A dataset whose results for the same model and `limit` are already in `outputs/` is skipped, so after adding a
dataset the script evaluates only the new one; set `overwrite: true` to evaluate again. The summary always covers
every dataset with results in `outputs/`, also those not listed in `datasets`; for a dataset with a full run and a
sample, it shows the run with more clips.

The datasets name their emotions differently, and emotion2vec+ has nine output classes: angry, disgusted, fearful,
happy, neutral, other, sad, surprised and `<unk>`. Each loader maps its dataset's labels to these classes:

| Dataset | Emotions in the dataset (where the label is) | Mapped to emotion2vec+ | Model classes not used |
|---|---|---|---|
| CREMA-D | 6, code in the file name: `ANG` `DIS` `FEA` `HAP` `NEU` `SAD` | ANG → angry, DIS → disgusted, FEA → fearful, HAP → happy, NEU → neutral, SAD → sad | surprised, other, `<unk>` |
| MCAE-SPPS | 7, second digit of the file name: 1 Neutral, 2 Happiness, 3 Anger, 4 Fear, 5 Sadness, 6 Disgust, 7 Surprise | Neutral → neutral, Happiness → happy, Anger → angry, Fear → fearful, Sadness → sad, Disgust → disgusted, Surprise → surprised | other, `<unk>` |
| THAI-SER (excluded for now) | 5, column `assigned_emo`: Neutral, Angry, Happy, Sad, Frustrated | Neutral → neutral, Angry → angry, Happy → happy, Sad → sad; **Frustrated has no match, so its 5,270 clips are left out** | disgusted, fearful, surprised, other, `<unk>` |
| nEMO | 6, column `emotion`: neutral, anger, fear, happiness, sadness, surprised | neutral → neutral, anger → angry, fear → fearful, happiness → happy, sadness → sad, surprised → surprised | disgusted, other, `<unk>` |
| MESS | 4, first letter of the file name: `A`ngry, `C`alm, `H`appy, `S`ad | A → angry, **C (calm) → neutral, the closest class**, H → happy, S → sad | disgusted, fearful, surprised, other, `<unk>` |
| ESD (excluded for now) | 5, folder name: Neutral, Angry, Happy, Sad, Surprise (the Mandarin transcripts say 中立, 生气, 快乐, 伤心, 惊喜) | Neutral → neutral, Angry → angry, Happy → happy, Sad → sad, Surprise → surprised | disgusted, fearful, other, `<unk>` |
| RAVDESS | 8, third number of the file name: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised | neutral, happy, sad, angry, fearful and surprised keep their names, disgust → disgusted; **calm has no match, so its 192 clips are left out** | other, `<unk>` |
| TESS | 7, last word of the file name: angry, disgust, fear, happy, neutral, ps, sad | angry, happy, neutral and sad keep their names, disgust → disgusted, fear → fearful, **ps (pleasant surprise) → surprised** | other, `<unk>` |
| JL-Corpus | 10, second part of the file name: angry, excited, happy, neutral, sad (primary); anxious, apologetic, assertive, concerned, encouraging (secondary) | angry, happy, neutral and sad keep their names; **excited and the five secondary emotions have no match, so their 1,440 clips are left out** | disgusted, fearful, surprised, other, `<unk>` |

How the nine probabilities become a prediction:

- **Main result** (*accuracy: acted emotion*): the prediction is the most probable of the dataset's own emotions; the
  other classes, including `other` and `<unk>`, are ignored. On MESS, for example, only angry, happy, neutral and
  sad compete, so chance level is 25%.
- **All 9 classes** (in each report): the prediction is the most probable of all nine classes, so choosing a class
  that the dataset does not have (for example surprised on CREMA-D) counts as wrong.
- **Listeners' votes** (CREMA-D, MCAE-SPPS, THAI-SER) use the same mapping; for a tied vote, any of the tied emotions
  counts.
- Some labels do not match exactly. MESS's *calm* is scored as *neutral*. Clips of emotions that the model does not
  have are left out: THAI-SER's *frustrated* (and a listeners' vote of *frustrated* can never match), RAVDESS's
  *calm*, and JL-Corpus's *excited* and secondary emotions. TESS's *pleasant surprise* and ESD's Mandarin label for
  Surprise, 惊喜, both mean a pleasant surprise, while the model's class is called 吃惊 (surprised).

Results:

| Dataset | Language | Clips | Emotions (chance) | Accuracy: acted emotion | Accuracy: acted or listeners' vote | In EmoBox |
|---|---|---|---|---|---|---|
| [CREMA-D](reports/crema_d_emotion2vec_plus_large.md) | English | 7,442 | 6 (16.7%) | **86.5%** | 89.6% | yes |
| [MCAE-SPPS](reports/mcae_spps_emotion2vec_plus_large.md) | Mandarin | 6,675 | 7 (14.3%) | **61.3%** | 61.3% | no |
| [nEMO](reports/nemo_emotion2vec_plus_large.md) | Polish | 4,481 | 6 (16.7%) | **52.0%** | – | no |
| [MESS](reports/mess_emotion2vec_plus_large.md) | English | 1,800 | 4 (25.0%) | **40.8%** | – | no |

*Acted or listeners' vote* also accepts the listeners' majority vote; it exists only for datasets that publish
listeners' labels (for MCAE-SPPS the two agree by construction, because only clips that listeners recognised were
kept). CREMA-D is in EmoBox, which emotion2vec+ was trained from, so its result is optimistic; the other three are a
more realistic picture.

ESD and THAI-SER are left out of the evaluation for now (`exclude`): ESD's licence is research-only and forbids
sharing it, it has no listener test, and it is in EmoBox; THAI-SER is in Thai, which matters less for this project.
Their data stays in `data/`; their earlier reports are in [reports/archive/](reports/archive/) (ESD on a 3,500-clip
sample: 97.0%; THAI-SER: 63.2%) and their predictions in `outputs/archive/`. Remove a dataset from `exclude` to
evaluate it again.

RAVDESS, TESS and JL-Corpus have loaders, checked on 20 clips each, but are not evaluated in full yet. To evaluate
only them, set `datasets: [ravdess, tess, jl_corpus]`; they are in EmoBox, so expect optimistic results.

The datasets have different emotions, and some emotions are much harder than others. On the four emotions that every
dataset has (angry, happy, neutral and sad; chance 25%), counting only clips of these emotions and taking the most
probable of the four as the prediction (also in the [summary](reports/emotion2vec_plus_large_summary.md)):

| Dataset | Language | Clips | Accuracy, 4 shared emotions | In EmoBox |
|---|---|---|---|---|
| [CREMA-D](reports/crema_d_emotion2vec_plus_large.md) | English | 4,900 | **93.2%** | yes |
| [MCAE-SPPS](reports/mcae_spps_emotion2vec_plus_large.md) | Mandarin | 4,297 | **87.8%** | no |
| [nEMO](reports/nemo_emotion2vec_plus_large.md) | Polish | 3,076 | **67.8%** | no |
| [MESS](reports/mess_emotion2vec_plus_large.md) | English | 1,800 | **40.8%** | no |

CREMA-D, which is in EmoBox, still comes first, but MCAE-SPPS, which is not, is close behind: its lower overall result
comes from its three other emotions (recall: fearful 8.5%, disgusted 8.5%, surprised 28.6%). nEMO and MESS differ
from CREMA-D in more than training data: language (Polish), recording conditions (at home) and, for MESS, *calm*
scored as *neutral*.

Speed on the M4 Max: for these evaluations the CPU is faster than MPS, because nearly every clip has a different
length and MPS compiles new kernels for every new length (THAI-SER, now archived: about 140 ms per clip on the CPU,
about 235 ms on MPS). MPS is fast only for lengths it has seen before (CREMA-D, run many times: 18 ms per clip). MPS also cannot
process clips longer than about 20 s; the scripts run those on the CPU automatically. On the CPU nEMO takes about
5 minutes and MESS 12 minutes (its 24,414 Hz audio is slow to resample; TESS has the same rate).

## 7. Results and examples in the browser

```bash
python scripts/make_viewer.py
```

Settings, in the `viewer` section of `config/config.yaml`: `correct` and `wrong` (how many clips of each kind,
default 10 and 10), `seed` (the same seed always picks the same clips), `predictions` (`all`, or a list of
predictions files), `exclude` (datasets whose clips are left out; default ESD and THAI-SER), `link_audio`, `out`
(the page for this machine) and `site` (the public page, see 7.1).

The script writes one self-contained page, `outputs/viewer.html` (audio embedded: about 13 MB with 20 clips from
each of the four evaluated datasets; open it in any browser). It has three parts:

- **Results**: the summary of every evaluated dataset, as in
  [`reports/emotion2vec_plus_large_summary.md`](reports/emotion2vec_plus_large_summary.md) (read from the JSON file
  next to it): the results table, every dataset's confusion matrix (click the figure to enlarge it) and the comparison
  on the emotions that every dataset has. Click a dataset to jump to its examples.
- **Examples**: random clips the model got right and wrong (below).
- **Datasets and licences**: the source, licence and credit of each dataset (`SOURCES` in `scripts/eval_datasets.py`).

The examples are a table with one row per clip: play the audio and compare the acted emotion (ground truth) and the listeners' vote with the model's
prediction and confidence; a ✓ marks the labels the prediction matches, and each row expands to show every class
probability. You can switch between the two ways of counting a prediction as correct, and filter by result (all,
correct, wrong) and by ground-truth emotion. Clips are picked using the first way (acted emotion). The script prints the seed it
used, so you can rebuild the same page later. `link_audio: true` links to the audio files instead of embedding them
(a much smaller page that only works inside this repository).

The page is built from the predictions files, so it grows with the project. Every `outputs/*_predictions.csv` (with
its JSON file) becomes one dataset-and-model run that you can switch between in the page. A new evaluation script
only has to write the same kind of files: a CSV with `audio_path`, `prediction`, `correct`, a ground-truth column and
optional `p_<class>` probabilities, plus a JSON file with `dataset`, `model` and the name of the ground-truth column.
The docstring of `scripts/make_viewer.py` lists the exact format.

### 7.1 Publish the page with GitHub Pages

The same run writes `site/index.html`, the page for the public: the same results, with embedded audio, but only clips
of datasets whose licence allows putting them on a public web page (`public_audio` in `SOURCES`). ESD's licence forbids
giving the data to anyone, so its clips never go on the public page, even if it is evaluated again; a dataset that
is not in `SOURCES` gets no clips either. The workflow `.github/workflows/pages.yml` publishes only the `site/` folder.

To turn it on, once: in the repository's Settings > Pages, set Source to "GitHub Actions". After that, every push to
`main` that changes `site/` publishes the page at https://davidlequnchen.github.io/emotion_recognition/ (the workflow
can also be run by hand from the Actions tab). Everything in `site/` becomes public, so look at the page before
pushing it. Each rebuild picks new clips and adds the whole page (about 20 MB) to the git history; set `viewer.seed`
to keep the same clips.

## Model: emotion2vec+ large

- Paper: *emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation* ([arXiv:2312.15185](https://arxiv.org/abs/2312.15185));
  code: [ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec).
- Fine-tuned on about 42,500 hours of pseudo-labelled speech emotion data.
- 9 output classes (`tokens.txt`): angry, disgusted, fearful, happy, neutral, other, sad, surprised, `<unk>`.
- Input: audio, converted to 16 kHz mono. Output: one probability per class for the whole utterance.
- Runtime: [FunASR](https://github.com/modelscope/FunASR) builds the PyTorch network from `config.yaml` and loads `model.pt`.
- License: see the model card (`models/emotion2vec_plus_large/README.md`, "model-license" from FunASR).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `MPS ... available=False` on the Mac | Python is x86_64 (Rosetta). `platform.machine()` must print `arm64`: reinstall Miniconda for Apple Silicon. |
| `CUDA ... available=False` on the server | `torch.__version__` must end in `+cu121`: reinstall with `requirements-cuda121.txt`. Otherwise the driver is too old: check `nvidia-smi`. |
| `ERROR: .../model.pt not found` | Run `python scripts/download_model.py`. |
| `ERROR: no audio files found` | Run `python scripts/download_data.py`. |
| `ERROR: ... file(s) failed` from `download_data.py` | A network error, or GitHub sent an error page instead of the audio (for example a temporary LFS bandwidth limit). Run the same command again: correct files are skipped. |
| Download from HuggingFace is slow or blocked | Set `download.hub: ms` in `config/config.yaml` and run `python scripts/download_model.py` again. |
| `Notice: ffmpeg is not installed ...` | Printed by FunASR on machines without ffmpeg. Harmless: wav, flac, mp3 and ogg are read without it. To silence it: `conda install -c conda-forge ffmpeg`. |
