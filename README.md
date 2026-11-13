# emotion_recognition

Two independent speech projects, each in its own folder with its own README, configuration, requirements and scripts.
Run each project's commands from its own folder.

Label Studio guides: [中文操作手册](label_studio/README.md) · [English guide](label_studio/README_en.md).

For the emotion recognition models, clone the metadata submodule and run its Release setup once:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Davidlequnchen/emotion_recognition.git
cd emotion_recognition
python3 emotion_recognition/scripts/common/setup_models.py
```

The main repository pins [emotion_recognition_models](https://github.com/Davidlequnchen/emotion_recognition_models)
at `emotion_recognition/models/`. Git clones the small model files; the setup command downloads and verifies the
model weights from GitHub Releases. By default it gets all eight models (about 64 GB); add model keys to get fewer.
The shallow-clone flags are optional: on 2026-11-09, historical data/model directory contents were removed from
the main repository's reachable history. Existing clones made before that rewrite should follow the
[history migration notes](emotion_recognition/docs/git_history_cleanup.md), preserving local results and work
before synchronising. Existing valid model files can be reused. Full environment instructions are below.

Datasets follow the same pattern at `emotion_recognition/data/`, through the pinned
[emotion_recognition_datasets](https://github.com/Davidlequnchen/emotion_recognition_datasets) submodule:

```bash
python3 emotion_recognition/scripts/common/setup_datasets.py
```

This restores the eleven mirrored datasets from Releases, including the retained DailyTalk and EmotionTalk
context-pilot candidates; the formal evaluation still has thirteen datasets. Add dataset keys to restore selected
mirrors only. CREMA-D, MESS, RAVDESS and Emozionalmente retain their
original download sources; use `download_data.py --dataset NAME --source` as described in the project README.
Existing data directories are preserved, verified files are reused, and local external datasets remain accessible.
Legacy clones may recover matching files from old Git objects already on that computer, including nEMO's old
Parquet parts. New clones and clones whose old objects were pruned use Releases; no old history is fetched.
Neither model nor dataset setup uses Git LFS.

| Folder | Project |
|---|---|
| [`emotion_recognition/`](emotion_recognition/README.md) | **Speech emotion recognition** (the main project): five SER models (emotion2vec+ large, MERaLiON-SER-v1, Melati and two Vox-Profile models) and three audio LLMs (Qwen2-Audio, Qwen2.5-Omni and Gemma 4), run in the `emotion_recognition_llm` PyTorch environment on an Apple Silicon Mac (MPS) or an NVIDIA server (CUDA 11.8), with one evaluation script and a results page in the browser. |
| [`antideepfake/`](antideepfake/README.md) | **Deepfake detection** (a side project): XLS-R-1B-AntiDeepfake from NII Yamagishi Lab tells real speech from synthetic or converted speech; the model is defined in plain PyTorch and its checkpoint comes from Zenodo. |

The public results page of the emotion recognition project, https://davidlequnchen.github.io/emotion_recognition/,
is built into `emotion_recognition/site/` and published by `.github/workflows/pages.yml`.
