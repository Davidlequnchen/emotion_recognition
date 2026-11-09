# emotion_recognition

Two independent speech projects, each in its own folder with its own README, configuration, requirements and scripts.
Run each project's commands from its own folder.

For the emotion recognition models, clone the metadata submodule and run its Release setup once:

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Davidlequnchen/emotion_recognition.git
cd emotion_recognition
python3 emotion_recognition/scripts/common/setup_models.py
```

The main repository pins [emotion_recognition_models](https://github.com/Davidlequnchen/emotion_recognition_models)
at `emotion_recognition/models/`. Git clones the small model files; the setup command downloads and verifies the
model weights from GitHub Releases. By default it gets all eight models (about 64 GB); add model keys to get fewer.
The shallow-clone flags avoid downloading the old model-weight history; omit them if you need the full Git history.
For an existing clone, run `git pull` and the same setup command. Existing model files are preserved and valid
weights are reused. Full environment instructions are in the project's README below.

Datasets follow the same pattern at `emotion_recognition/data/`, through the pinned
[emotion_recognition_datasets](https://github.com/Davidlequnchen/emotion_recognition_datasets) submodule:

```bash
python3 emotion_recognition/scripts/common/setup_datasets.py
```

This restores the nine mirrored datasets from Releases. CREMA-D, MESS, RAVDESS and Emozionalmente retain their
original download sources; use `download_data.py --dataset NAME --source` as described in the project README.
Existing data directories are preserved, verified files are reused, and local external datasets remain accessible.
If pulling this migration removes files that used to be tracked by the main repository, setup recovers matching
files from old Git objects already on that computer, including nEMO's old Parquet parts. Shallow clones do not
fetch old history for this recovery; missing files use Releases.
Neither model nor dataset setup uses Git LFS.

| Folder | Project |
|---|---|
| [`emotion_recognition/`](emotion_recognition/README.md) | **Speech emotion recognition** (the main project): five open-source models (emotion2vec+ large, MERaLiON-SER-v1, Melati and two Vox-Profile models), run with PyTorch on an Apple Silicon Mac (MPS) or an NVIDIA server (CUDA 12.1) and evaluated by one script on emotional speech datasets in English and Mandarin, with a results page in the browser. |
| [`antideepfake/`](antideepfake/README.md) | **Deepfake detection** (a side project): XLS-R-1B-AntiDeepfake from NII Yamagishi Lab tells real speech from synthetic or converted speech; the model is defined in plain PyTorch and its checkpoint comes from Zenodo. |

The public results page of the emotion recognition project, https://davidlequnchen.github.io/emotion_recognition/,
is built into `emotion_recognition/site/` and published by `.github/workflows/pages.yml`.
