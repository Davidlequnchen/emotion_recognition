# emotion_recognition

Two independent speech projects, each in its own folder with its own README, configuration, requirements and scripts.
Run each project's commands from its own folder.

| Folder | Project |
|---|---|
| [`emotion_recognition/`](emotion_recognition/README.md) | **Speech emotion recognition** (the main project): five open-source models (emotion2vec+ large, MERaLiON-SER-v1, Melati and two Vox-Profile models), run with PyTorch on an Apple Silicon Mac (MPS) or an NVIDIA server (CUDA 12.1) and evaluated by one script on emotional speech datasets in English and Mandarin, with a results page in the browser. |
| [`antideepfake/`](antideepfake/README.md) | **Deepfake detection** (a side project): XLS-R-1B-AntiDeepfake from NII Yamagishi Lab tells real speech from synthetic or converted speech; the model is defined in plain PyTorch and its checkpoint comes from Zenodo. |

The public results page of the emotion recognition project, https://davidlequnchen.github.io/emotion_recognition/,
is built into `emotion_recognition/site/` and published by `.github/workflows/pages.yml`.
