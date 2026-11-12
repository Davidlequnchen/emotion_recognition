# PyTorch CUDA 11.8 wheels

The three official wheels pinned by [`requirements-llm-cuda118.txt`](../../requirements-llm-cuda118.txt),
downloaded on 2026-11-12 for **CPython 3.11, Linux x86_64, glibc 2.28 or newer**. These wheels do not run on
macOS, Windows, Linux ARM, or other Python versions.

| Package | Version | Official index |
|---|---|---|
| torch | 2.7.1+cu118 | [PyTorch](https://download.pytorch.org/whl/cu118/torch/) |
| torchaudio | 2.7.1+cu118 | [PyTorch](https://download.pytorch.org/whl/cu118/torchaudio/) |
| torchvision | 0.22.1+cu118 | [PyTorch](https://download.pytorch.org/whl/cu118/torchvision/) |

[`manifest.json`](manifest.json) records the exact source URLs, file sizes and upstream SHA-256 values.
[`SHA256SUMS`](SHA256SUMS) contains the same upstream hashes for verification. The wheel files are unchanged
from PyTorch's downloads and are mirrored in the
[`deps-torch-cu118-cp311-linux-x86_64-v1` GitHub Release](https://github.com/Davidlequnchen/emotion_recognition/releases/tag/deps-torch-cu118-cp311-linux-x86_64-v1),
along with `manifest.json` and `SHA256SUMS`. Git tracks the downloader, metadata and documentation; downloaded
wheels are ignored locally.

## Fetch the wheels

After cloning or pulling, run from the **`emotion_recognition/` project folder**:

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py
```

[`download.py`](download.py) uses Python's standard library to download the three Release wheels into this folder
and verify their sizes and SHA-256 hashes. Matching local wheels are reused; an incomplete or failed download
does not become a final wheel file. Git LFS is not required.

If GitHub Releases are unavailable, use the original PyTorch download URLs instead:

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py --source upstream
```

Run `python dependencies/torch-cu118-cp311-linux_x86_64/download.py --help` for the available options.

## Install on the NVIDIA server

Activate the project's Python 3.11 environment and run from the **`emotion_recognition/` project folder**:

```bash
conda activate emotion_recognition_llm
python -m pip install --index-url https://pypi.org/simple ./dependencies/torch-cu118-cp311-linux_x86_64/*.whl
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchaudio, torchvision; print(torch.__version__, torchaudio.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

Use this wheel installation in place of `pip install -r requirements-llm-cuda118.txt`. Expected versions are
`2.7.1+cu118`, `2.7.1+cu118` and `0.22.1+cu118`; CUDA availability requires a compatible NVIDIA driver and
an accessible GPU.

`nvidia-smi` reports the driver's supported CUDA version. A driver reporting CUDA 12.4 can run these CUDA 11.8
wheels through NVIDIA's [driver backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html);
it does not need to report 11.8 or have a separate 11.8 toolkit installed.

This folder caches **only these three packages**, not a complete offline environment. Pip still downloads
missing dependencies, including the NVIDIA CUDA runtime packages, Triton, NumPy and Pillow, from PyPI.
The common packages in `requirements.txt` also require internet access unless already cached or installed.
A separate CUDA toolkit is not required for the prebuilt wheels.
