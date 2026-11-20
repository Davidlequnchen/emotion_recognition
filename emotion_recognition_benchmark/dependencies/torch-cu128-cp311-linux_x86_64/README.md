# PyTorch CUDA 12.8 offline wheelhouse

This bundle targets CPython 3.11 on Linux x86_64 with glibc 2.28 or newer. On a B200, check `nvidia-smi` for
an NVIDIA driver that supports CUDA 12.8 or newer (570 series or newer).

The 30 wheels total about 3.89 GB. They include `torch==2.7.1+cu128`, `torchaudio==2.7.1+cu128`,
`torchvision==0.22.1+cu128`, and their runtime
dependencies, including Triton 3.3.1. It also includes the optional certificate helper `truststore`.
[`manifest.json`](manifest.json) lists the versions, sources, and SHA-256 values. See [the installation order](INSTALL_ORDER.md)
for the individual wheel commands.

## Put pip temporary files on `/aip`

Create a temporary directory on your home filesystem once.

```bash
mkdir -p "$HOME/pip-tmp"
```

Add this line to `~/.bash_profile` for future Bash login sessions. Run the same line in your current terminal to
apply it immediately.

```bash
export TMPDIR="$HOME/pip-tmp"
```

Check the temporary directory and pip cache.

```bash
python -c 'import tempfile; print(tempfile.gettempdir())'
python -m pip cache dir
```

On the company server, the default pip cache is already under `$HOME/.cache/pip` on `/aip`. `TMPDIR` controls
where pip places temporary files while downloading and installing packages.

## Download the wheels

Git tracks the scripts, manifests, and documentation. The downloader fetches all 30 wheels from the CUDA 12.8
GitHub Release and checks each file's size and SHA-256. Rerun the downloader to resume interrupted downloads and
reuse verified files. [CUDA 12.8 Release](https://github.com/Davidlequnchen/emotion_recognition/releases/tag/deps-torch-cu128-cp311-linux-x86_64-v1)

From the repository root, update your `main` checkout and run the downloader.

```bash
git pull --ff-only
cd emotion_recognition_benchmark
python dependencies/torch-cu128-cp311-linux_x86_64/download.py
```

Use `--source upstream` to download directly from PyTorch and PyPI. If the server cannot reach the download sources,
run the downloader on a connected machine and copy the `.whl` files
into this directory in the server's repository checkout. The scripts use helpers from the neighboring CUDA 11.8
directory in that checkout. To use a company CA bundle, add `--ca-bundle /path/to/company-ca.pem` to the download
command. Verify transferred files offline with `--verify-only`.

```bash
python dependencies/torch-cu128-cp311-linux_x86_64/download.py --verify-only
```

## Install on the server

From `emotion_recognition_benchmark/`, activate the target Conda environment and run the installer.

```bash
conda activate emotion_recognition_llm
python dependencies/torch-cu128-cp311-linux_x86_64/install.py
```

The installer checks every wheel and the dependency order before installing one local wheel at a time. It uses
no package index and finishes with `pip check`. Add `--dry-run` to print the commands.

Check that PyTorch reports CUDA 12.8 and can see the GPU, then install the benchmark dependencies, including Accelerate.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements.txt
```
