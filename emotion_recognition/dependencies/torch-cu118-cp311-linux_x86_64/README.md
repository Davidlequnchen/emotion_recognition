# PyTorch CUDA 11.8 offline wheelhouse

This wheelhouse contains **27 wheels, about 2.93 GB**, targeting **CPython 3.11, Linux x86_64, glibc 2.28 or newer**:
the three packages pinned by [`requirements-llm-cuda118.txt`](../../requirements-llm-cuda118.txt), their 23 runtime
dependencies, and the optional `truststore` certificate helper. The full set must be installed in the matching
target environment; the download step can run on another machine with Python 3.9+.

| Package | Version | Official index |
|---|---|---|
| torch | 2.7.1+cu118 | [PyTorch](https://download.pytorch.org/whl/cu118/torch/) |
| torchaudio | 2.7.1+cu118 | [PyTorch](https://download.pytorch.org/whl/cu118/torchaudio/) |
| torchvision | 0.22.1+cu118 | [PyTorch](https://download.pytorch.org/whl/cu118/torchvision/) |

The runtime dependencies include the matching NVIDIA CUDA/cuDNN libraries, Triton, NumPy **2.4.6**, Pillow,
and the remaining packages required by this pinned stack. The complete list and dependency-first installation
sequence are in [`INSTALL_ORDER.md`](INSTALL_ORDER.md).

[`manifest.json`](manifest.json) records the exact PyTorch/PyPI source URLs, file sizes and upstream SHA-256 values.
[`SHA256SUMS`](SHA256SUMS) contains the same hashes for verification. The wheels are preserved unchanged
and mirrored in the
[`deps-torch-cu118-cp311-linux-x86_64-v1` GitHub Release](https://github.com/Davidlequnchen/emotion_recognition/releases/tag/deps-torch-cu118-cp311-linux-x86_64-v1),
along with `manifest.json`, `SHA256SUMS`, [`requirements-offline.txt`](requirements-offline.txt), and
`INSTALL_ORDER.md`. Git tracks the scripts, metadata and documentation; downloaded wheels are ignored locally.

This is the complete **default runtime dependency set for `requirements-llm-cuda118.txt`**, including the
certificate helper. Optional package extras and the rest of the project's common [`requirements.txt`](../../requirements.txt)
are outside this wheelhouse.

## Fetch the wheels

Use the scripts and manifest from a current **`main` checkout**, then run from the
**`emotion_recognition/` project folder**:

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py
```

[`download.py`](download.py) uses Python 3.9+ to download all 27 Release wheels into this folder
and verify their sizes and SHA-256 hashes. Matching local wheels are reused; an incomplete or failed download
does not become a final wheel file. Git LFS is not required.

If GitHub Releases are unavailable, use the original PyTorch/PyPI download URLs instead:

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py --source upstream
```

Run `python dependencies/torch-cu118-cp311-linux_x86_64/download.py --help` for the available options.

For a server without internet access, run the download on a machine that can access the Release, then copy this
folder, including all wheels, scripts and metadata, to the server. Verify the copied files without networking:

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py --verify-only
```

## Company HTTPS certificates

The downloader follows the same certificate handling as the data and model repositories. When optional
`truststore` is installed, it automatically uses the operating system's trust store, including company roots
installed by IT. Otherwise it uses Python's default TLS roots. Install the helper once in the same Python 3.10+
environment, then rerun the download command from the `emotion_recognition/` project folder:

```bash
python -m pip install -r dependencies/torch-cu118-cp311-linux_x86_64/requirements-download.txt
python dependencies/torch-cu118-cp311-linux_x86_64/download.py
```

Alternatively, provide an IT-issued company root/intermediate CA PEM bundle (also supported on Python 3.9):

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py --ca-bundle "/path/to/company-ca.pem"
```

`SSL_CERT_FILE=/path/to/company-ca.pem` is also supported; `--ca-bundle` takes precedence. These settings apply
to both GitHub Release downloads and `--source upstream`. Certificate and hostname verification remain enabled,
and existing verified wheels and resumable parts are preserved. If pip itself needs the company certificate,
use `python -m pip install --cert /path/to/company-ca.pem truststore`.

The wheelhouse also includes a `truststore` wheel. If it has already been copied to a Python 3.10+ environment,
the helper can be installed locally before downloading with the native certificate store:

```bash
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/truststore-*.whl
```

## Install on the NVIDIA server

Activate the project's Python 3.11 environment, with pip available, and run from the
**`emotion_recognition/` project folder**. Downloading requires network access; installation uses only the verified
local wheels:

```bash
conda activate emotion_recognition_llm
python dependencies/torch-cu118-cp311-linux_x86_64/download.py
python dependencies/torch-cu118-cp311-linux_x86_64/install.py
python -c "import torch, torchaudio, torchvision; print(torch.__version__, torchaudio.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

[`install.py`](install.py) first checks the target platform and verifies the size and SHA-256 of **every wheel**
before installing anything. It then runs one `python -m pip install --no-index --no-deps <wheel>` command per
package, in the manifest's dependency-first order, followed by `python -m pip check`. It stops if a step fails.

To inspect the numbered commands without installing, even on a different platform or before downloading:

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/install.py --dry-run
```

For manual, one-package-at-a-time installation, follow [`INSTALL_ORDER.md`](INSTALL_ORDER.md). An alternative
single pip command uses the fully pinned, hash-locked requirements and the same local wheelhouse:

```bash
python -m pip install --no-index --find-links ./dependencies/torch-cu118-cp311-linux_x86_64 --require-hashes -r dependencies/torch-cu118-cp311-linux_x86_64/requirements-offline.txt
python -m pip check
```

Use either installation route in place of `pip install -r requirements-llm-cuda118.txt`. Expected versions are
`2.7.1+cu118`, `2.7.1+cu118` and `0.22.1+cu118`; CUDA availability requires a compatible NVIDIA driver and
an accessible GPU.

`nvidia-smi` reports the driver's supported CUDA version. A driver reporting CUDA 12.4 can run these CUDA 11.8
wheels through NVIDIA's [driver backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html);
it does not need to report 11.8 or have a separate 11.8 toolkit installed.

The Torch/audio/vision stack and its default runtime dependencies can now be installed fully offline from this
folder. The common packages in `requirements.txt`, such as the model loaders, still need their own installation
and require internet access unless separately cached or installed:

```bash
python -m pip install -r requirements.txt
```

A separate CUDA toolkit is not required for the prebuilt wheels.
