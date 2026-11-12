# 逐个离线安装顺序

目标：CPython 3.11、Linux x86_64、glibc >= 2.28。
以下顺序由 wheel METADATA 的实际依赖关系生成；每个包的依赖均先安装。
所有 wheels 必须先下载到本目录并通过 SHA-256 校验。

在 `emotion_recognition/` 项目目录、激活目标 Conda 环境后执行：

```bash
python dependencies/torch-cu118-cp311-linux_x86_64/download.py --verify-only
python dependencies/torch-cu118-cp311-linux_x86_64/install.py
```

`install.py` 会先检查平台并验证整套文件，再逐个执行下列命令，最后运行 `pip check`。
也可以按下面顺序手工逐行执行；`--no-index --no-deps` 保证每条安装命令只使用指定的本地 wheel。

| 顺序 | 包 | 版本 | 先安装的直接依赖 |
|---|---|---|---|
| 1 | filelock | 4.2.10 | — |
| 2 | fsspec | 2026.9.0 | — |
| 3 | markupsafe | 3.0.4 | — |
| 4 | mpmath | 1.3.0 | — |
| 5 | networkx | 3.6.1 | — |
| 6 | numpy | 2.4.6 | — |
| 7 | nvidia-cublas-cu11 | 11.11.3.6 | — |
| 8 | nvidia-cuda-cupti-cu11 | 11.8.87 | — |
| 9 | nvidia-cuda-nvrtc-cu11 | 11.8.89 | — |
| 10 | nvidia-cuda-runtime-cu11 | 11.8.89 | — |
| 11 | nvidia-cufft-cu11 | 10.9.0.58 | — |
| 12 | nvidia-curand-cu11 | 10.3.0.86 | — |
| 13 | nvidia-cusparse-cu11 | 11.7.5.86 | — |
| 14 | nvidia-nccl-cu11 | 2.21.5 | — |
| 15 | nvidia-nvtx-cu11 | 11.8.86 | — |
| 16 | pillow | 13.0.0 | — |
| 17 | setuptools | 84.2.1 | — |
| 18 | typing-extensions | 4.16.0 | — |
| 19 | truststore | 0.10.5 | — |
| 20 | jinja2 | 3.1.6 | markupsafe |
| 21 | sympy | 1.14.0 | mpmath |
| 22 | nvidia-cudnn-cu11 | 9.1.0.70 | nvidia-cublas-cu11 |
| 23 | nvidia-cusolver-cu11 | 11.4.1.48 | nvidia-cublas-cu11 |
| 24 | triton | 3.3.1 | setuptools |
| 25 | torch | 2.7.1+cu118 | filelock, fsspec, jinja2, networkx, nvidia-cublas-cu11, nvidia-cuda-cupti-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-runtime-cu11, nvidia-cudnn-cu11, nvidia-cufft-cu11, nvidia-curand-cu11, nvidia-cusolver-cu11, nvidia-cusparse-cu11, nvidia-nccl-cu11, nvidia-nvtx-cu11, sympy, triton, typing-extensions |
| 26 | torchaudio | 2.7.1+cu118 | torch |
| 27 | torchvision | 0.22.1+cu118 | numpy, pillow, torch |

## 可逐行复制的命令

```bash
# 1. filelock==4.2.10
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/filelock-4.2.10-py3-none-any.whl
# 2. fsspec==2026.9.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/fsspec-2026.9.0-py3-none-any.whl
# 3. markupsafe==3.0.4
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/markupsafe-3.0.4-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
# 4. mpmath==1.3.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/mpmath-1.3.0-py3-none-any.whl
# 5. networkx==3.6.1
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/networkx-3.6.1-py3-none-any.whl
# 6. numpy==2.4.6
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/numpy-2.4.6-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 7. nvidia-cublas-cu11==11.11.3.6
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux2014_x86_64.whl
# 8. nvidia-cuda-cupti-cu11==11.8.87
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cuda_cupti_cu11-11.8.87-py3-none-manylinux2014_x86_64.whl
# 9. nvidia-cuda-nvrtc-cu11==11.8.89
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cuda_nvrtc_cu11-11.8.89-py3-none-manylinux2014_x86_64.whl
# 10. nvidia-cuda-runtime-cu11==11.8.89
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cuda_runtime_cu11-11.8.89-py3-none-manylinux2014_x86_64.whl
# 11. nvidia-cufft-cu11==10.9.0.58
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cufft_cu11-10.9.0.58-py3-none-manylinux2014_x86_64.whl
# 12. nvidia-curand-cu11==10.3.0.86
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_curand_cu11-10.3.0.86-py3-none-manylinux2014_x86_64.whl
# 13. nvidia-cusparse-cu11==11.7.5.86
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cusparse_cu11-11.7.5.86-py3-none-manylinux2014_x86_64.whl
# 14. nvidia-nccl-cu11==2.21.5
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_nccl_cu11-2.21.5-py3-none-manylinux2014_x86_64.whl
# 15. nvidia-nvtx-cu11==11.8.86
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_nvtx_cu11-11.8.86-py3-none-manylinux2014_x86_64.whl
# 16. pillow==13.0.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 17. setuptools==84.2.1
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/setuptools-84.2.1-py3-none-any.whl
# 18. typing-extensions==4.16.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/typing_extensions-4.16.0-py3-none-any.whl
# 19. truststore==0.10.5
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/truststore-0.10.5-py3-none-any.whl
# 20. jinja2==3.1.6
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/jinja2-3.1.6-py3-none-any.whl
# 21. sympy==1.14.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/sympy-1.14.0-py3-none-any.whl
# 22. nvidia-cudnn-cu11==9.1.0.70
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cudnn_cu11-9.1.0.70-py3-none-manylinux2014_x86_64.whl
# 23. nvidia-cusolver-cu11==11.4.1.48
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/nvidia_cusolver_cu11-11.4.1.48-py3-none-manylinux2014_x86_64.whl
# 24. triton==3.3.1
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/triton-3.3.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 25. torch==2.7.1+cu118
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/torch-2.7.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl
# 26. torchaudio==2.7.1+cu118
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/torchaudio-2.7.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl
# 27. torchvision==0.22.1+cu118
python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/torchvision-0.22.1+cu118-cp311-cp311-manylinux_2_28_x86_64.whl
python -m pip check
```

本集合覆盖 PyTorch/torchaudio/torchvision 的完整默认运行时依赖，并额外包含 truststore。
项目 `requirements.txt` 中的 FunASR、transformers 等其他包及可选 extras 不在本集合内。
