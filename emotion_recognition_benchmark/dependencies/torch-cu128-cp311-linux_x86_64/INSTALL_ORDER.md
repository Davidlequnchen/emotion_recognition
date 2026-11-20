# 逐个离线安装顺序

目标：CPython 3.11、Linux x86_64、glibc >= 2.28。
以下顺序由 wheel METADATA 的实际依赖关系生成；每个包的依赖均先安装。
所有 wheels 必须先下载到本目录并通过 SHA-256 校验。

在 `emotion_recognition_benchmark/` 项目目录、激活目标 Conda 环境后执行：

```bash
python dependencies/torch-cu128-cp311-linux_x86_64/download.py --verify-only
python dependencies/torch-cu128-cp311-linux_x86_64/install.py
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
| 7 | nvidia-cublas-cu12 | 12.8.3.14 | — |
| 8 | nvidia-cuda-cupti-cu12 | 12.8.57 | — |
| 9 | nvidia-cuda-nvrtc-cu12 | 12.8.61 | — |
| 10 | nvidia-cuda-runtime-cu12 | 12.8.57 | — |
| 11 | nvidia-nvjitlink-cu12 | 12.8.61 | — |
| 12 | nvidia-cufile-cu12 | 1.13.0.11 | — |
| 13 | nvidia-curand-cu12 | 10.3.9.55 | — |
| 14 | nvidia-cusparselt-cu12 | 0.6.3 | — |
| 15 | nvidia-nccl-cu12 | 2.26.2 | — |
| 16 | nvidia-nvtx-cu12 | 12.8.55 | — |
| 17 | pillow | 13.0.0 | — |
| 18 | setuptools | 84.2.1 | — |
| 19 | typing-extensions | 4.16.0 | — |
| 20 | truststore | 0.10.5 | — |
| 21 | jinja2 | 3.1.6 | markupsafe |
| 22 | sympy | 1.14.0 | mpmath |
| 23 | nvidia-cudnn-cu12 | 9.7.1.26 | nvidia-cublas-cu12 |
| 24 | nvidia-cufft-cu12 | 11.3.3.41 | nvidia-nvjitlink-cu12 |
| 25 | nvidia-cusparse-cu12 | 12.5.7.53 | nvidia-nvjitlink-cu12 |
| 26 | triton | 3.3.1 | setuptools |
| 27 | nvidia-cusolver-cu12 | 11.7.2.55 | nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-nvjitlink-cu12 |
| 28 | torch | 2.7.1+cu128 | filelock, fsspec, jinja2, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-cufile-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-cusparselt-cu12, nvidia-nccl-cu12, nvidia-nvjitlink-cu12, nvidia-nvtx-cu12, sympy, triton, typing-extensions |
| 29 | torchaudio | 2.7.1+cu128 | torch |
| 30 | torchvision | 0.22.1+cu128 | numpy, pillow, torch |

## 可逐行复制的命令

```bash
# 1. filelock==4.2.10
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/filelock-4.2.10-py3-none-any.whl
# 2. fsspec==2026.9.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/fsspec-2026.9.0-py3-none-any.whl
# 3. markupsafe==3.0.4
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/markupsafe-3.0.4-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
# 4. mpmath==1.3.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/mpmath-1.3.0-py3-none-any.whl
# 5. networkx==3.6.1
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/networkx-3.6.1-py3-none-any.whl
# 6. numpy==2.4.6
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/numpy-2.4.6-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 7. nvidia-cublas-cu12==12.8.3.14
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cublas_cu12-12.8.3.14-py3-none-manylinux_2_27_x86_64.whl
# 8. nvidia-cuda-cupti-cu12==12.8.57
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cuda_cupti_cu12-12.8.57-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 9. nvidia-cuda-nvrtc-cu12==12.8.61
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cuda_nvrtc_cu12-12.8.61-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
# 10. nvidia-cuda-runtime-cu12==12.8.57
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cuda_runtime_cu12-12.8.57-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 11. nvidia-nvjitlink-cu12==12.8.61
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_nvjitlink_cu12-12.8.61-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
# 12. nvidia-cufile-cu12==1.13.0.11
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cufile_cu12-1.13.0.11-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 13. nvidia-curand-cu12==10.3.9.55
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_curand_cu12-10.3.9.55-py3-none-manylinux_2_27_x86_64.whl
# 14. nvidia-cusparselt-cu12==0.6.3
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl
# 15. nvidia-nccl-cu12==2.26.2
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 16. nvidia-nvtx-cu12==12.8.55
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_nvtx_cu12-12.8.55-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 17. pillow==13.0.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/pillow-13.0.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 18. setuptools==84.2.1
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/setuptools-84.2.1-py3-none-any.whl
# 19. typing-extensions==4.16.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/typing_extensions-4.16.0-py3-none-any.whl
# 20. truststore==0.10.5
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/truststore-0.10.5-py3-none-any.whl
# 21. jinja2==3.1.6
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/jinja2-3.1.6-py3-none-any.whl
# 22. sympy==1.14.0
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/sympy-1.14.0-py3-none-any.whl
# 23. nvidia-cudnn-cu12==9.7.1.26
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cudnn_cu12-9.7.1.26-py3-none-manylinux_2_27_x86_64.whl
# 24. nvidia-cufft-cu12==11.3.3.41
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cufft_cu12-11.3.3.41-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 25. nvidia-cusparse-cu12==12.5.7.53
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cusparse_cu12-12.5.7.53-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
# 26. triton==3.3.1
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/triton-3.3.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 27. nvidia-cusolver-cu12==11.7.2.55
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/nvidia_cusolver_cu12-11.7.2.55-py3-none-manylinux_2_27_x86_64.whl
# 28. torch==2.7.1+cu128
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
# 29. torchaudio==2.7.1+cu128
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/torchaudio-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
# 30. torchvision==0.22.1+cu128
python -m pip install --no-index --no-deps ./dependencies/torch-cu128-cp311-linux_x86_64/torchvision-0.22.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
python -m pip check
```

本集合覆盖 PyTorch/torchaudio/torchvision 的完整默认运行时依赖，并额外包含 truststore。
项目 `requirements.txt` 中的 FunASR、transformers 等其他包及可选 extras 不在本集合内。
