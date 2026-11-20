"""Device checks and mixed precision policy for CPU, MPS and NVIDIA CUDA.

B200 is Blackwell compute capability 10.0. The dedicated environment uses the
official PyTorch CUDA 12.8 wheels; the benchmark's CUDA 11.8 wheels are not the
B200 deployment environment. Architecture metadata is only a preliminary check:
the standalone environment doctor also executes real matmul/Conv1d/backward.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
from importlib import metadata
import os
import platform
import re
import resource
import shutil
import subprocess
import sys

import torch
from torch import nn


PRECISIONS = {"auto", "float32", "bfloat16"}


def resolve_precision(device: torch.device | str, precision: str) -> str:
    """Use CUDA BF16 when supported; explicit unsupported requests are errors.

    CPU/MPS retain FP32. FP16 is intentionally excluded: this training loop does
    not use a GradScaler. Encoder and EMA parameters stay FP32 with CUDA BF16
    autocast; regression targets and scalar losses are reduced in FP32.
    """
    device = torch.device(device)
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; activate emotion_recognition_llm on the GPU server")
        if precision == "float32":
            return precision
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported()
        if precision == "auto":
            return "bfloat16" if supported else "float32"
        if precision == "bfloat16" and not supported:
            raise RuntimeError("Explicit bfloat16 requested, but the selected CUDA device does not support BF16")
    elif precision == "bfloat16":
        raise ValueError("Explicit bfloat16 is supported only on a CUDA device; use float32 or auto on CPU/MPS")
    return "float32" if precision == "auto" else precision


def training_autocast(device: torch.device | str, precision: str):
    """Context for the model forward pass; backward and optimizer stay outside."""
    device = torch.device(device)
    resolved = resolve_precision(device, precision)
    if resolved == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def memory_metrics(device: torch.device) -> dict:
    """GiB. CUDA: tensors allocated now, reserved by the caching allocator, peak allocated. MPS: tensors allocated
    and the Metal driver's total. All devices: this process's peak resident memory (DataLoader workers excluded)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    metrics = {"process_peak_rss_gib": peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2}
    if device.type == "cuda":
        metrics.update(cuda_allocated_gib=torch.cuda.memory_allocated(device) / 1024**3,
                       cuda_reserved_gib=torch.cuda.memory_reserved(device) / 1024**3,
                       cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3)
    elif device.type == "mps":
        metrics.update(mps_allocated_gib=torch.mps.current_allocated_memory() / 1024**3,
                       mps_driver_gib=torch.mps.driver_allocated_memory() / 1024**3)
    return metrics


def cuda_architecture_compatibility(capability: tuple[int, int], compiled_arches: list[str]) -> dict:
    """Check same-major cubin compatibility or forward-compatible generic PTX.

    Architecture-specific suffices such as compute_90a are not generic PTX and
    cannot establish Blackwell compatibility. A listed compatible architecture
    still does not prove every cuDNN/cuBLAS kernel works; run the kernel doctor.
    """
    major, minor = capability
    target = major * 10 + minor
    native, ptx = [], []
    for architecture in compiled_arches:
        match = re.fullmatch(r"(sm|compute)_(\d+)", architecture)
        if match is None:
            continue
        kind, number = match.group(1), int(match.group(2))
        if kind == "sm" and number // 10 == major and number % 10 <= minor:
            native.append(architecture)
        if kind == "compute" and number <= target:
            ptx.append(architecture)
    return {"compatible": bool(native or ptx), "matching_native": native, "forward_compatible_ptx": ptx,
            "target": f"sm_{target}"}


def _cuda_build_check(device: torch.device) -> dict:
    capability = torch.cuda.get_device_capability(device)
    cuda_version = torch.version.cuda
    if capability[0] >= 10:
        version = tuple(int(part) for part in (cuda_version or "0.0").split(".")[:2])
        if version < (12, 8):
            raise RuntimeError(
                f"Blackwell sm_{capability[0]}{capability[1]} detected, but PyTorch uses CUDA {cuda_version}. "
                "Install emotion_recognition_benchmark/requirements-llm-cuda128.txt (torch 2.7.1+cu128) in emotion_recognition_llm; "
                "the CUDA 11.8 build has no Blackwell kernels."
            )
    architectures = torch.cuda.get_arch_list()
    compatibility = cuda_architecture_compatibility(capability, architectures)
    if not compatibility["compatible"]:
        raise RuntimeError(
            f"PyTorch {torch.__version__} has no compatible native/PTX architecture for {compatibility['target']}; "
            f"compiled architectures: {architectures}. Install the official CUDA 12.8 B200 environment."
        )
    return {"compute_capability": list(capability), "torch_compiled_architectures": architectures,
            "architecture_check": compatibility}


def _kernel_check(device: torch.device, precision: str) -> dict:
    """Small real allocation + GEMM + audio Conv1d + FP32 loss + backward."""
    rng_devices = [torch.cuda.current_device() if device.index is None else device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        left = torch.randn(16, 32, device=device, requires_grad=True)
        right = torch.randn(32, 16, device=device, requires_grad=True)
        waveform = torch.randn(2, 1, 128, device=device, requires_grad=True)
        convolution = nn.Conv1d(1, 8, kernel_size=5, stride=2).to(device)
        with training_autocast(device, precision):
            product = left @ right
            frames = convolution(waveform)
            loss = product.float().square().mean() + frames.float().square().mean()
        loss.backward()
        gradients = [left.grad, right.grad, waveform.grad, convolution.weight.grad, convolution.bias.grad]
        if not torch.isfinite(loss) or any(value is None or not torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("Device kernel test produced non-finite outputs/gradients")
        if not any(value.abs().sum() > 0 for value in gradients):
            raise RuntimeError("Device kernel test produced no nonzero gradients")
        if precision == "bfloat16" and (product.dtype != torch.bfloat16 or frames.dtype != torch.bfloat16):
            raise RuntimeError("CUDA BF16 requested, but matmul/Conv1d did not execute with BF16 outputs")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"passed": True, "matmul_output_dtype": str(product.dtype), "conv1d_output_dtype": str(frames.dtype),
                "fp32_loss": float(loss.detach()), "finite_nonzero_backward_gradients": True}


def validate_device(device: torch.device | str, precision: str = "auto", *, run_kernels: bool = False) -> dict:
    """Fail fast on unavailable/unsupported hardware; never substitute a CPU."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; activate emotion_recognition_llm on the GPU server")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}")
    resolved = resolve_precision(device, precision)
    result = {"device": str(device), "requested_precision": precision, "resolved_precision": resolved,
              "kernel_checks_run": run_kernels}
    if device.type == "cuda":
        result.update(_cuda_build_check(device))
    if run_kernels:
        result["kernel_checks"] = _kernel_check(device, resolved)
    return result


def device_environment(device: torch.device | str | None = None) -> dict:
    """JSON-safe runtime inventory, including the actual interpreter and GPU."""
    packages = {}
    for name in ("torch", "torchaudio", "torchvision", "funasr", "numpy", "pyyaml", "soundfile", "matplotlib", "tqdm"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"), "packages": packages,
        "torch_cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_device": str(device) if device is not None else None, "cuda_devices": [],
    }
    if torch.cuda.is_available():
        result["torch_compiled_architectures"] = torch.cuda.get_arch_list()
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["cuda_devices"].append({"index": index, "name": properties.name,
                                           "compute_capability": [properties.major, properties.minor],
                                           "total_memory_bytes": properties.total_memory})
        if device is not None and torch.device(device).type == "cuda":
            selected = torch.device(device)
            selected_index = torch.cuda.current_device() if selected.index is None else selected.index
            result["selected_cuda_device"] = next(
                gpu for gpu in result["cuda_devices"] if gpu["index"] == selected_index)
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            result["nvidia_smi_gpu_driver_memory"] = completed.stdout.strip()
            if completed.returncode:
                result["nvidia_smi_error"] = completed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result["nvidia_smi_error"] = str(error)
    return result
