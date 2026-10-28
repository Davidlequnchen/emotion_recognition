"""
Minimal speech emotion recognition with emotion2vec+ (PyTorch, via FunASR).

Every step prints what it does, so you can check that Python, PyTorch, the
device (Apple GPU / NVIDIA GPU / CPU) and the model are all in order.

Usage (from the emotion_recognition/ folder):
    python scripts/emotion2vec/inference.py
The settings (AUDIO, DEVICE, MODEL_DIR) are at the top of this file; it reads no configuration file.
"""
import logging
import platform
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

# ======================================================================
# Settings (paths are relative to the emotion_recognition/ folder)
# ======================================================================
AUDIO = "data/samples"  # a file, or a folder: every audio file in it (wav, flac, mp3, ogg)
DEVICE = "auto"  # auto = cuda:0 if CUDA is available, else mps if available, else cpu; or "cuda:0", "mps", "cpu"
MODEL_DIR = "models/emotion2vec_plus_large"  # the emotion2vec+ large files (python scripts/common/download_model.py)

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/); the paths above are relative to it
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
MODEL_FILES = ["model.pt", "config.yaml", "configuration.json", "tokens.txt"]


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order with warnings, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: The settings at the top of this file, and the device that DEVICE = "auto" picks
    # ------------------------------------------------------------------
    step("Step 1/6 | Settings")
    model_dir = ROOT / MODEL_DIR  # an absolute path is kept as it is
    audio_path = ROOT / AUDIO
    device_name, why = DEVICE, ""
    if DEVICE == "auto":  # the best device this machine has
        if torch.cuda.is_available():
            device_name, why = "cuda:0", " (auto: CUDA is available)"
        elif torch.backends.mps.is_available():
            device_name, why = "mps", " (auto: no CUDA, MPS is available)"
        else:
            device_name, why = "cpu", " (auto: neither CUDA nor MPS is available)"
    print(f"  settings    : AUDIO, DEVICE, MODEL_DIR at the top of {Path(__file__).resolve().relative_to(ROOT)}")
    print(f"  device      : {device_name}{why}")
    print(f"  model       : {model_dir}")
    print(f"  audio       : {audio_path}")

    # ------------------------------------------------------------------
    # Step 2: Check the software stack and which accelerators PyTorch can see
    # ------------------------------------------------------------------
    step("Step 2/6 | Check environment")
    import torchaudio
    import funasr  # imported here rather than at the top because it takes a few seconds

    logging.getLogger().setLevel(logging.ERROR)  # importing FunASR turns on INFO logs; keep the output readable
    print(f"  Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  torchaudio  : {torchaudio.__version__}")
    print(f"  FunASR      : {funasr.__version__}")
    print(f"  CUDA        : available={torch.cuda.is_available()} (PyTorch built with CUDA {torch.version.cuda})")
    print(f"  MPS (Apple) : available={torch.backends.mps.is_available()} (built={torch.backends.mps.is_built()})")

    # ------------------------------------------------------------------
    # Step 3: Activate the requested device and check that it really works
    # ------------------------------------------------------------------
    step("Step 3/6 | Activate device")
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            sys.exit("ERROR: CUDA was requested but is not available.\n"
                     "       Check the driver with `nvidia-smi` and install requirements-cuda121.txt.")
        device = torch.device("cuda", device.index or 0)  # "cuda" -> "cuda:0"
        torch.cuda.set_device(device)
        gpu = torch.cuda.get_device_properties(device)
        print(f"  GPUs found  : {torch.cuda.device_count()}")
        print(f"  using       : {device} = {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB, "
              f"compute capability {gpu.major}.{gpu.minor})")
        print(f"  cuDNN       : {torch.backends.cudnn.version()}")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            sys.exit("ERROR: MPS was requested but is not available.\n"
                     "       Use an arm64 (not Rosetta) Python on Apple Silicon, or set DEVICE to \"cpu\".")
        print("  using       : Apple GPU via Metal (MPS)")
    else:
        print(f"  using       : CPU ({torch.get_num_threads()} threads)")
    x = torch.randn(512, 512, device=device)
    (x @ x).sum().item()  # .item() waits until the device has finished
    print(f"  sanity check: 512x512 matrix multiplication on {device} ... OK")

    # ------------------------------------------------------------------
    # Step 4: Load the model (config.yaml + model.pt) onto the device
    # ------------------------------------------------------------------
    step("Step 4/6 | Load model")
    for name in MODEL_FILES:
        f = model_dir / name
        if not f.exists():
            sys.exit(f"ERROR: {f} not found. Download the model first: python scripts/common/download_model.py")
        print(f"  {name:<19}: {f.stat().st_size / 1e6:,.2f} MB")

    from funasr import AutoModel

    t0 = time.perf_counter()
    # FunASR reads configuration.json, which names the other files: it builds the network from
    # config.yaml, loads the weights from model.pt, reads the class labels from tokens.txt and
    # moves the network to `device`.
    load = lambda dev: AutoModel(
        model=str(model_dir),
        device=str(dev),
        ncpu=torch.get_num_threads(),  # keep PyTorch's CPU thread count (FunASR would lower it to 4)
        # The pre-training decoder is not in model.pt and is never used for inference:
        # skip it instead of printing a "miss key" warning for each of its weights.
        excludes="modality_encoders.AUDIO.decoder.",
        disable_update=True,           # no online version check
        disable_pbar=True,
    )
    model = load(device)
    net = model.model  # the underlying torch.nn.Module
    print(f"  loaded in   : {time.perf_counter() - t0:.1f} s")
    print(f"  parameters  : {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M, on {next(net.parameters()).device}")
    if device.type == "cuda":
        print(f"  GPU memory  : {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    elif device.type == "mps":
        print(f"  GPU memory  : {torch.mps.current_allocated_memory() / 1024**3:.2f} GB allocated")
    labels = (model_dir / "tokens.txt").read_text(encoding="utf-8").split()
    print(f"  classes     : {len(labels)} -> {', '.join(lab.split('/')[-1] for lab in labels)}")

    # ------------------------------------------------------------------
    # Step 5: Collect the audio files
    # ------------------------------------------------------------------
    step("Step 5/6 | Collect audio")
    if audio_path.is_dir():
        files = sorted(p for p in audio_path.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    else:
        files = [audio_path] if audio_path.is_file() else []
    if not files:
        sys.exit(f"ERROR: no audio files found at {audio_path}. Download the test clips: python scripts/common/download_data.py")
    for f in files:
        info = sf.info(f)
        print(f"  {f.name:<24} {info.samplerate} Hz, {info.channels} ch, {info.duration:.2f} s")
    print("  (the model works on 16 kHz mono audio; other formats are converted automatically)")

    # ------------------------------------------------------------------
    # Step 6: Inference - one forward pass per file, softmax over the emotion classes
    # ------------------------------------------------------------------
    step("Step 6/6 | Run inference")
    cpu_model = None

    def predict(path):
        """MPS cannot run the model's first convolution on clips longer than about 20 s: run those on the CPU."""
        nonlocal cpu_model
        try:
            return model.generate(str(path), granularity="utterance", extract_embedding=False)[0]
        except NotImplementedError:
            print(f"  ({path.name} is longer than MPS supports: running it on the CPU)")
            if cpu_model is None:
                cpu_model = load("cpu")
            return cpu_model.generate(str(path), granularity="utterance", extract_embedding=False)[0]

    predict(files[0])
    print("  warm-up done (the first forward pass initialises the device kernels)\n")

    times = []
    for i, f in enumerate(files, 1):
        t0 = time.perf_counter()
        res = predict(f)
        times.append(time.perf_counter() - t0)
        # Labels look like "生气/angry": keep the English part, sort by probability.
        ranked = sorted(((lab.split("/")[-1], p) for lab, p in zip(res["labels"], res["scores"])),
                        key=lambda x: x[1], reverse=True)
        print(f"  [{i}/{len(files)}] {f.name:<24} -> {ranked[0][0]} ({ranked[0][1]:.2f})   {times[-1] * 1000:.0f} ms")
        print(f"        top-3: {' | '.join(f'{lab} {p:.2f}' for lab, p in ranked[:3])}")

    print(f"\nDone: {len(files)} file(s) on {device}, average {sum(times) / len(times) * 1000:.0f} ms per file.")


if __name__ == "__main__":
    main()
