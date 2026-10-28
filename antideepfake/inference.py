"""
Deepfake (spoofed speech) detection with AntiDeepfake XLS-R-1B (PyTorch).

Every step prints what it does, so you can check that Python, PyTorch, the device (Apple GPU / NVIDIA GPU / CPU)
and the model are all in order. For each audio file the model gives two logits, [fake, real]. Their softmax gives
P(real), the probability that the speech is real (bona fide): this is AntiDeepfake's own score, and higher means
more likely real. P(fake) = 1 - P(real) is the deepfake score.

Usage (from the antideepfake/ folder):
    python inference.py
Settings are in config.yaml: `mode` and `devices` choose the hardware, `inference.audio` the audio file(s) or
folder(s), `inference.threshold` the decision threshold. Get the checkpoint first, from Zenodo: python download_model.py

Works offline: the network is defined in this file and its weights are read from xlsr_1b.ckpt. Nothing is
downloaded at run time (no Hugging Face, no torch.hub, no fairseq).

How AntiDeepfake runs inference (https://github.com/nii-yamagishilab/AntiDeepfake), reproduced here:
  - dataio/dataio.py, test_audio_pipeline(): read the file, average the channels (mono), resample to 16 kHz with
    torchaudio, then normalise the waveform to zero mean and unit variance (layer_norm). Nothing is trimmed at
    test time: each file is scored in one pass, whatever its length.
  - models/W2V.py, Model: an XLS-R 1B front end (fairseq Wav2Vec2Model built from the 'xlsr_1b' entry of
    models/W2V_configs.py), average pooling over time, and a linear layer with 2 outputs.
  - evaluation.py: the outputs are [fake logit, real logit] (training labels: 1 = real, 0 = fake);
    softmax_score() takes the softmax probability of class 1 (real), and a file counts as real when it is >= 0.5.
The classes below re-implement the inference path of fairseq's Wav2Vec2Model (MIT licence) as used by
AntiDeepfake's models/W2V.py (BSD-3-Clause), in plain PyTorch and with the same parameter names, so xlsr_1b.ckpt
loads unchanged. fairseq itself is not used: it would downgrade hydra-core and omegaconf, which FunASR needs.
"""
import math
import platform
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parent  # project root (antideepfake/); paths in config.yaml are relative to it
CONFIG = ROOT / "config.yaml"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
SAMPLE_RATE = 16000  # the model works on 16 kHz mono audio
# Checkpoint parts that only wav2vec 2.0 pre-training uses (masking, quantizer): not needed for inference
PRETRAIN_ONLY = ("mask_emb", "quantizer.", "project_q.", "final_proj.")


# ----------------------------------------------------------------------
# The network: XLS-R 1B (wav2vec 2.0) front end + average pooling + linear layer
# ----------------------------------------------------------------------
class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):  # [batch, frames, dim]
        b, t, d = x.shape
        q, k, v = (proj(x).view(b, t, self.heads, -1).transpose(1, 2)  # [batch, heads, frames, head size]
                   for proj in (self.q_proj, self.k_proj, self.v_proj))
        x = F.scaled_dot_product_attention(q, k, v)  # softmax(q k^T / sqrt(head size)) v, for each head
        return self.out_proj(x.transpose(1, 2).reshape(b, t, d))


class EncoderLayer(nn.Module):
    """Transformer layer with the layer norm first (fairseq TransformerSentenceEncoderLayer, layer_norm_first)."""

    def __init__(self, dim, ffn, heads):
        super().__init__()
        self.self_attn = SelfAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class XLSR1B(nn.Module):
    """XLS-R 1B: fairseq Wav2Vec2Model with AntiDeepfake's 'xlsr_1b' configuration (models/W2V_configs.py):
    7 convolutions with layer norm (one frame per 20 ms), then 48 transformer layers of width 1280."""

    def __init__(self, dim=1280, ffn=5120, heads=16, layers=48):
        super().__init__()
        convs, n_in = [], 1
        for kernel, stride in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
            convs.append(nn.Sequential(
                nn.Conv1d(n_in, 512, kernel, stride=stride),
                nn.Identity(),  # dropout, only during training
                nn.Sequential(TransposeLast(), nn.LayerNorm(512), TransposeLast()),
                nn.GELU()))
            n_in = 512
        self.feature_extractor = nn.ModuleDict({"conv_layers": nn.ModuleList(convs)})
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, dim)
        self.encoder = nn.ModuleDict({
            "pos_conv": nn.Sequential(nn.Conv1d(dim, dim, 128, padding=64, groups=16)),  # position embedding
            "layers": nn.ModuleList(EncoderLayer(dim, ffn, heads) for _ in range(layers)),
            "layer_norm": nn.LayerNorm(dim),
        })

    def forward(self, wav):  # [batch, samples] -> [batch, frames, 1280]
        x = wav.unsqueeze(1)
        for conv in self.feature_extractor["conv_layers"]:
            x = conv(x)
        x = self.post_extract_proj(self.layer_norm(x.transpose(1, 2)))
        pos = self.encoder["pos_conv"](x.transpose(1, 2))[:, :, :-1]  # even kernel: drop the one extra frame
        x = x + F.gelu(pos).transpose(1, 2)
        for layer in self.encoder["layers"]:
            x = layer(x)
        return self.encoder["layer_norm"](x)  # with the layer norm first, the final layer norm comes last


class AntiDeepfake(nn.Module):
    """AntiDeepfake's models/W2V.py `Model`: front end, average over time, linear layer -> 2 logits."""

    def __init__(self):
        super().__init__()
        self.ssl = XLSR1B()
        self.proj_fc = nn.Linear(1280, 2)

    def forward(self, wav):
        return self.proj_fc(self.ssl(wav).mean(dim=1))  # [batch, 2]: index 0 = fake, index 1 = real


def load_model(path):
    """Build the network on the 'meta' device (no memory, no random values) and give it the checkpoint's weights."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    skipped = [k for k in state if k.removeprefix("m_ssl.model.").startswith(PRETRAIN_ONLY)]
    state = {k.replace("m_ssl.model.", "ssl."): v for k, v in state.items() if k not in skipped}
    # The position convolution is stored weight-normalised (weight = g * v / |v|): compute its weight once
    g, v = state.pop("ssl.encoder.pos_conv.0.weight_g"), state.pop("ssl.encoder.pos_conv.0.weight_v")
    state["ssl.encoder.pos_conv.0.weight"] = v * (g / v.norm(dim=(0, 1), keepdim=True))
    with torch.device("meta"):
        model = AntiDeepfake()
    model.load_state_dict(state, strict=True, assign=True)  # every name and shape must match
    return model.eval(), skipped


# ----------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------
def collect_audio(entries):
    """The files named in the config: a file, a folder (with its sub-folders) or a list of files and folders."""
    files = []
    for entry in entries if isinstance(entries, list) else [entries]:
        path = ROOT / entry  # an absolute path in the config is kept as it is
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
        elif path.is_file():
            files.append(path)
        else:
            sys.exit(f"ERROR: {path} not found.")
    return files


def load_audio(path):
    """Read a file as AntiDeepfake's test_audio_pipeline() does: average the channels, resample to 16 kHz."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels], values in [-1, 1]
    wav = torch.from_numpy(wav).mean(dim=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
    return wav


def score(model, wav, device, max_seconds):
    """Logits [fake, real] of one file: one pass over the whole file (AntiDeepfake), or, with max_seconds, the
    length-weighted mean of the logits of equal segments of at most max_seconds."""
    n = math.ceil(len(wav) / (max_seconds * SAMPLE_RATE)) if max_seconds else 1
    logits = torch.zeros(2)
    for seg in torch.tensor_split(wav, n):
        seg = F.layer_norm(seg, seg.shape)  # zero mean and unit variance over the file (or segment)
        logits += model(seg[None].to(device))[0].cpu() * len(seg) / len(wav)
    return logits


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order with warnings, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the configuration (config.yaml)
    # ------------------------------------------------------------------
    step("Step 1/6 | Load configuration")
    cfg = yaml.safe_load(CONFIG.read_text())
    mode = cfg["mode"]
    device_name = cfg["devices"][mode]
    ckpt = ROOT / cfg["model"]["dir"] / cfg["model"]["checkpoint"]
    audio, threshold, max_seconds = (cfg["inference"][k] for k in ("audio", "threshold", "max_seconds"))
    print(f"  config      : {CONFIG}")
    print(f"  mode        : {mode}")
    print(f"  device      : {device_name}")
    print(f"  checkpoint  : {ckpt}")
    print(f"  audio       : {audio}")
    print(f"  threshold   : real if P(real) >= {threshold}")
    print(f"  max_seconds : {max_seconds or 'none (each file in one pass, as AntiDeepfake does)'}")

    # ------------------------------------------------------------------
    # Step 2: Check the software stack and which accelerators PyTorch can see
    # ------------------------------------------------------------------
    step("Step 2/6 | Check environment")
    print(f"  Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  torchaudio  : {torchaudio.__version__} (resampling)")
    print(f"  soundfile   : {sf.__version__} (reading audio)")
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
                     "       Check the driver with `nvidia-smi` and install the CUDA build of PyTorch (see "
                     "requirements.txt).")
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
                     "       Use an arm64 (not Rosetta) Python on Apple Silicon, or set the device to cpu.")
        print("  using       : Apple GPU via Metal (MPS)")
    else:
        print(f"  using       : CPU ({torch.get_num_threads()} threads)")
    x = torch.randn(512, 512, device=device)
    (x @ x).sum().item()  # .item() waits until the device has finished
    print(f"  sanity check: 512x512 matrix multiplication on {device} ... OK")

    # ------------------------------------------------------------------
    # Step 4: Load the model (xlsr_1b.ckpt) onto the device
    # ------------------------------------------------------------------
    step("Step 4/6 | Load model")
    if not ckpt.exists():
        sys.exit(f"ERROR: {ckpt} not found. Download it first (from Zenodo): python download_model.py")
    print(f"  {ckpt.name:<12}: {ckpt.stat().st_size / 1e6:,.2f} MB")
    t0 = time.perf_counter()
    # The network is built from the definition in this file (XLS-R 1B: 48 layers, width 1280), then every
    # weight is read from the checkpoint; the names and shapes must all match.
    model, skipped = load_model(ckpt)
    model.to(device)
    print(f"  loaded in   : {time.perf_counter() - t0:.1f} s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters  : {n_params / 1e6:.1f} M, on {next(model.parameters()).device}")
    parts = sorted({k.split(".")[2] for k in skipped})  # e.g. "m_ssl.model.quantizer.vars" -> quantizer
    print(f"  not used    : {len(skipped)} pre-training tensors ({', '.join(parts)})")
    if device.type == "cuda":
        print(f"  GPU memory  : {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    elif device.type == "mps":
        print(f"  GPU memory  : {torch.mps.current_allocated_memory() / 1024**3:.2f} GB allocated")
    print("  outputs     : 2 logits per file -> index 0 = fake (spoof), index 1 = real (bona fide)")

    # ------------------------------------------------------------------
    # Step 5: Collect the audio files
    # ------------------------------------------------------------------
    step("Step 5/6 | Collect audio")
    files = collect_audio(audio)
    if not files:
        sys.exit(f"ERROR: no audio files found in {audio}.")
    for f in files:
        info = sf.info(f)
        print(f"  {f.name:<28} {info.samplerate} Hz, {info.channels} ch, {info.duration:.2f} s")
    print("  (each file is converted to 16 kHz mono and normalised to zero mean and unit variance)")

    # ------------------------------------------------------------------
    # Step 6: Inference - one forward pass per file, softmax over [fake, real]
    # ------------------------------------------------------------------
    step("Step 6/6 | Run inference")
    print("  score: P(real) = probability that the speech is real (bona fide); P(fake) = 1 - P(real)\n")
    cpu_model = None

    def predict(path):
        """MPS cannot run the first convolution on files longer than 20.48 s: score those on the CPU."""
        nonlocal cpu_model
        wav = load_audio(path)
        try:
            return score(model, wav, device, max_seconds)
        except NotImplementedError:
            print(f"  ({path.name} is longer than MPS supports: running it on the CPU)")
            if cpu_model is None:
                cpu_model = load_model(ckpt)[0]
            return score(cpu_model, wav, torch.device("cpu"), max_seconds)

    with torch.inference_mode():
        predict(files[0])
        print("  warm-up done (the first forward pass initialises the device kernels)\n")

        times, n_real = [], 0
        for i, f in enumerate(files, 1):
            t0 = time.perf_counter()
            logits = predict(f)
            times.append(time.perf_counter() - t0)
            p_fake, p_real = torch.softmax(logits, dim=0).tolist()
            label = "real" if p_real >= threshold else "FAKE"
            n_real += label == "real"
            print(f"  [{i}/{len(files)}] {f.name:<28} -> {label}   P(real) {p_real:.3f}   {times[-1] * 1000:.0f} ms")
            print(f"        P(fake) {p_fake:.3f} | logits [fake, real] = [{logits[0]:.2f}, {logits[1]:.2f}]")

    print(f"\nDone: {len(files)} file(s) on {device}, average {sum(times) / len(times) * 1000:.0f} ms per file.")
    print(f"  real: {n_real} | fake: {len(files) - n_real}   (real if P(real) >= {threshold})")


if __name__ == "__main__":
    main()
