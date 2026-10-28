"""Vox-Profile categorical emotion models, trained on MSP-Podcast; 9 classes:

    models/vox_profile_whisper  tiantiaf/whisper-large-v3-msp-podcast-emotion  Whisper-large-v3 encoder + head, 639.9M
    models/vox_profile_wavlm    tiantiaf/wavlm-large-categorical-emotion       WavLM-large + head, 316.2M

Which of the two a folder holds is read from its config.json (pretrain_model: whisper_large or wavlm_large). Files:
    model.safetensors          weights as released, float32 (Whisper: 6.19 GB with its text decoder; WavLM: 1.26 GB)
    model_slim.safetensors     Whisper only, written by prepare(): the same weights without the decoder, float16 where
                               that is exact, 1.29 GB; load() uses it when it is there
    config.json, README.md     the model's settings and its model card (licence and out-of-scope uses)
    whisper-large-v3/, wavlm-large/   config.json and preprocessor_config.json of the base model, saved by prepare()

Source: this re-implements the inference path of the official code, src/model/emotion/whisper_emotion.py and
wavlm_emotion.py of https://github.com/tiantiaf0627/vox-profile-release (commit 85100e6), so that the models load
offline with this project's transformers, on any device. Changes:
  - The backbone is built from the base model's config files in the model folder. The official code downloads the base
    weights (openai/whisper-large-v3, microsoft/wavlm-large) first, but the checkpoint holds every weight.
  - LoRA is merged into the Whisper encoder's fc1/fc2 weights at load time, as loralib does in eval mode (in the
    release, LoRA B is all zeros, so the encoder stays exactly openai/whisper-large-v3's). No loralib, no speechbrain
    (its padding mask is all ones for a single clip) and no `.cuda()` calls.
  - Only the categorical emotion head is built; the checkpoint's other heads (17 detailed emotions, arousal, valence,
    dominance) are not loaded.
The official code is under the Responsible AI Source Code License v1.1. Its full text is in vox_profile_LICENSE.txt
next to this file; the use restrictions of its Section 3.2 apply to this code too. The weights are under OpenRAIL: their
model cards exclude, among others, commercial use and surveillance.

Used by scripts/evaluation/evaluate.py through load(model_dir, device); prepare(model_dir) is run by
scripts/common/download_model.py after the download.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from audio_io import SR, load_audio  # scripts/models/audio_io.py

MAX_SECONDS = 15  # the model cards: clips of at most 15 s (they were trained on clips of 3-15 s)
# Official classes, in output order: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, Other
LABELS = ["angry", "contempt", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "other"]
# Base model of each pretrain_model: Hugging Face repository, its latest commit on 2026-10-28, and the sub-folder of the
# model folder that holds its config files
BASE = {"whisper_large": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", "whisper-large-v3"),
        "wavlm_large": ("microsoft/wavlm-large", "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c", "wavlm-large")}
WHISPER_REPO = "tiantiaf/whisper-large-v3-msp-podcast-emotion"
DECODER = "backbone_model.decoder."  # Whisper's text decoder: in model.safetensors, never used for emotions
SLIM = "model_slim.safetensors"


def settings(model_dir):
    """The model's config.json (a dict) and the folder with its base model's config files."""
    if not (model_dir / "config.json").exists():
        sys.exit(f"ERROR: {model_dir / 'config.json'} not found; run python scripts/common/download_model.py")
    cfg = json.loads((model_dir / "config.json").read_text())
    if cfg.get("pretrain_model") not in BASE:
        sys.exit(f"ERROR: {model_dir / 'config.json'}: pretrain_model must be one of {', '.join(BASE)}")
    return cfg, model_dir / BASE[cfg["pretrain_model"]][2]


def prepare(model_dir):
    """Save the base model's config.json and preprocessor_config.json in the model folder, so that loading needs no
    network. For Whisper, also write model_slim.safetensors: model.safetensors without the text decoder, every tensor
    in float16 where that is exact (the frozen encoder holds openai's float16 weights) and float32 otherwise."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    model_dir = Path(model_dir)
    cfg, base = settings(model_dir)
    repo, revision, _ = BASE[cfg["pretrain_model"]]
    base.mkdir(exist_ok=True)
    for name in ["config.json", "preprocessor_config.json"]:
        if not (base / name).exists():
            shutil.copyfile(hf_hub_download(repo, name, revision=revision), base / name)
            print(f"  saved {(base / name).relative_to(model_dir)} from {repo} (commit {revision[:8]})")
        else:
            print(f"  {(base / name).relative_to(model_dir)} is already there")
    if cfg["pretrain_model"] != "whisper_large":
        return
    source, slim = model_dir / "model.safetensors", model_dir / SLIM
    if slim.exists():
        print(f"  {SLIM} is already there ({slim.stat().st_size / 1e9:.2f} GB); delete it to write it again")
        return
    print(f"  {source.name}: {source.stat().st_size / 1e9:.2f} GB; computing its SHA-256 ...")
    with open(source, "rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    # The download records the commit and the file's etag, which is its SHA-256
    record = model_dir / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
    commit, etag = record.read_text().split()[:2] if record.exists() else ("unknown", "")
    if len(etag) == 64 and etag != digest:
        sys.exit(f"ERROR: {source} does not match its Hugging Face checksum; delete it and download it again")
    tensors, dropped, half = {}, [], 0
    with safe_open(source, "pt") as f:
        for name in f.keys():
            if name.startswith(DECODER):
                dropped.append(f.get_slice(name).get_shape())
                continue
            t = f.get_tensor(name)
            if t.half().float().equal(t):  # exact in float16 (the encoder, and LoRA B: all zeros): store it so
                t, half = t.half(), half + 1
            tensors[name] = t
    removed = sum(int(np.prod(shape)) for shape in dropped)
    save_file(tensors, slim, metadata={
        "source": f"https://huggingface.co/{WHISPER_REPO}", "source_commit": commit, "source_file": source.name,
        "source_sha256": digest,
        "removed": f"{len(dropped)} tensors {DECODER}* ({removed / 1e6:.1f}M parameters): Whisper's text decoder, "
                   "which the emotion model never runs",
        "converted": f"{half} tensors stored as float16 because that is exact (t.half().float() equals t); the other "
                     f"{len(tensors) - half} (positions, LoRA A, heads) stay float32. Cast every tensor to float32 to "
                     "get the original values",
        "created_by": "emotion_recognition/scripts/models/vox_profile.py, prepare()"})
    print(f"  wrote {SLIM}: {slim.stat().st_size / 1e9:.2f} GB instead of {source.stat().st_size / 1e9:.2f} GB; "
          f"{len(dropped)} decoder tensors removed, {half} of {len(tensors)} tensors stored as float16 (exact), "
          f"{len(tensors) - half} as float32")


class EmotionModel(nn.Module):
    """WhisperWrapper / WavLMWrapper of the official code, reduced to the categorical emotion output. The parameter
    names are the official ones, so that the checkpoint loads with strict checking."""

    def __init__(self, cfg, base):
        from transformers import WavLMConfig, WavLMModel, WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        super().__init__()
        self.whisper = cfg["pretrain_model"] == "whisper_large"
        if self.whisper:  # official: WhisperModel with 750 positions (15 s); only its encoder runs
            config = WhisperConfig.from_pretrained(base, max_source_positions=750)
            self.backbone_model = nn.ModuleDict({"encoder": WhisperEncoder(config)})
        else:
            config = WavLMConfig.from_pretrained(base)
            self.backbone_model = WavLMModel(config)
            self.weights = nn.Parameter(torch.ones(config.num_hidden_layers + 1))  # one weight per hidden state
        dim = cfg["hidden_dim"]
        self.model_seq = nn.Sequential(nn.Conv1d(config.hidden_size, dim, 1), nn.ReLU(), nn.Dropout(0.1),
                                       nn.Conv1d(dim, dim, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(dim, dim, 1))
        self.emotion_layer = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, cfg["output_class_num"]))

    def forward(self, x, frames=None):
        if self.whisper:  # official: the encoder's last hidden state, after its final layer norm
            hidden = self.backbone_model.encoder(x).last_hidden_state
        else:
            # Official: the 25 hidden states (the input of each layer, then the output after the final layer norm, as
            # transformers 4.46 returned them), summed with softmax(weights)
            out = self.backbone_model(x, output_hidden_states=True)
            stacked = torch.stack([*out.hidden_states[:-1], out.last_hidden_state])
            norm_weights = F.softmax(self.weights, dim=-1)
            hidden = (norm_weights.unsqueeze(-1) * stacked.view(len(stacked), -1)).sum(0).view(stacked.shape[1:])
        hidden = self.model_seq(hidden.transpose(1, 2)).transpose(1, 2)  # pointwise convolutions: (1, frames, 256)
        return self.emotion_layer(hidden[:, :frames].mean(1))  # mean over the clip's frames, then the classifier


def build(model_dir, device):
    """The model from model_dir only, in eval mode on `device`."""
    cfg, base = settings(model_dir)
    with torch.device("meta"):  # no random initialisation: every weight comes from the checkpoint
        model = EmotionModel(cfg, base)
    path = model_dir / SLIM if (model_dir / SLIM).exists() else model_dir / "model.safetensors"
    with safe_open(path, "pt") as f:  # float32, as the official code runs; without Whisper's decoder
        weights = {k: f.get_tensor(k).float() for k in f.keys() if not k.startswith(DECODER)}
    # LoRA (Whisper encoder layers 17-31): loralib.Linear in eval mode uses W + (B @ A) * lora_alpha / r, with r =
    # lora_rank and loralib's default lora_alpha = 1 (the official code sets only r)
    for a in [k for k in weights if k.endswith(".lora_A")]:
        layer = a.removesuffix(".lora_A")
        weights[f"{layer}.weight"] = weights[f"{layer}.weight"] + (
            weights.pop(f"{layer}.lora_B") @ weights.pop(a)) * (1 / cfg["lora_rank"])
    # The checkpoint's other heads (and Whisper's unused layer weights) are left out; everything else must match
    kept = {k.split(".")[0] for k in model.state_dict()}
    model.load_state_dict({k: v for k, v in weights.items() if k.split(".")[0] in kept}, strict=True, assign=True)
    return model.to(device).eval()


class Predictor:
    """Runs a Vox-Profile model on one clip and returns its probabilities over `labels`. A clip that the device cannot
    run runs on a CPU copy of the model instead (counted in on_cpu)."""

    def __init__(self, model_dir, device):
        import transformers
        from transformers import Wav2Vec2FeatureExtractor, WhisperFeatureExtractor

        self.model_dir, self.device = model_dir, device
        self.model, self.cpu_model, self.on_cpu = build(model_dir, device), None, 0
        base = settings(model_dir)[1]
        # The base model's audio front end, as in the official code (Whisper's with a 15-s window instead of 30 s)
        self.extractor = (WhisperFeatureExtractor.from_pretrained(base, chunk_length=MAX_SECONDS) if self.model.whisper
                          else Wav2Vec2FeatureExtractor.from_pretrained(base))
        self.labels = LABELS
        self.versions = {"transformers": transformers.__version__}
        self.notes = [
            ("Each clip (at most 15 s) becomes Whisper log-mel features padded with silence to 15 s, the model's fixed "
             "input; the encoder output is averaged over the clip's own frames only, as in the official code."
             if self.model.whisper else
             "Each clip (at most 15 s) is normalised to zero mean and unit variance and fed whole, without padding; "
             "the 25 hidden states are mixed with the model's learned layer weights and averaged over time, as in the "
             "official code.") + " float32 on every device.",
            "The model card warns that clips shorter than 3 s give unreliable predictions (it was trained on clips of "
            "3-15 s); clips are used as they are, not padded to 3 s.",
        ]

    def __call__(self, path):
        wav = load_audio(path)[:MAX_SECONDS * SR]
        if self.model.whisper:  # official: features of 15 s; the mean covers the clip's own encoder frames
            x = self.extractor(wav, sampling_rate=SR, max_length=MAX_SECONDS * SR, return_tensors="pt").input_features
            frames = (len(wav) // 160 - 1) // 2 + 1  # 160 samples per mel frame, then a stride-2 convolution
        else:  # official: the normalised samples of the clip; the mean covers all frames
            x = self.extractor(wav, sampling_rate=SR, padding=True, return_tensors="pt").input_values
            frames = None
        with torch.inference_mode():
            try:
                logits = self.model(x.to(self.device), frames)
            except NotImplementedError:  # an operation the device does not support
                if self.cpu_model is None:
                    self.cpu_model = build(self.model_dir, torch.device("cpu"))
                self.on_cpu += 1
                logits = self.cpu_model(x, frames)
        return logits.softmax(-1)[0].cpu().numpy().astype(np.float64)


def load(model_dir, device):
    """A Vox-Profile model from the files in model_dir (never the network), in eval mode on `device`."""
    model_dir = Path(model_dir)
    cfg, base = settings(model_dir)
    weights = SLIM if cfg["pretrain_model"] == "whisper_large" else "model.safetensors"
    if not (model_dir / weights).exists() and not (model_dir / "model.safetensors").exists():
        sys.exit(f"ERROR: {model_dir / weights} not found; run python scripts/common/join_parts.py (or "
                 "download_model.py)")
    if not (base / "preprocessor_config.json").exists():
        sys.exit(f"ERROR: {base} not found; run python scripts/common/download_model.py")
    return Predictor(model_dir, device)
