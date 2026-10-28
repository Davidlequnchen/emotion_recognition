"""Melati (Hugging Face Scicom-intl/Melati-Emotion-Whisper-Large-V3): 7 emotions from the Whisper-large-v3 encoder
with a small head on its frozen features (every encoder layer mean-pooled over the clip, a learned mix of the layers,
one hidden layer), 637.3M parameters.

Files in models/melati_emotion/ (the Hugging Face repository, unchanged):
    model.safetensors                  weights, 1.28 GB: the encoder in float16, the head in float32
    config.json, configuration_melati_head.py, modeling_melati_head.py   the model code (remote code, self-contained)
    preprocessor_config.json           Whisper feature extractor: 128 mel bins, every clip padded or cut to 30 s
    melati_emotion.py                  the official inference code, which this loader follows
    LICENSE                            non-commercial research only; it must stay with the weights

A clip is fed in as melati_emotion.py's Emotion.classify does: at most 30 s, clips under 0.1 s padded, Whisper
features padded to 30 s, and the encoder layers averaged over the clip's own frames only (50 per second). The
prediction is the most probable class of the raw softmax. Emotion.label()'s thresholds are not applied: they were
chosen for precise tagging of podcasts and would turn most clips into neutral.

Used by scripts/evaluation/evaluate.py through load(model_dir, device). It needs no prepare(): the model code builds
the encoder from the config in config.json, without the network.
"""
import sys

import numpy as np
import torch

from audio_io import SR, load_audio  # scripts/models/audio_io.py

# float16 encoder on a GPU (MPS, CUDA), as on the model card; the CPU always runs float32. On the Mac, MPS float16 gave
# the same prediction as CPU float32 for 72 of 72 test clips (largest probability difference 0.002) and is about 1.35
# times as fast as MPS float32
FLOAT16_ON_GPU = True


def build(model_dir, device):
    """The model from model_dir only, in eval mode on `device`: float16 on a GPU (MPS, CUDA) if FLOAT16_ON_GPU, else
    float32. The head stays float32 either way (the model code keeps it so)."""
    from transformers import AutoModel

    dtype = torch.float16 if FLOAT16_ON_GPU and device.type != "cpu" else torch.float32
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype, local_files_only=True)
    return model.to(device).eval()


class Predictor:
    """Runs Melati on one clip and returns its probabilities over `labels`. A clip that the device cannot run runs on
    a CPU copy of the model instead (counted in on_cpu)."""

    def __init__(self, model_dir, device):
        import transformers
        from transformers import AutoFeatureExtractor

        self.model_dir, self.device = model_dir, device
        self.model, self.cpu_model, self.on_cpu = build(model_dir, device), None, 0
        self.extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        self.labels = list(self.model.config.labels)  # already the project's names
        self.versions = {"transformers": transformers.__version__}
        precision = "float16 encoder" if self.model.encoder.conv1.weight.dtype == torch.float16 else "float32"
        self.notes = [
            "Each clip (at most 30 s) becomes Whisper-large-v3 log-mel features padded to 30 s; the encoder layers are "
            "averaged over the clip's own frames only (50 per second), as in the official melati_emotion.py. "
            f"Precision: {precision}, float32 head.",
            "The prediction is the most probable of the raw probabilities. The official thresholds (melati_emotion.py, "
            "label()) are not applied: they were set for precise tagging of podcasts, would turn most clips into "
            "neutral, and never return disgusted.",
        ]

    def __call__(self, path):
        clip = load_audio(path)[: 30 * SR]
        if len(clip) < SR // 10:  # under 0.1 s: padded, as in melati_emotion.py
            clip = np.pad(clip, (0, SR // 10 - len(clip)))
        features = self.extractor([clip], sampling_rate=SR, return_tensors="pt").input_features
        frames = torch.tensor([max(1, min(1500, int(len(clip) / SR * 50)))])  # 50 encoder frames per second
        with torch.inference_mode():
            try:
                logits = self.model(features.to(self.device), num_frames=frames.to(self.device)).logits
            except NotImplementedError:  # an operation the device does not support
                if self.cpu_model is None:
                    self.cpu_model = build(self.model_dir, torch.device("cpu"))
                self.on_cpu += 1
                logits = self.cpu_model(features, num_frames=frames).logits
        return logits.softmax(-1)[0].float().cpu().numpy()


def load(model_dir, device):
    """Melati from the files in model_dir (never the network), in eval mode on `device`."""
    if not (model_dir / "model.safetensors").exists():
        sys.exit(f"ERROR: {model_dir / 'model.safetensors'} not found; run python scripts/common/join_parts.py after "
                 "git pull, or download_model.py (see models/README.md)")
    return Predictor(model_dir, device)
