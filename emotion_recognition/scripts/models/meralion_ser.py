"""MERaLiON-SER-v1 (Hugging Face MERaLiON/MERaLiON-SER-v1): 7 emotions from the Whisper-medium encoder (LoRA merged)
with an ECAPA-TDNN head and attention pooling, about 313M parameters in use.

Files in models/meralion_ser_v1/ (the Hugging Face repository, plus one file added by prepare()):
    model.safetensors                  weights, 3.08 GB float32 (770M parameters; 457M of them are the Whisper decoder,
                                       which is never used)
    config.json, configuration_ser_whisper_ecapa.py, modeling_ser_whisper_ecapa.py   the model code (remote code)
    preprocessor_config.json           Whisper feature extractor: 80 mel bins, every clip padded or cut to 30 s
    whisper-medium/config.json         openai/whisper-medium's config.json, saved by prepare(): the model code builds
                                       its encoder from it, so loading needs no network
    MERaLiON-Public-Licence-SER-V1.pdf the licence (MIT with attribution terms); keep it with the weights

A clip is fed in as in the model card's CPU example: Whisper features of the clip padded with silence to 30 s,
float32. The model ignores attention masks and pools over the whole 30 s window, padding included; that is its
official behaviour and is kept.

Used by scripts/evaluation/evaluate.py through load(model_dir, device); prepare(model_dir) is run by
scripts/common/download_model.py after the download.
"""
import shutil
import sys

import torch

from audio_io import SR, load_audio  # scripts/models/audio_io.py

WHISPER_REPO = "openai/whisper-medium"
WHISPER_REVISION = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"  # its latest commit on 2026-10-28


def prepare(model_dir):
    """Save openai/whisper-medium's config.json in model_dir/whisper-medium/. The model code calls
    WhisperConfig.from_pretrained(config.whisper_name) with whisper_name "openai/whisper-medium", a download at every
    load; load() points whisper_name at this folder instead."""
    from huggingface_hub import hf_hub_download

    target = model_dir / "whisper-medium" / "config.json"
    if target.exists():
        print(f"  {target.relative_to(model_dir)} is already there")
        return
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(hf_hub_download(WHISPER_REPO, "config.json", revision=WHISPER_REVISION), target)
    print(f"  saved {target.relative_to(model_dir)} from {WHISPER_REPO} (commit {WHISPER_REVISION[:8]})")


def build(model_dir, device):
    """The model from model_dir only, in eval mode on `device`. The model is built from its class and then gets the
    weights, instead of from_pretrained(): transformers 5 builds models on the meta device, where the model's own code
    fails (it reads kernel sizes from a tensor with int())."""
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True,
                                        whisper_name=str(model_dir / "whisper-medium"))
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForAudioClassification"], model_dir)
    model = model_class(config)  # random weights, on the CPU
    model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=True)  # every weight, or an error
    return model.to(device).eval()


class Predictor:
    """Runs MERaLiON-SER-v1 on one clip and returns its probabilities over `labels`. A clip that the device cannot run
    runs on a CPU copy of the model instead (counted in on_cpu)."""

    def __init__(self, model_dir, device):
        import transformers
        from transformers import WhisperFeatureExtractor

        self.model_dir, self.device = model_dir, device
        self.model, self.cpu_model, self.on_cpu = build(model_dir, device), None, 0
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        id2label = self.model.config.id2label
        self.labels = [id2label[k] for k in sorted(id2label, key=int)]  # already the project's names
        self.versions = {"transformers": transformers.__version__}
        self.notes = [
            "Each clip becomes Whisper log-mel features padded with silence to 30 s (as in the model card); the model "
            "ignores attention masks and pools over the whole 30 s window, padding included, which is its official "
            "behaviour. float32 on every device, as released.",
        ]

    def __call__(self, path):
        features = self.extractor(load_audio(path), sampling_rate=SR, return_tensors="pt").input_features
        with torch.inference_mode():
            try:
                logits = self.model(input_features=features.to(self.device))["logits"]
            except NotImplementedError:  # an operation the device does not support
                if self.cpu_model is None:
                    self.cpu_model = build(self.model_dir, torch.device("cpu"))
                self.on_cpu += 1
                logits = self.cpu_model(input_features=features)["logits"]
        return logits.softmax(-1)[0].float().cpu().numpy()


def load(model_dir, device):
    """MERaLiON-SER-v1 from the files in model_dir (never the network), in eval mode on `device`."""
    for need, fix in [("model.safetensors", "python scripts/common/join_parts.py after git pull, or download_model.py"),
                      ("whisper-medium/config.json", "git pull, or python scripts/common/download_model.py")]:
        if not (model_dir / need).exists():
            sys.exit(f"ERROR: {model_dir / need} not found; run {fix} (see models/README.md)")
    return Predictor(model_dir, device)
