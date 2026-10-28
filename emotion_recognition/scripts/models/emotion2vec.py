"""
Loader of the emotion2vec+ models (FunASR), used by scripts/evaluation/evaluate.py.

load(model_dir, device) builds the model from the files of an emotion2vec+ release in model_dir (model.pt,
config.yaml, configuration.json, tokens.txt; see models/README.md), with the same settings as
scripts/emotion2vec/inference.py, and returns a Predictor. Called with the path of an audio file, the Predictor
returns the probabilities of the model's 9 classes in the order of tokens.txt. FunASR reads the file itself: it
averages stereo channels and resamples the audio to 16 kHz. Loading never touches the network: FunASR downloads
nothing when it is given a local folder, and disable_update skips its version check.
"""
import logging
from pathlib import Path

import numpy as np
import torch

MODEL_FILES = ["model.pt", "config.yaml", "configuration.json", "tokens.txt"]
# Added to the training-data caveat of every report and summary of an emotion2vec+ model
TRAINING_NOTE = ("emotion2vec+ seed, the first model in the chain that led to emotion2vec+ large, was fine-tuned on "
                 "EmoBox data; large was later trained on about 40k hours of pseudo-labelled data whose sources are "
                 "not published, so earlier exposure to other datasets cannot be ruled out completely.")


class Predictor:
    """Runs the model on one clip and returns its class probabilities. MPS cannot run emotion2vec's first
    convolution on clips longer than about 20 s (more than 65,536 output steps), so such clips run on a CPU
    copy of the model instead (counted in on_cpu)."""

    def __init__(self, model_dir, device):
        import funasr  # imported here because it takes a few seconds

        logging.getLogger().setLevel(logging.ERROR)  # importing FunASR turns on INFO logs; keep the output readable
        self.model_dir = model_dir
        self.model, self.cpu_model, self.on_cpu = self.build(device), None, 0
        # tokens.txt has one class per line, e.g. "生气/angry": keep the English part ("<unk>" stays as it is)
        self.labels = [t.split("/")[-1] for t in (model_dir / "tokens.txt").read_text(encoding="utf-8").split()]
        self.versions = {"FunASR": funasr.__version__}
        self.notes = ["One prediction per clip (FunASR, granularity utterance); FunASR averages stereo channels and "
                      "resamples the audio to 16 kHz."]
        if device.type == "mps":
            self.notes.append("Clips longer than about 20 s run on a CPU copy of the model, because MPS cannot run "
                              "the model's first convolution on them.")

    def build(self, device):
        """The FunASR model on `device`: it reads configuration.json, which names the other files."""
        from funasr import AutoModel

        return AutoModel(
            model=str(self.model_dir),
            device=str(device),
            ncpu=torch.get_num_threads(),  # keep PyTorch's CPU thread count (FunASR would lower it to 4)
            excludes="modality_encoders.AUDIO.decoder.",  # pre-training decoder: not in model.pt, unused for inference
            disable_update=True,  # no online version check
            disable_pbar=True,
        )

    def __call__(self, path):
        with torch.inference_mode():
            try:
                result = self.model.generate(path, granularity="utterance", extract_embedding=False)
            except NotImplementedError:  # too long for MPS: run the clip on the CPU
                if self.cpu_model is None:
                    self.cpu_model = self.build(torch.device("cpu"))
                self.on_cpu += 1
                result = self.cpu_model.generate(path, granularity="utterance", extract_embedding=False)
        return np.asarray(result[0]["scores"], dtype=np.float64)


def load(model_dir, device):
    """The Predictor of the emotion2vec+ model in model_dir, in eval mode on `device`."""
    model_dir = Path(model_dir)
    missing = [f for f in MODEL_FILES if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{model_dir}: {', '.join(missing)} not found. Download the model first: "
                                "python scripts/common/download_model.py")
    return Predictor(model_dir, device)
