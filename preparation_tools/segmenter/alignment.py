"""Word timestamps by CTC forced alignment of the existing transcript (doc §2).

The checkpoint is loaded from preparation_tools/model_weights/ only: its SHA-256 is checked first, then
torch.hub is pointed at that folder so torchaudio finds the file and never touches the network.
Words that cannot be spelled use a wildcard column (best non-blank score per frame), as WhisperX does.
"""
import hashlib
import json
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

from .textnorm import WILDCARD

SAMPLE_RATE = 16000
HOP = 320            # wav2vec 2.0 frame step in samples (20 ms)
RECEPTIVE = 400      # samples seen by one frame


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Aligner:
    """One CTC alignment model from model_weights/manifest.json (currently: torchaudio bundles)."""

    def __init__(self, model_key, weights_dir, device="cpu", chunk_s=60.0, context_s=2.0):
        weights_dir = Path(weights_dir).resolve()
        info = json.loads((weights_dir / "manifest.json").read_text(encoding="utf-8"))["models"][model_key]
        path = weights_dir / info["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint {path}. Run download.py (internet) and copy it here.")
        if sha256_of(path) != info["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}; delete it and run download.py again.")
        if info["type"] != "torchaudio_bundle":
            raise ValueError(f"Unsupported model type {info['type']!r} for {model_key}")
        bundle = getattr(torchaudio.pipelines, info["bundle"])
        if Path(info["path"]).parent.name != "checkpoints" or not info["url"].endswith("/" + info["filename"]):
            raise ValueError("The checkpoint must sit in checkpoints/ under its upstream file name.")
        previous = torch.hub.get_dir()
        torch.hub.set_dir(str(weights_dir))      # torchaudio reads <hub dir>/checkpoints/<file name>
        try:
            self.model = bundle.get_model().to(device).eval()
        finally:
            torch.hub.set_dir(previous)
        self.labels = bundle.get_labels()        # ('-', '|', 'E', 'T', ...): blank first, then '|' and letters
        self.index = {label: i for i, label in enumerate(self.labels)}
        self.key, self.device = model_key, device
        self.chunk = int(chunk_s * SAMPLE_RATE) // HOP * HOP
        self.context = max(RECEPTIVE, int(context_s * SAMPLE_RATE) // HOP * HOP)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def emissions(self, wave):
        """Log-probabilities (frames, labels) for a 1-D 16 kHz tensor. Long audio is processed in chunks with
        context on both sides; the frame count equals that of a single pass."""
        def run(piece):
            logits, _ = self.model(piece[None].to(self.device))
            return torch.log_softmax(logits[0].float(), dim=-1).cpu()

        n = wave.numel()
        if n <= self.chunk + 2 * self.context:
            return run(wave)
        total = (n - RECEPTIVE) // HOP + 1
        parts, done = [], 0
        for start in range(0, n, self.chunk):
            a, b = max(0, start - self.context), min(n, start + self.chunk + self.context)
            frames = run(wave[a:b])
            first = (start - a) // HOP
            count = min(self.chunk // HOP, total - done)
            parts.append(frames[first:first + count])
            done += count
            if done >= total:
                break
        return torch.cat(parts)

    def align(self, wave, words):
        """Add start, end (seconds) and confidence to each Word. Returns (ok, reason)."""
        emission = self.emissions(wave)
        wildcard = emission[:, 1:].max(dim=1).values          # best non-blank score per frame
        emission = torch.cat([emission, wildcard[:, None]], dim=1)
        wild_id, separator = emission.shape[1] - 1, self.index["|"]

        targets, owners = [], []                              # owners[k]: word index of token k, None for '|'
        for w_index, word in enumerate(words):
            for unit in word.units:
                if targets:
                    targets.append(separator)
                    owners.append(None)
                letters = [wild_id] if unit == WILDCARD else [self.index[ch] for ch in unit]
                targets += letters
                owners += [w_index] * len(letters)
        if not targets:
            return False, "no alignable text"
        repeats = sum(1 for x, y in zip(targets, targets[1:]) if x == y)
        if len(targets) + repeats > emission.shape[0]:
            return False, "transcript too long for the audio"

        path, scores = F.forced_align(emission[None], torch.tensor([targets], dtype=torch.int32), blank=0)
        spans = F.merge_tokens(path[0], scores[0].exp())
        if len(spans) != len(targets):
            return False, "alignment returned an unexpected number of tokens"
        seconds = wave.numel() / emission.shape[0] / SAMPLE_RATE   # about 0.02 s per frame
        for w_index, word in enumerate(words):
            mine = [s for s, owner in zip(spans, owners) if owner == w_index]
            word.start = round(mine[0].start * seconds, 4)
            word.end = round(mine[-1].end * seconds, 4)
            word.confidence = round(sum(s.score for s in mine) / len(mine), 4)   # mean letter score
        return True, "ok"
