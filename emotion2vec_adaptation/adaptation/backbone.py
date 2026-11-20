"""Offline, differentiable access to the existing Emotion2Vec+ encoder.

FunASR's public model contains feature extraction but no pretraining loss or
trained teacher/decoder. This adapter deliberately uses only its encoder. Audio
must already be mono 16 kHz and normalized *before* right padding, as in the
existing benchmark's FunASR inference implementation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
import yaml


def state_fingerprint(state: Mapping[str, Tensor]) -> str:
    """Content identity including names, shapes, dtypes and all tensor bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def valid_frame_mask(features: Tensor, padding_mask: Tensor | None) -> Tensor:
    """Return True for actual frames; padding never contributes to pooling/loss."""
    if padding_mask is None:
        return torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    if padding_mask.shape != features.shape[:2]:
        raise ValueError("Frame padding mask does not match encoder output.")
    valid = ~padding_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("An audio clip produced no valid encoder frames.")
    return valid


def masked_mean(features: Tensor, valid: Tensor) -> Tensor:
    """Mean over valid frames for each utterance."""
    weights = valid.unsqueeze(-1).to(features.dtype)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class EmotionBackbone(nn.Module):
    """An encoder without a classification head or unused SSL decoder."""

    def __init__(self, model: nn.Module, model_config_hash: str = ""):
        super().__init__()
        self.model = model
        self.embedding_dim = int(model.cfg.embed_dim)
        self.model_config_hash = model_config_hash

    @classmethod
    def load(cls, model_dir: str | Path, device: str | torch.device = "cpu") -> "EmotionBackbone":
        """Load local files strictly, with no hub resolution or network access.

        Only the published classification projection is removed. Missing or
        unexpected encoder weights cause an error instead of a partial load.
        ``weights_only`` avoids arbitrary pickle execution in local checkpoints.
        """
        from funasr.models.emotion2vec.model import Emotion2vec

        model_dir = Path(model_dir).resolve()
        for name in ("config.yaml", "model.pt"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Existing local model file missing: {model_dir / name}")
        config_bytes = (model_dir / "config.yaml").read_bytes()
        config = yaml.safe_load(config_bytes)
        if config.get("model") != "Emotion2vec":
            raise ValueError("model_dir must contain a FunASR Emotion2vec configuration.")
        model_conf = config["model_conf"]
        # The release omits these weights. Do not instantiate a random decoder.
        model_conf["modalities"]["audio"]["decoder"] = None
        model = Emotion2vec(model_conf=model_conf, vocab_size=-1)
        checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        raw_state = checkpoint.get("model", checkpoint)
        state = {}
        for name, tensor in raw_state.items():
            name = name.removeprefix("d2v_model.")
            if name in {"proj.weight", "proj.bias"}:
                continue
            state[name] = tensor
        model.load_state_dict(state, strict=True)
        return cls(model, hashlib.sha256(config_bytes).hexdigest()).to(device).eval()

    @property
    def fingerprint(self) -> str:
        """Current encoder identity; recomputed so training cannot leave it stale."""
        return state_fingerprint(self.model.state_dict())

    @property
    def num_extra_tokens(self) -> int:
        return int(self.model.modality_encoders["AUDIO"].modality_cfg.num_extra_tokens)

    def frames(
        self, sources: Tensor, padding_mask: Tensor | None = None,
        *, mask: bool = False, precomputed_mask: Tensor | None = None,
        remove_extra_tokens: bool = True,
    ) -> dict:
        """Return contextual frame features and native teacher layer outputs.

        Native ``layer_results`` retain the extra token prefix even when ``x``
        removes it. The SSL objective handles that explicitly.
        """
        return self.model(
            sources, padding_mask=padding_mask, mask=mask,
            features_only=True, precomputed_mask=precomputed_mask,
            remove_extra_tokens=remove_extra_tokens,
        )

    def embed(self, sources: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Option A: the last block's output averaged over valid frames, (batch, dim)."""
        result = self.frames(sources, padding_mask)
        features = result["x"]
        return masked_mean(features, valid_frame_mask(features, result["padding_mask"]))

    def blocks(self) -> list[nn.Module]:
        """Every Transformer block in order: the audio prenet blocks (4 in large), then the shared blocks (8)."""
        return [*self.model.modality_encoders["AUDIO"].context_encoder.blocks, *self.model.blocks]

    def layer_embeddings(self, sources: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Option B input: each block's output averaged over valid frames, (batch, blocks, dim).

        FunASR's ``layer_results`` are FFN outputs (SSL targets), not block outputs, so forward hooks
        capture each block's output; the extra-token prefix is dropped. The last row equals ``embed``.
        """
        outputs = []
        hooks = [block.register_forward_hook(lambda module, args, output: outputs.append(output[0]))
                 for block in self.blocks()]
        try:
            result = self.frames(sources, padding_mask)
        finally:
            for hook in hooks:
                hook.remove()
        if len(outputs) != len(hooks):
            raise RuntimeError("A block was skipped (layerdrop); extract layer embeddings in eval mode.")
        valid = valid_frame_mask(result["x"], result["padding_mask"])
        extra = self.num_extra_tokens
        return torch.stack([masked_mean(output[:, extra:], valid) for output in outputs], dim=1)

    def embed_each(self, sources: Tensor, padding_mask: Tensor | None = None, layers: bool = False) -> Tensor:
        """Run the encoder on one clip at a time with its padding cut off.

        FunASR's convolutional positional encoder also reads padded frames, so a padded clip's embedding depends
        on the other clips in its batch. One clip per forward pass gives every clip the same embedding as the
        benchmark's single-clip inference, whatever the batch.
        """
        lengths = [sources.shape[1]] * len(sources) if padding_mask is None else (~padding_mask).sum(dim=1).tolist()
        embed = self.layer_embeddings if layers else self.embed
        return torch.cat([embed(sources[row:row + 1, :length]) for row, length in enumerate(lengths)])

    def forward(self, sources: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        return self.embed(sources, padding_mask)

    def encoder_state_dict(self) -> dict[str, Tensor]:
        """Portable encoder state; classifier and predictor are never included."""
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def load_encoder(self, state: Mapping[str, Tensor]) -> None:
        self.model.load_state_dict(state, strict=True)
