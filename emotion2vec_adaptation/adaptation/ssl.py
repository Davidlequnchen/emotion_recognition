"""Unlabeled domain adaptation by masked EMA-teacher feature regression.

This is a new adaptation objective inspired by emotion2vec/data2vec, not a
resumption of their original pretraining run. The public release lacks the
original EMA teacher, decoder and optimizer states. We initialize a fresh teacher
from the local encoder and train a small predictor on masked latent frames only.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbone import EmotionBackbone, masked_mean, valid_frame_mask


def sample_span_mask(valid: Tensor, probability: float, span_length: int) -> Tensor:
    """Mask contiguous spans on valid frames, leaving at least one visible frame.

    Each example receives a deterministic target count given its valid length;
    randomized span order and boundaries provide the stochastic corruption. No
    padding is selected. Works with very short curated examples as well.
    """
    if not 0 < probability < 1 or span_length < 1:
        raise ValueError("mask_probability must be in (0,1); mask_length must be positive.")
    # Short clips have only a few hundred latent frames. Sampling this tiny
    # boolean mask on CPU avoids one CUDA launch/synchronization per span;
    # only the finished mask is copied back to the encoder's device.
    mask_device = valid.device
    valid = valid.detach().cpu()
    output = torch.zeros_like(valid, dtype=torch.bool)
    for row in range(valid.shape[0]):
        indices = valid[row].nonzero(as_tuple=True)[0]
        count = indices.numel()
        if count < 2:
            raise ValueError("Masked SSL needs at least two valid encoder frames per clip.")
        target = min(count - 1, max(1, round(count * probability)))
        order = torch.randperm(count, device=valid.device)
        selected = 0
        for start in order.tolist():
            span = indices[start:min(count, start + span_length)]
            span = span[~output[row, span]][:target - selected]
            output[row, span] = True
            selected += span.numel()
            if selected == target:
                break
    return output.to(mask_device)


def temporal_instance_normalize(features: Tensor, valid: Tensor) -> Tensor:
    """Per-channel normalization using only the utterance's valid frames."""
    weights = valid.unsqueeze(-1).to(features.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1)
    mean = (features * weights).sum(dim=1, keepdim=True) / count
    variance = ((features - mean).square() * weights).sum(dim=1, keepdim=True) / count
    return ((features - mean) * torch.rsqrt(variance + 1e-5)) * weights


TARGETS = ("final", "ffn_mean")


class MaskedTeacherObjective(nn.Module):
    """Student encoder + predictor with a frozen, always-eval EMA teacher.

    Call ``update_teacher`` once *after each optimizer step*, not after each
    microbatch. The predictor is newly initialized; it is not exported as part of
    the replacement encoder. ``mask_probability`` is the target fraction of valid
    frames covered, not a span-start probability.

    ``target`` picks what the masked frames regress onto:
    - ``final``: the teacher's last block output, layer-normalised per frame. The
      fine-tuned large model's last block is almost constant over a clip, so this
      mostly asks the student to recover the clip's representation from context.
    - ``ffn_mean``: the mean of the top ``target_top_layers`` FFN outputs, each
      normalised over the clip's valid frames (data2vec 2.0 / emotion2vec). From
      the fine-tuned large model this target is not predictable (docs/guide.md).
    ``utterance_weight`` adds the emotion2vec utterance loss: the student's extra
    tokens are pooled first, then compared with the layer-normalised mean of the
    teacher's last block over valid frames.
    """

    def __init__(
        self, backbone: EmotionBackbone, mask_probability: float = 0.55,
        mask_length: int = 5, ema_decay: float = 0.999,
        target_top_layers: int = 8, predictor_hidden: int = 1024,
        freeze_frontend: bool = True, utterance_weight: float = 0.0,
        target: str = "ffn_mean", frame_weight: float = 1.0,
    ):
        super().__init__()
        if not 0 < mask_probability < 1 or mask_length < 1:
            raise ValueError("Invalid SSL mask settings.")
        if not 0 <= ema_decay < 1 or target_top_layers < 1 or predictor_hidden < 1:
            raise ValueError("Invalid SSL teacher/predictor settings.")
        for name, weight in (("utterance_weight", utterance_weight), ("frame_weight", frame_weight)):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if frame_weight + utterance_weight == 0:
            raise ValueError("frame_weight and utterance_weight cannot both be zero.")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        self.student = backbone
        self.student.requires_grad_(True)
        if freeze_frontend:
            self.student.model.modality_encoders["AUDIO"].local_encoder.requires_grad_(False)
        self.teacher = copy.deepcopy(backbone).requires_grad_(False).eval()
        dim = backbone.embedding_dim
        self.predictor = nn.Sequential(nn.Linear(dim, predictor_hidden), nn.GELU(), nn.Linear(predictor_hidden, dim))
        self.predictor.to(next(backbone.parameters()).device)
        self.mask_probability = float(mask_probability)
        self.mask_length = int(mask_length)
        self.ema_decay = float(ema_decay)
        self.target_top_layers = int(target_top_layers)
        self.utterance_weight = float(utterance_weight)
        self.frame_weight = float(frame_weight)
        self.target = target
        if target == "ffn_mean" and target_top_layers > len(backbone.model.blocks):
            raise ValueError("target_top_layers exceeds available shared transformer blocks.")
        if utterance_weight > 0 and backbone.num_extra_tokens < 1:
            raise ValueError("Utterance extra-token loss needs an encoder with extra tokens.")

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, decay: float | None = None) -> None:
        decay = self.ema_decay if decay is None else float(decay)
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0,1).")
        student_params = dict(self.student.named_parameters())
        for name, parameter in self.teacher.named_parameters():
            parameter.mul_(decay).add_(student_params[name].detach(), alpha=1 - decay)
        student_buffers = dict(self.student.named_buffers())
        for name, buffer in self.teacher.named_buffers():
            buffer.copy_(student_buffers[name])

    def forward(self, sources: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, dict]:
        with torch.no_grad():
            teacher_result = self.teacher.frames(sources, padding_mask)
            teacher_frames = teacher_result["x"].float()
            valid = valid_frame_mask(teacher_frames, teacher_result["padding_mask"])
            extra = self.teacher.num_extra_tokens
            if self.target == "final":
                targets = F.layer_norm(teacher_frames, teacher_frames.shape[-1:])
            else:
                layers = teacher_result["layer_results"][-self.target_top_layers:]
                targets = torch.stack([
                    temporal_instance_normalize(layer[:, extra:].float(), valid) for layer in layers
                ]).mean(dim=0)
            teacher_pooled = masked_mean(teacher_frames, valid)
            target_utterance = F.layer_norm(teacher_pooled, teacher_pooled.shape[-1:])
            mask = sample_span_mask(valid, self.mask_probability, self.mask_length)

        student_result = self.student.frames(
            sources, padding_mask, mask=True, precomputed_mask=mask,
            remove_extra_tokens=False,
        )
        student_frames = student_result["x"][:, extra:]
        predictions = self.predictor(student_frames).float()
        # Index before reduction: padding and visible frames have exactly zero
        # direct contribution, irrespective of their count in the batch.
        frame_loss = F.mse_loss(predictions[mask], targets[mask])
        utterance_loss = frame_loss.new_zeros(())
        if self.utterance_weight:
            pooled_tokens = student_result["x"][:, :extra].mean(dim=1)
            utterance_loss = F.mse_loss(self.predictor(pooled_tokens).float(), target_utterance)
        loss = self.frame_weight * frame_loss + self.utterance_weight * utterance_loss
        with torch.no_grad():
            # Learnability baselines on the same masked frames: predicting zero, and
            # predicting the batch's per-channel mean target. A useful model beats both.
            masked_targets = targets[mask]
            zero_loss = self.frame_weight * masked_targets.square().mean()
            constant_loss = self.frame_weight * masked_targets.var(dim=0, unbiased=False).mean()
            if self.utterance_weight:
                zero_loss = zero_loss + self.utterance_weight * target_utterance.square().mean()
                constant_loss = constant_loss + self.utterance_weight * target_utterance.var(dim=0, unbiased=False).mean()
        # target_std cannot detect collapse: normalisation pins it near a fixed value
        # whatever the representation does. teacher_pooled_std is measured before any
        # normalisation, across the clips of the batch.
        target_std = targets[valid].std(dim=0, unbiased=False).mean()
        prediction_std = predictions[valid].std(dim=0, unbiased=False).mean()
        stats = {
            "loss": float(loss.detach()), "frame_loss": float(frame_loss.detach()),
            "utterance_loss": float(utterance_loss.detach()),
            "zero_loss": float(zero_loss), "constant_loss": float(constant_loss),
            "masked_frames": int(mask.sum()), "valid_frames": int(valid.sum()),
            "target_std": float(target_std), "prediction_std": float(prediction_std.detach()),
        }
        if len(sources) > 1:
            stats["teacher_pooled_std"] = float(teacher_pooled.std(dim=0, unbiased=False).mean())
        return loss, stats
