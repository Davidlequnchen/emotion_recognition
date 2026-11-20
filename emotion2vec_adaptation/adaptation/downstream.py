"""Downstream check during SSL: does the adapted encoder still separate the labelled emotions?

Every check embeds the same labelled training and validation clips with the current student (one clip at a time),
fits a fresh linear head and reports validation UAR, next to representation statistics measured before any
normalisation. SSL loss alone cannot say whether emotion information survives; this check can. The labelled test
split is never read here.
"""
import torch
from torch.nn import functional as F

from .audio import AudioReader, AudioSettings
from .data import select_training_subset
from utilities.hardware import training_autocast
from utilities.runtime import classification_metrics

HEAD_STEPS = 300            # full-batch steps of the linear probe head
HEAD_LEARNING_RATE = 0.01
HEAD_SEED = 0


def effective_rank(embeddings: torch.Tensor) -> float:
    """exp(entropy) of the covariance spectrum: how many directions the clips actually spread along."""
    centred = embeddings - embeddings.mean(dim=0)
    values = torch.linalg.svdvals(centred.double()).square()
    p = values / values.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))


class DownstreamProbe:
    def __init__(self, splits: dict, cfg: dict, device: torch.device):
        labels = cfg["labels"]
        train = select_training_subset(splits["train"], cfg["data"]["max_train_clips"], labels, cfg["runtime"]["seed"])
        reader = AudioReader(AudioSettings(**cfg["audio"]))
        # decoded once and kept on the CPU: about 0.3 GB for 1,000 five-second clips
        self.clips = {name: [reader.read(row.path) for row in rows] for name, rows in
                      (("train", train), ("validation", splits["validation"]))}
        self.targets = {name: torch.tensor([labels.index(row.label) for row in rows]) for name, rows in
                        (("train", train), ("validation", splits["validation"]))}
        self.labels, self.cfg, self.device = labels, cfg, device
        self.initial = None

    @torch.no_grad()
    def _embed(self, backbone, name):
        rows = []
        for wave in self.clips[name]:
            with training_autocast(self.device, self.cfg["runtime"]["precision"]):
                rows.append(backbone.embed(wave.unsqueeze(0).to(self.device)).float().cpu())
        return torch.cat(rows)

    def _fit_head(self, train_x, train_y):
        """Class-balanced linear head on standardised features. Its own generator: the global RNG that drives the
        SSL masks and data order is never touched, so a check does not change the training run."""
        generator = torch.Generator().manual_seed(HEAD_SEED)
        weight = (torch.randn(len(self.labels), train_x.shape[1], generator=generator) * 0.01).requires_grad_()
        bias = torch.zeros(len(self.labels), requires_grad=True)
        mean, std = train_x.mean(dim=0), train_x.std(dim=0).clamp_min(1e-6)
        counts = torch.bincount(train_y, minlength=len(self.labels)).float()
        class_weights = len(train_y) / (len(self.labels) * counts.clamp_min(1))
        optimizer = torch.optim.AdamW([weight, bias], lr=HEAD_LEARNING_RATE, weight_decay=0.01)
        x = (train_x - mean) / std
        for _ in range(HEAD_STEPS):
            optimizer.zero_grad()
            F.cross_entropy(F.linear(x, weight, bias), train_y, weight=class_weights).backward()
            optimizer.step()
        return lambda features: F.linear((features - mean) / std, weight, bias)

    def evaluate(self, backbone) -> dict:
        """Validation UAR / macro-F1 of a fresh linear head, plus drift and spread of the validation embeddings."""
        was_training = backbone.training
        backbone.eval()
        try:
            train_x, validation_x = self._embed(backbone, "train"), self._embed(backbone, "validation")
        finally:
            backbone.train(was_training)
        predict = self._fit_head(train_x, self.targets["train"])
        with torch.no_grad():
            metrics = classification_metrics(predict(validation_x), self.targets["validation"], self.labels)
        if self.initial is None:
            self.initial = validation_x
        cosine = F.cosine_similarity(validation_x, self.initial, dim=1)
        return {"downstream_uar": metrics["uar"], "downstream_macro_f1": metrics["macro_f1"],
                "embedding_cosine_to_initial_median": float(cosine.median()),
                "embedding_cosine_to_initial_min": float(cosine.min()),
                "embedding_spread": float(validation_x.std(dim=0).mean()),
                "embedding_effective_rank": effective_rank(validation_x)}
