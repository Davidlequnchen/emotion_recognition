"""Frozen utterance embeddings plus an independently trained classification head."""
import gc
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .audio import AudioDataset, AudioReader, AudioSettings, check_durations, collate_audio
from .backbone import EmotionBackbone
from utilities.config import resolve_path
from .data import read_manifest, select_training_subset, split_records
from utilities.hardware import resolve_precision, training_autocast
from utilities.monitoring import RunMonitor
from utilities.runtime import (append_json, atomic_save, audio_audit, classification_metrics, initialize_runtime,
                      new_run_directory, records_digest, save_records, write_json)


class ClassificationHead(nn.Module):
    """Option A: one embedding -> Linear (or a small MLP). Option B (num_layers > 0): a learned softmax-weighted
    sum of every block's embedding -> the same Linear/MLP. Each block's embedding is layer-normalised first so the
    weights compare blocks on one scale. The encoder stays frozen either way."""
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dim: int = 0, dropout: float = 0, num_layers: int = 0):
        super().__init__()
        # zeros = equal weights at the start; only option B has them
        self.layer_logits = nn.Parameter(torch.zeros(num_layers)) if num_layers else None
        self.network = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, num_classes)) if hidden_dim else nn.Linear(embedding_dim, num_classes)

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(0)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.layer_logits is not None:
            if embeddings.dim() != 3 or embeddings.shape[1] != len(self.layer_logits):
                raise ValueError("Option B expects (batch, blocks, dim) layer embeddings")
            embeddings = F.layer_norm(embeddings, embeddings.shape[-1:])
            embeddings = (self.layer_weights()[:, None] * embeddings).sum(dim=1)
        return self.network(embeddings)


def load_backbone(cfg: dict, device: torch.device) -> EmotionBackbone:
    backbone = EmotionBackbone.load(resolve_path(cfg["model"]["directory"]), device)
    if cfg["model"]["encoder_checkpoint"]:
        artifact = torch.load(resolve_path(cfg["model"]["encoder_checkpoint"]), map_location="cpu", weights_only=True)
        if artifact.get("format_version") != 1 or artifact.get("model_config_hash") != backbone.model_config_hash:
            raise ValueError("Encoder checkpoint format/architecture does not match the local model config")
        backbone.load_encoder(artifact["encoder_state"])
        if backbone.fingerprint != artifact["fingerprint"]:
            raise ValueError("Encoder checkpoint fingerprint mismatch")
    return backbone


def labeled_splits(cfg: dict, audit=None) -> dict:
    records = read_manifest(resolve_path(cfg["data"]["labeled_manifest"]), labeled=True, labels=cfg["labels"])
    return split_records(records, validation_fraction=cfg["data"]["validation_fraction"],
                         test_fraction=cfg["data"]["test_fraction"], seed=cfg["runtime"]["seed"], audit=audit)


def extract_embeddings(backbone: EmotionBackbone, records, cfg: dict, device: torch.device, monitor=None):
    """One clip per encoder pass, never padded: padding changes FunASR's output (EmotionBackbone.embed_each)."""
    reader = AudioReader(AudioSettings(**cfg["audio"]))
    dataset = AudioDataset(records, reader, label_to_id={name: i for i, name in enumerate(cfg["labels"])})
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=cfg["runtime"]["num_workers"], collate_fn=collate_audio,
                        pin_memory=device.type == "cuda")
    features, targets = [], []
    backbone.eval().requires_grad_(False)
    # option A caches (clips, dim); option B caches every block, (clips, blocks, dim)
    embed = backbone.layer_embeddings if cfg["head"]["layer_fusion"] == "weighted" else backbone.embed
    with torch.no_grad():
        batches = monitor.progress(loader, "Extracting frozen features") if monitor else loader
        for batch in batches:
            with training_autocast(device, cfg["runtime"]["precision"]):
                embeddings = embed(batch["sources"].to(device), batch["padding_mask"].to(device))
            if not torch.isfinite(embeddings).all():
                raise RuntimeError("Encoder produced non-finite embeddings")
            features.append(embeddings.detach().float().cpu())
            targets.append(batch["labels"])
    return torch.cat(features), torch.cat(targets)


MIN_VALIDATION_PER_CLASS = 10  # below this, one clip moves a class's recall by 10 points or more


def check_validation_classes(splits, labels, selection_metric, run, monitor):
    """Write clips per split and class; UAR/macro-F1 selection needs every class in validation."""
    counts = {split: {label: sum(row.label == label for row in rows) for label in labels} for split, rows in splits.items()}
    write_json(run / "split_counts.json", counts)
    missing = [label for label, count in counts["validation"].items() if count == 0]
    if missing and selection_metric != "loss":
        raise ValueError(f"Validation has no clips of {missing}; {selection_metric} selection cannot see these classes. "
                         "Fix the customer split, or set head.selection_metric: loss")
    few = {label: count for label, count in counts["validation"].items() if 0 < count < MIN_VALIDATION_PER_CLASS}
    if few:
        monitor.log(f"Warning: few validation clips for {few}; {selection_metric} will be noisy for these classes")


class HeadTrainer:
    """Extract once, release the encoder, and train only the head on the selected device."""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self) -> Path:
        cfg, settings = self.cfg, self.cfg["head"]
        device = initialize_runtime(cfg["runtime"])
        run = new_run_directory(resolve_path(settings["output_dir"]), cfg)
        with RunMonitor(run, "head", cfg.get("monitoring", {})) as monitor:
            return self._fit(run, device, monitor)

    def _fit(self, run, device, monitor):
        cfg, settings = self.cfg, self.cfg["head"]
        precision = resolve_precision(device, cfg["runtime"]["precision"])
        monitor.log(f"Head training device={device}, precision={precision}; encoder remains frozen")
        # content digests and headers are read once and cached across runs (data.audit_cache)
        audit = audio_audit(cfg)
        splits = labeled_splits(cfg, audit)
        splits["train"] = select_training_subset(splits["train"], cfg["data"]["max_train_clips"],
                                                  cfg["labels"], cfg["runtime"]["seed"])
        if any(not values for values in splits.values()):
            raise ValueError("Head training requires nonempty train, validation and test sets")
        for split, records in splits.items():
            save_records(run / f"{split}.csv", records)
        check_validation_classes(splits, cfg["labels"], settings["selection_metric"], run, monitor)
        check_durations([row for records in splits.values() for row in records], AudioSettings(**cfg["audio"]), audit)
        monitor.log(audit.summary())
        backbone = load_backbone(cfg, device)
        fingerprint, config_hash, embedding_dim = backbone.fingerprint, backbone.model_config_hash, backbone.embedding_dim
        num_layers = len(backbone.blocks()) if settings["layer_fusion"] == "weighted" else 0
        monitor.log(f"Option B: learned weighted sum of all {num_layers} blocks" if num_layers
                    else "Option A: last block, mean over valid frames")
        cached = {}
        for split, records in splits.items():
            monitor.log(f"Extracting {split}: {len(records)} clips on {device}")
            x, y = extract_embeddings(backbone, records, cfg, device, monitor)
            cached[split] = (x, y)
        if backbone.fingerprint != fingerprint:
            raise RuntimeError("Frozen encoder changed during feature extraction")
        atomic_save({"encoder_fingerprint": fingerprint, "source_digest": {k: records_digest(v, audit) for k, v in splits.items()},
                     "audio": cfg["audio"], "labels": cfg["labels"], "layer_fusion": settings["layer_fusion"],
                     "splits": cached}, run / "features.pt")
        del backbone
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if device.type == "mps":
            torch.mps.empty_cache()
        head = ClassificationHead(embedding_dim, len(cfg["labels"]), settings["hidden_dim"], settings["dropout"],
                                  num_layers).to(device)
        groups = [{"params": head.network.parameters()}]
        if num_layers:
            # at the head's rate the layer weights barely leave 1/num_layers; no decay pulls them back to equal
            groups.append({"params": [head.layer_logits], "lr": settings["layer_weights_learning_rate"], "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        train_x, train_y = cached["train"]
        counts = torch.bincount(train_y, minlength=len(cfg["labels"]))
        weights = (len(train_y) / (len(counts) * counts.float())).to(device) if settings["class_weights"] == "balanced" else None
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=settings["batch_size"], shuffle=True,
                            pin_memory=device.type == "cuda")
        best, stale = None, 0
        step = 0
        for epoch in monitor.progress(range(1, settings["epochs"] + 1), "Head epochs"):
            head.train()
            total_loss, total_objective_loss = 0.0, 0.0
            train_logits, train_targets = [], []
            for x, y in monitor.progress(loader, f"Head epoch {epoch}/{settings['epochs']}"):
                started = time.perf_counter()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with training_autocast(device, cfg["runtime"]["precision"]):
                    logits = head(x)
                loss = F.cross_entropy(logits.float(), y, weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite head training loss")
                loss.backward()
                optimizer.step()
                # Plot unweighted CE for comparable train/validation curves;
                # record the weighted optimization objective separately.
                observed_loss = F.cross_entropy(logits.detach().float(), y).item()
                total_loss += observed_loss * len(y)
                total_objective_loss += loss.item() * len(y)
                train_logits.append(logits.detach().float().cpu())
                train_targets.append(y.detach().cpu())
                step += 1
                monitor.record({"loss": observed_loss, "objective_loss": loss.item(),
                                "accuracy": (logits.argmax(-1) == y).float().mean().item(),
                                "learning_rate": settings["learning_rate"],
                                "clips_per_second": len(y) / max(time.perf_counter() - started, 1e-9)},
                               step=step, epoch=epoch)
            head.eval()
            with torch.no_grad():
                with training_autocast(device, cfg["runtime"]["precision"]):
                    validation_logits = head(cached["validation"][0].to(device)).float().cpu()
                validation_loss = F.cross_entropy(validation_logits, cached["validation"][1]).item()
                if not torch.isfinite(validation_logits).all() or not torch.isfinite(torch.tensor(validation_loss)):
                    raise RuntimeError("Non-finite head validation output")
            metrics = classification_metrics(validation_logits, cached["validation"][1], cfg["labels"])
            train_metrics = classification_metrics(torch.cat(train_logits), torch.cat(train_targets), cfg["labels"])
            epoch_metrics = {"train_loss": total_loss / len(train_y),
                             "train_objective_loss": total_objective_loss / len(train_y),
                             "validation_loss": validation_loss,
                             "train_accuracy": train_metrics["accuracy"], "validation_accuracy": metrics["accuracy"],
                             "train_macro_f1": train_metrics["macro_f1"], "validation_macro_f1": metrics["macro_f1"]}
            append_json(run / "history.jsonl", {"epoch": epoch, **epoch_metrics, **metrics})
            monitor.record(epoch_metrics, epoch=epoch, update_plot=True)
            # higher selection score is better; ties go to the lower validation loss
            score = (-validation_loss if settings["selection_metric"] == "loss" else metrics[settings["selection_metric"]],
                     -validation_loss)
            if best is None or score > best:
                best, stale = score, 0
                atomic_save({"format_version": 1, "state": head.state_dict(), "embedding_dim": embedding_dim,
                             "hidden_dim": settings["hidden_dim"], "dropout": settings["dropout"], "labels": cfg["labels"],
                             "encoder_fingerprint": fingerprint, "model_config_hash": config_hash, "audio": cfg["audio"],
                             "precision": precision, "best_epoch": epoch, "layer_fusion": settings["layer_fusion"],
                             "num_layers": num_layers, "selection_metric": settings["selection_metric"],
                             "selection_score": score[0]}, run / "head.pt")
            else:
                stale += 1
                if stale >= settings["patience"]:
                    monitor.log(f"Early stopping after {stale} epochs without validation "
                                f"{settings['selection_metric']} improvement")
                    break
        artifact = torch.load(run / "head.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(artifact["state"])
        head.eval()
        with torch.no_grad():
            with training_autocast(device, cfg["runtime"]["precision"]):
                test_logits = head(cached["test"][0].to(device)).float().cpu()
        write_json(run / "test_metrics.json", classification_metrics(test_logits, cached["test"][1], cfg["labels"]))
        if num_layers:
            layer_weights = [round(w, 4) for w in head.layer_weights().detach().cpu().tolist()]
            write_json(run / "layer_weights.json", {"blocks": list(range(1, num_layers + 1)), "weights": layer_weights,
                                                    "note": "block 1 is the first prenet block, the last is the final block"})
            monitor.log(f"Learned block weights: {layer_weights}")
        bundle = {"format_version": 1, "model_name": "emotion2vec_plus_large", "model_directory": str(resolve_path(cfg["model"]["directory"])),
                  "encoder_checkpoint": str(resolve_path(cfg["model"]["encoder_checkpoint"])) if cfg["model"]["encoder_checkpoint"] else None,
                  "head_checkpoint": "head.pt", "encoder_fingerprint": fingerprint, "audio": cfg["audio"],
                  "precision": precision, "labels": cfg["labels"], "layer_fusion": settings["layer_fusion"]}
        write_json(run / "model_bundle.json", bundle)
        monitor.log(f"Head model saved: {run / 'model_bundle.json'}")
        return run
