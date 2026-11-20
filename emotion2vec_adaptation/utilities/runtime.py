"""Small shared utilities for repeatable experiments and portable artifacts."""
import csv
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

from adaptation.audit import AudioAudit
from .config import resolve_path


def initialize_runtime(settings: dict) -> torch.device:
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if settings["cpu_threads"]:
        torch.set_num_threads(settings["cpu_threads"])
    name = settings["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(name)
    from .hardware import validate_device
    validate_device(device, settings.get("precision", "auto"), run_kernels=False)
    return device


def new_run_directory(path: Path, cfg: dict) -> Path:
    """Never overwrite another experiment's outputs."""
    path.mkdir(parents=True, exist_ok=False)
    (path / "settings.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    import funasr
    from .hardware import device_environment
    name = cfg["runtime"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    environment = device_environment(torch.device(name))
    environment.update(funasr=funasr.__version__, numpy=np.__version__)
    write_json(path / "environment.json", environment)
    return path


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_records(path: Path, records) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split", "clip_id"])
        writer.writeheader()
        for r in records:
            writer.writerow({"path": str(r.path), "label": r.label or "", "group": r.group,
                             "split": r.split, "clip_id": r.clip_id})


def audio_audit(cfg: dict) -> AudioAudit:
    """The audio audit at data.audit_cache, reused across runs; null keeps it in memory for this run only."""
    return AudioAudit(resolve_path(cfg["data"]["audit_cache"]) if cfg["data"]["audit_cache"] else None)


def records_digest(records, audit=None) -> str:
    """Record source content, not only filenames: each clip's SHA256 comes from the audio audit (in memory if None)."""
    records = list(records)
    digest = hashlib.sha256()
    for record, content in zip(records, (audit or AudioAudit()).digests(record.path for record in records)):
        digest.update(f"{record.path}\0{record.label}\0{record.group}\0{record.split}\0{content}\0".encode())
    return digest.hexdigest()


@contextmanager
def isolated_torch_rng(seed: int, device: torch.device):
    """Evaluate fixed validation masks without changing training's RNG sequence."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, classes: list[str]) -> dict:
    predicted = logits.argmax(-1).cpu()
    labels = labels.cpu()
    count = len(classes)
    confusion = torch.bincount(labels * count + predicted, minlength=count * count).reshape(count, count).double()
    support = confusion.sum(1)
    tp = confusion.diag()
    recall = tp / support.clamp_min(1)
    precision = tp / confusion.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"accuracy": (tp.sum() / support.sum()).item(), "uar": recall.mean().item(),
            "macro_f1": f1.mean().item(), "support": dict(zip(classes, support.long().tolist())),
            "confusion_matrix": confusion.long().tolist()}
