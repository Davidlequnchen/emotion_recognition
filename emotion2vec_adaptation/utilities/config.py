"""Read the fixed YAML files; all relative paths start at the repository root."""
from copy import deepcopy
import math
from pathlib import Path

import yaml

from adaptation.audio import AudioSettings

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_DIR.parent


def _read(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_settings(stage: str) -> dict:
    """Scripts select their own file; users edit YAML rather than command arguments."""
    if stage not in {"head", "pretrain", "predict", "labels"}:
        raise ValueError(f"Unknown stage: {stage}")
    common = _read(PROJECT_DIR / "config/common.yaml")
    stage_settings = _read(PROJECT_DIR / f"config/{stage}.yaml")
    section = "labels_import" if stage == "labels" else stage
    if set(stage_settings) != {section}:
        raise ValueError(f"config/{stage}.yaml must contain only '{section}:'")
    cfg = deepcopy(common)
    cfg.update(stage_settings)
    validate_settings(cfg)
    return cfg


def validate_settings(cfg: dict) -> None:
    """Fail on misspelled keys and invalid settings before loading large weights."""
    schemas = {
        "model": {"directory", "encoder_checkpoint"},
        "runtime": {"device", "precision", "cpu_threads", "seed", "num_workers"},
        "monitoring": None,
        "audio": {"sample_rate", "min_seconds", "max_seconds", "normalize"},
        "data": {"labeled_manifest", "unlabeled_source", "validation_fraction", "test_fraction", "max_train_clips", "audit_cache"},
        "labels": None,
        "head": {"output_dir", "layer_fusion", "layer_weights_learning_rate", "selection_metric", "batch_size", "epochs", "learning_rate", "weight_decay", "hidden_dim", "dropout", "class_weights", "patience"},
        "pretrain": {"output_dir", "batch_size", "epochs", "learning_rate", "predictor_learning_rate", "warmup_steps", "weight_decay", "gradient_clip", "target", "target_top_layers", "frame_weight", "utterance_weight", "mask_probability", "mask_length", "ema_decay", "predictor_hidden", "freeze_frontend", "anchor_weight", "probe_steps", "probe_min_gain", "downstream_every_steps", "downstream_max_drop", "bucket_by_duration", "save_every_steps", "keep_checkpoints", "save_training_state", "resume_checkpoint", "max_steps", "validation_fraction", "validation_manifest", "max_unlabeled_clips"},
        "predict": {"bundle", "source", "output_csv"},
        "labels_import": {"votes", "clips_manifest", "output_manifest"},
    }
    unknown = set(cfg) - set(schemas)
    if unknown:
        raise ValueError(f"Unknown YAML sections: {sorted(unknown)}")
    for section in ("model", "runtime", "audio", "data", "labels"):
        if section not in cfg:
            raise ValueError(f"Missing YAML section: {section}")
    for section, keys in schemas.items():
        if section in cfg and keys is not None:
            value = cfg[section]
            if not isinstance(value, dict) or set(value) != keys:
                actual = set(value) if isinstance(value, dict) else set()
                raise ValueError(f"{section}: missing {sorted(keys-actual)}, unknown {sorted(actual-keys)}")
    labels = cfg["labels"]
    if not isinstance(labels, list) or len(labels) < 2 or any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two unique nonempty names")
    runtime, data = cfg["runtime"], cfg["data"]
    from .monitoring import validate_monitoring_settings
    validate_monitoring_settings(cfg.get("monitoring", {}))
    if runtime["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("runtime.device must be auto, cpu, cuda or mps")
    if runtime["precision"] not in {"auto", "float32", "bfloat16"}:
        raise ValueError("runtime.precision must be auto, float32 or bfloat16")
    for key in ("cpu_threads", "num_workers", "seed"):
        _integer(runtime[key], f"runtime.{key}", minimum=0)
    for key in ("validation_fraction", "test_fraction"):
        if not isinstance(data[key], (int, float)) or not 0 < data[key] < 1:
            raise ValueError(f"data.{key} must be between zero and one")
    if data["validation_fraction"] + data["test_fraction"] >= 1:
        raise ValueError("Validation and test fractions must leave training data")
    if data["max_train_clips"] is not None:
        _integer(data["max_train_clips"], "data.max_train_clips", minimum=len(labels))
    if data["audit_cache"] is not None and (not isinstance(data["audit_cache"], str) or not data["audit_cache"].strip()):
        raise ValueError("data.audit_cache must be a file path or null")
    AudioSettings(**cfg["audio"])
    if cfg["audio"]["sample_rate"] != 16000 or cfg["audio"]["normalize"] is not True:
        raise ValueError("Emotion2Vec requires sample_rate: 16000 and normalize: true")
    for stage in ("head", "pretrain"):
        if stage not in cfg:
            continue
        settings = cfg[stage]
        for key in ("batch_size", "epochs"):
            _integer(settings[key], f"{stage}.{key}", minimum=1)
        _number(settings["learning_rate"], f"{stage}.learning_rate", 0, 1, inclusive_min=False)
        _number(settings["weight_decay"], f"{stage}.weight_decay", 0)
    if "head" in cfg:
        h = cfg["head"]
        _integer(h["hidden_dim"], "head.hidden_dim", 0)
        _integer(h["patience"], "head.patience", 1)
        _number(h["dropout"], "head.dropout", 0, 1)
        if h["class_weights"] not in {"none", "balanced"}:
            raise ValueError("head.class_weights must be none or balanced")
        if h["layer_fusion"] not in {"last", "weighted"}:
            raise ValueError("head.layer_fusion must be last (option A) or weighted (option B)")
        _number(h["layer_weights_learning_rate"], "head.layer_weights_learning_rate", 0, 1, inclusive_min=False)
        if h["selection_metric"] not in {"uar", "macro_f1", "loss"}:
            raise ValueError("head.selection_metric must be uar, macro_f1 or loss")
    if "pretrain" in cfg:
        s = cfg["pretrain"]
        for key in ("mask_length", "target_top_layers", "predictor_hidden"):
            _integer(s[key], f"pretrain.{key}", 1)
        _number(s["mask_probability"], "pretrain.mask_probability", 0, 1, inclusive_min=False)
        _number(s["ema_decay"], "pretrain.ema_decay", 0, 1)
        _number(s["gradient_clip"], "pretrain.gradient_clip", 0, inclusive_min=False)
        _number(s["utterance_weight"], "pretrain.utterance_weight", 0)
        _number(s["frame_weight"], "pretrain.frame_weight", 0)
        if s["frame_weight"] + s["utterance_weight"] == 0:
            raise ValueError("pretrain.frame_weight and pretrain.utterance_weight cannot both be 0")
        if s["target"] not in {"final", "ffn_mean"}:
            raise ValueError("pretrain.target must be final or ffn_mean")
        _number(s["predictor_learning_rate"], "pretrain.predictor_learning_rate", 0, 1, inclusive_min=False)
        _number(s["anchor_weight"], "pretrain.anchor_weight", 0)
        _number(s["probe_min_gain"], "pretrain.probe_min_gain", 0, 1)
        _number(s["downstream_max_drop"], "pretrain.downstream_max_drop", 0, 1)
        for key in ("warmup_steps", "probe_steps", "downstream_every_steps", "save_every_steps", "keep_checkpoints"):
            _integer(s[key], f"pretrain.{key}", 0)
        _number(s["validation_fraction"], "pretrain.validation_fraction", 0, 1)
        if s["max_unlabeled_clips"] is not None:
            _integer(s["max_unlabeled_clips"], "pretrain.max_unlabeled_clips", 1)
        for key in ("freeze_frontend", "save_training_state", "bucket_by_duration"):
            if not isinstance(s[key], bool):
                raise ValueError(f"pretrain.{key} must be boolean")
        if s["max_steps"] is not None:
            _integer(s["max_steps"], "pretrain.max_steps", 1)


def _integer(value, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name: str, minimum: float, maximum: float | None = None, inclusive_min=True) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (not inclusive_min and value == minimum)
            or (maximum is not None and value >= maximum)):
        raise ValueError(f"{name} must be finite and in {'[' if inclusive_min else '('}{minimum}, {maximum or 'infinity'})")
