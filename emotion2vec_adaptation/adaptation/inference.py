"""Use a matched encoder/head pair and refuse silently mismatched checkpoints."""
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .audio import AudioDataset, AudioReader, AudioSettings, check_durations, collate_audio
from utilities.config import resolve_path
from .data import read_unlabeled
from .head import ClassificationHead, load_backbone
from utilities.hardware import training_autocast, validate_device
from utilities.runtime import initialize_runtime


class AdaptedEmotionModel:
    """A small inference API independent of the training entry points."""
    def __init__(self, bundle_path: Path, device: torch.device):
        bundle_path = Path(bundle_path).resolve()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle.get("format_version") != 1:
            raise ValueError("Unknown model bundle format")
        self.device = device
        validate_device(device, "auto", run_kernels=False)
        self.backbone = load_backbone({"model": {"directory": bundle["model_directory"],
                                                  "encoder_checkpoint": bundle["encoder_checkpoint"]}}, device)
        artifact = torch.load(bundle_path.parent / bundle["head_checkpoint"], map_location="cpu", weights_only=True)
        fingerprint = self.backbone.fingerprint
        if fingerprint != bundle["encoder_fingerprint"] or fingerprint != artifact["encoder_fingerprint"]:
            raise ValueError("Head/encoder mismatch: retrain the classification head with the selected encoder")
        if artifact.get("format_version") != 1 or artifact["model_config_hash"] != self.backbone.model_config_hash:
            raise ValueError("Head architecture does not match the encoder")
        if artifact["labels"] != bundle["labels"] or artifact["audio"] != bundle["audio"]:
            raise ValueError("Bundle labels/audio settings do not match its head")
        # heads saved before option B existed are option A
        self.layer_fusion = artifact.get("layer_fusion", "last")
        if self.layer_fusion != bundle.get("layer_fusion", "last"):
            raise ValueError("Bundle layer_fusion does not match its head")
        self.labels = artifact["labels"]
        self.reader = AudioReader(AudioSettings(**artifact["audio"]))
        self.head = ClassificationHead(artifact["embedding_dim"], len(self.labels), artifact["hidden_dim"], artifact["dropout"],
                                       artifact.get("num_layers", 0))
        self.head.load_state_dict(artifact["state"], strict=True)
        self.head.to(device).eval()
        self.backbone.eval().requires_grad_(False)

    @torch.no_grad()
    def predict_batch(self, sources, padding_mask) -> torch.Tensor:
        """Class probabilities; the encoder still sees one unpadded clip at a time, so batching never changes them."""
        with training_autocast(self.device, "auto"):
            embeddings = self.backbone.embed_each(sources.to(self.device), padding_mask.to(self.device),
                                                  layers=self.layer_fusion == "weighted")
            logits = self.head(embeddings)
        return logits.float().softmax(-1).cpu()

    def predict_file(self, path: Path) -> dict:
        wave = self.reader.read(path)
        scores = self.predict_batch(wave.unsqueeze(0), torch.zeros(1, len(wave), dtype=torch.bool))[0]
        return {"label": self.labels[scores.argmax().item()], "scores": dict(zip(self.labels, scores.tolist()))}


def run_prediction(cfg: dict) -> Path:
    device = initialize_runtime(cfg["runtime"])
    settings = cfg["predict"]
    records = read_unlabeled(resolve_path(settings["source"]), training_only=False)
    output = resolve_path(settings["output_csv"])
    if output.exists():
        raise FileExistsError(f"Prediction output already exists: {output}")
    model = AdaptedEmotionModel(resolve_path(settings["bundle"]), device)
    check_durations(records, model.reader.settings)
    dataset = AudioDataset(records, model.reader)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_audio,
                        num_workers=cfg["runtime"]["num_workers"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "clip_id", "prediction"] + [f"score_{label}" for label in model.labels])
        writer.writeheader()
        for batch in loader:
            for record, scores in zip(batch["records"], model.predict_batch(batch["sources"], batch["padding_mask"])):
                writer.writerow({"path": str(record.path), "clip_id": record.clip_id,
                                 "prediction": model.labels[scores.argmax().item()],
                                 **{f"score_{label}": score for label, score in zip(model.labels, scores.tolist())}})
    print(f"Predictions saved: {output}", flush=True)
    return output
