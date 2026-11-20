"""A standalone unlabeled training loop; exports a replacement encoder only."""
import hashlib
import random
from dataclasses import replace
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .audio import AudioReader, AudioSettings, BatchOrder, ClipBatches, check_durations
from utilities.config import resolve_path
from .data import assert_disjoint_splits, exclude_holdouts, read_manifest, read_unlabeled, split_records
from utilities.hardware import memory_metrics, resolve_precision, training_autocast
from .downstream import DownstreamProbe
from .head import labeled_splits, load_backbone
from utilities.monitoring import RunMonitor
from utilities.runtime import (append_json, atomic_save, audio_audit, initialize_runtime, new_run_directory,
                      isolated_torch_rng, records_digest, save_records, write_json)
from .ssl import MaskedTeacherObjective

PREFETCH_BATCHES = 4  # batches each DataLoader worker prepares ahead: absorbs network-storage latency spikes


def _resume_signature(cfg: dict, source_digest: str) -> dict:
    # These controls may legitimately change when resuming into a new run folder.
    controls = {k: v for k, v in cfg["pretrain"].items()
                if k not in {"output_dir", "resume_checkpoint", "epochs", "max_steps", "save_training_state",
                             "save_every_steps", "keep_checkpoints"}}
    return {"audio": cfg["audio"], "seed": cfg["runtime"]["seed"], "controls": controls,
            "source_digest": source_digest}


def unlabeled_validation_split(records, holdouts, cfg):
    """Reserve independent unlabeled calls without touching the labeled holdouts: the fixed validation manifest, else
    the source CSV's validation rows, else validation_fraction of the groups. Groups and paths are checked here; byte
    content is checked after sampling, on the clips the run actually uses (ContinualPretrainer._prepare_data)."""
    settings = cfg["pretrain"]
    validation = [row for row in records if row.split == "validation"]
    records = [row for row in records if row.split in {"", "train"}]
    if not records:
        raise ValueError("No training audio remains in the unlabeled source")
    if settings["validation_manifest"]:
        validation = read_manifest(resolve_path(settings["validation_manifest"]), labeled=False)
        if any(row.split for row in validation):
            validation = [row for row in validation if row.split == "validation"]
        if not validation:
            raise ValueError("The explicit SSL validation manifest has no validation audio")
        reserved_groups = {row.group for row in validation}
        reserved_paths = {row.path for row in validation}
        records = [row for row in records if row.group not in reserved_groups and row.path not in reserved_paths]
        if not records:
            raise ValueError("The explicit SSL validation manifest reserves all training audio")
    elif not validation and settings["validation_fraction"]:
        parts = split_records([replace(row, split="") for row in records],
                              validation_fraction=settings["validation_fraction"], test_fraction=0,
                              seed=cfg["runtime"]["seed"], check_content=False)
        records, validation = parts["train"], parts["validation"]
    assert_disjoint_splits({"ssl_train": records, "ssl_validation": validation, "labeled_holdouts": holdouts},
                           check_content=False)
    return records, validation


def select_unlabeled_budget(records, max_clips, seed):
    """A reproducible sample: the clips with the smallest seeded path hash. With one seed a smaller budget is a
    subset of a larger one, so stages are nested and reuse the audio audit; adding source batches keeps most of it.
    Use a curated CSV for metadata stratification."""
    records = sorted(records, key=lambda row: (row.group, str(row.path)))
    if max_clips is None or max_clips >= len(records):
        return records
    selected = sorted(records, key=lambda row: hashlib.sha256(f"{seed}\0{row.path}".encode()).digest())[:max_clips]
    return sorted(selected, key=lambda row: (row.group, str(row.path)))


def evaluate_unlabeled(objective, loader, cfg, device) -> dict:
    """Fixed masks and eval mode, restoring RNG so validation cannot alter training. Returns the loss and its two
    learnability baselines (zero and per-channel constant prediction), averaged over masked frames."""
    was_training = objective.training
    objective.eval()
    totals, frames = {"loss": 0.0, "zero_loss": 0.0, "constant_loss": 0.0}, 0
    try:
        with torch.no_grad(), isolated_torch_rng(cfg["runtime"]["seed"] + 10000, device):
            for batch in loader:
                with training_autocast(device, cfg["runtime"]["precision"]):
                    loss, stats = objective(batch["sources"].to(device), batch["padding_mask"].to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite SSL validation loss")
                for key in totals:
                    totals[key] += stats[key] * stats["masked_frames"]
                frames += stats["masked_frames"]
    finally:
        objective.train(was_training)
    return {key: value / frames for key, value in totals.items()}


def encoder_learning_rate(step: int, settings: dict) -> float:
    """Encoder rate for 1-based step: zero during the probe, then a linear warm-up to learning_rate."""
    after = step - settings["probe_steps"]
    if after <= 0:
        return 0.0
    return settings["learning_rate"] * min(1.0, after / settings["warmup_steps"]) if settings["warmup_steps"] else settings["learning_rate"]


class ContinualPretrainer:
    """Masked EMA self-distillation on raw clips with no emotion-label objective.

    Two gates protect the encoder. The probe trains only the predictor for probe_steps with the encoder frozen and
    stops the run unless the validation loss beats the constant-prediction baseline by probe_min_gain. The
    downstream check refits a linear head on labelled clips every downstream_every_steps and stops the run when
    validation UAR falls more than downstream_max_drop below the original encoder's.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self) -> Path:
        cfg, settings = self.cfg, self.cfg["pretrain"]
        device = initialize_runtime(cfg["runtime"])
        run = new_run_directory(resolve_path(settings["output_dir"]), cfg)
        with RunMonitor(run, "pretrain", cfg.get("monitoring", {})) as monitor:
            return self._fit(run, device, monitor)

    def _prepare_data(self, run, monitor):
        """Groups and paths over the whole source, then the budget; only the clips actually used are read, once
        (the audio audit caches SHA256 and headers across runs), for the leakage, duration and resume checks."""
        cfg, settings = self.cfg, self.cfg["pretrain"]
        audit = audio_audit(cfg)
        holdouts, splits = [], None
        if resolve_path(cfg["data"]["labeled_manifest"]).exists():
            splits = labeled_splits(cfg, audit)
            holdouts = splits["validation"] + splits["test"]
        if settings["downstream_every_steps"] and splits is None:
            raise ValueError("The downstream check needs data.labeled_manifest; set pretrain.downstream_every_steps: 0 "
                             "to run without it (then nothing guards the emotion information)")
        candidates = read_unlabeled(resolve_path(cfg["data"]["unlabeled_source"]), exclude_records=holdouts,
                                    training_only=False, check_content=False)
        records, validation_records = unlabeled_validation_split(candidates, holdouts, cfg)
        available_training_clips = len(records)
        records = select_unlabeled_budget(records, settings["max_unlabeled_clips"], cfg["runtime"]["seed"])
        audit.entries([row.path for row in records + validation_records + holdouts], progress=monitor.progress)
        monitor.log(audit.summary())
        sampled = len(records)
        records = exclude_holdouts(records, holdouts, audit=audit)  # byte copies of held-out clips
        if not settings["validation_manifest"]:  # a fixed manifest is never edited silently: copies there raise below
            validation_records = exclude_holdouts(validation_records, holdouts, audit=audit)
        if not records:
            raise ValueError("No training audio remains after the labeled holdout exclusions")
        if settings["probe_steps"] and not validation_records:
            raise ValueError("The learnability probe needs SSL validation clips (pretrain.validation_fraction > 0 or "
                             "pretrain.validation_manifest); or set pretrain.probe_steps: 0")
        assert_disjoint_splits({"ssl_train": records, "ssl_validation": validation_records,
                                "labeled_holdouts": holdouts}, audit=audit)
        check_durations(records + validation_records, AudioSettings(**cfg["audio"]), audit)
        durations = [entry["duration"] for entry in audit.entries(row.path for row in records)]
        validation_durations = [entry["duration"] for entry in audit.entries(row.path for row in validation_records)]
        digests = (records_digest(records, audit),
                   records_digest(validation_records, audit) if validation_records else None)
        save_records(run / "unlabeled_train.csv", records)
        save_records(run / "unlabeled_validation.csv", validation_records)
        save_records(run / "excluded_holdouts.csv", holdouts)
        write_json(run / "data_audit.json", {"training_clips": len(records), "heldout_clips": len(holdouts),
                                            "available_training_clips": available_training_clips,
                                            "holdout_copies_dropped": sampled - len(records),
                                            "validation_clips": len(validation_records),
                                            "training_hours": sum(durations) / 3600,
                                            "validation_hours": sum(validation_durations) / 3600,
                                            "holdout_manifest_available": bool(holdouts), "source_digest": digests[0],
                                            "audio_audit": {"cache": str(audit.path) if audit.path else None,
                                                            "files_hashed": audit.hashed, "files_reused": audit.reused}})
        monitor.log(f"SSL data: {len(records)} training ({sum(durations) / 3600:.1f} h), {len(validation_records)} "
                    f"validation, {len(holdouts)} labeled holdout clips excluded, {sampled - len(records)} sampled "
                    "clips dropped as byte copies of holdouts")
        return splits, records, validation_records, durations, validation_durations, digests

    def _fit(self, run, device, monitor):
        cfg, settings = self.cfg, self.cfg["pretrain"]
        precision = resolve_precision(device, cfg["runtime"]["precision"])
        monitor.log(f"SSL training device={device}, precision={precision}; EMA teacher has no gradients")
        splits, records, validation_records, durations, validation_durations, digests = self._prepare_data(run, monitor)
        signature = _resume_signature(cfg, digests[0])
        signature.update(validation_digest=digests[1], precision=precision)
        backbone = load_backbone(cfg, device)
        initial_fingerprint = backbone.fingerprint
        objective = MaskedTeacherObjective(backbone, **{k: settings[k] for k in (
            "mask_probability", "mask_length", "ema_decay", "target_top_layers", "predictor_hidden",
            "freeze_frontend", "utterance_weight", "target", "frame_weight")})
        encoder_parameters = [p for p in objective.student.parameters() if p.requires_grad]
        predictor_parameters = list(objective.predictor.parameters())
        parameters = encoder_parameters + predictor_parameters
        optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": 0.0},
                                       {"params": predictor_parameters, "lr": settings["predictor_learning_rate"]}],
                                      weight_decay=settings["weight_decay"])
        # L2-SP: a pull back towards the original weights, kept on the device only when used
        anchors = [p.detach().clone() for p in encoder_parameters] if settings["anchor_weight"] else None
        downstream, baseline, best_uar = None, None, None
        if settings["downstream_every_steps"]:
            downstream = DownstreamProbe(splits, cfg, device)
            baseline = downstream.evaluate(objective.student)  # the original encoder, before any update or resume
            best_uar = baseline["downstream_uar"]
            append_json(run / "downstream_history.jsonl", {"step": 0, **baseline})
            monitor.log(f"Downstream baseline (original encoder): UAR {best_uar:.4f}")
        seed, bucket = cfg["runtime"]["seed"], settings["bucket_by_duration"]
        order = BatchOrder(durations, settings["batch_size"], seed, bucket, shuffle=True)
        start_epoch, skip, step, epoch_total, epoch_frames, probe_report = 1, 0, 0, 0.0, 0, None
        if settings["resume_checkpoint"]:
            state = torch.load(resolve_path(settings["resume_checkpoint"]), map_location=device, weights_only=True)
            if state.get("format_version") != 2:
                raise ValueError("The checkpoint predates exact mid-epoch resume (format 2); start a new run with its "
                                 "encoder.pt as model.encoder_checkpoint")
            if state.get("signature") != signature:
                raise ValueError("Resume data/objective/settings do not match the checkpoint")
            if state["model_config_hash"] != backbone.model_config_hash or state["initial_fingerprint"] != initial_fingerprint:
                raise ValueError("Resume base encoder does not match the checkpoint")
            if state["status"] != "completed":
                raise ValueError(f"The checkpoint's run ended with {state['status']}; start a new run from its "
                                 "encoder_best.pt instead")
            objective.load_state_dict(state["objective"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            step, best_uar, probe_report = state["step"], state["best_downstream_uar"], state["probe"]
            if state["completed_epoch"]:
                start_epoch = state["epoch"] + 1
            else:  # continue the interrupted epoch: same order, its consumed batches skipped without decoding audio
                start_epoch, skip = state["epoch"], state["batches_consumed"]
                epoch_total, epoch_frames = state["epoch_loss_sum"], state["epoch_masked_frames"]
            monitor.log(f"Resumed {settings['resume_checkpoint']}: epoch {start_epoch} from batch {skip + 1}, "
                        f"global step {step}")
            torch.set_rng_state(state["torch_rng"].cpu())
            random.setstate(state["python_rng"])
            if device.type == "cuda" and state.get("cuda_rng"):
                torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
            if device.type == "mps" and state.get("mps_rng") is not None:
                torch.mps.set_rng_state(state["mps_rng"].cpu())
            del state  # release the additional serialized teacher/optimizer tensor copies
        if start_epoch > settings["epochs"] or (settings["max_steps"] is not None and step >= settings["max_steps"]):
            raise ValueError("Resume checkpoint has already reached configured epochs/max_steps")
        reader = AudioReader(AudioSettings(**cfg["audio"]))
        workers = cfg["runtime"]["num_workers"]
        # Workers decode, resample, crop and normalise whole batches. The private generator keeps the DataLoader
        # from drawing on the global RNG that the masks use, so worker count never changes the training run.
        options = {"batch_size": None, "num_workers": workers, "pin_memory": device.type == "cuda",
                   "generator": torch.Generator().manual_seed(seed)}
        if workers:
            options["prefetch_factor"] = PREFETCH_BATCHES
        loader = DataLoader(ClipBatches(records, reader, seed, bucket), sampler=order,
                            persistent_workers=workers > 0, **options)
        validation_loader = None
        if validation_records:
            validation_loader = DataLoader(ClipBatches(validation_records, reader, seed, bucket),
                                           sampler=BatchOrder(validation_durations, settings["batch_size"], seed,
                                                              bucket, shuffle=False), **options)
        status, last_downstream = "completed", None
        checkpoints = run / "checkpoints"

        def set_encoder_trainable(trainable):
            for parameter in encoder_parameters:
                parameter.requires_grad_(trainable)

        def save(epoch, consumed):
            """Latest training state and encoder, plus a kept encoder-only checkpoint for later head comparisons."""
            started = time.perf_counter()
            if settings["save_training_state"]:
                atomic_save({"format_version": 2, "objective": objective.state_dict(), "optimizer": optimizer.state_dict(),
                             "signature": signature, "model_config_hash": backbone.model_config_hash,
                             "initial_fingerprint": initial_fingerprint, "epoch": epoch, "step": step,
                             "batches_consumed": consumed, "completed_epoch": consumed == order.batches,
                             "epoch_loss_sum": epoch_total, "epoch_masked_frames": epoch_frames, "status": status,
                             "probe": probe_report, "best_downstream_uar": best_uar, "torch_rng": torch.get_rng_state(),
                             "python_rng": random.getstate(),
                             "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                             "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None}, run / "last_training.pt")
            # Keep the last encoder; SSL loss is not an SER score (encoder_best.pt holds the best downstream one).
            self._save_encoder(run / "encoder.pt", backbone, initial_fingerprint, epoch, step)
            if settings["keep_checkpoints"]:
                checkpoints.mkdir(exist_ok=True)
                self._save_encoder(checkpoints / f"encoder_step_{step:08d}.pt", backbone, initial_fingerprint, epoch, step)
                for old in sorted(checkpoints.glob("encoder_step_*.pt"))[:-settings["keep_checkpoints"]]:
                    old.unlink()
            monitor.log(f"Saved step {step} (epoch {epoch}, batch {consumed}/{order.batches}) "
                        f"in {time.perf_counter() - started:.1f} s")

        set_encoder_trainable(step >= settings["probe_steps"])
        for epoch in monitor.progress(range(start_epoch, settings["epochs"] + 1), "SSL epochs"):
            objective.train()
            order.set_epoch(epoch, skip if epoch == start_epoch else 0)
            if not order.start:
                epoch_total, epoch_frames = 0.0, 0
            consumed = order.start
            if bucket:
                monitor.log(f"SSL epoch {epoch}: {order.batches} batches; cropping each batch to its shortest clip "
                            f"drops {order.cropped_fraction():.2%} of the audio")
            finished = requested = time.perf_counter()
            for batch in monitor.progress(loader, f"SSL epoch {epoch}/{settings['epochs']}"):
                fetched = time.perf_counter()
                consumed += 1
                encoder_lr = encoder_learning_rate(step + 1, settings)
                optimizer.param_groups[0]["lr"] = encoder_lr
                optimizer.zero_grad(set_to_none=True)
                with training_autocast(device, cfg["runtime"]["precision"]):
                    loss, stats = objective(batch["sources"].to(device, non_blocking=True),
                                            batch["padding_mask"].to(device, non_blocking=True))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite SSL loss at step {step + 1}")
                if anchors is not None and encoder_lr > 0:
                    anchor_loss = sum((p.float() - a.float()).square().sum() for p, a in zip(encoder_parameters, anchors))
                    loss = loss + settings["anchor_weight"] * anchor_loss
                    stats["anchor_loss"] = float(anchor_loss.detach())
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in parameters if p.grad is not None],
                                                           settings["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                objective.update_teacher()  # exactly once after the student optimizer update
                step += 1
                epoch_total += stats["loss"] * stats["masked_frames"]
                epoch_frames += stats["masked_frames"]
                done, clips = time.perf_counter(), len(batch["sources"])
                # wall time since the previous step ended: data waiting, logging and any check or save in between
                metrics = {"gradient_norm": float(grad_norm), "learning_rate": encoder_lr,
                           "predictor_learning_rate": settings["predictor_learning_rate"],
                           "data_wait_seconds": fetched - requested,
                           "clips_per_second": clips / max(done - finished, 1e-9),
                           "compute_clips_per_second": clips / max(done - fetched, 1e-9),
                           "cropped_fraction": batch["cropped_fraction"], **memory_metrics(device), **stats}
                finished = done
                append_json(run / "history.jsonl", {"epoch": epoch, "step": step, **metrics})
                monitor.record(metrics, step=step, epoch=epoch)

                if settings["probe_steps"] and step == settings["probe_steps"]:
                    probe_report = self._probe(objective, validation_loader, settings, step, run, monitor)
                    set_encoder_trainable(True)
                after_probe = step - settings["probe_steps"]
                if downstream is not None and after_probe > 0 and after_probe % settings["downstream_every_steps"] == 0:
                    last_downstream, best_uar, status = self._downstream(
                        downstream, objective.student, baseline, best_uar, settings, step, epoch, run, monitor,
                        initial_fingerprint)
                    if status != "completed":
                        break
                if settings["max_steps"] is not None and step >= settings["max_steps"]:
                    break
                if settings["save_every_steps"] and step % settings["save_every_steps"] == 0 and consumed < order.batches:
                    save(epoch, consumed)  # the epoch-end save covers the last batch
                requested = time.perf_counter()
            epoch_metrics = {"train_loss": epoch_total / max(epoch_frames, 1), "global_step": step}
            if bucket:
                epoch_metrics["cropped_fraction"] = order.cropped_fraction()
            if validation_loader is not None:
                validation = evaluate_unlabeled(objective, validation_loader, cfg, device)
                epoch_metrics.update(validation_loss=validation["loss"], validation_zero_loss=validation["zero_loss"],
                                     validation_constant_loss=validation["constant_loss"])
            append_json(run / "epoch_history.jsonl", {"epoch": epoch, **epoch_metrics})
            monitor.record(epoch_metrics, epoch=epoch, update_plot=True)
            save(epoch, consumed)
            if status != "completed" or (settings["max_steps"] is not None and step >= settings["max_steps"]):
                break
        write_json(run / "encoder_metadata.json", {"model_name": "emotion2vec_plus_large", "fingerprint": backbone.fingerprint,
                                                   "initial_fingerprint": initial_fingerprint, "requires_retrained_head": True,
                                                   "objective": f"masked_ema_{settings['target']}_v2", "steps": step,
                                                   "status": status, "probe": probe_report,
                                                   "downstream_baseline": baseline, "downstream_last": last_downstream,
                                                   "best_downstream_uar": best_uar,
                                                   "encoder_best_saved": (run / "encoder_best.pt").is_file()})
        monitor.log(f"SSL {status}. Encoder saved: {run / 'encoder.pt'}; retrain a head with an encoder before SER use.")
        return run

    @staticmethod
    def _save_encoder(path, backbone, initial_fingerprint, epoch, step, **extra):
        atomic_save({"format_version": 1, "encoder_state": backbone.encoder_state_dict(),
                     "fingerprint": backbone.fingerprint, "model_config_hash": backbone.model_config_hash,
                     "initial_fingerprint": initial_fingerprint, "objective": "masked_ema_feature_regression_v2",
                     "epoch": epoch, "step": step, **extra}, path)

    def _probe(self, objective, validation_loader, settings, step, run, monitor) -> dict:
        """After the predictor-only steps: can the predictor beat a constant prediction on held-out clips?"""
        result = evaluate_unlabeled(objective, validation_loader, self.cfg, next(objective.parameters()).device)
        gain = 1 - result["loss"] / result["constant_loss"]
        report = {"steps": step, **result, "gain_over_constant": gain, "min_gain": settings["probe_min_gain"],
                  "passed": gain >= settings["probe_min_gain"]}
        write_json(run / "probe.json", report)
        monitor.log(f"Learnability probe: validation loss {result['loss']:.4f}, constant baseline "
                    f"{result['constant_loss']:.4f}, zero baseline {result['zero_loss']:.4f}, gain {gain:.1%}")
        if not report["passed"]:
            raise RuntimeError(f"SSL target is not learnable from the student's features: the predictor beats a constant "
                               f"prediction by {gain:.1%} < {settings['probe_min_gain']:.0%} after {step} probe steps. "
                               "The encoder was not changed. See docs/guide.md (SSL target choice).")
        return report

    def _downstream(self, downstream, student, baseline, best_uar, settings, step, epoch, run, monitor, initial_fingerprint):
        """Refit a linear head on labelled clips; keep the best encoder; stop when UAR drops too far."""
        result = downstream.evaluate(student)
        append_json(run / "downstream_history.jsonl", {"step": step, **result})
        monitor.record(result, step=step, epoch=epoch)
        uar = result["downstream_uar"]
        monitor.log(f"Downstream check at step {step}: UAR {uar:.4f} (original {baseline['downstream_uar']:.4f}), "
                    f"cosine to original median {result['embedding_cosine_to_initial_median']:.3f}")
        if uar > best_uar:
            best_uar = uar
            self._save_encoder(run / "encoder_best.pt", student, initial_fingerprint, epoch, step, downstream=result)
        status = "completed"
        if uar < baseline["downstream_uar"] - settings["downstream_max_drop"]:
            status = "stopped_by_downstream_check"
            monitor.log(f"Stopping: downstream UAR fell {baseline['downstream_uar'] - uar:.4f} below the original encoder "
                        f"(limit {settings['downstream_max_drop']})")
        return result, best_uar, status
