"""Decode curated clips, enforce duration limits and create masked or padding-free audio batches."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audit import audio_header


@dataclass(frozen=True)
class AudioSettings:
    """The reader never crops or loudness-normalizes; only SSL batches may crop (ClipBatches)."""

    sample_rate: int = 16000
    min_seconds: float = 0.25
    max_seconds: float = 15.0
    normalize: bool = True

    def __post_init__(self):
        if self.sample_rate != 16000:
            raise ValueError("Emotion2Vec requires audio.sample_rate = 16000")
        if not (math.isfinite(self.min_seconds) and math.isfinite(self.max_seconds)
                and 0 < self.min_seconds < self.max_seconds):
            raise ValueError("Audio durations must satisfy 0 < min_seconds < max_seconds")
        if not isinstance(self.normalize, bool):
            raise ValueError("audio.normalize must be a YAML boolean")


class AudioReader:
    """Read mono float32 waveforms; standardize each unpadded clip if configured."""

    def __init__(self, settings: AudioSettings):
        self.settings = settings

    def read(self, path: Path) -> torch.Tensor:
        return self.normalize(self.decode(path), path)

    def decode(self, path: Path) -> torch.Tensor:
        """Mono 16 kHz samples, before the per-clip normalization."""
        path = Path(path)
        try:
            # Inspect first to avoid allocating an accidentally supplied hour-long call.
            info = sf.info(str(path))
            duration = info.frames / info.samplerate
            if not self.settings.min_seconds <= duration <= self.settings.max_seconds:
                raise ValueError(
                    f"{path}: duration {duration:.3f}s outside "
                    f"[{self.settings.min_seconds}, {self.settings.max_seconds}]s; "
                    "curate the clip rather than silently trimming it"
                )
            samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except (sf.LibsndfileError, OSError) as exc:
            raise ValueError(f"Cannot decode audio {path}: {exc}") from exc
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError(f"{path}: audio must contain finite, nonempty samples")
        wave = torch.from_numpy(samples.mean(axis=1)).float()
        if rate != self.settings.sample_rate:
            # Use the same sinc resampler as the existing preparation pipeline.
            from torchaudio.functional import resample

            wave = resample(wave, rate, self.settings.sample_rate)
        return wave

    def normalize(self, wave: torch.Tensor, path) -> torch.Tensor:
        if self.settings.normalize:
            # Emotion2Vec's input normalization uses layer_norm over the clip.
            wave = torch.nn.functional.layer_norm(wave, wave.shape)
        if not torch.isfinite(wave).all():
            raise ValueError(f"{path}: preprocessing produced nonfinite audio")
        return wave.contiguous()


def check_durations(records, settings: AudioSettings, audit=None) -> None:
    """Fail before any weights load, instead of mid-epoch inside the DataLoader. Durations come from the audio audit
    when given (no further file reads), otherwise from the headers."""
    records = list(records)
    entries = audit.entries(row.path for row in records) if audit else [audio_header(str(row.path)) for row in records]
    problems = []
    for row, entry in zip(records, entries):
        if entry["error"]:
            problems.append(f"{row.path}: cannot decode ({entry['error']})")
        elif not settings.min_seconds <= entry["duration"] <= settings.max_seconds:
            problems.append(f"{row.path}: {entry['duration']:.2f} s")
    if problems:
        shown = "\n  ".join(problems[:10]) + ("\n  ..." if len(problems) > 10 else "")
        raise ValueError(f"{len(problems)} clip(s) outside [{settings.min_seconds}, {settings.max_seconds}] s or "
                         f"unreadable; re-cut or remove them before training:\n  {shown}")


class AudioDataset(Dataset):
    """A manifest-backed dataset, equally usable with or without emotion labels."""

    def __init__(self, records, reader: AudioReader, label_to_id=None):
        self.records = list(records)
        self.reader = reader
        self.label_to_id = label_to_id
        if not self.records:
            raise ValueError("AudioDataset requires at least one clip")
        if label_to_id is not None:
            invalid = {row.label for row in self.records if row.label not in label_to_id}
            if invalid:
                raise ValueError(f"Dataset contains unconfigured labels: {sorted(invalid, key=str)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "source": self.reader.read(row.path),
            "label": None if self.label_to_id is None else self.label_to_id[row.label],
            "record": row,
        }


def collate_audio(items):
    """Right-pad clips; True in padding_mask always means an invalid sample."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([item["source"].numel() for item in items])
    sources = torch.zeros(len(items), int(lengths.max()), dtype=torch.float32)
    padding_mask = torch.arange(sources.shape[1])[None, :] >= lengths[:, None]
    for index, item in enumerate(items):
        sources[index, : lengths[index]] = item["source"]
    labels = [item["label"] for item in items]
    if any(label is None for label in labels) and not all(label is None for label in labels):
        raise ValueError("Cannot mix labeled and unlabeled items in a batch")
    return {
        "sources": sources,
        "padding_mask": padding_mask,
        "labels": None if labels[0] is None else torch.tensor(labels, dtype=torch.long),
        "records": [item["record"] for item in items],
    }


POOL_BATCHES = 100  # duration bucketing sorts random pools of this many batches; larger pools crop less


class BatchOrder:
    """SSL batches as clip indices; a DataLoader sampler whose keys (epoch, batch index, clip indices) feed ClipBatches.

    Training (shuffle=True): the order of an epoch is a pure function of (seed, epoch), so a resumed run rebuilds it
    and skips the consumed batches without decoding audio (set_epoch's start). With bucket=True the clips are
    shuffled, random pools of POOL_BATCHES batches are sorted by duration and cut into batches, and the batch order
    is shuffled: each batch holds clips of nearly equal duration. Validation (shuffle=False): fixed batches, in
    duration order when bucketing, keyed as epoch 0.
    """

    def __init__(self, durations, batch_size: int, seed: int, bucket: bool, shuffle: bool):
        self.durations = np.asarray(durations, dtype=np.float64)
        self.batch_size, self.seed, self.bucket, self.shuffle = batch_size, seed, bucket, shuffle
        self.batches = math.ceil(len(self.durations) / batch_size)
        self.set_epoch(0)

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        count = len(self.durations)
        rng = np.random.default_rng([self.seed, epoch])
        order = rng.permutation(count) if self.shuffle else np.arange(count)
        if self.bucket and self.shuffle:
            pools = np.arange(count) // (self.batch_size * POOL_BATCHES)
            order = order[np.lexsort((self.durations[order], pools))]  # stable: by pool, then by duration
        elif self.bucket:
            order = np.argsort(self.durations, kind="stable")
        self.order, self.epoch, self.start = order, epoch, start
        self.batch_order = rng.permutation(self.batches) if self.bucket and self.shuffle else np.arange(self.batches)

    def __len__(self):
        return self.batches - self.start

    def __iter__(self):
        for index in range(self.start, self.batches):
            first = self.batch_order[index] * self.batch_size
            yield self.epoch, index, self.order[first:first + self.batch_size]

    def cropped_fraction(self) -> float:
        """Share of this epoch's audio that cropping every batch to its shortest clip drops."""
        durations = self.durations[self.order]
        shortest = np.minimum.reduceat(durations, np.arange(0, len(durations), self.batch_size))
        sizes = np.diff(np.append(np.arange(0, len(durations), self.batch_size), len(durations)))
        return float(1 - (shortest * sizes).sum() / durations.sum())


class ClipBatches(Dataset):
    """Whole SSL batches, read for the keys of a BatchOrder (with DataLoader batch_size=None).

    crop=True: every clip is cropped to the batch's shortest clip at an offset drawn from (seed, epoch, batch index),
    then layer-normalised again, so no padding enters the encoder and a batch never depends on worker count or
    resume. crop=False: whole clips, right-padded as in collate_audio.
    """

    def __init__(self, records, reader: AudioReader, seed: int, crop: bool):
        # one bytes array instead of millions of Path objects: cheap to pickle, shared by workers without copying
        self.paths = np.array([str(row.path).encode() for row in records])
        self.reader, self.seed, self.crop = reader, seed, crop
        if not len(self.paths):
            raise ValueError("ClipBatches requires at least one clip")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        epoch, index, clips = key
        paths = [self.paths[clip].decode() for clip in clips]
        waves = [self.reader.decode(path) for path in paths]
        total = sum(len(wave) for wave in waves)
        if self.crop:
            shortest = min(len(wave) for wave in waves)
            rng = np.random.default_rng([self.seed, epoch, index])
            offsets = rng.integers(0, [len(wave) - shortest + 1 for wave in waves])
            waves = [wave[offset:offset + shortest] for wave, offset in zip(waves, offsets.tolist())]
        batch = collate_audio([{"source": self.reader.normalize(wave, path), "label": None, "record": path}
                               for wave, path in zip(waves, paths)])
        batch["cropped_fraction"] = 1 - sum(len(wave) for wave in waves) / total
        return batch
