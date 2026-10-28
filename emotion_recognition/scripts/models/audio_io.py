"""Read an audio file as the models expect it: mono, 16 kHz, float32 samples in [-1, 1].

Used by the model loaders in this folder whose library does not read files itself (FunASR, which runs emotion2vec,
does). Stereo files are averaged to mono; other sampling rates are resampled with torchaudio.
"""
import functools

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000  # every model here takes 16 kHz audio


@functools.cache
def resampler(rate, sr):
    """The resampler from `rate` to `sr` Hz, built once per pair of rates: building its filter is the slow part (for
    MESS's and TESS's 24,414 Hz about 0.2 s per call, while applying it takes about 10 ms)."""
    return torchaudio.transforms.Resample(rate, sr)


def load_audio(path, sr=SR):
    """The samples of an audio file as a 1-D float32 array at `sr` Hz."""
    wav, rate = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    wav = wav.mean(axis=1)
    if rate != sr:
        with torch.inference_mode():
            wav = resampler(rate, sr)(torch.from_numpy(wav)).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)
