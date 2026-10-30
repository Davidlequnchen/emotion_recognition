"""A simulated telephone channel (used by evaluate.py): what a call does to the audio before a call centre records it.

  1. resample to 8 kHz (narrowband telephony)
  2. band-pass to the telephone band (300-3,400 Hz by default)
  3. optional white noise at a given signal-to-noise ratio
  4. G.711 encoding and decoding, 8 bits per sample: A-law (Europe, most of Asia including Singapore) or mu-law
     (North America, Japan), with Python's audioop, which implements the ITU-T G.711 companding tables
  5. resample back to 16 kHz for the detector (the band above 4 kHz stays empty)
"""
import warnings

import numpy as np
import torch
import torchaudio

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # audioop is deprecated in 3.11-3.12 (removed in 3.13)
    import audioop


def describe(cfg):
    band = f"{cfg['bandpass'][0]}-{cfg['bandpass'][1]} Hz" if cfg["bandpass"] else "no band-pass"
    noise = f"white noise at {cfg['noise_snr_db']} dB SNR" if cfg["noise_snr_db"] is not None else "no noise"
    return f"{cfg['sample_rate']} Hz, {band}, {noise}, G.711 {cfg['codec']}"


def apply(wav, sr, cfg, rng):
    """wav: 1-D float tensor at sr Hz. Returns (the 16 kHz version for the detector, the 8 kHz version to listen to)."""
    tel_sr = cfg["sample_rate"]
    x = torchaudio.functional.resample(wav, sr, tel_sr)
    if cfg["bandpass"]:
        low, high = cfg["bandpass"]
        for _ in range(2):  # two 2nd-order sections each way: a 4th-order band-pass (24 dB per octave)
            x = torchaudio.functional.highpass_biquad(x, tel_sr, low)
            x = torchaudio.functional.lowpass_biquad(x, tel_sr, high)
    if cfg["noise_snr_db"] is not None:
        power = x.pow(2).mean().clamp_min(1e-10)
        noise = torch.from_numpy(rng.standard_normal(len(x)).astype(np.float32))
        x = x + noise * torch.sqrt(power / 10 ** (cfg["noise_snr_db"] / 10))
    # G.711: 16-bit linear PCM -> 8-bit A-law or mu-law -> back to 16-bit linear
    pcm = (x.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    if cfg["codec"] == "alaw":
        pcm = audioop.alaw2lin(audioop.lin2alaw(pcm, 2), 2)
    elif cfg["codec"] == "ulaw":
        pcm = audioop.ulaw2lin(audioop.lin2ulaw(pcm, 2), 2)
    else:
        raise ValueError(f"telephone.codec must be alaw or ulaw, not {cfg['codec']!r}")
    narrow = torch.from_numpy(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768)
    return torchaudio.functional.resample(narrow, tel_sr, 16000), narrow
