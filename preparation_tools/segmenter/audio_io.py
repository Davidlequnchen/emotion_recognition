"""Audio reading and writing with soundfile (WAV, FLAC, MP3 through the bundled libsndfile).

The source audio is decoded once and kept at its own sample rate and channels for export; a mono 16 kHz
copy is made for VAD and alignment only.
"""
import soundfile as sf
import torch
import torchaudio.functional as AF

ANALYSIS_RATE = 16000
# Linear PCM / float subtypes kept as they are when a WAV file supports them; anything else (MP3, mu-law, and
# PCM_S8 from FLAC, which WAV cannot store because WAV 8-bit is unsigned) is exported as 16-bit PCM.
LINEAR_SUBTYPES = {"PCM_U8", "PCM_S8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}


def read_audio(path):
    """(samples as float64 array [frames, channels], sample rate, source subtype)."""
    info = sf.info(str(path))
    data, rate = sf.read(str(path), dtype="float64", always_2d=True)
    return data, rate, info.subtype


def analysis_copy(data, rate):
    """Mono float32 tensor at 16 kHz (channel mean, then resampling)."""
    wave = torch.from_numpy(data.mean(axis=1)).float()
    if rate != ANALYSIS_RATE:
        wave = AF.resample(wave, rate, ANALYSIS_RATE)
    return wave


def wav_subtype(source_subtype):
    """The WAV subtype for a clip: the source subtype if WAV can hold it losslessly, else PCM_16."""
    if source_subtype in LINEAR_SUBTYPES and sf.check_format("WAV", source_subtype):
        return source_subtype
    return "PCM_16"


def write_clip(path, data, rate, source_subtype):
    """Write one segment as WAV at the source rate and channels."""
    subtype = wav_subtype(source_subtype)
    sf.write(str(path), data, rate, subtype=subtype, format="WAV")
    return subtype
