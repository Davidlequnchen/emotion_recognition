"""Chatterbox (English, Resemble AI) zero-shot voice cloning with the chatterbox-tts package (used by generate.py).

ChatterboxTTS.from_local() reads the model folder only (ve, t3_cfg, s3gen, tokenizer, conds); nothing is downloaded.
Upstream always adds Resemble's Perth watermark (inaudible, marks the clip as synthetic) to the output; it is kept.
"""
import numpy as np


def load(model_dir, device):
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_local(model_dir, str(device))


def components(model):
    """Parameter count of each part, in the order the speech is made."""
    return {"voice encoder (speaker embedding)": model.ve, "T3 LLM (text -> speech tokens)": model.t3,
            "S3Gen (tokens -> waveform)": model.s3gen}


def prepare(model, reference_wav, exaggeration):
    """Speaker conditioning from the reference (once per reference; the same for every clip)."""
    model.prepare_conditionals(str(reference_wav), exaggeration=exaggeration)


def synthesize(model, text, p):
    """One clip with the conditioning of prepare(). Returns float32 samples."""
    wav = model.generate(text, exaggeration=p["exaggeration"], cfg_weight=p["cfg_weight"], temperature=p["temperature"],
                         repetition_penalty=p["repetition_penalty"], min_p=p["min_p"], top_p=p["top_p"])
    return wav[0].cpu().numpy().astype(np.float32)
