"""CosyVoice2-0.5B and Fun-CosyVoice3-0.5B-2512 zero-shot voice cloning with the official code (used by generate.py).

The code is the FunAudioLLM/CosyVoice repository at a pinned commit, in third_party/CosyVoice (download_models.py
clones it). Two small changes are made at load time, without editing that code, so that it runs offline from the
model folder:
  - wetext (the text normaliser) would download its FSTs from ModelScope: it gets models/voice_clone/wetext instead.
  - Qwen2Encoder builds the Qwen2.5-0.5B backbone with from_pretrained(), which reads
    CosyVoice-BlankEN/model.safetensors (988 MB), whose weights llm.pt then overwrites, every one of them
    (load_state_dict(strict=True)). Here the backbone is built from CosyVoice-BlankEN/config.json alone, so that
    file is not needed; the model ends up with the same weights.
CosyVoice picks its device itself: CUDA (the current GPU) when PyTorch sees one, otherwise the CPU (no MPS).
"""
import logging
import sys

import torch


def load(model_dir, code_dir, wetext_dir):
    """Load a CosyVoice2 or CosyVoice3 model folder (the type comes from its cosyvoice2.yaml / cosyvoice3.yaml)."""
    sys.path[:0] = [str(code_dir), str(code_dir / "third_party" / "Matcha-TTS")]  # the official code and Matcha-TTS

    import wetext.wetext
    wetext.wetext.snapshot_download = lambda *args, **kwargs: str(wetext_dir)  # FSTs from the local folder

    import cosyvoice.llm.llm
    from transformers import AutoConfig, AutoModelForCausalLM

    def qwen_from_config(self, pretrain_path):
        torch.nn.Module.__init__(self)
        self.model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(pretrain_path),
                                                  torch_dtype=torch.float32)  # as from_pretrained (config says bf16)
    cosyvoice.llm.llm.Qwen2Encoder.__init__ = qwen_from_config  # backbone weights come from llm.pt

    from cosyvoice.cli.cosyvoice import AutoModel
    model = AutoModel(model_dir=str(model_dir))
    # CosyVoice silently drops text normalisation when wetext fails to load (a bare `except:`): fail loudly instead
    if model.frontend.text_frontend != "wetext":
        sys.exit(f"ERROR: CosyVoice's text frontend did not load (wetext FSTs missing in {wetext_dir}?).\n"
                 "       Run: python scripts/voice_clone/download_models.py")
    # CosyVoice sets the root logger to DEBUG, which floods the output (e.g. numba's compiler dumps from librosa)
    logging.getLogger().setLevel(logging.WARNING)
    return model


def components(model):
    """Parameter count of each part, in the order the speech is made."""
    return {"LLM (text -> speech tokens)": model.model.llm, "flow matching (tokens -> mel)": model.model.flow,
            "HiFT vocoder (mel -> waveform)": model.model.hift}


def synthesize(model, text, prompt_text, prompt_wav, speed, text_frontend):
    """One clip: zero-shot cloning of the voice in prompt_wav (whose transcript is prompt_text). CosyVoice splits a
    long text into sentences and makes them one after the other; the pieces are joined. Returns float32 samples."""
    pieces = [out["tts_speech"] for out in model.inference_zero_shot(
        text, prompt_text, str(prompt_wav), stream=False, speed=speed, text_frontend=text_frontend)]
    return torch.cat(pieces, dim=1)[0].cpu().numpy()
