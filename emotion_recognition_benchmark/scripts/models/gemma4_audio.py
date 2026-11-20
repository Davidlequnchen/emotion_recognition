"""Gemma 4 E4B instruction-tuned (Hugging Face google/gemma-4-E4B-it, Apache-2.0): a multimodal LLM with an audio
encoder of 304.8M parameters; 8.00B parameters with the per-layer embeddings (4.5B effective). It answers in text.

Files in models/gemma4_e4b/ (the Hugging Face repository, unchanged):
    model.safetensors                  weights, 16.0 GB bfloat16 (text, vision and audio parts)
    config.json, generation_config.json, processor_config.json (audio: 128 mel bins, 16 kHz, at most 750 audio tokens
                                       of 40 ms = 30 s), chat_template.jinja, tokenizer.json, tokenizer_config.json

The conversation follows the model card: the system prompt (Gemma 4 has a native system role), then one user turn
with the user prompt first and the audio after it (the card: "Audio content after the text in your prompt").
Thinking is off (enable_thinking=False; E4B then writes no thought block). The model takes at most 30 s of audio, so
a longer clip is cut to its first 30 s. Needs transformers >= 5.5 (Gemma 4). Prompts, decoding, parsing and the
cache: audio_llm.py. Used by scripts/evaluation/evaluate.py through load(model_dir, device, entry).
"""
import sys

from audio_llm import TRAINING_NOTE, Predictor, check_files  # noqa: F401 (TRAINING_NOTE is read by evaluate.py)

MAX_SECONDS = 30  # audio_seq_length 750 x 40 ms (processor_config.json)
TEMPLATE_KWARGS = {"enable_thinking": False}
DESCRIPTION = ("Gemma 4 E4B (thinking off) gets one conversation per clip in its chat template: the system prompt, "
               "then one user turn with the user prompt first and the audio after it (as the model card advises)")


def ready():
    """Why the model cannot run in this environment, or None when it can. Called by evaluate.py. Gemma 4's audio
    encoder builds its attention mask in a way that transformers allows only with PyTorch >= 2.6. The unified
    emotion_recognition_llm environment supports it; the optional legacy CUDA 12.1 stack (2.5.1) does not."""
    import torch
    import transformers

    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        return (f"Gemma 4's audio needs PyTorch >= 2.6 (this environment has {torch.__version__}); run it in the "
                "emotion_recognition_llm environment (requirements-llm-*.txt, see README section 1)")
    if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
        return f"Gemma 4 needs transformers >= 5.5 (this environment has {transformers.__version__})"
    return None


def build(model_dir, device, dtype):
    """The model and its processor from model_dir only, in eval mode on `device`."""
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, device_map=str(device), attn_implementation="sdpa", local_files_only=True)
    return model.eval(), processor


def conversation(system, user, turns):
    """The messages: system prompt; user turn with the user prompt and the audio; then each (answer, follow-up)."""
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": user}, {"type": "audio", "audio": "clip"}]}]
    for answer, follow_up in turns:
        messages += [{"role": "assistant", "content": [{"type": "text", "text": answer}]},
                     {"role": "user", "content": [{"type": "text", "text": follow_up}]}]
    return messages


def load(model_dir, device, entry):
    """Gemma 4 E4B from the files in model_dir (never the network), on `device`."""
    check_files(model_dir)
    if ready():
        sys.exit(f"ERROR: {ready()}")
    return Predictor(model_dir, device, entry, sys.modules[__name__])
