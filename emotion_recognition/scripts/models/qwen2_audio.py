"""Qwen2-Audio-7B-Instruct (Hugging Face Qwen/Qwen2-Audio-7B-Instruct, Apache-2.0): an audio LLM, a Whisper-large-v3
style audio encoder (637.0M parameters) in front of the Qwen-7B language model, 8.40B parameters; it answers in text.

Files in models/qwen2_audio_7b/ (the Hugging Face repository, unchanged):
    model-0000[1-5]-of-00005.safetensors, model.safetensors.index.json   weights, 16.8 GB bfloat16
    config.json, generation_config.json                                  model and default sampling settings
    preprocessor_config.json                     Whisper feature extractor: 128 mel bins, at most 30 s of audio
    tokenizer.json, tokenizer_config.json, vocab.json, merges.txt        tokenizer and chat template (ChatML)

The conversation follows the model card's "audio analysis" example: the system prompt, then one user turn with the
audio first ("Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>") and the user prompt after it. The model takes at most
30 s of audio (Whisper's window), so a longer clip is cut to its first 30 s. Prompts, decoding, parsing and the cache:
audio_llm.py. Used by scripts/evaluation/evaluate.py through load(model_dir, device, entry).
"""
import sys

from audio_llm import TRAINING_NOTE, Predictor, check_files  # noqa: F401 (TRAINING_NOTE is read by evaluate.py)

MAX_SECONDS = 30  # Whisper's window: the processor drops what comes after it
DESCRIPTION = ("Qwen2-Audio-7B-Instruct gets one conversation per clip in its chat template: the system prompt, then "
               "one user turn with the audio first and the user prompt after it (as in the model card)")


def build(model_dir, device, dtype):
    """The model and its processor from model_dir only, in eval mode on `device`."""
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, device_map=str(device), attn_implementation="sdpa", local_files_only=True)
    return model.eval(), processor


def conversation(system, user, turns):
    """The messages: system prompt; user turn with the audio and the user prompt; then each (answer, follow-up)."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "audio", "audio_url": "clip"}, {"type": "text", "text": user}]}]
    for answer, follow_up in turns:
        messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": follow_up}]
    return messages


def load(model_dir, device, entry):
    """Qwen2-Audio-7B-Instruct from the files in model_dir (never the network), on `device`."""
    check_files(model_dir)
    return Predictor(model_dir, device, entry, sys.modules[__name__])
