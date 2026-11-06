"""Qwen2.5-Omni-7B (Hugging Face Qwen/Qwen2.5-Omni-7B, Apache-2.0), its Thinker only: the part that takes text,
audio, images and video and answers in text (a Whisper-style audio encoder of 639.6M parameters, a vision encoder of
676.6M that is loaded but not used, and the Qwen2.5-7B language model: 8.93B parameters). The Talker and the
token2wav vocoder, which turn the answer into speech, are not loaded.

Files in models/qwen2_5_omni_7b/ (the Hugging Face repository without spk_dict.pt, the Talker's voices):
    model-0000[1-5]-of-00005.safetensors, model.safetensors.index.json   weights, 22.4 GB bfloat16 (Thinker, Talker
                                       and token2wav; shard 5 holds no Thinker weight, but transformers opens every
                                       shard of the index, so it stays)
    config.json, generation_config.json, preprocessor_config.json, chat_template.json, tokenizer files
    LICENSE                            Apache-2.0

The conversation follows the model card: the system prompt, then one user turn with the audio first and the user
prompt after it. The card's own system prompt ("You are Qwen, a virtual human ...") is needed only for speech output;
the Thinker gets the project's system prompt instead, as GPT Realtime does. No video, so use_audio_in_video does not
apply. Prompts, decoding, parsing and the cache: audio_llm.py. Used by scripts/evaluation/evaluate.py through
load(model_dir, device, entry).
"""
import sys

from audio_llm import TRAINING_NOTE, Predictor, check_files  # noqa: F401 (TRAINING_NOTE is read by evaluate.py)

MAX_SECONDS = None  # the audio encoder takes long audio in chunks
# The answer ends at the end of the assistant turn. The repository's generation_config.json is empty, so without these
# the Thinker writes on after its one-word answer (a new "Human:" turn) until max_new_tokens
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
DESCRIPTION = ("Qwen2.5-Omni-7B's Thinker (text output only; the Talker is not loaded) gets one conversation per clip "
               "in its chat template: the system prompt, then one user turn with the audio first and the user prompt "
               "after it (as in the model card)")


def build(model_dir, device, dtype):
    """The Thinker and the processor from model_dir only, in eval mode on `device`."""
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from transformers.utils import logging

    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True)
    # The files also hold the Talker's and token2wav's weights, which transformers lists one by one as unexpected, and
    # it warns about their token ids; those messages are silenced, and a missing Thinker weight stops the run instead
    verbosity = logging.get_verbosity()
    logging.set_verbosity_error()
    try:
        model, info = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, device_map=str(device), attn_implementation="sdpa", local_files_only=True,
            output_loading_info=True)
    finally:
        logging.set_verbosity(verbosity)
    if info["missing_keys"]:
        sys.exit(f"ERROR: {len(info['missing_keys'])} Thinker weights missing in {model_dir}, e.g. "
                 f"{sorted(info['missing_keys'])[:3]}")
    print(f"  Thinker     : loaded; {len(info['unexpected_keys']):,} Talker and token2wav weights left out")
    return model.eval(), processor


def conversation(system, user, turns):
    """The messages: system prompt; user turn with the audio and the user prompt; then each (answer, follow-up)."""
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "audio", "audio": "clip"}, {"type": "text", "text": user}]}]
    for answer, follow_up in turns:
        messages += [{"role": "assistant", "content": [{"type": "text", "text": answer}]},
                     {"role": "user", "content": [{"type": "text", "text": follow_up}]}]
    return messages


def load(model_dir, device, entry):
    """Qwen2.5-Omni-7B's Thinker from the files in model_dir (never the network), on `device`."""
    check_files(model_dir)
    return Predictor(model_dir, device, entry, sys.modules[__name__])
