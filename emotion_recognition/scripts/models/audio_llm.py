"""
Shared part of the loaders of the open-weight audio LLMs that run here (qwen2_5_omni.py, qwen2_audio.py,
gemma4_audio.py), used by scripts/evaluation/evaluate.py through those loaders.

They work like the API loader scripts/models/gpt_realtime.py, so their results compare with GPT Realtime's: the same
system and user prompts and labels (scripts/llm_eval/prompts.yaml, read by prompts.py), the same answer parser
(parse_label in scripts/llm_eval/parsing.py), a one-hot vector over the labels plus `invalid`, and every answer saved
in outputs/<model>/responses.jsonl. Per clip:
    1. the clip is read as 16 kHz mono float32 (load_audio in audio_io.py); a model that takes at most
       `max_seconds` of audio gets the first `max_seconds`;
    2. the conversation (system prompt; user turn with the audio and the user prompt) goes through the model's own
       chat template and processor, then model.generate() with greedy decoding (do_sample=False, no repetition
       penalty) and at most `max_new_tokens` (config/llm.yaml) new tokens;
    3. parse_label reads the answer as one label. Greedy decoding gives the same answer to the same question, so an
       answer that is not exactly one label is followed up in the same conversation instead (the answer as the
       assistant turn, then REMINDER as a new user turn), up to `retries_invalid` times (config/llm.yaml); still no
       label: the clip gets `invalid`.
Device and precision: bfloat16 on CUDA and on MPS (checked on the M4 Max: see test_scripts/check_local_audio_llms.py),
float32 on the CPU; scaled dot-product attention (sdpa); the weights load straight onto the device (device_map).

Every answer is saved at once as one JSON line in outputs/<model>/responses.jsonl: audio_path, hf_repo, prompts (a
hash of the prompts, labels and REMINDER), answers (the raw texts), label, requests, seconds, device, dtype, date. A
clip whose audio_path, hf_repo and prompts hash are already there is not run again, so an interrupted run goes on
where it stopped. The local models may get every dataset (nothing leaves the machine).
"""
import copy
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

from audio_io import SR, load_audio  # scripts/models/audio_io.py

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "llm.yaml"       # prompts, retries_invalid, max_new_tokens
sys.path.insert(0, str(ROOT / "scripts" / "llm_eval"))  # prompts.py, parsing.py
from parsing import parse_label
from prompts import load_prompts

INVALID = "invalid"  # the class of a clip without a readable answer
# The follow-up question after an answer that is not exactly one label ({labels}: the list, as in the prompts)
REMINDER = "Answer with exactly one word from this list, in lowercase, and nothing else: {labels}."
# Added to the training-data caveat of every report and summary of these models
TRAINING_NOTE = "The model may have heard these public datasets during training."


def dtype_for(device):
    """bfloat16 on a GPU (CUDA, MPS), float32 on the CPU."""
    return torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16


def plain_stop_check_on_mps():
    """On MPS, transformers (5.18) checks one step late whether generation should stop, through torch.Event, which
    crashes torch 2.5.1 (a segmentation fault in the first generate()). Use its plain check instead, as on CUDA: the
    same tokens, slightly slower."""
    from transformers.generation import utils

    utils.DeferredStopCheck.is_supported = staticmethod(lambda *args, **kwargs: False)


def check_files(model_dir):
    """Stop with a message when the model's weights are not in model_dir."""
    if not any(model_dir.glob("*.safetensors")):
        sys.exit(f"ERROR: no weights in {model_dir}; run python scripts/common/download_model.py")


class Predictor:
    """Runs one audio LLM on one clip at a time. `family` is the loader module, which gives:
        build(model_dir, device, dtype) -> (model, processor)
        conversation(system, user, turns) -> the messages for the chat template; turns: [(answer, follow-up), ...]
        TEMPLATE_KWARGS (dict, optional): extra arguments of apply_chat_template, e.g. enable_thinking=False
        STOP_TOKENS (list, optional): end-of-turn tokens that end the answer, when the model's generation settings
                     lack them
        MAX_SECONDS (number or None): the longest audio the model takes
        DESCRIPTION: what the model sees, a sentence for the reports"""

    def __init__(self, model_dir, device, entry, family):
        import transformers

        settings = yaml.safe_load(CONFIG.read_text())
        prompts = load_prompts(ROOT / settings["prompts"])
        self.system, self.user, self.choices = prompts["system"], prompts["user"], prompts["labels"]
        self.reminder = REMINDER.format(labels=", ".join(self.choices))
        self.labels = [*self.choices, INVALID]
        self.retries_invalid, self.max_new_tokens = int(settings["retries_invalid"]), int(settings["max_new_tokens"])
        # A hash of the prompts, labels and reminder: a cached answer counts only for the same questions
        self.prompt_hash = hashlib.sha256(json.dumps({**prompts, "reminder": self.reminder}, sort_keys=True)
                                          .encode()).hexdigest()[:12]
        self.family, self.device, self.dtype = family, torch.device(device), dtype_for(device)
        self.hf_repo = entry["hf_repo"]
        self.max_seconds = getattr(family, "MAX_SECONDS", None)
        self.template_kwargs = getattr(family, "TEMPLATE_KWARGS", {})
        # entry["cache_file"] only in the test scripts, so that their answers stay out of outputs/
        self.cache_file = Path(entry.get("cache_file") or ROOT / "outputs" / entry["key"] / "responses.jsonl")
        self.cache = self.read_cache()
        self.on_cpu, self.ran, self.cut = 0, 0, 0  # on_cpu: always 0 (evaluate.py reads it); ran: clips run now
        dtype = str(self.dtype).replace("torch.", "")
        if self.device.type == "mps":
            plain_stop_check_on_mps()
        self.model, self.processor = family.build(model_dir, self.device, self.dtype)
        # Greedy decoding: the model's generation settings (end and padding tokens) without its sampling settings
        # and repetition penalty (Qwen2-Audio's would also penalise the labels, which are in the prompt)
        self.generation = copy.deepcopy(self.model.generation_config)
        self.generation.update(do_sample=False, temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0,
                               max_new_tokens=self.max_new_tokens)
        # The end-of-turn tokens of the chat template, for a model whose generation settings lack them (Qwen2.5-Omni's
        # Thinker: its generation_config.json is empty, so it would write on after its answer)
        stops = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in getattr(family, "STOP_TOKENS", [])]
        if stops:
            known = self.generation.eos_token_id
            known = [] if known is None else [known] if isinstance(known, int) else list(known)
            self.generation.eos_token_id = list(dict.fromkeys(known + stops))
            if self.generation.pad_token_id is None:
                self.generation.pad_token_id = stops[-1]
        self.versions = {"transformers": transformers.__version__}
        where = self.cache_file.relative_to(ROOT) if self.cache_file.is_relative_to(ROOT) else self.cache_file
        self.notes = [
            f"{family.DESCRIPTION}, with the system and user prompts of `{settings['prompts']}` (the same as for GPT "
            f"Realtime); the clip as 16 kHz mono" + (f" (at most its first {self.max_seconds} s)" if self.max_seconds
                                                    else ""),
            f"Greedy decoding (do_sample=False, no repetition penalty), at most {self.max_new_tokens} new tokens, "
            f"{dtype} on `{self.device}`, sdpa attention; so the answers are the same from run to run (up to "
            "rounding differences between devices)",
            f"The answer is read by `scripts/llm_eval/parsing.py` and becomes a one-hot vector over the "
            f"{len(self.choices)} emotions plus `invalid`. An answer that is not exactly one label gets a follow-up "
            f"in the same conversation (\"{self.reminder}\"), up to {self.retries_invalid} time(s); then `invalid`",
            "An answer that is not one of the dataset's emotions counts as wrong (column *other answer* of the "
            f"confusion matrices); the raw answers are in `{where}`",
        ]
        print(f"  cache       : {where}, {len(self.cache):,} answer(s) for these prompts (hash {self.prompt_hash}) and "
              f"{self.hf_repo}; those clips are not run again")

    def read_cache(self):
        """The saved answers for this model and these prompts: audio_path -> record."""
        cache = {}
        if self.cache_file.exists():
            for n, line in enumerate(self.cache_file.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # e.g. a line cut off when a run was killed
                    print(f"  WARNING     : line {n} of {self.cache_file} is not JSON; ignored")
                    continue
                if record.get("hf_repo") == self.hf_repo and record.get("prompts") == self.prompt_hash:
                    cache[record["audio_path"]] = record
        return cache

    def generate(self, wav, turns):
        """The model's answer (text) to the clip and the conversation so far (turns: [(answer, follow-up), ...])."""
        messages = self.family.conversation(self.system, self.user, turns)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.template_kwargs)
        inputs = self.processor(text=text, audio=[wav], sampling_rate=SR, return_tensors="pt")
        # Floating-point inputs (audio features) in the model's dtype; token ids and masks stay as they are
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  if torch.is_tensor(v) else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, generation_config=self.generation)
        new = out[0, inputs["input_ids"].shape[1]:]  # the generated tokens only
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip()

    def __call__(self, path):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        record = self.cache.get(rel)
        if record is None:
            wav = load_audio(path)
            if self.max_seconds and len(wav) > self.max_seconds * SR:
                wav, self.cut = wav[:int(self.max_seconds * SR)], self.cut + 1
                print(f"  cut         : {rel} to its first {self.max_seconds} s (the longest audio the model takes)")
            t0, answers, label = time.perf_counter(), [], None
            for _ in range(self.retries_invalid + 1):  # follow up while the answer is not exactly one label
                answers.append(self.generate(wav, [(a, self.reminder) for a in answers]))
                label = parse_label(answers[-1], self.choices)
                if label:
                    break
            record = {"audio_path": rel, "hf_repo": self.hf_repo, "prompts": self.prompt_hash, "answers": answers,
                      "label": label or INVALID, "requests": len(answers),
                      "seconds": round(time.perf_counter() - t0, 2), "device": str(self.device),
                      "dtype": str(self.dtype).replace("torch.", ""), "date": str(date.today())}
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "a", encoding="utf-8") as fh:  # saved at once, so a stopped run resumes
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.cache[rel], self.ran = record, self.ran + 1
            if label is None:
                print(f"  invalid     : {rel}: {answers}")
        probs = np.zeros(len(self.labels))
        probs[self.labels.index(record["label"])] = 1.0
        return probs
