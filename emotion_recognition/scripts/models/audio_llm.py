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
clip whose audio_path, hf_repo and prompts hash are already there is not run again (unless it was given up after NaN
logits, below), so an interrupted run goes on where it stopped. The local models may get every dataset (nothing leaves the machine).

Two by-products of the same model.generate() call as the first answer (no extra pass), each switched on in
config/llm.yaml:
    save_label_probs   - "label_probs" and "label_mass" in the clip's JSON line: the model's probability of each
                         label at the first generated position (first-step logits, before any processing). A label's
                         probability is the sum over the first tokens of its forms "angry", "Angry", " angry",
                         " Angry" and of one-token words that parse_label reads as it ("anger", " fear" ...;
                         label_tokens), renormalised over the labels; label_mass is the share of the whole
                         vocabulary those tokens get (the rest goes to other tokens). evaluate.py writes them as
                         token_p_<label> and token_mass columns; the prediction stays the parsed answer.
    save_hidden_states - the prefill hidden states of `hidden_layers` layers of the language model, averaged over the
                         positions of the audio tokens (over all prompt tokens if they cannot be found), as float16 in
                         features/<model>/<dataset>/ (git-ignored; Features below, read with load_features).
Both come from the first request only (not from a follow-up after an invalid answer), and only for clips run now:
a clip answered earlier (in the cache) gets neither.

Not-a-number logits (seen now and then with Gemma 4 on MPS: from one clip on, every answer of the process was empty,
while a fresh process was fine). Every request also returns the raw logits, and NaN or inf at the first generated
position (or in the pooled hidden states) marks the request as failed; the clip is then run again from its first
request, up to `nan_retries` times (config/llm.yaml), each time with more done first:
    retry 1      the memory cache of the device is emptied (gc, torch.mps.empty_cache);
    retries 2-3  the model's weights are loaded again in this process (the old model freed first).
Still NaN: the clip is given up. Its line in responses.jsonl has label `invalid`, the answers of the last try and
"error": "nan after <n> retries", but no label probabilities and no features; it counts as invalid in this run
(gave_up lists these clips; evaluate.py reports how many per dataset). A line with an "error" is not read back as a
cached answer, so a later run of the model tries such a clip again.
"""
import copy
import gc
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
from parsing import VARIANTS, parse_label
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


def label_tokens(tokenizer, labels):
    """label -> the ids of the first token of each of its forms: the label as "angry", "Angry", " angry", " Angry"
    (its first token even when it is only a piece of the word, e.g. "f" of "f|ear|ful" in the Qwen tokenizer), and
    the same four forms of each other word that parse_label reads as it (VARIANTS of parsing.py: "anger", "fear",
    "surprise" ...) when that form is one whole token (a piece such as "h" of "h|appiness" would also start other
    words). Stops when a token starts two labels, since their probabilities could then not be told apart from the
    first step (scoring whole label sequences would need extra passes; not needed for the three models here, see
    README 8.1)."""
    def forms(word):
        return [word, word.capitalize(), " " + word, " " + word.capitalize()]

    ids = {}
    for lab in labels:
        tokens = {tokenizer.encode(form, add_special_tokens=False)[0] for form in forms(lab)}
        for variant in [v for v, target in VARIANTS.items() if target == lab]:
            tokens |= {t[0] for t in (tokenizer.encode(form, add_special_tokens=False) for form in forms(variant))
                       if len(t) == 1}
        ids[lab] = sorted(tokens)
    owner = {}
    for lab, tokens in ids.items():
        for t in tokens:
            if t in owner:
                sys.exit(f"ERROR: token {tokenizer.decode([t])!r} starts both {owner[t]} and {lab}; the first-step "
                         "label probabilities cannot tell them apart (set save_label_probs: false in config/llm.yaml)")
            owner[t] = lab
    return ids


def even_layers(n_layers, wanted):
    """The layers to save: `wanted` evenly spaced ones ending with the last (an int), or a list of layer numbers
    (1 = first transformer layer, n_layers = last; 0 would be the embeddings)."""
    if isinstance(wanted, int):
        return sorted({round(n_layers * k / wanted) for k in range(1, wanted + 1)})
    return [int(layer) for layer in wanted]


class Features:
    """The pooled hidden states of one model on one dataset, in features/<model>/<dataset>/:
        hidden.f16   float16, one row of (layers x hidden size) per clip, appended clip by clip
        index.jsonl  one line per row: row, audio_path, pooling ("audio" or "prompt": the positions averaged), tokens
        meta.json    model, hf_repo, prompts hash, layers, hidden size, dtype, shape of a row, how rows were pooled
    A row is written before its index line, so a stopped run leaves at most a row without its line, which is cut off
    when the folder is opened again."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.data, self.index_file = self.folder / "hidden.f16", self.folder / "index.jsonl"
        self.meta, self.rows = read_features(self.folder)
        if self.meta is not None:  # cut off what a stopped run left half-written
            with open(self.data, "ab") as fh:
                fh.truncate(len(self.rows) * self.row_bytes())
            self.index_file.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")
        self.paths = {r["audio_path"] for r in self.rows}

    def row_bytes(self):
        return int(np.prod(self.meta["row_shape"])) * 2  # float16: 2 bytes per value

    def add(self, rel, vectors, pooling, tokens, meta):
        """Append one clip's vectors (layers x hidden size); meta: what meta.json must hold (it must not change)."""
        if self.meta is None:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "meta.json").write_text(json.dumps(meta, indent=1))
            self.meta = meta
        elif any(self.meta.get(k) != v for k, v in meta.items()):
            sys.exit(f"ERROR: {self.folder} holds features of other settings ({self.folder / 'meta.json'}); move it "
                     "away or set save_hidden_states: false in config/llm.yaml")
        if rel in self.paths:  # already written (the run stopped before its answer was saved)
            return
        with open(self.data, "ab") as fh:
            fh.write(np.ascontiguousarray(vectors, dtype=np.float16).tobytes())
        row = {"row": len(self.rows), "audio_path": rel, "pooling": pooling, "tokens": tokens}
        with open(self.index_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.rows.append(row)
        self.paths.add(rel)


def read_features(folder):
    """(meta, rows) of a features folder without changing it: meta.json (None if there is none yet) and the lines of
    index.jsonl whose row is complete in hidden.f16."""
    folder = Path(folder)
    if not (folder / "meta.json").exists():
        return None, []
    meta, rows = json.loads((folder / "meta.json").read_text()), []
    index = folder / "index.jsonl"
    for line in index.read_text(encoding="utf-8").splitlines() if index.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a line cut off when a run was killed: it and what follows are dropped
            break
        if row["row"] != len(rows):
            break
        rows.append(row)
    size = (folder / "hidden.f16").stat().st_size if (folder / "hidden.f16").exists() else 0
    return meta, rows[:size // (int(np.prod(meta["row_shape"])) * 2)]  # float16: 2 bytes per value


def load_features(folder):
    """(vectors, index, meta) of a features/<model>/<dataset>/ folder, read-only (safe while a run writes to it):
    vectors is a clips x layers x hidden size float16 array (memory-mapped), index the rows of index.jsonl in the same
    order, meta meta.json. Example: vectors[:, -1] holds every clip's last-layer vector."""
    meta, rows = read_features(folder)
    vectors = np.memmap(Path(folder) / "hidden.f16", dtype=np.float16, mode="r", shape=(len(rows), *meta["row_shape"]))
    return vectors, rows, meta


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
        self.nan_retries = int(settings.get("nan_retries", 3))  # runs of a clip again after NaN logits
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
        self.gave_up = []  # clips given up after NaN logits in every retry (evaluate.py counts them per dataset)
        dtype = str(self.dtype).replace("torch.", "")
        if self.device.type == "mps":
            plain_stop_check_on_mps()
        self.model_dir = model_dir  # kept for a reload after NaN logits (recover())
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
        # Every request also returns the raw logits of each step (the NaN check, the label probabilities); the first
        # request of a clip with save_hidden_states also the hidden states (same tokens: outputs only)
        self.generation.update(return_dict_in_generate=True, output_logits=True)
        self.generation_hidden = copy.deepcopy(self.generation)
        self.generation_hidden.update(output_hidden_states=True)
        # By-products of the first answer (config/llm.yaml): label probabilities and pooled hidden states
        self.save_probs = bool(settings.get("save_label_probs", True))
        self.save_hidden = bool(settings.get("save_hidden_states", True))
        if self.save_probs:
            self.label_ids = label_tokens(self.processor.tokenizer, self.choices)
            print("  label probs : first tokens " + "; ".join(
                f"{lab} {[self.processor.tokenizer.decode([t]) for t in ids]}" for lab, ids in self.label_ids.items()))
        if self.save_hidden:
            config = self.model.config
            # The placeholder token of the audio in the prompt (Gemma 4: audio_token_id; the Qwen models:
            # audio_token_index); its positions are the ones averaged
            self.audio_token = getattr(config, "audio_token_id", None) or getattr(config, "audio_token_index", None)
            self.hidden_layers = settings.get("hidden_layers", 4)
            self.features_dir = Path(entry.get("features_dir") or ROOT / "features" / entry["key"])
            self.features = {}  # dataset -> Features
            self.model_key = entry["key"]
            folder = self.features_dir
            folder = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
            print(f"  features    : {folder}/<dataset>/, hidden_layers {self.hidden_layers}, audio token id "
                  f"{self.audio_token}")
        # What evaluate.py's predictions JSON says about the extra columns (extra())
        self.extra_doc = {"token_p_*": "audio LLMs: the model's probability of each label at the first generated "
                                       "position (first-token probabilities, renormalised over the labels; they add "
                                       "up to 1); the prediction is still the parsed answer",
                          "token_mass": "audio LLMs: the share of the whole vocabulary that the labels' first tokens "
                                        "get at that position"} if self.save_probs else {}
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
                # A clip given up after NaN logits ("error") is not a cached answer: a later run tries it again
                if (record.get("hf_repo") == self.hf_repo and record.get("prompts") == self.prompt_hash
                        and "error" not in record):
                    cache[record["audio_path"]] = record
        return cache

    def generate(self, wav, turns, extras=False):
        """The model's answer (text) to the clip and the conversation so far (turns: [(answer, follow-up), ...]), with
        extras=True what save_label_probs and save_hidden_states ask for, from the same generate() call (a dict:
        label_probs, label_mass, hidden, pooling, tokens), and whether the logits (and hidden states) are finite
        numbers (False: NaN or inf, see the module docstring)."""
        messages = self.family.conversation(self.system, self.user, turns)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  **self.template_kwargs)
        inputs = self.processor(text=text, audio=[wav], sampling_rate=SR, return_tensors="pt")
        # Floating-point inputs (audio features) in the model's dtype; token ids and masks stay as they are
        inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
                  if torch.is_tensor(v) else v for k, v in inputs.items()}
        found, hidden = {}, extras and self.save_hidden
        with torch.inference_mode():
            # Always also the raw logits of each step (those of the first step show NaN); with extras the hidden
            # states too (step 0: the prefill over the whole prompt, one tensor per layer plus the embeddings)
            result = self.model.generate(**inputs,
                                         generation_config=self.generation_hidden if hidden else self.generation)
            first = result.logits[0][0]  # first generated position, whole vocabulary
            finite = bool(torch.isfinite(first).all())
            if finite and extras and self.save_probs:
                found |= self.label_probs(first)
            if finite and hidden:
                found |= self.pooled(result.hidden_states[0], inputs["input_ids"][0])
                finite = bool(np.isfinite(found["hidden"]).all())  # float16 may also overflow to inf
        new = result.sequences[0, inputs["input_ids"].shape[1]:]  # the generated tokens only
        return self.processor.tokenizer.decode(new, skip_special_tokens=True).strip(), found, finite

    def label_probs(self, logits):
        """The labels' probabilities at the first generated position (logits: vocabulary, raw): each label's share of
        the probability of the labels' first tokens, and the share of the vocabulary those tokens get together."""
        p = torch.softmax(logits.float(), dim=-1)
        mass = {lab: float(p[ids].sum()) for lab, ids in self.label_ids.items()}
        total = sum(mass.values())
        return {"label_probs": {lab: round(m / total, 6) for lab, m in mass.items()}, "label_mass": round(total, 6)}

    def pooled(self, hidden, input_ids):
        """hidden: the prefill's hidden states (embeddings, then one per layer; each 1 x prompt length x hidden size)
        -> the chosen layers, averaged over the audio tokens' positions (layers x hidden size, float16), how they were
        pooled ("audio", or "prompt" when the audio tokens cannot be found: every prompt position) and over how many
        positions."""
        self.num_layers = len(hidden) - 1
        self.layers = even_layers(self.num_layers, self.hidden_layers)
        mask = input_ids == (self.audio_token if self.audio_token is not None else -1)  # -1: no token is that
        pooling = "audio" if hidden[0].shape[1] == len(input_ids) and mask.any() else "prompt"
        if pooling == "prompt":
            mask = torch.ones(hidden[0].shape[1], dtype=torch.bool, device=hidden[0].device)
        vectors = torch.stack([hidden[layer][0][mask].float().mean(0) for layer in self.layers])
        return {"hidden": vectors.to(torch.float16).cpu().numpy(), "pooling": pooling, "tokens": int(mask.sum())}

    def save_features(self, rel, found):
        """Append one clip's pooled hidden states to features/<model>/<dataset>/ (the dataset: the folder of the clip
        in data/)."""
        parts = Path(rel).parts
        dataset = parts[1] if len(parts) > 2 and parts[0] == "data" else "other"
        if dataset not in self.features:
            self.features[dataset] = Features(self.features_dir / dataset)
        meta = {"model": self.model_key, "hf_repo": self.hf_repo, "prompts": self.prompt_hash,
                "layers": self.layers, "num_layers": self.num_layers, "row_shape": list(found["hidden"].shape),
                "dtype": "float16", "pooling": "mean over the prefill positions of the audio tokens (index.jsonl: "
                "pooling = audio), or over every prompt position when they cannot be found (pooling = prompt); "
                "first request only; layer 0 = embeddings, num_layers = the last layer"}
        self.features[dataset].add(rel, found["hidden"], found["pooling"], found["tokens"], meta)

    def ask(self, wav):
        """One try at a clip: the first request, then follow-ups while the answer is not exactly one label. Returns
        the answers, the label (None if none), the by-products of the first request, and whether every request had
        finite logits (on NaN the follow-ups stop: they would be NaN too)."""
        answers, label, found = [], None, {}
        for _ in range(self.retries_invalid + 1):  # follow up while the answer is not exactly one label
            answer, extra, finite = self.generate(wav, [(a, self.reminder) for a in answers], extras=not answers)
            found = found or extra  # label probabilities and hidden states: from the first request only
            answers.append(answer)
            if not finite:
                return answers, None, {}, False
            label = parse_label(answers[-1], self.choices)
            if label:
                break
        return answers, label, found, True

    def empty_cache(self):
        """Free what Python and the device's memory cache hold (MPS or CUDA)."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def recover(self, retry, rel):
        """Before retry number `retry` of a clip after NaN logits: retry 1 empties the device's memory cache, later
        retries load the model's weights again (the old model freed first)."""
        what = "empty the memory cache" if retry == 1 else "reload the model weights"
        print(f"  WARNING     : NaN logits for {rel}; retry {retry}/{self.nan_retries}: {what}, then run it again")
        if retry > 1:
            self.model = None  # the only reference to the old model, so the next line frees it
        self.empty_cache()
        if retry > 1:
            t0 = time.perf_counter()
            self.model, self.processor = self.family.build(self.model_dir, self.device, self.dtype)
            print(f"  reloaded    : in {time.perf_counter() - t0:.1f} s")

    def __call__(self, path):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        record = self.cache.get(rel)
        if record is None:
            wav = load_audio(path)
            if self.max_seconds and len(wav) > self.max_seconds * SR:
                wav, self.cut = wav[:int(self.max_seconds * SR)], self.cut + 1
                print(f"  cut         : {rel} to its first {self.max_seconds} s (the longest audio the model takes)")
            t0 = time.perf_counter()
            for retry in range(self.nan_retries + 1):  # the clip again after NaN logits (module docstring)
                if retry:
                    self.recover(retry, rel)
                answers, label, found, finite = self.ask(wav)
                if finite:
                    break
            if "hidden" in found:  # written before the answer, so a stopped run never has an answer without them
                self.save_features(rel, found)
            record = {"audio_path": rel, "hf_repo": self.hf_repo, "prompts": self.prompt_hash, "answers": answers,
                      "label": label or INVALID, "requests": len(answers)}
            if not finite:  # given up: invalid, without label probabilities or features; a later run tries it again
                record["error"] = f"nan after {self.nan_retries} retries"
                self.gave_up.append(rel)
                print(f"  GAVE UP     : {rel}: NaN logits after {self.nan_retries} retries; counted as invalid "
                      f"({len(self.gave_up)} clip(s) given up in this run)")
            if "label_probs" in found:
                record |= {"label_probs": found["label_probs"], "label_mass": found["label_mass"]}
            record |= {"seconds": round(time.perf_counter() - t0, 2), "device": str(self.device),
                       "dtype": str(self.dtype).replace("torch.", ""), "date": str(date.today())}
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "a", encoding="utf-8") as fh:  # saved at once, so a stopped run resumes
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.cache[rel], self.ran = record, self.ran + 1
            if label is None and finite:
                print(f"  invalid     : {rel}: {answers}")
        probs = np.zeros(len(self.labels))
        probs[self.labels.index(record["label"])] = 1.0
        return probs

    def extra(self, path):
        """Extra columns of evaluate.py's predictions CSV for a clip already answered: the first-step probability of
        each label (token_p_<label>) and token_mass; {} for a clip answered without them."""
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        record = self.cache.get(rel, {})
        if "label_probs" not in record:
            return {}
        return {f"token_p_{lab}": record["label_probs"][lab] for lab in self.choices} | {
            "token_mass": record["label_mass"]}
