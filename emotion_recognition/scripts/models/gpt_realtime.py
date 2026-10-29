"""
Loader of GPT Realtime (OpenAI Realtime API, over WebSocket), used by scripts/evaluation/evaluate.py.

load(model_dir, device, entry) returns a Predictor; entry is the model's registry entry (config/models.yaml) with its
key added, and model_dir and device are not used. Called with the path of an audio file, the Predictor sends the clip
to the API and returns a one-hot vector over the labels of the prompts (the seven emotions) plus `invalid`:
    1. the clip becomes 24 kHz 16-bit mono PCM, base64 (prepare_audio in scripts/llm_eval/data_loader.py);
    2. one WebSocket session per clip: the system prompt as the session instructions, text output only, no turn
       detection; one user message with the audio and the user prompt; the answer comes with response.done
       (as in test_scripts/check_openai_realtime.py);
    3. parse_label (scripts/llm_eval/parsing.py) reads the answer as one label; an answer that is not exactly one
       label is asked again up to `retries_invalid` times, then the clip gets `invalid`.
A request that fails for a passing reason (network, rate limit, server error) is tried again up to `retries` times
with a growing pause; then the run stops. Settings and prompts: config/llm.yaml. One request at a time.

Every answer is saved at once as one JSON line in outputs/<model>/responses.jsonl: audio_path, api_model, prompts
(a hash of the prompts and labels), answers (the raw texts), label, usage (tokens), requests, seconds, model (the
snapshot the API reported), date. A clip whose audio_path, api_model and prompts hash are already there is not sent
again, so an interrupted run goes on where it stopped and nothing is paid twice.

The key comes from the OPENAI_API_KEY environment variable or a line OPENAI_API_KEY=... in the .env file at the
repository root (git-ignored); it is never printed. Clips of datasets whose licence does not allow giving their audio
to others (ESD) are refused.
"""
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
REPO = ROOT.parent                          # repository root, where .env lives
CONFIG = ROOT / "config" / "llm.yaml"       # prompts, retries, timeout
sys.path.insert(0, str(ROOT / "scripts" / "llm_eval"))  # data_loader.py, prompts.py, parsing.py
sys.path.insert(0, str(ROOT / "scripts" / "common"))    # eval_datasets.py
from data_loader import prepare_audio
from eval_datasets import SOURCES
from parsing import parse_label
from prompts import load_prompts

URL = "wss://api.openai.com/v1/realtime?model={}"
RATE = 24000          # the Realtime API takes 16-bit PCM at 24 kHz, mono
INVALID = "invalid"   # the class of a clip without a readable answer
# Added to the training-data caveat of every report and summary of the model
TRAINING_NOTE = "The model may have heard these public datasets during training."


class Transient(Exception):
    """A failure worth trying again: network, rate limit, server error."""


def find_key():
    """The OpenAI key from the environment or the .env file at the repository root ("" if there is none)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and (REPO / ".env").exists():
        for line in (REPO / ".env").read_text().splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def ready():
    """Why the model cannot run (no key, websockets missing), or None when it can. Called by evaluate.py."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        return "the websockets package is missing: pip install websockets"
    if not find_key():
        return f"no OPENAI_API_KEY: set it, or put a line OPENAI_API_KEY=... in {REPO / '.env'}"
    return None


async def session(audio, system, user, key, api_model, timeout):
    """One WebSocket session, one request: (answer text, usage, model snapshot)."""
    import websockets

    try:
        async with websockets.connect(URL.format(api_model), additional_headers={"Authorization": f"Bearer {key}"},
                                      max_size=None, open_timeout=timeout) as ws:
            async def send(event):
                await ws.send(json.dumps(event))

            async def wait_for(kind):
                """Read events until one of this type arrives; an error event stops the request."""
                while True:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                    if event["type"] == "error":
                        error = event.get("error", {})
                        message = f"{error.get('code') or error.get('type')}: {error.get('message')}"
                        if error.get("type") == "server_error" or "rate_limit" in str(error.get("code")):
                            raise Transient(message)
                        raise RuntimeError(f"the API refused the request ({message})")
                    if event["type"] == kind:
                        return event

            created = await wait_for("session.created")
            # Text output only, no voice activity detection (the clip is one complete turn), the system prompt as
            # the session instructions (they replace the default ones)
            await send({"type": "session.update", "session": {
                "type": "realtime", "instructions": system, "output_modalities": ["text"],
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": RATE}, "turn_detection": None}}}})
            await wait_for("session.updated")
            await send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [
                {"type": "input_audio", "audio": audio}, {"type": "input_text", "text": user}]}})
            await send({"type": "response.create"})
            response = (await wait_for("response.done"))["response"]
    except websockets.exceptions.InvalidStatus as e:  # the connection was refused, e.g. 401 (key) or 429 (rate)
        status = e.response.status_code
        if status == 429 or status >= 500:
            raise Transient(f"HTTP {status}") from e
        raise RuntimeError(f"the API refused the connection (HTTP {status}); check the key and the model name") from e
    except (OSError, asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
        raise Transient(f"{type(e).__name__}: {e}") from e
    if response.get("status") == "failed":
        raise Transient(f"response failed: {response.get('status_details')}")
    text = "".join(part.get("text", "") or part.get("transcript", "")
                   for item in response.get("output", []) for part in item.get("content", []))
    return text.strip(), response.get("usage", {}), created.get("session", {}).get("model", "")


def request(audio, system, user, key, api_model, timeout):
    """One paid request (the test scripts replace this function with a fake one): (text, usage, model snapshot)."""
    return asyncio.run(session(audio, system, user, key, api_model, timeout))


class Predictor:
    def __init__(self, entry):
        settings = yaml.safe_load(CONFIG.read_text())
        self.api_model = entry["api_model"]
        self.retries, self.retries_invalid = int(settings["retries"]), int(settings["retries_invalid"])
        self.timeout = float(settings["timeout_s"])
        prompts = load_prompts(ROOT / settings["prompts"])
        self.system, self.user, self.choices = prompts["system"], prompts["user"], prompts["labels"]
        self.labels = [*self.choices, INVALID]
        # A hash of the prompts and labels: a cached answer counts only for the same prompts
        self.prompt_hash = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()[:12]
        self.key = find_key()  # never printed
        self.cache_file = ROOT / "outputs" / entry["key"] / "responses.jsonl"
        self.cache = self.read_cache()
        self.on_cpu, self.sent = 0, 0  # on_cpu: always 0 (evaluate.py reads it); sent: requests in this run
        import websockets
        self.versions = {"api model": self.api_model, "websockets": websockets.__version__}
        self.notes = [
            f"Each clip is sent as 24 kHz 16-bit mono PCM to the `{self.api_model}` Realtime API (WebSocket, text "
            f"output, no turn detection), with the system prompt as the session instructions and the user prompt of "
            f"`{settings['prompts']}`",
            f"The one-word answer is read by `scripts/llm_eval/parsing.py` and becomes a one-hot vector over the "
            f"{len(self.choices)} emotions plus `invalid` (no readable answer after asking {self.retries_invalid} more "
            "time(s)); so the probabilities are 0 or 1",
            "The Realtime API does not allow temperature 0, so the answers can vary from run to run",
            "An answer that is not one of the dataset's emotions counts as wrong (column *other answer* of the "
            f"confusion matrices); the raw answers are in `{self.cache_file.relative_to(ROOT)}`",
        ]
        print(f"  cache       : {self.cache_file.relative_to(ROOT)}, {len(self.cache):,} answer(s) for these prompts "
              f"(hash {self.prompt_hash}) and {self.api_model}; those clips are not sent again")

    def read_cache(self):
        """The saved answers for this api_model and these prompts: audio_path -> record."""
        cache = {}
        if self.cache_file.exists():
            for n, line in enumerate(self.cache_file.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # e.g. a line cut off when a run was killed
                    print(f"  WARNING     : line {n} of {self.cache_file.relative_to(ROOT)} is not JSON; ignored")
                    continue
                if record.get("api_model") == self.api_model and record.get("prompts") == self.prompt_hash:
                    cache[record["audio_path"]] = record
        return cache

    def ask(self, audio):
        """One request, tried again after a passing failure: (text, usage, model snapshot)."""
        for attempt in range(self.retries + 1):
            try:
                self.sent += 1
                return request(audio, self.system, self.user, self.key, self.api_model, self.timeout)
            except Transient as e:
                if attempt == self.retries:
                    raise RuntimeError(f"{self.api_model} failed {attempt + 1} times ({e}). The answers so far are in "
                                       f"{self.cache_file.relative_to(ROOT)}; run again later to go on.") from e
                pause = 2 ** (attempt + 1)
                print(f"  retry       : {e}; trying again in {pause} s ({attempt + 1} of {self.retries})")
                time.sleep(pause)

    def __call__(self, path):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        parts = rel.split("/")
        if parts[0] == "data" and len(parts) > 1 and (parts[1] == "esd"
                                                 or not SOURCES.get(parts[1], {}).get("public_audio", True)):
            raise RuntimeError(f"{rel}: the licence of {parts[1]} does not allow giving its audio to others, so it is "
                               "never sent to an API")
        record = self.cache.get(rel)
        if record is None:
            audio, _, _ = prepare_audio(path, "pcm16_24k")
            t0, answers, usage, label, snapshot = time.perf_counter(), [], {}, None, ""
            for _ in range(self.retries_invalid + 1):  # ask again while the answer is not exactly one label
                text, used, snapshot = self.ask(audio)
                answers.append(text)
                details = used.get("input_token_details", {})
                for k, v in (("input_tokens", used.get("input_tokens")), ("output_tokens", used.get("output_tokens")),
                             ("audio_tokens", details.get("audio_tokens"))):
                    usage[k] = usage.get(k, 0) + (v or 0)
                label = parse_label(text, self.choices)
                if label:
                    break
            record = {"audio_path": rel, "api_model": self.api_model, "prompts": self.prompt_hash, "answers": answers,
                      "label": label or INVALID, "usage": usage, "requests": len(answers),
                      "seconds": round(time.perf_counter() - t0, 2), "model": snapshot, "date": str(date.today())}
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "a", encoding="utf-8") as fh:  # saved at once, so nothing is paid twice
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.cache[rel] = record
            if label is None:
                print(f"  invalid     : {rel}: {answers}")
        probs = np.zeros(len(self.labels))
        probs[self.labels.index(record["label"])] = 1.0
        return probs


def load(model_dir, device, entry):
    """The Predictor of the API model in entry (model_dir and device are not used: the model runs at OpenAI)."""
    reason = ready()
    if reason:
        raise RuntimeError(f"GPT Realtime cannot run: {reason}")
    return Predictor(entry)
