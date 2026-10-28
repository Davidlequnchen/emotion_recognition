"""
Informal check: can the OpenAI Realtime models classify one speech clip? (one paid request per model)

For each model in MODELS: open a Realtime session over WebSocket, ask for text output only, send one CREMA-D test
clip (converted to the 24 kHz 16-bit mono PCM that the Realtime API requires) with a one-line prompt, and print the
answer and the token usage. The Realtime API has no structured outputs, and its temperature cannot be 0.

The key comes from the OPENAI_API_KEY environment variable or from a line OPENAI_API_KEY=... in the .env file at the
repository root (git-ignored); it is never printed. Cost: a fraction of a cent per model.

Usage:
    python test_scripts/check_openai_realtime.py     (from emotion_recognition/; edit MODELS below to test others)
"""
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import websockets

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
REPO = PROJECT.parent                          # repository root, where .env lives
MODELS = ["gpt-realtime", "gpt-realtime-2.1"]
CLIP = PROJECT / "data" / "samples" / "1001_DFA_ANG_XX.wav"  # acted emotion: angry
PROMPT = ("Listen to the speaker. Which emotion do they express? Answer with exactly one word from this list: "
          "angry, disgusted, fearful, happy, neutral, sad, surprised.")
RATE = 24000  # the Realtime API takes 16-bit PCM at 24 kHz, mono


def find_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and (REPO / ".env").exists():
        for line in (REPO / ".env").read_text().splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def pcm16_base64(path):
    """The clip as base64 16-bit PCM at 24 kHz, mono."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(axis=1)), sr, RATE).numpy()
    return base64.b64encode((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()).decode()


async def classify(model, key, audio):
    """One session, one request: returns (answer text, usage, seconds)."""
    t0 = time.perf_counter()
    async with websockets.connect(f"wss://api.openai.com/v1/realtime?model={model}",
                                  additional_headers={"Authorization": f"Bearer {key}"},
                                  max_size=None, open_timeout=30) as ws:
        async def send(event):
            await ws.send(json.dumps(event))

        async def wait_for(*types):
            """Read events until one of the given types arrives; stop on an error event."""
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                if event["type"] == "error":
                    raise RuntimeError(event["error"].get("message"))
                if event["type"] in types:
                    return event

        # 1. Session: text output only, no voice activity detection (the clip is one complete turn)
        await wait_for("session.created")
        await send({"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["text"],
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": RATE}, "turn_detection": None}}}})
        await wait_for("session.updated")
        # 2. One user message with the clip and the prompt, then ask for a response
        await send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [
            {"type": "input_audio", "audio": audio}, {"type": "input_text", "text": PROMPT}]}})
        await send({"type": "response.create"})
        done = await wait_for("response.done")
    response = done["response"]
    text = "".join(part.get("text", "") or part.get("transcript", "")
                   for item in response.get("output", []) for part in item.get("content", []))
    return text.strip(), response.get("usage", {}), time.perf_counter() - t0


def main():
    key = find_key()
    if not key:
        sys.exit(f"ERROR: no key. Set OPENAI_API_KEY, or put a line OPENAI_API_KEY=... in {REPO / '.env'}")
    print(f"key     : found ({len(key)} characters, not shown)")
    print(f"clip    : {CLIP.name} (acted emotion: angry), sent as 24 kHz 16-bit mono PCM")
    audio = pcm16_base64(CLIP)
    for model in MODELS:
        try:
            text, usage, seconds = asyncio.run(classify(model, key, audio))
        except Exception as e:  # report and go on with the next model; no retry, to keep the cost down
            print(f"\n{model}: FAILED: {type(e).__name__}: {e}")
            continue
        details = usage.get("input_token_details", {})
        print(f"\n{model}: answered {text!r} in {seconds:.1f} s")
        print(f"  tokens  : input {usage.get('input_tokens')} (audio {details.get('audio_tokens')}, "
              f"text {details.get('text_tokens')}), output {usage.get('output_tokens')}")


if __name__ == "__main__":
    main()
