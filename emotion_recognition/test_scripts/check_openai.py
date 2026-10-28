"""
Informal check: does the OpenAI API key work, and can an audio model classify one speech clip? (one paid request)

Step 1 finds the key, step 2 lists the models it can use (free), step 3 sends one CREMA-D test clip with a one-line
prompt to MODEL through Chat Completions and prints the answer and the token usage.

The key comes from the OPENAI_API_KEY environment variable or from a line OPENAI_API_KEY=... in the .env file at the
repository root (git-ignored); it is never printed. Only the Python standard library is used. Cost: a fraction of a
cent.

Usage:
    python test_scripts/check_openai.py     (from emotion_recognition/; edit MODEL below to test another model)
"""
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
REPO = PROJECT.parent                          # repository root, where .env lives
API = "https://api.openai.com/v1"
MODEL = "gpt-audio-mini"  # tested 2026-10-27; gpt-audio-1.5 is the current (more expensive) audio model
CLIP = PROJECT / "data" / "samples" / "1001_DFA_ANG_XX.wav"  # acted emotion: angry
PROMPT = ("Listen to the speaker. Which emotion do they express? Answer with exactly one word from this list: "
          "angry, disgusted, fearful, happy, neutral, sad, surprised.")


def find_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and (REPO / ".env").exists():
        for line in (REPO / ".env").read_text().splitlines():
            if line.strip().startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


def call(method, path, key, body=None):
    """One API request; returns (HTTP status, parsed JSON). Errors come back as their status and message."""
    request = urllib.request.Request(f"{API}{path}", method=method,
                                     data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {"error": {"message": e.reason}}


def main():
    # Step 1: find the key
    key = find_key()
    if not key:
        sys.exit(f"ERROR: no key. Set OPENAI_API_KEY, or put a line OPENAI_API_KEY=... in {REPO / '.env'}")
    print(f"Step 1 | key found ({len(key)} characters, not shown)")

    # Step 2: list the models this key can use (free)
    status, data = call("GET", "/models", key)
    if status != 200:
        sys.exit(f"Step 2 | listing models failed: HTTP {status}: {data.get('error', {}).get('message')}")
    models = sorted(m["id"] for m in data["data"])
    print(f"Step 2 | the key works: {len(models)} models available")
    for word in ["realtime", "audio", "transcribe"]:
        print(f"  {word:<10}: {', '.join(m for m in models if word in m) or '(none)'}")

    # Step 3: one clip and a one-line prompt to an audio model, text answer only
    status, data = call("POST", "/chat/completions", key, {
        "model": MODEL, "modalities": ["text"], "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(CLIP.read_bytes()).decode(),
                                                    "format": "wav"}}]}]})
    if status != 200:
        sys.exit(f"Step 3 | request to {MODEL} failed: HTTP {status}: {data.get('error', {}).get('message')}")
    usage = data["usage"]
    print(f"Step 3 | {data['model']} answered {data['choices'][0]['message']['content']!r} for {CLIP.name} "
          f"(acted emotion: angry)")
    print(f"  tokens  : input {usage['prompt_tokens']} (audio {usage['prompt_tokens_details']['audio_tokens']}), "
          f"output {usage['completion_tokens']}")


if __name__ == "__main__":
    main()
