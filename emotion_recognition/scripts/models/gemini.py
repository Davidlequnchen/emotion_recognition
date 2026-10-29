"""
Loader of Google Gemini, used by scripts/evaluation/evaluate.py: a PLACEHOLDER, not implemented yet.

ready() says so, and evaluate.py skips the model with that message. The implementation will work like
scripts/models/gpt_realtime.py, with the same interface, prompts, parsing and cache:
    - load(model_dir, device, entry) returns a Predictor (labels: the seven emotions of the prompts plus `invalid`;
      versions, notes, on_cpu = 0); entry["api_model"] is the Gemini model name;
    - each clip becomes a 16 kHz mono 16-bit WAV file (prepare_audio(path, "wav16k") in
      scripts/llm_eval/data_loader.py), sent as inline data (mime type audio/wav) with the user prompt, through the
      google-genai SDK (pip install google-genai); the system prompt is the system instruction, temperature 0;
    - parse_label (scripts/llm_eval/parsing.py) reads the answer; one-hot vector; the same retries and
      retries_invalid (config/llm.yaml);
    - every answer saved in outputs/<model>/responses.jsonl, so nothing is paid twice;
    - the key from GEMINI_API_KEY or the .env file at the repository root, never printed; no ESD clips. Use a paid
      plan: on the free tier Google may use what is sent to improve its products.
"""


def ready():
    """Why the model cannot run, or None when it can. Called by evaluate.py."""
    return "Gemini is not implemented yet (placeholder)"


def load(model_dir, device, entry):
    raise NotImplementedError(ready())
