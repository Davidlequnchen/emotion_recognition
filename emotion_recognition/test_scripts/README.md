# Informal test scripts

Quick checks that are not part of the pipelines in `scripts/`. Run them from the `emotion_recognition/` folder.
The OpenAI checks read the key from the `OPENAI_API_KEY` environment variable or from a line `OPENAI_API_KEY=...` in
the `.env` file at the repository root (git-ignored), and never print it.

| Script | What it does | Cost |
|---|---|---|
| `check_openai.py` | Lists the models the key can use (free), then sends one test clip to an audio model (`gpt-audio-mini`) through Chat Completions | one request, a fraction of a cent |
| `check_openai_realtime.py` | Sends the same clip to each Realtime model (`gpt-realtime`, `gpt-realtime-2.1`) over WebSocket, as 24 kHz 16-bit mono PCM, and prints the answer and token usage | one request per model, a fraction of a cent each |

Results on 2026-10-27 with `data/samples/1001_DFA_ANG_XX.wav` (acted emotion: angry): `gpt-audio-mini`,
`gpt-realtime` and `gpt-realtime-2.1` all answered `neutral`. The clip took 22 audio tokens (about 10 per second).
A Realtime session adds about 110 text tokens of default instructions unless the session sets its own, and
`gpt-realtime-2.1` used 44 output tokens for a one-word answer (its reasoning is counted as output).
