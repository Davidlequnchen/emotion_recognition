"""
Voice cloning: one synthetic clip per line of texts.csv, in the voice of a reference recording, with each model of
config/voice_clone.yaml (CosyVoice2-0.5B, Fun-CosyVoice3-0.5B-2512, Chatterbox).

Every step prints what it does, so you can check that Python, PyTorch, the device and the models are all in order.
Models run one after the other, and clips one at a time: a model is freed before the next one is loaded.

Usage (from the antideepfake/ folder, in the voice_clone environment):
    python scripts/voice_clone/generate.py
Settings are in config/voice_clone.yaml: `mode` and `devices` choose the hardware, `models` the models, `generate`
the reference recording, the texts, the seed and the output folder. Get the models first:
python scripts/voice_clone/download_models.py

Reference: data/voice_clone/reference/<speaker>_<YYYYmmdd_HHMMSS>.wav with its transcript in the .txt of the same
name (scripts/voice_clone/record_reference.py writes both). CosyVoice uses the audio and the transcript, Chatterbox
the audio only.
Output: outputs/voice_clone/<speaker>/<model>/<id>.wav (16-bit PCM, the model's own sample rate) and a manifest.csv
in that folder with one row per clip (text, reference and its SHA-256, seed, duration, timing, versions).

Works offline: HF_HUB_OFFLINE=1 is set below, and every file comes from models/voice_clone/ and third_party/.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # never contact Hugging Face: everything is in models/voice_clone/
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # no tokenizer threads forked with the process (quiet, no warning)
os.environ["TQDM_DISABLE"] = "1"  # no progress bars from inside the models: one line per clip is printed instead

import csv
import gc
import hashlib
import json
import platform
import random
import re
import resource
import subprocess
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")  # from perth and pyworld
warnings.filterwarnings("ignore", category=FutureWarning)  # deprecation notices from diffusers and torch

ROOT = Path(__file__).resolve().parents[2]  # project root (antideepfake/); paths in the config are relative to it
CONFIG = ROOT / "config" / "voice_clone.yaml"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # the loaders next to this file
STAMP = re.compile(r"^(?P<speaker>.+)_(?P<stamp>\d{8}_\d{6})$")  # <speaker>_<YYYYmmdd_HHMMSS>
MANIFEST_COLUMNS = ["id", "scenario", "role", "text", "file", "reference", "reference_sha256", "seed", "sample_rate",
                    "duration_s", "synth_seconds", "rtf", "model", "hf_repo", "hf_revision", "code", "torch",
                    "transformers", "device", "params"]


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def peak_ram_gb():
    """Peak resident memory of this process so far (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2


def find_reference(g):
    """The reference .wav of the config, its transcript, and the speaker name (the stem without the time stamp)."""
    if g["reference"] == "latest":
        folder = ROOT / g["reference_dir"]
        candidates = [p for p in folder.glob("*.wav") if STAMP.match(p.stem) and p.with_suffix(".txt").exists()]
        if not candidates:
            sys.exit(f"ERROR: no <speaker>_<YYYYmmdd_HHMMSS>.wav with its .txt transcript in {folder}.\n"
                     "       Record one with: python scripts/voice_clone/record_reference.py")
        wav = max(candidates, key=lambda p: STAMP.match(p.stem)["stamp"])  # the newest time stamp
    else:
        wav = ROOT / g["reference"]
        if not wav.exists() or not wav.with_suffix(".txt").exists():
            sys.exit(f"ERROR: {wav} or its transcript {wav.with_suffix('.txt').name} not found.")
    transcript = wav.with_suffix(".txt").read_text(encoding="utf-8").strip()
    if not transcript:
        sys.exit(f"ERROR: the transcript {wav.with_suffix('.txt')} is empty.")
    match = STAMP.match(wav.stem)
    return wav, transcript, match["speaker"] if match else wav.stem


def read_manifest(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS


def free(device):
    """Give the memory of the model back before the next one is loaded."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order with warnings, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the configuration (config/voice_clone.yaml)
    # ------------------------------------------------------------------
    step("Step 1/6 | Load configuration")
    cfg = yaml.safe_load(CONFIG.read_text())
    registry, g = cfg["registry"], cfg["generate"]
    keys = list(registry) if cfg["models"] == "all" else cfg["models"]
    unknown = [k for k in keys if k not in registry]
    if unknown:
        sys.exit(f"ERROR: unknown model(s) {unknown} in `models`. Registered: {', '.join(registry)}")
    mode = cfg["mode"]
    if mode == "auto":
        mode = "cuda" if torch.cuda.is_available() else "mac"
    print(f"  config      : {CONFIG}")
    print(f"  mode        : {cfg['mode']} -> {mode}")
    print(f"  models      : {', '.join(keys)}")
    print(f"  texts       : {g['texts']} (limit: {g['limit'] or 'none'})")
    print(f"  seed        : {g['seed']} + row index")
    print(f"  overwrite   : {g['overwrite']}")
    print(f"  output      : {g['out']}")

    # ------------------------------------------------------------------
    # Step 2: Check the software stack and which accelerators PyTorch can see
    # ------------------------------------------------------------------
    step("Step 2/6 | Check environment")
    print(f"  Python       : {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"  PyTorch      : {torch.__version__}, torchaudio {version('torchaudio')}")
    print(f"  transformers : {version('transformers')}")
    print(f"  chatterbox   : chatterbox-tts {version('chatterbox-tts')}")
    print(f"  onnxruntime  : {version('onnxruntime')} (CosyVoice's speech tokenizer and speaker encoder)")
    print(f"  CUDA         : available={torch.cuda.is_available()} (PyTorch built with CUDA {torch.version.cuda})")
    print(f"  MPS (Apple)  : available={torch.backends.mps.is_available()} (built={torch.backends.mps.is_built()})")
    print(f"  offline      : HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}, "
          f"TRANSFORMERS_OFFLINE={os.environ['TRANSFORMERS_OFFLINE']}")
    code_dir = ROOT / cfg["cosyvoice_code"]["folder"]
    if {"cosyvoice2", "cosyvoice3"} & set(keys):
        if not (code_dir / "cosyvoice").exists():
            sys.exit(f"ERROR: {code_dir} not found. Run: python scripts/voice_clone/download_models.py")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir, capture_output=True, text=True).stdout.strip()
        print(f"  CosyVoice    : {code_dir} @ {commit[:12]}")
        if commit != cfg["cosyvoice_code"]["commit"]:
            sys.exit(f"ERROR: {code_dir} is at {commit}, not at the pinned {cfg['cosyvoice_code']['commit']}.\n"
                     "       Run: python scripts/voice_clone/download_models.py")

    # ------------------------------------------------------------------
    # Step 3: The reference recording and its transcript
    # ------------------------------------------------------------------
    step("Step 3/6 | Reference recording")
    ref_wav, transcript, speaker = find_reference(g)
    info = sf.info(ref_wav)
    ref_sha = sha256(ref_wav)
    print(f"  file        : {ref_wav.relative_to(ROOT) if ref_wav.is_relative_to(ROOT) else ref_wav}")
    print(f"  speaker     : {speaker}")
    print(f"  audio       : {info.samplerate} Hz, {info.channels} ch, {info.subtype}, {info.duration:.2f} s")
    print(f"  transcript  : {transcript}")
    print(f"  SHA-256     : {ref_sha}")
    if info.duration > 30:
        sys.exit("ERROR: the reference is longer than 30 s, which CosyVoice's speech tokenizer refuses. Use 5-10 s.")
    if not 5 <= info.duration <= 10:
        print("  WARNING: 5-10 s is recommended (Chatterbox uses the first 6 s for the LLM and 10 s for S3Gen)")

    # ------------------------------------------------------------------
    # Step 4: The texts (one clip per row)
    # ------------------------------------------------------------------
    step("Step 4/6 | Texts")
    texts_path = ROOT / g["texts"]
    if not texts_path.exists():
        sys.exit(f"ERROR: {texts_path} not found.")
    with open(texts_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    missing = {"id", "scenario", "role", "text"} - set(rows[0] if rows else {})
    if missing:
        sys.exit(f"ERROR: {texts_path} has no column(s) {sorted(missing)} (expected id,scenario,role,text).")
    rows = [dict(r, seed=g["seed"] + i) for i, r in enumerate(rows)][:g["limit"] or None]  # seed from the full file
    print(f"  {texts_path.name}: {len(rows)} row(s) to make")
    for r in rows[:3]:
        print(f"  {r['id']} [{r['scenario']}, {r['role']}] {r['text'][:90]}")

    # ------------------------------------------------------------------
    # Step 5: Each model in turn: load, make every clip, free
    # ------------------------------------------------------------------
    step("Step 5/6 | Generate")
    summary = []
    for n, key in enumerate(keys, 1):
        m = registry[key]
        model_dir = ROOT / m["folder"]
        out_dir = ROOT / g["out"] / speaker / key
        family = "chatterbox" if key == "chatterbox" else "cosyvoice"
        device = torch.device(cfg["devices"][mode] if mode == "cuda" else m.get("mac_device", cfg["devices"]["mac"]))
        print(f"\n--- [{n}/{len(keys)}] {key}: {m['name']} ({m['licence']}) ---")
        if not model_dir.exists() or not all((model_dir / f).exists() for f in m["files"]):
            sys.exit(f"ERROR: {model_dir} is incomplete. Run: python scripts/voice_clone/download_models.py")

        # Skip the clips that exist already (unless overwrite), and never mix two references in one folder
        manifest_path = out_dir / "manifest.csv"
        manifest = read_manifest(manifest_path)
        other = {r["reference_sha256"] for r in manifest.values()} - {ref_sha}
        if other and not g["overwrite"]:
            sys.exit(f"ERROR: {out_dir} holds clips made from another reference recording.\n"
                     "       Move that folder away, or set generate.overwrite: true to replace them.")
        todo = [r for r in rows if g["overwrite"] or not (out_dir / f"{r['id']}.wav").exists()]
        print(f"  output      : {out_dir.relative_to(ROOT)}  ({len(rows) - len(todo)} done, {len(todo)} to make)")
        if not todo:
            continue

        # Device: CosyVoice chooses it itself (CUDA if available, else the CPU); Chatterbox uses the configured one
        if device.type == "cuda":
            if not torch.cuda.is_available():
                sys.exit("ERROR: CUDA was requested but is not available. Check `nvidia-smi` and the PyTorch build.")
            torch.cuda.set_device(device)  # "cuda" in CosyVoice's code then means this GPU
            print(f"  device      : {device} = {torch.cuda.get_device_name(device)}")
        elif device.type == "mps":
            if family == "cosyvoice":
                sys.exit(f"ERROR: CosyVoice's code runs on CUDA or the CPU only: set mac_device: cpu for {key}.")
            if not torch.backends.mps.is_available():
                sys.exit("ERROR: MPS was requested but is not available. Set mac_device: cpu.")
            print("  device      : Apple GPU via Metal (MPS)")
        else:
            print(f"  device      : CPU ({torch.get_num_threads()} threads)")

        # Load the model
        t0 = time.perf_counter()
        if family == "cosyvoice":
            import cosyvoice_tts as tts
            model = tts.load(model_dir, code_dir, ROOT / cfg["wetext"]["folder"])
            sample_rate = model.sample_rate
            device = model.model.device  # the device CosyVoice chose
            params = {k: m[k] for k in ("speed", "text_frontend", "prompt_prefix")}
            code = f"CosyVoice@{cfg['cosyvoice_code']['commit'][:12]}"
        else:
            import chatterbox_tts as tts
            model = tts.load(model_dir, device)
            sample_rate = model.sr
            params = {k: m[k] for k in ("exaggeration", "cfg_weight", "temperature", "repetition_penalty", "min_p",
                                        "top_p")}
            code = f"chatterbox-tts {version('chatterbox-tts')}"
        print(f"  loaded in   : {time.perf_counter() - t0:.1f} s, on {device}")
        total = 0
        for part, module in tts.components(model).items():
            count = sum(p.numel() for p in module.parameters())
            total += count
            print(f"  {part:<36}: {count / 1e6:.1f}M")
        print(f"  {'total (PyTorch parts)':<36}: {total / 1e6:.1f}M, output {sample_rate} Hz")
        print(f"  parameters  : {json.dumps(params)}")
        if family == "chatterbox":
            t0 = time.perf_counter()
            tts.prepare(model, ref_wav, m["exaggeration"])  # speaker conditioning, once for all clips
            print(f"  conditioning: {time.perf_counter() - t0:.1f} s (speaker embedding and prompt tokens)")

        # One clip at a time
        out_dir.mkdir(parents=True, exist_ok=True)
        times, audio_s = [], []
        for i, r in enumerate(todo, 1):
            set_seed(r["seed"])
            t0 = time.perf_counter()
            with torch.inference_mode():
                if family == "cosyvoice":
                    wav = tts.synthesize(model, r["text"], m["prompt_prefix"] + transcript, ref_wav, m["speed"],
                                         m["text_frontend"])
                else:
                    wav = tts.synthesize(model, r["text"], params)
            synth = time.perf_counter() - t0
            wav = np.clip(wav, -1.0, 1.0)
            path = out_dir / f"{r['id']}.wav"
            sf.write(path, wav, sample_rate, subtype="PCM_16")
            duration = len(wav) / sample_rate
            times.append(synth)
            audio_s.append(duration)
            manifest[r["id"]] = {
                "id": r["id"], "scenario": r["scenario"], "role": r["role"], "text": r["text"], "file": path.name,
                "reference": ref_wav.name, "reference_sha256": ref_sha, "seed": r["seed"], "sample_rate": sample_rate,
                "duration_s": f"{duration:.3f}", "synth_seconds": f"{synth:.2f}", "rtf": f"{synth / duration:.3f}",
                "model": key, "hf_repo": m["hf_repo"], "hf_revision": m["revision"], "code": code,
                "torch": torch.__version__, "transformers": version("transformers"), "device": str(device),
                "params": json.dumps(params)}
            write_manifest(manifest_path, manifest)  # after every clip, so an interrupted run keeps its rows
            peak = float(np.abs(wav).max()) if len(wav) else 0.0
            print(f"  [{i}/{len(todo)}] {r['id']}: {duration:5.2f} s audio in {synth:5.1f} s "
                  f"(RTF {synth / duration:.2f}), peak {peak:.2f}  {r['text'][:60]}")

        rtf = sum(times) / sum(audio_s)
        print(f"  {key}: {len(todo)} clip(s), {sum(audio_s):.1f} s of audio in {sum(times):.1f} s "
              f"(RTF {rtf:.2f}), peak RAM of the process so far {peak_ram_gb():.1f} GB")
        summary.append((key, len(todo), sum(audio_s), sum(times), rtf, str(device)))

        # Free the model before the next one
        del model
        free(device)

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    step("Step 6/6 | Summary")
    for key, count, audio, synth, rtf, dev in summary:
        print(f"  {key:<12} {count:>4} clip(s)  {audio:7.1f} s audio  {synth:7.1f} s  RTF {rtf:.2f}  on {dev}")
    if not summary:
        print("  nothing to make: every clip exists (set generate.overwrite: true to make them again)")
    print(f"\nDone. Clips and manifest.csv in {(ROOT / g['out'] / speaker).relative_to(ROOT)}/<model>/")


if __name__ == "__main__":
    main()
