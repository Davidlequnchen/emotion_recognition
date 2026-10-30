"""
Record a 5-10 s reference clip of your voice for voice cloning, in a small window.

Usage (from the antideepfake/ folder, in the voice_clone environment):
    python scripts/voice_clone/record_reference.py
Settings are in config/record.yaml (speaker name, length, countdown, microphone, sentences to read).

In the window: read the sentence shown, press "Record", wait for the countdown, then read it aloud. The take is
saved automatically as data/voice_clone/reference/<speaker>_<YYYYmmdd_HHMMSS>.wav (mono, 16-bit PCM), with the
sentence in the text box saved next to it as the transcript (.txt). "Play" plays the last take back; record again
if it sounds wrong (every take is kept; generate.py uses the newest one).

macOS asks once for microphone access for the app that runs Python (Terminal, iTerm or VS Code): allow it in
System Settings > Privacy & Security > Microphone. Without it, the recording is silent.
"""
import platform
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (antideepfake/); paths in the config are relative to it
CONFIG = ROOT / "config" / "record.yaml"


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # ------------------------------------------------------------------
    # Step 1: Read the settings
    # ------------------------------------------------------------------
    step("Step 1/3 | Settings (config/record.yaml)")
    cfg = yaml.safe_load(CONFIG.read_text())
    out_dir = ROOT / cfg["out"]
    sr, seconds, countdown = int(cfg["sample_rate"]), float(cfg["seconds"]), int(cfg["countdown"])
    prompts = cfg["prompts"]
    print(f"  speaker     : {cfg['speaker']}")
    print(f"  take        : {seconds:g} s at {sr} Hz, mono, after a {countdown} s countdown")
    print(f"  saved to    : {out_dir}")
    print(f"  sentences   : {len(prompts)}")

    # ------------------------------------------------------------------
    # Step 2: Check the audio stack and the microphone
    # ------------------------------------------------------------------
    step("Step 2/3 | Microphone")
    print(f"  Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"  sounddevice : {sd.__version__} (PortAudio {sd.get_portaudio_version()[1]})")
    print("  input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"    [{i}] {d['name']} ({d['max_input_channels']} ch, {d['default_samplerate']:.0f} Hz)")
    device = cfg["input_device"]
    try:
        info = sd.query_devices(device, kind="input")
        sd.check_input_settings(device=device, channels=1, samplerate=sr, dtype="float32")
    except Exception as e:  # no microphone, unknown device name, or unsupported sample rate
        sys.exit(f"ERROR: the microphone cannot record mono at {sr} Hz: {e}\n"
                 "       Set `input_device` or `sample_rate` in config/record.yaml.")
    print(f"  using       : {info['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 3: The recording window
    # ------------------------------------------------------------------
    step("Step 3/3 | Recording window (close it to quit)")
    state = {"prompt": 0, "audio": None}

    root = tk.Tk()
    root.title("Record a reference clip")
    root.geometry("760x400")
    tk.Label(root, text=f"Speaker: {cfg['speaker']}   |   {seconds:g} s per take   |   read the sentence below aloud",
             font=("Helvetica", 14)).pack(pady=(14, 6))
    text = tk.Text(root, height=4, wrap="word", font=("Helvetica", 20), padx=10, pady=8)
    text.pack(fill="x", padx=16)
    status = tk.Label(root, text="Press Record, wait for the countdown, then read.", font=("Helvetica", 16), fg="#333")
    status.pack(pady=12)
    buttons = tk.Frame(root)
    buttons.pack(pady=6)

    def show_prompt(delta=0):
        state["prompt"] = (state["prompt"] + delta) % len(prompts)
        text.delete("1.0", "end")
        text.insert("1.0", prompts[state["prompt"]])

    def set_buttons(enabled):
        for b in (b_prev, b_next, b_rec, b_play):
            b.config(state="normal" if enabled else "disabled")

    def record():
        set_buttons(False)
        tick(countdown)

    def tick(n):  # countdown, then start recording
        if n > 0:
            status.config(text=f"Get ready ... {n}", fg="#b26b00")
            root.after(1000, tick, n - 1)
            return
        state["audio"] = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
        state["t0"] = time.time()
        recording()

    def recording():  # show the time left while sd.rec runs in the background
        left = seconds - (time.time() - state["t0"])
        if left > 0:
            status.config(text=f"● Recording ... {left:.1f} s left", fg="#c00000")
            root.after(100, recording)
        else:
            sd.wait()
            save()

    def save():
        audio = state["audio"][:, 0]
        peak, rms = float(np.abs(audio).max()), float(np.sqrt(np.mean(audio ** 2)))
        transcript = " ".join(text.get("1.0", "end").split())
        stem = f"{cfg['speaker']}_{time.strftime('%Y%m%d_%H%M%S')}"
        wav, txt = out_dir / f"{stem}.wav", out_dir / f"{stem}.txt"
        sf.write(wav, audio, sr, subtype="PCM_16")
        txt.write_text(transcript + "\n", encoding="utf-8")
        level = f"peak {peak:.2f}, RMS {20 * np.log10(max(rms, 1e-9)):.0f} dBFS"
        if peak < 0.05:
            msg, colour = f"Saved {wav.name}, but it is almost silent ({level}): check the microphone access", "#c00000"
        elif peak > 0.99:
            msg, colour = f"Saved {wav.name}, but it is clipped ({level}): speak softer or move back", "#b26b00"
        else:
            msg, colour = f"Saved {wav.name} ({level})", "#00802b"
        print(f"  {msg}\n    transcript: {transcript}")
        status.config(text=msg, fg=colour)
        set_buttons(True)

    def play():
        if state["audio"] is not None:
            sd.play(state["audio"][:, 0], sr)

    b_prev = tk.Button(buttons, text="◀ Previous sentence", font=("Helvetica", 14), command=lambda: show_prompt(-1))
    b_rec = tk.Button(buttons, text="● Record", font=("Helvetica", 16, "bold"), width=10, command=record)
    b_play = tk.Button(buttons, text="▶ Play", font=("Helvetica", 14), command=play)
    b_next = tk.Button(buttons, text="Next sentence ▶", font=("Helvetica", 14), command=lambda: show_prompt(1))
    for b in (b_prev, b_rec, b_play, b_next):
        b.pack(side="left", padx=6)
    tk.Label(root, text=f"Takes are saved in {cfg['out']}/ (not in git). Only record people who have agreed to it.",
             font=("Helvetica", 12), fg="#666").pack(side="bottom", pady=10)
    show_prompt()
    root.mainloop()


if __name__ == "__main__":
    main()
