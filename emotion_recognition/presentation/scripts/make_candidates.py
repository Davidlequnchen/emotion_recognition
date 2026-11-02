"""Candidate clips to replace the 'Ambiguous ground truth' example (MCAE-SPPS 12007): writes
assets/audio_candidates/*.mp3 and assets/audio_candidates/candidates.json. The deck does not use these files.

Each candidate has a doubtful ground truth (the listeners split, the acted label gets a weak plurality or none)
and the two top models disagree with the acted label or with each other. The MP3s are made exactly as in
make_audio.py (trimmed, single clip at about -18 dBFS RMS, peaks at most -1 dBFS, 16 kHz mono 64 kbps).
Reads the same inputs as make_audio.py. Needs ffmpeg and ffprobe on PATH.
Run (from the repository root or anywhere):  python emotion_recognition/presentation/scripts/make_candidates.py
"""
import json
import shutil
import tempfile

from common import ASSETS, ER, ClipTables, active_audio, encode_mp3, group_gains

# ---------------------------------------------------------------- settings
OUT = ASSETS / "audio_candidates"
# (dataset, stem, why it is interesting)
CANDIDATES = [
    ("crema_d", "1022_IEO_FEA_HI",
     "Acted fearful (high intensity), but the 10 listeners split 4 fearful vs 4 happy (2 neutral). The two top models "
     "split the same way: emotion2vec+ says happy at 1.00, MERaLiON fearful at 0.85, although both were trained on "
     "CREMA-D's acted labels."),
    ("crema_d", "1006_IEO_ANG_MD",
     "Acted angry (medium intensity), but the 9 listeners split three ways (angry 3, happy 3, neutral 3): there is no "
     "majority at all. Both top models say happy (0.92 and 0.86): scored as wrong, yet as many listeners heard happy "
     "as angry."),
    ("mcae_spps", "54003",
     "Acted fear; 20 of 40 listeners chose fear and 17 chose sad, so the acted label wins by only 3 votes. Both top "
     "models say sad (emotion2vec+ 0.99, MERaLiON 0.78), on a dataset neither was trained on."),
    ("mcae_spps", "66127",
     "Acted disgust; 19 of 40 listeners chose disgust and 16 neutral. The models split like the listeners and are "
     "unsure: emotion2vec+ says neutral at 0.38, MERaLiON disgusted at 0.43."),
]

print(f"Candidates: {len(CANDIDATES)} | output: {OUT.relative_to(ER)}")
assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe must be on PATH"
OUT.mkdir(parents=True, exist_ok=True)

print("Step 1: load predictions, listener votes and speaker data")
T = ClipTables()

print("Step 2: trim, level (each clip alone) and encode MP3s")
entries = []
tmp_dir = tempfile.mkdtemp()
for ds, stem, why in CANDIDATES:
    y, sr, rms, peak = active_audio(ER / T.row(ds, stem)["audio_path"])
    gain = group_gains([(rms, peak)])[0]
    name = f"{ds}__{stem}.mp3"
    dur = encode_mp3(y, sr, gain, OUT / name, tmp_dir)
    e = T.entry(ds, stem, f"audio_candidates/{name}")
    e.update({"why_interesting": why, "duration_s": round(dur, 2)})
    entries.append(e)
    p = e["predictions"]
    print(f"  {name:35s} {dur:4.2f}s gain {gain:+5.1f} dB  acted {e['acted_label']:9s} votes {e['listener_votes']}  "
          f"e2v {p['emotion2vec+ large']}  MERaLiON {p['MERaLiON-SER-v1']}")
shutil.rmtree(tmp_dir)

print("Step 3: write candidates.json")
(OUT / "candidates.json").write_text(json.dumps({"candidates": entries}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Done: {len(entries)} MP3s and candidates.json in {OUT.relative_to(ER)}")
