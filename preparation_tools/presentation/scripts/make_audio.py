"""Audio for the two listening slides: the whole input turn and every exported segment, as small MP3s.

Source: the integration run on synthetic DailyTalk calls (CC BY-SA 4.0). Each synthetic turn joins consecutive
utterances of one DailyTalk speaker with inserted pauses, so these clips are adaptations (joined and cut) of DailyTalk
audio and stay under CC BY-SA 4.0 with attribution. Only the clips shown on the slides are written.
Processing: none apart from the format: 16 kHz mono MP3 at 64 kbps (ffmpeg, metadata stripped); no trimming or
levelling, so the clips sound exactly like the pipeline's output.
Writes assets/audio/<name>.mp3 and assets/audio/manifest.json (start, end, duration and transcript of every clip).
Needs ffmpeg on PATH. Run:  python preparation_tools/presentation/scripts/make_audio.py
"""
import csv
import json
import shutil
import subprocess

from style import ASSETS, TOOLS

OUT = TOOLS / "local" / "integration" / "output"
TURNS = TOOLS / "local" / "integration" / "input" / "turns"
LISTEN_TURNS = ["d418_long_en", "d1562_long_en"]
AUDIO = ASSETS / "audio"


def to_mp3(src, dst):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
                    "-c:a", "libmp3lame", "-b:a", "64k", "-map_metadata", "-1", "-fflags", "+bitexact",
                    "-flags:a", "+bitexact", str(dst)], check=True)


def main():
    assert shutil.which("ffmpeg"), "ffmpeg must be on PATH"
    print("ffmpeg:", subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0])
    AUDIO.mkdir(parents=True, exist_ok=True)
    for old in AUDIO.glob("*.mp3"):
        old.unlink()
    rows = list(csv.DictReader(open(OUT / "final" / "manifest.csv", newline="", encoding="utf-8")))
    manifest = {}
    for turn_id in LISTEN_TURNS:
        print(f"Turn {turn_id}")
        src = next(p for p in TURNS.glob(f"{turn_id}.*") if p.suffix in (".wav", ".mp3"))
        whole = AUDIO / f"{turn_id}_whole.mp3"
        to_mp3(src, whole)
        segs = sorted((r for r in rows if r["turn_id"] == turn_id), key=lambda r: int(r["segment_index"]))
        entry = {"whole": {"file": f"assets/audio/{whole.name}", "duration_s": round(sum(float(r["duration_s"]) for r in segs), 3),
                           "source": src.name}, "segments": []}
        for r in segs:
            dst = AUDIO / f"{r['segment_id']}.mp3"
            to_mp3(OUT / r["audio_path"], dst)
            text = (OUT / r["transcript_path"]).read_text(encoding="utf-8").strip()
            entry["segments"].append({"file": f"assets/audio/{dst.name}", "segment_id": r["segment_id"],
                                      "start_s": float(r["segment_start_in_turn_s"]), "end_s": float(r["segment_end_in_turn_s"]),
                                      "duration_s": float(r["duration_s"]), "text": text})
            print(f"  {dst.name}: {float(r['segment_start_in_turn_s']):.1f}-{float(r['segment_end_in_turn_s']):.1f} s")
        manifest[turn_id] = entry
    (AUDIO / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    size = sum(p.stat().st_size for p in AUDIO.glob("*.mp3"))
    print(f"wrote {len(list(AUDIO.glob('*.mp3')))} MP3s, {size / 1e6:.2f} MB, and assets/audio/manifest.json")


if __name__ == "__main__":
    main()
