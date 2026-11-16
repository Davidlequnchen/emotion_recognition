#!/usr/bin/env python3
"""Split long customer turns into labelling-ready segments (all settings: config/segment_turns.yaml).

Steps, call by call and turn by turn: decode -> mono 16 kHz copy for analysis -> language check (Spanish or mixed
turns go whole to non_english) -> English turns <= 10 s stay whole (final) -> forced alignment of the existing
transcript (failures go whole to alignment_issues) -> word gaps -> boundary scores -> dynamic programming within
2-15 s (no partition: whole to boundary_issues) -> cut at the quietest VAD frame of each chosen gap -> write
<group>/<call_id>/<segment_id>.wav + .txt and review files. After each call: prev/next links across all groups,
then that call's rows go to every <group>/manifest.csv and to call_timeline.csv. Runs fully offline after
install.py.
"""
from pathlib import Path
import platform
import sys

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "segment_turns.yaml"


def main():
    import soundfile
    import torch
    import torchaudio
    from importlib.metadata import version

    # 1) environment
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"torch {torch.__version__}, torchaudio {torchaudio.__version__}, silero-vad {version('silero-vad')}, "
          f"soundfile {soundfile.__version__} (libsndfile {soundfile.__libsndfile_version__})")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    print(f"Config: {CONFIG}; device {cfg['device']}")

    # 2) run the pipeline
    from segmenter.pipeline import run
    rows, failures = run(cfg, ROOT)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
