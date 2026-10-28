"""Download the active model of config/config.yaml into models/<name>/.

Usage (from the emotion_recognition/ folder):
    python scripts/emotion2vec/download_model.py
Settings are in config/config.yaml: `model` names the model to download, `download.hub` where to get it
(hf = HuggingFace, ms = ModelScope, faster in mainland China).
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "config.yaml"

cfg = yaml.safe_load(CONFIG.read_text())
name, hub = cfg["model"], cfg["download"]["hub"]
entry = cfg["models"][name]
target = ROOT / entry["dir"]
print(f"Downloading '{name}' from {'HuggingFace' if hub == 'hf' else 'ModelScope'} -> {target}")

if hub == "hf":
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=entry["hf_repo"], local_dir=target)
else:
    from modelscope import snapshot_download
    snapshot_download(entry["ms_repo"], local_dir=str(target))

print("Done. Files:")
for f in sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts):
    print(f"  {str(f.relative_to(target)):<28} {f.stat().st_size / 1e6:>10.2f} MB")
