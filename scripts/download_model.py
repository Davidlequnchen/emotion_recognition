"""Download a model from the registry in config/config.yaml into models/<name>/.

Usage:
    python scripts/download_model.py              # active model, from HuggingFace
    python scripts/download_model.py --hub ms     # from ModelScope (faster in mainland China)
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=ROOT / "config" / "config.yaml")
parser.add_argument("--model", help="model key in the config (default: the active `model`)")
parser.add_argument("--hub", choices=["hf", "ms"], default="hf", help="hf = HuggingFace, ms = ModelScope")
args = parser.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text())
name = args.model or cfg["model"]
entry = cfg["models"][name]
target = ROOT / entry["dir"]
print(f"Downloading '{name}' from {'HuggingFace' if args.hub == 'hf' else 'ModelScope'} -> {target}")

if args.hub == "hf":
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=entry["hf_repo"], local_dir=target)
else:
    from modelscope import snapshot_download
    snapshot_download(entry["ms_repo"], local_dir=str(target))

print("Done. Files:")
for f in sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts):
    print(f"  {str(f.relative_to(target)):<28} {f.stat().st_size / 1e6:>10.2f} MB")
