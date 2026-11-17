"""Download the pinned, text-output EmotionThinker checkpoint outside the models submodule."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = "ddwang2000/EmotionThinker"
REVISION = "f80d3b343ab1eaf903ffc03aa775cfef4bffa850"
ROOT = Path(__file__).resolve().parents[2]


def main():
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destination", type=Path, default=ROOT / "local_models/emotionthinker")
    args = ap.parse_args()
    target = args.destination.resolve()
    snapshot_download(REPO, revision=REVISION, local_dir=target, max_workers=4,
                      ignore_patterns=["spk_dict.pt", "*.png", "*.wav", ".gitattributes"])
    index = json.loads((target / "model.safetensors.index.json").read_text())
    weight_files = sorted(set(index["weight_map"].values()))
    actual_size = sum((target / name).stat().st_size for name in weight_files)
    # Index total_size excludes the small safetensors headers.
    if actual_size < index["metadata"]["total_size"]:
        raise RuntimeError("Downloaded shards are smaller than the tensor payload")
    files = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(target.iterdir())
             if p.is_file() and p.name != "download_manifest.json"]
    metadata = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.json")
                if p.name != "download_manifest.json"}
    (target / "download_manifest.json").write_text(json.dumps({"repo": REPO, "revision": REVISION,
        "files": files, "metadata_sha256": metadata, "tensor_payload_bytes": index["metadata"]["total_size"],
        "shard_bytes": actual_size}, indent=2))
    print(f"Ready: {target}; {actual_size / 1e9:.3f} GB of weight shards")


if __name__ == "__main__":
    main()
