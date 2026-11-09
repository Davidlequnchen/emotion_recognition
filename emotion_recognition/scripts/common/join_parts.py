"""Join legacy Git file parts and verify SHA-256 (primarily the nEMO dataset Parquet files).

Usage (from the emotion_recognition/ folder), after `git pull`:
    python scripts/common/join_parts.py
Older main-repository commits stored nEMO's Parquet files in 90 MiB parts. For example:
    data/nemo/data/train-00000-of-00002.parquet.part000, .part001, ...
    data/nemo/data/train-00000-of-00002.parquet.sha256
This script finds every such set under data/ and models/, excluding hidden download caches, writes the whole file
next to its parts and checks it.
Files that are already there with the right checksum are skipped, so it can simply be re-run.
Searching models/ remains supported for older checkouts. Current datasets are restored by setup_datasets.py
and models by setup_models.py from pinned Releases; neither requires this join step. Hidden Release download
caches are excluded from discovery. Do not commit new datasets or weights into main-repository history.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
FOLDERS = [ROOT / "data", ROOT / "models"]  # where split files are searched


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately
    firsts = sorted(p for folder in FOLDERS if folder.is_dir() for p in folder.rglob("*.part000")
                    if not any(part.startswith(".") for part in p.relative_to(folder).parts))
    if not firsts:
        print("No legacy Git parts found. Current dataset/model Releases are restored by setup_datasets.py "
              "and setup_models.py; no legacy join is needed.")
        return
    print(f"Found {len(firsts)} split file(s) under {' and '.join(f.name + '/' for f in FOLDERS)}")

    failed = 0
    for first in firsts:
        target = first.with_suffix("")  # x.parquet.part000 -> x.parquet
        name = target.relative_to(ROOT)
        parts = sorted(target.parent.glob(target.name + ".part[0-9][0-9][0-9]"))
        expected = (target.parent / f"{target.name}.sha256").read_text().split()[0]

        # 1. Skip files that were joined before
        if target.exists() and sha256(target) == expected:
            print(f"  {name}: already joined and checked")
            continue
        # 2. All parts must be there: .part000, .part001, ... without gaps
        if [p.suffix for p in parts] != [f".part{i:03d}" for i in range(len(parts))]:
            print(f"  ERROR {name}: parts are missing (found {', '.join(p.suffix[1:] for p in parts)})")
            failed += 1
            continue
        # 3. Join them into a temporary file, check the SHA-256, then give it its real name
        print(f"  {name}: joining {len(parts)} parts ...")
        joining = target.parent / f"{target.name}.joining"
        with open(joining, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        if sha256(joining) != expected:
            joining.unlink()
            print(f"  ERROR {name}: SHA-256 does not match; run git pull again, then this script")
            failed += 1
            continue
        joining.replace(target)
        print(f"  {name}: OK, {target.stat().st_size / 1e6:,.0f} MB, SHA-256 checked")

    if failed:
        sys.exit(f"ERROR: {failed} file(s) could not be joined")
    print("Done")


if __name__ == "__main__":
    main()
