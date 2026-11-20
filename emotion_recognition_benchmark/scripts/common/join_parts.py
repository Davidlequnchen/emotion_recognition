"""Join Git file parts and verify SHA-256.

Usage (from the emotion_recognition_benchmark/ folder), after `git pull`:
    python scripts/common/join_parts.py
    python scripts/common/join_parts.py --file dependencies/INSTALLER.sh
With --file, join only the specified file using its neighboring parts and .sha256 file.
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
import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition_benchmark/)
FOLDERS = [ROOT / "data", ROOT / "models"]  # where split files are searched


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def join(target, name):
    checksum_file = target.with_name(target.name + ".sha256")
    checksum = checksum_file.read_text().split(maxsplit=1)
    if not checksum:
        raise ValueError(f"{checksum_file.name} is empty")
    expected = checksum[0]

    if target.is_file() and sha256(target) == expected:
        print(f"  {name}: already joined and checked")
        return

    parts = sorted(target.parent.glob(target.name + ".part[0-9][0-9][0-9]"))
    if not parts or [p.suffix for p in parts] != [f".part{i:03d}" for i in range(len(parts))]:
        raise ValueError("parts are missing; run git pull again, then this script")

    print(f"  {name}: joining {len(parts)} parts ...")
    with TemporaryDirectory(dir=target.parent, prefix=".join-parts-") as temp_dir:
        joining = Path(temp_dir) / target.name
        with joining.open("wb") as out:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, out)
        if sha256(joining) != expected:
            raise ValueError("SHA-256 does not match; check that all parts are present and intact")
        joining.replace(target)
    print(f"  {name}: OK, {target.stat().st_size / 1e6:,.0f} MB, SHA-256 checked")


def main():
    parser = argparse.ArgumentParser(description="Join Git file parts and verify SHA-256.")
    parser.add_argument("--file", type=Path, help="Join only this file using its neighboring parts")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately
    if args.file is not None:
        targets = [args.file]
    else:
        targets = sorted(p.with_suffix("") for folder in FOLDERS if folder.is_dir()
                         for p in folder.rglob("*.part000")
                         if not any(part.startswith(".") for part in p.relative_to(folder).parts))
    if not targets:
        print("No legacy Git parts found. Current dataset/model Releases are restored by setup_datasets.py "
              "and setup_models.py; no legacy join is needed.")
        return

    failed = 0
    for target in targets:
        name = target.relative_to(ROOT) if target.is_relative_to(ROOT) else target
        try:
            join(target, name)
        except (OSError, ValueError) as exc:
            print(f"  ERROR {name}: {exc}")
            failed += 1

    if failed:
        sys.exit(f"ERROR: {failed} file(s) could not be joined")
    print("Done")


if __name__ == "__main__":
    main()
