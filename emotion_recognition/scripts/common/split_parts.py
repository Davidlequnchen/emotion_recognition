"""Cut large files into parts that git can take, for a machine that cannot download them (e.g. a server without access
to Hugging Face). GitHub refuses files over 100 MB, so model weights travel through git as 90 MiB parts.

Usage (from the emotion_recognition/ folder):
    python scripts/common/split_parts.py
The settings are at the top of this file: FILES lists the files to cut, PART_MIB the size of a part. Next to each file
it writes:
    <file>.part000, <file>.part001, ...   the parts, in order (the last one is smaller)
    <file>.sha256                         SHA-256 of the whole file ("<hex>  <name>", as sha256sum writes it)
When to run it: after downloading or changing one of these files, before committing its parts. A file whose parts are
already there and match its SHA-256 is skipped, the old parts of a changed file are replaced, and a file that does
not exist yet is skipped with a message, so the script can simply be re-run.

Then commit the parts and the .sha256 files, never the whole file. git ignores models/ and data/, so add them with
`git add -f`, and push in batches of at most about 1.5 GB (GitHub refuses a push over 2 GB). On the other machine,
`git pull` and then `python scripts/common/join_parts.py` puts each file back together and checks its SHA-256
(see models/README.md).
"""
import hashlib
import sys
from pathlib import Path

# ======================================================================
# Settings (paths are relative to the emotion_recognition/ folder)
# ======================================================================
FILES = [  # the files to cut, each into <file>.part000, .part001, ... and <file>.sha256
    "models/meralion_ser_v1/model.safetensors",
    "models/melati_emotion/model.safetensors",
    "models/vox_profile_whisper/model_slim.safetensors",
    "models/vox_profile_wavlm/model.safetensors",
]
PART_MIB = 90  # size of each part, in MiB (GitHub refuses files over 100 MB)

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def sha256(paths):
    """SHA-256 of the files in `paths`, read one after the other as if they were one file."""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 24):
                digest.update(chunk)
    return digest.hexdigest()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately

    # ------------------------------------------------------------------
    # Step 1: The settings at the top of this file
    # ------------------------------------------------------------------
    step("Step 1/3 | Settings")
    size = PART_MIB * 2**20  # bytes per part
    print(f"  settings    : FILES, PART_MIB at the top of {Path(__file__).resolve().relative_to(ROOT)}")
    print(f"  part size   : {PART_MIB} MiB ({size / 1e6:.1f} MB)")
    print(f"  files       : {len(FILES)}")

    # ------------------------------------------------------------------
    # Step 2: Cut each file into parts, unless its parts are already there and match its SHA-256
    # ------------------------------------------------------------------
    step("Step 2/3 | Split files")
    done, missing = [], []
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            print(f"  {name}: not there yet, skipped (download it first)")
            missing.append(name)
            continue
        old = sorted(path.parent.glob(path.name + ".part[0-9][0-9][0-9]"))
        total = path.stat().st_size
        n = -(-total // size)  # number of parts, rounded up
        names = [path.with_name(f"{path.name}.part{i:03d}") for i in range(n)]
        sidecar = path.with_name(path.name + ".sha256")
        print(f"  {name}: {total / 1e6:,.1f} MB, computing its SHA-256 ...")
        digest = sha256([path])
        # 1. Skip a file whose parts are all there with the right sizes, whose SHA-256 is recorded, and whose parts
        #    together give that SHA-256
        if (old == names and all(p.stat().st_size == min(size, total - i * size) for i, p in enumerate(old))
                and sidecar.exists() and sidecar.read_text().split()[0] == digest and sha256(old) == digest):
            print(f"    already split: {n} parts, SHA-256 checked")
            done.append((name, n, total))
            continue
        # 2. Otherwise remove old parts (from an earlier version of the file, or an interrupted run) and cut it again
        for p in old:
            p.unlink()
        if old:
            print(f"    removed {len(old)} old part(s)")
        with open(path, "rb") as fh:
            for part in names:
                part.write_bytes(fh.read(size))
        # 3. The SHA-256 last: written only once every part is there
        sidecar.write_text(f"{digest}  {path.name}\n")
        print(f"    written: {n} parts ({names[0].suffix[1:]} to {names[-1].suffix[1:]}) and {sidecar.name}")
        done.append((name, n, total))

    # ------------------------------------------------------------------
    # Step 3: Summary, and what to commit
    # ------------------------------------------------------------------
    step("Step 3/3 | Summary")
    for name, n, nbytes in done:
        print(f"  {name:<52} {n:>3} parts {nbytes / 1e9:>6.2f} GB")
    print(f"  {'total':<52} {sum(n for _, n, _ in done):>3} parts "
          f"{sum(nbytes for _, _, nbytes in done) / 1e9:>6.2f} GB")
    if missing:
        print(f"  not there yet: {', '.join(missing)}")
    print("\nCommit the parts and the .sha256 files, never the whole files (git ignores models/ and data/: use")
    print(f"`git add -f`). GitHub limits: each file under 100 MB (a part is {size / 1e6:.1f} MB), each push under 2 GB,")
    print(f"so push in batches of at most about 1.5 GB ({int(1.5e9 // size)} parts). On the other machine: git pull, then")
    print("python scripts/common/join_parts.py.")


if __name__ == "__main__":
    main()
