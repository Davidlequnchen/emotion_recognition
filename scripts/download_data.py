"""Download CREMA-D audio: the six test clips (default) or the whole dataset.

CREMA-D (https://github.com/CheyneyComputerScience/CREMA-D) is an open emotional speech dataset
(Open Database License): 7,442 clips from 91 actors, six emotions. Every file is checked against
the SHA-256 published in the CREMA-D repository, so every machine gets exactly the same data.
An interrupted download can simply be re-run: files that are already correct are skipped.

Usage:
    python scripts/download_data.py                 # 6 test clips -> data/samples/ (used by inference.py)
    python scripts/download_data.py --subset full   # all 7,442 WAV files + label CSVs -> data/crema_d/ (606 MB)
"""
import argparse
import hashlib
import http.client
import io
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/CheyneyComputerScience/CREMA-D"
COMMIT = "1658cd342dff90010aa843eaeebd53610a08b1dc"  # pinned CREMA-D version (2025-03-21)
# CREMA-D stores its audio with Git LFS, so the files are served from media.githubusercontent.com.
MEDIA = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{COMMIT}/"
SAMPLES = {  # test clips: actor 1001 saying "Don't forget a jacket" with each emotion -> SHA-256
    "1001_DFA_ANG_XX.wav": "c6f94b2592487c7556bd86df7df6e8f4420d1b9c11414aeec9d8e5e640b51495",
    "1001_DFA_DIS_XX.wav": "c992e624c48a6119b5635ef7b606aea28d45d02c397b9bd85f7e5a2c2656ac65",
    "1001_DFA_FEA_XX.wav": "cc84aa476b106f71364c079ac3cf08448d2bc0b5ac79deb7ec10cbe69c19d74c",
    "1001_DFA_HAP_XX.wav": "91d63d7a792a37b2ff8ac6267d3e5364bc1b10d5b0e0be288c0af0534d4fdae5",
    "1001_DFA_NEU_XX.wav": "95e32bd83c449d26afb653c5451fa0389a83b1245c24385778cd02a18cb50cb6",
    "1001_DFA_SAD_XX.wav": "c75f74e03ae5783830733519c806c85996ca0bd833e17b535f8830166bc002e9",
}
# Label and metadata files saved with the full dataset (same paths as in the CREMA-D repository).
METADATA = ["README.md", "LICENSE.txt", "VideoDemographics.csv", "SentenceFilenames.csv",
            "processedResults/summaryTable.csv", "processedResults/tabulatedVotes.csv"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def full_file_list(out):
    """Read the repository archive (~25 MB): save the metadata files, and return
    {"AudioWAV/<clip>.wav": SHA-256} taken from the Git LFS pointer of every audio file."""
    archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f"{REPO}/archive/{COMMIT}.zip", timeout=300).read()))
    files = {}
    for entry in archive.namelist():
        rel = entry.split("/", 1)[1]  # drop the top-level "CREMA-D-<commit>/" folder
        if rel in METADATA:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(archive.read(entry))
        elif rel.startswith("AudioWAV/") and rel.endswith(".wav"):
            files[rel] = re.search(r"sha256:([0-9a-f]{64})", archive.read(entry).decode()).group(1)
    return files


def fetch(url, dest, checksum):
    """Download one file, check its SHA-256 and save it, with up to 3 attempts. Returns True on success."""
    for attempt in range(3):
        if attempt:
            time.sleep(2 ** attempt)  # wait 2 s, then 4 s, before retrying
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            if sha256(data) == checksum:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return True
        except (OSError, http.client.HTTPException):  # network error, HTTP error or truncated response
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=["samples", "full"], default="samples",
                        help="samples = the 6 test clips (default), full = all 7,442 CREMA-D clips")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (default: 8)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately, also when redirected to a log

    # Build the download jobs: (URL, destination, expected SHA-256)
    if args.subset == "samples":
        out = ROOT / "data" / "samples"
        print(f"Downloading {len(SAMPLES)} CREMA-D test clips -> {out}")
        jobs = [(MEDIA + "AudioWAV/" + name, out / name, sha) for name, sha in SAMPLES.items()]
    else:
        out = ROOT / "data" / "crema_d"
        print(f"Downloading the full CREMA-D audio -> {out}")
        print("  reading the file list from the repository archive (~25 MB) ...")
        jobs = [(MEDIA + rel, out / rel, sha) for rel, sha in full_file_list(out).items()]
        print(f"  saved metadata: {', '.join(METADATA)}")

    # Skip files that are already there and correct, then download the rest in parallel.
    todo = [job for job in jobs if not (job[1].exists() and sha256(job[1].read_bytes()) == job[2])]
    print(f"  already present and verified: {len(jobs) - len(todo)} of {len(jobs)} files")
    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, *job): job[1].name for job in todo}
        for i, future in enumerate(as_completed(futures), 1):
            if not future.result():
                failed.append(futures[future])
            if i % 500 == 0 or i == len(todo):
                print(f"  downloaded and verified: {i - len(failed)} of {len(todo)} files")
    if failed:
        sys.exit(f"ERROR: {len(failed)} file(s) failed (e.g. {', '.join(failed[:3])}). "
                 "Run the same command again to retry them.")
    print(f"Done: {len(jobs)} files, {sum(job[1].stat().st_size for job in jobs) / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
