"""Download the AntiDeepfake checkpoint named in config/antideepfake.yaml (xlsr_1b.ckpt, 3.86 GB) into
models/antideepfake/, from Zenodo.

Usage (from the antideepfake/ folder):
    python scripts/antideepfake/download_model.py
Settings are in the `model` section of config/antideepfake.yaml: `zenodo_record`, `checkpoint` and `dir`. The download uses Zenodo
only, never Hugging Face or ModelScope, so it also works on a machine that cannot reach them.
The checkpoint is streamed to disk and checked against the size and MD5 that Zenodo publishes.
A file that is already present and correct is skipped; an interrupted download resumes where it stopped when
the script is run again.
"""
import hashlib
import http.client
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (antideepfake/); paths in the config are relative to it
CONFIG = ROOT / "config" / "antideepfake.yaml"
# Two small files of the same Zenodo record, kept next to the checkpoint: its licence and its readme (2 kB)
ZENODO_EXTRAS = ["LICENSE-CHECKPOINT", "README.txt"]


def list_files(target):
    print("Done. Files:")
    for f in sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts):
        print(f"  {str(f.relative_to(target)):<28} {f.stat().st_size / 1e6:>10.2f} MB")


def digest(path, algo):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, algo).hexdigest()


def fetch(url, dest, size, checksum, algo):
    """Stream one file to dest via dest.part, then check its size and checksum. Zenodo can close a long download
    early (after about 1 GB): the download then resumes where it stopped, up to 20 times."""
    if dest.exists() and dest.stat().st_size == size and digest(dest, algo) == checksum:
        print(f"  {dest.name}: already present, {algo.upper()} verified")
        return
    part = dest.with_name(dest.name + ".part")
    for _ in range(20):
        done = part.stat().st_size if part.exists() else 0
        if done >= size:
            break
        if done:
            print(f"  {dest.name}: resuming at {done / 1e9:.2f} of {size / 1e9:.2f} GB")
        request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "ab") as fh:
                if done and response.status != 206:  # the server ignored the range and sends the whole file
                    fh.truncate(0)
                    done = 0
                t0, start = time.time(), done
                checked, at_check = t0, done  # throughput check once a minute
                while chunk := response.read(1 << 16):  # 64 kB at a time: the file never has to fit in memory
                    fh.write(chunk)
                    done += len(chunk)
                    if size > 1e8 and done * 20 // size > (done - len(chunk)) * 20 // size:  # progress every 5%
                        rate = (done - start) / max(time.time() - t0, 1e-6) / 1e6
                        print(f"  downloaded {done / 1e9:.2f} of {size / 1e9:.2f} GB ({done / size:.0%}), "
                              f"{rate:.1f} MB/s")
                    if time.time() - checked >= 60:
                        if done - at_check < 1e6:  # a connection can stall without closing: open a new one
                            print("  under 1 MB in the last minute: reconnecting")
                            break
                        checked, at_check = time.time(), done
        except (OSError, http.client.HTTPException) as e:  # network error, HTTP error or dropped connection
            print(f"  connection lost: {e}")
            time.sleep(5)
    if not part.exists() or part.stat().st_size < size:
        sys.exit(f"ERROR: {dest.name} is incomplete. Run the same command again: it resumes where it stopped.")
    print(f"  {dest.name}: checking size and {algo.upper()} ...")
    if part.stat().st_size != size or digest(part, algo) != checksum:
        part.unlink()
        sys.exit(f"ERROR: {dest.name} does not match the size and {algo.upper()} published by Zenodo, so it was "
                 "deleted.\n       Run the same command again.")
    part.replace(dest)
    print(f"  {dest.name}: {size:,} bytes, {algo.upper()} {checksum} verified")


def main():
    """The AntiDeepfake checkpoint of config/antideepfake.yaml (xlsr_1b.ckpt) and two small files of its Zenodo record."""
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately, also when redirected to a log
    m = yaml.safe_load(CONFIG.read_text())["model"]
    target, record = ROOT / m["dir"], m["zenodo_record"]
    print(f"Downloading AntiDeepfake '{m['checkpoint']}' from Zenodo record {record} -> {target}")
    # The record lists every file with its size, checksum ("md5:<hex>") and download link
    for attempt in range(1, 6):  # Zenodo sometimes answers 502 or 504 when it is busy: try again
        try:
            with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as response:
                meta = json.load(response)
            break
        except (OSError, http.client.HTTPException) as e:
            if attempt == 5:
                sys.exit(f"ERROR: Zenodo does not answer ({e}). Run the same command again later.")
            print(f"  Zenodo: {e}; trying again in {15 * attempt} s")
            time.sleep(15 * attempt)
    info = meta["metadata"]
    print(f"  record      : {info['title']} (version {info.get('version')}, {info['publication_date']})")
    print(f"  licence     : {info['license']['id']}   https://doi.org/{meta['doi']}")
    files = {f["key"]: f for f in meta["files"]}
    target.mkdir(parents=True, exist_ok=True)
    for name in [m["checkpoint"]] + ZENODO_EXTRAS:
        if name not in files:
            sys.exit(f"ERROR: Zenodo record {record} has no file '{name}'. Files: {', '.join(sorted(files))}")
        algo, checksum = files[name]["checksum"].split(":")
        fetch(files[name]["links"]["self"], target / name, files[name]["size"], checksum, algo)
    list_files(target)


if __name__ == "__main__":
    main()
