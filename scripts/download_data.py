"""Download the speech emotion datasets of this project into data/.

Usage:
    python scripts/download_data.py
Settings are in the `download` section of config/config.yaml:
    dataset: crema_d   with  crema_d: samples   -> the 6 CREMA-D test clips in data/samples/ (default)
    dataset: crema_d   with  crema_d: full      -> all 7,442 CREMA-D clips and label files in data/crema_d/
    dataset: nemo                               -> another dataset in data/nemo/ (list below)
    dataset: all                                -> every dataset (CREMA-D in full)

Datasets: like CREMA-D, each records the same sentences in different emotions (see docs/datasets_survey.md).
    name        language   clips    download   format              licence            host
    crema_d     English    7,442    0.61 GB    WAV                 ODbL               GitHub (Git LFS), pinned commit
    mcae_spps   Mandarin   6,675    0.66 GB    zip -> WAV          CC BY-NC-ND 4.0    OSF 9jyzc
    thai_ser    Thai       27,854   12.7 GB    Parquet (audio in)  CC BY-SA 4.0       Hugging Face airesearch/thai-ser
    nemo        Polish     4,481    0.53 GB    Parquet (audio in)  CC BY-NC-SA 4.0    Hugging Face amu-cai/nEMO
    mess        English    1,800    0.19 GB    zip -> WAV          CC BY 4.0          Zenodo 3813437
    esd         Eng + Man  35,000   2.45 GB    zip -> WAV          research only (*)  Google Drive (HLTSingapore)
(*) ESD: before using it, complete its licence form and email it to the authors; see
    https://github.com/HLTSingapore/Emotional-Speech-Data
Zips are unpacked into data/<name>/audio/ and kept, so these datasets take about twice their download size.
Several licences forbid commercial use or sharing modified copies: use the data for research only and
never commit it (data/ is git-ignored, except the CREMA-D test clips).

Every file is checked against the checksum its host publishes, and files that are already correct are
skipped, so an interrupted download can simply be re-run.
"""
import hashlib
import http.client
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config.yaml"
DATA = ROOT / "data"

# CREMA-D: hosted on GitHub with its audio in Git LFS (served from media.githubusercontent.com)
REPO = "https://github.com/CheyneyComputerScience/CREMA-D"
COMMIT = "1658cd342dff90010aa843eaeebd53610a08b1dc"  # pinned CREMA-D version (2025-03-21)
MEDIA = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{COMMIT}/"
SAMPLES = {  # test clips: actor 1001 saying "Don't forget a jacket" with each emotion -> SHA-256
    "1001_DFA_ANG_XX.wav": "c6f94b2592487c7556bd86df7df6e8f4420d1b9c11414aeec9d8e5e640b51495",
    "1001_DFA_DIS_XX.wav": "c992e624c48a6119b5635ef7b606aea28d45d02c397b9bd85f7e5a2c2656ac65",
    "1001_DFA_FEA_XX.wav": "cc84aa476b106f71364c079ac3cf08448d2bc0b5ac79deb7ec10cbe69c19d74c",
    "1001_DFA_HAP_XX.wav": "91d63d7a792a37b2ff8ac6267d3e5364bc1b10d5b0e0be288c0af0534d4fdae5",
    "1001_DFA_NEU_XX.wav": "95e32bd83c449d26afb653c5451fa0389a83b1245c24385778cd02a18cb50cb6",
    "1001_DFA_SAD_XX.wav": "c75f74e03ae5783830733519c806c85996ca0bd833e17b535f8830166bc002e9",
}
# Label and metadata files saved with the full dataset (same paths as in the CREMA-D repository)
METADATA = ["README.md", "LICENSE.txt", "VideoDemographics.csv", "SentenceFilenames.csv",
            "processedResults/summaryTable.csv", "processedResults/tabulatedVotes.csv"]

# The other datasets: where each one is hosted and its identifier there
DATASETS = {
    # MCAE-SPPS keeps every emotion both as a folder of single files and as a zip; download only the zips
    "mcae_spps": {"host": "osf", "id": "9jyzc", "skip": r"^(Anger|Disgust|Fear|Happiness|Neutral|Sadness|Surprise)/"},
    "thai_ser": {"host": "huggingface", "id": "airesearch/thai-ser"},
    "nemo": {"host": "huggingface", "id": "amu-cai/nEMO"},
    "mess": {"host": "zenodo", "id": "3813437"},
    # ESD is one public Google Drive file. Google Drive publishes no checksum: the zip's own CRCs are checked,
    # and its SHA-256 (taken at the first download, 2026-10-26) catches any later change of the file.
    "esd": {"host": "gdrive", "id": "1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v", "file": "Emotional Speech Dataset (ESD).zip",
            "sha256": "36e076368d132b46cd97a040430c34f597a3ce65db7d8d4b9b695bcb62cf5122"},
}


def digest(path, algo):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, algo).hexdigest()


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(url, dest, checksum, algo):
    """Download one file, check its checksum and save it, with up to 3 attempts. Returns True on success."""
    for attempt in range(3):
        if attempt:
            time.sleep(2 ** attempt)  # wait 2 s, then 4 s, before retrying
        try:
            data = urllib.request.urlopen(url, timeout=300).read()
            if hashlib.new(algo, data).hexdigest() == checksum:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return True
        except (OSError, http.client.HTTPException):  # network error, HTTP error or truncated response
            pass
    return False


def download_all(jobs, workers):
    """jobs: (URL, destination, checksum, algorithm). Skip files that are already correct, fetch the rest in parallel."""
    todo = [job for job in jobs if not (job[1].exists() and digest(job[1], job[3]) == job[2])]
    print(f"  already present and verified: {len(jobs) - len(todo)} of {len(jobs)} files")
    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, *job): job[1].name for job in todo}
        for i, future in enumerate(as_completed(futures), 1):
            if not future.result():
                failed.append(futures[future])
            if i % 500 == 0 or i == len(todo):
                print(f"  downloaded and verified: {i - len(failed)} of {len(todo)} files")
    if failed:
        sys.exit(f"ERROR: {len(failed)} file(s) failed (e.g. {', '.join(failed[:3])}). "
                 "Run the same command again to retry them.")


def unzip(path, out):
    """Extract a zip into out/, skipping macOS metadata (__MACOSX, ._*, .DS_Store) and files that are already there."""
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and "__MACOSX" not in m.filename
                   and not Path(m.filename).name.startswith(".")]
        for m in members:
            if not ((out / m.filename).exists() and (out / m.filename).stat().st_size == m.file_size):
                archive.extract(m, out)
    print(f"  unzipped {len(members):,} files from {path.name}")


def crema_d_files(out):
    """Read the CREMA-D repository archive (~25 MB): save the metadata files and return
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


def osf_files(node):
    """Every file of an OSF project as (path, checksum, algorithm, download URL)."""
    pending = [(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/?page%5Bsize%5D=100", "")]
    while pending:
        url, prefix = pending.pop()
        while url:
            page = get_json(url)
            for item in page["data"]:
                a = item["attributes"]
                if a["kind"] == "folder":
                    href = item["relationships"]["files"]["links"]["related"]["href"]
                    pending.append((href + ("&" if "?" in href else "?") + "page%5Bsize%5D=100", prefix + a["name"] + "/"))
                else:
                    yield prefix + a["name"], a["extra"]["hashes"]["sha256"], "sha256", item["links"]["download"]
            url = page["links"].get("next")


def zenodo_files(record):
    """Every file of a Zenodo record as (path, checksum, algorithm, download URL)."""
    for f in get_json(f"https://zenodo.org/api/records/{record}")["files"]:
        algo, checksum = f["checksum"].split(":")
        yield f["key"], checksum, algo, f["links"]["self"]


def download_huggingface(repo, out):
    """Download a Hugging Face dataset repository, then check every large file against its SHA-256 on the Hub."""
    from huggingface_hub import HfApi, snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out)
    files = HfApi().dataset_info(repo, files_metadata=True).siblings
    bad = [f.rfilename for f in files if f.lfs and digest(out / f.rfilename, "sha256") != f.lfs.sha256]
    if bad:
        sys.exit(f"ERROR: checksum mismatch for {bad[:3]}. Delete them and run the same command again.")
    print(f"  verified {sum(1 for f in files if f.lfs)} large files against their SHA-256 on the Hub")


def download_crema_d(subset, workers):
    if subset == "samples":
        out = DATA / "samples"
        print(f"Downloading {len(SAMPLES)} CREMA-D test clips -> {out}")
        jobs = [(MEDIA + "AudioWAV/" + name, out / name, sha, "sha256") for name, sha in SAMPLES.items()]
    else:
        out = DATA / "crema_d"
        print(f"Downloading the full CREMA-D audio -> {out}")
        print("  reading the file list from the repository archive (~25 MB) ...")
        jobs = [(MEDIA + rel, out / rel, sha, "sha256") for rel, sha in crema_d_files(out).items()]
        print(f"  saved metadata: {', '.join(METADATA)}")
    download_all(jobs, workers)
    print(f"Done: {len(jobs)} files, {sum(job[1].stat().st_size for job in jobs) / 1e6:.1f} MB in {out}")


def download_gdrive(file_id, dest, checksum):
    """Download one public Google Drive file (streamed to disk), unless it is already there and correct."""
    if dest.exists() and (checksum is None or digest(dest, "sha256") == checksum):
        print(f"  already present: {dest.name}")
        return
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response, open(part, "wb") as fh:
        total, done, shown = int(response.headers.get("Content-Length", 0)), 0, 0
        while chunk := response.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total and done / total >= shown + 0.1:  # progress every 10%
                shown = round(done / total, 1)
                print(f"  downloaded {done / 1e9:.2f} of {total / 1e9:.2f} GB")
    with zipfile.ZipFile(part) as archive:  # every member's CRC must match
        if archive.testzip() is not None:
            sys.exit(f"ERROR: {dest.name} is damaged. Run the same command again.")
    sha = digest(part, "sha256")
    if checksum is not None and sha != checksum:
        sys.exit(f"ERROR: {dest.name} differs from the file pinned in this script (SHA-256 {sha}).")
    part.replace(dest)
    print(f"  verified the zip's CRCs; SHA-256 {sha}")


def download_dataset(name, workers):
    entry = DATASETS[name]
    host, ident = entry["host"], entry["id"]
    out = DATA / name
    print(f"Downloading {name} from {host} ({ident}) -> {out}")
    if host == "huggingface":
        download_huggingface(ident, out)
    elif host == "gdrive":
        download_gdrive(ident, out / entry["file"], entry["sha256"])
        unzip(out / entry["file"], out / "audio")
    else:
        files = osf_files(ident) if host == "osf" else zenodo_files(ident)
        skip = entry.get("skip")
        download_all([(url, out / path, checksum, algo) for path, checksum, algo, url in files
                      if not (skip and re.match(skip, path))], workers)
        for archive in sorted(out.rglob("*.zip")):  # zips are only packaging: unpack the audio into out/audio/
            unzip(archive, out / "audio")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and ".cache" not in f.parts)
    print(f"Done: {size / 1e9:.2f} GB in {out}")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately, also when redirected to a log
    settings = yaml.safe_load(CONFIG.read_text())["download"]
    dataset, workers = settings["dataset"], settings["workers"]
    if dataset not in ["crema_d", *DATASETS, "all"]:
        sys.exit(f"ERROR: unknown download.dataset '{dataset}' in config/config.yaml; "
                 f"use one of: crema_d, {', '.join(DATASETS)}, all")

    for name in (["crema_d", *DATASETS] if dataset == "all" else [dataset]):
        if name == "crema_d":
            download_crema_d("full" if dataset == "all" else settings["crema_d"], workers)
        else:
            download_dataset(name, workers)


if __name__ == "__main__":
    main()
