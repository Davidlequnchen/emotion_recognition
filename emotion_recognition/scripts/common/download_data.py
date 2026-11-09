"""Download the speech emotion datasets of this project into data/.

Usage (from the emotion_recognition/ folder):
    python scripts/common/download_data.py
Settings are in config/download_data.yaml:
    dataset: crema_d   with  crema_d: samples   -> the 6 CREMA-D test clips in data/samples/ (default)
    dataset: crema_d   with  crema_d: full      -> all 7,442 CREMA-D clips and label files in data/crema_d/
    provider: github  and dataset: mirrors      -> the nine pinned Release mirrors through setup_datasets.py
    provider: source  and dataset: nemo          -> the original publisher's data/nemo/ (list below)
    dataset: all with --source                   -> all 13 evaluated datasets (CREMA-D full; ESD excluded)

Use --dataset NAME --source for a particular original source without editing YAML. The default CREMA-D samples
remain the six small clips tracked in the data submodule; no Release archive is downloaded for that mode.

Datasets: like CREMA-D, most record the same sentences in different emotions (MELD: utterances from a TV series;
see docs/datasets_survey.md).
    name        language   clips    download   format              licence            host
    crema_d     English    7,442    0.61 GB    WAV                 ODbL               GitHub (Git LFS), pinned commit
    mcae_spps   Mandarin   6,675    0.66 GB    zip -> WAV          CC BY-NC 4.0       OSF 9jyzc
    thai_ser    Thai       27,854   12.7 GB    Parquet (audio in)  CC BY-SA 4.0       Hugging Face airesearch/thai-ser
    nemo        Polish     4,481    0.53 GB    Parquet (audio in)  CC BY-NC-SA 4.0    Hugging Face amu-cai/nEMO
    mess        English    1,800    0.19 GB    zip -> WAV          CC BY 4.0          Zenodo 3813437
    esd         Eng + Man  35,000   2.45 GB    zip -> WAV          research only (*)  Google Drive (HLTSingapore)
    ravdess     English    1,440    0.21 GB    zip -> WAV          CC BY-NC-SA 4.0    Zenodo 1188976 (speech audio only)
    tess        English    2,800    0.28 GB    WAV                 CC BY-NC 4.0       Borealis Dataverse doi:10.5683/SP2/E8H2MF
    jl_corpus   English    2,400    1.28 GB    zip -> WAV          CC0                Kaggle tli725/jl-corpus
    mead        English    31,734   3.07 GB    tar (m4a) -> WAV    none published     Google Drive (MEAD Part0, audio.tar only)
    meld        English    2,610    10.88 GB   tar.gz (mp4) -> WAV GPL-3.0 (repo)     Hugging Face declare-lab/MELD (test split only)
    emovo       Italian    588      0.33 GB    WAV                 none published     DagsHub kingabzpro/EMOVO (unofficial mirror)
    emozionalmente Italian 6,902    0.56 GB    zip -> WAV          CC BY 4.0          Zenodo 12616095
    savee       English    480      0.11 GB    zip -> WAV          research only (**) Kaggle ejlok1/surrey-audiovisual-expressed-emotion-savee
(*) ESD: before using it, complete its licence form and email it to the authors; see
    https://github.com/HLTSingapore/Emotional-Speech-Data
(**) SAVEE: free for research after registration at http://kahlan.eps.surrey.ac.uk/savee/; its licence forbids
    redistribution, so the Kaggle copy used here is unofficial.
Zips are unpacked into data/<name>/audio/ and then deleted, so only the extracted files take disk space. The files
of each unpacked zip are listed in data/<name>/unpacked.json: a later run downloads a zip again only when one of
them is missing or has changed size.
THAI-SER: its audio (studio condenser / Zoom microphone) is extracted to data/thai_ser/audio/ and its labels to
data/thai_ser/labels.csv, then the 12.7 GB of Parquet files are deleted; a later run downloads them again only when
labels.csv or a clip is missing.
MEAD (AAC in .m4a) and MELD (video in .mp4) are converted to 16 kHz mono WAV in data/<name>/audio/ with ffmpeg,
which must be installed (brew install ffmpeg / apt install ffmpeg). MEAD deletes each actor's tar after unpacking and
the .m4a files (data/mead/m4a/) once every actor is converted; MELD deletes the 10.88 GB archive and the videos once
every test clip is converted (a later run downloads again only what has no WAV yet).
Selected research copies of nine datasets are versioned as Release archives in the data submodule's repository,
with source terms and scope recorded in data/README.md. Four external/Zenodo datasets retain their original
sources, and ESD remains local and excluded. No dataset archive is committed as a Git or Git LFS object.

Every file is checked against the checksum its host publishes, and files that are already correct are
skipped, so an interrupted download can simply be re-run. Google Drive and Kaggle publish no checksum: their
single zip is checked with its own CRCs and a SHA-256 pinned in this script. MEAD's tars (Google Drive) are only
checked by unpacking them, and EMOVO's 588 files (DagsHub) by one SHA-256 over all of them pinned in this script.
"""
import argparse
import hashlib
import importlib.util
import http.client
import io
import json
import csv
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "download_data.yaml"  # this script's settings
DATA = ROOT / "data"

_HTTPS_OPENER = None


def configure_https(ca_bundle=None):
    """Use the shared optional system trust store without changing global SSL behavior."""
    global _HTTPS_OPENER
    context = None
    for file in (DATA / "tls.py", ROOT / "models" / "tls.py"):
        if file.is_file():
            spec = importlib.util.spec_from_file_location("dataset_source_tls", file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            context = module.create_ssl_context(ca_bundle)
            break
    if context is None:
        context = ssl.create_default_context()
        if ca_bundle is not None:
            context.load_verify_locations(cafile=str(ca_bundle))
    _HTTPS_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def urlopen(*args, **kwargs):
    return (_HTTPS_OPENER.open if _HTTPS_OPENER is not None else urllib.request.urlopen)(*args, **kwargs)


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
    # RAVDESS: only the speech audio; the record also holds the song audio and the videos (25 GB)
    "ravdess": {"host": "zenodo", "id": "1188976", "keep": r"^Audio_Speech_Actors_01-24\.zip$"},
    "tess": {"host": "dataverse", "id": "doi:10.5683/SP2/E8H2MF", "server": "https://borealisdata.ca"},
    # JL-Corpus: one public Kaggle zip (no login needed); SHA-256 taken at the first download, 2026-10-27
    "jl_corpus": {"host": "kaggle", "id": "tli725/jl-corpus", "file": "jl-corpus.zip",
                  "sha256": "4f1feb9af5584a428a997da492be6fbc9185a18c2f7191d07196c2f559553292"},
    # MEAD: the official Google Drive folder "Part0" has one sub-folder per actor (48) with audio.tar and video.tar;
    # only audio.tar is downloaded (3.07 GB in all). Google Drive publishes no checksum: each tar is checked by
    # unpacking it
    "mead": {"host": "gdrive_folder", "id": "1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j", "file": "audio.tar"},
    # MELD: the authors' Hugging Face repository has one archive of the raw videos of every split (10.88 GB, SHA-256
    # from its Git LFS pointer); only the test split is kept. Its labels come from the authors' GitHub repository,
    # pinned at a commit (SHA-256 taken at the first download, 2026-11-05)
    "meld": {"host": "meld", "id": "declare-lab/MELD", "file": "MELD.Raw.tar.gz",
             "sha256": "a56b4407d574195cbce470d86f9c9d72fcfea59b0e34502ecd4babee4a5c613e",
             "labels": "https://raw.githubusercontent.com/declare-lab/MELD/e8cedf27b5d2877e198332c957127e16eb214afe/"
                       "data/MELD/test_sent_emo.csv",
             "labels_sha256": "8d37103938f7067600839fe29d5a114a6cd1bcdafb75bec101e06464c5006888"},
    # EMOVO: the official site (voice.fub.it) is gone and there is no official mirror; this is an unofficial copy on
    # DagsHub. It publishes no checksum and no file list without login: the 588 file names follow the corpus' naming
    # (<emotion>-<speaker>-<sentence>.wav), and one SHA-256 over all files (taken at the first download, 2026-11-05)
    # catches any later change
    "emovo": {"host": "dagshub", "id": "kingabzpro/EMOVO", "sha256": "14a1af4d80a46b7ea00c8946d8c1df64819429b89afe514e2bc8efd997dc4343"},
    "emozionalmente": {"host": "zenodo", "id": "12616095"},
    # SAVEE: the official copy needs registration (http://kahlan.eps.surrey.ac.uk/savee/) and its licence forbids
    # redistribution; this is an unofficial public Kaggle zip (no login needed), audio only. SHA-256 taken at the
    # first download, 2026-11-06
    "savee": {"host": "kaggle", "id": "ejlok1/surrey-audiovisual-expressed-emotion-savee", "file": "savee.zip",
              "sha256": "2110db62fdc2952db15bf324a414649b76a2a4f9f2fbb9460eae1dade7b57dd0"},
}
# Download URLs of the hosts that serve a dataset as one zip
ZIP_URLS = {"gdrive": "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t",
            "kaggle": "https://www.kaggle.com/api/v1/datasets/download/{}"}


def digest(path, algo):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, algo).hexdigest()


def get_json(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(url, dest, checksum, algo):
    """Download one file, check its checksum (unless it is None) and save it, with up to 3 attempts. Returns True on
    success."""
    for attempt in range(3):
        if attempt:
            time.sleep(2 ** attempt)  # wait 2 s, then 4 s, before retrying
        try:
            data = urlopen(url, timeout=300).read()
            if checksum is None or hashlib.new(algo, data).hexdigest() == checksum:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return True
        except (OSError, http.client.HTTPException):  # network error, HTTP error or truncated response
            pass
    return False


def download_all(jobs, workers):
    """jobs: (URL, destination, checksum, algorithm). Skip files that are already correct (or, with checksum None, there),
    fetch the rest in parallel."""
    todo = [job for job in jobs if not (job[1].exists() and (job[2] is None or digest(job[1], job[3]) == job[2]))]
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


def unpacked_zips(out):
    """The zips that an earlier run unpacked into out/audio/ and deleted (listed with their files in
    out/unpacked.json) whose files are all still there with the same sizes: these are not downloaded again."""
    record = out / "unpacked.json"
    done = json.loads(record.read_text()) if record.exists() else {}
    return {name for name, files in done.items()
            if all((out / "audio" / f).is_file() and (out / "audio" / f).stat().st_size == size
                   for f, size in files.items())}


def unzip(path, out):
    """Extract a zip into out/audio/, skipping macOS metadata (__MACOSX, ._*, .DS_Store) and files that are already
    there; then list its files in out/unpacked.json and delete the zip (only the extracted files are kept)."""
    with zipfile.ZipFile(path) as archive:  # extracting checks each member's CRC
        members = [m for m in archive.infolist() if not m.is_dir() and "__MACOSX" not in m.filename
                   and not Path(m.filename).name.startswith(".")]
        for m in members:
            dest = out / "audio" / m.filename
            if not (dest.exists() and dest.stat().st_size == m.file_size):
                archive.extract(m, out / "audio")
    record = out / "unpacked.json"  # zip (path in out/) -> {file in out/audio/: size}
    done = json.loads(record.read_text()) if record.exists() else {}
    done[path.relative_to(out).as_posix()] = {m.filename: m.file_size for m in members}
    record.write_text(json.dumps(done, indent=1))
    path.unlink()
    if path.parent != out and not any(path.parent.iterdir()):  # e.g. MCAE-SPPS's "Audio zips by different emotion/"
        path.parent.rmdir()
    print(f"  unzipped {len(members):,} files from {path.name}, then deleted the zip")


def crema_d_files(out):
    """Read the CREMA-D repository archive (~25 MB): save the metadata files and return
    {"AudioWAV/<clip>.wav": SHA-256} taken from the Git LFS pointer of every audio file."""
    archive = zipfile.ZipFile(io.BytesIO(urlopen(f"{REPO}/archive/{COMMIT}.zip", timeout=300).read()))
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


def dataverse_files(server, doi):
    """Every file of a Dataverse dataset (latest version) as (path, checksum, algorithm, download URL)."""
    version = get_json(f"{server}/api/datasets/:persistentId/?persistentId={doi}")["data"]["latestVersion"]
    for f in version["files"]:
        d = f["dataFile"]
        path = f"{f['directoryLabel']}/{d['filename']}" if f.get("directoryLabel") else d["filename"]
        yield path, d["checksum"]["value"], d["checksum"]["type"].lower(), f"{server}/api/access/datafile/{d['id']}"


def download_huggingface(repo, out):
    """Download a Hugging Face dataset repository, then check every large file against its SHA-256 on the Hub."""
    from huggingface_hub import HfApi, snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=out)
    files = HfApi().dataset_info(repo, files_metadata=True).siblings
    bad = [f.rfilename for f in files if f.lfs and digest(out / f.rfilename, "sha256") != f.lfs.sha256]
    if bad:
        sys.exit(f"ERROR: checksum mismatch for {bad[:3]}. Delete them and run the same command again.")
    print(f"  verified {sum(1 for f in files if f.lfs)} large files against their SHA-256 on the Hub")


def download_thai_ser(ident, out):
    """THAI-SER: download the Parquet files, extract the audio and labels.csv through the evaluation loader
    (scripts/common/eval_datasets.py), check that every clip is there, then delete the Parquet files (12.7 GB)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_datasets
    if (out / "labels.csv").exists() and not list((out / "data").glob("*.parquet")):
        clips = eval_datasets.thai_ser().clips
        if all(c["audio"].exists() for c in clips):
            print(f"  already extracted: {len(clips):,} clips in {out / 'audio'} and {out / 'labels.csv'}")
            return
        (out / "labels.csv").unlink()  # a clip is missing: start again from the Parquet files
    download_huggingface(ident, out)
    clips = eval_datasets.thai_ser().clips  # extracts the clips that are missing and writes labels.csv
    missing = [c["audio"] for c in clips if not c["audio"].exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} clips were not extracted, e.g. {missing[0]}; the Parquet files are kept")
    shutil.rmtree(out / "data")
    shutil.rmtree(out / ".cache", ignore_errors=True)  # the download records of huggingface_hub
    print(f"  extracted {len(clips):,} clips and labels.csv; deleted the Parquet files")


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


def download_zip(url, dest, checksum):
    """Download one zip (streamed to disk) from a host that publishes no checksum, unless it is already there and
    correct: check the CRCs of its members and, when this script pins one, its SHA-256."""
    if dest.exists() and (checksum is None or digest(dest, "sha256") == checksum):
        print(f"  already present: {dest.name}")
        return
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=300) as response, open(part, "wb") as fh:
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


def to_wav(src, dst):
    """Convert one audio or video file to a 16 kHz mono 16-bit WAV with ffmpeg (audio track only). Returns True on
    success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.stem + ".part.wav")  # renamed when complete, so a half-written WAV never counts as done
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(part)]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0 and part.exists() and part.stat().st_size > 44
    if ok:
        part.replace(dst)
    else:
        part.unlink(missing_ok=True)
    return ok


def convert_all(pairs, workers):
    """pairs: (source, WAV destination). Convert those whose WAV is not there yet, `workers` ffmpeg processes at a
    time. Returns the sources that ffmpeg could not convert."""
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg is not installed (brew install ffmpeg / apt install ffmpeg).")
    todo = [p for p in pairs if not p[1].exists()]
    print(f"  already converted: {len(pairs) - len(todo)} of {len(pairs)} files; converting {len(todo)} to 16 kHz "
          f"mono WAV with {workers} ffmpeg processes ...")
    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(to_wav, *p): p[0] for p in todo}
        for i, future in enumerate(as_completed(futures), 1):
            if not future.result():
                failed.append(futures[future])
            if i % 1000 == 0 or i == len(todo):
                print(f"  converted: {i - len(failed)} of {len(todo)} files")
    return sorted(failed)


def download_stream(url, dest, checksum=None):
    """Download one large file streamed to disk, resuming a partial download (dest.part), and check its SHA-256 when
    one is given. Returns False when the host answers with an HTML page instead of the file (Google Drive does that
    when a file's download quota is used up)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urlopen(request, timeout=300) as response:
        if response.headers.get_content_type() == "text/html":
            return False
        if done and response.status != 206:  # the host ignored the range: start again
            done = 0
        total, shown = done + int(response.headers.get("Content-Length", 0)), done
        with open(part, "ab" if done else "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total > 1e9 and done - shown >= total / 10:  # progress every 10% of a large file
                    shown = done
                    print(f"  downloaded {done / 1e9:.2f} of {total / 1e9:.2f} GB")
    if checksum is not None and digest(part, "sha256") != checksum:
        part.unlink()
        sys.exit(f"ERROR: {dest.name} does not match its SHA-256. Run the same command again.")
    part.replace(dest)
    return True


def gdrive_folder(folder):
    """The entries of a public Google Drive folder as {name: file or folder id}, read from its embedded view."""
    html = urlopen(f"https://drive.google.com/embeddedfolderview?id={folder}", timeout=60).read().decode()
    return {name: ident for ident, name in
            re.findall(r'id="entry-([\w-]+)".*?class="flip-entry-title">([^<]+)<', html, re.S)}


def download_mead(entry, workers, converters):
    """MEAD: download each actor's audio.tar from the Google Drive folder, unpack it into m4a/<actor>/ (deleting the
    tar), then convert every clip to audio/<actor>/<emotion>/level_<n>/<number>.wav."""
    out = DATA / "mead"
    actors = gdrive_folder(entry["id"])
    print(f"  {len(actors)} actor folders: {', '.join(actors)}")

    def get(actor):
        if (out / "m4a" / actor).exists() or (out / "audio" / actor).exists():  # unpacked or converted earlier
            return True
        tar, tmp = out / "tar" / f"{actor}.tar", out / "tmp" / actor
        url = f"https://drive.usercontent.google.com/download?id={gdrive_folder(actors[actor])[entry['file']]}" \
              "&export=download&confirm=t"
        for attempt in range(3):  # Google Drive answers with an HTML page when the quota is used up: wait and retry
            try:
                if tar.exists() or download_stream(url, tar):
                    break
            except (OSError, http.client.HTTPException):  # network error: the next attempt resumes the download
                pass
            time.sleep(60 * (attempt + 1))
        else:
            return False
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            with tarfile.open(tar) as archive:
                archive.extractall(tmp, filter="data")
        except tarfile.TarError:  # a damaged tar: delete it, so that the next run downloads it again
            tar.unlink()
            return False
        (out / "m4a").mkdir(exist_ok=True)
        (tmp / "audio").rename(out / "m4a" / actor)  # the tar holds audio/<emotion>/level_<n>/<number>.m4a
        tar.unlink()
        return True

    # Google Drive throttles many large downloads at once: at most 4 in parallel
    with ThreadPoolExecutor(min(workers, 4)) as pool:
        results = dict(zip(actors, pool.map(get, actors)))
    shutil.rmtree(out / "tmp", ignore_errors=True)
    if (out / "tar").exists() and not any((out / "tar").iterdir()):  # every tar unpacked and deleted
        (out / "tar").rmdir()
    missing = [a for a, ok in results.items() if not ok]
    print(f"  unpacked the audio of {len(actors) - len(missing)} of {len(actors)} actors into {out / 'm4a'}")
    if missing:
        print(f"  WARNING: Google Drive did not serve {', '.join(missing)} (probably its download quota): "
              "run the same command again later")
    m4a = sorted((out / "m4a").glob("*/*/*/*.m4a"))
    failed = convert_all([(f, out / "audio" / f.relative_to(out / "m4a").with_suffix(".wav")) for f in m4a],
                         converters)
    if failed:  # at the first download: 5 files of the original tars that hold no audio (345-349 bytes)
        print(f"  WARNING: ffmpeg could not convert {len(failed)} files (left out), e.g. {failed[0]}")
    if not missing:  # every actor converted: the .m4a files are no longer needed (2.9 GB)
        shutil.rmtree(out / "m4a", ignore_errors=True)
        print(f"  deleted {out / 'm4a'} (the WAVs in {out / 'audio'} are complete)")


def download_meld(entry, converters):
    """MELD test split: download the labels and the archive of every split, take the test videos out of it, convert
    those listed in the labels to audio/dia<dialogue>_utt<utterance>.wav, then delete the archive and the videos."""
    out = DATA / "meld"
    labels = out / "test_sent_emo.csv"
    if not (labels.exists() and digest(labels, "sha256") == entry["labels_sha256"]):
        if not fetch(entry["labels"], labels, entry["labels_sha256"], "sha256"):
            sys.exit(f"ERROR: could not download {entry['labels']}")
    with open(labels, newline="", encoding="utf-8") as fh:
        clips = [f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}" for r in csv.DictReader(fh)]
    failed_list = out / "failed.txt"  # clips that ffmpeg could not decode on an earlier run
    known_bad = set(failed_list.read_text().split()) if failed_list.exists() else set()
    todo = [c for c in clips if not (out / "audio" / f"{c}.wav").exists() and c not in known_bad]
    print(f"  {len(clips):,} test utterances in {labels.name}; WAVs still to make: {len(todo):,}")
    if todo:
        videos = out / "mp4"
        if not all((videos / f"{c}.mp4").exists() for c in todo):
            archive, inner = out / entry["file"], out / "test.tar.gz"
            if not inner.exists():
                if not archive.exists():
                    url = f"https://huggingface.co/datasets/{entry['id']}/resolve/main/{entry['file']}"
                    print(f"  downloading {entry['file']} (10.88 GB, every split) ...")
                    download_stream(url, archive, entry["sha256"])
                print(f"  verified the SHA-256 of {archive.name}; taking out the test split (2.24 GB) ...")
                with tarfile.open(archive, "r:gz") as outer:
                    with outer.extractfile("MELD.Raw/test.tar.gz") as src, open(inner, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
            print("  unpacking the test videos ...")
            wanted = {f"{c}.mp4" for c in todo}  # the archive also holds 137 final_videos_test*.mp4 not in the labels
            videos.mkdir(exist_ok=True)
            with tarfile.open(inner, "r:gz") as archive_test:
                for m in archive_test:
                    if m.isfile() and Path(m.name).name in wanted:
                        with archive_test.extractfile(m) as src, open(videos / Path(m.name).name, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
        absent = [c for c in todo if not (videos / f"{c}.mp4").exists()]
        failed = [Path(f).stem for f in convert_all([(videos / f"{c}.mp4", out / "audio" / f"{c}.wav")
                                                      for c in todo if c not in absent], converters)]
        if absent or failed:
            failed_list.write_text("\n".join(sorted(known_bad | set(absent) | set(failed))) + "\n")
            print(f"  WARNING: {len(absent)} clips have no video and ffmpeg could not decode {len(failed)}; listed in "
                  f"{failed_list.name} and left out")
    made = sum((out / "audio" / f"{c}.wav").exists() for c in clips)
    print(f"  WAVs: {made:,} of {len(clips):,} test utterances")
    if made + len(set(failed_list.read_text().split()) if failed_list.exists() else ()) >= len(clips):
        # every clip is done: the archive and the videos are no longer needed
        leftovers = [path for path in (out / entry["file"], out / "test.tar.gz", out / "mp4") if path.exists()]
        for path in leftovers:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        if leftovers:
            print("  deleted the archive and the videos (a later run downloads the archive again only if a WAV is "
                  "missing)")


def download_emovo(entry, workers):
    """EMOVO: download its 588 WAV files from the DagsHub copy into audio/<speaker>/, then check one SHA-256 over all
    of them against the one pinned in this script."""
    out = DATA / "emovo"
    speakers = ["f1", "f2", "f3", "m1", "m2", "m3"]  # 3 female and 3 male actors
    emotions = ["dis", "gio", "neu", "pau", "rab", "sor", "tri"]
    sentences = ["b1", "b2", "b3", "d1", "d2", "l1", "l2", "l3", "l4", "n1", "n2", "n3", "n4", "n5"]
    names = [f"{s}/{e}-{s}-{t}.wav" for s in speakers for e in emotions for t in sentences]
    raw = f"https://dagshub.com/api/v1/repos/{entry['id']}/raw/master/Dataset/"
    download_all([(raw + name, out / "audio" / name, None, "sha256") for name in names], min(workers, 4))
    # One SHA-256 over the list of "<file> <SHA-256>" lines, in the order above
    manifest = "".join(f"{name} {digest(out / 'audio' / name, 'sha256')}\n" for name in names)
    sha = hashlib.sha256(manifest.encode()).hexdigest()
    if sha != entry["sha256"]:
        sys.exit(f"ERROR: the 588 files differ from those pinned in this script (SHA-256 of the list {sha}).")
    print("  verified the 588 files against the SHA-256 pinned in this script")


def download_dataset(name, workers, converters):
    entry = DATASETS[name]
    host, ident = entry["host"], entry["id"]
    out = DATA / name
    print(f"Downloading {name} from {host} ({ident}) -> {out}")
    if name == "thai_ser":
        download_thai_ser(ident, out)
    elif host == "huggingface":
        download_huggingface(ident, out)
    elif host == "gdrive_folder":
        download_mead(entry, workers, converters)
    elif host == "meld":
        download_meld(entry, converters)
    elif host == "dagshub":
        download_emovo(entry, workers)
    elif host in ZIP_URLS:
        if entry["file"] in unpacked_zips(out):
            print(f"  already unpacked: {entry['file']} (its files in {out / 'audio'} are complete)")
        else:
            download_zip(ZIP_URLS[host].format(ident), out / entry["file"], entry["sha256"])
            unzip(out / entry["file"], out)
    else:
        files = {"osf": lambda: osf_files(ident), "zenodo": lambda: zenodo_files(ident),
                 "dataverse": lambda: dataverse_files(entry["server"], ident)}[host]()
        skip, keep = entry.get("skip"), entry.get("keep")  # regular expressions on the file path
        jobs = [(url, out / path, checksum, algo) for path, checksum, algo, url in files
                if not (skip and re.match(skip, path)) and not (keep and not re.match(keep, path))]
        done = unpacked_zips(out)  # zips unpacked by an earlier run: not downloaded again
        if done:
            print(f"  already unpacked: {len(done)} zip(s) ({', '.join(sorted(done))})")
        jobs = [job for job in jobs if job[1].relative_to(out).as_posix() not in done]
        download_all(jobs, workers)
        for job in jobs:  # zips are only packaging: unpack the audio into out/audio/, then delete the zip
            if job[1].suffix == ".zip":
                unzip(job[1], out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and ".cache" not in f.parts)
    print(f"Done: {size / 1e9:.2f} GB in {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Restore mirrored datasets or download from original sources.")
    parser.add_argument("--dataset", help="override dataset in config/download_data.yaml; mirrors selects the nine Release mirrors")
    parser.add_argument("--source", action="store_true", help="download from the original publisher instead of the pinned GitHub mirror")
    parser.add_argument("--provider", choices=("github", "source"), help="override the configured provider")
    parser.add_argument("--ca-bundle", type=Path, help="company/custom CA PEM bundle for HTTPS downloads")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    settings = yaml.safe_load(CONFIG.read_text())
    dataset = args.dataset or settings["dataset"]
    provider = "source" if args.source else (args.provider or settings.get("provider", "github"))
    workers, converters = settings["workers"], settings.get("convert_workers", 4)
    if provider not in ("github", "source"):
        parser.error("provider must be github or source")
    if dataset not in ["crema_d", *DATASETS, "all", "mirrors"]:
        parser.error(f"unknown dataset {dataset!r}; use crema_d, {', '.join(DATASETS)}, mirrors or all")
    if provider == "github" and not (dataset == "crema_d" and settings.get("crema_d", "samples") == "samples"):
        command = [sys.executable, str(ROOT / "scripts" / "common" / "setup_datasets.py"), dataset,
                   "--workers", str(workers)]
        if args.ca_bundle is not None:
            command.extend(["--ca-bundle", str(args.ca_bundle)])
        return subprocess.call(command)
    if provider == "github":
        # The six small test clips are tracked in the data submodule; initialize it without fetching any archives.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from setup_datasets import ensure_submodule
        ensure_submodule()
    if dataset == "mirrors":
        parser.error("mirrors requires provider github; choose one original dataset or all with --source")
    configure_https(args.ca_bundle)
    # ESD is excluded from the 13-dataset evaluation; only an explicit --dataset esd fetches it.
    names = ["crema_d", *(name for name in DATASETS if name != "esd")] if dataset == "all" else [dataset]
    for name in names:
        if name == "crema_d":
            download_crema_d("full" if dataset == "all" else settings["crema_d"], workers)
        else:
            download_dataset(name, workers, converters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
