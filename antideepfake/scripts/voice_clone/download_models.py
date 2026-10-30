"""Download everything the voice-cloning models of config/voice_clone.yaml need, so that generate.py runs offline.

Usage (from the antideepfake/ folder, in the voice_clone environment):
    python scripts/voice_clone/download_models.py
Settings are in config/voice_clone.yaml: `models` chooses the models, `registry` gives each one's Hugging Face
repository, pinned revision, folder and files. The script fetches, in this order:
  1. the official CosyVoice code from GitHub at a pinned commit, with its Matcha-TTS submodule -> third_party/CosyVoice
  2. the chosen models from Hugging Face, only the files listed in the registry -> models/voice_clone/<folder>
  3. the wetext text-normalisation FSTs from ModelScope, with git and Git LFS (CosyVoice's text frontend would
     otherwise fetch them at run time) -> models/voice_clone/wetext
Needs git and Git LFS. Files that are already present are skipped. A Hugging Face .gitattributes is never kept in a
model folder.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (antideepfake/); paths in the config are relative to it
CONFIG = ROOT / "config" / "voice_clone.yaml"
COSYVOICE_MODELS = {"cosyvoice2", "cosyvoice3"}


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def list_files(target):
    """Print every file of a folder with its size (the download caches .cache/ and .git/ are left out)."""
    files = sorted(p for p in target.rglob("*") if p.is_file() and not {".cache", ".git"} & set(p.relative_to(target).parts))
    for f in files:
        print(f"  {str(f.relative_to(target)):<44} {f.stat().st_size / 1e6:>10.2f} MB")
    print(f"  {'total':<44} {sum(f.stat().st_size for f in files) / 1e9:>10.2f} GB")


def digest(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress immediately, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the configuration
    # ------------------------------------------------------------------
    step("Step 1/4 | Load configuration")
    cfg = yaml.safe_load(CONFIG.read_text())
    registry = cfg["registry"]
    keys = list(registry) if cfg["models"] == "all" else cfg["models"]
    unknown = [k for k in keys if k not in registry]
    if unknown:
        sys.exit(f"ERROR: unknown model(s) {unknown} in `models`. Registered: {', '.join(registry)}")
    print(f"  config : {CONFIG}")
    print(f"  models : {', '.join(keys)}")

    # ------------------------------------------------------------------
    # Step 2: The CosyVoice code from GitHub, at the pinned commit, with its Matcha-TTS submodule
    # ------------------------------------------------------------------
    step("Step 2/4 | CosyVoice code (GitHub)")
    code = cfg["cosyvoice_code"]
    repo_dir = ROOT / code["folder"]
    if not COSYVOICE_MODELS & set(keys):
        print("  not needed (no CosyVoice model chosen)")
    else:
        if not (repo_dir / ".git").exists():
            print(f"  cloning {code['repo']} -> {repo_dir}")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            git("clone", "--quiet", code["repo"], str(repo_dir))
        if git("rev-parse", "HEAD", cwd=repo_dir) != code["commit"]:
            print(f"  checking out commit {code['commit']}")
            git("fetch", "--quiet", "origin", cwd=repo_dir)
            git("checkout", "--quiet", code["commit"], cwd=repo_dir)
        git("submodule", "update", "--init", "--recursive", "--quiet", cwd=repo_dir)
        print(f"  CosyVoice  : {git('log', '-1', '--format=%H (%cs) %s', cwd=repo_dir)}")
        for line in git("submodule", "status", cwd=repo_dir).splitlines():
            print(f"  submodule  : {line.strip()}")

    # ------------------------------------------------------------------
    # Step 3: The models (Hugging Face), only the files that the inference code reads
    # ------------------------------------------------------------------
    step("Step 3/4 | Models (Hugging Face)")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for key in keys:
        m = registry[key]
        target = ROOT / m["folder"]
        print(f"\n  [{key}] {m['hf_repo']} @ {m['revision'][:12]} ({m['licence']}) -> {target}")
        info = api.model_info(m["hf_repo"], revision=m["revision"], files_metadata=True)
        sizes = {s.rfilename: s.size for s in info.siblings}
        missing = [f for f in m["files"] if f not in sizes]
        if missing:
            sys.exit(f"ERROR: {m['hf_repo']} at {m['revision']} has no file(s) {missing}.")
        print(f"  files      : {len(m['files'])} of {len(sizes)} in the repository, "
              f"{sum(sizes[f] for f in m['files']) / 1e9:.2f} of {sum(sizes.values()) / 1e9:.2f} GB")
        t0 = time.perf_counter()
        snapshot_download(repo_id=m["hf_repo"], revision=m["revision"], allow_patterns=m["files"], local_dir=target)
        if (target / ".gitattributes").exists():  # never keep Hugging Face's Git LFS rules in a model folder
            (target / ".gitattributes").unlink()
        wrong = [f for f in m["files"] if (target / f).stat().st_size != sizes[f]]
        if wrong:
            sys.exit(f"ERROR: size mismatch for {wrong} in {target}. Delete them and run the script again.")
        print(f"  done in    : {time.perf_counter() - t0:.0f} s, sizes checked")
        list_files(target)

    # ------------------------------------------------------------------
    # Step 4: The wetext FSTs (ModelScope), used by CosyVoice's text frontend
    # ------------------------------------------------------------------
    step("Step 4/4 | wetext text-normalisation FSTs (ModelScope)")
    wt = cfg["wetext"]
    wt_dir = ROOT / wt["folder"]
    if not COSYVOICE_MODELS & set(keys):
        print("  not needed (no CosyVoice model chosen)")
    elif all((wt_dir / f).exists() and digest(wt_dir / f) == h for f, h in wt["files"].items()):
        print(f"  already present in {wt_dir}, SHA-256 verified")
        list_files(wt_dir)
    else:
        # A clone without the large files (Git LFS pointers only), then only the needed files with Git LFS
        print(f"  {wt['ms_repo']} @ {wt['revision'][:12]} ({wt['licence']}) -> {wt_dir}")
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "wetext"
            subprocess.run(["git", "clone", "--quiet", f"https://www.modelscope.cn/{wt['ms_repo']}.git", str(clone)],
                           check=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
            git("checkout", "--quiet", wt["revision"], cwd=clone)
            try:
                git("lfs", "pull", "--include", ",".join(wt["files"]), cwd=clone)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                sys.exit(f"ERROR: git lfs pull failed ({e}). Install Git LFS (`brew install git-lfs`, "
                         "`git lfs install`) and run the script again.")
            for name, sha256 in wt["files"].items():
                if digest(clone / name) != sha256:
                    sys.exit(f"ERROR: {name} does not match the SHA-256 in config/voice_clone.yaml.")
                (wt_dir / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(clone / name, wt_dir / name)
                print(f"  {name}: SHA-256 verified")
        list_files(wt_dir)

    print("\nDone. Generate with: python scripts/voice_clone/generate.py")


if __name__ == "__main__":
    main()
