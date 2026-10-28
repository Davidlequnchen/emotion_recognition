"""Download the models of config/models.yaml into models/<key>/, ready to load without the network.

Usage (from the emotion_recognition/ folder):
    python scripts/common/download_model.py
Settings are in config/download_model.yaml:
    models: all   every model registered in config/models.yaml, or a list of their keys such as [meralion_ser_v1]
    hub: hf       Hugging Face; ms = ModelScope (faster in mainland China; only for models with an ms_repo)
Each model's repository is downloaded into its `dir`. Then, if the model's loader (scripts/models/<loader>.py) has a
prepare() function, it runs: it fetches the small extra files the model needs to load offline (e.g. a base model's
config.json) or derives a smaller weights file. Files that are already complete are not downloaded again, so the script
can simply be re-run.

A machine that cannot reach Hugging Face gets the models through git instead: split_parts.py here, then `git pull`
and join_parts.py there (see models/README.md).
"""
import importlib
import platform
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "download_model.yaml"  # this script's settings
MODELS_CONFIG = ROOT / "config" / "models.yaml"    # the model registry
LOADERS = ROOT / "scripts" / "models"       # the model loaders: scripts/models/<loader>.py
# Not downloaded: compiled Python files, and the repository's .gitattributes: its Git LFS rules (*.png, *.pdf, ...)
# would apply to the model folder in this repository and send the licence and figures to LFS when committed
IGNORE = ["__pycache__/*", "*.pyc", ".gitattributes"]


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the settings (config/download_model.yaml) and the model registry (config/models.yaml)
    # ------------------------------------------------------------------
    step("Step 1/4 | Load configuration")
    settings = yaml.safe_load(CONFIG.read_text())
    registry = yaml.safe_load(MODELS_CONFIG.read_text())
    wanted, hub = settings["models"], settings["hub"]
    keys = list(registry) if wanted == "all" else [wanted] if isinstance(wanted, str) else list(wanted)
    unknown = [k for k in keys if k not in registry]
    if unknown:
        sys.exit(f"ERROR: unknown model(s) {unknown} in `models` of config/download_model.yaml; use: "
                 f"{', '.join(registry)} or all")
    if hub not in ("hf", "ms"):
        sys.exit(f"ERROR: `hub` in config/download_model.yaml is {hub!r}; use hf (Hugging Face) or ms (ModelScope)")
    no_ms = [k for k in keys if hub == "ms" and not registry[k].get("ms_repo")]
    if no_ms:
        sys.exit(f"ERROR: not on ModelScope (no ms_repo in config/models.yaml): {', '.join(no_ms)}. In "
                 "config/download_model.yaml, set `hub: hf`, or `models` to models that have an ms_repo.")
    no_loader = [registry[k]["loader"] for k in keys if not (LOADERS / f"{registry[k]['loader']}.py").exists()]
    if no_loader:
        sys.exit(f"ERROR: loader(s) not found: {', '.join(f'scripts/models/{n}.py' for n in no_loader)}")
    print(f"  config      : {CONFIG}")
    print(f"  registry    : {MODELS_CONFIG}")
    print(f"  hub         : {'Hugging Face' if hub == 'hf' else 'ModelScope'}")
    for key in keys:
        entry = registry[key]
        print(f"  {key:<22}: {entry['hf_repo' if hub == 'hf' else 'ms_repo']} -> {entry['dir']}/")

    # ------------------------------------------------------------------
    # Step 2: Check the software stack
    # ------------------------------------------------------------------
    step("Step 2/4 | Check environment")
    if hub == "hf":
        import huggingface_hub as library
        from huggingface_hub import HfApi, snapshot_download
    else:
        import modelscope as library
        from modelscope import snapshot_download
    print(f"  Python      : {platform.python_version()}")
    print(f"  library     : {library.__name__} {library.__version__}")

    # ------------------------------------------------------------------
    # Step 3: Download each model, then run its loader's prepare() (extra files for loading offline)
    # ------------------------------------------------------------------
    step(f"Step 3/4 | Download {len(keys)} model(s)")
    sys.path.insert(0, str(LOADERS))  # the loaders import their helpers (audio_io.py) from there
    expected = {}  # Hugging Face only: file -> size there, checked in step 4
    for key in keys:
        entry = registry[key]
        target = ROOT / entry["dir"]
        print(f"\n  {key}")
        t0 = time.perf_counter()
        if hub == "hf":
            # Pin the latest commit, so that every file comes from the same version of the repository
            info = HfApi().model_info(entry["hf_repo"], files_metadata=True)
            print(f"  commit      : {info.sha}")
            snapshot_download(repo_id=entry["hf_repo"], revision=info.sha, local_dir=target, ignore_patterns=IGNORE)
            expected |= {target / s.rfilename: s.size for s in info.siblings}
        else:
            snapshot_download(entry["ms_repo"], local_dir=str(target), ignore_patterns=IGNORE)
        print(f"  downloaded  : {time.perf_counter() - t0:.0f} s")
        if (target / ".gitattributes").exists():  # left by an earlier download (see IGNORE)
            (target / ".gitattributes").unlink()
            print("  removed     : .gitattributes (Hugging Face's Git LFS rules)")
        loader = importlib.import_module(entry["loader"])
        if hasattr(loader, "prepare"):
            print(f"  prepare     : scripts/models/{entry['loader']}.py")
            loader.prepare(target)
        else:
            print("  prepare     : none needed")

    # ------------------------------------------------------------------
    # Step 4: List the files of each model; check the sizes of the downloaded ones against Hugging Face
    # ------------------------------------------------------------------
    step("Step 4/4 | Files")
    wrong = 0
    for key in keys:
        target = ROOT / registry[key]["dir"]
        files = sorted(p for p in target.rglob("*") if p.is_file()
                       and not {".cache", "__pycache__"} & set(p.relative_to(target).parts))
        parts = [p for p in files if p.match("*.part[0-9][0-9][0-9]")]  # copies for git, from split_parts.py
        print(f"\n  {key} ({target.relative_to(ROOT)}/)")
        for f in files:
            if f in parts:
                continue
            size, check = f.stat().st_size, ""
            if f in expected:
                check = "  OK" if size == expected[f] else "  ERROR: size differs from Hugging Face"
                wrong += size != expected[f]
            print(f"    {str(f.relative_to(target)):<44} {size / 1e6:>10,.2f} MB{check}")
        print(f"    total {sum(f.stat().st_size for f in files if f not in parts) / 1e9:,.2f} GB"
              + (f", plus {len(parts)} parts for git (split_parts.py)" if parts else ""))
    if wrong:
        sys.exit(f"ERROR: {wrong} file(s) differ in size from Hugging Face; delete them and run this script again")
    print("\nDone. On a machine without Hugging Face, see models/README.md.")


if __name__ == "__main__":
    main()
