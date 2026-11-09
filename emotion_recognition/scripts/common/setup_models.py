"""Initialize the pinned models submodule, reuse local weights and fetch missing Release assets.

Run from the project folder: python scripts/common/setup_models.py [MODEL ...]
Only Python 3.9+ and Git are required. With no MODEL argument, every model is selected.
The models repository's download.py handles checksums, resumable downloads and reconstruction.
"""
import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import uuid


PROJECT = Path(__file__).resolve().parents[2]
SUBMODULE_NAME = "emotion-models"
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    """Resolve a manifest path while rejecting traversal and escaping symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid model file path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Invalid model file path: {relative!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Model path escapes its directory: {relative!r}")
    return target


def run_git(repository, arguments, git_command=("git",), capture=False):
    command = [*git_command, "-C", str(repository), *arguments]
    if capture:
        return subprocess.check_output(command, text=True, stderr=subprocess.PIPE).strip()
    subprocess.run(command, check=True)


def ensure_submodule(project=PROJECT, git_command=("git",)):
    """Preserve an old non-Git models folder before initializing only this submodule."""
    project = Path(project).resolve()
    repository = project.parent
    models = project / "models"
    relative = models.relative_to(repository).as_posix()
    try:
        declared = run_git(repository, ["config", "--file", ".gitmodules", "--get",
                                       f"submodule.{SUBMODULE_NAME}.path"], git_command, capture=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("The models submodule is not registered here. Run git pull in the main repository "
                           "first, then rerun setup_models.py.") from exc
    if declared != relative:
        raise RuntimeError(f"Unexpected models submodule path: {declared!r}; expected {relative!r}")
    if models.is_symlink() or (models.exists() and not models.is_dir()):
        raise RuntimeError(f"{models} must be a directory; its existing content has not been changed.")
    backup = None
    if models.exists() and not (models / ".git").exists() and any(models.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = project / f"models.before-submodule-{stamp}"
        if backup.exists():
            backup = backup.with_name(backup.name + "-" + uuid.uuid4().hex[:8])
        models.rename(backup)
        print(f"Preserved the previous models directory at {backup}", flush=True)
    try:
        run_git(repository, ["submodule", "sync", "--", relative], git_command)
        run_git(repository, ["submodule", "update", "--init", "--", relative], git_command)
    except (OSError, subprocess.CalledProcessError) as exc:
        backups = sorted(project.glob("models.before-submodule-*"))
        recovery = (" Previous model files remain in " + ", ".join(str(p) for p in backups) + ".") if backups else ""
        raise RuntimeError("Could not initialize the models submodule. Check access to its GitHub repository "
                           "and rerun this command." + recovery) from exc
    if not (models / "download.py").is_file() or not (models / "manifest.json").is_file():
        raise RuntimeError(f"The pinned submodule at {models} has no download.py or manifest.json; "
                           "check the main repository's models submodule version.")
    return models


def selected_models(manifest, wanted):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ValueError("Unsupported models manifest schema")
    models = manifest["models"]
    keys = list(models) if not wanted or wanted == ["all"] else list(dict.fromkeys(wanted))
    for key in keys:
        if not isinstance(key, str) or not KEY.fullmatch(key) or key not in models:
            raise ValueError(f"Unknown model {key!r}; choose from: {', '.join(models)}")
        for item in models[key].get("files", []):
            if not isinstance(item.get("size"), int) or item["size"] <= 0:
                raise ValueError(f"Invalid file size for {key}/{item.get('path')}")
            if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError(f"Invalid SHA-256 for {key}/{item.get('path')}")
    return keys


def reuse_weights(models, manifest, keys):
    """Hardlink verified old weights into missing targets; leave every backup in place."""
    backups = sorted(models.parent.glob("models.before-submodule-*"), reverse=True)
    restored = 0
    for key in keys:
        for item in manifest["models"][key]["files"]:
            target = safe_path(models, f"{key}/{item['path']}")
            if target.exists() or target.is_symlink():
                continue  # download.py verifies existing weights and replaces damaged files atomically.
            for backup in backups:
                if not backup.is_dir() or backup.is_symlink():
                    continue
                source = safe_path(backup, f"{key}/{item['path']}")
                if source.is_symlink() or not source.is_file() or source.stat().st_size != item["size"]:
                    continue
                print(f"Checking previous local weight {source}", flush=True)
                if sha256(source) != item["sha256"]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except FileExistsError:
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                        raise
                    temporary = target.with_name(target.name + ".setup-" + uuid.uuid4().hex + ".partial")
                    with source.open("rb") as original, temporary.open("xb") as output:
                        shutil.copyfileobj(original, output, length=4 * 1024 * 1024)
                    if temporary.stat().st_size != item["size"] or sha256(temporary) != item["sha256"]:
                        raise RuntimeError(f"Copy verification failed; original remains at {source}; "
                                           f"partial copy remains at {temporary}")
                    if target.exists() or target.is_symlink():
                        raise RuntimeError(f"{target} appeared during setup; verified copy remains at {temporary}")
                    temporary.replace(target)
                restored += 1
                print(f"Reused verified local weight: {key}/{item['path']}", flush=True)
                break
    if backups:
        print(f"Reused {restored} weight file(s). Previous model directories are retained: "
              + ", ".join(str(p) for p in backups), flush=True)
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", metavar="MODEL", help="model keys; omit for all")
    parser.add_argument("--verify-only", action="store_true", help="verify existing weights without downloading them")
    parser.add_argument("--workers", type=int, default=2, help="parallel downloads (default: 2)")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if any(not KEY.fullmatch(key) for key in args.models):
        parser.error("MODEL values must be model keys, not paths")
    try:
        models = ensure_submodule()
        manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
        keys = selected_models(manifest, args.models)
        reuse_weights(models, manifest, keys)
        command = [sys.executable, str(models / "download.py"), *keys, "--workers", str(args.workers)]
        if args.verify_only:
            command.append("--verify-only")
        return subprocess.call(command)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
