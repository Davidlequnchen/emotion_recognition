"""Initialize the pinned data submodule and restore its nine mirrored datasets.

Run from the project folder: python3 scripts/common/setup_datasets.py [DATASET ...]
Python 3.9+ and Git are sufficient. Omit DATASET (or use mirrors) for the nine
Release mirrors. External sources remain available through download_data.py.
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
import tempfile
import uuid


PROJECT = Path(__file__).resolve().parents[2]
SUBMODULE_NAME = "emotion-datasets"
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
PRESERVED_EXTERNAL = {"crema_d", "mess", "ravdess", "emozionalmente", "esd"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid dataset file path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Invalid dataset file path: {relative!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes its directory: {relative!r}")
    return target


def run_git(repository, arguments, git_command=("git",), capture=False):
    command = [*git_command, "-C", str(repository), *arguments]
    if capture:
        return subprocess.check_output(command, text=True, stderr=subprocess.PIPE).strip()
    subprocess.run(command, check=True)


def ensure_submodule(project=PROJECT, git_command=("git",)):
    """Keep an old data directory intact before initializing only the data submodule."""
    project = Path(project).resolve()
    repository = project.parent
    data = project / "data"
    relative = data.relative_to(repository).as_posix()
    try:
        declared = run_git(repository, ["config", "--file", ".gitmodules", "--get",
                                       f"submodule.{SUBMODULE_NAME}.path"], git_command, capture=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("The datasets submodule is not registered here. Pull the main repository, "
                           "then rerun setup_datasets.py.") from exc
    if declared != relative:
        raise RuntimeError(f"Unexpected data submodule path: {declared!r}; expected {relative!r}")
    if data.is_symlink() or (data.exists() and not data.is_dir()):
        raise RuntimeError(f"{data} must be a directory; its existing content has not been changed.")
    if data.exists() and not (data / ".git").exists() and any(data.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = project / f"data.before-submodule-{stamp}"
        if backup.exists():
            backup = backup.with_name(backup.name + "-" + uuid.uuid4().hex[:8])
        data.rename(backup)
        print(f"Preserved the previous data directory at {backup}", flush=True)
    try:
        run_git(repository, ["submodule", "sync", "--", relative], git_command)
        run_git(repository, ["submodule", "update", "--init", "--", relative], git_command)
    except (OSError, subprocess.CalledProcessError) as exc:
        backups = sorted(project.glob("data.before-submodule-*"))
        recovery = (" Previous data remain in " + ", ".join(str(p) for p in backups) + ".") if backups else ""
        raise RuntimeError("Could not initialize the datasets submodule. Check GitHub access and rerun this "
                           "command." + recovery) from exc
    if not (data / "download.py").is_file() or not (data / "manifest.json").is_file():
        raise RuntimeError(f"The pinned data submodule at {data} has no download.py or manifest.json; "
                           "check the main repository's submodule version.")
    return data


def selected_datasets(manifest, wanted=None, listing=False):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("datasets"), dict):
        raise ValueError("Unsupported dataset manifest schema")
    datasets = manifest["datasets"]
    if not wanted:
        keys = list(datasets) if listing else [k for k, value in datasets.items() if value.get("storage") == "release"]
    elif wanted == ["mirrors"]:
        keys = [k for k, value in datasets.items() if value.get("storage") == "release"]
    elif wanted == ["all"]:
        keys = list(datasets)
    else:
        keys = list(dict.fromkeys(wanted))
    for key in keys:
        if not isinstance(key, str) or not KEY.fullmatch(key) or key not in datasets:
            raise ValueError(f"Unknown dataset {key!r}; choose from: {', '.join(datasets)}")
        entry = datasets[key]
        if entry.get("storage") not in ("release", "external") or not isinstance(entry.get("files", []), list):
            raise ValueError(f"Invalid dataset entry: {key}")
        for item in entry.get("files", []):
            if (not isinstance(item, dict) or not isinstance(item.get("size"), int)
                    or isinstance(item["size"], bool) or item["size"] < 0):
                raise ValueError(f"Invalid file size for {key}")
            if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError(f"Invalid SHA-256 for {key}/{item.get('path')}")
            if not isinstance(item.get("path"), str):
                raise ValueError(f"Invalid file path for {key}")
            safe_path(Path("/validation-root"), f"{key}/{item['path']}")
    return keys


def link_or_copy(source, target, expected=None):
    """Populate a missing target without deleting or modifying the original file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        temporary = target.with_name(target.name + ".setup-" + uuid.uuid4().hex + ".partial")
        with source.open("rb") as original, temporary.open("xb") as output:
            shutil.copyfileobj(original, output, length=4 * 1024 * 1024)
        size = expected["size"] if expected else source.stat().st_size
        checksum = expected["sha256"] if expected else sha256(source)
        if temporary.stat().st_size != size or sha256(temporary) != checksum:
            raise RuntimeError(f"Copy verification failed; original remains at {source}; partial remains at {temporary}")
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"{target} appeared during setup; verified copy remains at {temporary}")
        temporary.replace(target)
    return True


def data_backups(data):
    return [p for p in sorted(data.parent.glob("data.before-submodule-*"), reverse=True)
            if p.is_dir() and not p.is_symlink()]


def reuse_files(data, manifest, keys):
    """Reuse only manifest-verified mirrored data; leave additional materials in backups."""
    backups = data_backups(data)
    restored = 0
    for key in keys:
        if manifest["datasets"][key]["storage"] != "release":
            continue
        count = 0
        for item in manifest["datasets"][key]["files"]:
            target = safe_path(data, f"{key}/{item['path']}")
            if target.exists() or target.is_symlink():
                continue
            for backup in backups:
                source = safe_path(backup, f"{key}/{item['path']}")
                if source.is_symlink() or not source.is_file() or source.stat().st_size != item["size"]:
                    continue
                if sha256(source) != item["sha256"]:
                    continue
                if link_or_copy(source, target, item):
                    restored += 1
                    count += 1
                break
        if count:
            print(f"Reused {count:,} verified file(s) for {key}", flush=True)
    return restored




def legacy_target(data, relative):
    """Reject every symlink component before publishing recovered bytes."""
    target = safe_path(data, relative)
    cursor = data
    if cursor.is_symlink():
        raise ValueError("Symlink data root is not allowed for legacy Git recovery")
    for component in PurePosixPath(relative).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError(f"Symlink dataset path is not allowed for legacy Git recovery: {relative!r}")
    return target


def legacy_blob_map(data, legacy, git_command=("git",)):
    """Read only local history; missing/shallow history never triggers a fetch."""
    if not isinstance(legacy, dict):
        return None
    commit, prefix = legacy.get("commit"), legacy.get("prefix")
    if not isinstance(commit, str) or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ValueError("Invalid legacy Git commit in dataset manifest")
    if not isinstance(prefix, str):
        raise ValueError("Invalid legacy Git prefix in dataset manifest")
    repository = data.parent.parent
    if prefix != data.relative_to(repository).as_posix():
        raise ValueError("Legacy Git prefix does not match this data submodule")
    environment = os.environ.copy()
    environment["GIT_NO_LAZY_FETCH"] = "1"
    command = [*git_command, "-C", str(repository)]
    available = subprocess.run([*command, "cat-file", "-e", commit + "^{commit}"], env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if available.returncode:
        print("Previous dataset Git history is not available locally; using Releases without fetching history.",
              flush=True)
        return None
    tree = subprocess.run([*command, "ls-tree", "-r", "-l", "-z", commit, "--", prefix],
                          env=environment, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if tree.returncode:
        print("Previous dataset Git tree is incomplete locally; using Releases without fetching history.", flush=True)
        return None
    blobs = {}
    start = prefix + "/"
    for row in tree.stdout.split(b"\0"):
        if not row:
            continue
        header, raw_path = row.split(b"\t", 1)
        mode, kind, oid, size = header.split()
        if mode not in (b"100644", b"100755") or kind != b"blob" or not size.isdigit():
            continue  # never restore symlink or submodule objects as ordinary dataset files.
        name = os.fsdecode(raw_path)
        if name.startswith(start):
            blobs[name[len(start):]] = (oid.decode("ascii"), int(size))
    return repository, environment, blobs


class GitBlobReader:
    """Stream requested blobs through one cat-file process, without loading them into memory."""
    def __init__(self, repository, environment, git_command=("git",)):
        self.errors = tempfile.TemporaryFile()
        self.process = subprocess.Popen([*git_command, "-C", str(repository), "cat-file", "--batch"],
                                        env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.errors)

    def copy(self, oid, expected_size, output, digest):
        self.process.stdin.write((oid + "\n").encode("ascii"))
        self.process.stdin.flush()
        fields = self.process.stdout.readline(200).strip().split()
        if len(fields) == 2 and fields[1] == b"missing":
            return False
        if (len(fields) != 3 or fields[0] != oid.encode("ascii") or fields[1] != b"blob"
                or not fields[2].isdigit()):
            raise RuntimeError("Could not read a local legacy dataset Git object")
        size = int(fields[2])
        remaining = size
        while remaining:
            block = self.process.stdout.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise RuntimeError("A local legacy dataset Git object is incomplete")
            if size == expected_size:
                output.write(block)
                digest.update(block)
            remaining -= len(block)
        if self.process.stdout.read(1) != b"\n":
            raise RuntimeError("Invalid boundary in local legacy dataset Git object stream")
        return size == expected_size

    def close(self):
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.process.stdout.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.errors.close()


def legacy_file_blobs(blobs, part_groups, relative, expected_size):
    direct = blobs.get(relative)
    if direct is not None and direct[1] == expected_size:
        return [direct]
    # nEMO's old main-repository history held the Parquet files as consecutive .partNNN blobs.
    parts = part_groups.get(relative, [])
    if (not parts or [number for number, _ in parts] != list(range(len(parts)))
            or sum(value[1] for _, value in parts) != expected_size):
        return None
    return [value for _, value in parts]


def reuse_legacy_git(data, manifest, keys, git_command=("git",)):
    """Recover checkout-deleted tracked files from local Git blobs, validated against the new manifest."""
    if "legacy_git" not in manifest:
        return 0
    missing = [(key, item) for key in keys if manifest["datasets"][key]["storage"] == "release"
               for item in manifest["datasets"][key]["files"]
               if not legacy_target(data, f"{key}/{item['path']}").exists()]
    if not missing:
        return 0
    local = legacy_blob_map(data, manifest["legacy_git"], git_command)
    if local is None:
        return 0
    repository, environment, blobs = local
    part_groups = {}
    for name, blob in blobs.items():
        match = re.fullmatch(r"(.*)\.part([0-9]{3})", name)
        if match is not None:
            part_groups.setdefault(match[1], []).append((int(match[2]), blob))
    part_groups = {name: sorted(parts) for name, parts in part_groups.items()}
    reader = None
    restored = 0
    counts = {}
    try:
        for key, item in missing:
            relative = f"{key}/{item['path']}"
            target = legacy_target(data, relative)
            if target.exists() or target.is_symlink():
                continue
            objects = legacy_file_blobs(blobs, part_groups, relative, item["size"])
            if objects is None:
                continue
            if reader is None:
                reader = GitBlobReader(repository, environment, git_command)
            legacy_target(data, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".legacy-git-" + uuid.uuid4().hex + ".partial")
            digest = hashlib.sha256()
            try:
                with temporary.open("xb") as output:
                    valid = all(reader.copy(oid, size, output, digest) for oid, size in objects)
                if (not valid or temporary.stat().st_size != item["size"]
                        or digest.hexdigest() != item["sha256"]):
                    continue  # a different/corrupt old version is left for the Release downloader.
                legacy_target(data, relative)
                if target.exists() or target.is_symlink():
                    continue
                # Atomic no-clobber publication. A file created by another process is never replaced.
                try:
                    os.link(temporary, target)
                except (FileExistsError, PermissionError, NotImplementedError):
                    continue
                restored += 1
                counts[key] = counts.get(key, 0) + 1
            finally:
                if temporary.exists():
                    temporary.unlink()
    except (OSError, RuntimeError) as exc:
        print(f"Local legacy Git recovery stopped ({exc}); missing files will use Releases.", flush=True)
    finally:
        if reader is not None:
            reader.close()
    for key, count in counts.items():
        print(f"Recovered {count:,} verified file(s) for {key} from local Git history", flush=True)
    return restored

def preserve_external_files(data):
    """Restore local external/archived data paths; no such files are uploaded or downloaded here."""
    count = 0
    for backup in data_backups(data):
        for key in sorted(PRESERVED_EXTERNAL):
            folder = backup / key
            if not folder.is_dir() or folder.is_symlink():
                continue
            for source in folder.rglob("*"):
                relative = source.relative_to(backup)
                if ".git" in relative.parts or source.name == ".gitattributes":
                    continue  # keep repository/filter metadata in the preserved backup only.
                if source.is_symlink() or not source.is_file():
                    continue
                if not source.resolve().is_relative_to(backup.resolve()):
                    continue
                target = safe_path(data, relative.as_posix())
                if target.exists() or target.is_symlink():
                    continue
                count += link_or_copy(source, target)
    if count:
        print(f"Preserved {count:,} existing file(s) from external/archived datasets at their original paths", flush=True)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help="dataset keys; omit for the nine Release mirrors")
    parser.add_argument("--list", action="store_true", help="list all registered datasets without downloading")
    parser.add_argument("--verify-only", action="store_true", help="verify mirrored files without downloading them")
    parser.add_argument("--workers", type=int, default=2, help="parallel Release downloads (default: 2)")
    parser.add_argument("--ca-bundle", type=Path, help="company/custom CA PEM bundle for Release downloads")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if any(not KEY.fullmatch(key) for key in args.datasets):
        parser.error("DATASET values must be dataset keys, not paths")
    try:
        data = ensure_submodule()
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        keys = selected_datasets(manifest, args.datasets, args.list)
        external = [key for key in keys if manifest["datasets"][key]["storage"] == "external"]
        if not args.list:
            preserve_external_files(data)
        if external and not args.list and not args.verify_only:
            instructions = "; ".join(f"python scripts/common/download_data.py --dataset {key} --source" for key in external)
            raise ValueError("These datasets retain their original sources: " + ", ".join(external) + ". Run: " + instructions)
        if not args.list:
            reuse_files(data, manifest, keys)
            reuse_legacy_git(data, manifest, keys)
        if not keys:
            print("No mirrored datasets selected", flush=True)
            return 0
        command = [sys.executable, str(data / "download.py"), *keys, "--workers", str(args.workers)]
        if args.list:
            command.append("--list")
        if args.verify_only:
            command.append("--verify-only")
        if args.ca_bundle is not None:
            command.extend(["--ca-bundle", str(args.ca_bundle)])
        return subprocess.call(command)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
