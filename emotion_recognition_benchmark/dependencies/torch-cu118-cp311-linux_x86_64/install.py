#!/usr/bin/env python3
"""Install the verified Linux CPython 3.11 wheel set one package at a time.

The manifest lists wheels in dependency-first order. No package index is used.
Use --dry-run on any platform to print the commands without installing or
requiring the wheel files to be present.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SYNC_HINT = ("Synchronize the entire wheel directory from the current main branch, including manifest.json "
             "and all Python helpers; keep existing .whl files, then rerun download.py.")


def package_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("Invalid package name in manifest: " + repr(value))
    return re.sub(r"[-_.]+", "-", value).lower()


def check_platform():
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise ValueError("Installation requires CPython 3.11; use --dry-run to inspect commands on other versions.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Installation requires Linux x86_64; use --dry-run to inspect commands on other platforms.")
    libc, version = platform.libc_ver()
    match = re.match(r"^(\d+)\.(\d+)", version)
    if libc != "glibc" or not match or tuple(map(int, match.groups())) < (2, 28):
        raise ValueError("Installation requires glibc >= 2.28; detected {} {}.".format(libc or "unknown libc", version))


def load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema; expected 1.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("The manifest must contain a nonempty files list.")
    print("Wheel count: {}".format(len(files)), flush=True)
    roots = manifest.get("root_packages")
    if not isinstance(roots, list) or not roots:
        raise ValueError("Outdated or incomplete manifest: missing root_packages. " + SYNC_HINT)
    root_names = {package_name(name) for name in roots}
    missing_roots = {"torch", "torchaudio", "torchvision"} - root_names
    if missing_roots:
        raise ValueError("Manifest omits required roots: {}. {}".format(
            ", ".join(sorted(missing_roots)), SYNC_HINT))
    names = set()
    packages = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid wheel entry in manifest.")
        name = package_name(item.get("package"))
        if name in packages:
            raise ValueError("Repeated package in manifest: " + name)
        packages[name] = len(packages)
        if not isinstance(item.get("version"), str) or not item["version"]:
            raise ValueError("Missing package version for " + name)
        if not isinstance(item.get("requires"), list):
            raise ValueError("Outdated or incomplete manifest: missing dependencies for {}. {}".format(name, SYNC_HINT))
        filename = item.get("filename")
        if (not isinstance(filename, str) or Path(filename).name != filename
                or "/" in filename or "\\" in filename or not filename.endswith(".whl")):
            raise ValueError("Invalid wheel filename in manifest: " + repr(filename))
        if filename in names:
            raise ValueError("Repeated wheel filename in manifest: " + filename)
        names.add(filename)
        size = item.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("Invalid size_bytes for " + filename)
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid SHA-256 for " + filename)
    for name in root_names:
        if name not in packages:
            raise ValueError("Manifest is missing root package {}. {}".format(name, SYNC_HINT))
    for item in files:
        name = package_name(item["package"])
        for value in item["requires"]:
            dependency = package_name(value)
            if dependency not in packages:
                raise ValueError("Incomplete manifest: {} requires missing package {}. {}".format(name, dependency, SYNC_HINT))
            if packages[dependency] >= packages[name]:
                raise ValueError("Manifest is not dependency-first: {} must precede {}. {}".format(dependency, name, SYNC_HINT))
    return manifest


def verify_dependency_closure(root, manifest):
    """Verify actual wheel metadata, rather than trusting only declared manifest edges."""
    if not (root / "audit.py").is_file():
        raise ValueError("Missing dependency auditor audit.py. " + SYNC_HINT)
    try:
        auditor = importlib.import_module("audit")
    except ImportError as exc:
        raise ValueError("Cannot load the wheel dependency auditor: {}. {}".format(exc, SYNC_HINT)) from exc
    auditor.audit(root, manifest)


def verify_all(root, files):
    """Finish validating the entire wheel set before the first pip command."""
    invalid = []
    for item in files:
        path = root / item["filename"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            invalid.append(item["filename"] + " (missing or wrong size)")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            invalid.append(item["filename"] + " (SHA-256 mismatch)")
        else:
            print("Verified: " + item["filename"], flush=True)
    if invalid:
        raise ValueError("Wheel verification failed; nothing was installed. Run download.py to restore:\n  "
                         + "\n  ".join(invalid))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="print numbered offline install commands without checking platform or local wheel files")
    args = parser.parse_args(argv)
    try:
        root = ROOT.resolve()
        print("Manifest: " + str(root / "manifest.json"), flush=True)
        manifest = load_manifest(root)
        files = manifest["files"]
        if not args.dry_run:
            check_platform()
            verify_all(root, files)
            verify_dependency_closure(root, manifest)
            print("Verified complete Linux dependency closure and install order.", flush=True)
        commands = [
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(root / item["filename"])]
            for item in files
        ]
        commands.append([sys.executable, "-m", "pip", "check"])
        for index, command in enumerate(commands, 1):
            print("{}. {}".format(index, shlex.join(command)), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print("ERROR: Command failed with exit code {}: {}. Installation stopped.".format(
            exc.returncode, shlex.join(exc.cmd)), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Installation stopped; completed pip steps remain installed.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
