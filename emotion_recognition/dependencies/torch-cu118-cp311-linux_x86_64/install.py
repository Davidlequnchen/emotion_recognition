#!/usr/bin/env python3
"""Install the verified Linux CPython 3.11 wheel set one package at a time.

The manifest lists wheels in dependency-first order. No package index is used.
Use --dry-run on any platform to print the commands without installing or
requiring the wheel files to be present.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def check_platform():
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise ValueError("Installation requires CPython 3.11; use --dry-run to inspect commands on other versions.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Installation requires Linux x86_64; use --dry-run to inspect commands on other platforms.")
    libc, version = platform.libc_ver()
    match = re.match(r"^(\d+)\.(\d+)", version)
    if libc != "glibc" or not match or tuple(map(int, match.groups())) < (2, 28):
        raise ValueError("Installation requires glibc >= 2.28; detected {} {}.".format(libc or "unknown libc", version))


def load_files(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema; expected 1.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("The manifest must contain a nonempty files list.")
    names = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid wheel entry in manifest.")
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
    return files


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
        files = load_files(root)
        if not args.dry_run:
            check_platform()
            verify_all(root, files)
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
