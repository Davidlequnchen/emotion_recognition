#!/usr/bin/env python3
"""Restore the pinned CUDA 11.8 wheels beside this script (Python 3.9+, standard library only).

Existing verified wheels are reused. Interrupted downloads resume on rerun;
files are renamed to .whl only after their size and SHA-256 match the manifest.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent


def matches(path, item):
    if not path.is_file() or path.stat().st_size != item["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


def download(item, url, root=ROOT):
    target = root / item["filename"]
    if matches(target, item):
        print("Already verified: " + target.name, flush=True)
        return
    partial = target.with_name(target.name + ".download")
    expected = item["size_bytes"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected:
        if matches(partial, item):
            partial.replace(target)
            print("Verified: " + target.name, flush=True)
            return
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "emotion-recognition-wheel-downloader/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    print("Downloading {} ({:,}/{:,} bytes)".format(target.name, offset, expected), flush=True)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.getcode()
        if status == 206:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if (not content_range or int(content_range[1]) != offset or int(content_range[3]) != expected
                    or not offset <= int(content_range[2]) < expected):
                raise ValueError("Invalid Content-Range for " + target.name)
        elif status == 200:
            # Some servers ignore Range; safely restart instead of appending a second full copy.
            offset = 0
        else:
            raise ValueError("Unexpected HTTP status: " + str(status))
        last_report = offset
        with partial.open("ab" if offset else "wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                offset += len(chunk)
                if offset > expected:
                    raise ValueError("Download exceeds expected size: " + target.name)
                stream.write(chunk)
                if offset - last_report >= 32 * 1024 * 1024:
                    print("  {:.0%} ({:,}/{:,} bytes)".format(offset / expected, offset, expected), flush=True)
                    last_report = offset
    if not matches(partial, item):
        if partial.stat().st_size == expected:
            partial.unlink()  # A complete corrupt download must not be resumed.
        raise ValueError("Size or SHA-256 mismatch: " + target.name)
    partial.replace(target)
    print("Verified: " + target.name, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("release", "upstream"), default="release",
                        help="download from GitHub Release (default) or the official PyTorch source")
    parser.add_argument("--verify-only", action="store_true", help="verify local wheels without network access")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema")
        valid = True
        for item in manifest["files"]:
            filename = item["filename"]
            if Path(filename).name != filename or not filename.endswith(".whl"):
                raise ValueError("Invalid wheel filename in manifest")
            if args.verify_only:
                found = matches(ROOT / filename, item)
                print(("OK: " if found else "MISSING OR INVALID: ") + filename)
                valid = found and valid
            else:
                url = item["url"] if args.source == "upstream" else "{}/{}".format(
                    manifest["release_url"].replace("/tag/", "/download/"),
                    urllib.parse.quote(filename, safe=""))
                download(item, url)
        return 0 if valid else 1
    except (OSError, ValueError, KeyError, urllib.error.URLError, http.client.HTTPException) as exc:
        print("ERROR: {}. Rerun to resume, or try --source upstream.".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
