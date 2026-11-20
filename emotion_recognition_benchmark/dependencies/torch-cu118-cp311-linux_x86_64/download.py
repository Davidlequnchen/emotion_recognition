#!/usr/bin/env python3
"""Restore the pinned wheels to their wheel directory (Python 3.9+).

Existing verified wheels are reused. Interrupted downloads resume on rerun;
files are renamed to .whl only after their size and SHA-256 match the manifest.
Optional truststore support uses the operating system's certificate trust.
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

from tls import create_ssl_context, certificate_error, certificate_error_message


ROOT = Path(__file__).resolve().parent


class DownloadResponseError(ValueError):
    """The remote server did not return the expected wheel bytes."""


def matches(path, item):
    if not path.is_file() or path.stat().st_size != item["size_bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


def download(item, url, root=ROOT, context=None, *, resume=True):
    target = root / item["filename"]
    if matches(target, item):
        print("Already verified: " + target.name, flush=True)
        return
    partial = target.with_name(target.name + ".download")
    expected = item["size_bytes"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= expected or (offset and not resume):
        if matches(partial, item):
            partial.replace(target)
            print("Verified: " + target.name, flush=True)
            return
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "emotion-recognition-wheel-downloader/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = "bytes={}-".format(offset)
    host = urllib.parse.urlsplit(url).hostname
    print(f"Downloading {target.name} from {host} ({offset:,}/{expected:,} bytes)", flush=True)
    request = urllib.request.Request(url, headers=headers)
    if context is None:
        context = create_ssl_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        status = response.getcode()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type in {"text/html", "application/xhtml+xml", "application/json", "application/problem+json"}:
            raise DownloadResponseError(f"{host} returned HTTP {status} {content_type} for {target.name}")
        end = expected
        if status == 206:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if (not content_range or int(content_range[1]) != offset or int(content_range[3]) != expected
                    or not offset <= int(content_range[2]) < expected):
                raise DownloadResponseError("Invalid Content-Range for " + target.name)
            end = int(content_range[2]) + 1
        elif status == 200:
            # Some servers ignore Range; safely restart instead of appending a second full copy.
            offset = 0
        else:
            raise DownloadResponseError("Unexpected HTTP status: " + str(status))
        length = response.headers.get("Content-Length")
        if length and length.isdecimal() and int(length) != end - offset:
            raise DownloadResponseError(
                f"{host} advertised {int(length):,} bytes for {target.name}; expected {end - offset:,}")
        start = last_report = offset
        with partial.open("ab" if offset else "wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                offset += len(chunk)
                if offset > end:
                    raise DownloadResponseError(f"{host} returned more bytes than expected for {target.name}")
                stream.write(chunk)
                if offset - last_report >= 32 * 1024 * 1024:
                    print("  {:.0%} ({:,}/{:,} bytes)".format(offset / expected, offset, expected), flush=True)
                    last_report = offset
    if offset != expected:
        raise DownloadResponseError(f"Incomplete download for {target.name} from {host}: {offset:,}/{expected:,} bytes")
    if not matches(partial, item):
        partial.unlink()  # A complete corrupt download must not be resumed.
        if start:
            print(f"Checksum mismatch after resuming {target.name}. Retrying from byte 0.", flush=True)
            return download(item, url, root=root, context=context, resume=False)
        raise DownloadResponseError(f"SHA-256 mismatch for {target.name} from {host} ({expected:,} bytes)")
    partial.replace(target)
    print("Verified: " + target.name, flush=True)


def release_asset(release_url, filename):
    return "{}/{}".format(release_url.replace("/tag/", "/download/"), urllib.parse.quote(filename, safe=""))


def main(argv=None, *, root=None, default_source="release"):
    root = ROOT if root is None else root
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("release", "upstream"), default=default_source,
                        help="download from GitHub Release or the official sources (default: %(default)s)")
    parser.add_argument("--verify-only", action="store_true", help="verify local wheels without network access")
    parser.add_argument("--ca-bundle", type=Path,
                        help="company/custom CA PEM bundle; otherwise SSL_CERT_FILE or system/default roots")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported manifest schema")
        if not args.verify_only and args.source == "release" and not manifest.get("release_url"):
            raise ValueError("No Release is configured for this wheel set; use --source upstream")
        context = None if args.verify_only else create_ssl_context(args.ca_bundle)
        valid = True
        for item in manifest["files"]:
            filename = item["filename"]
            if Path(filename).name != filename or not filename.endswith(".whl"):
                raise ValueError("Invalid wheel filename in manifest")
            if args.verify_only:
                found = matches(root / filename, item)
                print(("OK: " if found else "MISSING OR INVALID: ") + filename)
                valid = found and valid
            else:
                url = release_asset(manifest["release_url"], filename) if args.source == "release" else item["url"]
                download(item, url, root=root, context=context)
        return 0 if valid else 1
    except (OSError, ValueError, KeyError, urllib.error.URLError, http.client.HTTPException) as exc:
        if certificate_error(exc):
            message = certificate_error_message(exc)
        else:
            message = "{}. Rerun to resume.".format(exc)
            if args.source == "release":
                message += " For the official URLs, use --source upstream."
        print("ERROR: " + message, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
