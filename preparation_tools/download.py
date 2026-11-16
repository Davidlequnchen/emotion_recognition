#!/usr/bin/env python3
"""Download the segmenter's pinned wheels and alignment checkpoint (Python 3.9+, standard library only).

By default everything comes from this repository's GitHub Release RELEASE_TAG, which holds the files unchanged
from PyPI and download.pytorch.org (those upstream hosts remain selectable with SOURCE). It fetches what this
machine needs: dependencies/any/, the wheel folder of this platform and the checkpoint in model_weights/.
  - a file that already matches its manifest (size and SHA-256) is reused without any network access;
  - downloads go to <name>.download and resume from there after an interruption;
  - a file gets its final name only after its size and SHA-256 match dependencies/ or model_weights/manifest.json.
Private repository: set GH_TOKEN (or GITHUB_TOKEN), or log in once with `gh auth login`. The token is sent only
to the GitHub API, is never printed, and is dropped when GitHub redirects the download to its storage host.
Next step: python install.py (here, or after copying the whole preparation_tools/ folder to an offline machine).
"""
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from tls import certificate_error, certificate_error_message, create_ssl_context

# --- settings (edit here; no command-line flags; only the GitHub token comes from the environment) ---
SOURCE = "release"     # "release": the GitHub Release below; "upstream": PyPI and download.pytorch.org
PLATFORM = "auto"      # "auto" (this machine), "cp311-linux_x86_64", "cp311-macosx_arm64" or "all"
GET_WEIGHTS = True     # also fetch the wav2vec2 checkpoint (378 MB)
CA_BUNDLE = None       # company root CA file (PEM) if HTTPS verification fails; SSL_CERT_FILE works too
USE_GH_CLI = True      # without GH_TOKEN / GITHUB_TOKEN, ask the GitHub CLI for its token (`gh auth token`)

REPOSITORY = "Davidlequnchen/emotion_recognition"
RELEASE_TAG = "preparation-tools-segmenter-v1"
DOWNLOAD_URL = "https://github.com"   # anonymous: <DOWNLOAD_URL>/<repository>/releases/download/<tag>/<asset>
API_URL = "https://api.github.com"    # with a token: the GitHub REST API
PART_SIZE = 1536 * 1024 * 1024        # a file above 1.5 GiB is stored as <name>.part000, .part001, ... assets
ATTEMPTS = 3                          # tries per asset after network errors; every retry resumes
RETRY_WAIT_S = 5

ROOT = Path(__file__).resolve().parent
CHUNK = 1024 * 1024
USER_AGENT = "emotion-recognition-preparation-tools/1"
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")   # flat names that GitHub keeps unchanged as asset names


class DownloadError(Exception):
    """A file cannot be restored safely; retrying the same request would not help."""


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, size, sha256):
    return path.is_file() and path.stat().st_size == size and sha256_of(path) == sha256


def release_parts(name, size):
    """Release assets of one file as [(asset, first byte, length)]; above PART_SIZE the file is split in order."""
    if size <= PART_SIZE:
        return [(name, 0, size)]
    return [(f"{name}.part{index:03d}", start, min(PART_SIZE, size - start))
            for index, start in enumerate(range(0, size, PART_SIZE))]


def load_manifests():
    deps = json.loads((ROOT / "dependencies" / "manifest.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT / "model_weights" / "manifest.json").read_text(encoding="utf-8"))
    if deps.get("schema_version") != 1 or weights.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version (expected 1)")
    return deps, weights


def wheel_platforms(deps):
    """Wheel folders to fetch besides any/: this machine's, or the one set in PLATFORM."""
    known = list(deps["platforms"])
    if PLATFORM == "all":
        return known
    if PLATFORM != "auto":
        if PLATFORM not in known:
            raise ValueError(f"PLATFORM must be \"auto\", \"all\" or one of {known}; it is {PLATFORM!r}")
        return [PLATFORM]
    system, machine = platform.system(), platform.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return ["cp311-linux_x86_64"]
    if system == "Darwin" and machine == "arm64":
        return ["cp311-macosx_arm64"]
    raise ValueError(f"no wheel set for {system} {machine}; set PLATFORM at the top of download.py to one of "
                     f"{known} (or \"all\") and copy the folder to the target machine")


def planned_files(deps, weights, platforms, with_weights):
    """Files to restore: name (= Release asset), target path, size, SHA-256 and upstream URL. Validated first."""
    files = []
    for item in deps["files"]:
        if item["platform"] != "any" and item["platform"] not in deps["platforms"]:
            raise ValueError(f"unknown platform in dependencies/manifest.json: {item['platform']!r}")
        if item["platform"] == "any" or item["platform"] in platforms:
            files.append({"name": item["filename"], "folder": ROOT / "dependencies",
                          "target": ROOT / "dependencies" / item["platform"] / item["filename"],
                          "size": item["size_bytes"], "sha256": item["sha256"], "url": item["url"]})
    if with_weights:
        for model in weights["models"].values():
            files.append({"name": model["filename"], "folder": ROOT / "model_weights",
                          "target": ROOT / "model_weights" / model["path"],
                          "size": model["size_bytes"], "sha256": model["sha256"], "url": model["url"]})
    names = set()
    for entry in files:
        name, size = entry["name"], entry["size"]
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or entry["target"].name != name:
            raise ValueError(f"unsafe file name in a manifest: {name!r}")
        if not entry["target"].resolve().is_relative_to(entry["folder"].resolve()):
            raise ValueError(f"path leaves its folder: {entry['target']}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"invalid size for {name}")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"invalid SHA-256 for {name}")
        if name in names:
            raise ValueError(f"file name used twice (Release assets must be unique): {name}")
        names.add(name)
    return files


def github_token():
    """(token, where it came from) or (None, None). The token itself is never printed."""
    token, origin = None, None
    for variable in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(variable, "").strip():
            token, origin = os.environ[variable].strip(), variable
            break
    if token is None and USE_GH_CLI and shutil.which("gh"):
        try:
            result = subprocess.run(["gh", "auth", "token", "--hostname", "github.com"],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                token, origin = result.stdout.strip(), "`gh auth token`"
        except (OSError, subprocess.SubprocessError):
            pass
    if token is not None and any(ord(char) <= 32 or ord(char) >= 127 for char in token):
        raise ValueError(f"the GitHub token from {origin} contains spaces or non-ASCII characters")
    return token, origin


class DropAuthorizationOnRedirect(urllib.request.HTTPRedirectHandler):
    """GitHub answers an asset request with a redirect to its storage host, which must not receive the token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")   # second guard: urllib already drops unredirected headers
        return redirected


class Release:
    """Requests for Release assets: anonymous download URLs, or the REST API when a token is available."""

    def __init__(self, opener, token):
        self.opener, self.token, self.assets = opener, token, None

    def api_request(self, url, accept, headers=None):
        request = urllib.request.Request(url, headers={**(headers or {}), "User-Agent": USER_AGENT,
                                                       "Accept": accept, "X-GitHub-Api-Version": "2022-11-28"})
        request.add_unredirected_header("Authorization", "Bearer " + self.token)   # never sent after a redirect
        return request

    def request(self, asset, size, headers):
        quote = urllib.parse.quote
        if not self.token:
            return urllib.request.Request(f"{DOWNLOAD_URL}/{REPOSITORY}/releases/download/"
                                          f"{quote(RELEASE_TAG, safe='')}/{quote(asset, safe='')}", headers=headers)
        if self.assets is None:   # one API call per run, only when something has to be downloaded
            print(f"    GitHub API: reading the asset list of {RELEASE_TAG}", flush=True)
            url = f"{API_URL}/repos/{REPOSITORY}/releases/tags/{quote(RELEASE_TAG, safe='')}"
            with self.opener.open(self.api_request(url, "application/vnd.github+json"), timeout=60) as response:
                self.assets = {item["name"]: item for item in json.load(response).get("assets", [])}
            print(f"    GitHub API: {len(self.assets)} assets", flush=True)
        item = self.assets.get(asset)
        if item is None:
            raise DownloadError(f"{asset} is not in Release {RELEASE_TAG}; it is incomplete (see publish.py)")
        if item.get("size") != size:
            raise DownloadError(f"{asset} has {item.get('size')} bytes on the Release, the manifest says {size:,}")
        if not isinstance(item.get("id"), int) or isinstance(item["id"], bool):
            raise DownloadError(f"invalid asset id for {asset}")
        return self.api_request(f"{API_URL}/repos/{REPOSITORY}/releases/assets/{item['id']}",
                                "application/octet-stream", headers)


def fetch_part(opener, request_for, asset, partial, start, end, total):
    """Append bytes [current size of partial, end) of a file; this asset holds bytes [start, end) of it."""
    for attempt in range(1, ATTEMPTS + 1):
        done = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if done > start:
            headers["Range"] = f"bytes={done - start}-"
        try:
            with opener.open(request_for(asset, end - start, headers), timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != done - start or int(match[3]) != end - start:
                        raise DownloadError(f"invalid Content-Range for {asset}")
                elif response.status == 200:
                    if done > start:   # the server ignored Range: write this asset again from its first byte
                        with partial.open("r+b") as stream:
                            stream.truncate(start)
                        done = start
                else:
                    raise DownloadError(f"unexpected HTTP status {response.status} for {asset}")
                last_report = done
                with partial.open("ab") as stream:
                    for chunk in iter(lambda: response.read(CHUNK), b""):
                        if done + len(chunk) > end:
                            raise DownloadError(f"{asset} is larger than the manifest says")
                        stream.write(chunk)
                        done += len(chunk)
                        if done - last_report >= 32 * CHUNK:
                            print(f"    {done / total:.0%} ({done:,}/{total:,} bytes)", flush=True)
                            last_report = done
            if done != end:
                raise ConnectionError(f"connection closed after {done:,} of {end:,} bytes")
            return done
        except urllib.error.HTTPError as exc:
            if (exc.code not in (408, 429) and exc.code < 500) or attempt == ATTEMPTS:
                raise
            error = f"HTTP {exc.code}"
        except DownloadError:
            raise
        except (OSError, http.client.HTTPException) as exc:   # includes URLError, timeouts and early closes
            if certificate_error(exc) or attempt == ATTEMPTS:
                raise
            error = str(exc)
        print(f"    attempt {attempt}/{ATTEMPTS} failed ({error}); resuming in {RETRY_WAIT_S} s", flush=True)
        time.sleep(RETRY_WAIT_S)


def restore(entry, opener, request_for, split):
    """Download one file unless it is already verified. Returns True when it was downloaded."""
    target, size, sha256 = entry["target"], entry["size"], entry["sha256"]
    shown = target.relative_to(ROOT)
    if verified(target, size, sha256):
        print(f"  already verified: {shown}", flush=True)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    done = partial.stat().st_size if partial.exists() else 0
    if done > size or (done == size and not verified(partial, size, sha256)):   # too long or wrong: start again
        partial.unlink()
        done = 0
    print(f"  {'resuming' if done else 'downloading'} {shown} ({done:,}/{size:,} bytes)", flush=True)
    parts = release_parts(entry["name"], size) if split else [(entry["name"], 0, size)]
    for asset, start, length in parts:
        if done < start + length:
            done = fetch_part(opener, request_for, asset, partial, start, start + length, size)
    if not verified(partial, size, sha256):
        partial.unlink()   # complete but wrong bytes must never be resumed
        raise DownloadError(f"SHA-256 mismatch for {target.name}; the download was deleted (run again to retry)")
    partial.replace(target)
    print(f"  verified: {shown}", flush=True)
    return True


def restore_all(info):
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}; folder {ROOT}", flush=True)

    print("Step 1: read the manifests", flush=True)
    deps, weights = load_manifests()
    platforms = wheel_platforms(deps)
    files = planned_files(deps, weights, platforms, GET_WEIGHTS)
    wheels = [entry for entry in files if entry["name"].endswith(".whl")]
    print(f"  {len(wheels)} wheels (any + {', '.join(platforms)}), {len(files) - len(wheels)} checkpoint(s), "
          f"{sum(entry['size'] for entry in files) / 1e6:.1f} MB in total", flush=True)

    print("Step 2: choose the source", flush=True)
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=create_ssl_context(CA_BUNDLE)),
                                         DropAuthorizationOnRedirect())
    if SOURCE == "release":
        token, info["token_origin"] = github_token()
        request_for = Release(opener, token).request
        how = (f"through the GitHub API with the token from {info['token_origin']} (not shown)" if token else
               "anonymously (works for a public repository; a private one needs GH_TOKEN)")
        print(f"  GitHub Release {RELEASE_TAG} of {REPOSITORY}, {how}", flush=True)
    elif SOURCE == "upstream":
        urls = {entry["name"]: entry["url"] for entry in files}

        def request_for(asset, size, headers):
            return urllib.request.Request(urls[asset], headers=headers)
        print("  upstream: PyPI and download.pytorch.org (URLs from the manifests)", flush=True)
    else:
        raise ValueError(f"SOURCE must be \"release\" or \"upstream\"; it is {SOURCE!r}")

    print(f"Step 3: download and verify {len(files)} files", flush=True)
    downloaded = 0
    for index, entry in enumerate(files, 1):
        print(f"[{index}/{len(files)}] {entry['name']} ({entry['size'] / 1e6:.1f} MB)", flush=True)
        downloaded += restore(entry, opener, request_for, split=SOURCE == "release")
    print(f"Done: {downloaded} downloaded, {len(files) - downloaded} already verified; every file matches its "
          "size and SHA-256.", flush=True)
    print("Next: python install.py (here, or after copying the whole preparation_tools/ folder)", flush=True)
    return 0


def main():
    info = {"token_origin": None}
    try:
        return restore_all(info)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404) and info["token_origin"]:
            hint = (f"the token from {info['token_origin']} cannot read Release {RELEASE_TAG} of {REPOSITORY}: it "
                    "needs read access to the repository (fine-grained token: Contents read-only; classic: repo), "
                    "or the Release is not published yet (publish.py)")
        elif exc.code in (401, 403, 404) and SOURCE == "release":
            hint = (f"if {REPOSITORY} is private, set GH_TOKEN (or GITHUB_TOKEN) or run `gh auth login`; "
                    f"otherwise check that Release {RELEASE_TAG} exists (publish.py)")
        else:
            hint = "run the script again to resume"
        print(f"ERROR: HTTP {exc.code} {exc.reason}: {hint}.", file=sys.stderr)
    except (OSError, ValueError, KeyError, DownloadError, http.client.HTTPException) as exc:
        if certificate_error(exc):
            print(f"ERROR: {certificate_error_message(exc)}", file=sys.stderr)
        else:
            print(f"ERROR: {exc}. Run the script again to resume.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted. Run the same command again to resume.", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
