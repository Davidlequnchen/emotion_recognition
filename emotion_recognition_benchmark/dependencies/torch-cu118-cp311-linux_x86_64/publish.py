#!/usr/bin/env python3
"""Mirror verified wheels to the existing dependency Release using GitHub CLI.

Wheel assets are immutable: existing names must have exactly the expected size
and SHA-256. Only the four named metadata assets may be replaced, after all
wheel assets have been uploaded and verified on GitHub.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPOSITORY = "Davidlequnchen/emotion_recognition"
RELEASE_TAG = "deps-torch-cu118-cp311-linux-x86_64-v1"
# Update the manifest last, after the complete wheel set and other metadata exist.
METADATA = ("SHA256SUMS", "requirements-offline.txt", "INSTALL_ORDER.md", "manifest.json")


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_assets(root, manifest):
    """Validate every local payload before any remote mutation."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if manifest.get("repository") != REPOSITORY or manifest.get("release_tag") != RELEASE_TAG:
        raise ValueError("This publisher is restricted to the pinned repository and Release tag")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("The manifest must contain wheel files")
    expected = {}
    wheels = []
    for item in files:
        name = item["filename"]
        if (not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
                or not name.endswith(".whl") or name in expected):
            raise ValueError("Invalid or repeated wheel filename: " + repr(name))
        expected[name] = (item["size_bytes"], item["sha256"])
        wheels.append(name)
    for name in METADATA:
        path = root / name
        expected[name] = (path.stat().st_size, file_digest(path))
    for name, (size, digest) in expected.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != size or file_digest(path) != digest:
            raise ValueError("Local size or SHA-256 mismatch: " + name)
    return wheels, expected


def release_assets():
    output = subprocess.check_output(
        ["gh", "release", "view", RELEASE_TAG, "--repo", REPOSITORY,
         "--json", "tagName,isDraft,assets"], text=True)
    release = json.loads(output)
    if release.get("tagName") != RELEASE_TAG or release.get("isDraft"):
        raise ValueError("Expected the existing published dependency Release")
    assets = {}
    for asset in release["assets"]:
        if asset["name"] in assets:
            raise ValueError("Repeated remote asset name: " + asset["name"])
        assets[asset["name"]] = asset
    return assets


def asset_matches(asset, expected):
    size, digest = expected
    return (asset.get("state") == "uploaded" and asset.get("size") == size
            and asset.get("digest") == "sha256:" + digest)


def verify_remote(names, expected, assets):
    for name in names:
        if name not in assets or not asset_matches(assets[name], expected[name]):
            raise ValueError("Remote asset is missing or has different bytes: " + name)


def publish(root=ROOT):
    configured_repo = os.environ.get("GH_REPO", REPOSITORY)
    if configured_repo != REPOSITORY:
        raise ValueError("GH_REPO does not match the pinned repository")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    wheels, expected = expected_assets(root, manifest)
    assets = release_assets()
    # Reject immutable-name conflicts before uploading anything.
    verify_remote([name for name in wheels if name in assets], expected, assets)
    uploaded = 0
    for name in wheels:
        if name in assets:
            print("Reusing verified Release wheel: " + name, flush=True)
            continue
        print("Uploading new Release wheel: " + name, flush=True)
        subprocess.run(["gh", "release", "upload", RELEASE_TAG, str(root / name),
                        "--repo", REPOSITORY], check=True)
        uploaded += 1
    verify_remote(wheels, expected, release_assets())
    # The wheel set is now complete; replace only explicitly whitelisted metadata.
    assets = release_assets()
    for name in METADATA:
        if name in assets and asset_matches(assets[name], expected[name]):
            print("Reusing verified Release metadata: " + name, flush=True)
            continue
        print("Updating Release metadata: " + name, flush=True)
        subprocess.run(["gh", "release", "upload", RELEASE_TAG, str(root / name),
                        "--repo", REPOSITORY, "--clobber"], check=True)
    verify_remote(expected, expected, release_assets())
    print("Verified {} wheels and {} metadata assets on GitHub; {} new wheels uploaded.".format(
        len(wheels), len(METADATA), uploaded), flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as stream:
            stream.write("## Verified CUDA 11.8 wheel mirror\n\n")
            stream.write("- Release: https://github.com/{}/releases/tag/{}\n".format(REPOSITORY, RELEASE_TAG))
            stream.write("- {} wheels, {:,} bytes; {} new wheel assets uploaded.\n".format(
                len(wheels), sum(expected[name][0] for name in wheels), uploaded))
            stream.write("- All wheel and metadata sizes and SHA-256 values match GitHub's asset digests.\n")


def main():
    try:
        publish()
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun to reuse already verified Release assets.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
