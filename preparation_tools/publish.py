#!/usr/bin/env python3
"""Publish the segmenter's wheels and checkpoint to their pinned GitHub Release (maintainers; needs GitHub CLI).

Run where all files are present and verified (every platform's wheels and the checkpoint; download.py with
PLATFORM = "all" fetches them). The repository, tag and part size are pinned in download.py, so the two scripts
always agree. Steps:
  1. verify every local file against the manifests; nothing on GitHub is touched before this passes;
  2. find the Release; if it does not exist, create it as a draft (the exact command is printed first);
  3. stop if an existing asset has the same name but other bytes: assets are immutable;
  4. upload the missing wheels and checkpoint (files above 1.5 GiB as <name>.part000, .part001, ...);
  5. upload the metadata last: SHA256SUMS, both manifests and LICENSES.md (only these may be replaced);
  6. check every asset's size and SHA-256 against GitHub's digests, then publish the draft.
Rerunning is safe: assets already verified on GitHub are reused. Needs `gh auth login` with write access.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

import download

# --- settings (edit here; no command-line flags) ---
DRY_RUN = False   # True: verify the local files and print the plan; GitHub is only read, never changed
TARGET = "main"   # branch the new tag points at (used only when the Release is created)
TITLE = "preparation_tools: turn segmenter wheels and checkpoint (v1)"
NOTES = ("Offline bundle for preparation_tools/ (turn segmenter for call-centre audio): pinned wheels for CPython 3.11 "
         "on Linux x86_64 and macOS arm64, and the torchaudio WAV2VEC2_ASR_BASE_960H checkpoint, unchanged from PyPI "
         "and download.pytorch.org. Restore them with preparation_tools/download.py, which checks every size and "
         "SHA-256 against the manifests in Git (copies attached here). Assets are immutable; a changed file gets a "
         "new tag. Licences: LICENSES.md.")
GH = "gh"         # GitHub CLI executable

REPOSITORY, TAG = download.REPOSITORY, download.RELEASE_TAG
METADATA = {"SHA256SUMS": None, "dependencies-manifest.json": "dependencies/manifest.json",
            "model_weights-manifest.json": "model_weights/manifest.json", "LICENSES.md": "LICENSES.md"}


def local_assets(staging):
    """Step 1: [(asset name, local file, size, SHA-256)] for the payload and for the metadata, all verified."""
    deps, weights = download.load_manifests()
    files = download.planned_files(deps, weights, list(deps["platforms"]), with_weights=True)
    payload = []
    for entry in files:
        print(f"  {entry['target'].relative_to(download.ROOT)} ({entry['size']:,} bytes)", flush=True)
        if not download.verified(entry["target"], entry["size"], entry["sha256"]):
            raise ValueError(f"{entry['target']} is missing or differs from its manifest; run download.py with "
                             "PLATFORM = \"all\" first")
        parts = download.release_parts(entry["name"], entry["size"])
        if len(parts) == 1:
            payload.append((entry["name"], entry["target"], entry["size"], entry["sha256"]))
            continue
        with entry["target"].open("rb") as source:   # above PART_SIZE: write the parts to the staging folder
            for asset, _, length in parts:
                with (staging / asset).open("wb") as part:
                    remaining = length
                    while remaining:
                        chunk = source.read(min(download.CHUNK, remaining))
                        part.write(chunk)
                        remaining -= len(chunk)
                payload.append((asset, staging / asset, length, download.sha256_of(staging / asset)))
                print(f"    part {asset} ({length:,} bytes)", flush=True)
    names = [name for name, _, _, _ in payload] + list(METADATA)
    if len(set(names)) != len(names):
        raise ValueError("Release asset names are not unique")
    (staging / "SHA256SUMS").write_text("".join(f"{sha}  {name}\n" for name, _, _, sha in payload), encoding="utf-8")
    for name, source in METADATA.items():
        if source:
            shutil.copyfile(download.ROOT / source, staging / name)
    metadata = [(name, staging / name, (staging / name).stat().st_size, download.sha256_of(staging / name))
                for name in METADATA]
    return payload, metadata


def release_state():
    """(is_draft, {asset name: asset}) of the Release, or None if it does not exist. Read-only."""
    result = subprocess.run([GH, "release", "view", TAG, "--repo", REPOSITORY, "--json", "tagName,isDraft,assets"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        if "not found" in result.stderr.lower():
            return None
        raise ValueError("gh release view failed: " + result.stderr.strip())
    release = json.loads(result.stdout)
    if release.get("tagName") != TAG:
        raise ValueError(f"gh returned Release {release.get('tagName')!r} instead of {TAG}")
    assets = {}
    for asset in release["assets"]:
        if asset["name"] in assets:
            raise ValueError("asset name repeated on GitHub: " + asset["name"])
        assets[asset["name"]] = asset
    return release["isDraft"], assets


def same(asset, size, sha256):
    return (asset.get("state") == "uploaded" and asset.get("size") == size
            and asset.get("digest") == "sha256:" + sha256)


def check_remote(entries, assets):
    bad = [name for name, _, size, sha in entries if name not in assets or not same(assets[name], size, sha)]
    if bad:
        raise ValueError("missing on GitHub, or size/SHA-256 digest differs: " + ", ".join(bad))


def gh(*args):
    """Run one GitHub CLI command that changes the Release (printed first)."""
    print("    $ " + shlex.join([GH, *args]), flush=True)
    subprocess.run([GH, *args], check=True)


def publish():
    if os.environ.get("GH_REPO", REPOSITORY) != REPOSITORY:
        raise ValueError("GH_REPO points at another repository; unset it")
    url = f"https://github.com/{REPOSITORY}/releases/tag/{TAG}"
    print(f"Release {url}" + ("  (DRY RUN: GitHub is only read)" if DRY_RUN else ""), flush=True)
    with tempfile.TemporaryDirectory(prefix="preparation-tools-release-") as folder:
        print("Step 1: verify every local file against the manifests", flush=True)
        payload, metadata = local_assets(Path(folder))
        print(f"  {len(payload)} files ({sum(e[2] for e in payload):,} bytes) and {len(metadata)} metadata files "
              "verified", flush=True)

        print("Step 2: find the Release", flush=True)
        state = release_state()
        if state is None:
            create = ["release", "create", TAG, "--repo", REPOSITORY, "--draft", "--latest=false",
                      "--target", TARGET, "--title", TITLE, "--notes", NOTES]
            print(f"  not found: creating it as a DRAFT, tag {TAG} on {TARGET}; it is published only in step 6, "
                  "after every asset is verified on GitHub", flush=True)
            if DRY_RUN:
                print("    would run: " + shlex.join([GH, *create]), flush=True)
                state = (True, {})
            else:
                gh(*create)
                state = release_state()
                if state is None:
                    raise ValueError("the new draft Release cannot be found")
        draft, assets = state
        print(f"  {'draft' if draft else 'published'}, {len(assets)} assets", flush=True)

        print("Step 3: compare existing assets (immutable: the same name must hold the same bytes)", flush=True)
        conflicts = [f"{name} (GitHub: {assets[name].get('state')}, {assets[name].get('size')} bytes, "
                     f"{assets[name].get('digest')})" for name, _, size, sha in payload
                     if name in assets and not same(assets[name], size, sha)]
        if conflicts:
            raise ValueError("these Release assets differ from the local files, so nothing was uploaded. A changed "
                             "file needs a new tag (RELEASE_TAG in download.py): " + "; ".join(conflicts))
        extra = sorted(set(assets) - {name for name, _, _, _ in payload + metadata})
        if extra:
            print("  not managed by publish.py, left unchanged: " + ", ".join(extra), flush=True)
        missing = [entry for entry in payload if entry[0] not in assets]
        changed = [entry for entry in metadata if entry[0] not in assets or not same(assets[entry[0]], *entry[2:])]
        print(f"  {len(payload) - len(missing)} files already on GitHub and verified, {len(missing)} to upload; "
              f"metadata to upload: {', '.join(e[0] for e in changed) or 'none'}", flush=True)
        if DRY_RUN:
            for name, _, size, _ in missing + changed:
                print(f"    would upload {name} ({size:,} bytes)", flush=True)
            print("DRY RUN finished; set DRY_RUN = False to publish.", flush=True)
            return

        print("Step 4: upload the wheels and the checkpoint", flush=True)
        for name, path, size, _ in missing:
            print(f"  uploading {name} ({size:,} bytes)", flush=True)
            gh("release", "upload", TAG, str(path), "--repo", REPOSITORY)
        check_remote(payload, release_state()[1])

        print("Step 5: upload the metadata (last; only these four names may be replaced)", flush=True)
        for name, path, size, _ in changed:
            print(f"  {'replacing' if name in assets else 'uploading'} {name} ({size:,} bytes)", flush=True)
            gh("release", "upload", TAG, str(path), "--repo", REPOSITORY, "--clobber")

        print("Step 6: verify every asset on GitHub, then publish", flush=True)
        draft, assets = release_state()
        check_remote(payload + metadata, assets)
        print(f"  all {len(payload) + len(metadata)} assets match their size and SHA-256 on GitHub", flush=True)
        if draft:
            gh("release", "edit", TAG, "--repo", REPOSITORY, "--draft=false", "--latest=false")
            if release_state()[0]:
                raise ValueError("the Release is still a draft")
            print("  published", flush=True)
    print(f"Done: {url}", flush=True)


def main():
    try:
        publish()
        return 0
    except (OSError, ValueError, KeyError) as exc:   # OSError includes a missing `gh`
        print(f"ERROR: {exc}", file=sys.stderr)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed ({exc.returncode}): {shlex.join(map(str, exc.cmd))}. Rerun to reuse the "
              "assets that are already verified on GitHub.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted. Rerun to reuse the assets that are already verified on GitHub.", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
