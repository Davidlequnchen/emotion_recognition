#!/usr/bin/env python3
"""Install the segmenter's dependencies offline, from the verified wheels in dependencies/ (no package index).

Run inside the target Python 3.11 environment (the one that has torch 2.7.1 and torchaudio 2.7.1):
  1. check Python, platform and the torch/torchaudio prerequisites (they are not duplicated here);
  2. verify the size and SHA-256 of every wheel and of the alignment checkpoint before installing anything;
  3. check that every wheel's own dependencies (METADATA) are covered by this set or the prerequisites;
  4. pip install --no-index --no-deps, one wheel at a time, dependencies first;
  5. pip check;
  6. smoke test in a fresh interpreter: Silero VAD, the offline wav2vec2 checkpoint, WAV and MP3 decoding.
"""
from email.parser import Parser
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from zipfile import ZipFile

# --- settings (edit here; no command-line flags) ---
RUN_TORCH_INSTALLER = False   # Linux: if torch/torchaudio are missing, run the repository's CUDA 11.8 installer first

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TORCH_INSTALLER = REPO / "emotion_recognition" / "dependencies" / "torch-cu118-cp311-linux_x86_64" / "install.py"
MAC_REQUIREMENTS = REPO / "emotion_recognition" / "requirements-llm-mac.txt"


def platform_key():
    """Wheel folder for this machine."""
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 11):
        raise ValueError(f"CPython 3.11 is required; this is {platform.python_implementation()} "
                         f"{platform.python_version()}.")
    system, machine = platform.system(), platform.machine()
    if system == "Linux" and machine == "x86_64":
        libc, version = platform.libc_ver()
        if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
            raise ValueError(f"glibc >= 2.28 is required (soundfile wheel); found {libc} {version}.")
        return "cp311-linux_x86_64"
    if system == "Darwin" and machine == "arm64":
        return "cp311-macosx_arm64"
    raise ValueError(f"No wheel set for {system} {machine}; available: Linux x86_64 and macOS arm64.")


def installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_prerequisites(manifest, key):
    """torch/torchaudio (and numpy, typing-extensions that come with them) must already be installed."""
    missing = []
    for name, wanted in manifest["prerequisites"].items():
        if name == "note":
            continue
        found = installed(name)
        ok = found is not None and (wanted == "any" or found.split("+")[0] == wanted)
        print(f"  {name}: {found or 'not installed'} (needed: {wanted}) {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(name)
    if not missing:
        return
    if key.startswith("cp311-linux") and RUN_TORCH_INSTALLER and TORCH_INSTALLER.is_file():
        print(f"  running {TORCH_INSTALLER}")
        subprocess.run([sys.executable, str(TORCH_INSTALLER)], check=True)
        return check_prerequisites(manifest, key)
    hint = (f"python {TORCH_INSTALLER}  (offline CUDA 11.8 wheelhouse of this repository; "
            "or set RUN_TORCH_INSTALLER = True at the top of install.py)" if key.startswith("cp311-linux")
            else f"python -m pip install -r {MAC_REQUIREMENTS}  (needs internet)")
    raise ValueError(f"Missing prerequisites {', '.join(missing)}. Install them first:\n  {hint}")


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(entries):
    """entries: (path, size, sha256). Everything is checked before the first pip command."""
    bad = []
    for path, size, sha256 in entries:
        if not path.is_file() or path.stat().st_size != size:
            bad.append(f"{path} (missing or wrong size)")
        elif sha256_of(path) != sha256:
            bad.append(f"{path} (SHA-256 mismatch)")
        else:
            print(f"  verified: {path.relative_to(ROOT)}")
    if bad:
        raise ValueError("Verification failed; nothing was installed. Run download.py on a machine with internet "
                         "and copy the files again:\n  " + "\n  ".join(bad))


def check_metadata(wheels, manifest):
    """Each wheel's Requires-Dist (without extras) must be an earlier wheel or an installed prerequisite."""
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:                                  # packaging is one of our wheels; pip vendors a copy
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.utils import canonicalize_name
    available = {canonicalize_name(n): installed(n) for n in manifest["prerequisites"] if n != "note"}
    for item, path in wheels:
        with ZipFile(path) as wheel:
            name = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA") and n.count("/") == 1)
            meta = Parser().parsestr(wheel.read(name).decode("utf-8"))
        if canonicalize_name(meta["Name"]) != canonicalize_name(item["package"]) or meta["Version"] != item["version"]:
            raise ValueError(f"{path.name}: METADATA does not match the manifest")
        for line in meta.get_all("Requires-Dist", []):
            req = Requirement(line)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue                                 # optional extras (onnxruntime, torchcodec, ...) are not used
            have = available.get(canonicalize_name(req.name))
            if have is None or not req.specifier.contains(have.split("+")[0], prereleases=True):
                raise ValueError(f"{item['package']} requires {req}, which is not covered (found {have})")
        available[canonicalize_name(item["package"])] = item["version"]
        print(f"  {item['package']} {item['version']}: dependencies covered")


def smoke_test():
    """Runs in a fresh interpreter after installation."""
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, str(ROOT))
    from segmenter import vad
    from segmenter.alignment import Aligner

    model = vad.load_vad()
    probs = vad.speech_probabilities(model, torch.zeros(16000))
    print(f"  Silero VAD (JIT from the wheel): {len(probs)} frames of 32 ms for 1 s of silence, "
          f"max speech probability {probs.max():.3f}")
    aligner = Aligner("wav2vec2_asr_base_960h", ROOT / "model_weights", "cpu")
    frames = aligner.emissions(torch.zeros(16000)).shape
    print(f"  wav2vec2 checkpoint loaded offline: {aligner.parameters / 1e6:.1f}M parameters, "
          f"{len(aligner.labels)} labels, emissions {tuple(frames)} for 1 s")
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    with tempfile.TemporaryDirectory() as folder:
        for fmt in ("WAV", "FLAC", "MP3"):
            path = Path(folder) / f"tone.{fmt.lower()}"
            sf.write(str(path), tone, 16000, format=fmt)
            data, rate = sf.read(str(path), dtype="float32")
            print(f"  soundfile {fmt}: wrote and decoded {len(data)} samples at {rate} Hz")
            if rate != 16000 or abs(len(data) - len(tone)) > 1152:
                raise RuntimeError(f"{fmt} round trip failed")
    print("  smoke test passed")


def main():
    print(f"Python {platform.python_version()} ({sys.executable}) on {platform.system()} {platform.machine()}")
    manifest = json.loads((ROOT / "dependencies" / "manifest.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT / "model_weights" / "manifest.json").read_text(encoding="utf-8"))

    print("Step 1: platform and prerequisites")
    key = platform_key()
    print(f"  wheel set: any + {key}")
    check_prerequisites(manifest, key)

    print("Step 2: verify wheels and checkpoint (nothing is installed if one fails)")
    wheels = [(item, ROOT / "dependencies" / item["platform"] / item["filename"])
              for item in manifest["files"] if item["platform"] in ("any", key)]
    checkpoints = [(ROOT / "model_weights" / m["path"], m["size_bytes"], m["sha256"])
                   for m in weights["models"].values()]
    verify_files([(path, item["size_bytes"], item["sha256"]) for item, path in wheels] + checkpoints)

    print("Step 3: dependency check from the wheels' METADATA")
    check_metadata(wheels, manifest)

    print(f"Step 4: install {len(wheels)} wheels offline")
    commands = [[sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(path)] for _, path in wheels]
    for index, command in enumerate(commands, 1):
        print(f"  {index}. {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)

    print("Step 5: pip check", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

    print("Step 6: smoke test", flush=True)
    subprocess.run([sys.executable, "-c", "import install; install.smoke_test()"], cwd=ROOT, check=True)
    print("Done. Next: put turns in local/input/turns/ and run  python segment_turns.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed ({exc.returncode}): {shlex.join(map(str, exc.cmd))}. "
              "Installation stopped; earlier steps stay installed.", file=sys.stderr)
        sys.exit(1)
