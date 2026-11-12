#!/usr/bin/env python3
"""Audit the complete Linux CPython 3.11 wheel closure and dependency-first install order.

Maintainers may use --write-lock to regenerate the ordered manifest, checksums,
offline requirements and manual installation list after changing the wheel set.
Uses packaging from pip when packaging is not installed separately; no network access.
"""
import argparse
from email.parser import Parser
from graphlib import TopologicalSorter
import json
from pathlib import Path
import sys
from zipfile import ZipFile, BadZipFile

try:
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import cpython_tags, compatible_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.tags import cpython_tags, compatible_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import Version

from download import matches

ROOT = Path(__file__).resolve().parent
TARGET = {
    "os_name": "posix", "sys_platform": "linux", "platform_machine": "x86_64",
    "platform_system": "Linux", "platform_release": "", "platform_version": "",
    "python_version": "3.11", "python_full_version": "3.11.0",
    "implementation_name": "cpython", "implementation_version": "3.11.0",
    "platform_python_implementation": "CPython", "extra": "",
}


def target_tags():
    platforms = []
    aliases = {17: "manylinux2014_x86_64", 12: "manylinux2010_x86_64", 5: "manylinux1_x86_64"}
    for minor in range(28, 4, -1):
        platforms.append("manylinux_2_{}_x86_64".format(minor))
        if minor in aliases:
            platforms.append(aliases[minor])
    platforms.append("linux_x86_64")
    return set(cpython_tags((3, 11), abis=["cp311"], platforms=platforms)) | set(
        compatible_tags((3, 11), interpreter="cp311", platforms=platforms))


def audit(root, manifest, check_order=True):
    records, metadata, requirements = {}, {}, {}
    allowed_tags = target_tags()
    for item in manifest["files"]:
        name = canonicalize_name(item["package"])
        filename = item["filename"]
        if Path(filename).name != filename or "\\" in filename or name in records:
            raise ValueError("Invalid or repeated wheel: " + filename)
        path = root / filename
        if not matches(path, item):
            raise ValueError("Missing wheel or size/SHA-256 mismatch: " + filename)
        wheel_name, wheel_version, _, tags = parse_wheel_filename(filename)
        if wheel_name != name or wheel_version != Version(item["version"]) or not tags & allowed_tags:
            raise ValueError("Wheel name, version or target ABI mismatch: " + filename)
        with ZipFile(path) as wheel:
            # setuptools vendors other distributions; only the wheel's top-level metadata is authoritative.
            entries = [entry for entry in wheel.namelist()
                       if entry.endswith(".dist-info/METADATA") and entry.count("/") == 1]
            if len(entries) != 1:
                raise ValueError("Expected exactly one METADATA: " + filename)
            message = Parser().parsestr(wheel.read(entries[0]).decode("utf-8"))
        if (canonicalize_name(message["Name"]) != name
                or Version(message["Version"]) != Version(item["version"])):
            raise ValueError("Wheel METADATA differs from manifest: " + filename)
        python = message.get("Requires-Python", "")
        if python and not SpecifierSet(python).contains(Version(TARGET["python_full_version"])):
            raise ValueError("Wheel does not support Python 3.11: " + filename)
        records[name], metadata[name] = item, message
        requirements[name] = [Requirement(value) for value in message.get_all("Requires-Dist", [])]

    roots = manifest.get("root_packages")
    if not isinstance(roots, list) or not roots:
        raise ValueError("Manifest must specify nonempty root_packages")
    for name in roots:
        if not isinstance(name, str) or canonicalize_name(name) not in records:
            raise ValueError("Missing root package: " + str(name))
    graph = {name: set() for name in sorted(records)}
    extras = {name: {""} for name in records}
    changed = True
    while changed:
        changed = False
        for name in graph:
            for req in requirements[name]:
                if req.marker and not any(req.marker.evaluate(dict(TARGET, extra=extra)) for extra in extras[name]):
                    continue
                dependency = canonicalize_name(req.name)
                if dependency not in records:
                    raise ValueError("{} requires missing dependency {}".format(name, req))
                if not req.specifier.contains(Version(records[dependency]["version"]), prereleases=True):
                    raise ValueError("{} requires {}, found {}".format(name, req, records[dependency]["version"]))
                graph[name].add(dependency)
                added = set(req.extras) - extras[dependency]
                if added:
                    extras[dependency].update(added)
                    changed = True
    order = list(TopologicalSorter(graph).static_order())
    if check_order:
        positions = {name: index for index, name in enumerate(records)}
        for name, dependencies in graph.items():
            if any(positions[dependency] >= positions[name] for dependency in dependencies):
                raise ValueError("Install order is not dependency-first for " + name)
            if "requires" in records[name] and set(records[name]["requires"]) != dependencies:
                raise ValueError("Recorded dependencies differ from METADATA for " + name)
    return [dict(records[name], requires=sorted(graph[name])) for name in order]


def write_lock(root, manifest, ordered):
    manifest["files"] = ordered
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "SHA256SUMS").write_text("".join(
        "{}  {}\n".format(item["sha256"], item["filename"]) for item in ordered), encoding="utf-8")
    (root / "requirements-offline.txt").write_text(
        "# Complete pinned CUDA 11.8 runtime closure for CPython 3.11 / Linux x86_64.\n"
        "# Includes truststore for the downloader. Install with --no-index --find-links and --require-hashes.\n"
        + "".join("{}=={} --hash=sha256:{}\n".format(item["package"], item["version"], item["sha256"])
                  for item in ordered), encoding="utf-8")
    lines = ["# 逐个离线安装顺序", "", "目标：CPython 3.11、Linux x86_64、glibc >= 2.28。",
             "以下顺序由 wheel METADATA 的实际依赖关系生成；每个包的依赖均先安装。",
             "所有 wheels 必须先下载到本目录并通过 SHA-256 校验。", "",
             "在 `emotion_recognition/` 项目目录、激活目标 Conda 环境后执行：", "", "```bash",
             "python dependencies/torch-cu118-cp311-linux_x86_64/download.py --verify-only",
             "python dependencies/torch-cu118-cp311-linux_x86_64/install.py", "```", "",
             "`install.py` 会先检查平台并验证整套文件，再逐个执行下列命令，最后运行 `pip check`。",
             "也可以按下面顺序手工逐行执行；`--no-index --no-deps` 保证每条安装命令只使用指定的本地 wheel。", "",
             "| 顺序 | 包 | 版本 | 先安装的直接依赖 |", "|---|---|---|---|"]
    for index, item in enumerate(ordered, 1):
        lines.append("| {} | {} | {} | {} |".format(index, item["package"], item["version"],
                                                    ", ".join(item["requires"]) or "—"))
    lines += ["", "## 可逐行复制的命令", "", "```bash"]
    for index, item in enumerate(ordered, 1):
        lines += ["# {}. {}=={}".format(index, item["package"], item["version"]),
                  "python -m pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/{}".format(item["filename"])]
    lines += ["python -m pip check", "```", "",
              "本集合覆盖 PyTorch/torchaudio/torchvision 的完整默认运行时依赖，并额外包含 truststore。",
              "项目 `requirements.txt` 中的 FunASR、transformers 等其他包及可选 extras 不在本集合内。", ""]
    (root / "INSTALL_ORDER.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-lock", action="store_true", help="regenerate locks and manual install order")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        ordered = audit(ROOT, manifest, check_order=not args.write_lock)
        if args.write_lock:
            write_lock(ROOT, manifest, ordered)
        print("Verified {} wheels: hashes, Python/ABI compatibility, full Linux dependency closure and install order.".format(len(ordered)))
        return 0
    except (OSError, ValueError, KeyError, BadZipFile) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
