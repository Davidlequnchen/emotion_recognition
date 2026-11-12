"""Tiny wheel fixtures for the complete offline runtime bundle audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The data/model repositories also have modules called tls and download.
tls = load_module("wheel_bundle_test_tls", ROOT / "tls.py")
with mock.patch.dict(sys.modules, {"tls": tls}):
    downloader = load_module("wheel_bundle_test_download", ROOT / "download.py")
with mock.patch.dict(sys.modules, {"download": downloader}):
    auditor = load_module("wheel_bundle_test_audit", ROOT / "audit.py")


class BundleAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tiny-wheel-bundle-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def wheel(self, name, version="1.0", requirements=(), python=">=3.9", tag="py3-none-any", vendored=False):
        distribution = name.replace("-", "_")
        filename = "{}-{}-{}.whl".format(distribution, version, tag)
        metadata = "Metadata-Version: 2.1\nName: {}\nVersion: {}\n".format(name, version)
        if python:
            metadata += "Requires-Python: {}\n".format(python)
        metadata += "".join("Requires-Dist: {}\n".format(value) for value in requirements)
        path = self.root / filename
        with ZipFile(path, "w") as wheel:
            directory = "{}-{}.dist-info/".format(distribution, version)
            wheel.writestr(directory + "METADATA", metadata + "\n")
            wheel.writestr(directory + "WHEEL", "Wheel-Version: 1.0\nTag: {}\n".format(tag))
            if vendored:
                wheel.writestr("setuptools/_vendor/foreign-9.9.dist-info/METADATA",
                               "Metadata-Version: 2.1\nName: foreign\nVersion: 9.9\n"
                               "Requires-Dist: should-not-be-installed>=123\n\n")
        body = path.read_bytes()
        return {"package": name, "version": version, "filename": filename,
                "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}

    def manifest(self, files, roots=None):
        return {"schema_version": 1, "root_packages": roots if roots is not None else
                [item["package"] for item in files], "files": files}

    def test_linux_marker_dependencies_are_required_on_any_host(self):
        consumer = self.wheel("consumer", requirements=[
            'linux-runtime>=1; platform_system == "Linux" and platform_machine == "x86_64"',
            'macos-runtime; sys_platform == "darwin"',
        ])
        with self.assertRaisesRegex(ValueError, "missing dependency linux-runtime"):
            auditor.audit(self.root, self.manifest([consumer]))
        runtime = self.wheel("linux-runtime")
        ordered = auditor.audit(self.root, self.manifest([runtime, consumer]))
        self.assertEqual(ordered[-1]["requires"], ["linux-runtime"])

    def test_dependency_version_must_satisfy_metadata(self):
        dependency = self.wheel("dependency", version="1.0")
        consumer = self.wheel("consumer", requirements=["dependency>=2"])
        with self.assertRaisesRegex(ValueError, "requires dependency>=2, found 1.0"):
            auditor.audit(self.root, self.manifest([dependency, consumer]))

    def test_dependency_must_precede_consumer(self):
        dependency = self.wheel("dependency")
        consumer = self.wheel("consumer", requirements=["dependency"])
        manifest = self.manifest([consumer, dependency], roots=["consumer"])
        with self.assertRaisesRegex(ValueError, "not dependency-first"):
            auditor.audit(self.root, manifest)
        ordered = auditor.audit(self.root, manifest, check_order=False)
        self.assertEqual([item["package"] for item in ordered], ["dependency", "consumer"])

    def test_bad_sha256_is_rejected_before_metadata_is_trusted(self):
        item = self.wheel("consumer")
        item["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "size/SHA-256 mismatch"):
            auditor.audit(self.root, self.manifest([item]))

    def test_requires_python_must_allow_target_python_311(self):
        item = self.wheel("consumer", python=">=3.12")
        with self.assertRaisesRegex(ValueError, "does not support Python 3.11"):
            auditor.audit(self.root, self.manifest([item]))

    def test_wheel_tags_must_support_target_abi_architecture_and_glibc(self):
        for tag in ("cp312-cp312-manylinux_2_28_x86_64", "cp311-cp311-manylinux_2_28_aarch64",
                    "cp311-cp311-manylinux_2_31_x86_64"):
            with self.subTest(tag=tag):
                item = self.wheel("consumer", tag=tag)
                with self.assertRaisesRegex(ValueError, "target ABI mismatch"):
                    auditor.audit(self.root, self.manifest([item]))

    def test_correct_closure_writes_all_four_metadata_files(self):
        dependency = self.wheel("dependency")
        consumer = self.wheel("consumer", requirements=["dependency>=1"])
        manifest = self.manifest([consumer, dependency], roots=["consumer"])
        ordered = auditor.audit(self.root, manifest, check_order=False)
        auditor.write_lock(self.root, manifest, ordered)
        for filename in ("manifest.json", "SHA256SUMS", "requirements-offline.txt", "INSTALL_ORDER.md"):
            self.assertTrue((self.root / filename).is_file(), filename)
        saved = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual([item["package"] for item in saved["files"]], ["dependency", "consumer"])
        self.assertEqual(saved["files"][1]["requires"], ["dependency"])
        self.assertEqual(auditor.audit(self.root, saved), ordered)
        sums = (self.root / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(sums, ["{}  {}".format(item["sha256"], item["filename"]) for item in ordered])
        requirements = (self.root / "requirements-offline.txt").read_text()
        for item in ordered:
            self.assertIn("{}=={} --hash=sha256:{}".format(item["package"], item["version"], item["sha256"]),
                          requirements)
        instructions = (self.root / "INSTALL_ORDER.md").read_text()
        dependency_command = "pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/" + dependency["filename"]
        consumer_command = "pip install --no-index --no-deps ./dependencies/torch-cu118-cp311-linux_x86_64/" + consumer["filename"]
        self.assertLess(instructions.index(dependency_command), instructions.index(consumer_command))
        self.assertIn("python -m pip check", instructions)

    def test_setuptools_vendored_metadata_is_not_another_distribution(self):
        item = self.wheel("setuptools", vendored=True)
        ordered = auditor.audit(self.root, self.manifest([item]))
        self.assertEqual(len(ordered), 1)
        self.assertEqual(ordered[0]["requires"], [])

    def test_declared_root_package_must_be_present_even_if_other_files_are_closed(self):
        item = self.wheel("dependency")
        with self.assertRaisesRegex(ValueError, "root"):
            auditor.audit(self.root, self.manifest([item], roots=["missing-root"]))

    def test_requested_dependency_extra_adds_its_transitive_requirements(self):
        consumer = self.wheel("consumer", requirements=["dependency[feature]>=1"])
        dependency = self.wheel("dependency", requirements=['feature-runtime; extra == "feature"'])
        manifest = self.manifest([dependency, consumer], roots=["consumer"])
        with self.assertRaisesRegex(ValueError, "missing dependency feature-runtime"):
            auditor.audit(self.root, manifest)
        runtime = self.wheel("feature-runtime")
        ordered = auditor.audit(self.root, self.manifest([runtime, dependency, consumer], roots=["consumer"]))
        self.assertEqual([item["package"] for item in ordered], ["feature-runtime", "dependency", "consumer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
