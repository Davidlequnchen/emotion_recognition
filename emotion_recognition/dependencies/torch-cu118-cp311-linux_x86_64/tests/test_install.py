"""Installer preflight regression checks for mixed or truncated wheel directories."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wheel_install_guard_tests", ROOT / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def item(package, version="1.0", requires=()):
    return {
        "package": package, "version": version,
        "filename": "{}-{}-py3-none-any.whl".format(package.replace("-", "_"), version),
        "size_bytes": 10, "sha256": "f" * 64, "requires": list(requires),
    }


def complete_manifest():
    return {
        "schema_version": 1,
        "root_packages": ["torch", "torchaudio", "torchvision"],
        "files": [
            item("runtime"),
            item("torch", "2.7.1+cu118", ["runtime"]),
            item("torchaudio", "2.7.1+cu118", ["torch"]),
            item("torchvision", "0.22.1+cu118", ["torch"]),
        ],
    }


class InstallerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="wheel-installer-guard-")
        self.root = Path(self.temporary.name).resolve()
        output = mock.patch("sys.stdout", new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, manifest):
        (self.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_legacy_three_wheel_manifest_is_rejected_without_running_pip(self):
        legacy = complete_manifest()
        legacy.pop("root_packages")
        legacy["files"] = legacy["files"][1:]
        for entry in legacy["files"]:
            entry.pop("requires")
        self.save(legacy)
        for arguments in ([], ["--dry-run"]):
            with self.subTest(arguments=arguments):
                with mock.patch.object(installer, "ROOT", self.root), \
                        mock.patch.object(installer, "check_platform") as platform, \
                        mock.patch.object(installer, "verify_all") as hashes, \
                        mock.patch.object(installer, "verify_dependency_closure") as closure, \
                        mock.patch.object(installer.subprocess, "run") as pip, \
                        mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
                    result = installer.main(arguments)
                self.assertEqual(result, 1)
                platform.assert_not_called()
                hashes.assert_not_called()
                closure.assert_not_called()
                pip.assert_not_called()
                self.assertTrue(errors.getvalue())

    def test_truncated_declared_dependency_is_rejected(self):
        manifest = complete_manifest()
        manifest["files"] = manifest["files"][1:]
        self.save(manifest)
        with self.assertRaises(ValueError):
            installer.load_manifest(self.root)

    def test_dependency_must_precede_consumer(self):
        manifest = complete_manifest()
        manifest["files"][0], manifest["files"][1] = manifest["files"][1], manifest["files"][0]
        self.save(manifest)
        with self.assertRaises(ValueError):
            installer.load_manifest(self.root)

    def test_declared_root_package_must_be_in_the_bundle(self):
        manifest = complete_manifest()
        manifest["files"] = [entry for entry in manifest["files"] if entry["package"] != "torchaudio"]
        self.save(manifest)
        with self.assertRaises(ValueError):
            installer.load_manifest(self.root)

    def test_root_declarations_must_include_the_torch_stack(self):
        manifest = complete_manifest()
        manifest["root_packages"] = ["runtime"]
        self.save(manifest)
        with self.assertRaises(ValueError):
            installer.load_manifest(self.root)

    def test_every_wheel_needs_an_explicit_dependency_list(self):
        for value in (None, "runtime", [None]):
            with self.subTest(value=value):
                manifest = complete_manifest()
                manifest["files"][1]["requires"] = value
                self.save(manifest)
                with self.assertRaises(ValueError):
                    installer.load_manifest(self.root)
        manifest = complete_manifest()
        del manifest["files"][1]["requires"]
        self.save(manifest)
        with self.assertRaises(ValueError):
            installer.load_manifest(self.root)

    def test_valid_dry_run_needs_no_wheels_platform_check_or_auditor(self):
        self.save(complete_manifest())
        with mock.patch.object(installer, "ROOT", self.root), \
                mock.patch.object(installer, "check_platform") as platform, \
                mock.patch.object(installer, "verify_all") as hashes, \
                mock.patch.object(installer, "verify_dependency_closure") as closure, \
                mock.patch.object(installer.subprocess, "run") as pip, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            result = installer.main(["--dry-run"])
        self.assertEqual(result, 0)
        platform.assert_not_called()
        hashes.assert_not_called()
        closure.assert_not_called()
        pip.assert_not_called()
        self.assertIn(str(self.root / "manifest.json"), output.getvalue())
        self.assertIn("Wheel count: 4", output.getvalue())
        self.assertEqual(output.getvalue().count("pip install --no-index --no-deps"), 4)

    def test_metadata_audit_failure_stops_before_every_pip_operation(self):
        self.save(complete_manifest())
        events = []

        def fail_audit(*arguments):
            events.append("closure")
            raise ValueError("torch requires a missing Linux runtime")

        with mock.patch.object(installer, "ROOT", self.root), \
                mock.patch.object(installer, "check_platform", side_effect=lambda: events.append("platform")), \
                mock.patch.object(installer, "verify_all", side_effect=lambda *args: events.append("hashes")), \
                mock.patch.object(installer, "verify_dependency_closure", side_effect=fail_audit), \
                mock.patch.object(installer.subprocess, "run") as pip, \
                mock.patch("sys.stdout", new_callable=io.StringIO), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as errors:
            result = installer.main([])
        self.assertEqual(result, 1)
        self.assertEqual(events, ["platform", "hashes", "closure"])
        pip.assert_not_called()
        self.assertIn("missing Linux runtime", errors.getvalue())

    def test_valid_actual_install_checks_everything_before_ordered_pip_calls(self):
        manifest = complete_manifest()
        self.save(manifest)
        events = []
        commands = []

        def run(command, check):
            self.assertTrue(check)
            events.append("pip")
            commands.append(command)

        with mock.patch.object(installer, "ROOT", self.root), \
                mock.patch.object(installer, "check_platform", side_effect=lambda: events.append("platform")), \
                mock.patch.object(installer, "verify_all", side_effect=lambda *args: events.append("hashes")), \
                mock.patch.object(installer, "verify_dependency_closure", side_effect=lambda *args: events.append("closure")), \
                mock.patch.object(installer.subprocess, "run", side_effect=run), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            result = installer.main([])
        self.assertEqual(result, 0)
        self.assertEqual(events[:3], ["platform", "hashes", "closure"])
        self.assertEqual(events[3:], ["pip"] * 5)
        expected = [[sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                     str(self.root / entry["filename"])] for entry in manifest["files"]]
        expected.append([sys.executable, "-m", "pip", "check"])
        self.assertEqual(commands, expected)

    def test_missing_auditor_has_a_clear_recovery_error(self):
        with self.assertRaises(ValueError) as caught:
            installer.verify_dependency_closure(self.root, complete_manifest())
        self.assertRegex(str(caught.exception), r"audit\.py|sync|Sync|main")

    def test_missing_auditor_import_dependency_has_a_clear_recovery_error(self):
        (self.root / "audit.py").write_text("# fixture helper exists\n")
        with mock.patch.object(installer.importlib, "import_module", side_effect=ImportError("fixture helper absent")):
            with self.assertRaises(ValueError) as caught:
                installer.verify_dependency_closure(self.root, complete_manifest())
        self.assertRegex(str(caught.exception), r"audit\.py|sync|Sync|main|pip")


if __name__ == "__main__":
    unittest.main(verbosity=2)
