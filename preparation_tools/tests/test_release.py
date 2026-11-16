"""download.py and publish.py against local stand-ins for GitHub and the GitHub CLI.

Offline and light: tiny fake wheels and checkpoint in a temporary folder, two HTTP servers on 127.0.0.1 (one for
github.com + api.github.com, one for the storage host that GitHub redirects to) and a fake `gh` that keeps the
Release in a local folder. Nothing contacts GitHub; the real files in dependencies/ and model_weights/ are untouched.
"""
import contextlib
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import download  # noqa: E402
import publish  # noqa: E402

TOKEN = "ghp_TEST0token0never0printed"
LINUX, MAC = "cp311-linux_x86_64", "cp311-macosx_arm64"
WHEELS = {("any", "alpha-1.0-py3-none-any.whl"): 5000,
          (LINUX, "beta-1.0-cp311-cp311-manylinux_2_28_x86_64.whl"): 3000,
          (MAC, "beta-1.0-cp311-cp311-macosx_11_0_arm64.whl"): 3100}
CHECKPOINT = ("checkpoints/tiny.pth", 10_000)
LINUX_FILES = ["dependencies/any/alpha-1.0-py3-none-any.whl",
               "dependencies/cp311-linux_x86_64/beta-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
               "model_weights/checkpoints/tiny.pth"]

FAKE_GH = '''#!@PYTHON@ -IS
"""Fake GitHub CLI for tests: one Release kept in local folders; it never contacts GitHub."""
import hashlib, json, shutil, sys
from pathlib import Path
store, assets = Path("@STORE@"), Path("@ASSETS@")
args = sys.argv[1:]
with (store / "calls.jsonl").open("a") as log:
    log.write(json.dumps(args) + "\\n")
state = store / "release.json"
release = json.loads(state.read_text()) if state.exists() else None
if args[:2] == ["release", "view"]:
    if release is None:
        sys.exit("release not found")
    listing = [{"name": p.name, "size": p.stat().st_size, "state": "uploaded",
                "digest": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(assets.iterdir())]
    print(json.dumps({"tagName": release["tag"], "isDraft": release["draft"], "assets": listing}))
elif args[:2] == ["release", "create"]:
    state.write_text(json.dumps({"tag": args[2], "draft": "--draft" in args}))
elif args[:2] == ["release", "upload"]:
    source = Path(args[3])
    if (assets / source.name).exists() and "--clobber" not in args:
        sys.exit("asset under the same filename already exists")
    shutil.copyfile(source, assets / source.name)
elif args[:2] == ["release", "edit"] and "--draft=false" in args:
    release["draft"] = False
    state.write_text(json.dumps(release))
else:
    sys.exit("unexpected gh call: " + " ".join(args))
'''


def send(handler, code, body=b"", content_type="text/plain"):
    handler.send_response(code)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def redirect(handler, location):
    handler.send_response(302)
    handler.send_header("Location", location)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


class FakeGitHub:
    """github.com download URLs, the REST API (JSON + octet-stream redirect) and a separate storage host."""

    def __init__(self, assets):
        self.assets = assets         # folder with the Release assets
        self.private = False         # True: anonymous requests get 404, as for a private repository
        self.ignore_range = False    # True: the storage host always answers 200 with the whole asset
        self.corrupt = set()         # asset names served with one flipped byte
        self.cut_once = set()        # asset names whose next response stops half way
        self.log = []                # (host, path, headers)
        self.github, self.storage = self.server("github"), self.server("storage")

    def server(self, role):
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.log.append((role, urllib.parse.unquote(self.path), dict(self.headers)))
                (fake.handle_github if role == "github" else fake.handle_storage)(self)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        return server

    @staticmethod
    def url(server):
        return f"http://127.0.0.1:{server.server_address[1]}"

    def close(self):
        for server in (self.github, self.storage):
            server.shutdown()
            server.server_close()

    def ids(self):
        return {path.name: 1000 + index for index, path in enumerate(sorted(self.assets.iterdir()))}

    def requests(self, role):
        return [(path, headers) for host, path, headers in self.log if host == role]

    def handle_github(self, handler):
        repo, tag = download.REPOSITORY, download.RELEASE_TAG
        path = urllib.parse.unquote(handler.path)
        authorized = handler.headers.get("Authorization") == "Bearer " + TOKEN
        storage = f"{self.url(self.storage)}/storage/"
        if path.startswith(f"/{repo}/releases/download/{tag}/") or path.startswith("/upstream/"):
            name = path.rsplit("/", 1)[1]
            if (self.private and not path.startswith("/upstream/")) or not (self.assets / name).is_file():
                return send(handler, 404)
            return redirect(handler, storage + name + "?X-Signature=abc")
        if not authorized:   # GitHub answers 404 for a private repository without a valid token
            return send(handler, 404)
        if path == f"/api/repos/{repo}/releases/tags/{tag}":
            assets = [{"id": number, "name": name, "size": (self.assets / name).stat().st_size}
                      for name, number in self.ids().items()]
            return send(handler, 200, json.dumps({"tag_name": tag, "assets": assets}).encode(), "application/json")
        match = re.fullmatch(rf"/api/repos/{repo}/releases/assets/(\d+)", path)
        names = {number: name for name, number in self.ids().items()}
        if match and int(match[1]) in names:
            if handler.headers.get("Accept") != "application/octet-stream":   # GitHub then returns JSON
                return send(handler, 200, json.dumps({"id": int(match[1])}).encode(), "application/json")
            return redirect(handler, storage + names[int(match[1])] + "?X-Signature=abc")
        send(handler, 404)

    def handle_storage(self, handler):
        if "Authorization" in handler.headers:
            return send(handler, 400, b"credentials were forwarded to the storage host")
        name = urllib.parse.urlsplit(handler.path).path.rsplit("/", 1)[1]
        if not (self.assets / name).is_file():
            return send(handler, 404)
        data = bytearray((self.assets / name).read_bytes())
        if name in self.corrupt:
            data[len(data) // 2] ^= 0xFF
        start, match = 0, re.fullmatch(r"bytes=(\d+)-", handler.headers.get("Range", ""))
        if match and not self.ignore_range:
            start = int(match[1])
            if start >= len(data):
                return send(handler, 416)
            handler.send_response(206)
            handler.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            handler.send_response(200)
        body = bytes(data[start:])
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if name in self.cut_once:   # simulate a dropped connection
            self.cut_once.discard(name)
            handler.wfile.write(body[:len(body) // 2])
            handler.close_connection = True
            return
        handler.wfile.write(body)


def make_bundle(root, upstream):
    """The publisher's folder: manifests, LICENSES.md and tiny random files in place of wheels and checkpoint."""
    entries = []
    for (platform_key, name), size in WHEELS.items():
        data = os.urandom(size)
        path = root / "dependencies" / platform_key / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append({"package": name.split("-")[0], "version": "1.0", "platform": platform_key,
                        "filename": name, "url": f"{upstream}/upstream/{name}",
                        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": size, "requires": []})
    (root / "dependencies" / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "platforms": {LINUX: "Linux", MAC: "macOS"}, "files": entries}))
    relative, size = CHECKPOINT
    data = os.urandom(size)
    (root / "model_weights" / relative).parent.mkdir(parents=True)
    (root / "model_weights" / relative).write_bytes(data)
    (root / "model_weights" / "manifest.json").write_text(json.dumps({"schema_version": 1, "models": {"tiny": {
        "filename": Path(relative).name, "path": relative, "url": f"{upstream}/upstream/{Path(relative).name}",
        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": size}}}))
    (root / "LICENSES.md").write_text("Licences of the test files\n")


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="release-test-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.source, self.dest, self.assets, self.store = (base / "source", base / "dest", base / "assets",
                                                           base / "gh_store")
        self.assets.mkdir()
        self.store.mkdir()
        self.fake = FakeGitHub(self.assets)
        self.addCleanup(self.fake.close)
        github = self.fake.url(self.fake.github)
        make_bundle(self.source, github)
        # The downloading machine starts with the manifests from Git only.
        shutil.copytree(self.source, self.dest, ignore=shutil.ignore_patterns("*.whl", "*.pth"))
        fake_gh = self.store / "gh"
        fake_gh.write_text(FAKE_GH.replace("@PYTHON@", sys.executable).replace("@STORE@", str(self.store))
                           .replace("@ASSETS@", str(self.assets)))
        fake_gh.chmod(0o755)
        settings = [(download, "ROOT", self.dest), (download, "DOWNLOAD_URL", github),
                    (download, "API_URL", github + "/api"), (download, "PLATFORM", LINUX),
                    (download, "SOURCE", "release"), (download, "GET_WEIGHTS", True), (download, "CA_BUNDLE", None),
                    (download, "USE_GH_CLI", False), (download, "RETRY_WAIT_S", 0),
                    (publish, "GH", str(fake_gh)), (publish, "DRY_RUN", False)]
        for module, name, value in settings:
            patcher = mock.patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_REPO", "SSL_CERT_FILE"):
            os.environ.pop(name, None)
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost"   # no system proxy for the fakes

    # --- helpers ---
    def fill_release(self):
        """Put the publisher's files on the fake Release as publish.py names them (split above PART_SIZE)."""
        with mock.patch.object(download, "ROOT", self.source):
            deps, weights = download.load_manifests()
            for entry in download.planned_files(deps, weights, [LINUX, MAC], True):
                data = entry["target"].read_bytes()
                for asset, start, length in download.release_parts(entry["name"], entry["size"]):
                    (self.assets / asset).write_bytes(data[start:start + length])

    def run_script(self, module):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = module.main()
        return code, out.getvalue(), err.getvalue()

    def assert_restored(self, files=LINUX_FILES):
        for relative in files:
            self.assertEqual((self.dest / relative).read_bytes(), (self.source / relative).read_bytes(), relative)
        self.assertEqual(list(self.dest.rglob("*.download")), [])

    def gh_calls(self):
        log = self.store / "calls.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    # --- download.py ---
    def test_fresh_anonymous_download(self):
        self.fill_release()
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()
        self.assertEqual(list((self.dest / "dependencies" / MAC).glob("*.whl")), [])   # only this platform's wheels
        self.assertEqual(len(self.fake.requests("storage")), 3)
        self.assertTrue(all("/releases/download/" in path for path, _ in self.fake.requests("github")))
        self.assertIn("anonymously", out)

    def test_resume_from_half_file(self):
        self.fill_release()
        target = self.dest / "model_weights" / CHECKPOINT[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        Path(str(target) + ".download").write_bytes((self.source / "model_weights" / CHECKPOINT[0]).read_bytes()[:5000])
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()
        ranges = [headers.get("Range") for path, headers in self.fake.requests("storage") if "tiny.pth" in path]
        self.assertEqual(ranges, ["bytes=5000-"])
        self.assertIn("resuming model_weights/checkpoints/tiny.pth (5,000/10,000 bytes)", out)

    def test_hash_mismatch_is_rejected(self):
        self.fill_release()
        target = self.dest / "model_weights" / CHECKPOINT[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * CHECKPOINT[1])   # right size, wrong bytes: must not be reused
        self.fake.corrupt = {"tiny.pth"}
        code, out, err = self.run_script(download)
        self.assertEqual(code, 1)
        self.assertIn("SHA-256 mismatch for tiny.pth", err)
        self.assertEqual(target.read_bytes(), b"x" * CHECKPOINT[1])   # the bad download did not replace anything
        self.assertFalse(Path(str(target) + ".download").exists())    # and was deleted, not kept for resuming
        self.fake.corrupt = set()
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()

    def test_verified_files_are_reused_without_network(self):
        self.fill_release()
        self.assertEqual(self.run_script(download)[0], 0)
        self.fake.log.clear()
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assertEqual(self.fake.log, [])
        self.assertEqual(out.count("  already verified: "), 3)

    def test_private_repository_with_token(self):
        self.fill_release()
        self.fake.private = True
        os.environ["GH_TOKEN"] = TOKEN
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()
        github = self.fake.requests("github")
        self.assertEqual(sum(path.endswith("/releases/tags/" + download.RELEASE_TAG) for path, _ in github), 1)
        assets = [headers for path, headers in github if "/releases/assets/" in path]
        self.assertEqual(len(assets), 3)
        self.assertTrue(all(h["Authorization"] == "Bearer " + TOKEN and h["Accept"] == "application/octet-stream"
                            for h in assets))
        storage = self.fake.requests("storage")
        self.assertEqual(len(storage), 3)
        self.assertFalse(any(key.lower() == "authorization" for _, headers in storage for key in headers))
        self.assertNotIn(TOKEN, out + err)
        self.assertIn("token from GH_TOKEN (not shown)", out)

    def test_private_repository_without_or_with_wrong_token(self):
        self.fill_release()
        self.fake.private = True
        code, out, err = self.run_script(download)
        self.assertEqual(code, 1)
        self.assertIn("set GH_TOKEN", err)
        os.environ["GITHUB_TOKEN"] = "ghp_WRONG0token"
        code, out, err = self.run_script(download)
        self.assertEqual(code, 1)
        self.assertIn("token from GITHUB_TOKEN cannot read", err)
        self.assertNotIn("ghp_WRONG0token", out + err)

    def test_token_sources(self):
        os.environ["GITHUB_TOKEN"] = "second"
        self.assertEqual(download.github_token(), ("second", "GITHUB_TOKEN"))
        os.environ["GH_TOKEN"] = "first"
        self.assertEqual(download.github_token(), ("first", "GH_TOKEN"))
        del os.environ["GH_TOKEN"], os.environ["GITHUB_TOKEN"]
        self.assertEqual(download.github_token(), (None, None))   # USE_GH_CLI is off in these tests
        answer = subprocess.CompletedProcess([], 0, stdout=TOKEN + "\n", stderr="")
        with mock.patch.object(download, "USE_GH_CLI", True), \
                mock.patch.object(download.shutil, "which", return_value="/usr/bin/gh"), \
                mock.patch.object(download.subprocess, "run", return_value=answer) as run:
            self.assertEqual(download.github_token(), (TOKEN, "`gh auth token`"))
        self.assertEqual(run.call_args[0][0], ["gh", "auth", "token", "--hostname", "github.com"])
        os.environ["GH_TOKEN"] = "has space"
        with self.assertRaises(ValueError) as caught:
            download.github_token()
        self.assertNotIn("has space", str(caught.exception))

    def test_split_parts_resume_mid_part_and_server_ignoring_range(self):
        with mock.patch.object(download, "PART_SIZE", 4096):
            self.fill_release()
            self.assertEqual(sorted(p.name for p in self.assets.glob("tiny.pth.part*")),
                             ["tiny.pth.part000", "tiny.pth.part001", "tiny.pth.part002"])
            target = self.dest / "model_weights" / CHECKPOINT[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            source = (self.source / "model_weights" / CHECKPOINT[0]).read_bytes()
            Path(str(target) + ".download").write_bytes(source[:6000])
            code, out, err = self.run_script(download)
            self.assertEqual(code, 0, err)
            self.assert_restored()
            requests = [(path.split("/")[-1].split("?")[0], headers.get("Range"))
                        for path, headers in self.fake.requests("storage") if "tiny.pth" in path]
            self.assertEqual(requests, [("tiny.pth.part001", "bytes=1904-"), ("tiny.pth.part002", None)])
            # A server that ignores Range: the current part is written again from its first byte.
            target.unlink()
            Path(str(target) + ".download").write_bytes(source[:6000])
            self.fake.ignore_range = True
            code, out, err = self.run_script(download)
            self.assertEqual(code, 0, err)
            self.assert_restored()

    def test_dropped_connection_is_resumed(self):
        self.fill_release()
        self.fake.cut_once = {"tiny.pth"}
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()
        ranges = [headers.get("Range") for path, headers in self.fake.requests("storage") if "tiny.pth" in path]
        self.assertEqual(ranges, [None, "bytes=5000-"])
        self.assertIn("attempt 1/3 failed", out)

    def test_upstream_source(self):
        self.fill_release()
        with mock.patch.object(download, "SOURCE", "upstream"):
            code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()
        self.assertTrue(all(path.startswith("/upstream/") for path, _ in self.fake.requests("github")))

    def test_platform_detection(self):
        deps, _ = download.load_manifests()
        with mock.patch.object(download, "PLATFORM", "auto"):
            for system, machine, expected in (("Linux", "x86_64", [LINUX]), ("Darwin", "arm64", [MAC])):
                with mock.patch("platform.system", return_value=system), \
                        mock.patch("platform.machine", return_value=machine):
                    self.assertEqual(download.wheel_platforms(deps), expected)
            with mock.patch("platform.system", return_value="Windows"), \
                    mock.patch("platform.machine", return_value="AMD64"), self.assertRaises(ValueError):
                download.wheel_platforms(deps)
        with mock.patch.object(download, "PLATFORM", "all"):
            self.assertEqual(download.wheel_platforms(deps), [LINUX, MAC])

    # --- publish.py ---
    def publish_from_source(self):
        with mock.patch.object(download, "ROOT", self.source):
            return self.run_script(publish)

    def test_publish_then_download(self):
        code, out, err = self.publish_from_source()
        self.assertEqual(code, 0, err)
        calls = [call[:2] for call in self.gh_calls()]
        self.assertIn(["release", "create"], calls)
        create = next(call for call in self.gh_calls() if call[:2] == ["release", "create"])
        self.assertIn("--draft", create)
        self.assertLess(max(i for i, c in enumerate(self.gh_calls()) if c[:2] == ["release", "upload"]
                            and "--clobber" not in c),
                        min(i for i, c in enumerate(self.gh_calls()) if "--clobber" in c))   # metadata last
        self.assertEqual(calls[-2], ["release", "edit"])
        self.assertFalse(json.loads((self.store / "release.json").read_text())["draft"])
        names = sorted(p.name for p in self.assets.iterdir())
        self.assertEqual(names, sorted([name for _, name in WHEELS] + ["tiny.pth"] + list(publish.METADATA)))
        sums = (self.assets / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(len(sums), 4)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}  \S+", line) for line in sums))
        # The company machine downloads what was published, through the token path.
        self.fake.private = True
        os.environ["GH_TOKEN"] = TOKEN
        code, out, err = self.run_script(download)
        self.assertEqual(code, 0, err)
        self.assert_restored()
        # Rerunning publish uploads nothing.
        before = len(self.gh_calls())
        code, out, err = self.publish_from_source()
        self.assertEqual(code, 0, err)
        self.assertEqual({tuple(call[:2]) for call in self.gh_calls()[before:]}, {("release", "view")})

    def test_publish_refuses_changed_asset_and_replaces_only_metadata(self):
        self.assertEqual(self.publish_from_source()[0], 0)
        (self.source / "LICENSES.md").write_text("Licences, second edition\n")
        before = len(self.gh_calls())
        code, out, err = self.publish_from_source()
        self.assertEqual(code, 0, err)
        uploads = [call for call in self.gh_calls()[before:] if call[:2] == ["release", "upload"]]
        self.assertEqual([Path(call[3]).name for call in uploads], ["LICENSES.md"])
        self.assertIn("--clobber", uploads[0])
        (self.assets / "tiny.pth").write_bytes(b"other bytes")   # same name, different bytes on GitHub
        before = len(self.gh_calls())
        code, out, err = self.publish_from_source()
        self.assertEqual(code, 1)
        self.assertIn("differ from the local files", err)
        self.assertEqual({tuple(call[:2]) for call in self.gh_calls()[before:]}, {("release", "view")})

    def test_publish_dry_run_changes_nothing(self):
        with mock.patch.object(publish, "DRY_RUN", True):
            code, out, err = self.publish_from_source()
        self.assertEqual(code, 0, err)
        self.assertIn("would run: ", out)
        self.assertEqual([call[:2] for call in self.gh_calls()], [["release", "view"]])
        self.assertEqual(list(self.assets.iterdir()), [])

    def test_publish_splits_large_files(self):
        with mock.patch.object(download, "PART_SIZE", 4096):
            code, out, err = self.publish_from_source()
            self.assertEqual(code, 0, err)
            names = {p.name for p in self.assets.iterdir()}
            self.assertTrue({"tiny.pth.part000", "tiny.pth.part001", "tiny.pth.part002",
                             "alpha-1.0-py3-none-any.whl.part000", "alpha-1.0-py3-none-any.whl.part001"} <= names)
            self.assertNotIn("tiny.pth", names)
            code, out, err = self.run_script(download)
            self.assertEqual(code, 0, err)
            self.assert_restored()

    def test_publish_needs_every_local_file(self):
        (self.source / "dependencies" / MAC / "beta-1.0-cp311-cp311-macosx_11_0_arm64.whl").unlink()
        code, out, err = self.publish_from_source()
        self.assertEqual(code, 1)
        self.assertIn("is missing or differs from its manifest", err)
        self.assertEqual(self.gh_calls(), [])   # nothing on GitHub was touched, not even read


if __name__ == "__main__":
    unittest.main()
