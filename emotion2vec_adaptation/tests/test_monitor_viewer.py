"""Check viewer confinement, YAML settings and optional live loopback HTTP."""

import errno
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from threading import Thread
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.monitor_viewer import (
    ViewerSettings, dashboard_paths, load_viewer_settings, make_handler, resolve_request_path,
)


class MonitorViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "outputs"
        self.dashboard = self.root / "run one/monitoring/dashboard.html"
        self.dashboard.parent.mkdir(parents=True)
        self.dashboard.write_text('<html><meta http-equiv="refresh" content="5">live dashboard</html>')
        self.png = self.dashboard.parent / "curves.png"
        self.png.write_bytes(b"\x89PNG\r\n\x1a\n")
        self.outside = self.base / "outside.txt"
        self.outside.write_text("must never be served")
        (self.root / "outside_link").symlink_to(self.outside)
        (self.root / "outside_dir").symlink_to(self.base, target_is_directory=True)

    def test_path_resolution_handles_query_encoding_and_blocks_root_escapes(self):
        self.assertEqual(resolve_request_path(self.root, "/run%20one/monitoring/curves.png?v=123"), self.png)
        for request in ("/../outside.txt", "/%2e%2e%2foutside.txt", "/outside_link",
                        "/outside_dir/outside.txt", "/%00.wav"):
            with self.subTest(request=request), self.assertRaises((ValueError, OSError)):
                resolve_request_path(self.root, request)

    def test_dashboard_index_ignores_outside_symlink_targets(self):
        remote = self.base / "elsewhere/monitoring/dashboard.html"
        remote.parent.mkdir(parents=True)
        remote.write_text("outside dashboard")
        destination = self.root / "bad run/monitoring/dashboard.html"
        destination.parent.mkdir(parents=True)
        destination.symlink_to(remote)
        self.assertEqual(dashboard_paths(self.root), [self.dashboard.relative_to(self.root)])

    def test_bind_host_and_port_are_strictly_local_and_valid(self):
        settings = ViewerSettings(self.root)
        self.assertEqual((settings.bind_host, settings.port), ("127.0.0.1", 8766))
        for host in ("0.0.0.0", "192.0.2.1", "::1", "localhost", 2130706433):
            with self.subTest(host=host), self.assertRaises(ValueError):
                ViewerSettings(self.root, bind_host=host)
        for port in (0, -1, 65536, True, "8766"):
            with self.subTest(port=port), self.assertRaises(ValueError):
                ViewerSettings(self.root, port=port)
        with self.assertRaises(ValueError):
            ViewerSettings(self.base / "missing")

    def test_yaml_loading_rejects_unknown_fields_and_bad_shapes(self):
        config = self.base / "viewer.yaml"
        valid = {"viewer": {"output_root": str(self.root), "bind_host": "127.0.0.1", "port": 8766}}
        config.write_text(yaml.safe_dump(valid))
        self.assertEqual(load_viewer_settings(config).output_root, self.root)
        for value in (None, [], {"other": {}}, {"viewer": {**valid["viewer"], "public": True}},
                      {"viewer": {**valid["viewer"], "output_root": 1}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.write_text(yaml.safe_dump(value))
                load_viewer_settings(config)

    def test_http_handler_without_socket_serves_assets_and_enforces_read_only(self):
        """Exercise the real request parser even when sandbox socket binds fail."""
        handler = make_handler(self.root)
        handler.log_message = lambda *args: None

        class MemorySocket:
            def __init__(self, request):
                self.request = BytesIO(request)
                self.response = BytesIO()

            def makefile(self, *args):
                return self.request

            def sendall(self, data):
                self.response.write(data)

        def request(method, target):
            connection = MemorySocket(f"{method} {target} HTTP/1.0\r\n\r\n".encode())
            handler(connection, ("127.0.0.1", 12345), None)
            header, _, body = connection.response.getvalue().partition(b"\r\n\r\n")
            return int(header.split(b" ", 2)[1]), body

        status, body = request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"/run%20one/monitoring/dashboard.html", body)
        self.assertIn(b'http-equiv="refresh" content="5"', body)
        self.assertEqual(request("GET", "/run%20one/monitoring/curves.png?v=updated"),
                         (200, self.png.read_bytes()))
        self.assertEqual(request("HEAD", "/run%20one/monitoring/dashboard.html"), (200, b""))
        self.assertIn(b"live dashboard", request("GET", "/run%20one/monitoring/dashboard.html")[1])
        for path in ("/../outside.txt", "/%2e%2e%2foutside.txt", "/outside_link", "/outside_dir/outside.txt"):
            self.assertEqual(request("GET", path)[0], 403)
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            self.assertEqual(request(method, "/new-file")[0], 405)
        self.assertFalse((self.root / "new-file").exists())

    def test_live_loopback_get_head_post_and_escape_responses(self):
        handler = make_handler(self.root)
        handler.log_message = lambda *args: None
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        except OSError as error:
            if error.errno in {errno.EPERM, errno.EACCES}:
                self.skipTest("Sandbox blocks loopback socket binding; pure path/settings tests still run")
            raise
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path):
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            try:
                connection.request(method, path)
                response = connection.getresponse()
                return response.status, response.read(), response.getheader("Content-Type")
            finally:
                connection.close()

        try:
            status, body, mime = request("GET", "/")
            self.assertEqual(status, 200)
            self.assertIn(b"/run%20one/monitoring/dashboard.html", body)
            self.assertNotIn(b"outside.txt", body)
            status, body, _ = request("GET", "/run%20one/monitoring/dashboard.html")
            self.assertEqual(status, 200)
            self.assertIn(b"live dashboard", body)
            self.assertEqual(request("HEAD", "/run%20one/monitoring/dashboard.html")[:2], (200, b""))
            self.assertEqual(request("GET", "/run%20one/monitoring/curves.png?v=updated")[:2],
                             (200, self.png.read_bytes()))
            for path in ("/../outside.txt", "/%2e%2e%2foutside.txt", "/outside_link", "/outside_dir/outside.txt"):
                self.assertEqual(request("GET", path)[0], 403)
            self.assertEqual(request("POST", "/new-file")[0], 405)
            self.assertFalse((self.root / "new-file").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
