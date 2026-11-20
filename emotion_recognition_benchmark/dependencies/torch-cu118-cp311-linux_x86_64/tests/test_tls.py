"""Local HTTPS checks for the wheel downloader's company-certificate support."""
from contextlib import contextmanager
import hashlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BODY = b"small verified wheel fixture\n"
FILENAME = "fixture+cu118-cp311-cp311-linux_x86_64.whl"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Do not accidentally import a model/dataset module also named "tls".
tls = load_module("wheel_downloader_test_tls", ROOT / "tls.py")
with mock.patch.dict(sys.modules, {"tls": tls}):
    downloader = load_module("wheel_downloader_test_main", ROOT / "download.py")


def manifest(url):
    return {
        "schema_version": 1,
        "release_url": "https://example.invalid/releases/tag/fixture",
        "files": [{
            "filename": FILENAME,
            "size_bytes": len(BODY),
            "sha256": hashlib.sha256(BODY).hexdigest(),
            "url": url,
        }],
    }


@contextmanager
def local_https(certificate, key):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate), str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class TLSContextTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {"SSL_CERT_FILE": ""})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def test_default_context_checks_certificate_and_hostname(self):
        with mock.patch.object(tls.importlib, "import_module", side_effect=ImportError("optional helper absent")):
            context = tls.create_ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_native_context_restores_checks_without_global_patch(self):
        original = ssl.SSLContext
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        constructor = mock.Mock(return_value=context)
        native = types.SimpleNamespace(SSLContext=constructor)
        with mock.patch.object(tls.importlib, "import_module", return_value=native):
            actual = tls.create_ssl_context()
        self.assertIs(actual, context)
        constructor.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
        self.assertIs(ssl.SSLContext, original)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_verify_only_does_not_construct_tls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest("https://unused.invalid/fixture")))
            (root / FILENAME).write_bytes(BODY)
            with mock.patch.object(downloader, "ROOT", root), \
                    mock.patch.object(downloader, "create_ssl_context", side_effect=AssertionError("TLS is unused")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                result = downloader.main(["--verify-only", "--ca-bundle", str(root / "absent.pem")])
            self.assertEqual(result, 0)
            self.assertIn("OK: " + FILENAME, output.getvalue())


@unittest.skipUnless(shutil.which("openssl"), "openssl is required for local test certificates")
class WheelDownloadCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wheel-tls-certificates-")
        root = Path(cls.temporary.name)
        cls.ca = root / "company root.pem"
        cls.key = root / "server.key"
        cls.certificate = root / "server.pem"
        config = root / "ca.cnf"
        config.write_text("[req]\nprompt=no\ndistinguished_name=dn\nx509_extensions=v3_ca\n"
                          "[dn]\nCN=Ephemeral fixture company root\n[v3_ca]\n"
                          "basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\n"
                          "subjectKeyIdentifier=hash\n")
        extensions = root / "server.ext"
        extensions.write_text("basicConstraints=critical,CA:FALSE\n"
                              "keyUsage=critical,digitalSignature,keyEncipherment\n"
                              "extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost\n")

        def command(*arguments):
            subprocess.run(["openssl", *map(str, arguments)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        command("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", root / "ca.key",
                "-out", cls.ca, "-days", "1", "-config", config)
        command("req", "-newkey", "rsa:2048", "-nodes", "-keyout", cls.key,
                "-out", root / "server.csr", "-subj", "/CN=localhost")
        command("x509", "-req", "-in", root / "server.csr", "-CA", cls.ca,
                "-CAkey", root / "ca.key", "-CAcreateserial", "-out", cls.certificate,
                "-days", "1", "-extfile", extensions)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @contextmanager
    def workspace(self, url):
        with tempfile.TemporaryDirectory(prefix="wheel-tls-download-") as directory:
            root = Path(directory)
            for name in ("download.py", "tls.py"):
                shutil.copyfile(ROOT / name, root / name)
            (root / "manifest.json").write_text(json.dumps(manifest(url)))
            yield root

    def run_cli(self, root, *arguments, ca_environment=""):
        environment = dict(os.environ, SSL_CERT_FILE=ca_environment,
                           NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1")
        return subprocess.run([sys.executable, str(root / "download.py"), "--source", "upstream", *arguments],
                              capture_output=True, text=True, timeout=15, env=environment)

    def test_cli_company_ca_option_downloads_verified_wheel(self):
        with local_https(self.certificate, self.key) as port:
            with self.workspace("https://localhost:{}/wheel".format(port)) as root:
                result = self.run_cli(root, "--ca-bundle", str(self.ca))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((root / FILENAME).read_bytes(), BODY)
                self.assertFalse((root / (FILENAME + ".download")).exists())

    def test_cli_ssl_cert_file_downloads_verified_wheel(self):
        with local_https(self.certificate, self.key) as port:
            with self.workspace("https://localhost:{}/wheel".format(port)) as root:
                result = self.run_cli(root, ca_environment=str(self.ca))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((root / FILENAME).read_bytes(), BODY)

    def test_cli_explicit_ca_takes_precedence_over_environment(self):
        with local_https(self.certificate, self.key) as port:
            with self.workspace("https://localhost:{}/wheel".format(port)) as root:
                result = self.run_cli(root, "--ca-bundle", str(self.ca),
                                      ca_environment=str(root / "unused-environment-ca.pem"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((root / FILENAME).read_bytes(), BODY)

    def test_cli_untrusted_company_root_explains_fix_and_keeps_partial(self):
        with local_https(self.certificate, self.key) as port:
            with self.workspace("https://localhost:{}/wheel".format(port)) as root:
                partial = root / (FILENAME + ".download")
                partial.write_bytes(BODY[:8])
                result = self.run_cli(root)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(partial.read_bytes(), BODY[:8])
                self.assertFalse((root / FILENAME).exists())
                self.assertIn("truststore", result.stderr)
                self.assertIn("--ca-bundle", result.stderr)
                self.assertIn("SSL_CERT_FILE", result.stderr)
                self.assertIn("verification remain enabled", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_cli_company_ca_does_not_disable_hostname_check(self):
        with local_https(self.certificate, self.key) as port:
            with self.workspace("https://127.0.0.1:{}/wheel".format(port)) as root:
                result = self.run_cli(root, "--ca-bundle", str(self.ca))
                self.assertEqual(result.returncode, 1)
                self.assertFalse((root / FILENAME).exists())
                self.assertIn("certificate verification failed", result.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
