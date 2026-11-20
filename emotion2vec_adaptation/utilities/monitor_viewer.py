"""Read-only, loopback-only HTTP access to an existing training output folder."""

from dataclasses import dataclass
from html import escape
from http.server import SimpleHTTPRequestHandler
from io import BytesIO
import ipaddress
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import yaml

from .config import PROJECT_DIR, resolve_path


@dataclass(frozen=True)
class ViewerSettings:
    output_root: Path
    bind_host: str = "127.0.0.1"
    port: int = 8766

    def __post_init__(self):
        if not isinstance(self.bind_host, str):
            raise ValueError("viewer.bind_host must be an IPv4 loopback address string")
        try:
            address = ipaddress.IPv4Address(self.bind_host)
        except (ipaddress.AddressValueError, TypeError) as exc:
            raise ValueError("viewer.bind_host must be an IPv4 loopback address such as 127.0.0.1") from exc
        if not address.is_loopback:
            raise ValueError("viewer.bind_host must be loopback; use SSH forwarding for remote viewing")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ValueError("viewer.port must be an integer between 1 and 65535")
        root = Path(self.output_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"viewer.output_root must be an existing directory: {root}")
        object.__setattr__(self, "output_root", root)


def load_viewer_settings(path: Path | None = None) -> ViewerSettings:
    path = path or PROJECT_DIR / "config/monitor_viewer.yaml"
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"viewer"}:
        raise ValueError("monitor_viewer.yaml must contain only a viewer mapping")
    settings = value["viewer"]
    if not isinstance(settings, dict) or set(settings) != {"output_root", "bind_host", "port"}:
        raise ValueError("viewer requires exactly output_root, bind_host and port")
    if not isinstance(settings["output_root"], str) or not settings["output_root"].strip():
        raise ValueError("viewer.output_root must be a nonempty path")
    return ViewerSettings(resolve_path(settings["output_root"]), settings["bind_host"], settings["port"])


def resolve_request_path(root: Path, request_path: str) -> Path:
    """Resolve decoded request paths and symlinks, rejecting any root escape."""
    root = Path(root).resolve()
    decoded = unquote(urlsplit(request_path).path, errors="strict")
    candidate = (root / decoded.lstrip("/")).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Request escapes the configured output root")
    return candidate


def dashboard_paths(root: Path) -> list[Path]:
    """Return newest-first dashboard paths without following outside symlinks."""
    return sorted((path.relative_to(root) for path in root.glob("**/monitoring/dashboard.html")
                   if path.is_file() and path.resolve().is_relative_to(root)),
                  key=lambda path: (root / path).stat().st_mtime, reverse=True)


def make_handler(output_root: Path):
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Monitoring output root does not exist: {root}")

    class ReadOnlyHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def translate_path(self, path):
            return str(resolve_request_path(root, path))

        def send_head(self):
            try:
                path = self.translate_path(self.path)
                return self.list_directory(path) if Path(path) == root else super().send_head()
            except (ValueError, OSError, RuntimeError):
                self.send_error(403, "Path is outside the configured output root or inaccessible")
                return None

        def list_directory(self, path):
            if Path(path) != root:
                return super().list_directory(path)
            links = "".join(f'<li><a href="/{quote(item.as_posix(), safe="/")}">{escape(str(item.parent.parent))}</a></li>'
                            for item in dashboard_paths(root)) or "<li>No dashboards yet. Start a training run.</li>"
            data = (f'<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">'
                    '<title>Training monitors</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;'
                    'padding:0 20px}li{margin:14px 0}a{color:#2563eb}code{overflow-wrap:anywhere}</style></head>'
                    f'<body><h1>Training monitors</h1><p>Read-only output root: <code>{escape(str(root))}</code></p>'
                    f'<p>This list refreshes every 5 seconds.</p><ul>{links}</ul></body></html>').encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return BytesIO(data)

        def do_POST(self):
            self.send_error(405, "Read-only viewer; only GET and HEAD are supported")

        do_PUT = do_PATCH = do_DELETE = do_POST

    return ReadOnlyHandler
