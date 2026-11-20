"""Serve remote training monitors through an SSH-forwardable loopback endpoint.

Run without arguments. Edit config/monitor_viewer.yaml to choose the output root
and port. Stop with Ctrl+C; this script never opens a browser or starts a daemon.
"""

from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import quote

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.monitor_viewer import dashboard_paths, load_viewer_settings, make_handler


def main():
    settings = load_viewer_settings()
    with ThreadingHTTPServer((settings.bind_host, settings.port), make_handler(settings.output_root)) as server:
        base = f"http://{settings.bind_host}:{settings.port}"
        print(f"Read-only training monitors: {base}/", flush=True)
        print(f"Output root: {settings.output_root}", flush=True)
        for path in dashboard_paths(settings.output_root):
            print(f"  {base}/{quote(path.as_posix(), safe='/')}", flush=True)
        print("Press Ctrl+C to stop. For remote training, forward this loopback port over SSH.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Monitoring viewer stopped.", flush=True)


if __name__ == "__main__":
    main()
