#!/usr/bin/env python3
"""Segment Explorer: check the turn segmenter on one audio file and its transcript in the browser.

    python preparation_tools/explorer/app.py

Starts a small web server on 127.0.0.1 (this computer only) and opens the page. Choose an audio file and its
transcript, click "Run segmenter", look at the result, change settings and re-run, export one static HTML page.
The decision comes from segmenter.pipeline.segment_turn with the settings of config/segment_turns.yaml, the same
code segment_turns.py runs. Uploads stay in memory (a temporary copy for decoding is deleted at once); the
temporary folder is removed when the server stops. Transcript text is never printed. Python standard library only
for the server; the models run in the same environment as segment_turns.py. Stop with Ctrl+C.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import unquote
import webbrowser

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # preparation_tools/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

import core  # noqa: E402

# Settings (constants instead of command-line options)
HOST = "127.0.0.1"           # this computer only; never serve internal call audio on another address
PORT = 8765                  # the next free port is used when this one is busy
OPEN_BROWSER = True          # open the page in the default browser at start
CONFIG = ROOT / "config" / "segment_turns.yaml"
PAGE = HERE / "page.html"
TMP_DIR = ROOT / "local" / "explorer_tmp"           # git-ignored; emptied at start, removed at stop
DISPLAY_ALIGNMENT = True     # also align short and non-English turns, for display only (never changes the result)
KEEP_UPLOADS = 3             # uploads kept in memory for instant re-runs
MAX_UPLOAD_MB = 500
MIN_FREE_MEMORY = 25         # percent (macOS); the models are not loaded below this. 0 switches the check off


def free_memory_percent():
    """System-wide free memory in percent on macOS (memory_pressure -Q), None elsewhere."""
    if platform.system() != "Darwin":
        return None
    text = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    match = re.search(r"free percentage: (\d+)%", text)
    return int(match.group(1)) if match else None


def model_loader(cfg):
    """The function that loads Silero VAD and the alignment model once, on the first upload."""
    def load():
        free = free_memory_percent()
        print(f"Loading the models (free memory {free if free is not None else '?'}%, need {MIN_FREE_MEMORY}%)")
        if free is not None and free < MIN_FREE_MEMORY:
            raise RuntimeError(f"Only {free}% of memory is free (need {MIN_FREE_MEMORY}%). Close other programs "
                               "and try again, or lower MIN_FREE_MEMORY at the top of explorer/app.py.")
        from segmenter.pipeline import load_models
        return load_models(cfg, ROOT)
    return load


class UserError(Exception):
    """An error whose message is safe to print and to show on the page (status code given)."""

    def __init__(self, status, message, code=""):
        super().__init__(message)
        self.status, self.code = status, code


def make_server(explorer, host=HOST, port=PORT, page=PAGE):
    """A ThreadingHTTPServer bound to host; tries port, port + 1, ... port + 20, then any free port.
    One computation at a time (a lock); the page can still be loaded while a run is busy."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "SegmentExplorer"
        sys_version = ""

        def log_message(self, fmt, *args):            # no request bodies or query strings are ever printed
            pass

        # ---- guards: only this page on this computer may call the API ----
        def allowed(self):
            port_ = self.server.server_address[1]
            hosts = {f"127.0.0.1:{port_}", f"localhost:{port_}"}
            if self.headers.get("Host") not in hosts:
                return False                          # blocks DNS rebinding
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{h}" for h in hosts}:
                return False                          # blocks other web pages
            return self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none")

        def send(self, status, body, content_type, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, obj):
            self.send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def read_body(self, content_type):
            if self.headers.get("Content-Type", "").split(";")[0].strip() != content_type:
                raise UserError(415, f"Expected Content-Type {content_type}")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise UserError(400, "Empty request")
            if length > MAX_UPLOAD_MB * 1024 * 1024:
                raise UserError(413, f"File larger than {MAX_UPLOAD_MB} MB (MAX_UPLOAD_MB in explorer/app.py)")
            return self.rfile.read(length)

        def read_json(self):
            try:
                return json.loads(self.read_body("application/json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise UserError(400, "The request is not valid JSON") from None

        def do_GET(self):
            if not self.allowed():
                return self.send(403, b"Forbidden", "text/plain")
            path = self.path.split("?")[0]
            if path == "/":
                return self.send(200, Path(page).read_bytes(), "text/html; charset=utf-8")
            if path == "/api/defaults":
                cfg = explorer.base_cfg
                return self.send_json(200, {"values": core.editable_values(cfg),
                                            "version": core.settings_version(cfg),
                                            "word_min_confidence": cfg["alignment"]["word_min_confidence"]})
            self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            if not self.allowed():
                return self.send(403, b"Forbidden", "text/plain")
            path = self.path.split("?")[0]
            t0 = time.time()
            try:
                if path == "/api/audio":
                    name = unquote(self.headers.get("X-File-Name", "")).replace("\\", "/").split("/")[-1]
                    raw = self.read_body("application/octet-stream")
                    with lock:
                        print(f"Upload: {name} ({len(raw) / 1e6:.1f} MB): decoding, analysis copy, VAD")
                        audio_id, upload = explorer.add_audio(name, raw)
                    seconds = upload["frames"] / upload["rate"]
                    print(f"  audio {audio_id}: {seconds:.2f} s, {upload['rate']} Hz, {upload['channels']} ch, "
                          f"{upload['subtype']}; decode {upload['timings']['decode_s']:.2f} s, "
                          f"VAD {upload['timings']['vad_s']:.2f} s")
                    return self.send_json(200, {"audio_id": audio_id, "duration_s": round(seconds, 4)})
                if path in ("/api/run", "/api/export"):
                    request = self.read_json()
                    transcript = request.get("transcript")
                    if not isinstance(transcript, str):
                        raise UserError(400, "Missing transcript text")
                    audio_id = str(request.get("audio_id", ""))
                    with lock:
                        try:
                            result = explorer.run(audio_id, transcript, str(request.get("transcript_name", "")),
                                                  request.get("settings") or {})
                        except KeyError:
                            raise UserError(404, "This audio is no longer in memory; upload it again.",
                                            "unknown_audio") from None
                    changed = ", ".join(result["settings"]["changed"]) or "none"
                    lengths = ", ".join(f"{s['duration_s']:.1f}" for s in result["segments"])
                    print(f"{'Export' if path == '/api/export' else 'Run'}: audio {audio_id}, settings "
                          f"{result['settings']['version']} (changed: {changed}) -> {result['group']}/"
                          f"{result['status']}, {len(result['segments'])} segment(s) [{lengths}]; alignment "
                          f"{result['timings']['alignment']}; {time.time() - t0:.2f} s")
                    if path == "/api/run":
                        return self.send_json(200, result)
                    upload = explorer.uploads[audio_id]
                    page_html = core.build_export_html(Path(page).read_text(encoding="utf-8"), result,
                                                       upload["raw"], upload["mime"])
                    name = core.export_name(upload["name"])
                    print(f"  export page {name}: {len(page_html) / 1e6:.1f} MB (contains the audio and transcript)")
                    return self.send(200, page_html.encode("utf-8"), "text/html; charset=utf-8",
                                     {"Content-Disposition": f'attachment; filename="{name}"'})
                self.send(404, b"Not found", "text/plain")
            except UserError as exc:
                print(f"  refused ({exc.status}): {exc}")
                self.send_json(exc.status, {"error": str(exc), "code": exc.code})
            except (ValueError, RuntimeError) as exc:      # settings, audio decoding, memory check: no transcript
                print(f"  error: {type(exc).__name__}: {exc}")
                self.send_json(400, {"error": f"{exc}"})
            except Exception as exc:                       # the message may quote transcript words: not printed
                print(f"  error: {type(exc).__name__} (details on the page only)")
                self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    for candidate in [*range(port, port + 21), 0] if port else [0]:
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            server.daemon_threads = True
            return server
        except OSError:
            continue
    raise OSError("No free port")


def stop_on_signal(*_):
    raise KeyboardInterrupt


def main():
    import soundfile
    import torch
    import torchaudio
    from importlib.metadata import version

    # 1) environment and settings
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"torch {torch.__version__}, torchaudio {torchaudio.__version__}, silero-vad {version('silero-vad')}, "
          f"soundfile {soundfile.__version__} (libsndfile {soundfile.__libsndfile_version__})")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    print(f"Step 1: settings from {CONFIG} ({core.settings_version(cfg)}); device {cfg['device']}")

    # 2) temporary folder (git-ignored), emptied now and removed at stop
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True)
    explorer = core.Explorer(cfg, model_loader(cfg), TMP_DIR, KEEP_UPLOADS, DISPLAY_ALIGNMENT)

    # 3) server on this computer only
    server = make_server(explorer, HOST, PORT)
    url = f"http://{HOST}:{server.server_address[1]}/"
    print(f"Step 2: Segment Explorer at {url} (this computer only); models load on the first upload")
    print("        Stop with Ctrl+C (uploads are dropped and the temporary folder is removed)")
    if OPEN_BROWSER:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    signal.signal(signal.SIGTERM, stop_on_signal)    # a kill also drops the uploads and the temporary folder
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping")
    finally:
        server.server_close()
        explorer.clear()
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"Step 3: stopped; uploads dropped, {TMP_DIR} removed")


if __name__ == "__main__":
    main()
