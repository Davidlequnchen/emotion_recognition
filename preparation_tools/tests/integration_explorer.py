#!/usr/bin/env python3
"""End-to-end check of the Segment Explorer with the real models, on synthetic DailyTalk turns made by
tests/integration_dailytalk.py (run that first). Not part of unittest.

Starts the explorer's server on a free port of 127.0.0.1 (no browser), then over HTTP: uploads a long English turn
and its transcript, runs the segmenter, compares the cuts with the batch run's review file, re-runs with changed
settings from the cache, and exports the static page; then a short English turn, a long Spanish-text turn and the
MP3 turn. The exported examples go to local/explorer_examples/ (git-ignored); they contain the synthetic audio.
"""
import http.client
import json
from pathlib import Path
import re
import shutil
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "explorer"))
import app  # noqa: E402
import core  # noqa: E402

TURNS = ROOT / "local" / "integration" / "input" / "turns"
REVIEW = ROOT / "local" / "integration" / "output" / "review"
EXAMPLES = ROOT / "local" / "explorer_examples"
TMP = ROOT / "local" / "explorer_tmp_check"
LONG, SHORT, SPANISH, MP3 = "d418_long_en.wav", "d418_short_en.wav", "d418_es.wav", "d2107_long_en.mp3"
CHANGED = {"segmentation.target_s": 5, "segmentation.cut_weight_lambda": 0.5}


class Client:
    def __init__(self, port):
        self.port = port

    def call(self, path, body, content_type, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        connection.request("POST", path, body=body, headers={"Content-Type": content_type, **(headers or {})})
        response = connection.getresponse()
        data = response.read()
        connection.close()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {data[:300]!r}")
        return data, dict(response.getheaders())

    def upload(self, name):
        data, _ = self.call("/api/audio", (TURNS / name).read_bytes(), "application/octet-stream",
                            {"X-File-Name": name})
        return json.loads(data)["audio_id"]

    def run(self, audio_id, name, settings=None, path="/api/run"):
        text_path = (TURNS / name).with_suffix(".txt")
        body = json.dumps({"audio_id": audio_id, "transcript": text_path.read_text(encoding="utf-8"),
                           "transcript_name": text_path.name, "settings": settings or {}}).encode()
        data, headers = self.call(path, body, "application/json")
        return (json.loads(data), headers) if path == "/api/run" else (data.decode("utf-8"), headers)


def check_export(page, name):
    """Save one exported page and check that it is self-contained."""
    urls = re.findall(r"https?://[^\s\"'<>]*", page)
    assert not urls, urls
    assert "connect-src 'none'" in page and 'id="btnRun"' not in page
    assert "approved internal channels" in page
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    (EXAMPLES / name).write_text(page, encoding="utf-8")
    print(f"  exported {EXAMPLES / name}: {len(page) / 1e6:.2f} MB, no external URLs")


def describe(result):
    lengths = ", ".join(f"{s['duration_s']:.2f}" for s in result["segments"])
    return (f"{result['group']}/{result['status']}, {len(result['segments'])} segments [{lengths}] s, "
            f"alignment {result['timings']['alignment']}, run {result['timings']['run_s']:.3f} s")


def main():
    free = app.free_memory_percent()
    print(f"Step 0: free memory {free}% (need {app.MIN_FREE_MEMORY}%)")
    if free is not None and free < app.MIN_FREE_MEMORY:
        print("Skipped: not enough free memory for the model run.")
        return 2
    if not (TURNS / LONG).is_file():
        print(f"Missing {TURNS / LONG}: run tests/integration_dailytalk.py first.")
        return 2
    cfg = yaml.safe_load(app.CONFIG.read_text(encoding="utf-8"))
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    explorer = core.Explorer(cfg, app.model_loader(cfg), TMP, app.KEEP_UPLOADS, app.DISPLAY_ALIGNMENT)
    server = app.make_server(explorer, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = Client(server.server_address[1])
    print(f"Step 1: server on 127.0.0.1:{server.server_address[1]}")
    try:
        # long English turn: same cuts as the batch, re-run from the cache, export
        print(f"Step 2: {LONG}")
        t0 = time.time()
        audio_id = client.upload(LONG)
        result, _ = client.run(audio_id, LONG)
        print(f"  first run (models, decoding, VAD, alignment): {time.time() - t0:.1f} s -> {describe(result)}")
        assert (result["group"], result["status"]) == ("final", "cut")
        batch = json.loads((REVIEW / Path(LONG).with_suffix(".json").name).read_text(encoding="utf-8"))
        same = result["dp"]["cuts_s"] == batch["dp"]["cuts_s"]
        print(f"  cuts {result['dp']['cuts_s']} vs batch review {batch['dp']['cuts_s']}: "
              f"{'identical' if same else 'DIFFERENT'}; settings {result['settings']['version']} "
              f"(batch {batch['segmentation_version']})")
        t0 = time.time()
        rerun, _ = client.run(audio_id, LONG, CHANGED)
        print(f"  re-run with {CHANGED}: {time.time() - t0:.3f} s over HTTP -> {describe(rerun)}")
        assert rerun["timings"]["alignment"] == "cached" and rerun["settings"]["changed"] == list(CHANGED)
        page, headers = client.run(audio_id, LONG, rerun["settings"]["values"], "/api/export")
        check_export(page, "long_cut_" + core.export_name(LONG))
        # short English turn: whole, display-only alignment
        print(f"Step 3: {SHORT}")
        result, _ = client.run(client.upload(SHORT), SHORT)
        print(f"  {describe(result)}; display only: {result['display_only']}, {len(result['words'])} words")
        assert (result["group"], result["status"], result["display_only"]) == ("final", "whole_short", True)
        # long Spanish-text turn: non_english, display-only alignment, export
        print(f"Step 4: {SPANISH}")
        audio_id = client.upload(SPANISH)
        result, _ = client.run(audio_id, SPANISH)
        print(f"  {describe(result)}; {result['duration_s']:.1f} s; reason {result['reason']!r}; "
              f"display only: {result['display_only']}, {len(result['words'])} words")
        assert (result["group"], result["status"]) == ("non_english", "whole_language")
        page, _ = client.run(audio_id, SPANISH, None, "/api/export")
        check_export(page, "non_english_" + core.export_name(SPANISH))
        # MP3 turn
        print(f"Step 5: {MP3}")
        result, _ = client.run(client.upload(MP3), MP3)
        batch = json.loads((REVIEW / Path(MP3).with_suffix(".json").name).read_text(encoding="utf-8"))
        print(f"  {describe(result)}; cuts {'identical to' if result['dp']['cuts_s'] == batch['dp']['cuts_s'] else 'DIFFERENT from'} the batch")
        assert result["status"] == "cut"
    finally:
        server.shutdown()
        server.server_close()
        explorer.clear()
    leftovers = [p.name for p in TMP.iterdir()]
    shutil.rmtree(TMP)
    print(f"Step 6: temporary uploads left: {leftovers or 'none'}; {TMP} removed")
    return 1 if leftovers else 0


if __name__ == "__main__":
    sys.exit(main())
