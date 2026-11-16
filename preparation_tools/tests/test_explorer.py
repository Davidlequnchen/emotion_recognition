"""Segment Explorer (explorer/): settings, waveform peaks, the cached re-run, display-only alignment, the page data,
the self-contained export and the local server, with the fake models of fakes.py (no real models, no browser)."""
import http.client
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "explorer"))
from fakes import LONG_EN, RATE, SHORT_EN, SPANISH, FakeAligner, constant_probs  # noqa: E402
from segmenter import pipeline  # noqa: E402
import app  # noqa: E402
import core  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "segment_turns.yaml").read_text(encoding="utf-8"))
MODEL = "wav2vec2_asr_base_960h"
TEMPLATE = (ROOT / "explorer" / "page.html").read_text(encoding="utf-8")


def wav_bytes(seconds, rate=RATE):
    """A quiet test tone as WAV bytes (so the waveform peaks are not all zero)."""
    t = np.arange(int(seconds * rate)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, 0.1 * np.sin(2 * np.pi * 220 * t), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class CountingAligner(FakeAligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def align(self, wave, words):
        self.calls += 1
        return super().align(wave, words)


class FakeSession:
    """An Explorer with fake models; counts the VAD and alignment runs."""

    def __init__(self, tmp, probs=0.05):
        self.aligner = CountingAligner()
        self.vad_calls = 0
        fake_probs = constant_probs(probs)

        def speech_probabilities(model, wave):
            self.vad_calls += 1
            return fake_probs(model, wave)

        self.patch = mock.patch.object(pipeline.vad, "speech_probabilities", side_effect=speech_probabilities)
        self.patch.start()
        self.explorer = core.Explorer(CFG, lambda: (None, {MODEL: self.aligner}), tmp)

    def stop(self):
        self.patch.stop()


class Settings(unittest.TestCase):
    def test_defaults_are_unchanged(self):
        cfg, changed = core.apply_settings(CFG, core.editable_values(CFG))
        self.assertEqual((changed, pipeline.settings_version(cfg)), ([], pipeline.settings_version(CFG)))
        self.assertEqual(core.yaml_snippet(cfg, CFG, changed), "")

    def test_changed_values_version_and_yaml(self):
        cfg, changed = core.apply_settings(CFG, {"segmentation.target_s": 6.0, "scoring.weight_pause": 0.6})
        self.assertEqual(changed, ["segmentation.target_s", "scoring.weight_pause"])
        self.assertEqual(cfg["segmentation"]["target_s"], 6)               # 6.0 -> 6, as YAML writes it
        self.assertIsInstance(cfg["segmentation"]["target_s"], int)
        self.assertEqual(CFG["segmentation"]["target_s"], 7)               # the YAML copy is never changed
        self.assertNotEqual(pipeline.settings_version(cfg), pipeline.settings_version(CFG))
        snippet = core.yaml_snippet(cfg, CFG, changed)
        self.assertIn("segmentation:\n  target_s: 6  # YAML: 7\n", snippet)
        self.assertIn("scoring:\n  weight_pause: 0.6  # YAML: 0.5\n", snippet)
        pasted = yaml.safe_load(snippet)                                    # valid YAML with the new values
        self.assertEqual(pasted, {"segmentation": {"target_s": 6}, "scoring": {"weight_pause": 0.6}})

    def test_invalid_values(self):
        for values in ({"segmentation.min_s": 20}, {"segmentation.target_s": 0}, {"scoring.weight_text": 1.5},
                       {"segmentation.max_s": -1}, {"segmentation.max_s": "15"}, {"segmentation.max_s": True},
                       {"segmentation.max_s": float("nan")}, {"vad.trim_edges": True}):
            with self.assertRaises(ValueError, msg=values):
                core.apply_settings(CFG, values)


class Peaks(unittest.TestCase):
    def test_min_max_per_bin(self):
        samples = np.zeros(1000)
        samples[150], samples[420] = 0.5, -0.25
        peaks = core.waveform_peaks(samples, rate=1000, per_second=10)    # 10 bins of 100 samples
        self.assertEqual(len(peaks["max"]), 10)
        self.assertAlmostEqual(peaks["bin_s"], 0.1)
        self.assertEqual((peaks["max"][1], peaks["min"][4], peaks["max"][0]), (1000, -500, 0))
        self.assertEqual(peaks["peak"], 0.5)

    def test_bin_cap_and_short_input(self):
        peaks = core.waveform_peaks(np.ones(10_000), rate=1000, per_second=100, max_bins=7)
        self.assertLessEqual(len(peaks["max"]), 7)
        self.assertAlmostEqual(peaks["bin_s"] * len(peaks["max"]), 10.0, delta=peaks["bin_s"])
        self.assertEqual(core.waveform_peaks(np.zeros(3), 16000)["max"], [0])
        self.assertEqual(core.waveform_peaks(np.zeros(0), 16000)["max"], [])


class PrecomputedAnalysis(unittest.TestCase):
    def test_same_result_without_running_vad_again(self):
        data = np.zeros((35 * RATE, 1))
        aligners = {MODEL: FakeAligner()}
        with mock.patch.object(pipeline.vad, "speech_probabilities", side_effect=constant_probs(0.05)) as vad:
            analysis = pipeline.analyse_audio(data, RATE, None)
            first = pipeline.segment_turn(data, RATE, LONG_EN, CFG, None, aligners)
            self.assertEqual(vad.call_count, 2)
            again = pipeline.segment_turn(np.empty((data.shape[0], 0)), RATE, LONG_EN, CFG, None, aligners, analysis)
            self.assertEqual(vad.call_count, 2)                            # nothing recomputed
        self.assertEqual(first[0], again[0])
        self.assertEqual(first[1]["dp"], again[1]["dp"])


class CachedRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = FakeSession(self.tmp.name)
        self.explorer = self.session.explorer

    def tearDown(self):
        self.session.stop()
        self.tmp.cleanup()

    def test_rerun_uses_cache_and_changes_only_scoring(self):
        audio_id, upload = self.explorer.add_audio("turn.wav", wav_bytes(35))
        self.assertFalse(any(Path(self.tmp.name).iterdir()))               # temporary copy deleted at once
        first = self.explorer.run(audio_id, LONG_EN, "turn.txt")
        self.assertEqual((first["group"], first["status"], len(first["segments"])), ("final", "cut", 5))
        self.assertEqual(first["timings"]["alignment"], "computed")
        second = self.explorer.run(audio_id, LONG_EN, "turn.txt", {"segmentation.target_s": 14})
        self.assertEqual(second["timings"]["alignment"], "cached")
        self.assertEqual((self.session.vad_calls, self.session.aligner.calls), (1, 1))   # VAD and alignment once
        self.assertEqual([round(s["duration_s"]) for s in second["segments"]], [14, 14, 7])
        self.assertEqual(second["settings"]["changed"], ["segmentation.target_s"])
        self.assertNotEqual(second["settings"]["version"], first["settings"]["version"])
        self.assertEqual(self.explorer.add_audio("same.wav", wav_bytes(35))[0], audio_id)  # same bytes: cached
        self.assertEqual(self.session.vad_calls, 1)

    def test_payload_marks_chosen_gaps_and_is_json(self):
        audio_id, _ = self.explorer.add_audio("turn.wav", wav_bytes(35))
        result = self.explorer.run(audio_id, LONG_EN, "turn.txt")
        json.dumps(result)                                                # JSON-ready
        chosen = [g for g in result["gaps"] if g["chosen"]]
        self.assertEqual([g["cut_time"] for g in chosen], result["dp"]["cuts_s"])
        self.assertEqual([round(g["cut_time"], 2) for g in chosen], [round(s["start_s"], 2) for s in result["segments"][1:]])
        self.assertEqual(" ".join(s["text"] for s in result["segments"]), LONG_EN)
        self.assertEqual(len(result["words"]), 35)
        self.assertFalse(result["display_only"])
        self.assertEqual(len(result["vad"]["probs"]), int(np.ceil(35 * RATE / 512)))
        self.assertTrue(all(s["low_speech"] for s in result["segments"]))   # constant probability 0.05
        self.assertIn("cut by the dynamic programme into 5 segments", result["decision"])

    def test_short_turn_display_only_alignment(self):
        audio_id, _ = self.explorer.add_audio("short.wav", wav_bytes(8))
        result = self.explorer.run(audio_id, SHORT_EN, "short.txt")
        self.assertEqual((result["group"], result["status"], len(result["segments"])), ("final", "whole_short", 1))
        self.assertTrue(result["display_only"])
        self.assertIn("display only", result["display_note"])
        self.assertEqual(result["alignment"]["status"], "not_needed")        # the pipeline's own record
        self.assertEqual(len(result["words"]), len(SHORT_EN.split()))
        self.assertFalse(any(g["chosen"] for g in result["gaps"]))
        self.assertIsNone(result["dp"])
        # lowering keep_whole_max_s makes the pipeline cut it, reusing the display alignment from the cache
        cut = self.explorer.run(audio_id, SHORT_EN, "short.txt", {"segmentation.keep_whole_max_s": 5})
        self.assertEqual(cut["alignment"]["status"], "ok")
        self.assertEqual((cut["timings"]["alignment"], self.session.aligner.calls), ("cached", 1))

    def test_spanish_turn_and_switching_display_alignment_off(self):
        audio_id, _ = self.explorer.add_audio("es.wav", wav_bytes(20))
        result = self.explorer.run(audio_id, SPANISH, "es.txt")
        self.assertEqual((result["group"], result["reason"]), ("non_english", "language: es"))
        self.assertIn("non_english/", result["decision"])
        self.assertIn("rough", result["display_note"])
        self.assertTrue(result["display_only"] and result["words"])
        self.explorer.display_alignment = False
        plain = self.explorer.run(audio_id, SPANISH, "es.txt")
        self.assertEqual((plain["words"], plain["gaps"], plain["group"]), ([], [], "non_english"))

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.explorer.add_audio("notes.txt", b"abc")
        with self.assertRaises(KeyError):
            self.explorer.run("unknown", LONG_EN)
        with self.assertRaises(Exception):
            self.explorer.add_audio("broken.wav", b"not audio at all")
        self.assertFalse(any(Path(self.tmp.name).iterdir()))

    def test_keeps_only_the_last_uploads(self):
        self.explorer.keep_uploads = 2
        ids = [self.explorer.add_audio(f"t{k}.wav", wav_bytes(2 + k))[0] for k in range(3)]
        self.assertEqual(list(self.explorer.uploads), ids[1:])


class Export(unittest.TestCase):
    def test_self_contained_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = FakeSession(tmp)
            try:
                text = LONG_EN.replace("help", "help</script><script>alert(1)</script>", 1)   # still one token
                audio_id, upload = session.explorer.add_audio("turn.wav", wav_bytes(35))
                result = session.explorer.run(audio_id, text, "turn.txt", {"segmentation.target_s": 8})
            finally:
                session.stop()
        page = core.build_export_html(TEMPLATE, result, upload["raw"], upload["mime"])
        for live_only in ('id="btnRun"', 'id="btnRerun"', 'id="btnExport"', 'id="fileAudio"', 'id="fileText"',
                          "<!--LIVE-->", "Run segmenter"):
            self.assertNotIn(live_only, page)
        self.assertIn("connect-src 'none'", page)
        self.assertNotIn("connect-src 'self'", page)
        self.assertEqual(re.findall(r"https?://", page), [])
        self.assertIn("approved internal channels", page)
        self.assertEqual(page.lower().count("</script>"), 3)               # data, audio, code: none from the text
        data = re.search(r'<script id="explorer-data" type="application/json">(.*?)</script>', page, re.S).group(1)
        embedded = json.loads(data)
        self.assertEqual(embedded["segments"], result["segments"])
        self.assertEqual(embedded["settings"]["values"]["segmentation.target_s"], 8)
        self.assertEqual(embedded["settings"]["changed"], ["segmentation.target_s"])
        uri = re.search(r'<script id="explorer-audio" type="text/plain">(.*?)</script>', page, re.S).group(1)
        self.assertTrue(uri.startswith("data:audio/wav;base64,"))
        import base64
        self.assertEqual(base64.b64decode(uri.split(",", 1)[1]), upload["raw"])
        self.assertEqual(core.export_name("call 01/turn A.wav"), "segment_explorer_turn_A.html")

    def test_template_has_no_external_urls(self):
        self.assertEqual(re.findall(r"https?://", TEMPLATE), [])
        self.assertEqual(TEMPLATE.count("<!--LIVE-->"), TEMPLATE.count("<!--/LIVE-->"))


class Server(unittest.TestCase):
    """The local server end to end with fake models: upload, run, re-run from cache, export, and the guards."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.session = FakeSession(self.tmp.name)
        self.server = app.make_server(self.session.explorer, port=0)
        self.port = self.server.server_address[1]
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.session.stop()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, dict(response.getheaders()), data

    def post_json(self, path, obj):
        return self.request("POST", path, json.dumps(obj).encode(), {"Content-Type": "application/json"})

    def test_upload_run_rerun_export(self):
        status, headers, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Run segmenter", page)
        status, _, body = self.request("GET", "/api/defaults")
        self.assertEqual(json.loads(body)["values"], core.editable_values(CFG))
        status, _, body = self.request("POST", "/api/audio", wav_bytes(35),
                                       {"Content-Type": "application/octet-stream", "X-File-Name": "d418%20long.wav"})
        self.assertEqual(status, 200, body)
        audio_id = json.loads(body)["audio_id"]
        request = {"audio_id": audio_id, "transcript": LONG_EN, "transcript_name": "d418 long.txt"}
        status, _, body = self.post_json("/api/run", request)
        first = json.loads(body)
        self.assertEqual((status, first["status"], first["files"]["audio"]), (200, "cut", "d418 long.wav"))
        status, _, body = self.post_json("/api/run", {**request, "settings": {"segmentation.target_s": 14}})
        second = json.loads(body)
        self.assertEqual((len(first["segments"]), len(second["segments"])), (5, 3))
        self.assertEqual((self.session.vad_calls, self.session.aligner.calls), (1, 1))
        status, headers, page = self.post_json("/api/export", {**request, "settings": second["settings"]["values"]})
        self.assertEqual(status, 200)
        self.assertIn('filename="segment_explorer_d418_long.html"', headers["Content-Disposition"])
        self.assertIn(b"connect-src 'none'", page)
        self.assertNotIn(b'id="btnRun"', page)
        status, _, body = self.post_json("/api/run", {**request, "settings": {"segmentation.min_s": 99}})
        self.assertEqual(status, 400)
        self.assertIn("min_s", json.loads(body)["error"])
        status, _, body = self.post_json("/api/run", {**request, "audio_id": "gone"})
        self.assertEqual((status, json.loads(body)["code"]), (404, "unknown_audio"))

    def test_guards(self):
        status, _, _ = self.request("GET", "/", headers={"Host": f"attacker.example:{self.port}"})
        self.assertEqual(status, 403)                                     # DNS rebinding
        status, _, _ = self.request("POST", "/api/run", b"{}", {"Content-Type": "application/json",
                                                                "Origin": "http://attacker.example"})
        self.assertEqual(status, 403)                                     # another web page
        status, _, _ = self.request("POST", "/api/run", b"{}", {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)                                     # only JSON or raw audio bytes
        status, _, _ = self.request("GET", "/../config/segment_turns.yaml")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
