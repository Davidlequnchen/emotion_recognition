"""Cut placement, word-to-segment assignment and the routing of turns to the four groups, on synthetic word
timings (no models needed)."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fakes import LONG_EN, MIXED, RATE, SHORT_EN, SPANISH, FakeAligner, constant_probs  # noqa: E402
from segmenter import pipeline, vad  # noqa: E402
from segmenter.boundaries import assign_words  # noqa: E402
from segmenter.pipeline import alignment_quality, choose_cuts, trim_span  # noqa: E402
from segmenter.textnorm import tokenize  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "segment_turns.yaml").read_text(encoding="utf-8"))


def timed_words(text, spans, confidence=0.9):
    words = tokenize(text)
    for word, (start, end) in zip(words, spans):
        word.start, word.end, word.confidence = start, end, confidence
    return words


def probs_for(words, total):
    """Speech probability 0.95 inside words, 0.05 in pauses."""
    probs = np.full(int(np.ceil(total / vad.FRAME_S)), 0.05)
    centres = vad.frame_centres(probs)
    for w in words:
        probs[(centres >= w.start) & (centres <= w.end)] = 0.95
    return probs


def slow_turn(n=60, word_s=0.35, pause_s=0.15, long_every=6, long_s=1.0):
    """n words; a long pause after every long_every-th word, which ends a sentence."""
    text, spans, t = [], [], 0.3
    for k in range(n):
        sentence_end = (k + 1) % long_every == 0
        text.append("word." if sentence_end else "word")
        spans.append((round(t, 3), round(t + word_s, 3)))
        t += word_s + (long_s if sentence_end else pause_s)
    return " ".join(text), spans, t + 0.3


class GapSilence(unittest.TestCase):
    def test_cut_at_quietest_frame_inside_gap(self):
        probs = np.array([0.9, 0.9, 0.4, 0.1, 0.3, 0.9, 0.9])          # frames of 32 ms
        v, cut = vad.gap_silence(probs, 0.064, 0.160)                   # frames 2-4 lie inside
        self.assertAlmostEqual(cut, 3.5 * vad.FRAME_S)
        self.assertAlmostEqual(v, 1 - np.mean([0.4, 0.1, 0.3]))

    def test_cut_in_the_middle_of_a_quiet_plateau(self):
        probs = np.array([0.9, 0.02, 0.06, 0.05, 0.08, 0.04, 0.9])     # near-silent pause, lowest frame first
        v, cut = vad.gap_silence(probs, 0.032, 0.192)                   # frames 1-5 lie inside
        self.assertAlmostEqual(cut, 3.5 * vad.FRAME_S)                  # middle of the pause, not frame 1

    def test_short_gap_cut_in_the_middle(self):
        v, cut = vad.gap_silence(np.array([0.9, 0.2, 0.9]), 0.040, 0.044)   # no frame centre inside
        self.assertAlmostEqual(cut, 0.042)
        self.assertAlmostEqual(v, 0.8)


class Assignment(unittest.TestCase):
    def test_every_word_once_in_order(self):
        words = timed_words("a b c d", [(0.1, 0.5), (0.7, 1.0), (1.4, 1.8), (2.0, 2.5)])
        self.assertEqual(assign_words(words, [0.0, 1.2, 3.0]), [0, 0, 1, 1])

    def test_cut_inside_a_word_is_rejected(self):
        words = timed_words("a b", [(0.1, 0.5), (0.7, 1.0)])
        with self.assertRaises(AssertionError):
            assign_words(words, [0.0, 0.8, 2.0])


class ChooseCuts(unittest.TestCase):
    def test_slow_speaker_cuts_at_sentence_ends(self):
        text, spans, total = slow_turn()
        words = timed_words(text, spans)
        cuts, review, status = choose_cuts(words, probs_for(words, total), text, 0.0, total, CFG)
        self.assertEqual(status, "cut")
        bounds = [0.0] + [c["cut_time"] for c in cuts] + [total]
        lengths = np.diff(bounds)
        self.assertTrue(all(2.0 <= d <= 15.0 for d in lengths), lengths)
        self.assertTrue(all(c["q"] == 1.0 for c in cuts))               # all cuts at the long sentence pauses
        self.assertEqual(review["dp"]["limits_s"], [2, 15])
        assign_words(words, bounds)

    def test_no_cut_next_to_low_confidence_word(self):
        text, spans, total = slow_turn()
        words = timed_words(text, spans)
        cuts, *_ = choose_cuts(words, probs_for(words, total), text, 0.0, total, CFG)
        weak = cuts[0]["after_word"]
        words[weak].confidence = 0.05
        cuts2, review, _ = choose_cuts(words, probs_for(words, total), text, 0.0, total, CFG)
        self.assertNotIn(weak, [c["after_word"] for c in cuts2])
        self.assertFalse(review["gaps"][weak]["candidate"])

    def test_one_window_no_second_pass(self):
        # 30 one-second words; low-confidence words leave a stretch without any usable gap
        def run(weak_words):
            text = " ".join(["word"] * 30)
            words = timed_words(text, [(k + 0.1, k + 0.9) for k in range(30)])
            for k in weak_words:
                words[k].confidence = 0.05
            return choose_cuts(words, probs_for(words, 30.0), text, 0.0, 30.0, CFG)

        cuts, review, status = run(range(5, 17))                        # about 14 s without a cut point: fits
        self.assertEqual(status, "cut")
        bounds = [0.0] + [c["cut_time"] for c in cuts] + [30.0]
        self.assertAlmostEqual(max(np.diff(bounds)), 14.0, delta=0.1)
        cuts, review, status = run(range(3, 21))                        # about 20 s: no partition, keep whole
        self.assertEqual((status, cuts), ("whole_boundary", []))
        self.assertEqual(review["dp"], {"feasible": False, "limits_s": [2, 15]})

    def test_alignment_quality(self):
        words = timed_words("a b c [NAME]", [(0, 1), (1, 2), (2, 3), (3, 4)], confidence=0.8)
        self.assertEqual(alignment_quality(words, CFG["alignment"]), ([], 0.8, 0.25))
        words[0].confidence = words[1].confidence = 0.1
        problems, mean_conf, share = alignment_quality(words, CFG["alignment"])
        self.assertEqual(problems, ["mean confidence 0.33 < 0.5", "75% of words weak or unspelled > 40%"])


class Routing(unittest.TestCase):
    """segment_turn puts every turn into exactly one of the four groups; the language check comes first."""

    def route(self, seconds, text, aligner=None, probs=0.05):
        data = np.zeros((int(seconds * RATE), 1))
        aligners = {"wav2vec2_asr_base_960h": aligner or FakeAligner()}
        with mock.patch.object(pipeline.vad, "speech_probabilities", side_effect=constant_probs(probs)):
            segments, info = pipeline.segment_turn(data, RATE, text, CFG, None, aligners)
        return segments, info

    def assert_whole(self, segments, seconds):
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0]["start_sample"], segments[0]["end_sample"]), (0, int(seconds * RATE)))

    def test_short_english_is_final_whole(self):
        segments, info = self.route(5.0, SHORT_EN, aligner=FakeAligner(fail="must not be called"))
        self.assertEqual((info["group"], info["status"], info["reason"]), ("final", "whole_short", ""))
        self.assert_whole(segments, 5.0)

    def test_long_english_is_cut_into_final(self):
        segments, info = self.route(35.0, LONG_EN)
        self.assertEqual((info["group"], info["status"], info["reason"]), ("final", "cut", ""))
        self.assertEqual([round(s["duration_s"]) for s in segments], [7, 7, 7, 7, 7])   # cuts at sentence ends
        self.assertEqual(" ".join(s["text"] for s in segments), LONG_EN)

    def test_ten_to_fifteen_seconds_can_stay_whole_in_final(self):
        text = " ".join(["word"] * 12)                                  # no punctuation, no quiet gap
        segments, info = self.route(11.6, text, probs=0.95)
        self.assertEqual((info["group"], info["status"]), ("final", "whole_dp"))
        self.assert_whole(segments, 11.6)

    def test_short_spanish_is_non_english(self):
        segments, info = self.route(3.0, "Sí, gracias. Está bien, señora.")
        self.assertEqual((info["group"], info["status"], info["reason"]), ("non_english", "whole_language", "language: es"))
        self.assert_whole(segments, 3.0)

    def test_long_spanish_and_mixed_are_non_english_whole(self):
        for text, language in ((SPANISH, "es"), (MIXED, "mixed")):
            segments, info = self.route(25.0, text, aligner=FakeAligner(fail="must not be called"))
            self.assertEqual((info["group"], info["reason"]), ("non_english", f"language: {language}"))
            self.assertEqual(info["alignment"]["status"], "skipped_language")
            self.assert_whole(segments, 25.0)

    def test_alignment_failure_and_low_confidence(self):
        segments, info = self.route(35.0, LONG_EN, aligner=FakeAligner(fail="transcript too long for the audio"))
        self.assertEqual((info["group"], info["status"], info["reason"]),
                         ("alignment_issues", "whole_alignment", "alignment: transcript too long for the audio"))
        self.assert_whole(segments, 35.0)
        segments, info = self.route(35.0, LONG_EN, aligner=FakeAligner(confidence=0.41))
        self.assertEqual((info["group"], info["reason"]), ("alignment_issues", "alignment: mean confidence 0.41 < 0.5"))
        self.assert_whole(segments, 35.0)

    def test_long_turn_without_text(self):
        segments, info = self.route(20.0, "  \n")
        self.assertEqual((info["group"], info["status"], info["reason"]),
                         ("alignment_issues", "whole_no_text", "alignment: empty transcript"))
        self.assert_whole(segments, 20.0)

    def test_no_partition_goes_to_boundary_issues(self):
        # the second word lasts 17.8 s, so every partition has a piece under 2 s or over 15 s
        spans = [(0.1, 0.9), (1.1, 18.9), (19.1, 19.9), (20.1, 20.9)]
        segments, info = self.route(22.0, "one two three four", aligner=FakeAligner(spans=spans))
        self.assertEqual((info["group"], info["status"], info["reason"]),
                         ("boundary_issues", "whole_boundary", "no partition within 2-15 s"))
        self.assert_whole(segments, 22.0)


class EdgeTrimming(unittest.TestCase):
    """VAD that misses the first 3 s of a 9 s turn must never cost audio in a whole-turn export."""

    def run_short_turn(self, trim):
        cfg = copy.deepcopy(CFG)
        cfg["vad"]["trim_edges"] = trim
        rate, total = 16000, 9.0
        probs = np.full(int(total / vad.FRAME_S), 0.05)
        probs[int(3.0 / vad.FRAME_S):] = 0.95         # VAD hears speech only from 3 s on
        data = np.zeros((int(total * rate), 1))
        with mock.patch.object(pipeline.vad, "speech_probabilities", return_value=probs):
            segments, info = pipeline.segment_turn(data, rate, "hello there I need help", cfg, None, {})
        return segments, info

    def test_default_keeps_whole_turn(self):
        self.assertFalse(CFG["vad"]["trim_edges"])
        segments, info = self.run_short_turn(trim=False)
        self.assertEqual(info["status"], "whole_short")
        self.assertEqual((segments[0]["start_sample"], segments[0]["end_sample"]), (0, 9 * 16000))

    def test_whole_export_untrimmed_even_with_trimming_on(self):
        segments, info = self.run_short_turn(trim=True)
        self.assertEqual(info["status"], "whole_short")
        self.assertEqual((segments[0]["start_sample"], segments[0]["end_sample"]), (0, 9 * 16000))
        self.assertAlmostEqual(info["decision_span_s"][0], 2.8, delta=0.04)  # only the decision span is trimmed


class Trim(unittest.TestCase):
    def test_trims_only_long_edge_silence(self):
        settings = {"trim_edges": True, "trim_min_silence_s": 1.0, "trim_margin_s": 0.2}
        self.assertEqual(trim_span((2.0, 8.0), 12.0, settings), (1.8, 8.2))
        self.assertEqual(trim_span((0.5, 11.5), 12.0, settings), (0.0, 12.0))
        self.assertEqual(trim_span(None, 12.0, settings), (0.0, 12.0))


if __name__ == "__main__":
    unittest.main()
