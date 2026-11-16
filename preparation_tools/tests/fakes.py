"""Stand-ins for the models and tiny synthetic turns, shared by the unit tests (no models, no real audio)."""
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
LONG_EN = " ".join(["I need some help with my bill."] * 5)        # 35 words, a sentence end every 7th
SHORT_EN = "Okay, thank you so much for your help."
SPANISH = "Hola, me cobraron dos veces en mi cuenta y quiero saber por qué."
MIXED = "Yes I called yesterday pero nadie me ayudó con el problema, and now the bill is wrong."


class FakeAligner:
    """Instead of segmenter.alignment.Aligner: word k gets [k + 0.1, k + 0.9] s unless spans are given."""
    parameters = 0

    def __init__(self, *args, spans=None, confidence=0.9, fail=None, **kwargs):
        self.spans, self.confidence, self.fail = spans, confidence, fail

    def align(self, wave, words):
        if self.fail:
            return False, self.fail
        for k, word in enumerate(words):
            word.start, word.end = self.spans[k] if self.spans else (k + 0.1, k + 0.9)
            word.confidence = self.confidence
        return True, "ok"


def constant_probs(value):
    """A speech_probabilities replacement: the same probability for every 32 ms frame of the wave."""
    return lambda model, wave: np.full(int(np.ceil(wave.numel() / 512)), value)


def write_turn(folder, turn_id, seconds, text):
    """A silent WAV turn of the given length and its transcript."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / f"{turn_id}.wav"), np.zeros((int(seconds * RATE), 1)), RATE, subtype="PCM_16")
    (folder / f"{turn_id}.txt").write_text(text, encoding="utf-8")
