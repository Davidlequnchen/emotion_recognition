"""Silero VAD speech probabilities, one per 32 ms frame (doc §3).

Only the frame probabilities are used: to score how quiet a word gap is, to place the cut at the quietest
frame inside the chosen gap, to measure speech time, and to find long silence at the turn edges.
VAD never deletes audio inside a turn. The JIT model is packaged inside the silero-vad wheel (no download).
"""
import numpy as np
import torch

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512                       # Silero's window at 16 kHz
FRAME_S = FRAME_SAMPLES / SAMPLE_RATE     # 0.032 s


def load_vad():
    """Load the Silero JIT model on the CPU; keep PyTorch's thread setting (silero sets 1 thread on import)."""
    threads = torch.get_num_threads()
    from silero_vad import load_silero_vad
    model = load_silero_vad(onnx=False)
    torch.set_num_threads(threads)
    return model


@torch.inference_mode()
def speech_probabilities(model, wave):
    """1-D float tensor at 16 kHz -> numpy array of speech probabilities, frame i covers [i, i+1) x 32 ms."""
    model.reset_states()
    probs = []
    for start in range(0, wave.numel(), FRAME_SAMPLES):
        chunk = wave[start:start + FRAME_SAMPLES]
        if chunk.numel() < FRAME_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, FRAME_SAMPLES - chunk.numel()))
        probs.append(model(chunk, SAMPLE_RATE).item())
    return np.asarray(probs, dtype=np.float64)


def frame_centres(probs):
    return (np.arange(len(probs)) + 0.5) * FRAME_S


QUIET_TOLERANCE = 0.1  # frames within this of the gap's lowest probability count as equally quiet


def gap_silence(probs, start, end, tolerance=QUIET_TOLERANCE):
    """(v, cut time) for a gap [start, end]: v = 1 - mean probability inside the gap; the cut goes to the centre
    of the quiet frame nearest the gap middle (quiet = within `tolerance` of the lowest probability in the gap),
    or to the middle of a gap shorter than one frame. Taking the middle of the quiet part, not the single lowest
    frame, splits a pause evenly and keeps the cut away from word onsets. The cut always stays within [start, end]."""
    middle = (start + end) / 2
    centres = frame_centres(probs)
    inside = np.flatnonzero((centres >= start) & (centres <= end))
    if inside.size == 0:
        frame = min(int(middle / FRAME_S), len(probs) - 1)
        return 1.0 - float(probs[frame]), middle
    values = probs[inside]
    quiet = inside[values <= values.min() + tolerance]
    best = quiet[np.argmin(np.abs(centres[quiet] - middle))]
    return 1.0 - float(values.mean()), float(centres[best])


def speech_seconds(probs, start, end, threshold):
    """Seconds of frames with probability >= threshold whose centre lies in [start, end)."""
    centres = frame_centres(probs)
    return float(np.sum((centres >= start) & (centres < end) & (probs >= threshold)) * FRAME_S)


def speech_edges(probs, threshold):
    """(first speech time, last speech time) or None when no frame reaches the threshold."""
    speech = np.flatnonzero(probs >= threshold)
    if speech.size == 0:
        return None
    return float(speech[0] * FRAME_S), float((speech[-1] + 1) * FRAME_S)
