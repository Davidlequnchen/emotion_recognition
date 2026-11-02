"""Shared loaders for the presentation assets (make_figures.py, make_audio.py, make_candidates.py).

Everything is read from this repository: our evaluation outputs (outputs/<model>/<dataset>_predictions.csv,
reports/summary.json) and the dataset metadata in data/ (download the datasets first with
scripts/common/download_data.py). Nothing here writes files. Standard library + numpy + openpyxl only (no pandas).
"""
import csv
import json
import re
from pathlib import Path

import openpyxl

# ---------------------------------------------------------------- paths (found from this file, no CLI arguments)
SCRIPTS = Path(__file__).resolve().parent      # emotion_recognition/presentation/scripts
DECK = SCRIPTS.parent                          # emotion_recognition/presentation
ER = DECK.parent                               # emotion_recognition (project folder: outputs/, reports/, data/)
ASSETS = DECK / "assets"                       # what the deck reads: fig/, audio/, stats.json, manifest.json

# ---------------------------------------------------------------- models, datasets, label orders
MODELS = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper", "vox_profile_wavlm"]
MODEL_NAMES = {
    "emotion2vec_plus_large": "emotion2vec+ large",
    "meralion_ser_v1": "MERaLiON-SER-v1",
    "melati_emotion": "Melati",
    "vox_profile_whisper": "Vox-Profile Whisper",
    "vox_profile_wavlm": "Vox-Profile WavLM",
}
# Only datasets whose licence allows public clips (never ESD or THAI-SER).
DATASETS = ["crema_d", "mcae_spps", "mess", "ravdess", "tess", "jl_corpus"]
DATASET_NAMES = {"crema_d": "CREMA-D", "mcae_spps": "MCAE-SPPS", "mess": "MESS", "ravdess": "RAVDESS",
                 "tess": "TESS", "jl_corpus": "JL-Corpus"}
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
ORDER = ["angry", "disgusted", "fearful", "excited", "happy", "neutral", "sad", "surprised"]  # row order on figures

# ---------------------------------------------------------------- dataset metadata
CREMA_CODE = {"A": "angry", "D": "disgusted", "F": "fearful", "H": "happy", "N": "neutral", "S": "sad"}
CREMA_SENT = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.", "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.", "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.", "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
CREMA_LEVEL = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
RAVDESS_SENT = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}
# MCAE-SPPS vote columns (Corpus information.xlsx) -> our labels; this order is used to list the votes
MCAE_COLS = {"Neutral": "neutral", "Happiness": "happy", "Anger": "angry", "Fear": "fearful",
             "Sadness": "sad", "Disgust": "disgusted", "Surprise": "surprised"}
# English translations of the MCAE-SPPS sentences that appear on the slides
MCAE_EN = {
    "他们听了一个讲座": "They attended a lecture.", "他去看看": "He goes to take a look.",
    "你们看过这本书": "You (plural) have read this book.", "我看着远处": "I look into the distance.",
    "他听了一个讲座": "He attended a lecture.", "他们有个计划": "They have a plan.", "我去买菜": "I go to buy groceries.",
    "我想起了从前": "I recalled the past.", "你们去买菜": "You (plural) go to buy groceries.",  # candidates
}


def load_summary():
    """reports/summary.json: accuracy per model and dataset, and whether the dataset is in the model's training data."""
    return json.loads((ER / "reports" / "summary.json").read_text(encoding="utf-8"))


def training_membership(summary):
    """{model: set of datasets in its training data}, from summary.json (in_training_data)."""
    res = summary["results"]
    return {m: {ds for ds in DATASETS if res[m][ds]["in_training_data"]} for m in MODELS}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_predictions(dataset):
    """One dict per clip (in the order of the first model's CSV): audio_path, stem, acted_emotion, listener_vote (if
    the dataset has one) and for every model m: pred_m (label), prob_m (the model's probability of that label),
    correct_m (bool) and predall_m (prediction over all classes)."""
    rows, index = None, None
    for m in MODELS:
        recs = read_csv(ER / "outputs" / m / f"{dataset}_predictions.csv")
        if rows is None:
            rows = []
            for r in recs:
                row = {"audio_path": r["audio_path"], "acted_emotion": r["acted_emotion"],
                       "stem": Path(r["audio_path"]).stem}
                if "listener_vote" in r:
                    row["listener_vote"] = r["listener_vote"]
                rows.append(row)
            index = {row["audio_path"]: row for row in rows}
        for r in recs:
            row = index[r["audio_path"]]
            row[f"pred_{m}"] = r["prediction"]
            row[f"prob_{m}"] = float(r[f"p_{r['prediction']}"])
            row[f"correct_{m}"] = r["correct"] == "True"
            row[f"predall_{m}"] = r["prediction_all_classes"]
    assert all(f"pred_{m}" in row for row in rows for m in MODELS), f"{dataset}: a model is missing clips"
    return rows


def crema_votes():
    """{file name: {A, D, F, H, N, S, numResponses}} voice-only vote counts from tabulatedVotes.csv
    (the first digit of the response id is 1 for voice-only ratings)."""
    out = {}
    for r in read_csv(ER / "data" / "crema_d" / "processedResults" / "tabulatedVotes.csv"):
        if str(r[""]).startswith("1"):
            out[r["fileName"]] = {k: int(r[k]) for k in list(CREMA_CODE) + ["numResponses"]}
    return out


def crema_demographics():
    """{actor id: {Age, Sex}} from VideoDemographics.csv."""
    return {int(r["ActorID"]): r for r in read_csv(ER / "data" / "crema_d" / "VideoDemographics.csv")}


def mcae_info():
    """{file stem: row} from MCAE-SPPS Corpus information.xlsx (column names stripped; 'rr' = recognition rate,
    the share of the 40 listeners who chose the intended emotion)."""
    ws = openpyxl.load_workbook(ER / "data" / "mcae_spps" / "Corpus information.xlsx", read_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    head = [str(c).strip() for c in next(it)]
    out = {}
    for vals in it:
        if vals[0] is None:
            continue
        r = dict(zip(head, vals))
        r["rr"] = float(r["Recognition rate"])
        out[str(r["Filename"])] = r
    return out


def parse_meta(dataset, stem, audio_path=None):
    """Speaker, sentence id and (where known) sentence text and intensity of a clip, from its file name."""
    if dataset == "crema_d":
        a, s, e, lvl = stem.split("_")
        return {"speaker": a, "sentence_id": s, "sentence": CREMA_SENT[s], "intensity": CREMA_LEVEL[lvl]}
    if dataset == "mcae_spps":
        return {"speaker": stem[0], "sentence_id": stem[2:]}
    if dataset == "mess":
        m = re.match(r"([ACHS])([MF])(\d)([VTM]\d\d)", stem)
        return {"speaker": m.group(2) + m.group(3), "sentence_id": m.group(4)}
    if dataset == "ravdess":
        p = stem.split("-")
        return {"speaker": p[6], "sentence_id": p[4], "sentence": RAVDESS_SENT[p[4]],
                "intensity": {"01": "normal", "02": "strong"}[p[3]], "repetition": p[5]}
    if dataset == "tess":
        sp, word, _ = stem.split("_")
        sp = "OAF" if sp == "OA" else sp
        return {"speaker": sp, "sentence_id": word, "sentence": f'"Say the word {word}."'}
    if dataset == "jl_corpus":
        sp, emo, sent, rep = stem.split("_")
        txt = (ER / Path(audio_path).with_suffix(".txt")).read_text(encoding="utf-8", errors="replace").strip()
        return {"speaker": sp, "sentence_id": re.sub(r"[a-z]$", "", sent), "session": sent[-1], "repetition": rep,
                "sentence": txt}
    raise ValueError(dataset)


# ================================================================ audio clips (make_audio.py, make_candidates.py)
AUDIO_TARGET_DB = -18.0      # RMS of the loudest clip of a group (dBFS)
AUDIO_MAX_DROP_DB = 8.0      # quieter clips of a group keep half of their level difference, at most this much
AUDIO_PEAK_DB = -1.0         # peak ceiling (dBFS)
TRIM_DB, FRAME_S, PAD_S = 35, 0.02, 0.12  # trim frames 35 dB below the loudest 20 ms frame, keep 0.12 s padding
MP3_ARGS = ["-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k", "-map_metadata", "-1"]


class ClipTables:
    """Everything needed to describe a clip: predictions of all models, listener votes, speaker data."""

    def __init__(self):
        self.data = {ds: {r["stem"]: r for r in load_predictions(ds)} for ds in DATASETS}
        self.summary = load_summary()
        self.seen = training_membership(self.summary)
        self.crema = crema_votes()
        self.mcae = mcae_info()
        self.demo = crema_demographics()

    def row(self, ds, stem):
        return self.data[ds][stem]

    def speaker_info(self, ds, sp):
        if ds == "crema_d":
            d = self.demo[int(sp)]
            return f"{sp} ({d['Sex'].lower()}, {d['Age']})"
        if ds == "mcae_spps":
            return f"{sp} ({'male' if int(sp) <= 3 else 'female'})"
        if ds == "mess":
            return f"{sp} ({'male' if sp[0] == 'M' else 'female'})"
        if ds == "ravdess":
            return f"Actor {sp} ({'male' if int(sp) % 2 else 'female'})"
        if ds == "tess":
            return {"OAF": "OAF (female, 64)", "YAF": "YAF (female, 26)"}[sp]
        return f"{sp} ({'female' if sp.startswith('female') else 'male'})"

    def listener(self, ds, stem):
        """(text, {emotion: votes} sorted by votes, majority) for the datasets with per-clip listener ratings."""
        if ds == "crema_d":
            v = self.crema[stem]
            votes = sorted(((CREMA_CODE[k], v[k]) for k in "ADFHNS" if v[k] > 0), key=lambda t: -t[1])
            maj = self.data[ds][stem]["listener_vote"]
            text = (f"Listeners (voice only, n={v['numResponses']}): " + " · ".join(f"{e} {n}" for e, n in votes)
                    + f" → majority {maj}")
            return text, dict(votes), maj
        if ds == "mcae_spps":
            r = self.mcae[stem]
            votes = sorted(((MCAE_COLS[k], int(r[k])) for k in MCAE_COLS if r[k] > 0), key=lambda t: -t[1])
            text = (f"Listeners (n=40): " + " · ".join(f"{e} {n}" for e, n in votes)
                    + f" (recognition rate {r['rr'] * 100:.1f}%)")
            return text, dict(votes), max(dict(votes), key=dict(votes).get)
        return "No per-clip listener ratings for this dataset", None, None

    def entry(self, ds, stem, mp3_rel):
        """Manifest entry of one clip (without group, notes and duration)."""
        r = self.row(ds, stem)
        meta = parse_meta(ds, stem, r["audio_path"])
        sentence = meta.get("sentence")
        if ds == "mcae_spps":
            zh = self.mcae[stem]["sentence content"].strip()
            sentence = f"{zh} ({MCAE_EN[zh]})" if zh in MCAE_EN else zh
        if ds == "mess":
            sentence = None  # MESS has no transcripts
        text, votes, maj = self.listener(ds, stem)
        e = {"path": mp3_rel, "dataset": DATASET_NAMES[ds], "original_path": r["audio_path"],
             "speaker": self.speaker_info(ds, meta["speaker"]), "sentence": sentence,
             "sentence_id": meta.get("sentence_id"), "acted_label": r["acted_emotion"]}
        if "intensity" in meta:
            e["acted_intensity"] = meta["intensity"]
        e.update({"listener_info": text, "listener_votes": votes, "listener_majority": maj,
                  "predictions": {MODEL_NAMES[m]: [r[f"pred_{m}"], round(r[f"prob_{m}"], 3)] for m in MODELS},
                  "models_in_training_data": [MODEL_NAMES[m] for m in MODELS if ds in self.seen[m]]})
        return e


def active_audio(path):
    """Mono signal trimmed to the active part, with its RMS (dBFS, active part only) and peak (dBFS)."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    hop = int(FRAME_S * sr)
    fr = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2) + 1e-12) for i in range(0, len(x), hop)])
    db = 20 * np.log10(fr)
    on = np.where(db > db.max() - TRIM_DB)[0]
    pad = int(PAD_S * sr)
    a, b = max(0, on[0] * hop - pad), min(len(x), (on[-1] + 1) * hop + pad)
    y = x[a:b]
    rms = 20 * np.log10(np.sqrt(np.mean(x[on[0] * hop:(on[-1] + 1) * hop] ** 2)))
    peak = 20 * np.log10(np.abs(y).max())
    return y, sr, rms, peak


def group_gains(levels):
    """Gain (dB) per clip of a group from [(rms, peak), ...]: the loudest clip to -18 dBFS RMS, quieter clips keep
    half of their level difference (at most 8 dB), so a sad clip still sounds softer than an angry one but stays
    audible; peaks at most -1 dBFS. A single clip simply goes to -18 dBFS RMS."""
    top = max(rms for rms, _ in levels)
    return [min(AUDIO_TARGET_DB - min(AUDIO_MAX_DROP_DB, 0.5 * (top - rms)) - rms, AUDIO_PEAK_DB - peak)
            for rms, peak in levels]


def encode_mp3(y, sr, gain_db, out_path, tmp_dir):
    """Apply the gain, write a float WAV in tmp_dir, encode 16 kHz mono 64 kbps MP3 with ffmpeg; return its duration."""
    import subprocess
    import numpy as np
    import soundfile as sf
    tmp = Path(tmp_dir) / (Path(out_path).stem + ".wav")
    sf.write(str(tmp), (y * 10 ** (gain_db / 20)).astype(np.float32), sr, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), *MP3_ARGS, str(out_path)], check=True)
    tmp.unlink()
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(out_path)], capture_output=True, text=True, check=True).stdout.strip()
    return float(out)
