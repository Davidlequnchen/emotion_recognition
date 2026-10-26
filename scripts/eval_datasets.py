"""
Evaluation datasets: one loader per dataset turns its files and labels into a common list of clips.

Every loader returns an EvalSet whose labels use emotion2vec+ class names
(angry, disgusted, fearful, happy, neutral, sad, surprised). To evaluate on a new dataset, write a
loader below and add it to LOADERS; scripts/evaluate.py, its report and scripts/make_viewer.py then
work without other changes. Download the data first with scripts/download_data.py.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOT_IN_EMOBOX = ("**Not in EmoBox**, so emotion2vec+ seed was not fine-tuned on it. emotion2vec+ large was later "
                 "trained on 40k hours of pseudo-labelled data whose sources are not published, so earlier exposure "
                 "cannot be ruled out completely.")
IN_EMOBOX = ("**In EmoBox.** EmoBox data fine-tuned emotion2vec+ seed, the first model in the chain that led to "
             "emotion2vec+ large, so the model has probably seen these clips or labels derived from them. "
             "Treat these numbers as optimistic.")


@dataclass
class EvalSet:
    key: str            # short name used in file names, e.g. "crema_d"
    name: str           # display name, e.g. "CREMA-D"
    language: str
    classes: list       # the emotions that are evaluated, as emotion2vec+ class names
    clips: list         # one dict per clip: {"audio": Path, "acted": class name, "listener": label(s) or ""}
    ground_truth: str   # what "acted" means here, for the report
    listener: str = ""  # what "listener" means here, if the dataset has listeners' labels
    notes: list = field(default_factory=list)    # extra setup lines for the report (mappings, exclusions)
    caveats: list = field(default_factory=list)  # caveats for the report
    in_emobox: bool = False


def extract_audio(parquet_files, columns, out, pick):
    """Write the audio stored inside Parquet files to out/ (only files that are not there yet).
    pick(row) returns (file name, audio bytes) or None for rows that are not needed."""
    import pyarrow.parquet as pq
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in parquet_files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                got = pick(row)
                if got and not (out / got[0]).exists():
                    (out / got[0]).write_bytes(got[1])
                    written += 1
    print(f"  extracted {written:,} audio files to {out}")


def crema_d():
    root = DATA / "crema_d"
    codes = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
    letters = {"A": "angry", "D": "disgusted", "F": "fearful", "H": "happy", "N": "neutral", "S": "sad"}
    with open(root / "processedResults" / "summaryTable.csv", newline="") as fh:
        votes = {row["FileName"]: row["VoiceVote"] for row in csv.DictReader(fh)}  # e.g. "N" or a tie "N:S"
    clips = [{"audio": f, "acted": codes[f.stem.split("_")[2]],
              "listener": " or ".join(letters[c] for c in votes[f.stem].split(":")) if votes.get(f.stem) else ""}
             for f in sorted((root / "AudioWAV").glob("*.wav"))]
    return EvalSet(
        "crema_d", "CREMA-D", "English", list(codes.values()), clips,
        ground_truth="the emotion each actor was asked to portray, from the file name (`ANG`, `DIS`, `FEA`, `HAP`, "
                     "`NEU`, `SAD`)",
        listener="the majority vote of crowd-sourced listeners who heard the voice only (`summaryTable.csv`, "
                 "`VoiceVote`); a tie counts every tied emotion",
        caveats=[IN_EMOBOX,
                 "One clip, `1076_MTI_SAD_XX.wav`, contains no audio (a known CREMA-D issue); it is kept."],
        in_emobox=True)


def mcae_spps():
    import openpyxl
    root = DATA / "mcae_spps"
    codes = {"1": "neutral", "2": "happy", "3": "angry", "4": "fearful", "5": "sad", "6": "disgusted", "7": "surprised"}
    # Columns with the number of the 40 listeners who chose each emotion
    columns = {"Neutral": "neutral", "Happiness": "happy", "Anger": "angry", "Fear": "fearful", "Sadness": "sad",
               "Disgust": "disgusted", "Surprise": "surprised"}
    audio = {p.stem: p for p in (root / "audio").rglob("*.wav")}
    rows = openpyxl.load_workbook(root / "Corpus information.xlsx", read_only=True).worksheets[0].iter_rows(values_only=True)
    header = [str(h).strip() for h in next(rows)]
    clips = []
    for row in rows:
        r = dict(zip(header, row))
        votes = {emotion: r[col] for col, emotion in columns.items()}
        top = max(votes.values())
        clips.append({"audio": audio[str(r["Filename"])], "acted": codes[str(r["Filename"])[1]],
                      "listener": " or ".join(e for e, n in votes.items() if n == top)})
    return EvalSet(
        "mcae_spps", "MCAE-SPPS", "Mandarin", list(columns.values()), clips,
        ground_truth="the emotion each actor was asked to portray, from the second digit of the file name",
        listener="the majority vote of the 40 listeners who rated each clip (`Corpus information.xlsx`); "
                 "a tie counts every tied emotion",
        notes=["Label mapping: Neutral = neutral, Happiness = happy, Anger = angry, Fear = fearful, Sadness = sad, "
               "Disgust = disgusted, Surprise = surprised."],
        caveats=[NOT_IN_EMOBOX,
                 "**Validated clips only.** The dataset keeps a clip only if at least 43% of its 40 listeners chose the "
                 "intended emotion, so the acted label and the listeners' vote usually agree.",
                 "Mandarin speech, 48 kHz, resampled to 16 kHz by the model's loader."])


def thai_ser():
    root = DATA / "thai_ser"
    files = sorted((root / "data").glob("*.parquet"))
    names = {"Neutral": "neutral", "Angry": "angry", "Happy": "happy", "Sad": "sad"}  # "Frustrated" has no model class
    import pyarrow.parquet as pq
    rows = pq.ParquetDataset(files).read(columns=["audio_id", "assigned_emo", "majority_emo"]).to_pylist()
    rows = [r for r in rows if r["assigned_emo"] in names]
    audio = root / "audio"
    if any(not (audio / r["audio_id"]).exists() for r in rows):
        # Each clip was recorded by several microphones: use the condenser microphone in the studio and the
        # Zoom recording for online sessions (each clip's audio_id names exactly that file)
        keep = {r["audio_id"] for r in rows}
        print("  first run: extracting the audio of the studio condenser / Zoom microphone from the Parquet files ...")
        extract_audio(files, ["audio_id", "mic_con", "mic_zoom"], audio, lambda r: (r["audio_id"], (
            r["mic_con"] if r["mic_con"] and r["mic_con"]["bytes"] else r["mic_zoom"])["bytes"]) if r["audio_id"] in keep else None)
    clips = [{"audio": audio / r["audio_id"], "acted": names[r["assigned_emo"]],
              "listener": names.get(r["majority_emo"], str(r["majority_emo"]).lower() if r["majority_emo"] not in ("None", None) else "")}
             for r in rows]
    return EvalSet(
        "thai_ser", "THAI-SER", "Thai", list(names.values()), clips,
        ground_truth="the emotion assigned to the actor for the scene (`assigned_emo`)",
        listener="the crowd-sourced majority label (`majority_emo`); clips without a majority have none",
        notes=["Label mapping: Neutral = neutral, Angry = angry, Happy = happy, Sad = sad. Clips assigned "
               "*Frustrated* (5,270 of 27,854) are left out because emotion2vec+ has no such class; a listener vote of "
               "*frustrated* can therefore never match.",
               "Audio: the condenser microphone for studio sessions and the Zoom recording for online sessions "
               "(44.1 kHz FLAC, resampled to 16 kHz)."],
        caveats=[NOT_IN_EMOBOX,
                 "**Labels often disagree.** The assigned and the crowd-majority labels differ for many clips; the "
                 "dataset authors recommend the high-agreement subset (agreement >= 0.71) for clean comparisons.",
                 "Some sessions were recorded in an untreated studio room or over Zoom, so audio quality varies."])


def nemo():
    root = DATA / "nemo"
    files = sorted((root / "data").glob("*.parquet"))
    names = {"neutral": "neutral", "anger": "angry", "fear": "fearful", "happiness": "happy", "sadness": "sad",
             "surprised": "surprised"}
    import pyarrow.parquet as pq
    rows = pq.ParquetDataset(files).read(columns=["file_id", "emotion"]).to_pylist()
    audio = root / "audio"
    if any(not (audio / f"{r['file_id']}.wav").exists() for r in rows):
        print("  first run: extracting the audio from the Parquet files ...")
        extract_audio(files, ["file_id", "audio"], audio, lambda r: (f"{r['file_id']}.wav", r["audio"]["bytes"]))
    clips = [{"audio": audio / f"{r['file_id']}.wav", "acted": names[r["emotion"]], "listener": ""} for r in rows]
    return EvalSet(
        "nemo", "nEMO", "Polish", ["angry", "fearful", "happy", "neutral", "sad", "surprised"], clips,
        ground_truth="the emotion each speaker was asked to portray (`emotion`)",
        notes=["Label mapping: anger = angry, fear = fearful, happiness = happy, sadness = sad, surprised = surprised, "
               "neutral = neutral."],
        caveats=[NOT_IN_EMOBOX,
                 "No listeners' labels are published: listeners only screened the recordings, so there is one way of "
                 "counting a prediction as correct.",
                 "Recorded at home (24 kHz); only 3 of the 9 speakers are professional voice actors."])


def mess():
    root = DATA / "mess" / "audio"
    letters = {"A": "angry", "C": "neutral", "H": "happy", "S": "sad"}  # C = calm, scored as neutral
    clips = [{"audio": f, "acted": letters[f.name[0]], "listener": ""} for f in sorted(root.glob("*.wav"))]
    return EvalSet(
        "mess", "MESS", "English", ["angry", "happy", "neutral", "sad"], clips,
        ground_truth="the emotion each speaker was asked to portray, from the first letter of the file name "
                     "(`A`ngry, `C`alm, `H`appy, `S`ad)",
        notes=["Label mapping: angry = angry, happy = happy, sad = sad, and **calm = neutral** (emotion2vec+ has no "
               "*calm* class)."],
        caveats=[NOT_IN_EMOBOX,
                 "*Calm* is not quite *neutral*: calm clips predicted as neutral count as correct.",
                 "The public version has 1,800 clips and acted labels only; the listeners' ratings are available "
                 "from the author on request. Audio is sampled at 24,414 Hz, resampled to 16 kHz."])


def esd():
    root = DATA / "esd" / "audio" / "Emotion Speech Dataset"  # <speaker>/<Emotion>/<speaker>_<number>.wav
    folders = {"Neutral": "neutral", "Angry": "angry", "Happy": "happy", "Sad": "sad", "Surprise": "surprised"}
    clips = [{"audio": f, "acted": folders[f.parent.name], "listener": ""} for f in sorted(root.glob("*/*/*.wav"))]
    return EvalSet(
        "esd", "ESD", "English, Mandarin", list(folders.values()), clips,
        ground_truth="the emotion each speaker was asked to portray, from the folder name (`Neutral`, `Angry`, "
                     "`Happy`, `Sad`, `Surprise`; the transcripts give the same labels)",
        notes=["Label mapping: Neutral = neutral, Angry = angry, Happy = happy, Sad = sad, Surprise = surprised.",
               "Speakers 0001–0010 speak Mandarin and 0011–0020 English. Each says the same 350 sentences in all five "
               "emotions (for about 3.5% of them the wording differs slightly between emotions)."],
        caveats=[IN_EMOBOX,
                 "No listeners' labels are published, so there is one way of counting a prediction as correct.",
                 "Audio: 16 kHz WAV, the model's own sampling rate."],
        in_emobox=True)


LOADERS = {"crema_d": crema_d, "mcae_spps": mcae_spps, "thai_ser": thai_ser, "nemo": nemo, "mess": mess, "esd": esd}
