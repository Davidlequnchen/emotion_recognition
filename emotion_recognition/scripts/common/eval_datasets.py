"""
Evaluation datasets: one loader per dataset turns its files and labels into a common list of clips.

Every loader returns an EvalSet whose labels use the project's class names
(angry, disgusted, fearful, happy, neutral, sad, surprised), which every model loader also uses. A label that has no
such class can stay as it is when the dataset says which classes count as correct for it (EvalSet.accept, e.g.
JL-Corpus's excited: happy or surprised); accepted(ds, label) gives these classes for any label, and sample(ds, limit)
the clips that an evaluation with `limit` (config/evaluation.yaml) uses. To evaluate on a new dataset, write a
loader below and add it to LOADERS; scripts/evaluation/evaluate.py, its report and
scripts/viewer/make_viewer.py then work without other changes. Whether a dataset is in a model's training data is
not set here but in the model registry (`training_data` in config/models.yaml). Download the data
first with scripts/common/download_data.py.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
DATA = ROOT / "data"


@dataclass
class EvalSet:
    key: str            # short name used in file names, e.g. "crema_d"
    name: str           # display name, e.g. "CREMA-D"
    language: str
    classes: list       # the classes a prediction is chosen from, as the project's class names
    clips: list         # one dict per clip: {"audio": Path, "acted": label, "listener": label(s) or ""}; "acted" is
                        # a class name, or a composite label from accept
    ground_truth: str   # what "acted" means here, for the report
    listener: str = ""  # what "listener" means here, if the dataset has listeners' labels
    # Composite labels: a ground-truth label that is not a model class -> the classes that count as a correct
    # prediction for it, e.g. {"excited": ["happy", "surprised"]} (the first one is used where one class is needed)
    accept: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)    # extra setup lines for the report (mappings, exclusions)
    caveats: list = field(default_factory=list)  # caveats for the report (the training-data caveat comes from the
                                                 # model registry and is added by the evaluation)


def accepted(ds, label):
    """The model classes that count as a correct prediction for a ground-truth label: the label itself when it is a
    model class, or the classes of a composite label (ds.accept)."""
    return set(ds.accept.get(label, [label]))


def truth_labels(ds):
    """The ground-truth labels that the dataset's clips have, in a fixed order: its classes first (in the order of
    ds.classes), then its composite labels (in the order of ds.accept). The rows of the confusion matrices."""
    found = {c["acted"] for c in ds.clips}
    unknown = found - set(ds.classes) - set(ds.accept)
    if unknown:
        raise ValueError(f"{ds.name}: the labels {sorted(unknown)} are neither in `classes` nor in `accept`")
    return [label for label in [*ds.classes, *ds.accept] if label in found]


def sample(ds, limit):
    """The clips that scripts/evaluation/evaluate.py evaluates with `limit` (config/evaluation.yaml): every clip when
    limit is null or not smaller than the dataset, otherwise `limit` clips taken at even intervals, ground-truth label
    by label (in the order of truth_labels, and in file order within each), so that every label is sampled even when
    the file order repeats (TESS: the 7 emotions of each word in a row)."""
    if not limit or limit >= len(ds.clips):
        return ds.clips
    order = truth_labels(ds)
    ordered = sorted(ds.clips, key=lambda c: order.index(c["acted"]))  # a stable sort keeps the file order
    return ordered[::len(ordered) // limit][:limit]


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
        caveats=["One clip, `1076_MTI_SAD_XX.wav`, contains no audio (a known CREMA-D issue); it is kept."])


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
        caveats=["**Validated clips only.** The dataset keeps a clip only if at least 43% of its 40 listeners chose "
                 "the intended emotion, so the acted label and the listeners' vote usually agree.",
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
               "*Frustrated* (5,270 of 27,854) are left out because none of the project's emotions matches it; a "
               "listener vote of *frustrated* can therefore never match.",
               "Audio: the condenser microphone for studio sessions and the Zoom recording for online sessions "
               "(44.1 kHz FLAC, resampled to 16 kHz)."],
        caveats=["**Labels often disagree.** The assigned and the crowd-majority labels differ for many clips; the "
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
        caveats=["No listeners' labels are published: listeners only screened the recordings, so there is one way of "
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
        notes=["Label mapping: angry = angry, happy = happy, sad = sad, and **calm = neutral** (the project's "
               "emotions have no *calm*)."],
        caveats=["*Calm* is not quite *neutral*: calm clips predicted as neutral count as correct.",
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
        caveats=["No listeners' labels are published, so there is one way of counting a prediction as correct.",
                 "Audio: 16 kHz WAV, the models' own sampling rate."])


def ravdess():
    root = DATA / "ravdess" / "audio"  # Actor_<nn>/03-01-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav
    codes = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
             "07": "disgusted", "08": "surprised"}  # 02 = calm, scored as neutral
    # Ordered by emotion code, then actor: acted neutral comes before acted calm, so that a sample (`limit`)
    # of the neutral clips holds both in proportion (within each actor, 4 neutral clips precede 8 calm ones)
    files = sorted(root.glob("Actor_*/*.wav"), key=lambda f: (f.stem.split("-")[2], f))
    clips = [{"audio": f, "acted": codes[f.stem.split("-")[2]], "listener": ""} for f in files]
    neutral, calm = (sum(f.stem.split("-")[2] == code for f in files) for code in ("01", "02"))
    return EvalSet(
        "ravdess", "RAVDESS", "English", list(dict.fromkeys(codes.values())), clips,
        ground_truth="the emotion each actor was asked to portray, from the third number in the file name (01 neutral, "
                     "02 calm, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust, 08 surprised)",
        notes=["Label mapping: neutral, happy, sad, angry, fearful and surprised keep their names; disgust = disgusted, "
               f"and **calm = neutral** (the project's emotions have no *calm*), so neutral has {neutral + calm:,} "
               f"clips: {neutral:,} acted neutral and {calm:,} acted calm.",
               "24 professional actors (12 female, 12 male) say two sentences, \"Kids are talking by the door\" and "
               "\"Dogs are sitting by the door\", twice in each emotion, at two intensities (normal and strong) except "
               "for neutral."],
        caveats=["*Calm* is not quite *neutral*: calm clips predicted as neutral count as correct.",
                 "Listeners rated every file 10 times (247 listeners). The paper publishes their ratings as "
                 "recognition rates per file, not as a vote for an emotion, so this report has no listeners' labels.",
                 "Audio: 48 kHz WAV, resampled to 16 kHz; 5 files are stereo, and their channels are averaged."])


def tess():
    root = DATA / "tess"  # <speaker>_<word>_<emotion>.wav; speaker OAF (aged 64) or YAF (aged 26)
    words = {"angry": "angry", "disgust": "disgusted", "fear": "fearful", "happy": "happy", "neutral": "neutral",
             "ps": "surprised", "sad": "sad"}  # ps = pleasant surprise
    clips = [{"audio": f, "acted": words[f.stem.rsplit("_", 1)[1].lower()], "listener": ""}
             for f in sorted(root.glob("*.wav"))]
    return EvalSet(
        "tess", "TESS", "English", list(words.values()), clips,
        ground_truth="the emotion each actress was asked to portray, from the last word of the file name (`angry`, "
                     "`disgust`, `fear`, `happy`, `neutral`, `ps` for pleasant surprise, `sad`)",
        notes=["Label mapping: angry, happy, neutral and sad keep their names; disgust = disgusted, fear = fearful, "
               "**pleasant surprise = surprised**.",
               "Two actresses, aged 26 and 64, say \"Say the word ...\" with the same 200 target words in each of the "
               "seven emotions."],
        caveats=["Only two speakers, so the result says little about other voices; models often score close to 100% on "
                 "TESS.",
                 "No listeners' labels are published with the dataset. Audio: 24,414 Hz WAV (one file, "
                 "`OAF_food_fear.wav`, is 96 kHz), resampled to 16 kHz."])


def jl_corpus():
    # <speaker>_<emotion>_<sentence><session>_<repetition>.wav, e.g. female1_angry_10a_1.wav, with its transcript in .txt
    root = DATA / "jl_corpus" / "audio" / "Raw JL corpus (unchecked and unannotated)" / "JL(wav+txt)"
    primary = ["angry", "happy", "neutral", "sad", "excited"]  # the five secondary emotions have no model class
    files = sorted(root.glob("*.wav"))
    clips = [{"audio": f, "acted": f.stem.split("_")[1], "listener": ""} for f in files
             if f.stem.split("_")[1] in primary]
    return EvalSet(
        "jl_corpus", "JL-Corpus", "English (New Zealand)", ["angry", "happy", "neutral", "sad", "surprised"], clips,
        ground_truth="the emotion each speaker was asked to portray, from the second part of the file name",
        accept={"excited": ["happy", "surprised"]},  # excited has no model class
        notes=[f"Only the five primary emotions ({len(clips):,} of {len(files):,} clips): angry, happy, neutral and sad "
               "keep their names; **excited has no model class, so a prediction of happy or surprised counts as "
               "correct** for it (its label stays excited). The prediction is therefore chosen from angry, happy, "
               "neutral, sad and surprised, although no clip is acted surprised. The five secondary emotions (anxious, "
               "apologetic, assertive, concerned, encouraging) have no model class and are left out.",
               "Four trained voice actors of New Zealand English (two female, two male) say the same 15 sentences in "
               "each primary emotion, in two sessions, twice each."],
        caveats=["*Excited* is neither *happy* nor *surprised*: excited clips predicted as happy or surprised count as "
                 "correct. Surprised is a possible prediction only because of excited, and it is wrong for a clip of "
                 "any other emotion.",
                 "This is the raw corpus (\"unchecked and unannotated\"). In the authors' perception test (120 listeners) "
                 "about 70% of the primary-emotion clips were recognised; the results per clip are not published, so "
                 "poorly acted clips are included.",
                 "Audio: 44.1 kHz WAV, resampled to 16 kHz."])


LOADERS = {"crema_d": crema_d, "mcae_spps": mcae_spps, "thai_ser": thai_ser, "nemo": nemo, "mess": mess, "esd": esd,
           "ravdess": ravdess, "tess": tess, "jl_corpus": jl_corpus}

# Source, licence and credit of each dataset, shown on the viewer page (scripts/viewer/make_viewer.py). public_audio:
# the licence allows putting its clips on a public web page. A dataset that is not listed here counts as not allowed.
CC = "https://creativecommons.org/licenses/"
SOURCES = {
    "crema_d": {"url": "https://github.com/CheyneyComputerScience/CREMA-D", "public_audio": True,
                "licence": "ODbL 1.0", "licence_url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "credit": "Cao, Cooper, Keutmann, Gur, Nenkova, Verma. CREMA-D: Crowd-sourced Emotional Multimodal "
                          "Actors Dataset. IEEE Transactions on Affective Computing, 2014."},
    "mcae_spps": {"url": "https://osf.io/9jyzc/", "public_audio": True,
                  "licence": "CC BY-NC 4.0", "licence_url": CC + "by-nc/4.0/",
                  "credit": "Li, Zhou, Yan, Li, Ma, Wu. A validated Mandarin Chinese Auditory Emotion Database of "
                            "Subject-Personal-Pronoun Sentences (MCAE-SPPS). Scientific Data, 2026."},
    "thai_ser": {"url": "https://huggingface.co/datasets/airesearch/thai-ser", "public_audio": True,
                 "licence": "CC BY-SA 4.0", "licence_url": CC + "by-sa/4.0/",
                 "credit": "Wongpithayadisai et al. Thai Speech Emotion Recognition (THAI-SER) Corpus. Language "
                           "Resources and Evaluation, 2026."},
    "nemo": {"url": "https://huggingface.co/datasets/amu-cai/nEMO", "public_audio": True,
             "licence": "CC BY-NC-SA 4.0", "licence_url": CC + "by-nc-sa/4.0/",
             "credit": "Christop. nEMO: Dataset of Emotional Speech in Polish. LREC-COLING 2024."},
    "mess": {"url": "https://zenodo.org/records/3813437", "public_audio": True,
             "licence": "CC BY 4.0", "licence_url": CC + "by/4.0/",
             "credit": "Morgan. Categorical and Dimensional Ratings of Emotional Speech: Behavioral Findings From the "
                       "Morgan Emotional Speech Set. Journal of Speech, Language, and Hearing Research, 2019."},
    # ESD's licence forbids giving the data to anyone else, so its clips stay off public pages
    "esd": {"url": "https://github.com/HLTSingapore/Emotional-Speech-Data", "public_audio": False,
            "licence": "research only, under a signed licence", "licence_url": None,
            "credit": "Zhou, Sisman, Liu, Li. Emotional voice conversion: theory, databases and ESD. Speech "
                      "Communication, 2022. (Portions of) the research on this page used the ESD Database made "
                      "available by the HLT lab, National University of Singapore, Singapore."},
    "ravdess": {"url": "https://zenodo.org/records/1188976", "public_audio": True,
                "licence": "CC BY-NC-SA 4.0", "licence_url": CC + "by-nc-sa/4.0/",
                "credit": "Livingstone, Russo. The Ryerson Audio-Visual Database of Emotional Speech and Song "
                          "(RAVDESS). PLOS ONE, 2018."},
    "tess": {"url": "https://doi.org/10.5683/SP2/E8H2MF", "public_audio": True,
             "licence": "CC BY-NC 4.0", "licence_url": CC + "by-nc/4.0/",
             "credit": "Pichora-Fuller, Dupuis. Toronto emotional speech set (TESS). Borealis, 2020."},
    "jl_corpus": {"url": "https://www.kaggle.com/datasets/tli725/jl-corpus", "public_audio": True,
                  "licence": "CC0 1.0", "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                  "credit": "James, Tian, Watson. An Open Source Emotional Speech Corpus for Human Robot Interaction "
                            "Applications. Interspeech 2018."},
}
