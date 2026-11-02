"""Audio examples of the presentation: writes assets/audio/*.mp3 and assets/manifest.json.

The clips are the fixed list CLIPS below (change it on purpose only: the slides refer to these files).
Reads: the WAV files in data/<dataset>/, outputs/<model>/<dataset>_predictions.csv, reports/summary.json,
data/crema_d/processedResults/tabulatedVotes.csv, data/crema_d/VideoDemographics.csv,
data/mcae_spps/Corpus information.xlsx and assets/stats.json (for the figure captions; run make_figures.py first).
Needs ffmpeg and ffprobe on PATH.
Run (from the repository root or anywhere):  python emotion_recognition/presentation/scripts/make_audio.py
"""
import json
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from common import (ASSETS, DATASET_NAMES, DATASETS, ER, MODEL_NAMES, ClipTables, active_audio, encode_mp3,
                    group_gains)

# ---------------------------------------------------------------- settings
AUD = ASSETS / "audio"

# Four clips per dataset: same speaker, same sentence, four emotions (levelled together as a quartet).
EX = {
    "crema_d": ["1089_IOM_NEU_XX", "1089_IOM_ANG_XX", "1089_IOM_HAP_XX", "1089_IOM_SAD_XX"],
    "mcae_spps": ["61195", "63195", "62195", "65195"],
    "mess": ["CF3V12_SCR", "AF3V12_SCR", "HF3V12_SCR", "SF3V12_SCR"],
    "ravdess": ["03-01-01-01-01-01-02", "03-01-05-02-01-01-02", "03-01-03-02-01-01-02", "03-01-04-02-01-01-02"],
    "tess": ["YAF_bar_neutral", "YAF_bar_angry", "YAF_bar_happy", "YAF_bar_sad"],
    "jl_corpus": ["female2_neutral_1a_2", "female2_angry_1b_2", "female2_happy_1a_1", "female2_sad_1b_2"],
}
EX_NOTE = {
    "crema_d": "Same actress (1089), same sentence, four emotions; the listener majority matches the acted label on all four (70-100% of raters); both top models are right on all four (18 of 20 predictions right).",
    "mcae_spps": "Same actress, same Mandarin sentence, four emotions; 88-95% of 40 listeners chose the intended emotion; both top models are right on all four.",
    "mess": "Same actress, same sentence, four emotions (calm counts as neutral). MERaLiON gets all four; emotion2vec+ calls the angry and sad clips neutral. MESS clips are RMS-normalised, so loudness gives no cue.",
    "ravdess": "Same actress, same sentence and repetition, strong intensity (neutral has only normal); all five models are right on all four.",
    "tess": "Same speaker (YAF, aged 26), same carrier phrase and target word, four emotions; emotion2vec+ (trained on TESS) gets all four, MERaLiON calls the neutral clip sad.",
    "jl_corpus": "Same speaker (female2), same sentence, four emotions; all five models are right on all four.",
}
# Single clips: listeners vs actor (subjectivity of the labels).
SUBJ = [
    ("crema_d", "1014_WSI_HAP_XX", "Acted happy, yet all 10 listeners heard angry. MERaLiON follows the listeners; emotion2vec+ (trained on CREMA-D's acted labels) says happy."),
    ("crema_d", "1089_TAI_FEA_XX", "Acted fearful, all 10 listeners heard sad; the top two models still say fearful (CREMA-D is in their training data)."),
    ("crema_d", "1029_IEO_DIS_HI", "Acted disgusted at high intensity, 9 of 10 listeners heard angry."),
    ("crema_d", "1082_TAI_SAD_XX", "Acted sad, all 10 listeners heard neutral: the most common CREMA-D mismatch (67% of acted-sad clips get a neutral majority)."),
    ("mcae_spps", "37166", "Only 47.5% of listeners recognised the intended surprise; a third heard neutral."),
    ("mcae_spps", "42139", "Only 47.5% recognised happy; 13 of 40 heard surprised."),
    ("mcae_spps", "23035", "Contrast: 40 of 40 listeners agree on angry, and all five models are right."),
]
# Single clips: error analysis, by kind of error.
FAIL = [
    ("crema_d", "1068_IEO_HAP_HI", "i_models_side_with_listeners",
     "Acted happy (high intensity), 9 of 10 listeners heard angry, and both top models say angry: is the model wrong or the label?"),
    ("crema_d", "1066_ITS_ANG_XX", "i_models_side_with_listeners",
     "Acted angry, 8 of 10 listeners heard disgusted; both top models say disgusted (emotion2vec+ at 100%)."),
    ("mess", "AM2V15_SCR", "ii_confident_on_unseen_data",
     "Acted angry on unseen MESS; all five models say neutral, emotion2vec+ at 100%. emotion2vec+ calls 81% of MESS clips neutral."),
    ("mcae_spps", "57175", "ii_confident_on_unseen_data",
     "38 of 40 listeners heard surprise, yet emotion2vec+, MERaLiON and Melati say angry. Mandarin surprise is heard as angry by models (MERaLiON: 46% of surprised clips)."),
    ("tess", "YAF_read_neutral", "ii_confident_on_unseen_data",
     "Neutral on TESS (unseen for MERaLiON): MERaLiON says sad at 0.91, as it does for all 400 TESS neutral clips; emotion2vec+ (trained on TESS) is right."),
    ("ravdess", "03-01-08-02-02-01-17", "ii_confident_on_unseen_data",
     "Surprised (strong) on RAVDESS: emotion2vec+ (trained on RAVDESS) says surprised at 100%, MERaLiON says fearful at 0.91."),
    ("mess", "SF3T02_SCR", "iii_models_disagree",
     "Acted sad: four models say sad, but emotion2vec+ says neutral at 100%; a confident outlier."),
    ("mcae_spps", "62191", "iii_models_disagree",
     "The models split exactly as the listeners do: 22 of 40 heard happy, 17 neutral; MERaLiON says happy, emotion2vec+ neutral."),
    ("crema_d", "1006_IEO_ANG_MD", "iv_questionable_ground_truth",
     "Acted angry (medium intensity), but the 9 listeners split three ways (angry 3, happy 3, neutral 3): there is no majority at all. Both top models say happy (0.92 and 0.86): scored as wrong, yet as many listeners heard happy as angry."),
    ("crema_d", "1030_WSI_FEA_XX", "iv_questionable_ground_truth",
     "Acted fearful, but none of 10 listeners chose fear and votes are spread over five emotions; any prediction is defensible (15% of CREMA-D clips get no vote for the acted emotion)."),
]
# (dataset, stem, group, subgroup, note) in manifest order
CLIPS = ([(ds, s, "dataset_example", ds, EX_NOTE[ds]) for ds, stems in EX.items() for s in stems]
         + [(ds, s, "subjectivity", ds, n) for ds, s, n in SUBJ]
         + [(ds, s, "failure", sub, n) for ds, s, sub, n in FAIL])

NOTES = {
    "audio_processing": "Leading/trailing silence trimmed (35 dB below the loudest 20 ms frame, 0.12 s padding); loudest clip of each quartet at about -18 dBFS RMS, quieter emotions keep half of their level difference (at most 8 dB) so relative loudness is still audible; single clips at about -18 dBFS RMS; peaks at most -1 dBFS; 16 kHz mono MP3 at 64 kbps.",
    "predictions": "Label = the most probable of the dataset's emotions (as in the evaluation); probability = that model's own softmax probability for the label.",
    "licences": "CREMA-D ODbL (contents DbCL); MCAE-SPPS CC BY-NC 4.0; MESS CC BY; RAVDESS CC BY-NC-SA 4.0; TESS CC BY-NC 4.0; JL-Corpus CC0.",
}

print("Versions: numpy", np.__version__, "| soundfile", sf.__version__, "|",
      subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0])
print(f"Clips: {len(CLIPS)} | output: {AUD.relative_to(ER)}")
assert shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe must be on PATH"
AUD.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Step 1: tables
print("Step 1: load predictions, listener votes and speaker data")
T = ClipTables()
SEEN = T.seen
stats = json.loads((ASSETS / "stats.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- Step 2: trim and level
print("Step 2: trim silence and compute the gain of every clip (quartets are levelled together)")
groups = {}
for ds, stem, group, sub, note in CLIPS:
    key = (group, sub) if group == "dataset_example" else (group, stem)
    groups.setdefault(key, []).append((ds, stem))
cache, gains = {}, {}
for key, items in groups.items():
    for ds, stem in items:
        cache[(ds, stem)] = active_audio(ER / T.row(ds, stem)["audio_path"])
    levels = [(cache[c][2], cache[c][3]) for c in items]
    gains.update(zip(items, group_gains(levels)))

# ---------------------------------------------------------------- Step 3: encode and describe
print("Step 3: encode MP3s and write one manifest entry per clip")
manifest = {"audio": [], "figures": []}
tmp_dir = tempfile.mkdtemp()
for ds, stem, group, sub, note in CLIPS:
    y, sr, rms, peak = cache[(ds, stem)]
    name = f"{ds}__{stem}".replace(" ", "_") + ".mp3"
    dur = encode_mp3(y, sr, gains[(ds, stem)], AUD / name, tmp_dir)
    e = T.entry(ds, stem, f"audio/{name}")
    e.update({"group": group, "subgroup": sub, "why_interesting": note, "duration_s": round(dur, 2)})
    manifest["audio"].append(e)
    p = e["predictions"]
    print(f"  {name:45s} {dur:4.2f}s gain {gains[(ds, stem)]:+5.1f} dB  {e['acted_label']:9s} "
          f"{p['emotion2vec+ large']} {p['MERaLiON-SER-v1']}")
shutil.rmtree(tmp_dir)

# ---------------------------------------------------------------- Step 4: figure captions
print("Step 4: figure captions")
FIGS = []
for ds in DATASETS:
    for m in ["emotion2vec_plus_large", "meralion_ser_v1"]:
        acc = stats["models"][m]["accuracy"][ds] * 100
        seen = "in its training data" if ds in SEEN[m] else "not in its training data"
        FIGS.append((f"fig/cm_{m}_{ds}.png",
                     f"{MODEL_NAMES[m]} on {DATASET_NAMES[ds]} ({seen}), accuracy {acc:.1f}%: rows = acted emotion, columns = prediction, % of each row."))
    FIGS.append((f"fig/cm_pair_{ds}.png",
                 f"{DATASET_NAMES[ds]}: emotion2vec+ large vs MERaLiON-SER-v1, row-normalised confusion matrices with shared colour scale."))
c = stats["crema_d_listeners"]
mc = stats["mcae_spps_listeners"]["recognition_rate"]
FIGS += [
    ("fig/crema_acted_vs_listener.png",
     f"CREMA-D: acted emotion vs voice-only listener majority (about 9 raters per clip). Majority matches the actor on only {c['listener_majority_equals_acted'] * 100:.1f}% of clips; most mismatches are heard as neutral."),
    ("fig/mcae_recognition_hist.png",
     f"MCAE-SPPS: share of 40 listeners who chose the intended emotion, per clip (mean {mc['mean'] * 100:.1f}%, median {mc['median'] * 100:.1f}%; the corpus kept only clips with a plurality for the intended emotion)."),
    ("fig/seen_vs_unseen.png",
     "Mean accuracy on datasets inside vs outside each model's training data (the Vox-Profile models saw none of the six)."),
    ("fig/emotion_counts.png", "Evaluated clips per acted emotion in each of the six datasets."),
]
for p, cap in FIGS:
    assert (ASSETS / p).exists(), f"missing {p}: run make_figures.py first"
    manifest["figures"].append({"path": p, "caption": cap})
manifest["notes"] = NOTES
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Done: {len(manifest['audio'])} MP3s in assets/audio/ and assets/manifest.json "
      f"({len(manifest['figures'])} figure captions)")
