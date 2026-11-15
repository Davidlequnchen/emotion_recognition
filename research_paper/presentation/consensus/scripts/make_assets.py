"""Slide figures, slide numbers and facts.md for the consensus deck, from the saved consensus outputs only.

Run from this folder after `python ../../scripts/consensus.py` (rebuild.sh does both):
    python scripts/make_assets.py
Reads   ../../output/consensus_thresholds.csv, consensus_by_class.csv, consensus_baselines.csv, consensus_manifest.json
Writes  fig/*.png (consensus_plot.plot_slides, 300 dpi at slide size), assets/numbers.json (read by build.js),
        facts.md (every number on the slides with its source row).
No models, audio or APIs are used.
"""
import csv
import json
from pathlib import Path
import statistics
import sys

DECK = Path(__file__).resolve().parents[1]
PAPER = DECK.parents[1]                                   # research_paper/
OUT = PAPER / "output"
sys.path.insert(0, str(PAPER / "scripts"))
import consensus                                          # noqa: E402  group definitions and display names
import consensus_plot                                     # noqa: E402  the single source of the figures

DIALOGUE = ("meld", "emotiontalk", "dailytalk", "multidialog")
FIRST, SECOND = [d for d, _, _ in consensus_plot.DATASETS[:8]], [d for d, _, _ in consensus_plot.DATASETS[8:]]
NAMES = {key: name for key, name, _ in consensus_plot.DATASETS}
facts = []                                                # (slide, fact, value, source) for facts.md


def step(text):
    print(f"\n== {text}")


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pct(value, digits=1):
    return None if value in (None, "") else round(100 * float(value), digits)


def fact(slide, what, value, source):
    facts.append((slide, what, value, source))
    return value


def row_info(r):
    k, m = int(r["min_votes"]), int(r["model_count"])
    return {"rule": f"{'' if k == m else '≥'}{k}/{m}", "k": k, "m": m, "accepted": int(r["accepted"]),
            "correct": int(r["accepted_correct"]), "errors": int(r["accepted_errors"]),
            "common": int(r["common_samples"]), "coverage": pct(r["coverage"], 2),
            "accuracy": pct(r["conditional_precision"], 2), "complete": r["complete"] == "True",
            "cohort": r["cohort_sha256"]}


def src(r, columns):
    return (f"`output/consensus_thresholds.csv` dataset={r['dataset']}, group={r['group']}, "
            f"min_votes={r['min_votes']} ({columns})")


# Step 1: saved consensus outputs (written by scripts/consensus.py)
step("1. Read the saved consensus outputs")
thresholds, by_class, baselines = read("consensus_thresholds.csv"), read("consensus_by_class.csv"), \
    read("consensus_baselines.csv")
manifest = json.loads((OUT / "consensus_manifest.json").read_text(encoding="utf-8"))
pending = consensus.pending_members(manifest["cohorts"])
print(f"threshold rows {len(thresholds)}, class rows {len(by_class)}, baseline rows {len(baselines)}; "
      f"consensus refreshed {manifest['generated_at_singapore']}")

# Step 2: slide figures from consensus_plot.py (same colours and markers as the paper figures)
step("2. Draw the slide figures")
for path in consensus_plot.plot_slides(thresholds, by_class, DECK / "fig", pending=pending):
    print("  wrote", path.relative_to(DECK))

# Step 3: numbers for the slides
step("3. Compute the slide numbers")
unanimous = {(r["dataset"], r["group"]): r for r in thresholds if r["min_votes"] == r["model_count"]}
specs = consensus.GROUPS | consensus.EXTRA_GROUPS
groups = {}
for key in consensus.MAIN_GROUPS + consensus.APPENDIX_GROUPS:
    label, colour, marker, members = consensus_plot.GROUPS[key]
    rows = [unanimous.get((d, key)) for d in consensus.DATASETS]
    complete = sum(r is not None and r["complete"] == "True" for r in rows)
    partial = sorted(NAMES[d] for (g, d) in pending if g == key)
    missing = sorted({n for (g, _), names in pending.items() if g == key for n in names})
    groups[key] = {"label": label, "colour": colour.lstrip("#"), "marker": marker, "describe": members,
                   "members": [consensus.SHORT_NAMES.get(m, m) for m in specs[key]["models"]],
                   "size": len(specs[key]["models"]), "complete": complete, "partial": len(partial),
                   "missing": missing, "waits": specs[key]["require_all"]}
    fact("7, A1–A2", f"{label}: datasets with every member / with missing members", f"{complete} / {len(partial)}",
         "`output/consensus_manifest.json` cohorts (complete, missing_members)")
pending_names = sorted({n for names in pending.values() for n in names})
pending_datasets = max((groups[g]["partial"] for g in consensus.MAIN_GROUPS), default=0)

crema = [row_info(r) for r in thresholds if r["dataset"] == "crema_d" and r["group"] == "SER5"]
for r, raw in zip(crema, [r for r in thresholds if r["dataset"] == "crema_d" and r["group"] == "SER5"]):
    fact("3, 6", f"CREMA-D SER5 {r['rule']}: accepted, correct / wrong, coverage, accuracy",
         f"{r['accepted']:,}; {r['correct']:,} / {r['errors']:,}; {r['coverage']}%; {r['accuracy']}% "
         f"(of {r['common']:,} clips)", src(raw, "accepted, accepted_correct, accepted_errors, coverage, "
                                                 "conditional_precision, common_samples"))

meld = []
for g in consensus.MAIN_GROUPS:
    raw = unanimous.get(("meld", g))
    if raw:
        info = row_info(raw) | {"group": g, "label": groups[g]["label"]}
        meld.append(info)
        fact("10", f"MELD {g} unanimity {info['rule']}: accepted, correct / wrong, coverage, accuracy",
             f"{info['accepted']:,}; {info['correct']:,} / {info['errors']:,}; {info['coverage']}%; "
             f"{info['accuracy']}% (of {info['common']:,} clips)", src(raw, "accepted, accepted_correct, accepted_errors, "
                                                                     "coverage, conditional_precision, common_samples"))
meld_matched = len(meld) == 4 and len({m["cohort"] for m in meld}) == 1 and all(m["complete"] for m in meld)
fact("10", "MELD: the four main groups are complete on one identical cohort", str(meld_matched),
     "`output/consensus_thresholds.csv` cohort_sha256 and complete of the four MELD unanimity rows")

table = {}
for d in consensus.DATASETS:
    for g in consensus.MAIN_GROUPS:
        raw = unanimous.get((d, g))
        if raw:
            table.setdefault(d, {})[g] = {"coverage": pct(raw["coverage"], 2), "accuracy": pct(raw["conditional_precision"], 2),
                                          "rule": row_info(raw)["rule"], "partial": raw["complete"] != "True"}


def values(group, metric, datasets):
    return [table[d][group][metric] for d in datasets if d in table and group in table[d]
            and table[d][group][metric] is not None]


ser5_acc, ser5_cov = values("SER5", "accuracy", consensus.DATASETS), values("SER5", "coverage", consensus.DATASETS)
acted = [d for d in consensus.DATASETS if d not in DIALOGUE]
english_acted = ("crema_d", "ravdess", "savee", "jl_corpus")
stats = {
    "ser5_u95": sum(a >= 95 for a in ser5_acc), "datasets": len(consensus.DATASETS),
    "ser5_u95_names": [NAMES[d] for d in consensus.DATASETS if table.get(d, {}).get("SER5", {}).get("accuracy", 0) >= 95],
    "ser5_cov": [min(ser5_cov), max(ser5_cov)],
    "ser5_dialogue_acc": [min(values("SER5", "accuracy", DIALOGUE)), max(values("SER5", "accuracy", DIALOGUE))],
    "ser5_english_acted_acc": [min(values("SER5", "accuracy", english_acted)),
                               max(values("SER5", "accuracy", english_acted))],
    "ser5_english_acted_cov": [min(values("SER5", "coverage", english_acted)),
                               max(values("SER5", "coverage", english_acted))],
    "ser5_second_ge90": sum(a >= 90 for a in values("SER5", "accuracy", SECOND)),
    "audio_second_below50": sum(a < 50 for a in values("AudioLLM5", "accuracy", SECOND)),
    "llm3_second_below50": sum(a < 50 for a in values("LLM3", "accuracy", SECOND)),
    "all10_cov_max": max(values("All10", "coverage", consensus.DATASETS)),
    "all10_cov_median": round(statistics.median(values("All10", "coverage", consensus.DATASETS)), 1),
    "all10_zero": [NAMES[d] for d in consensus.DATASETS if table.get(d, {}).get("All10", {}).get("coverage") == 0],
    "all10_acc_ge90": sum(a >= 90 for a in values("All10", "accuracy", consensus.DATASETS)),
    "all10_acc_defined": len(values("All10", "accuracy", consensus.DATASETS)),
}
fact("11, 12", "SER5 unanimity ≥95% accurate", f"{stats['ser5_u95']}/16 ({', '.join(stats['ser5_u95_names'])})",
     "`output/consensus_thresholds.csv` group=SER5, min_votes=5, conditional_precision")
fact("11, 12", "SER5 unanimous coverage range", f"{stats['ser5_cov'][0]}–{stats['ser5_cov'][1]}%",
     "`output/consensus_thresholds.csv` group=SER5, min_votes=5, coverage (16 datasets)")
fact("8, 12", "SER5 unanimous accuracy on the four dialogue corpora",
     f"{stats['ser5_dialogue_acc'][0]}–{stats['ser5_dialogue_acc'][1]}%",
     "`output/consensus_thresholds.csv` group=SER5, min_votes=5, datasets meld/emotiontalk/dailytalk/multidialog")
fact("8", "SER5 unanimity on CREMA-D, RAVDESS, SAVEE, JL-Corpus: accuracy / coverage",
     f"{stats['ser5_english_acted_acc'][0]}–{stats['ser5_english_acted_acc'][1]}% / "
     f"{stats['ser5_english_acted_cov'][0]}–{stats['ser5_english_acted_cov'][1]}%",
     "`output/consensus_thresholds.csv` group=SER5, min_votes=5")
fact("9", "SER5 unanimity ≥90% on the second eight datasets", f"{stats['ser5_second_ge90']}/8",
     "`output/consensus_thresholds.csv` group=SER5, min_votes=5 (TESS … Emozionalmente)")
fact("9", "AudioLLM5 / LLM3 unanimity < 50% accurate on the second eight datasets",
     f"{stats['audio_second_below50']}/8 / {stats['llm3_second_below50']}/8",
     "`output/consensus_thresholds.csv` groups AudioLLM5, LLM3 at min_votes = model_count")
fact("11, 12", "All10 unanimous coverage: maximum / median", f"{stats['all10_cov_max']}% / {stats['all10_cov_median']}%",
     "`output/consensus_thresholds.csv` group=All10, min_votes = model_count")
fact("11", "All10 unanimity accepts no clip", ", ".join(stats["all10_zero"]) or "none",
     "`output/consensus_thresholds.csv` group=All10, accepted = 0")
fact("11 (notes)", "All10 unanimity ≥90% accurate", f"{stats['all10_acc_ge90']}/{stats['all10_acc_defined']} datasets with accepted clips",
     "`output/consensus_thresholds.csv` group=All10, min_votes = model_count")

meld_single = sorted((float(r["accuracy"]), consensus.SHORT_NAMES.get(r["model"], r["model"]))
                     for r in baselines if r["dataset"] == "meld" and r["group"] == "All10")
stats["meld_single"] = [round(100 * meld_single[0][0], 1), round(100 * meld_single[-1][0], 1)] if meld_single else None
stats["meld_single_names"] = [meld_single[0][1], meld_single[-1][1]] if meld_single else None
if meld_single:
    fact("2, 10", "MELD single-model accuracy on the All10 cohort (lowest–highest)",
         f"{stats['meld_single'][0]}% ({meld_single[0][1]}) – {stats['meld_single'][1]}% ({meld_single[-1][1]})",
         "`output/consensus_baselines.csv` dataset=meld, group=All10, accuracy")


def neutral(dataset, group):
    for r in by_class:
        if (r["dataset"], r["group"], r["predicted_class"]) == (dataset, group, "neutral") and \
                r["min_votes"] == r["model_count"]:
            return pct(r["share_of_accepted"], 0)
    return None


stats["neutral"] = {f"{d}_{g}": neutral(d, g) for d in ("crema_d", "multidialog", "meld", "mead")
                    for g in ("SER5", "AudioLLM5", "All10")}
best = [d for d in consensus.DATASETS if d in table and table[d].get("All10", {}).get("accuracy") is not None
        and table[d]["All10"]["accuracy"] >= max(v["accuracy"] for v in table[d].values() if v["accuracy"] is not None)]
stats["all10_best"] = len(best)
fact("9 (notes)", "Datasets where All10 unanimity is the most accurate main group (ties included)", f"{len(best)}/16",
     "`output/consensus_thresholds.csv` the four main groups at min_votes = model_count, conditional_precision")
stats["mead_audio_acc"] = table.get("mead", {}).get("AudioLLM5", {}).get("accuracy")
fact("9 (notes)", "MEAD AudioLLM5 unanimous accuracy", f"{stats['mead_audio_acc']}%",
     "`output/consensus_thresholds.csv` dataset=mead, group=AudioLLM5, min_votes = model_count")
stats["meld_best"] = max((m["accuracy"] for m in meld if m["accuracy"] is not None), default=None)
stats["extra_zero"] = {g: [NAMES[d] for d in consensus.DATASETS if (d, g) in unanimous and unanimous[d, g]["accepted"] == "0"]
                       for g in consensus.APPENDIX_GROUPS}
fact("A1–A2 (notes)", "Appendix groups whose unanimity accepts no clip", "; ".join(f"{g}: {', '.join(v)}" for g, v in
     stats["extra_zero"].items() if v) or "none", "`output/consensus_thresholds.csv` accepted = 0")
for key, value in stats["neutral"].items():
    d, g = key.rsplit("_", 1)
    fact("12, 13, A3, 9 (notes)", f"{NAMES[d]} {g} unanimity: share of agreed labels that are neutral", f"{value}%",
         f"`output/consensus_by_class.csv` dataset={d}, group={g}, predicted_class=neutral, min_votes = model_count, "
         "share_of_accepted")

numbers = {"refreshed": manifest["generated_at_singapore"], "runs": manifest["selected_run_count"],
           "main": list(consensus.MAIN_GROUPS), "extra": list(consensus.APPENDIX_GROUPS), "groups": groups,
           "pending": {"names": pending_names, "datasets": pending_datasets}, "crema": crema, "meld": meld,
           "meld_matched": meld_matched, "unanimous": table, "stats": stats,
           "names": NAMES, "first": FIRST, "second": SECOND}
fact("1, 7, 11, 13", "Models with pending results / datasets where main groups lack them",
     f"{', '.join(pending_names) or 'none'} / {pending_datasets}", "`output/consensus_manifest.json` cohorts.missing_members")
fact("1", "Compatible saved prediction runs", str(manifest["selected_run_count"]),
     "`output/consensus_manifest.json` selected_run_count")
(DECK / "assets").mkdir(exist_ok=True)
(DECK / "assets/numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("  wrote assets/numbers.json")

# Step 4: facts.md, every number with its source
step("4. Write facts.md")
lines = ["# Facts for the consensus deck", "",
         "Generated by `scripts/make_assets.py`; do not edit by hand. Every number on the slides comes from the saved "
         "consensus outputs in `research_paper/output/` (written by `research_paper/scripts/consensus.py` from saved "
         "predictions; no model was run). Percentages are rounded on the slides; the CSVs keep full precision.", "",
         f"Consensus refreshed: {manifest['generated_at_singapore']}. Paths below are relative to `research_paper/`.", "",
         "Definitions (slides 4–5): coverage = accepted / common clips; accuracy of agreed labels = correct accepted / "
         "accepted (undefined when nothing is accepted); strict majority = more than half of the m participating "
         "models; invalid answers abstain but stay in m; ties for the top label are rejected; the common cohort is the "
         "intersection of the members' audio paths (`research_paper/scripts/consensus.py`, `decision`, "
         "`consensus_details.py`, `build_details`). The voting example on slide 5 and the 1,000-clip example on slide 4 "
         "are illustrations (the latter from the report's figure guide), not data.", "",
         "| Slide | Fact | Value | Source |", "|---|---|---|---|"]
lines += [f"| {s} | {w} | {v} | {src_} |" for s, w, v, src_ in facts]
lines += ["", "Heat map (slide 11) and trade-off grids (slides 8–9, A1–A2): every cell/point is one row of "
          "`output/consensus_thresholds.csv` (coverage, conditional_precision); the class-share heat map (A3) uses "
          "`output/consensus_by_class.csv` (predicted_class=neutral, share_of_accepted). Group colours and markers: "
          "`research_paper/scripts/consensus_plot.py` (`GROUPS`), Okabe–Ito colours plus Tol's indigo.", ""]
(DECK / "facts.md").write_text("\n".join(lines), encoding="utf-8")
print(f"  wrote facts.md ({len(facts)} facts)")
