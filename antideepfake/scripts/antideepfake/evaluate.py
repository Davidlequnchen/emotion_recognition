"""
Evaluate AntiDeepfake on real speech versus voice clones, on clean audio and through a simulated telephone channel.

Every step prints what it does. For each file and condition the detector gives P(real); the script then measures,
per group of fake clips (one per voice-cloning model) against all real speech:
  - detection rate: share of fake clips with P(real) < threshold (and, per real group, the share kept as real)
  - ROC-AUC: chance that a random fake clip scores as more fake than a random real one (1 = perfect, 0.5 = chance)
  - EER: the error rate where real clips flagged as fake and fake clips passed as real are equally frequent,
    and the P(real) threshold at which that happens

Usage (from the antideepfake/ folder, in the emotion_recognition or antideepfake environment):
    python scripts/antideepfake/evaluate.py
Settings are in config/evaluation.yaml (groups of real and fake files, conditions, telephone channel, output);
the detector (checkpoint, mode, devices, max_seconds) in config/antideepfake.yaml.

Output, in outputs/antideepfake/<name>/ (not in git):
    scores.csv       one row per file and condition: group, label, P(real), logits, duration
    summary.md/json  the metrics above, per condition
    scores.png       distribution of the scores per group and condition; roc.png: ROC curves
    telephone_audio/ the telephone versions of the files (8 kHz), when telephone.save_audio is true
"""
import csv
import json
import os
import platform
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # inference.py and telephone.py next to this file
import telephone  # noqa: E402
from inference import ROOT, SAMPLE_RATE, collect_audio, load_audio, load_model, score  # noqa: E402

CONFIG = ROOT / "config" / "evaluation.yaml"
DETECTOR_CONFIG = ROOT / "config" / "antideepfake.yaml"


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def rel(path):
    """A path relative to antideepfake/, for the tables (absolute when it is on another drive)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def auc_eer(real, fake):
    """real, fake: log-odds of being real (logit_real - logit_fake). Returns ROC-AUC, EER and the P(real) threshold
    at the EER, with 'fake' as the positive class (a lower log-odds is more fake)."""
    real, fake = np.asarray(real), np.asarray(fake)
    # AUC = P(fake scores lower than real), ties count half (Mann-Whitney)
    auc = ((fake[:, None] < real[None, :]).mean() + 0.5 * (fake[:, None] == real[None, :]).mean())
    # EER: flag a file as fake when its log-odds <= t; sweep t over every score
    best = None
    for t in np.unique(np.concatenate([real, fake])):
        fpr = (real <= t).mean()  # real flagged as fake
        fnr = (fake > t).mean()   # fake passed as real
        if best is None or abs(fpr - fnr) < best[0]:
            best = (abs(fpr - fnr), (fpr + fnr) / 2, t)
    return float(auc), float(best[1]), float(1 / (1 + np.exp(-best[2])))


def roc_points(real, fake):
    """False-positive and true-positive rates (fake = positive) over every threshold, for the ROC plot."""
    real, fake = np.asarray(real), np.asarray(fake)
    ts = np.concatenate([[-np.inf], np.unique(np.concatenate([real, fake])), [np.inf]])
    return [(real <= t).mean() for t in ts], [(fake <= t).mean() for t in ts]


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # ------------------------------------------------------------------
    # Step 1: Read the two configurations
    # ------------------------------------------------------------------
    step("Step 1/7 | Load configuration")
    cfg = yaml.safe_load(CONFIG.read_text())
    det = yaml.safe_load(DETECTOR_CONFIG.read_text())
    out_dir = ROOT / "outputs" / "antideepfake" / cfg["name"]
    device_name = det["devices"][det["mode"]]
    tel = cfg["telephone"]
    print(f"  config      : {rel(CONFIG)} (run '{cfg['name']}'), detector: {rel(DETECTOR_CONFIG)}")
    print(f"  conditions  : {', '.join(cfg['conditions'])}")
    if "telephone" in cfg["conditions"]:
        print(f"  telephone   : {telephone.describe(tel)}")
    print(f"  threshold   : P(real) >= {cfg['threshold']} counts as real")
    print(f"  output      : {rel(out_dir)}")

    # ------------------------------------------------------------------
    # Step 2: Software stack and device
    # ------------------------------------------------------------------
    step("Step 2/7 | Environment and device")
    print(f"  Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"  PyTorch     : {torch.__version__}, torchaudio {torchaudio.__version__}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: CUDA was requested (config/antideepfake.yaml) but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        sys.exit("ERROR: MPS was requested (config/antideepfake.yaml) but is not available.")
    print(f"  device      : {device} (mode '{det['mode']}' in config/antideepfake.yaml)")

    # ------------------------------------------------------------------
    # Step 3: Load the detector
    # ------------------------------------------------------------------
    step("Step 3/7 | Load model")
    ckpt = ROOT / det["model"]["dir"] / det["model"]["checkpoint"]
    if not ckpt.exists():
        sys.exit(f"ERROR: {ckpt} not found. Download it first: python scripts/antideepfake/download_model.py")
    t0 = time.perf_counter()
    model = load_model(ckpt)[0].to(device)
    print(f"  {ckpt.name}: loaded in {time.perf_counter() - t0:.1f} s, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, on {device}")

    # ------------------------------------------------------------------
    # Step 4: Collect the files of every group (at most per_group_limit, drawn with the seed)
    # ------------------------------------------------------------------
    step("Step 4/7 | Collect audio")
    groups = []  # (group, label, files)
    for label in ("real", "fake"):
        for group, entries in cfg[label].items():
            files = collect_audio(entries)
            if not files:
                sys.exit(f"ERROR: no audio files in group '{group}' ({entries}).")
            if cfg["per_group_limit"] and len(files) > cfg["per_group_limit"]:
                files = sorted(random.Random(cfg["seed"]).sample(files, cfg["per_group_limit"]))
            groups.append((group, label, files))
            print(f"  {label:<4} {group:<14} {len(files):>4} files  ({', '.join(entries)})")
    total = sum(len(f) for _, _, f in groups) * len(cfg["conditions"])
    print(f"  to score    : {total} (files x conditions)")

    # ------------------------------------------------------------------
    # Step 5: Score every file under every condition, one file at a time
    # ------------------------------------------------------------------
    step("Step 5/7 | Score")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, done, t_start = [], 0, time.perf_counter()
    cpu_model = None

    def predict(wav):
        """MPS cannot run the first convolution on files longer than 20.48 s: score those on the CPU."""
        nonlocal cpu_model
        try:
            return score(model, wav, device, det["inference"]["max_seconds"])
        except NotImplementedError:
            if cpu_model is None:
                cpu_model = load_model(ckpt)[0]
            return score(cpu_model, wav, torch.device("cpu"), det["inference"]["max_seconds"])

    with torch.inference_mode():
        for condition in cfg["conditions"]:
            print(f"\n  --- {condition} ---")
            for group, label, files in groups:
                odds = []
                for f in files:
                    wav = load_audio(f)  # 16 kHz mono, as AntiDeepfake reads it
                    if condition == "telephone":
                        rng = np.random.default_rng(cfg["seed"] + zlib.crc32(rel(f).encode()))  # same noise each run
                        wav, narrow = telephone.apply(wav, SAMPLE_RATE, tel, rng)
                        if tel["save_audio"]:
                            target = out_dir / "telephone_audio" / group / f.name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            sf.write(target, narrow.numpy(), tel["sample_rate"], subtype="PCM_16")
                    logits = predict(wav)
                    p_real = torch.softmax(logits, dim=0)[1].item()
                    odds.append(float(logits[1] - logits[0]))
                    rows.append({"condition": condition, "group": group, "label": label, "file": rel(f),
                                 "p_real": f"{p_real:.6f}", "logit_fake": f"{logits[0]:.4f}",
                                 "logit_real": f"{logits[1]:.4f}", "duration_s": f"{len(wav) / SAMPLE_RATE:.2f}"})
                    done += 1
                p = 1 / (1 + np.exp(-np.array(odds)))
                kept = (p >= cfg["threshold"]).mean() if label == "real" else (p < cfg["threshold"]).mean()
                print(f"  {label:<4} {group:<14} {len(files):>4} files  median P(real) {np.median(p):.3f}  "
                      f"{'kept as real' if label == 'real' else 'detected as fake'} {kept:.1%}   "
                      f"[{done}/{total}, {time.perf_counter() - t_start:.0f} s]")

    with open(out_dir / "scores.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  scores      : {rel(out_dir / 'scores.csv')} ({len(rows)} rows)")

    # ------------------------------------------------------------------
    # Step 6: Metrics per condition: each fake group against all real speech, and all fakes together
    # ------------------------------------------------------------------
    step("Step 6/7 | Metrics")
    odds_of = lambda c, g: [float(r["logit_real"]) - float(r["logit_fake"]) for r in rows
                            if r["condition"] == c and r["group"] == g]
    real_groups = [g for g, lab, _ in groups if lab == "real"]
    fake_groups = [g for g, lab, _ in groups if lab == "fake"]
    summary = {"name": cfg["name"], "threshold": cfg["threshold"], "telephone": tel, "conditions": {}}
    lines = [f"# AntiDeepfake evaluation: {cfg['name']}", "",
             f"Threshold: P(real) >= {cfg['threshold']} counts as real. Telephone channel: {telephone.describe(tel)}.",
             "AUC and EER: each fake group against all real groups together (fake = positive class).", ""]
    for c in cfg["conditions"]:
        real_all = [x for g in real_groups for x in odds_of(c, g)]
        res = {"real": {}, "fake": {}}
        lines += [f"## {c}", "", "| Group | Label | Files | Median P(real) | Correct at threshold | ROC-AUC | EER | "
                  "EER threshold (P(real)) |", "|---|---|---|---|---|---|---|---|"]
        for g in real_groups:
            p = 1 / (1 + np.exp(-np.array(odds_of(c, g))))
            res["real"][g] = {"files": len(p), "median_p_real": float(np.median(p)),
                              "kept_as_real": float((p >= cfg["threshold"]).mean())}
            lines.append(f"| {g} | real | {len(p)} | {np.median(p):.3f} | {res['real'][g]['kept_as_real']:.1%} | "
                         "– | – | – |")
        for g in fake_groups + ["all fakes"]:
            fake = [x for fg in fake_groups for x in odds_of(c, fg)] if g == "all fakes" else odds_of(c, g)
            p = 1 / (1 + np.exp(-np.array(fake)))
            auc, eer, thr = auc_eer(real_all, fake)
            res["fake"][g] = {"files": len(p), "median_p_real": float(np.median(p)),
                              "detected_as_fake": float((p < cfg["threshold"]).mean()),
                              "roc_auc": auc, "eer": eer, "eer_threshold_p_real": thr}
            lines.append(f"| {g} | fake | {len(p)} | {np.median(p):.3f} | {res['fake'][g]['detected_as_fake']:.1%} | "
                         f"{auc:.3f} | {eer:.1%} | {thr:.3f} |")
        summary["conditions"][c] = res
        lines.append("")
        print(f"\n  --- {c} ---")
        for g, m in res["fake"].items():
            print(f"  {g:<14} detected {m['detected_as_fake']:>6.1%}   AUC {m['roc_auc']:.3f}   EER {m['eer']:>6.1%}")
        for g, m in res["real"].items():
            print(f"  {g:<14} kept as real {m['kept_as_real']:>6.1%}")
    (out_dir / "summary.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  summary     : {rel(out_dir / 'summary.md')}, summary.json")

    # ------------------------------------------------------------------
    # Step 7: Plots: score distribution per group, ROC curves per fake group
    # ------------------------------------------------------------------
    step("Step 7/7 | Plots")
    if not cfg["plots"]:
        print("  skipped (plots: false)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = real_groups + fake_groups
    fig, axes = plt.subplots(1, len(cfg["conditions"]), figsize=(6 * len(cfg["conditions"]), 5), squeeze=False)
    for ax, c in zip(axes[0], cfg["conditions"]):
        data = [odds_of(c, g) for g in names]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, g in zip(box["boxes"], names):
            patch.set_facecolor("#9fd8a8" if g in real_groups else "#f2a5a0")
        for i, d in enumerate(data, 1):  # every file as a dot
            ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(d)), d, s=6, color="k", alpha=0.35, zorder=3)
        ax.axhline(np.log(cfg["threshold"] / (1 - cfg["threshold"])), color="grey", ls="--", lw=1)
        ax.set_xticks(range(1, len(names) + 1), names, rotation=30, ha="right")
        ax.set_title(c)
        ax.set_ylabel("log-odds of real (logit_real - logit_fake); > 0 = real")
    fig.tight_layout()
    fig.savefig(out_dir / "scores.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(cfg["conditions"]), figsize=(5 * len(cfg["conditions"]), 5), squeeze=False)
    for ax, c in zip(axes[0], cfg["conditions"]):
        real_all = [x for g in real_groups for x in odds_of(c, g)]
        for g in fake_groups:
            fpr, tpr = roc_points(real_all, odds_of(c, g))
            m = summary["conditions"][c]["fake"][g]
            ax.plot(fpr, tpr, label=f"{g} (AUC {m['roc_auc']:.3f}, EER {m['eer']:.1%})")
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set_xlabel("real flagged as fake (false-positive rate)")
        ax.set_ylabel("fake detected (true-positive rate)")
        ax.set_title(c)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=200)
    plt.close(fig)
    print(f"  plots       : {rel(out_dir / 'scores.png')}, {rel(out_dir / 'roc.png')}")


if __name__ == "__main__":
    main()
