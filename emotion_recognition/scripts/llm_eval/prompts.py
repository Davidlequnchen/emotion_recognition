"""
Prompts for the audio-LLM evaluation: read prompts.yaml and fill in the labels.

prompts.yaml has a `system` prompt, a `user` prompt and the `labels` the model chooses from. load_prompts() puts the
labels into the prompts in place of {labels} ("angry, disgusted, ...") and checks that every label is one lowercase
word, that at least one prompt lists the labels and that no placeholder is left unfilled.

Usage (from the emotion_recognition/ folder):
    python scripts/llm_eval/prompts.py     prints the filled prompts of `prompts` in config/llm.yaml
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "llm.yaml"  # the settings of scripts/llm_eval/
REMINDER = "Answer with exactly one word from this list, in lowercase, and nothing else: {labels}."


def load_prompts(path, labels=None):
    """Fill a prompt with a dataset's allowed labels, in the configured vocabulary's fixed order.

    With labels=None, use the complete vocabulary (standalone previews/checks). Dataset evaluation passes
    ds.classes, never a clip's ground-truth label or listener vote.
    """
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    vocabulary = spec.get("labels") or []
    # Every label must be one lowercase word (the prompts ask for exactly one), and each may appear only once
    if (not vocabulary or len(set(vocabulary)) < len(vocabulary)
            or not all(re.fullmatch(r"[a-z]+", str(x)) for x in vocabulary)):
        sys.exit(f"ERROR: `labels` in {path} must be a list of different lowercase words, e.g. [angry, happy, sad]")
    if labels is None:
        labels = list(vocabulary)
    else:
        requested = list(labels) if not isinstance(labels, str) else [labels]
        if not requested or len(set(requested)) != len(requested) or any(lab not in vocabulary for lab in requested):
            sys.exit(f"ERROR: dataset labels must be a non-empty, distinct subset of `labels` in {path}; "
                     f"got {requested}")
        labels = [lab for lab in vocabulary if lab in requested]
    missing = [k for k in ("system", "user") if not isinstance(spec.get(k), str) or not spec[k].strip()]
    if missing:
        sys.exit(f"ERROR: {path} has no {' and no '.join(missing)} prompt")
    if not any("{labels}" in spec[k] for k in ("system", "user")):
        sys.exit(f"ERROR: neither prompt in {path} contains {{labels}}, so the model would not see the list")
    # Fill in the list of labels, then check that no other placeholder such as {language} is left
    filled = {k: spec[k].replace("{labels}", ", ".join(labels)).strip() for k in ("system", "user")}
    left = re.findall(r"\{[^{}]*\}", filled["system"] + " " + filled["user"])
    if left:
        sys.exit(f"ERROR: unfilled placeholder(s) {', '.join(left)} in {path}; only {{labels}} is filled in")
    return {"system": filled["system"], "user": filled["user"], "labels": list(labels)}


def prompt_protocol(prompts, local=False):
    """The inference protocol used by both loaders and result-reuse checks, without loading any model.

    Include the label scope even when a dataset happens to use all seven labels: pre-fix answers must not be
    mistaken for results of the dataset-constrained protocol. Local models also hash their retry reminder.
    """
    payload = {**prompts, "label_scope": "dataset"}
    if local:
        payload["reminder"] = REMINDER.format(labels=", ".join(prompts["labels"]))
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return {"label_scope": "dataset", "labels": list(prompts["labels"]), "prompt_hash": fingerprint}


def configured_protocol(entry, classes, config=None, root=None):
    """The expected inference protocol for an LLM registry entry, usable by evaluation, plots and the viewer.

    Reading configuration and filling the prompt needs no model, audio, API key or inference libraries.
    """
    if not (entry.get("api") or entry.get("llm")):
        return None
    settings = yaml.safe_load(Path(config or CONFIG).read_text())
    prompts = load_prompts(Path(root or ROOT) / settings["prompts"], labels=classes)
    protocol = prompt_protocol(prompts, local=not entry.get("api", False))
    return protocol | {"model_id": entry["api_model"] if entry.get("api") else entry["hf_repo"]}


def main():
    path = ROOT / yaml.safe_load(CONFIG.read_text())["prompts"]
    prompts = load_prompts(path)
    print(f"prompts : {path.relative_to(ROOT)}")
    print(f"labels  : {', '.join(prompts['labels'])}")
    for k in ("system", "user"):
        print(f"\n{k} ({len(prompts[k])} characters):\n{prompts[k]}")


if __name__ == "__main__":
    main()
