"""
Prompts for the audio-LLM evaluation: read prompts.yaml and fill in the labels.

prompts.yaml has a `system` prompt, a `user` prompt and the `labels` the model chooses from. load_prompts() puts the
labels into the prompts in place of {labels} ("angry, disgusted, ...") and checks that every label is one lowercase
word, that at least one prompt lists the labels and that no placeholder is left unfilled.

Usage (from the emotion_recognition/ folder):
    python scripts/llm_eval/prompts.py     prints the filled prompts of `llm.prompts` in config/config.yaml
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "config.yaml"


def load_prompts(path):
    """{"system": text, "user": text, "labels": [label, ...]} from a prompts YAML file, with {labels} filled in."""
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    labels = spec.get("labels") or []
    # Every label must be one lowercase word (the prompts ask for exactly one), and each may appear only once
    if not labels or len(set(labels)) < len(labels) or not all(re.fullmatch(r"[a-z]+", str(x)) for x in labels):
        sys.exit(f"ERROR: `labels` in {path} must be a list of different lowercase words, e.g. [angry, happy, sad]")
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


def main():
    path = ROOT / yaml.safe_load(CONFIG.read_text())["llm"]["prompts"]
    prompts = load_prompts(path)
    print(f"prompts : {path.relative_to(ROOT)}")
    print(f"labels  : {', '.join(prompts['labels'])}")
    for k in ("system", "user"):
        print(f"\n{k} ({len(prompts[k])} characters):\n{prompts[k]}")


if __name__ == "__main__":
    main()
