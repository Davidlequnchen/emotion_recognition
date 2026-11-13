#!/usr/bin/env python3
"""Normalize Label Studio decisions while preserving every annotator's vote."""

import argparse
import json
from pathlib import Path
import sys


EMOTIONS = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "unknown"}
DECISIONS = EMOTIONS | {"bad_clip"}


def normalize(tasks):
    if not isinstance(tasks, list):
        raise ValueError("Input must be a Label Studio JSON array of tasks")
    rows = []
    cancelled_count = 0
    empty_task_count = 0
    for task_index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise ValueError(f"Task {task_index}: expected an object")
        annotations = task.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"Task {task_index}: annotations must be an array")
        if not annotations:
            empty_task_count += 1
            continue
        for annotation_index, annotation in enumerate(annotations, 1):
            context = f"Task {task_index}, annotation {annotation_index}"
            if not isinstance(annotation, dict):
                raise ValueError(f"{context}: expected an annotation object")
            if annotation.get("was_cancelled", False):
                cancelled_count += 1
                continue
            data = task.get("data", {})
            clip_id = data.get("clip_id") if isinstance(data, dict) else None
            if not isinstance(clip_id, str) or not clip_id.strip():
                raise ValueError(f"{context}: missing or empty data.clip_id")
            results = annotation.get("result", [])
            if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
                raise ValueError(f"{context}: result must be an array of objects")
            decisions = [item for item in results
                         if item.get("from_name") == "decision" and item.get("type") == "choices"]
            if len(decisions) != 1:
                raise ValueError(f"{context}: expected exactly one decision choices result")
            value = decisions[0].get("value", {})
            choices = value.get("choices") if isinstance(value, dict) else None
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError(f"{context}: decision must contain exactly one choice")
            decision = choices[0]
            if not isinstance(decision, str) or decision not in DECISIONS:
                raise ValueError(f"{context}: unknown decision value")
            annotator_id = annotation.get("completed_by")
            if isinstance(annotator_id, dict):
                annotator_id = annotator_id.get("id")
            if not isinstance(annotator_id, int) or isinstance(annotator_id, bool):
                raise ValueError(f"{context}: completed_by must be an integer or object with integer id")
            bad_clip = decision == "bad_clip"
            rows.append({
                "clip_id": clip_id,
                "task_id": task.get("id"),
                "annotation_id": annotation.get("id"),
                "annotator_id": annotator_id,
                "lead_time": annotation.get("lead_time"),
                "decision": decision,
                "status": "bad_clip" if bad_clip else "accepted",
                "emotion": None if bad_clip else decision,
            })
    return rows, cancelled_count, empty_task_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Native Label Studio JSON export")
    parser.add_argument("--output", required=True, help="New normalized JSON file; existing files are never overwritten")
    args = parser.parse_args()
    try:
        source = Path(args.input).expanduser()
        output = Path(args.output).expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists; choose a new filename: {output}")
        tasks = json.loads(source.read_text(encoding="utf-8-sig"))
        rows, cancelled_count, empty_task_count = normalize(tasks)
        serialized = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(rows)} annotation rows to {output}")
    print(f"Skipped cancelled annotations: {cancelled_count}; tasks without annotations: {empty_task_count}.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
