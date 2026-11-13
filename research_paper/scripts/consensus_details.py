"""Pure same-cohort voting diagnostics for validated consensus analysis jobs."""
from collections import Counter
import hashlib
import json


_COHORT_FIELDS = ("dataset", "group", "model_count", "target_count", "complete", "members",
                  "common_samples", "union_samples", "cohort_sha256")
_RATE_FIELDS = ("accepted", "accepted_correct", "accepted_errors", "coverage",
                "conditional_precision", "selective_risk", "accepted_correct_fraction",
                "accepted_fraction_of_union")
THRESHOLD_FIELDS = _COHORT_FIELDS + ("min_votes",) + _RATE_FIELDS
CLASS_FIELDS = _COHORT_FIELDS + ("min_votes", "predicted_class") + _RATE_FIELDS + ("share_of_accepted",)
BASELINE_FIELDS = _COHORT_FIELDS + ("model", "correct", "accuracy", "invalid_predictions")


def _rates(accepted, correct, common, union):
    precision = correct / accepted if accepted else None
    return {"accepted": accepted, "accepted_correct": correct,
            "accepted_errors": accepted - correct, "coverage": accepted / common,
            "conditional_precision": precision,
            "selective_risk": (accepted - correct) / accepted if accepted else None,
            "accepted_correct_fraction": correct / common,
            "accepted_fraction_of_union": accepted / union}


def build_details(jobs):
    """Return threshold, predicted-class and individual-model rows without I/O.

    Each job is ``(fields, runs, common_paths)`` after the caller has validated
    matching gold labels, classes and composite-label rules. The cohort SHA256
    covers the dataset, sorted path/gold pairs, classes and accepted-gold rules;
    model membership and predictions do not affect it. Class coverage uses the
    full common cohort; ``share_of_accepted`` uses the threshold's accepted set.
    All rates use fractions, and precision/risk are undefined for empty sets.
    """
    threshold_rows, class_rows, baseline_rows = [], [], []
    for fields, runs, common in jobs:
        paths = sorted(common)
        n, m = len(paths), len(runs)
        if not n or m < 2:
            raise ValueError("Voting details require at least two models and a nonempty cohort")
        first = runs[0]
        union = len(set.union(*(set(run.rows) for run in runs)))
        identity = {"dataset": fields["dataset"], "classes": first.classes,
                    "accept": first.accept,
                    "samples": [[path, first.rows[path]["gold"]] for path in paths]}
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
        base = {key: fields[key] for key in ("dataset", "group", "model_count", "target_count",
                                             "complete", "members")}
        base |= {"common_samples": n, "union_samples": union, "cohort_sha256": digest}
        observations = []
        for path in paths:
            gold = first.rows[path]["gold"]
            accepted_gold = first.accept.get(gold, [gold])
            counts = Counter(run.rows[path]["prediction"] for run in runs
                             if run.rows[path]["prediction"] is not None)
            winning = max(counts.values(), default=0)
            leaders = [label for label, votes in counts.items() if votes == winning]
            winner = leaders[0] if len(leaders) == 1 else None
            observations.append((winner, winning, accepted_gold))
        for minimum in range(m // 2 + 1, m + 1):
            predicted, correct = Counter(), Counter()
            for winner, votes, accepted_gold in observations:
                if winner is not None and votes >= minimum:
                    predicted[winner] += 1
                    correct[winner] += winner in accepted_gold
            accepted, accepted_correct = sum(predicted.values()), sum(correct.values())
            threshold = base | {"min_votes": minimum}
            threshold_rows.append(threshold | _rates(accepted, accepted_correct, n, union))
            for label in first.classes:
                class_rows.append(threshold | {"predicted_class": label}
                                  | _rates(predicted[label], correct[label], n, union)
                                  | {"share_of_accepted": predicted[label] / accepted if accepted else None})
        for run in runs:
            correct = invalid = 0
            for path in paths:
                row = run.rows[path]
                prediction = row["prediction"]
                correct += prediction in run.accept.get(row["gold"], [row["gold"]])
                invalid += prediction is None
            baseline_rows.append(base | {"model": run.model, "correct": correct,
                                         "accuracy": correct / n, "invalid_predictions": invalid})
    return threshold_rows, class_rows, baseline_rows
