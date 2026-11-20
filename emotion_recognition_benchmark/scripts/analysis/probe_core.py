"""CPU-only, group-disjoint linear probing primitives.

Split assignment uses labels for stratification, never features or predictions.
Fit preprocessing on training examples only and choose regularization using
validation UAR only. Test features and labels are deliberately absent from the
fitting API.
"""

from __future__ import annotations

import itertools
import math
import random
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPLIT_NAMES = ("train", "validation", "test")


def _class_names(classes: Sequence[str]) -> list[str]:
    names = list(classes)
    if len(names) < 2 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("classes must contain at least two non-empty string labels")
    if len(set(names)) != len(names):
        raise ValueError("classes contains duplicate labels")
    return names


def _labels(values: Sequence[str], classes: list[str], name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty one-dimensional label sequence")
    if any(not isinstance(value, (str, np.str_)) for value in result):
        raise ValueError(f"{name} labels must be strings")
    unknown = set(result.tolist()) - set(classes)
    if unknown:
        raise ValueError(f"{name} contains unknown labels: {sorted(unknown)!r}")
    return result


def _features(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric feature matrix") from exc
    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must have shape (n_samples, n_features), both non-zero")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite features")
    return result


def _group_counts(
    n_groups: int,
    fractions: Sequence[float],
    group_counts: Sequence[int] | Mapping[str, int] | None,
) -> np.ndarray:
    proportions = np.asarray(fractions, dtype=np.float64)
    if (
        proportions.shape != (3,)
        or not np.isfinite(proportions).all()
        or np.any(proportions <= 0)
        or not np.isclose(proportions.sum(), 1.0)
    ):
        raise ValueError("fractions must be three positive finite values summing to 1")
    if group_counts is not None:
        if isinstance(group_counts, Mapping):
            if set(group_counts) != set(SPLIT_NAMES):
                raise ValueError("group_counts mapping must contain train, validation, and test")
            values = [group_counts[name] for name in SPLIT_NAMES]
        else:
            values = list(group_counts)
        if len(values) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
            for value in values
        ):
            raise ValueError("group_counts must contain three positive integers")
        if sum(values) != n_groups:
            raise ValueError("group_counts must sum to the actual number of groups")
        return np.asarray(values, dtype=np.int64)

    targets = proportions * n_groups
    counts = np.maximum(1, np.floor(targets).astype(np.int64))
    while counts.sum() < n_groups:
        counts[int(np.argmax(targets - counts))] += 1
    while counts.sum() > n_groups:
        excess = np.where(counts > 1, counts - targets, -np.inf)
        counts[int(np.argmax(excess))] -= 1
    return counts


def make_group_splits(
    paths: list[str],
    labels: list[str],
    groups: list[str],
    seed: int,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    group_counts: Sequence[int] | Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Return deterministic, class-balanced, group-disjoint clip assignments.

    The input must describe the full corpus before excluding missing features.
    Reuse the returned assignment for every model; do not call this on each
    model's available-feature subset. Groups are sorted before seeded search,
    making assignment independent of CSV row order. Explicit group counts must
    match the corpus exactly. Small group sets are searched exhaustively; larger
    sets use deterministic random initialization and count-preserving swaps.

    Every class must be present in every partition. An impossible or unresolved
    group-stratification problem raises rather than falling back to clip splits.
    Only label counts enter the balancing objective, never model performance.
    """
    if not (len(paths) == len(labels) == len(groups)) or not paths:
        raise ValueError("paths, labels, and groups must have the same non-zero length")
    for name, values in (("paths", paths), ("labels", labels), ("groups", groups)):
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{name} must contain non-empty strings")
    if len(set(paths)) != len(paths):
        raise ValueError("paths must be unique so no clip assignment can be overwritten")
    ordered_groups = sorted(set(groups))
    if len(ordered_groups) < 3:
        raise ValueError("at least three independent groups are required; clip-random fallback is forbidden")
    counts = _group_counts(len(ordered_groups), fractions, group_counts)
    classes = sorted(set(labels))
    group_index = {group: index for index, group in enumerate(ordered_groups)}
    class_index = {label: index for index, label in enumerate(classes)}
    histograms = np.zeros((len(ordered_groups), len(classes)), dtype=np.int64)
    for label, group in zip(labels, groups):
        histograms[group_index[group], class_index[label]] += 1
    class_group_counts = np.count_nonzero(histograms, axis=0)
    rare = [label for label, count in zip(classes, class_group_counts) if count < 3]
    if rare:
        raise ValueError(f"each class needs at least three independent groups; insufficient: {rare!r}")

    # Fixed group counts define the intended split proportions. This also
    # respects corpus-specific protocols such as 61/15/15 speakers.
    proportions = counts / len(ordered_groups)
    totals = histograms.sum(axis=0)
    target = proportions[:, None] * totals[None, :]
    clip_target = proportions * len(paths)

    def partition_counts(assignment: np.ndarray) -> np.ndarray:
        return np.stack([histograms[assignment == part].sum(axis=0) for part in range(3)])

    def objective(partition: np.ndarray) -> float:
        missing = int(np.count_nonzero(partition == 0))
        # Relative errors give each class the same influence despite imbalance.
        class_error = np.square((partition - target) / np.maximum(target, 1.0)).mean()
        size_error = np.square((partition.sum(axis=1) - clip_target) / clip_target).mean()
        return missing * 1_000_000.0 + float(class_error + 0.1 * size_error)

    best_assignment: np.ndarray | None = None
    best_score = math.inf

    def consider(assignment: np.ndarray) -> None:
        nonlocal best_assignment, best_score
        score = objective(partition_counts(assignment))
        if score < best_score - 1e-12:
            best_assignment = assignment.copy()
            best_score = score

    n_groups = len(ordered_groups)
    assignment_count = math.comb(n_groups, int(counts[0])) * math.comb(
        n_groups - int(counts[0]), int(counts[1])
    )
    if assignment_count <= 20_000:
        # Seeded ordering allows different seeds while retaining exhaustive
        # coverage and reproducible tie-breaking for small speaker sets.
        order = list(range(n_groups))
        random.Random(seed).shuffle(order)
        for train_indices in itertools.combinations(order, int(counts[0])):
            train_set = set(train_indices)
            remaining = [index for index in order if index not in train_set]
            for val_indices in itertools.combinations(remaining, int(counts[1])):
                assignment = np.full(n_groups, 2, dtype=np.int64)
                assignment[list(train_indices)] = 0
                assignment[list(val_indices)] = 1
                consider(assignment)
    else:
        rng = random.Random(seed)
        template = np.repeat(np.arange(3), counts)
        for _ in range(512):
            order = list(range(n_groups))
            rng.shuffle(order)
            assignment = np.empty(n_groups, dtype=np.int64)
            assignment[order] = template
            consider(assignment)
        assert best_assignment is not None
        # Optimize swaps without ever changing the requested group counts.
        partition = partition_counts(best_assignment)
        for _ in range(12):
            swap: tuple[int, int] | None = None
            next_partition = partition
            next_score = best_score
            for left in range(n_groups):
                left_part = int(best_assignment[left])
                for right in range(left + 1, n_groups):
                    right_part = int(best_assignment[right])
                    if left_part == right_part:
                        continue
                    candidate = partition.copy()
                    delta = histograms[right] - histograms[left]
                    candidate[left_part] += delta
                    candidate[right_part] -= delta
                    score = objective(candidate)
                    if score < next_score - 1e-12:
                        swap, next_partition, next_score = (left, right), candidate, score
            if swap is None:
                break
            left, right = swap
            best_assignment[left], best_assignment[right] = best_assignment[right], best_assignment[left]
            partition, best_score = next_partition, next_score

    assert best_assignment is not None
    if np.any(partition_counts(best_assignment) == 0):
        raise ValueError(
            "could not construct three group-disjoint partitions with every class; "
            "review class/group support or the requested group counts"
        )
    return {
        path: SPLIT_NAMES[int(best_assignment[group_index[group]])]
        for path, group in zip(paths, groups)
    }


def score_predictions(
    y_true: Sequence[str], probabilities: Any, classes: Sequence[str]
) -> dict[str, Any]:
    """Score genuine class probabilities in the exact supplied class order.

    Brier score is the multiclass mean sum of squared probability errors (not
    divided by the number of classes). UAR and macro F1 use the fixed class list;
    missing reference classes therefore contribute zero recall and are exposed
    in support. The caller should require full support for comparable results.
    """
    names = _class_names(classes)
    truth = _labels(y_true, names, "y_true")
    proba = np.asarray(probabilities, dtype=np.float64)
    if proba.shape != (len(truth), len(names)):
        raise ValueError("probabilities must have shape (n_samples, len(classes))")
    if not np.isfinite(proba).all():
        raise ValueError("probabilities contains non-finite values")
    if np.any(proba < 0) or np.any(proba > 1):
        raise ValueError("probabilities must be within [0, 1]")
    if not np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("probability rows must sum to 1")
    predicted_indices = proba.argmax(axis=1)
    predicted = np.asarray(names)[predicted_indices]
    matrix = confusion_matrix(truth, predicted, labels=names)
    support = matrix.sum(axis=1)
    recall = np.divide(
        matrix.diagonal(), support, out=np.zeros(len(names), dtype=float), where=support != 0
    )
    label_index = {label: index for index, label in enumerate(names)}
    true_indices = np.asarray([label_index[label] for label in truth], dtype=np.int64)
    one_hot = np.eye(len(names), dtype=np.float64)[true_indices]
    # Numeric labels prevent sklearn's lexical label sorting from silently
    # changing the meaning of columns when names is in a custom class order.
    result = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "uar": float(recall.mean()),
        "macro_f1": float(f1_score(truth, predicted, labels=names, average="macro", zero_division=0)),
        "log_loss": float(log_loss(true_indices, proba, labels=np.arange(len(names)))),
        "brier_score": float(np.square(proba - one_hot).sum(axis=1).mean()),
        "confusion_matrix": matrix.tolist(),
        "classes": names,
        "support": support.tolist(),
        "n_samples": int(len(truth)),
    }
    if not all(np.isfinite(result[name]) for name in ("accuracy", "uar", "macro_f1", "log_loss", "brier_score")):
        raise ValueError("scoring produced non-finite metrics")
    return result


def fit_linear_probe(
    x_train: Any,
    y_train: Sequence[str],
    x_val: Any,
    y_val: Sequence[str],
    classes: Sequence[str],
    c_values: Sequence[float],
    class_weight: str | Mapping[str, float] | None = "balanced",
    max_iter: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit standardized linear logistic probes and select C by validation UAR.

    StandardScaler and LogisticRegression see training rows only, including
    balanced class weights. For multiclass data, lbfgs uses multinomial softmax;
    binary logistic probabilities use its equivalent sigmoid parameterization.
    No test inputs are accepted and the winner is not refitted on validation.
    Equal validation UAR keeps the first C in the provided grid. Convergence
    warnings are returned for every fit rather than suppressed silently.

    The returned model is a native sklearn Pipeline. Its predict_proba columns
    follow probability_class_order; index those columns using
    probability_column_indices to score in the supplied classes order.
    """
    names = _class_names(classes)
    train = _features(x_train, "x_train")
    validation = _features(x_val, "x_val")
    if train.shape[1] != validation.shape[1]:
        raise ValueError("training and validation feature dimensions differ")
    train_labels = _labels(y_train, names, "y_train")
    val_labels = _labels(y_val, names, "y_val")
    if len(train_labels) != len(train) or len(val_labels) != len(validation):
        raise ValueError("feature and label sample counts differ")
    missing_train = set(names) - set(train_labels.tolist())
    if missing_train:
        raise ValueError(f"training data is missing classes: {sorted(missing_train)!r}")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    grid = list(c_values)
    if not grid:
        raise ValueError("c_values must not be empty")
    try:
        grid = [float(value) for value in grid]
    except (TypeError, ValueError) as exc:
        raise ValueError("c_values must contain positive finite numbers") from exc
    if any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("c_values must contain positive finite numbers")

    search: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_uar = -math.inf
    best_index = -1
    best_columns: list[int] = []
    for c_value in grid:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight=class_weight,
                        max_iter=int(max_iter),
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train, train_labels)
        native_order = model.classes_.tolist()
        columns = [native_order.index(label) for label in names]
        probabilities = model.predict_proba(validation)[:, columns]
        metrics = score_predictions(val_labels, probabilities, names)
        warning_rows = [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in caught
        ]
        converged = not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
        row = {
            "C": c_value,
            "validation_uar": metrics["uar"],
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "converged": converged,
            "n_iter": model.named_steps["classifier"].n_iter_.astype(int).tolist(),
            "warnings": warning_rows,
        }
        search.append(row)
        if metrics["uar"] > best_uar:
            best_model, best_uar, best_index, best_columns = model, metrics["uar"], len(search) - 1, columns
    assert best_model is not None
    selected = search[best_index]
    return {
        "model": best_model,
        "selected_C": selected["C"],
        "classes": names,
        "probability_class_order": best_model.classes_.tolist(),
        "probability_column_indices": best_columns,
        "validation_search": search,
        "selected_validation_uar": best_uar,
        "converged": selected["converged"],
        "convergence_warnings": [row for row in selected["warnings"] if row["category"] == "ConvergenceWarning"],
        "any_search_nonconvergence": any(not row["converged"] for row in search),
    }
