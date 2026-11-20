"""Graded training-source evidence shared by reports, plots and the viewer.

An absent dataset is unknown overall, not evidence of no backbone/pre-training exposure.
Old prediction booleans are historical metadata and cannot establish current exposure.
"""
from copy import deepcopy


EXPOSURE_LABEL = "Training involvement"
EXPOSURE_NOTE = ("Direct use and upstream teacher use both count as training involvement.\n"
                 "No pattern: no corpus-specific evidence, not verified unseen. Corpus use is not test-clip leakage.")
EXPOSURE_LEVELS = {
    "confirmed_exposure": {"label": "Used directly", "hatch": "///", "marker": "†"},
    "probable_exposure": {"label": "Used via upstream training", "hatch": "...", "marker": "‡"},
    "possible_exposure": {"label": "Possible association", "hatch": "xx", "marker": "?"},
    "not_established": {"label": "No corpus-specific evidence", "hatch": "", "marker": ""},
}


def training_hatch(status):
    """Return a distinct pattern for each evidence level (unknown has no pattern)."""
    return EXPOSURE_LEVELS.get(status, EXPOSURE_LEVELS["not_established"])["hatch"]


def training_marker(record):
    """Compact table marker, with a legacy-boolean fallback for external records."""
    status = record.get("training_status") or ("confirmed_exposure" if record.get("in_training_data")
                                               else "not_established")
    return EXPOSURE_LEVELS.get(status, EXPOSURE_LEVELS["not_established"])["marker"]


def training_metadata(key, entry=None):
    """Resolve corpus-specific evidence; never infer unseen from an absent entry."""
    entry = entry or {}
    listed = entry.get("training_data") or []
    listed = [listed] if isinstance(listed, str) else listed
    detail = (entry.get("training_exposure") or {}).get(key) or {}
    status = detail.get("status", "confirmed_exposure" if key in listed else "not_established")
    if status not in EXPOSURE_LEVELS:
        raise ValueError(f"Unknown training evidence status: {status!r}")
    fields = {"in_training_data": status in ("confirmed_exposure", "probable_exposure"),
              "training_status": status,
              "training_source": entry.get("training_source", "")}
    for source, target in (("scope", "training_scope"), ("reason", "training_reason")):
        if detail.get(source):
            fields[target] = detail[source]
    evidence = detail.get("evidence", entry.get("training_evidence"))
    if isinstance(evidence, dict) and key in evidence:
        fields["training_evidence"] = deepcopy(evidence[key])
    elif isinstance(evidence, (list, tuple)):
        fields["training_evidence"] = deepcopy(list(evidence))
    return fields


def with_training_provenance(record, entry=None):
    """Return current presentation metadata without changing scores, protocols or the saved record."""
    current = dict(record)
    for field in ("training_evidence", "training_scope", "training_reason"):
        current.pop(field, None)  # do not retain evidence from a superseded registry
    current.update(training_metadata(record.get("key"), entry))
    return current
