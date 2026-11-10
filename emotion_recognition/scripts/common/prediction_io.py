"""Read saved prediction CSVs without changing the source artifacts.

CSV exports from other machines can have a UTF-8 BOM and Yes/No correctness
columns. Present the same canonical True/False strings to all result consumers.
"""
import csv
from pathlib import Path


def parse_correct(value):
    """Parse an explicit correctness flag; reject unknown values instead of scoring them wrong."""
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise ValueError(f"Unrecognized prediction correctness value: {value!r}")


def read_prediction_rows(path):
    """Return rows with a BOM-free header and normalized correctness fields, in memory only."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for line, row in enumerate(rows, 2):
        for key in row:
            if isinstance(key, str) and key.startswith("correct"):
                try:
                    row[key] = str(parse_correct(row[key]))
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{line}, {key}: {exc}") from exc
    return rows
