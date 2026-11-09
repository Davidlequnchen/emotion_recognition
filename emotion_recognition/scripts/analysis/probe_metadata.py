"""Auditable speaker/dialogue identities for the 13 evaluation datasets.

This module only reads small CSV metadata; it never loads audio, imports a
dataset loader (some loaders extract audio), or writes dataset files. Filename
rules were checked against ``common/eval_datasets.py`` and the local source
documentation: nEMO's README, MCAE's ``Files description_sentence corpus.txt``,
JL-Corpus's ``Format_Intro.txt``, and ``docs/datasets_survey_en.md`` section 6.
MESS's rule is explicitly an inference recorded in that project survey, rather
than an official naming specification; its provenance remains in every row.

THAI-SER identities come from ``labels.csv:actor_id``, never the session or
microphone. Emozionalmente's numeric basenames are sample IDs, so its identities
must come from ``samples.csv:actor``. TESS's known ``OA_bite_neutral.wav`` typo is
the OAF actress. MELD uses dialogue IDs: dialogue disjointness does not imply
speaker disjointness, and this project's MELD audio is official **test only**.

Unsupported/ambiguous names fail closed. The caller can supply an audited
``groups.csv`` instead; no clip ID is ever substituted for an unknown speaker.
"""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
import re


DATASET_KEYS = frozenset({
    "crema_d", "mcae_spps", "thai_ser", "nemo", "mess", "ravdess", "tess",
    "jl_corpus", "mead", "meld", "emovo", "emozionalmente", "savee",
})

_FILENAME_RULES = {
    "crema_d": (
        r"(?P<speaker>\d{4})_[A-Z]{3}_(?:ANG|DIS|FEA|HAP|NEU|SAD)_(?:XX|LO|MD|HI)",
        "filename <actor>_<sentence>_<emotion>_<intensity>; "
        "docs/datasets_survey_en.md section 6 (CREMA-D)",
    ),
    "mcae_spps": (
        r"(?P<speaker>[1-6])[1-7]\d{3}",
        "filename first digit is actor 1-6; "
        "data/mcae_spps/Files description_sentence corpus.txt: Audio files coding rule",
    ),
    "nemo": (
        r"(?P<speaker>[A-Z]{2}\d)_(?:anger|fear|happiness|neutral|sadness|surprised)_\d+",
        "filename <speaker_id>_<emotion>_<sentence_id>; data/nemo/README.md: Data Fields",
    ),
    "mess": (
        r"[ACHS](?P<speaker>[FM][1-3])[MTV]\d{2}_SCR",
        "filename gender+speaker number (F1-F3/M1-M3), inferred from corpus pattern; "
        "docs/datasets_survey_en.md section 6 (MESS; no official naming scheme supplied)",
    ),
    "jl_corpus": (
        r"(?P<speaker>(?:female|male)[12])_[a-z]+_\d+[ab]_[12]",
        "filename gender+speaker.ID prefix, shared across sessions/repetitions; "
        "data/jl_corpus/audio/Raw JL corpus (unchecked and unannotated)/Format_Intro.txt",
    ),
    "savee": (
        r"(?P<speaker>DC|JE|JK|KL)_(?:a|d|f|h|n|sa|su)\d{2}",
        "filename <speaker>_<emotion><sentence>; "
        "docs/datasets_survey_en.md section 6 (SAVEE: DC/JE/JK/KL)",
    ),
}


def _failure(dataset_key: str, detail: str) -> ValueError:
    return ValueError(
        f"{dataset_key}: {detail}. Restore the source metadata or provide an audited "
        "groups.csv with audio_path, group and group_kind (speaker or dialogue); "
        "per-clip or random fallback groups are not allowed."
    )


def _relative_path(dataset_key: str, audio_path: str) -> PurePosixPath:
    """Normalize separators without permitting absolute paths or parent traversal."""
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise _failure(dataset_key, f"empty or invalid audio_path {audio_path!r}")
    path = PurePosixPath(audio_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", str(path)):
        raise _failure(dataset_key, f"audio_path must be project-relative: {audio_path!r}")
    # Basenames remain usable for filename-defined corpora. When a full loader
    # path is supplied, catch accidental cross-dataset inputs instead of guessing.
    if "data" in path.parts:
        at = path.parts.index("data")
        if len(path.parts) <= at + 1 or path.parts[at + 1] != dataset_key:
            raise _failure(dataset_key, f"audio_path belongs to another dataset: {audio_path!r}")
    if path.suffix.lower() not in {".wav", ".flac"}:
        raise _failure(dataset_key, f"unsupported audio filename {audio_path!r}")
    return path


def _project_root(project_root: Path) -> Path:
    """Accept the inner project root, or its enclosing Git repository root."""
    root = Path(project_root)
    if not (root / "data").is_dir() and (root / "emotion_recognition" / "data").is_dir():
        root = root / "emotion_recognition"
    return root


def _csv_identities(
    dataset_key: str,
    project_root: Path,
    candidates: tuple[str, ...],
    file_column: str,
    speaker_column: str,
) -> tuple[dict[str, str], str]:
    """Read the source's clip-to-actor mapping and reject missing/conflicting IDs."""
    metadata = next((project_root / p for p in candidates if (project_root / p).is_file()), None)
    if metadata is None:
        raise _failure(dataset_key, f"required speaker metadata is missing: {candidates[0]}")
    relative = metadata.relative_to(project_root).as_posix()
    identities: dict[str, str] = {}
    try:
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {file_column, speaker_column} <= set(reader.fieldnames or []):
                raise _failure(
                    dataset_key,
                    f"{relative} needs columns {file_column!r} and {speaker_column!r}",
                )
            for line, row in enumerate(reader, 2):
                filename = (row.get(file_column) or "").strip()
                speaker = (row.get(speaker_column) or "").strip()
                if not filename or not speaker or speaker.lower() in {"none", "null", "nan"}:
                    raise _failure(dataset_key, f"missing filename/actor in {relative}:{line}")
                filename = PurePosixPath(filename.replace("\\", "/")).name
                if filename in identities and identities[filename] != speaker:
                    raise _failure(dataset_key, f"conflicting actors for {filename!r} in {relative}:{line}")
                identities[filename] = speaker
    except (OSError, UnicodeError, csv.Error) as exc:
        raise _failure(dataset_key, f"cannot read {relative}: {exc}") from exc
    if not identities:
        raise _failure(dataset_key, f"{relative} contains no actor identities")
    return identities, f"{relative}: {file_column} -> {speaker_column}"


def infer_groups(
    dataset_key: str, audio_paths: list[str], project_root: Path,
) -> dict[str, dict]:
    """Map each original relative path to its verified identity and provenance.

    Returns ``{audio_path: {group, group_kind, source}}`` with dataset-namespaced
    group IDs. Identity derivation is independent of input order, emotion,
    intensity, session, and microphone. No audio file needs to exist for the
    filename-defined corpora. THAI-SER and Emozionalmente require their original
    source CSVs; a filename alone does not establish the actor for these corpora.

    ``ValueError`` requests an audited ``groups.csv`` when a dataset, path or
    identity cannot be understood safely. Deciding whether a corpus has enough
    groups for a requested split, or whether test-only MELD may be repartitioned,
    is the caller's responsibility.
    """
    if dataset_key not in DATASET_KEYS:
        raise _failure(dataset_key, "unknown dataset; automatic grouping is available for the 13 evaluation datasets")
    root = _project_root(project_root)
    paths = {original: _relative_path(dataset_key, original) for original in audio_paths}
    if not paths:
        return {}

    identities: dict[str, str] = {}
    metadata_source = ""
    if dataset_key == "thai_ser":
        identities, metadata_source = _csv_identities(
            dataset_key, root, ("data/thai_ser/labels.csv",), "audio_id", "actor_id",
        )
        metadata_source += "; actor identity shared across microphones/sessions; session_id is not the group"
    elif dataset_key == "emozionalmente":
        identities, metadata_source = _csv_identities(
            dataset_key, root,
            ("data/emozionalmente/audio/emozionalmente/metadata/samples.csv",
             "data/emozionalmente/metadata/samples.csv"),
            "file_name", "actor",
        )
        metadata_source += "; anonymized actor ID, not numeric sample basename"

    result: dict[str, dict] = {}
    for original, path in paths.items():
        kind = "speaker"
        if dataset_key in {"thai_ser", "emozionalmente"}:
            if path.name not in identities:
                raise _failure(dataset_key, f"no actor metadata for {original!r} in {metadata_source}")
            identity, source = identities[path.name], metadata_source
        elif dataset_key in _FILENAME_RULES:
            pattern, source = _FILENAME_RULES[dataset_key]
            match = re.fullmatch(pattern, path.stem)
            if match is None:
                raise _failure(dataset_key, f"unrecognized speaker filename {original!r}")
            identity = match["speaker"]
        elif dataset_key == "ravdess":
            match = re.fullmatch(
                r"03-01-0[1-8]-0[12]-0[12]-0[12]-(?P<speaker>0[1-9]|1\d|2[0-4])", path.stem,
            )
            if match is None:
                raise _failure(dataset_key, f"unrecognized speech actor filename {original!r}")
            identity = match["speaker"]
            if path.parent.name.startswith("Actor_") and path.parent.name != f"Actor_{identity}":
                raise _failure(dataset_key, f"actor folder/filename conflict in {original!r}")
            source = "filename seventh field is actor 01-24; docs/datasets_survey_en.md section 6 (RAVDESS)"
        elif dataset_key == "tess":
            match = re.fullmatch(
                r"(?P<speaker>OAF|YAF)_[^_]+_(?:angry|disgust|fear|happy|neutral|ps|sad)",
                path.stem, flags=re.IGNORECASE,
            )
            if path.name.lower() == "oa_bite_neutral.wav":
                identity = "OAF"
                source = "known OA_bite_neutral.wav prefix typo is OAF; docs/datasets_survey_en.md section 6 (TESS)"
            elif match is not None:
                identity = match["speaker"].upper()
                source = "filename OAF/YAF actress prefix; docs/datasets_survey_en.md section 6 (TESS; two actresses)"
            else:
                raise _failure(dataset_key, f"unrecognized actress filename {original!r}")
        elif dataset_key == "mead":
            match = re.fullmatch(
                r"(?P<speaker>[MW]\d{3})/(?:angry|contempt|disgusted|fear|happy|neutral|sad|surprised)"
                r"/level_[123]/\d{3}\.wav", "/".join(path.parts[-4:]),
            )
            if match is None:
                raise _failure(dataset_key, f"missing <actor>/<emotion>/level_<n>/<number>.wav layout in {original!r}")
            identity = match["speaker"]
            source = "actor directory Mnnn/Wnnn, shared across emotions/intensities; common/eval_datasets.py:mead"
        elif dataset_key == "emovo":
            match = re.fullmatch(r"(?:rab|dis|pau|gio|neu|tri|sor)-(?P<speaker>[fm][1-3])-[bdln]\d", path.stem)
            if match is None:
                raise _failure(dataset_key, f"unrecognized actor filename {original!r}")
            identity = match["speaker"]
            if re.fullmatch(r"[fm][1-3]", path.parent.name) and path.parent.name != identity:
                raise _failure(dataset_key, f"actor folder/filename conflict in {original!r}")
            source = "filename <emotion>-<speaker>-<sentence>; common/eval_datasets.py:emovo (six actors)"
        elif dataset_key == "meld":
            match = re.fullmatch(r"dia(?P<dialogue>\d+)_utt\d+", path.stem)
            if match is None:
                raise _failure(dataset_key, f"unrecognized dialogue filename {original!r}")
            identity, kind = f"dia{int(match['dialogue'])}", "dialogue"
            source = (
                "filename dia<Dialogue_ID>_utt<Utterance_ID>; common/eval_datasets.py:meld and "
                "data/meld/test_sent_emo.csv; only the official test split is available locally; "
                "dialogue grouping does not guarantee speaker independence"
            )
        else:  # Exhaustive guard if DATASET_KEYS changes without a parser.
            raise _failure(dataset_key, "speaker grouping has not been implemented")
        result[original] = {"group": f"{dataset_key}:{kind}:{identity}", "group_kind": kind, "source": source}
    return result
