"""A reusable audio audit: each file is read once, and its SHA256 and header are reused across runs.

The cache is JSON Lines, one entry per resolved path: size, mtime_ns, sha256, frames, sample_rate, duration and
the decoding error, if any. An entry is reused while the file's size and mtime_ns are unchanged, and recomputed
otherwise. Leakage checks, resume digests and duration checks all read their content digests and headers here, so
they stay byte-content based without reading millions of files again.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path

import soundfile as sf

THREADS = 16         # reading is I/O bound (network storage especially); hashlib and file reads release the GIL
CHUNK = 10000        # entries per append: an interrupted first audit keeps what it has already hashed


def audio_header(source) -> dict:
    """Frames, sample rate and duration from a path or file object, or the decoding error."""
    try:
        info = sf.info(source)
    except (sf.LibsndfileError, OSError) as exc:
        return {"frames": None, "sample_rate": None, "duration": None, "error": getattr(exc, "error_string", str(exc))}
    return {"frames": info.frames, "sample_rate": info.samplerate, "duration": info.frames / info.samplerate,
            "error": None}


def _measure(path: str, cached: dict | None) -> tuple[dict, bool]:
    """Reuse the cached entry if size and mtime_ns match; otherwise read the file once for both hash and header."""
    stat = os.stat(path)
    if cached and cached["size"] == stat.st_size and cached["mtime_ns"] == stat.st_mtime_ns:
        return cached, False
    data = Path(path).read_bytes()
    return {"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(data).hexdigest(), **audio_header(io.BytesIO(data))}, True


class AudioAudit:
    """path=None keeps the audit in memory only (one read per file within this process)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._cached, self._verified = {}, {}  # from the file / checked against the disk in this process
        self.hashed = self.reused = self.hashed_bytes = self.ignored_lines = 0
        if self.path and self.path.is_file():
            with self.path.open("rb") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                        self._cached[entry["path"]] = entry
                    except (ValueError, KeyError, TypeError):
                        self.ignored_lines += 1  # e.g. an append cut short by a crash; that file is simply re-read

    def entries(self, paths, progress=None) -> list[dict]:
        """Entries in the order of paths. Files not yet checked in this process are stat-ed, and read only when new
        or changed, in a thread pool; new entries are appended to the cache in chunks."""
        paths = [str(path) for path in paths]
        pending = [path for path in dict.fromkeys(paths) if path not in self._verified]
        if pending:
            with ThreadPoolExecutor(THREADS) as pool:
                def measured():
                    for start in range(0, len(pending), CHUNK):
                        chunk = pending[start:start + CHUNK]
                        yield from pool.map(_measure, chunk, [self._cached.pop(path, None) for path in chunk])
                fresh = []
                results = progress(measured(), "Auditing audio files", total=len(pending)) if progress else measured()
                for entry, is_new in results:
                    self._verified[entry["path"]] = entry
                    if is_new:
                        self.hashed += 1
                        self.hashed_bytes += entry["size"]
                        fresh.append(entry)
                        if len(fresh) == CHUNK:
                            self._append(fresh)
                            fresh = []
                    else:
                        self.reused += 1
                self._append(fresh)
        return [self._verified[path] for path in paths]

    def digests(self, paths) -> list[str]:
        return [entry["sha256"] for entry in self.entries(paths)]

    def _append(self, entries):
        if not entries or not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab+") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell():
                handle.seek(-1, os.SEEK_END)
                newline = b"" if handle.read(1) == b"\n" else b"\n"  # never continue a line cut short
            else:
                newline = b""
            handle.write(newline + "".join(json.dumps(entry) + "\n" for entry in entries).encode())
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self) -> str:
        where = self.path or "memory only"
        ignored = f"; {self.ignored_lines} unreadable cache lines ignored" if self.ignored_lines else ""
        return (f"Audio audit ({where}): {self.hashed} files read and hashed ({self.hashed_bytes / 1e9:.2f} GB), "
                f"{self.reused} reused unchanged{ignored}")
