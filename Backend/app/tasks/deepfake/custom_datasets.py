"""Researcher-built datasets — named collections of the visitor's own audio.

The other tasks offer a "Manage Datasets" dialog backed by the shared
`/upload/dataset/*` routes. Those store files where the deepfake detectors
cannot resolve them, so this task keeps its own, with the same three actions
(create, upload files, delete) plus one the others do not need: an optional
LABEL FILE, so a researcher can measure EER and the DET curve on their own
corpus rather than only on the ASVspoof subset.

GROUND TRUTH STAYS AGGREGATE
----------------------------
Labels are read by `evaluation.py` only. No listing or per-clip response ever
returns one — exactly the rule `dataset.py` sets for the ASVspoof protocol —
so a custom dataset can be judged blind like the built-in one. The listing
reports only how many clips a label file matched.

SAFETY (as uploads.py)
----------------------
- Clip ids are opaque (`cd_` + 16 random hex chars), never derived from the
  uploaded filename; the filename is display metadata only.
- Ownership is structural: everything lives under the caller's validated
  session directory, so another session's id or dataset name simply misses.
- Dataset names are validated by pattern before they are ever joined into a
  path.
- Datasets expire `DATASET_TTL_SECONDS` after their last change.

Layout: `{BACKEND_DIR}/uploads/deepfake_datasets/{sid}/{name}/`
    dataset.json            name, created/updated timestamps
    labels.json             {filename stem: [attack, "bonafide"|"spoof"]}   (optional)
    cd_<hex>.wav / .json    16 kHz mono PCM clip + display sidecar
"""

from __future__ import annotations

import csv
import io
import json
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.session import InvalidSessionId, validate_session_id
from app.core.settings import BACKEND_DIR

from .service import TARGET_SAMPLE_RATE, _load_waveform
from .uploads import MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, UploadRejected, _clean_display_name

CLIP_ID_PREFIX = "cd_"
_CLIP_ID_PATTERN = re.compile(r"^cd_[0-9a-f]{16}$")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-]{0,47}$")

MAX_DATASETS_PER_SESSION = 10
MAX_CLIPS_PER_DATASET = 300
MAX_LABEL_FILE_BYTES = 2 * 1024 * 1024
DATASET_TTL_SECONDS = 7 * 24 * 60 * 60

BONAFIDE_WORDS = {"bonafide", "bona-fide", "bona_fide", "genuine", "real", "human", "0"}
SPOOF_WORDS = {"spoof", "fake", "synthetic", "deepfake", "1"}


class DatasetError(ValueError):
    """A request that breaks a rule (bad name, limit, unreadable file)."""


class DatasetNotFound(LookupError):
    """Unknown, expired, or another session's dataset or clip."""


@dataclass(frozen=True, slots=True)
class ClipInfo:
    # Same shape as dataset.RecordingInfo so the frontend treats it the same.
    recording_id: str
    display_filename: str
    extension: str
    size_bytes: int
    duration_seconds: float | None
    created_at: float


def is_custom_clip_id(recording_id: str) -> bool:
    return recording_id.startswith(CLIP_ID_PREFIX)


def _root() -> Path:
    return BACKEND_DIR / "uploads" / "deepfake_datasets"


def _session_dir(sid: str | None) -> Path:
    try:
        return _root() / validate_session_id(sid)
    except InvalidSessionId as error:
        raise DatasetNotFound("No valid session for custom datasets.") from error


def validate_name(name: str | None) -> str:
    cleaned = (name or "").strip()
    if not _NAME_PATTERN.fullmatch(cleaned):
        raise DatasetError(
            "Dataset names are 1-48 characters: letters, digits, spaces, '-' and '_', "
            "starting with a letter or digit."
        )
    return cleaned


def _dataset_dir(sid: str | None, name: str, must_exist: bool = True) -> Path:
    directory = _session_dir(sid) / validate_name(name)
    if must_exist:
        meta = _read_json(directory / "dataset.json")
        if meta is None:
            raise DatasetNotFound(f"No dataset named '{name}'.")
        if time.time() - meta.get("updated_at", 0) > DATASET_TTL_SECONDS:
            shutil.rmtree(directory, ignore_errors=True)
            raise DatasetNotFound(f"Dataset '{name}' has expired.")
    return directory


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _touch(directory: Path) -> None:
    meta = _read_json(directory / "dataset.json") or {}
    meta["updated_at"] = time.time()
    (directory / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")


def _clips(directory: Path) -> list[ClipInfo]:
    clips = []
    for meta_path in directory.glob("cd_*.json"):
        data = _read_json(meta_path)
        if data is None or not (directory / f"{meta_path.stem}.wav").is_file():
            continue
        try:
            clips.append(ClipInfo(**data))
        except TypeError:
            continue
    clips.sort(key=lambda clip: (clip.display_filename.lower(), clip.created_at))
    return clips


def _labels(directory: Path) -> dict[str, list[str]]:
    data = _read_json(directory / "labels.json")
    return data if isinstance(data, dict) else {}


def _stem(filename: str) -> str:
    return Path(filename).stem.lower()


def _summary(directory: Path) -> dict:
    meta = _read_json(directory / "dataset.json") or {}
    clips = _clips(directory)
    labels = _labels(directory)
    matched = [labels[_stem(clip.display_filename)] for clip in clips if _stem(clip.display_filename) in labels]
    return {
        "dataset_name": meta.get("name", directory.name),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
        "total_files": len(clips),
        "total_duration_seconds": round(sum(clip.duration_seconds or 0.0 for clip in clips), 2),
        # Counts only: which clip carries which label is never returned.
        "labels": {
            "provided": bool(labels),
            "matched_files": len(matched),
            "bonafide": sum(1 for _, label in matched if label == "bonafide"),
            "spoof": sum(1 for _, label in matched if label == "spoof"),
        },
    }


def _prune(sid: str | None) -> list[Path]:
    try:
        root = _session_dir(sid)
    except DatasetNotFound:
        return []
    alive = []
    if not root.is_dir():
        return alive
    now = time.time()
    for directory in root.iterdir():
        meta = _read_json(directory / "dataset.json") if directory.is_dir() else None
        if meta is None or now - meta.get("updated_at", 0) > DATASET_TTL_SECONDS:
            shutil.rmtree(directory, ignore_errors=True)
            continue
        alive.append(directory)
    alive.sort(key=lambda path: (_read_json(path / "dataset.json") or {}).get("created_at", 0))
    return alive


def list_datasets(sid: str | None) -> list[dict]:
    return [_summary(directory) for directory in _prune(sid)]


def create_dataset(sid: str | None, name: str) -> dict:
    name = validate_name(name)
    existing = _prune(sid)
    if any(path.name.lower() == name.lower() for path in existing):
        raise DatasetError(f"A dataset named '{name}' already exists.")
    if len(existing) >= MAX_DATASETS_PER_SESSION:
        raise DatasetError(f"At most {MAX_DATASETS_PER_SESSION} datasets per session; delete one first.")
    directory = _dataset_dir(sid, name, must_exist=False)
    directory.mkdir(parents=True, exist_ok=False)
    now = time.time()
    (directory / "dataset.json").write_text(
        json.dumps({"name": name, "created_at": now, "updated_at": now}), encoding="utf-8"
    )
    return _summary(directory)


def delete_dataset(sid: str | None, name: str) -> None:
    shutil.rmtree(_dataset_dir(sid, name), ignore_errors=True)


def add_clip(sid: str | None, name: str, raw_path: Path, filename: str | None) -> ClipInfo:
    """Decode, validate and store one file as 16 kHz mono WAV. Blocking."""
    import soundfile as sf

    directory = _dataset_dir(sid, name)
    if len(_clips(directory)) >= MAX_CLIPS_PER_DATASET:
        raise DatasetError(f"A dataset holds at most {MAX_CLIPS_PER_DATASET} files.")
    try:
        waveform, sample_rate = _load_waveform(raw_path)
    except ValueError as error:
        raise UploadRejected("Could not read this file as audio. Try WAV, FLAC, OGG or MP3.") from error

    duration = waveform.shape[1] / float(sample_rate)
    if duration < MIN_CLIP_SECONDS:
        raise UploadRejected(f"Too short ({duration:.2f}s); at least {MIN_CLIP_SECONDS:g}s is needed.")
    if duration > MAX_CLIP_SECONDS:
        raise UploadRejected(f"{duration:.0f}s long; the limit is {MAX_CLIP_SECONDS:.0f}s.")
    if float(waveform.abs().max()) < 1e-4:
        raise UploadRejected("The file is silent.")

    clip_id = f"{CLIP_ID_PREFIX}{secrets.token_hex(8)}"
    audio_path = directory / f"{clip_id}.wav"
    sf.write(audio_path, waveform.squeeze(0).numpy(), TARGET_SAMPLE_RATE, subtype="PCM_16")
    info = ClipInfo(
        recording_id=clip_id,
        display_filename=_clean_display_name(filename, "upload"),
        extension=".wav",
        size_bytes=audio_path.stat().st_size,
        duration_seconds=round(duration, 2),
        created_at=time.time(),
    )
    (directory / f"{clip_id}.json").write_text(json.dumps(asdict(info)), encoding="utf-8")
    _touch(directory)
    return info


def list_clips(sid: str | None, name: str) -> list[ClipInfo]:
    return _clips(_dataset_dir(sid, name))


def resolve_clip_path(sid: str | None, clip_id: str) -> Path:
    """A `cd_` id from any of the caller's datasets -> its WAV path."""
    if not _CLIP_ID_PATTERN.fullmatch(clip_id or ""):
        raise DatasetNotFound(f"Unknown clip id: {clip_id}")
    for directory in _prune(sid):
        path = directory / f"{clip_id}.wav"
        if path.is_file():
            return path
    raise DatasetNotFound(f"Unknown clip id: {clip_id}")


def _normalise_label(raw: str) -> str | None:
    word = raw.strip().lower()
    if word in BONAFIDE_WORDS:
        return "bonafide"
    if word in SPOOF_WORDS:
        return "spoof"
    return None


def parse_label_file(text: str) -> dict[str, list[str]]:
    """Two formats, detected per line:

    ASVspoof protocol  `SPEAKER FILE_ID - ATTACK bonafide|spoof` (whitespace)
    CSV                `filename,label[,attack]` with an optional header row

    Keys are lower-cased filename stems, so `clip01.wav`, `clip01.flac` and
    `clip01` all match the uploaded `clip01.mp3`.
    """
    labels: dict[str, list[str]] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
            continue
        if len(row) == 1:
            parts = row[0].split()
            if len(parts) >= 5:  # ASVspoof protocol line
                filename, attack, label = parts[1], parts[3], parts[4]
            elif len(parts) >= 2:
                filename, label, attack = parts[0], parts[1], parts[2] if len(parts) > 2 else "-"
            else:
                continue
        else:
            filename, label = row[0], row[1]
            attack = row[2].strip() if len(row) > 2 and row[2].strip() else "-"
        normalised = _normalise_label(label)
        if normalised is None:
            continue  # unreadable line; a header row lands here too
        if normalised == "spoof" and attack in {"", "-"}:
            attack = "spoof"
        labels[_stem(filename.strip())] = [attack if normalised == "spoof" else "-", normalised]
    if not labels:
        raise DatasetError(
            "No labels could be read. Use 'filename,label' CSV lines (label: bonafide/spoof) "
            "or ASVspoof protocol lines."
        )
    return labels


def set_labels(sid: str | None, name: str, text: str) -> dict:
    directory = _dataset_dir(sid, name)
    labels = parse_label_file(text)
    (directory / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    _touch(directory)
    return _summary(directory)


def labelled_clips(sid: str | None, name: str) -> tuple[list[tuple[str, Path]], dict[str, tuple[str, str]]]:
    """For evaluation.py only: (stem, path) per clip and the stem -> (attack, label) map."""
    directory = _dataset_dir(sid, name)
    labels = _labels(directory)
    clips = [(_stem(clip.display_filename), directory / f"{clip.recording_id}.wav") for clip in _clips(directory)]
    return clips, {stem: (entry[0], entry[1]) for stem, entry in labels.items()}
