"""Read-only discovery and listing for the deepfake task's built-in datasets.

Three labelled 200-clip subsets, each behind opaque, stable recording ids
(same pattern as the verification and diarization tasks):

    asvspoof2019-la   ASVspoof 2019 LA eval (in-domain for most detectors)
    asvspoof5         ASVspoof 5 eval (newer attacks, codecs, crowdsourced speech)
    in-the-wild       In-the-Wild (Müller et al. 2022; real-world generalisation)

GROUND-TRUTH SAFETY: an ASVspoof file id (`LA_E_2834763`) encodes nothing, so
it is safe to display -- but the protocol's `key` (bonafide/spoof) and
`system_id` (the attack, A07..A19) are exactly what this task asks the user to
judge. They are parsed only by `load_ground_truth()`, which is reserved for
offline evaluation, and must never reach a runtime response.

Expected on-disk layout, one directory per dataset under DEEPFAKE_DATASET_ROOT
(built by scripts/prepare_asvspoof_la_subset.py and
scripts/prepare_deepfake_eval_subsets.py):

    Backend/data/deepfake/<directory>/
        <audio_subdir>/*.flac|*.wav
        protocol.txt

Every protocol.txt is written in the ASVspoof 2019 CM format, whatever the
source corpus used, so `load_ground_truth` reads all three the same way.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import settings

EXPECTED_RECORDING_COUNT = 200
SUPPORTED_AUDIO_EXTENSIONS = {".flac", ".wav"}
PROTOCOL_FILENAME = "protocol.txt"
AUDIO_MEDIA_TYPES = {".flac": "audio/flac", ".wav": "audio/wav"}


@dataclass(frozen=True, slots=True)
class BuiltinDataset:
    dataset_id: str
    label: str
    directory: str
    audio_subdir: str
    audio_extension: str
    description: str
    citation: str
    license: str


BUILTIN_DATASETS: dict[str, BuiltinDataset] = {
    dataset.dataset_id: dataset
    for dataset in (
        BuiltinDataset(
            dataset_id="asvspoof2019-la",
            label="ASVspoof 2019 LA (subset)",
            directory="asvspoof2019_la",
            audio_subdir="flac",
            audio_extension=".flac",
            description="Studio-quality VCTK speech against 13 TTS and voice-conversion attacks. "
            "Models B-F were trained on its training partition.",
            citation="Wang et al., Computer Speech & Language 2020",
            license="ODC-By 1.0",
        ),
        BuiltinDataset(
            dataset_id="asvspoof5",
            label="ASVspoof 5 (subset)",
            directory="asvspoof5",
            audio_subdir="flac",
            audio_extension=".flac",
            description="Crowdsourced audiobook speech against 16 newer attacks, some with "
            "adversarial or codec processing. Not in the training data of Models B-F.",
            citation="Wang et al., ASVspoof 5, Computer Speech & Language 2025",
            license="ODC-By 1.0",
        ),
        BuiltinDataset(
            dataset_id="in-the-wild",
            label="In-the-Wild (subset)",
            directory="in_the_wild",
            audio_subdir="wav",
            audio_extension=".wav",
            description="Public-figure speech and deepfakes collected from the internet. "
            "Tests real-world generalisation; not in the training data of Models B-F.",
            citation="Müller et al., Interspeech 2022",
            license="see attribution.txt of the release",
        ),
    )
}
DEFAULT_DATASET_ID = "asvspoof2019-la"
# Kept for callers that only know the original single dataset.
DATASET_ID = DEFAULT_DATASET_ID


class DatasetUnavailable(RuntimeError):
    """Raised when the demo dataset directory is missing or unreadable."""


class UnknownDataset(ValueError):
    """Raised when a dataset id is not one of BUILTIN_DATASETS."""


class RecordingNotFound(ValueError):
    """Raised when a recording id does not match a known recording."""


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    recording_id: str
    display_filename: str
    extension: str
    size_bytes: int
    # None when the header could not be read. The shared Audio Dataset table
    # renders this column, and without it every row reads "0.00s".
    duration_seconds: float | None = None


def get_builtin_dataset(dataset_id: str | None) -> BuiltinDataset:
    try:
        return BUILTIN_DATASETS[dataset_id or DEFAULT_DATASET_ID]
    except KeyError as error:
        raise UnknownDataset(f"Unknown dataset: {dataset_id}") from error


def _dataset_root(dataset_id: str | None = None) -> Path:
    return settings.DEEPFAKE_DATASET_ROOT / get_builtin_dataset(dataset_id).directory


def _audio_dir(dataset_id: str | None = None) -> Path:
    return _dataset_root(dataset_id) / get_builtin_dataset(dataset_id).audio_subdir


def _recording_id_for(filename: str, dataset_id: str | None = None) -> str:
    # The original dataset hashes the bare filename, so its ids (and every
    # client-side reference to them) are unchanged. The others are namespaced:
    # In-the-Wild's "0.wav" must never collide with anything else.
    dataset_id = dataset_id or DEFAULT_DATASET_ID
    key = filename if dataset_id == DEFAULT_DATASET_ID else f"{dataset_id}/{filename}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"rec_{digest[:16]}"


def _iter_audio_files(audio_dir: Path, dataset_id: str | None = None):
    if not audio_dir.is_dir():
        raise DatasetUnavailable(f"Deepfake demo dataset not found: {dataset_id or DEFAULT_DATASET_ID}")

    for entry in audio_dir.iterdir():
        # Top level only: protocol.txt sits in the parent directory and is
        # offline-eval ground truth, so it is never reachable from here.
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        yield entry


def _read_duration(path: Path) -> float | None:
    """Clip length from the file header only -- no decoding, no audio read."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return round(info.frames / float(info.samplerate), 2)
    except Exception:
        # A listing must not fail because one file is unreadable.
        return None


def _info_for(path: Path, recording_id: str) -> RecordingInfo:
    return RecordingInfo(
        recording_id=recording_id,
        display_filename=path.name,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        duration_seconds=_read_duration(path),
    )


def _discover(dataset_id: str | None = None) -> list[tuple[RecordingInfo, Path]]:
    recordings = [
        (_info_for(entry, _recording_id_for(entry.name, dataset_id)), entry)
        for entry in _iter_audio_files(_audio_dir(dataset_id), dataset_id)
    ]
    recordings.sort(key=lambda pair: pair[0].display_filename)
    return recordings


def get_dataset_info(dataset_id: str | None = None) -> dict[str, object]:
    """Summarize one built-in dataset without raising when it is absent from
    disk. An id outside BUILTIN_DATASETS still raises `UnknownDataset`."""

    dataset = get_builtin_dataset(dataset_id)
    try:
        total = sum(1 for _ in _iter_audio_files(_audio_dir(dataset.dataset_id), dataset.dataset_id))
        available = True
    except DatasetUnavailable:
        total, available = 0, False

    return {
        "dataset_id": dataset.dataset_id,
        "expected_recording_count": EXPECTED_RECORDING_COUNT,
        "total_recordings": total,
        "audio_extensions": [dataset.audio_extension],
        "available": available,
    }


def list_builtin_datasets() -> list[dict[str, object]]:
    """Every built-in dataset, present on disk or not (the menu greys out
    the missing ones). Counts only -- no labels."""

    return [
        {
            **get_dataset_info(dataset.dataset_id),
            "label": dataset.label,
            "description": dataset.description,
            "citation": dataset.citation,
            "license": dataset.license,
        }
        for dataset in BUILTIN_DATASETS.values()
    ]


def list_recordings(dataset_id: str | None = None) -> list[RecordingInfo]:
    """List every recording in one built-in subset. Raises if unavailable.

    Carries no label: see the module docstring.
    """

    return [info for info, _path in _discover(dataset_id)]


def list_recordings_with_paths(dataset_id: str | None = None) -> list[tuple[RecordingInfo, Path]]:
    """`list_recordings` plus each clip's path, for scoring a whole subset
    without resolving every id again. Internal use only."""

    return _discover(dataset_id)


def get_recording(recording_id: str) -> RecordingInfo:
    """Look up a single recording by its opaque id, in any built-in dataset."""

    return _info_for(_locate(recording_id), recording_id)


def _locate(recording_id: str) -> Path:
    found_any = False
    for dataset_id in BUILTIN_DATASETS:
        try:
            for entry in _iter_audio_files(_audio_dir(dataset_id), dataset_id):
                found_any = True
                if _recording_id_for(entry.name, dataset_id) == recording_id:
                    return entry
        except DatasetUnavailable:
            continue
    if not found_any:
        raise DatasetUnavailable("No deepfake demo dataset is installed.")
    raise RecordingNotFound(f"Unknown recording id: {recording_id}")


def resolve_recording_path(recording_id: str) -> Path:
    """Resolve a safe recording id to its on-disk path, whichever built-in
    dataset holds it. Internal use only -- unknown or path-traversal ids
    simply miss and raise `RecordingNotFound` without touching the
    filesystem with untrusted input."""

    return _locate(recording_id)


def audio_media_type(path: Path) -> str:
    return AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def load_ground_truth(dataset_id: str | None = None) -> dict[str, tuple[str, str]]:
    """`{file_id: (system_id, key)}` parsed from protocol.txt.

    OFFLINE EVALUATION ONLY (Feature 1's score distribution / DET / EER). The
    `key` is the bona fide/spoof answer the workbench exists to let a user
    judge for themselves, so this must never be called from a route handler.

    Protocol lines are the ASVspoof 2019 CM format:
        SPEAKER_ID  FILE_ID  -  SYSTEM_ID  KEY
    """

    protocol_path = _dataset_root(dataset_id) / PROTOCOL_FILENAME
    if not protocol_path.is_file():
        raise DatasetUnavailable(
            f"Ground-truth protocol not found for {dataset_id or DEFAULT_DATASET_ID}: {protocol_path}"
        )

    truth: dict[str, tuple[str, str]] = {}
    with open(protocol_path, "r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 5:
                continue
            _speaker_id, file_id, _unused, system_id, key = fields[:5]
            truth[file_id] = (system_id, key)
    return truth
