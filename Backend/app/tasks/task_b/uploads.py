"""Session-scoped audio uploads for Speaker Diarization.

Lets a user diarize their own recording instead of only the three read-only
AMI meetings in `dataset.py`. Uploads are deliberately thinner than the
Speaker Verification task's `session_assets.py`: there is no Redis registry
and no sweep loop here, because the upload id **is** the filename stem, so
the filesystem is the only source of truth it needs. Diarization results
themselves are still cached in Redis by the *content* hash of the file, which
is what makes re-running an upload instant -- see `router._get_or_compute`.

Storage layout: `{BACKEND_DIR}/uploads/diarization_sessions/{sid}/{upload_id}{ext}`,
structurally separate from the flat legacy `uploads/` directory, from Speaker
Verification's `uploads/verification_sessions/`, and from the permanent demo
dataset directory -- an upload can never be mistaken for an AMI recording.

Trade-off accepted for Phase 1: nothing ever deletes these files. A session
that goes away leaves its directory behind.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

from app.core.settings import BACKEND_DIR

from .dataset import RecordingInfo

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
MAX_FILE_BYTES = 50 * 1024 * 1024
UPLOAD_ID_PREFIX = "upl_"
# Perturbed clips (Phase 2) share this directory but are a different kind of
# thing: derived, deterministic, and never listed as "your uploads".
PERTURBED_ID_PREFIX = "prt_"

_SID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TEMP_PREFIX = ".tmp-"


class InvalidSessionId(Exception):
    """Raised when a session id fails the exact-format check."""


class UploadNotFound(ValueError):
    """Raised when an upload id is unknown or not owned by the caller's session."""


class PerturbedNotFound(ValueError):
    """Raised when a perturbed-clip id is unknown or not owned by the caller's session."""


def _storage_root() -> Path:
    return BACKEND_DIR / "uploads" / "diarization_sessions"


def validate_and_canonicalize_sid(sid: str | None) -> str:
    """Full-match only against the exact 32-lowercase-hex-char shape
    `ensure_session()` mints (`uuid.uuid4().hex`) -- a value that doesn't
    match exactly is rejected outright, never re-cased or "fixed up", so two
    differently-formatted strings are never treated as the same session."""

    if not sid or not _SID_PATTERN.fullmatch(sid):
        raise InvalidSessionId("Session id is missing or malformed.")
    return sid


def _session_dir(validated_sid: str) -> Path:
    return _storage_root() / validated_sid


def _iter_upload_files(validated_sid: str):
    """Top level of one session's directory only; partially-written temp
    files are never visible. Yields nothing when the session has never
    uploaded anything -- an empty list is a legitimate state, not an error."""

    session_dir = _session_dir(validated_sid)
    if not session_dir.is_dir():
        return

    for entry in session_dir.iterdir():
        if not entry.is_file() or entry.name.startswith(_TEMP_PREFIX):
            continue
        if entry.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            continue
        yield entry


def begin_upload_write(sid: str, extension: str) -> Path:
    """Validates sid, ensures the session's storage directory exists, and
    returns a fresh temp path *inside that same directory* (so promotion is a
    same-filesystem atomic rename, not a cross-filesystem copy). Ownership:
    this only allocates the path -- the caller must delete it if anything
    fails before `promote_upload()` is reached.

    Sync and blocking (mkdir); callers thread-pool it."""

    validated_sid = validate_and_canonicalize_sid(sid)
    session_dir = _session_dir(validated_sid)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir / f"{_TEMP_PREFIX}{secrets.token_hex(8)}{extension}"


def promote_upload(temp_path: Path, sid: str, extension: str) -> RecordingInfo:
    """Rename the temp file into its final `{upload_id}{ext}` name and
    describe it. Every returned field is derived from the file on disk, never
    from a caller-supplied value -- in particular the client's original
    filename is never used on disk and never echoed back, matching
    `dataset._recording_id_for`'s opaque-id convention.

    Returns the same `RecordingInfo` the demo dataset returns, so an upload
    and an AMI recording are indistinguishable to the frontend apart from
    their id prefix. Sync and blocking; callers thread-pool it."""

    validated_sid = validate_and_canonicalize_sid(sid)
    upload_id = f"{UPLOAD_ID_PREFIX}{secrets.token_hex(16)}"
    final_path = _session_dir(validated_sid) / f"{upload_id}{extension}"
    temp_path.replace(final_path)
    return RecordingInfo(
        recording_id=upload_id,
        display_filename=final_path.name,
        extension=extension,
        size_bytes=final_path.stat().st_size,
    )


def list_uploads(sid: str) -> list[RecordingInfo]:
    """This session's uploads, newest first, so a page refresh restores the
    list with the most recently uploaded recording at the top.

    Only `upl_` entries: perturbed clips live in the same directory but are
    derived artefacts, not things the user uploaded, and listing them here
    would let a perturbation masquerade as a source recording.
    """

    validated_sid = validate_and_canonicalize_sid(sid)
    entries = sorted(
        (
            entry
            for entry in _iter_upload_files(validated_sid)
            if entry.stem.startswith(UPLOAD_ID_PREFIX)
        ),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    return [
        RecordingInfo(
            recording_id=entry.stem,
            display_filename=entry.name,
            extension=entry.suffix.lower(),
            size_bytes=entry.stat().st_size,
        )
        for entry in entries
    ]


def resolve_upload_path(upload_id: str, sid: str) -> Path:
    """Resolve an upload id to its on-disk path. The untrusted `upload_id` is
    never joined into a path -- it is compared against the stems of files
    discovered by listing, exactly like `dataset.resolve_recording_path`, so
    path traversal is structurally impossible. Only the caller's own session
    directory is scanned, so one session can never resolve another's file."""

    validated_sid = validate_and_canonicalize_sid(sid)
    for entry in _iter_upload_files(validated_sid):
        if entry.stem == upload_id:
            return entry
    raise UploadNotFound(f"Unknown upload id: {upload_id}")


def resolve_perturbed_path(perturbed_id: str, sid: str) -> Path:
    """Resolve a perturbed-clip id to its on-disk path, with exactly the same
    guarantees as `resolve_upload_path`: the untrusted id is compared against
    stems discovered by listing and never joined into a path, and only the
    caller's own session directory is scanned."""

    validated_sid = validate_and_canonicalize_sid(sid)
    for entry in _iter_upload_files(validated_sid):
        if entry.stem == perturbed_id:
            return entry
    raise PerturbedNotFound(f"Unknown perturbed clip id: {perturbed_id}")
