"""The visitor's own clips — uploaded files and microphone recordings.

Every per-clip feature (detection, silence probe, saliency, embedding view)
takes a file path, so a user clip only has to become a path. This module owns
that: it decodes whatever arrives, re-encodes it as 16 kHz mono PCM WAV (the
rate every detector resamples to anyway, and a format every browser can play
back), and files it under the caller's session.

SAFETY
------
- Ids are opaque (`up_` + 16 random hex chars) and never derived from the
  uploaded filename; the filename is kept only as display metadata.
- Ownership is structural: a clip lives in its session's own directory, so a
  lookup with another session's id simply misses. The session id itself is
  validated before it is ever joined into a path.
- Unknown or malformed ids are rejected by pattern before touching the disk.
- Clips expire after `UPLOAD_TTL_SECONDS` and each session holds at most
  `MAX_CLIPS_PER_SESSION`; both are enforced lazily on every write/list.

Unlike the ASVspoof subset these clips have no protocol file, so there is no
ground truth to leak — the model's score is the only opinion on record.

Storage layout: `{BACKEND_DIR}/uploads/deepfake_sessions/{sid}/{clip_id}.wav`
plus a `{clip_id}.json` sidecar with the display name and duration.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.core.session import InvalidSessionId, validate_session_id
from app.core.settings import BACKEND_DIR

from .service import TARGET_SAMPLE_RATE, _load_waveform

UPLOAD_ID_PREFIX = "up_"
_UPLOAD_ID_PATTERN = re.compile(r"^up_[0-9a-f]{16}$")

# libsndfile decodes these natively. The browser converts anything else
# (m4a, webm/opus from MediaRecorder) to WAV before upload.
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MIN_CLIP_SECONDS = 0.5
# Every detector analyses at most 30 s; longer clips are refused rather than
# stored and then silently truncated by every feature.
MAX_CLIP_SECONDS = 60.0
MAX_CLIPS_PER_SESSION = 20
UPLOAD_TTL_SECONDS = 24 * 60 * 60
MAX_DISPLAY_NAME = 120


class UploadRejected(ValueError):
    """The file is not usable audio, or breaks a size/duration limit."""


class UploadNotFound(LookupError):
    """Unknown, expired, malformed, or another session's clip id."""


@dataclass(frozen=True, slots=True)
class UploadInfo:
    # Same shape as dataset.RecordingInfo so the frontend can treat a user
    # clip exactly like a demo clip.
    recording_id: str
    display_filename: str
    extension: str
    size_bytes: int
    duration_seconds: float | None
    source: str  # "upload" | "recording"
    created_at: float


def is_upload_id(recording_id: str) -> bool:
    return recording_id.startswith(UPLOAD_ID_PREFIX)


def _storage_root() -> Path:
    return BACKEND_DIR / "uploads" / "deepfake_sessions"


def _session_dir(sid: str | None) -> Path:
    try:
        return _storage_root() / validate_session_id(sid)
    except InvalidSessionId as error:
        raise UploadNotFound("No valid session for user clips.") from error


def _clean_display_name(filename: str | None, source: str) -> str:
    name = Path(filename or "").name.strip()
    # Printable, no path separators; fall back to something readable.
    name = re.sub(r"[^\w .()\-]+", "_", name)[:MAX_DISPLAY_NAME].strip()
    if not name:
        stamp = time.strftime("%H-%M-%S")
        name = f"recording-{stamp}.wav" if source == "recording" else f"clip-{stamp}.wav"
    return name


def _read_meta(meta_path: Path) -> UploadInfo | None:
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return UploadInfo(**data)
    except Exception:
        return None


def _delete_clip(directory: Path, clip_id: str) -> None:
    for suffix in (".wav", ".json"):
        (directory / f"{clip_id}{suffix}").unlink(missing_ok=True)


def _prune(directory: Path, now: float | None = None) -> list[UploadInfo]:
    """Drop expired or broken clips; return the survivors, newest first."""
    now = time.time() if now is None else now
    alive: list[UploadInfo] = []
    if not directory.is_dir():
        return alive
    for meta_path in directory.glob("up_*.json"):
        clip_id = meta_path.stem
        info = _read_meta(meta_path)
        audio = directory / f"{clip_id}.wav"
        if info is None or not audio.is_file() or now - info.created_at > UPLOAD_TTL_SECONDS:
            _delete_clip(directory, clip_id)
            continue
        alive.append(info)
    # Orphaned audio with no sidecar (a crash mid-write).
    for audio in directory.glob("up_*.wav"):
        if not (directory / f"{audio.stem}.json").exists():
            audio.unlink(missing_ok=True)
    alive.sort(key=lambda info: info.created_at, reverse=True)
    return alive


def save_upload(sid: str | None, raw_path: Path, filename: str | None, source: str) -> UploadInfo:
    """Decode `raw_path`, store it as 16 kHz mono WAV, return its metadata.

    Blocking (decode + resample + write): call from a threadpool.
    """
    import soundfile as sf

    source = "recording" if source == "recording" else "upload"
    directory = _session_dir(sid)

    try:
        waveform, sample_rate = _load_waveform(raw_path)
    except ValueError as error:
        raise UploadRejected(
            "Could not read this file as audio. Try WAV, FLAC, OGG or MP3."
        ) from error

    duration = waveform.shape[1] / float(sample_rate)
    if duration < MIN_CLIP_SECONDS:
        raise UploadRejected(f"The clip is too short ({duration:.2f}s); record at least {MIN_CLIP_SECONDS:g}s.")
    if duration > MAX_CLIP_SECONDS:
        raise UploadRejected(
            f"The clip is {duration:.0f}s long; the limit is {MAX_CLIP_SECONDS:.0f}s. Trim it and try again."
        )
    if float(waveform.abs().max()) < 1e-4:
        raise UploadRejected("The clip is silent — check the microphone and try again.")

    directory.mkdir(parents=True, exist_ok=True)
    existing = _prune(directory)
    # Make room by evicting the oldest clips rather than refusing the new one.
    for stale in existing[MAX_CLIPS_PER_SESSION - 1 :]:
        _delete_clip(directory, stale.recording_id)

    clip_id = f"{UPLOAD_ID_PREFIX}{secrets.token_hex(8)}"
    audio_path = directory / f"{clip_id}.wav"
    sf.write(audio_path, waveform.squeeze(0).numpy(), TARGET_SAMPLE_RATE, subtype="PCM_16")

    info = UploadInfo(
        recording_id=clip_id,
        display_filename=_clean_display_name(filename, source),
        extension=".wav",
        size_bytes=audio_path.stat().st_size,
        duration_seconds=round(duration, 2),
        source=source,
        created_at=time.time(),
    )
    # Sidecar last: a clip only "exists" once its metadata does.
    (directory / f"{clip_id}.json").write_text(json.dumps(asdict(info)), encoding="utf-8")
    return info


def list_uploads(sid: str | None) -> list[UploadInfo]:
    try:
        directory = _session_dir(sid)
    except UploadNotFound:
        return []
    return _prune(directory)


def _checked(sid: str | None, clip_id: str) -> tuple[Path, str]:
    if not _UPLOAD_ID_PATTERN.fullmatch(clip_id or ""):
        raise UploadNotFound(f"Unknown clip id: {clip_id}")
    return _session_dir(sid), clip_id


def get_upload(sid: str | None, clip_id: str) -> UploadInfo:
    directory, clip_id = _checked(sid, clip_id)
    info = _read_meta(directory / f"{clip_id}.json")
    if info is None or not (directory / f"{clip_id}.wav").is_file():
        raise UploadNotFound(f"Unknown clip id: {clip_id}")
    if time.time() - info.created_at > UPLOAD_TTL_SECONDS:
        _delete_clip(directory, clip_id)
        raise UploadNotFound(f"Clip {clip_id} has expired.")
    return info


def resolve_upload_path(sid: str | None, clip_id: str) -> Path:
    get_upload(sid, clip_id)
    directory, clip_id = _checked(sid, clip_id)
    return directory / f"{clip_id}.wav"


def delete_upload(sid: str | None, clip_id: str) -> None:
    get_upload(sid, clip_id)
    directory, clip_id = _checked(sid, clip_id)
    _delete_clip(directory, clip_id)
