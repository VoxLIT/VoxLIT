"""Session-scoped uploads for Speaker Diarization (task_b `uploads.py` + the
`/uploads` routes).

Storage is redirected to `tmp_path` by monkeypatching `uploads._storage_root`
— the pattern from `test_speaker_verification_session_assets.py` — so the real
`Backend/uploads/diarization_sessions/` is never written to. No model loads:
`run_diarization` is patched with a counting fake, which is what proves the
Phase 1 exit criterion (a re-run of the same upload is an instant cache hit).
"""

import io
import os
import secrets
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient

from app.main import app
from app.tasks.task_b import uploads

BASE = "/tasks/task-b"
MODEL = "pyannote-3.1"


def _wav_bytes(freq: float = 220.0, seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


def _valid_sid() -> str:
    return secrets.token_hex(16)  # 32 lowercase hex chars, matches ensure_session()


@pytest.fixture
def storage_root(monkeypatch, tmp_path) -> Path:
    """Redirect session storage; never touch the real uploads directory."""

    root = tmp_path / "diarization_sessions"
    monkeypatch.setattr(uploads, "_storage_root", lambda: root)
    return root


def _diarization_payload() -> dict:
    return {
        "model": MODEL,
        "duration": 0.5,
        "num_speakers": 2,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "segments": [
            {
                "id": "seg_000",
                "start": 0.0,
                "end": 0.2,
                "speaker": "SPEAKER_00",
                "confidence": 0.9,
                "confidence_bucket": "high",
            },
            {
                "id": "seg_001",
                "start": 0.3,
                "end": 0.5,
                "speaker": "SPEAKER_01",
                "confidence": 0.7,
                "confidence_bucket": "high",
            },
        ],
        "embeddings": {"seg_000": [1.0, 0.0], "seg_001": [0.0, 1.0]},
    }


class _CountingDiarization:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, model_key, audio_path):
        self.calls.append(str(audio_path))
        return _diarization_payload()

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fake_inference():
    fake = _CountingDiarization()
    with patch("app.tasks.task_b.router.run_diarization", new=fake):
        yield fake


async def _upload(client, filename="meeting.wav", content=None, content_type="audio/wav"):
    payload = _wav_bytes() if content is None else content
    return await client.post(
        f"{BASE}/uploads",
        files={"file": (filename, io.BytesIO(payload), content_type)},
    )


def _session_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [entry for session in root.iterdir() for entry in session.iterdir()]


# ---------------------------------------------------------------------------
# sid validation
# ---------------------------------------------------------------------------


def test_sid_validation_accepts_the_exact_hex_form():
    sid = _valid_sid()
    assert uploads.validate_and_canonicalize_sid(sid) == sid


@pytest.mark.parametrize(
    "bad_sid",
    [
        None,
        "",
        "not-a-session-id",
        secrets.token_hex(16).upper(),  # uppercase is rejected, never "fixed up"
        secrets.token_hex(16)[:31],  # too short
        secrets.token_hex(16) + "a",  # too long
        "../../../etc/passwd",
        f"{secrets.token_hex(16)}/..",
        f" {secrets.token_hex(16)} ",
    ],
)
def test_sid_validation_rejects_non_exact_forms(bad_sid):
    with pytest.raises(uploads.InvalidSessionId):
        uploads.validate_and_canonicalize_sid(bad_sid)


# ---------------------------------------------------------------------------
# Extension and size validation
# ---------------------------------------------------------------------------


def test_allowed_extensions_and_size_cap_are_the_documented_values():
    assert uploads.ALLOWED_AUDIO_EXTENSIONS == {".wav", ".mp3", ".m4a", ".flac"}
    assert uploads.MAX_FILE_BYTES == 50 * 1024 * 1024
    assert uploads.UPLOAD_ID_PREFIX == "upl_"
    assert uploads.PERTURBED_ID_PREFIX == "prt_"


@pytest.mark.asyncio
async def test_upload_accepts_a_wav(client, storage_root):
    response = await _upload(client)

    assert response.status_code == 200
    body = response.json()
    assert body["recording_id"].startswith("upl_")
    assert body["extension"] == ".wav"
    assert body["size_bytes"] > 0
    assert set(body) == {"recording_id", "display_filename", "extension", "size_bytes"}


@pytest.mark.parametrize("filename", ["clip.mp3", "clip.m4a", "clip.flac", "CLIP.WAV"])
@pytest.mark.asyncio
async def test_upload_accepts_every_allowed_extension_case_insensitively(
    client, storage_root, filename
):
    response = await _upload(client, filename=filename)

    assert response.status_code == 200
    assert response.json()["extension"] == Path(filename).suffix.lower()


@pytest.mark.parametrize(
    "filename",
    ["notes.txt", "payload.exe", "noextension", "clip.wav.txt", "archive.zip", "clip."],
)
@pytest.mark.asyncio
async def test_upload_rejects_disallowed_extensions(client, storage_root, filename):
    response = await _upload(client, filename=filename)

    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]
    assert _session_files(storage_root) == []


@pytest.mark.asyncio
async def test_upload_rejects_an_empty_file_and_leaves_nothing_behind(client, storage_root):
    response = await _upload(client, content=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    assert _session_files(storage_root) == []


@pytest.mark.asyncio
async def test_upload_rejects_an_oversized_file_and_leaves_nothing_behind(
    client, storage_root, monkeypatch
):
    monkeypatch.setattr(uploads, "MAX_FILE_BYTES", 1024)

    response = await _upload(client, content=b"RIFF" + b"\x00" * 4096)

    assert response.status_code == 413
    assert "50 MB" in response.json()["detail"]
    assert _session_files(storage_root) == []


@pytest.mark.asyncio
async def test_a_rejected_upload_leaves_no_temp_file(client, storage_root):
    """`begin_upload_write` allocates a `.tmp-` path before validation of the
    bytes; the route owns cleaning it up on every failure path."""

    await _upload(client, content=b"")

    leftovers = _session_files(storage_root)
    assert leftovers == []
    assert not any(entry.name.startswith(".tmp-") for entry in leftovers)


# ---------------------------------------------------------------------------
# Storage: opaque ids, no filename echo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_never_echoes_the_client_filename(client, storage_root):
    response = await _upload(client, filename="my-secret-meeting-2026.wav")

    body = response.json()
    assert "my-secret-meeting" not in str(body)
    assert body["display_filename"] == f"{body['recording_id']}.wav"
    # ...and it is not on disk under that name either.
    on_disk = _session_files(storage_root)
    assert len(on_disk) == 1
    assert "my-secret-meeting" not in on_disk[0].name


@pytest.mark.asyncio
async def test_upload_ids_are_opaque_and_distinct(client, storage_root):
    first = (await _upload(client)).json()["recording_id"]
    second = (await _upload(client)).json()["recording_id"]

    assert first != second
    for upload_id in (first, second):
        digest = upload_id.removeprefix("upl_")
        assert len(digest) == 32
        assert all(character in "0123456789abcdef" for character in digest)


@pytest.mark.asyncio
async def test_uploaded_bytes_are_stored_verbatim(client, storage_root):
    payload = _wav_bytes(freq=440.0)

    await _upload(client, content=payload)

    on_disk = _session_files(storage_root)
    assert len(on_disk) == 1
    assert on_disk[0].read_bytes() == payload


@pytest.mark.asyncio
async def test_uploads_are_stored_under_the_session_directory(client, storage_root):
    await _upload(client)

    sessions = list(storage_root.iterdir())
    assert len(sessions) == 1
    assert uploads.validate_and_canonicalize_sid(sessions[0].name) == sessions[0].name


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_listing_is_empty_for_a_session_that_never_uploaded(client, storage_root):
    response = await client.get(f"{BASE}/uploads")

    assert response.status_code == 200
    assert response.json() == {"total_uploads": 0, "uploads": []}


@pytest.mark.asyncio
async def test_listing_returns_uploads_newest_first(client, storage_root):
    ids = [
        (await _upload(client, filename=name)).json()["recording_id"]
        for name in ("a.wav", "b.wav", "c.wav")
    ]

    # Ordering is by mtime, and three uploads inside one test can land in the
    # same filesystem timestamp tick. Stamp them explicitly so this asserts the
    # sort rather than the clock's resolution.
    session_dir = next(storage_root.iterdir())
    for offset, upload_id in enumerate(ids):
        path = session_dir / f"{upload_id}.wav"
        os.utime(path, (1_700_000_000 + offset, 1_700_000_000 + offset))

    response = await client.get(f"{BASE}/uploads")

    assert response.status_code == 200
    body = response.json()
    assert body["total_uploads"] == 3
    listed = [entry["recording_id"] for entry in body["uploads"]]
    assert listed == list(reversed(ids))  # newest first


@pytest.mark.asyncio
async def test_listing_excludes_perturbed_clips(client, storage_root):
    """Perturbed clips live in the same directory but are derived artefacts;
    listing one as an upload would let it masquerade as a source recording."""

    upload_id = (await _upload(client)).json()["recording_id"]
    session_dir = next(storage_root.iterdir())
    (session_dir / "prt_deadbeef.wav").write_bytes(_wav_bytes())

    body = (await client.get(f"{BASE}/uploads")).json()

    assert [entry["recording_id"] for entry in body["uploads"]] == [upload_id]


@pytest.mark.asyncio
async def test_listing_excludes_temp_and_non_audio_files(client, storage_root):
    upload_id = (await _upload(client)).json()["recording_id"]
    session_dir = next(storage_root.iterdir())
    (session_dir / ".tmp-halfwritten.wav").write_bytes(b"partial")
    (session_dir / "upl_notes.txt").write_text("not audio")

    body = (await client.get(f"{BASE}/uploads")).json()

    assert [entry["recording_id"] for entry in body["uploads"]] == [upload_id]


# ---------------------------------------------------------------------------
# Audio playback and session isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uploaded_audio_is_served_back(client, storage_root):
    payload = _wav_bytes(freq=330.0)
    upload_id = (await _upload(client, content=payload)).json()["recording_id"]

    response = await client.get(f"{BASE}/uploads/{upload_id}/audio")

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"] == "audio/wav"


@pytest.mark.parametrize(
    "bad_id",
    ["upl_unknown", "not-an-upload-id", "../../etc/passwd", "upl_" + "0" * 32],
)
@pytest.mark.asyncio
async def test_uploaded_audio_404s_for_unknown_and_traversal_ids(client, storage_root, bad_id):
    await _upload(client)

    response = await client.get(f"{BASE}/uploads/{bad_id}/audio")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_one_session_cannot_read_another_sessions_upload(client, storage_root):
    upload_id = (await _upload(client)).json()["recording_id"]

    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other:
        response = await other.get(f"{BASE}/uploads/{upload_id}/audio")
        listing = await other.get(f"{BASE}/uploads")

    assert response.status_code == 404
    assert listing.json() == {"total_uploads": 0, "uploads": []}


@pytest.mark.asyncio
async def test_one_session_cannot_diarize_another_sessions_upload(
    client, storage_root, fake_inference
):
    upload_id = (await _upload(client)).json()["recording_id"]

    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as other:
        response = await other.post(
            f"{BASE}/run", json={"model": MODEL, "recording_id": upload_id}
        )

    assert response.status_code == 404
    assert fake_inference.call_count == 0


# ---------------------------------------------------------------------------
# Resolver-level guarantees
# ---------------------------------------------------------------------------


def test_resolve_upload_path_rejects_traversal_shaped_ids(storage_root):
    sid = _valid_sid()
    session_dir = storage_root / sid
    session_dir.mkdir(parents=True)
    (session_dir / "upl_real.wav").write_bytes(b"RIFF")
    outside = storage_root / "outside.wav"
    outside.write_bytes(b"RIFF-not-yours")

    for bad_id in ("../outside", "../../etc/passwd", str(outside), "upl_real.wav", ""):
        with pytest.raises(uploads.UploadNotFound):
            uploads.resolve_upload_path(bad_id, sid)

    assert uploads.resolve_upload_path("upl_real", sid) == session_dir / "upl_real.wav"


def test_resolve_perturbed_path_rejects_traversal_shaped_ids(storage_root):
    sid = _valid_sid()
    session_dir = storage_root / sid
    session_dir.mkdir(parents=True)
    (session_dir / "prt_real.wav").write_bytes(b"RIFF")

    for bad_id in ("../outside", "prt_unknown", ""):
        with pytest.raises(uploads.PerturbedNotFound):
            uploads.resolve_perturbed_path(bad_id, sid)

    assert uploads.resolve_perturbed_path("prt_real", sid) == session_dir / "prt_real.wav"


def test_resolvers_reject_a_malformed_sid_before_touching_the_filesystem(storage_root):
    with pytest.raises(uploads.InvalidSessionId):
        uploads.resolve_upload_path("upl_anything", "../../etc")
    with pytest.raises(uploads.InvalidSessionId):
        uploads.list_uploads(None)


def test_list_uploads_is_empty_for_a_session_with_no_directory(storage_root):
    assert uploads.list_uploads(_valid_sid()) == []


def test_promote_upload_derives_every_field_from_disk(storage_root):
    sid = _valid_sid()
    temp_path = uploads.begin_upload_write(sid, ".wav")
    temp_path.write_bytes(b"RIFF-0123456789")

    recording = uploads.promote_upload(temp_path, sid, ".wav")

    assert recording.recording_id.startswith("upl_")
    assert recording.display_filename == f"{recording.recording_id}.wav"
    assert recording.size_bytes == len(b"RIFF-0123456789")
    assert not temp_path.exists()  # renamed, not copied


def test_begin_upload_write_allocates_inside_the_session_dir(storage_root):
    """Same-directory temp path is what makes promotion an atomic rename."""

    sid = _valid_sid()
    temp_path = uploads.begin_upload_write(sid, ".wav")

    assert temp_path.parent == storage_root / sid
    assert temp_path.name.startswith(".tmp-")
    assert temp_path.parent.is_dir()


# ---------------------------------------------------------------------------
# The Phase 1 exit criterion: uploads flow through the content-hash cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_works_on_an_upload(client, storage_root, fake_inference):
    upload_id = (await _upload(client)).json()["recording_id"]

    response = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": upload_id})

    assert response.status_code == 200
    body = response.json()
    assert body["recording_id"] == upload_id
    assert body["cached"] is False
    assert body["num_speakers"] == 2
    assert fake_inference.call_count == 1


@pytest.mark.asyncio
async def test_rerunning_the_same_upload_is_an_instant_cache_hit(
    client, storage_root, fake_inference
):
    upload_id = (await _upload(client)).json()["recording_id"]
    payload = {"model": MODEL, "recording_id": upload_id}

    first = await client.post(f"{BASE}/run", json=payload)
    second = await client.post(f"{BASE}/run", json=payload)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert second.json()["segments"] == first.json()["segments"]
    assert fake_inference.call_count == 1


@pytest.mark.asyncio
async def test_the_same_bytes_uploaded_twice_share_one_cache_entry(
    client, storage_root, fake_inference
):
    """Two uploads of the same audio get different ids but identical content,
    so the second run hits the cache — the key is the file's bytes, not its id."""

    payload = _wav_bytes(freq=512.0)
    first_id = (await _upload(client, filename="a.wav", content=payload)).json()["recording_id"]
    second_id = (await _upload(client, filename="b.wav", content=payload)).json()["recording_id"]
    assert first_id != second_id

    first = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": first_id})
    second = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": second_id})

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert fake_inference.call_count == 1
    assert second.json()["recording_id"] == second_id


@pytest.mark.asyncio
async def test_different_uploads_do_not_share_a_cache_entry(client, storage_root, fake_inference):
    first_id = (await _upload(client, content=_wav_bytes(freq=220.0))).json()["recording_id"]
    second_id = (await _upload(client, content=_wav_bytes(freq=880.0))).json()["recording_id"]

    first = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": first_id})
    second = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": second_id})

    assert first.json()["cached"] is False
    assert second.json()["cached"] is False
    assert fake_inference.call_count == 2


@pytest.mark.asyncio
async def test_projection_works_on_an_upload(client, storage_root, fake_inference):
    upload_id = (await _upload(client)).json()["recording_id"]

    response = await client.get(
        f"{BASE}/projection", params={"model": MODEL, "recording_id": upload_id}
    )

    assert response.status_code == 200
    assert len(response.json()["points"]) == 2
