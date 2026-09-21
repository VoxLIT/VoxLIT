"""The visitor's own clips: upload/record, list, stream, delete, and use them
with every per-clip feature.

No model is loaded — `run_detection` is stubbed. What is tested is storage,
validation, session ownership, and that `up_...` ids route through the same
per-clip endpoints as demo ids.
"""

from __future__ import annotations

import io
from importlib import import_module

import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient

from app.main import app
from app.tasks.deepfake import uploads

deepfake_router = import_module("app.tasks.deepfake.router")

SAMPLE_RATE = 16_000


def _wav_bytes(seconds: float = 1.5, sample_rate: int = 44_100, amplitude: float = 0.3, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def storage(monkeypatch, tmp_path):
    root = tmp_path / "deepfake_sessions"
    monkeypatch.setattr(uploads, "_storage_root", lambda: root)
    return root


@pytest.fixture
def stub_detection(monkeypatch):
    calls = []

    def _fake(model_key, audio_path):
        calls.append((model_key, str(audio_path)))
        return {"model": model_key, "decision": "bonafide", "spoof_probability": 0.2}

    monkeypatch.setattr(deepfake_router, "run_detection", _fake)
    return calls


async def _upload(client, content: bytes | None = None, name: str = "me.wav", source: str = "upload"):
    return await client.post(
        "/tasks/deepfake/uploads",
        files={"file": (name, content if content is not None else _wav_bytes(), "audio/wav")},
        data={"source": source},
    )


async def test_upload_is_normalised_to_16k_mono_wav(client, storage):
    response = await _upload(client, _wav_bytes(channels=2))

    assert response.status_code == 200, response.text
    clip = response.json()
    assert clip["recording_id"].startswith("up_")
    assert clip["display_filename"] == "me.wav"
    assert clip["source"] == "upload"
    assert clip["duration_seconds"] == pytest.approx(1.5, abs=0.01)

    (stored,) = storage.glob("*/up_*.wav")
    info = sf.info(str(stored))
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 1


async def test_listing_and_audio_and_delete(client):
    clip = (await _upload(client, source="recording", name="mic.wav")).json()
    assert clip["source"] == "recording"

    listing = await client.get("/tasks/deepfake/uploads")
    assert [row["recording_id"] for row in listing.json()["recordings"]] == [clip["recording_id"]]

    audio = await client.get(f"/tasks/deepfake/uploads/{clip['recording_id']}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"

    deleted = await client.delete(f"/tasks/deepfake/uploads/{clip['recording_id']}")
    assert deleted.status_code == 200
    assert (await client.get("/tasks/deepfake/uploads")).json()["recordings"] == []


@pytest.mark.parametrize(
    "content, name, status",
    [
        (b"not audio at all", "x.wav", 422),
        (b"", "x.wav", 400),
        (_wav_bytes(), "x.exe", 400),
        (_wav_bytes(seconds=0.2), "x.wav", 422),
        (_wav_bytes(amplitude=0.0), "x.wav", 422),
    ],
)
async def test_bad_uploads_are_refused(client, storage, content, name, status):
    response = await _upload(client, content, name)

    assert response.status_code == status
    assert not list(storage.glob("*/up_*.wav"))


async def test_overlong_clip_is_refused(client, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_CLIP_SECONDS", 1.0)
    response = await _upload(client, _wav_bytes(seconds=2.0))
    assert response.status_code == 422


async def test_oversized_file_is_refused(client, monkeypatch):
    monkeypatch.setattr(deepfake_router, "MAX_UPLOAD_BYTES", 100)
    response = await _upload(client)
    assert response.status_code == 413


async def test_display_name_cannot_carry_a_path(client, storage):
    clip = (await _upload(client, name="../../etc/passwd.wav")).json()
    assert clip["display_filename"] == "passwd.wav"
    assert all(path.parent.parent == storage for path in storage.glob("*/up_*"))


def test_nameless_clips_get_a_readable_name():
    assert uploads._clean_display_name("", "recording").startswith("recording-")
    assert uploads._clean_display_name(None, "upload").startswith("clip-")


async def test_oldest_clips_are_evicted_at_the_cap(client, monkeypatch):
    monkeypatch.setattr(uploads, "MAX_CLIPS_PER_SESSION", 2)
    first = (await _upload(client)).json()["recording_id"]
    await _upload(client)
    await _upload(client)

    ids = [row["recording_id"] for row in (await client.get("/tasks/deepfake/uploads")).json()["recordings"]]
    assert len(ids) == 2
    assert first not in ids


async def test_expired_clips_disappear(client, monkeypatch):
    clip_id = (await _upload(client)).json()["recording_id"]
    monkeypatch.setattr(uploads, "UPLOAD_TTL_SECONDS", -1)

    assert (await client.get("/tasks/deepfake/uploads")).json()["recordings"] == []
    assert (await client.get(f"/tasks/deepfake/uploads/{clip_id}/audio")).status_code == 404


async def test_another_session_cannot_see_or_use_the_clip(client, stub_detection):
    clip_id = (await _upload(client)).json()["recording_id"]

    async with AsyncClient(app=app, base_url="http://test") as stranger:
        assert (await stranger.get("/tasks/deepfake/uploads")).json()["recordings"] == []
        assert (await stranger.get(f"/tasks/deepfake/uploads/{clip_id}/audio")).status_code == 404
        assert (await stranger.delete(f"/tasks/deepfake/uploads/{clip_id}")).status_code == 404
        run = await stranger.post("/tasks/deepfake/run", json={"model": "xlsr-deepfake", "recording_id": clip_id})
        assert run.status_code == 404
    assert stub_detection == []


@pytest.mark.parametrize("clip_id", ["up_../../x", "up_ZZZZZZZZZZZZZZZZ", "up_", "up_0123456789abcdef0"])
async def test_malformed_clip_ids_miss(client, clip_id):
    response = await client.get(f"/tasks/deepfake/uploads/{clip_id}/audio")
    assert response.status_code == 404


@pytest.mark.parametrize("model", ["xlsr-deepfake", "ast-fakeaudio", "xlsr-mamba"])
async def test_run_scores_a_user_clip_with_every_model(client, stub_detection, model):
    clip_id = (await _upload(client)).json()["recording_id"]

    response = await client.post("/tasks/deepfake/run", json={"model": model, "recording_id": clip_id})

    assert response.status_code == 200
    assert response.json()["recording_id"] == clip_id
    assert stub_detection[-1][0] == model
    assert stub_detection[-1][1].endswith(f"{clip_id}.wav")


async def test_silence_probe_and_saliency_accept_user_clips(client, monkeypatch):
    clip_id = (await _upload(client)).json()["recording_id"]
    seen = []
    monkeypatch.setattr(deepfake_router, "run_silence_probe", lambda model, path: seen.append(path) or {"ok": 1})
    monkeypatch.setattr(deepfake_router, "generate_saliency", lambda model, path: seen.append(path) or {"ok": 2})

    probe = await client.post("/tasks/deepfake/silence-probe", json={"model": "xlsr-mamba", "recording_id": clip_id})
    salient = await client.post("/tasks/deepfake/saliency", json={"model": "ast-fakeaudio", "recording_id": clip_id})

    assert probe.status_code == 200 and probe.json()["recording_id"] == clip_id
    assert salient.status_code == 200 and salient.json()["recording_id"] == clip_id
    assert all(str(path).endswith(f"{clip_id}.wav") for path in seen)


async def test_embeddings_place_user_clips_on_the_dataset_map(client, monkeypatch):
    clip_id = (await _upload(client, name="mine.wav")).json()["recording_id"]
    captured = {}

    async def _fake_project(model, method, components, extra_clips):
        captured["extra"] = extra_clips
        return {"recordings": [], "coordinates": []}

    monkeypatch.setattr(deepfake_router, "project_dataset", _fake_project)

    response = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "extra_recording_ids": [clip_id, clip_id]},
    )

    assert response.status_code == 200
    ((got_id, name, path),) = captured["extra"]
    assert got_id == clip_id and name == "mine.wav" and path.name == f"{clip_id}.wav"


async def test_embeddings_refuse_demo_ids_as_extras(client):
    response = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "extra_recording_ids": ["rec_0123456789abcdef"]},
    )
    assert response.status_code == 400
