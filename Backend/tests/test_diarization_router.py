"""Endpoint contracts for the Speaker Diarization router (task_b).

Mounted at `/tasks/task-b`. Every test runs against a fake `ami_subset/` under
`tmp_path` (never the real AMI data) and patches
`app.tasks.task_b.router.run_diarization`, so no model is ever loaded — the
fake's call count is what proves the Redis cache is actually being hit.
"""

import json
from unittest.mock import patch

import pytest

from app.core import redis as redis_module
from app.core.settings import settings
from app.tasks.task_b import dataset

BASE = "/tasks/task-b"
MODEL = "pyannote-3.1"
ALL_MODELS = ["pyannote-3.1", "reverb-v1", "reverb-v2"]

REAL_DATASET_DIR = settings.speaker_diarization_ami_dataset_dir

FAKE_RECORDINGS = {
    "ES2004a.Mix-Headset.wav": b"RIFF-fake-es2004a",
    "IS1009a.Mix-Headset.wav": b"RIFF-fake-is1009a",
    "TS3003a.Mix-Headset.wav": b"RIFF-fake-ts3003a",
}


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)
    assert settings.speaker_diarization_ami_dataset_dir != REAL_DATASET_DIR

    dataset_dir = tmp_path / "ami_subset"
    dataset_dir.mkdir(parents=True)
    for filename, payload in FAKE_RECORDINGS.items():
        (dataset_dir / filename).write_bytes(payload)
    (dataset_dir / "rttm").mkdir()
    (dataset_dir / "rttm" / "ES2004a.rttm").write_text("SPEAKER ES2004a 1 0.0 1.0 <NA>\n")
    return dataset_dir


@pytest.fixture
def missing_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)
    assert settings.speaker_diarization_ami_dataset_dir != REAL_DATASET_DIR
    return tmp_path


def _diarization_payload(num_embeddings: int = 3) -> dict:
    """What a real `run_diarization` returns, in miniature."""

    segments = [
        {
            "id": f"seg_{index:03d}",
            "start": float(index * 2),
            "end": float(index * 2 + 1),
            "speaker": f"SPEAKER_{index % 2:02d}",
            "confidence": 0.8,
            "confidence_bucket": "high",
        }
        for index in range(3)
    ]
    embeddings = {
        f"seg_{index:03d}": [1.0, 0.0, 0.0] if index % 2 == 0 else [0.0, 1.0, 0.0]
        for index in range(num_embeddings)
    }
    for segment in segments:
        if segment["id"] not in embeddings:
            segment["confidence"] = None
            segment["confidence_bucket"] = None
    return {
        "model": MODEL,
        "duration": 10.0,
        "num_speakers": 2,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "segments": segments,
        "embeddings": embeddings,
    }


class _CountingDiarization:
    """Stands in for `run_diarization`; counts how often inference really ran."""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload if payload is not None else _diarization_payload()
        self.calls: list[tuple] = []

    def __call__(self, model_key, audio_path):
        self.calls.append((model_key, str(audio_path)))
        return self.payload

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fake_inference():
    fake = _CountingDiarization()
    with patch("app.tasks.task_b.router.run_diarization", new=fake):
        yield fake


async def _any_recording_id(client) -> str:
    response = await client.get(f"{BASE}/dataset/recordings")
    return response.json()["recordings"][0]["recording_id"]


BAD_IDS = ["not-a-real-id", "..%2F..%2Fetc%2Fpasswd", "rec_0000000000000000"]


# ---------------------------------------------------------------------------
# Models and dataset metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_endpoint(client):
    response = await client.get(f"{BASE}/models")

    assert response.status_code == 200
    models = response.json()["models"]
    assert [m["key"] for m in models] == ALL_MODELS
    assert [m["key"] for m in models if m["recommended"]] == [MODEL]
    # The workbench header renders this; without it the UI would imply the
    # three models differ more than they do.
    assert all(m["glass_box_note"].strip() for m in models)




@pytest.mark.asyncio
async def test_dataset_endpoint(client, fake_dataset_dir):
    response = await client.get(f"{BASE}/dataset")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "ami-subset"
    assert body["total_recordings"] == len(FAKE_RECORDINGS)
    assert body["available"] is True


@pytest.mark.asyncio
async def test_dataset_endpoint_reports_unavailable_rather_than_404(client, missing_dataset_dir):
    """`/dataset` is a summary and must stay 200 so the UI can say "not set up"."""

    response = await client.get(f"{BASE}/dataset")

    assert response.status_code == 200
    assert response.json()["available"] is False


@pytest.mark.asyncio
async def test_dataset_recordings_endpoint(client, fake_dataset_dir):
    response = await client.get(f"{BASE}/dataset/recordings")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "ami-subset"
    assert body["total_recordings"] == len(FAKE_RECORDINGS)
    assert len(body["recordings"]) == len(FAKE_RECORDINGS)
    for recording in body["recordings"]:
        assert set(recording) == {"recording_id", "display_filename", "extension", "size_bytes"}
        assert recording["recording_id"].startswith("rec_")
    # Offline-eval ground truth never reaches a runtime response.
    assert "rttm" not in json.dumps(body).lower()


@pytest.mark.asyncio
async def test_dataset_single_recording_endpoint(client, fake_dataset_dir):
    recording_id = await _any_recording_id(client)

    response = await client.get(f"{BASE}/dataset/recordings/{recording_id}")

    assert response.status_code == 200
    assert response.json()["recording_id"] == recording_id


@pytest.mark.parametrize("bad_id", BAD_IDS)
@pytest.mark.asyncio
async def test_dataset_single_recording_rejects_bad_ids(client, fake_dataset_dir, bad_id):
    response = await client.get(f"{BASE}/dataset/recordings/{bad_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_dataset_endpoints_404_when_dataset_missing(client, missing_dataset_dir):
    assert (await client.get(f"{BASE}/dataset/recordings")).status_code == 404
    assert (await client.get(f"{BASE}/dataset/recordings/rec_anything")).status_code == 404
    assert (await client.get(f"{BASE}/dataset/recordings/rec_anything/audio")).status_code == 404


# ---------------------------------------------------------------------------
# Audio streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recording_audio_returns_exact_bytes(client, fake_dataset_dir):
    listing = (await client.get(f"{BASE}/dataset/recordings")).json()["recordings"]
    recording = listing[0]

    response = await client.get(f"{BASE}/dataset/recordings/{recording['recording_id']}/audio")

    # No Range header: 200 and the whole file, advertising that ranges work.
    assert response.status_code == 200
    assert response.content == FAKE_RECORDINGS[recording["display_filename"]]
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["accept-ranges"] == "bytes"


@pytest.mark.asyncio
async def test_recording_audio_honours_range_with_a_206(client, fake_dataset_dir):
    """Served through verification's `stream_audio_file`, so a seek deep into
    a long meeting fetches only the bytes it needs instead of the whole file."""

    recording = (await client.get(f"{BASE}/dataset/recordings")).json()["recordings"][0]
    payload = FAKE_RECORDINGS[recording["display_filename"]]

    response = await client.get(
        f"{BASE}/dataset/recordings/{recording['recording_id']}/audio", headers={"Range": "bytes=0-3"}
    )

    assert response.status_code == 206
    assert response.content == payload[0:4]
    assert response.headers["content-range"] == f"bytes 0-3/{len(payload)}"
    assert response.headers["content-length"] == "4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"] == "audio/wav"


@pytest.mark.asyncio
async def test_recording_audio_open_ended_range_returns_the_tail(client, fake_dataset_dir):
    recording = (await client.get(f"{BASE}/dataset/recordings")).json()["recordings"][0]
    payload = FAKE_RECORDINGS[recording["display_filename"]]

    response = await client.get(
        f"{BASE}/dataset/recordings/{recording['recording_id']}/audio", headers={"Range": "bytes=4-"}
    )

    assert response.status_code == 206
    assert response.content == payload[4:]
    assert response.headers["content-range"] == f"bytes 4-{len(payload) - 1}/{len(payload)}"


@pytest.mark.parametrize("bad_id", BAD_IDS)
@pytest.mark.asyncio
async def test_recording_audio_rejects_bad_ids(client, fake_dataset_dir, bad_id):
    response = await client.get(f"{BASE}/dataset/recordings/{bad_id}/audio")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /run -- cache miss, cache hit, and what the key is made of
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_the_diarization_payload(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)

    response = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": recording_id})

    assert response.status_code == 200
    body = response.json()
    assert body["recording_id"] == recording_id
    assert body["cached"] is False
    assert body["num_speakers"] == 2
    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert len(body["segments"]) == 3
    assert fake_inference.call_count == 1


@pytest.mark.asyncio
async def test_run_second_identical_request_is_a_cache_hit(client, fake_dataset_dir, fake_inference):
    """The Phase 0 exit criterion: a repeat must not re-run inference."""

    recording_id = await _any_recording_id(client)
    payload = {"model": MODEL, "recording_id": recording_id}

    first = await client.post(f"{BASE}/run", json=payload)
    second = await client.post(f"{BASE}/run", json=payload)

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert fake_inference.call_count == 1
    # The cached response carries the same science as the fresh one.
    assert second.json()["segments"] == first.json()["segments"]


@pytest.mark.asyncio
async def test_cache_key_is_the_content_hash_not_the_recording_id(
    client, monkeypatch, tmp_path, fake_inference
):
    """Two different recordings holding identical bytes share one cache entry —
    this is what makes a re-uploaded file instant."""

    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "ami_subset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "first.wav").write_bytes(b"RIFF-identical-bytes")
    (dataset_dir / "second.wav").write_bytes(b"RIFF-identical-bytes")

    first_id = dataset._recording_id_for("first.wav")
    second_id = dataset._recording_id_for("second.wav")
    assert first_id != second_id

    first = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": first_id})
    second = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": second_id})

    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert fake_inference.call_count == 1
    # Each response still reports the id that was actually asked for.
    assert first.json()["recording_id"] == first_id
    assert second.json()["recording_id"] == second_id


@pytest.mark.asyncio
async def test_cache_key_is_namespaced_per_model(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)
    await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": recording_id})

    keys = await redis_module.redis.keys("result:diar:*")

    assert len(keys) == 1
    assert keys[0].startswith(f"result:diar:{MODEL}:")


@pytest.mark.asyncio
async def test_the_three_models_cache_independently(client, fake_dataset_dir, fake_inference):
    """Same recording, three models -> three entries, not one shared result.
    A collision here would silently show one model's timeline under another's
    name, which is exactly what the comparison workbench exists to avoid."""

    recording_id = await _any_recording_id(client)
    for model_key in ALL_MODELS:
        response = await client.post(
            f"{BASE}/run", json={"model": model_key, "recording_id": recording_id}
        )
        assert response.status_code == 200
        assert response.json()["cached"] is False

    keys = await redis_module.redis.keys("result:diar:*")

    assert len(keys) == 3
    assert {key.rsplit(":", 1)[0] for key in keys} == {
        f"result:diar:{model_key}" for model_key in ALL_MODELS
    }
    # Every model really ran; none of them was served another's cache entry.
    assert fake_inference.call_count == 3
    assert [call[0] for call in fake_inference.calls] == ALL_MODELS


@pytest.mark.asyncio
@pytest.mark.parametrize("model_key", ALL_MODELS)
async def test_run_accepts_every_registered_model(
    client, fake_dataset_dir, fake_inference, model_key
):
    recording_id = await _any_recording_id(client)

    response = await client.post(
        f"{BASE}/run", json={"model": model_key, "recording_id": recording_id}
    )

    assert response.status_code == 200
    assert fake_inference.calls[0][0] == model_key


@pytest.mark.asyncio
async def test_run_is_written_to_the_cache_with_a_ttl(client, fake_dataset_dir, fake_inference):
    """Diarization costs minutes of CPU, so the entry must actually persist."""

    recording_id = await _any_recording_id(client)
    await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": recording_id})

    keys = await redis_module.redis.keys("result:diar:*")
    ttl = await redis_module.redis.ttl(keys[0])

    assert 0 < ttl <= 7 * 24 * 60 * 60


@pytest.mark.asyncio
async def test_run_rejects_an_unknown_model(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)

    response = await client.post(
        f"{BASE}/run", json={"model": "pyannote-4.0", "recording_id": recording_id}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    for model_key in ALL_MODELS:
        assert model_key in detail
    assert fake_inference.call_count == 0


@pytest.mark.asyncio
async def test_run_validates_the_model_before_touching_the_filesystem(client, fake_dataset_dir):
    """A bad model plus a bad recording id is a 400, not a 404: the cheap,
    unambiguous error wins."""

    response = await client.post(
        f"{BASE}/run", json={"model": "nope", "recording_id": "not-a-real-id"}
    )
    assert response.status_code == 400


@pytest.mark.parametrize("bad_id", ["not-a-real-id", "rec_0000000000000000", "../../etc/passwd"])
@pytest.mark.asyncio
async def test_run_404s_for_an_unknown_recording(client, fake_dataset_dir, fake_inference, bad_id):
    response = await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": bad_id})

    assert response.status_code == 404
    assert fake_inference.call_count == 0


@pytest.mark.parametrize(
    "body",
    [{}, {"model": MODEL}, {"recording_id": "rec_x"}, {"model": None, "recording_id": "rec_x"}],
)
@pytest.mark.asyncio
async def test_run_rejects_malformed_bodies(client, fake_dataset_dir, body):
    response = await client.post(f"{BASE}/run", json=body)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_run_returns_503_when_the_model_cannot_load(client, fake_dataset_dir):
    """Gated pyannote weights or a drifted venv -- infrastructure, not input."""

    from app.tasks.task_b.service import DiarizationModelUnavailable

    recording_id = await _any_recording_id(client)

    def _unavailable(model_key, audio_path):
        raise DiarizationModelUnavailable("HF_TOKEN is not set.")

    with patch("app.tasks.task_b.router.run_diarization", new=_unavailable):
        response = await client.post(
            f"{BASE}/run", json={"model": MODEL, "recording_id": recording_id}
        )

    assert response.status_code == 503
    assert "HF_TOKEN" in response.json()["detail"]


@pytest.mark.asyncio
async def test_a_failed_run_is_not_cached(client, fake_dataset_dir):
    from app.tasks.task_b.service import DiarizationModelUnavailable

    recording_id = await _any_recording_id(client)

    def _unavailable(model_key, audio_path):
        raise DiarizationModelUnavailable("nope")

    with patch("app.tasks.task_b.router.run_diarization", new=_unavailable):
        await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": recording_id})

    assert await redis_module.redis.keys("result:diar:*") == []


# ---------------------------------------------------------------------------
# GET /projection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_projection_returns_one_point_per_embedding(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)

    response = await client.get(
        f"{BASE}/projection", params={"model": MODEL, "recording_id": recording_id}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL
    assert body["recording_id"] == recording_id
    assert len(body["points"]) == 3
    for point in body["points"]:
        assert set(point) == {"id", "x", "y", "speaker", "confidence"}
        assert isinstance(point["x"], float)
        assert point["speaker"].startswith("SPEAKER_")


@pytest.mark.asyncio
async def test_projection_reuses_the_cached_run(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)
    await client.post(f"{BASE}/run", json={"model": MODEL, "recording_id": recording_id})

    response = await client.get(
        f"{BASE}/projection", params={"model": MODEL, "recording_id": recording_id}
    )

    assert response.status_code == 200
    assert fake_inference.call_count == 1


@pytest.mark.asyncio
async def test_projection_422s_with_fewer_than_two_embeddings(client, fake_dataset_dir):
    """PCA to 2D is undefined with a single point; it must not 500."""

    single = _CountingDiarization(_diarization_payload(num_embeddings=1))
    recording_id = await _any_recording_id(client)

    with patch("app.tasks.task_b.router.run_diarization", new=single):
        response = await client.get(
            f"{BASE}/projection", params={"model": MODEL, "recording_id": recording_id}
        )

    assert response.status_code == 422
    assert "projection" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_projection_defaults_to_2d_with_explained_variance(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)

    response = await client.get(
        f"{BASE}/projection", params={"model": MODEL, "recording_id": recording_id}
    )

    body = response.json()
    assert body["dims"] == 2
    assert len(body["explained_variance"]) == 2
    assert all(0.0 <= share <= 1.0 for share in body["explained_variance"])
    assert all("z" not in point for point in body["points"])


@pytest.mark.asyncio
async def test_projection_in_3d_adds_a_z_axis(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)

    response = await client.get(
        f"{BASE}/projection",
        params={"model": MODEL, "recording_id": recording_id, "dims": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dims"] == 3
    assert len(body["explained_variance"]) == 3
    assert len(body["points"]) == 3
    for point in body["points"]:
        assert set(point) == {"id", "x", "y", "z", "speaker", "confidence"}
        assert isinstance(point["z"], float)
    # Same cached run as the 2D map -- asking for 3D must not re-diarize.
    await client.get(f"{BASE}/projection", params={"model": MODEL, "recording_id": recording_id})
    assert fake_inference.call_count == 1


@pytest.mark.asyncio
async def test_projection_3d_422s_with_fewer_than_three_embeddings(client, fake_dataset_dir):
    """Two points are enough for 2D but not for three PCA components."""

    two = _CountingDiarization(_diarization_payload(num_embeddings=2))
    recording_id = await _any_recording_id(client)

    with patch("app.tasks.task_b.router.run_diarization", new=two):
        flat = await client.get(
            f"{BASE}/projection", params={"model": MODEL, "recording_id": recording_id}
        )
        deep = await client.get(
            f"{BASE}/projection",
            params={"model": MODEL, "recording_id": recording_id, "dims": 3},
        )

    assert flat.status_code == 200
    assert deep.status_code == 422
    assert "3D projection" in deep.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("dims", [1, 4])
async def test_projection_rejects_unsupported_dims(client, fake_dataset_dir, fake_inference, dims):
    recording_id = await _any_recording_id(client)

    response = await client.get(
        f"{BASE}/projection",
        params={"model": MODEL, "recording_id": recording_id, "dims": dims},
    )

    assert response.status_code == 422
    assert fake_inference.call_count == 0


@pytest.mark.asyncio
async def test_projection_rejects_an_unknown_model(client, fake_dataset_dir, fake_inference):
    recording_id = await _any_recording_id(client)

    response = await client.get(
        f"{BASE}/projection", params={"model": "nope", "recording_id": recording_id}
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_projection_404s_for_an_unknown_recording(client, fake_dataset_dir, fake_inference):
    response = await client.get(
        f"{BASE}/projection", params={"model": MODEL, "recording_id": "not-a-real-id"}
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_projection_requires_both_query_params(client, fake_dataset_dir):
    assert (await client.get(f"{BASE}/projection", params={"model": MODEL})).status_code == 422
    assert (await client.get(f"{BASE}/projection", params={"recording_id": "x"})).status_code == 422


# ---------------------------------------------------------------------------
# POST /perturbation -- endpoint contract only; the diff algorithm is tested
# in test_diarization_diff.py and the transforms in test_perturbation_service.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perturbation_rejects_an_unknown_model_before_perturbing(
    client, fake_dataset_dir, fake_inference
):
    recording_id = await _any_recording_id(client)

    with patch("app.tasks.task_b.router.perturbation.perturb_to_session") as perturb:
        response = await client.post(
            f"{BASE}/perturbation",
            json={
                "model": "pyannote-4.0",
                "recording_id": recording_id,
                "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
            },
        )

    assert response.status_code == 400
    perturb.assert_not_called()


@pytest.mark.asyncio
async def test_perturbation_404s_for_an_unknown_recording(client, fake_dataset_dir, fake_inference):
    with patch("app.tasks.task_b.router.perturbation.perturb_to_session") as perturb:
        response = await client.post(
            f"{BASE}/perturbation",
            json={
                "model": MODEL,
                "recording_id": "not-a-real-id",
                "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
            },
        )

    assert response.status_code == 404
    perturb.assert_not_called()


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "reverb", "params": {}},  # unsupported type
        {"type": "noise", "params": {}},  # missing noise_level
        {"type": "noise", "params": {"noise_level": 5.0}},  # out of range
        {"type": "noise", "params": {"noise_level": 0.05, "extra": 1}},  # forbidden key
        {"type": "time_masking", "params": {"mask_start_percent": 60, "mask_end_percent": 10}},
    ],
)
@pytest.mark.asyncio
async def test_perturbation_422s_on_bad_params_and_writes_nothing(
    client, fake_dataset_dir, fake_inference, spec
):
    """A rejected request is the user's problem, not an outage -- and it must
    leave no clip on disk and no diarization run behind it."""

    recording_id = await _any_recording_id(client)

    response = await client.post(
        f"{BASE}/perturbation",
        json={"model": MODEL, "recording_id": recording_id, "perturbation": spec},
    )

    assert response.status_code == 422
    assert fake_inference.call_count == 0
    assert await redis_module.redis.keys("result:diar-delta:*") == []


@pytest.mark.parametrize(
    "body",
    [
        {"model": MODEL, "recording_id": "rec_x"},  # no perturbation block
        {"model": MODEL, "perturbation": {"type": "noise", "params": {}}},
        {},
    ],
)
@pytest.mark.asyncio
async def test_perturbation_rejects_malformed_bodies(client, fake_dataset_dir, body):
    response = await client.post(f"{BASE}/perturbation", json=body)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /perturbation/preview -- build the clip so it can be heard, never diarize
# ---------------------------------------------------------------------------

NOISE_SPEC = {"type": "noise", "params": {"noise_level": 0.05}}


@pytest.mark.asyncio
async def test_preview_builds_the_clip_without_diarizing(
    client, fake_dataset_dir, fake_inference, tmp_path
):
    recording_id = await _any_recording_id(client)
    clip = tmp_path / "prt_preview.wav"
    clip.write_bytes(b"RIFF")

    with patch(
        "app.tasks.task_b.router.perturbation.perturb_to_session",
        return_value=("prt_preview", clip, {"noise_level": 0.05}),
    ) as perturb:
        response = await client.post(
            f"{BASE}/perturbation/preview",
            json={"recording_id": recording_id, "perturbation": NOISE_SPEC},
        )

    assert response.status_code == 200
    assert response.json() == {
        "perturbed_id": "prt_preview",
        "perturbation": {"type": "noise", "params": {"noise_level": 0.05}},
    }
    perturb.assert_called_once()
    assert perturb.call_args.args[3:] == ("noise", {"noise_level": 0.05})
    assert fake_inference.call_count == 0
    assert await redis_module.redis.keys("result:diar*") == []


@pytest.mark.asyncio
async def test_preview_and_full_run_build_the_same_clip(client, fake_dataset_dir, fake_inference):
    """Both endpoints go through one helper with the same arguments, so the
    deterministic `prt_` id -- and the file -- a preview builds is the one the
    full run reuses."""

    recording_id = await _any_recording_id(client)
    body = {"recording_id": recording_id, "perturbation": NOISE_SPEC}

    with patch(
        "app.tasks.task_b.router.perturbation.perturb_to_session",
        side_effect=ValueError("stop after perturbing"),
    ) as perturb:
        await client.post(f"{BASE}/perturbation/preview", json=body)
        await client.post(f"{BASE}/perturbation", json={**body, "model": MODEL})

    assert perturb.call_count == 2
    preview_args, run_args = (call.args for call in perturb.call_args_list)
    assert preview_args[2:] == run_args[2:]  # source hash, type, params


@pytest.mark.asyncio
async def test_preview_404s_for_an_unknown_recording(client, fake_dataset_dir, fake_inference):
    with patch("app.tasks.task_b.router.perturbation.perturb_to_session") as perturb:
        response = await client.post(
            f"{BASE}/perturbation/preview",
            json={"recording_id": "not-a-real-id", "perturbation": NOISE_SPEC},
        )

    assert response.status_code == 404
    perturb.assert_not_called()


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "reverb", "params": {}},
        {"type": "noise", "params": {}},
        {"type": "noise", "params": {"noise_level": 5.0}},
        {"type": "time_masking", "params": {"mask_start_percent": 60, "mask_end_percent": 10}},
    ],
)
@pytest.mark.asyncio
async def test_preview_422s_on_bad_params(client, fake_dataset_dir, fake_inference, spec):
    recording_id = await _any_recording_id(client)

    response = await client.post(
        f"{BASE}/perturbation/preview",
        json={"recording_id": recording_id, "perturbation": spec},
    )

    assert response.status_code == 422
    assert fake_inference.call_count == 0


@pytest.mark.parametrize(
    "body",
    [{"recording_id": "rec_x"}, {"perturbation": NOISE_SPEC}, {}],
)
@pytest.mark.asyncio
async def test_preview_rejects_malformed_bodies(client, fake_dataset_dir, body):
    response = await client.post(f"{BASE}/perturbation/preview", json=body)
    assert response.status_code == 422
