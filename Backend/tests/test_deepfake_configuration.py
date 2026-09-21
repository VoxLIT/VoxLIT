"""Configuration tests for /tasks/deepfake (Test Plan 3.1.8).

This task has four knobs that change what a number MEANS rather than merely
how fast it arrives, and each one is written into a cache key so that turning
it cannot serve a stale answer computed under the old setting:

    THRESHOLD_VERSION   the operating point a decision was taken at
    SILENCE_TOP_DB      what counted as "silence" in the probe
    EMBEDDING_VERSION   which vector the embedding view is plotting
    MAX_SALIENCY_SECONDS  how much audio the attribution covers

A stale hit across any of those is the worst kind of bug this task can have:
the response is well-formed, plausible, and quietly answers a different
question than the one asked. These tests pin each knob into its key.

`DEEPFAKE_DATASET_ROOT` is covered too, since every other test in the suite
depends on being able to redirect it.
"""

import importlib
from importlib import import_module

import pytest

from app.core.settings import settings

deepfake_router = import_module("app.tasks.deepfake.router")
deepfake_service = import_module("app.tasks.deepfake.service")
deepfake_evaluation = import_module("app.tasks.deepfake.evaluation")
deepfake_embeddings = import_module("app.tasks.deepfake.embeddings")

PROTOCOL = {
    "LA_E_8000001": ("-", "bonafide"),
    "LA_E_8000002": ("A10", "spoof"),
}


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    for index, file_id in enumerate(PROTOCOL):
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC" + bytes([index]))
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "\n".join(
            f"LA_0069 {file_id} - {system_id} {key}"
            for file_id, (system_id, key) in PROTOCOL.items()
        )
        + "\n"
    )
    return audio_dir


async def _first_recording_id(client) -> str:
    response = await client.get("/tasks/deepfake/dataset/recordings")
    return response.json()["recordings"][0]["recording_id"]


# ---------------------------------------------------------------------------
# Dataset location
# ---------------------------------------------------------------------------


def test_the_dataset_root_setting_redirects_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    assert settings.asvspoof2019_la_dataset_dir == tmp_path / "asvspoof2019_la"

    from app.tasks.deepfake.dataset import get_dataset_info

    # Pointed somewhere empty, the listing reports unavailable rather than
    # falling back to a default location and scoring the wrong corpus.
    assert get_dataset_info()["available"] is False


def test_the_expected_recording_count_is_declared(fake_dataset):
    """The demo subset is built to a fixed size by the prepare script.

    `get_dataset_info` reports both the expected and the actual count, so a
    half-downloaded dataset is visible in the UI instead of silently producing
    an EER over 37 clips.
    """
    from app.tasks.deepfake.dataset import EXPECTED_RECORDING_COUNT, get_dataset_info

    info = get_dataset_info()
    assert EXPECTED_RECORDING_COUNT == 200
    assert info["expected_recording_count"] == 200
    assert info["total_recordings"] == len(PROTOCOL)
    assert info["total_recordings"] != info["expected_recording_count"]


# ---------------------------------------------------------------------------
# Cache-key participation
# ---------------------------------------------------------------------------


async def test_bumping_the_threshold_version_invalidates_cached_decisions(
    client, fake_dataset, monkeypatch
):
    """The threshold version is in the /run key, so v1 cannot read v0's cache."""
    calls = []

    def _run_detection(model_key, audio_path):
        calls.append(str(audio_path))
        return {"spoof_probability": 0.77, "decision": "spoof", "threshold": 0.5}

    monkeypatch.setattr(deepfake_router, "run_detection", _run_detection)
    recording_id = await _first_recording_id(client)
    body = {"model": "xlsr-deepfake", "recording_id": recording_id}

    assert (await client.post("/tasks/deepfake/run", json=body)).json()["cached"] is False
    assert (await client.post("/tasks/deepfake/run", json=body)).json()["cached"] is True
    assert len(calls) == 1

    monkeypatch.setattr(deepfake_router, "THRESHOLD_VERSION", "deepfake-threshold-v1-eer")
    assert (await client.post("/tasks/deepfake/run", json=body)).json()["cached"] is False
    assert len(calls) == 2


async def test_the_silence_threshold_is_part_of_the_probe_cache_key(
    client, fake_dataset, monkeypatch
):
    """Change what counts as silence and the old probe result must not be reused."""
    calls = []

    def _probe(model_key, audio_path):
        calls.append(str(audio_path))
        return {"variants": {}, "silence_top_db": deepfake_router.SILENCE_TOP_DB}

    monkeypatch.setattr(deepfake_router, "run_silence_probe", _probe)
    recording_id = await _first_recording_id(client)
    body = {"model": "xlsr-deepfake", "recording_id": recording_id}

    assert (await client.post("/tasks/deepfake/silence-probe", json=body)).json()["cached"] is False
    assert (await client.post("/tasks/deepfake/silence-probe", json=body)).json()["cached"] is True

    monkeypatch.setattr(deepfake_router, "SILENCE_TOP_DB", 20)
    assert (await client.post("/tasks/deepfake/silence-probe", json=body)).json()["cached"] is False
    assert len(calls) == 2


async def test_the_embedding_version_is_part_of_the_embedding_cache_key(
    client, fake_dataset, monkeypatch
):
    """`head-input-v1` means a specific vector. Read a different layer, bump it."""
    calls = []

    def _run_embedding(model_key, audio_path):
        calls.append(str(audio_path))
        return {"embedding": [0.1, 0.2, 0.3, 0.4], "spoof_probability": 0.6}

    monkeypatch.setattr(deepfake_embeddings, "run_embedding", _run_embedding)
    body = {"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2}

    assert (await client.post("/tasks/deepfake/embeddings", json=body)).status_code == 200
    assert len(calls) == len(PROTOCOL)

    # Same version -> cache hit, no new forward passes.
    assert (await client.post("/tasks/deepfake/embeddings", json=body)).status_code == 200
    assert len(calls) == len(PROTOCOL)

    monkeypatch.setattr(deepfake_embeddings, "EMBEDDING_VERSION", "penultimate-layer-v2")
    assert (await client.post("/tasks/deepfake/embeddings", json=body)).status_code == 200
    assert len(calls) == 2 * len(PROTOCOL)


async def test_the_saliency_method_is_part_of_its_cache_key(client, fake_dataset, monkeypatch):
    """An attribution computed by a different method is a different answer."""
    calls = []

    def _saliency(model_key, audio_path, segment_count=60):
        calls.append(str(audio_path))
        return {"method": deepfake_router.SALIENCY_METHOD, "segments": [], "series": []}

    monkeypatch.setattr(deepfake_router, "generate_saliency", _saliency)
    recording_id = await _first_recording_id(client)
    body = {"model": "xlsr-deepfake", "recording_id": recording_id}

    assert (await client.post("/tasks/deepfake/saliency", json=body)).json()["cached"] is False
    assert (await client.post("/tasks/deepfake/saliency", json=body)).json()["cached"] is True

    monkeypatch.setattr(deepfake_router, "SALIENCY_METHOD", "integrated-gradients")
    assert (await client.post("/tasks/deepfake/saliency", json=body)).json()["cached"] is False
    assert len(calls) == 2


async def test_each_model_gets_its_own_cache_slot(client, fake_dataset, monkeypatch):
    """Model A's score must never be served for Model C."""
    calls = []

    def _run_detection(model_key, audio_path):
        calls.append(model_key)
        return {"spoof_probability": 0.5 if model_key == "xlsr-deepfake" else 0.9}

    monkeypatch.setattr(deepfake_router, "run_detection", _run_detection)
    recording_id = await _first_recording_id(client)

    a = await client.post(
        "/tasks/deepfake/run", json={"model": "xlsr-deepfake", "recording_id": recording_id}
    )
    c = await client.post(
        "/tasks/deepfake/run", json={"model": "xlsr-mamba", "recording_id": recording_id}
    )

    assert a.json()["spoof_probability"] == 0.5
    assert c.json()["spoof_probability"] == 0.9
    assert calls == ["xlsr-deepfake", "xlsr-mamba"]


# ---------------------------------------------------------------------------
# Environment-driven limits
# ---------------------------------------------------------------------------


def test_the_saliency_cap_is_read_from_the_shared_environment_variable(monkeypatch):
    """Same env var and default as the shared saliency service (SRS DF-15).

    Read at import time, so this reloads the module -- which is also what
    proves the value is not hardcoded.
    """
    from app.tasks.deepfake import saliency

    assert saliency.MAX_SALIENCY_SECONDS == 12  # the shared default

    monkeypatch.setenv("MAX_SALIENCY_SECONDS", "4")
    reloaded = importlib.reload(saliency)
    try:
        assert reloaded.MAX_SALIENCY_SECONDS == 4
    finally:
        monkeypatch.delenv("MAX_SALIENCY_SECONDS")
        importlib.reload(saliency)


def test_the_analysis_window_cap_is_declared_not_implied():
    """A truncated score must never be reported as if it saw the whole clip."""
    assert deepfake_service.MAX_ANALYSIS_SECONDS == 30.0
    assert deepfake_service.FRAMES_PER_SECOND == 100.0


def test_every_registered_model_declares_its_threshold_as_uncalibrated():
    """Until Feature 1 produces an EER threshold, every spec must say so.

    The evaluation in Model Research/deepfake_evaluation measured an EER
    threshold of 0.142884 for Model A on the demo subset -- adopting it means
    changing these flags AND bumping THRESHOLD_VERSION, which is why the two
    are tested together.
    """
    for spec in deepfake_service.MODEL_SPECS.values():
        assert spec.threshold == deepfake_service.DEFAULT_THRESHOLD
        assert spec.threshold_calibrated is False
    assert "uncalibrated" in deepfake_service.THRESHOLD_VERSION


def test_no_checkpoint_needs_a_token():
    """Every detector is public: the task must run on a fresh checkout with no HF_TOKEN.

    Model B used to be the gated WpythonW checkpoint; it was replaced by a
    public one precisely so no access request stands between a user and a score.
    """
    gated = {key for key, spec in deepfake_service.MODEL_SPECS.items() if spec.gated}
    assert gated == set()


async def test_an_absent_dataset_is_reported_not_raised(client, tmp_path, monkeypatch):
    """A fresh checkout has no dataset; the UI must still load."""
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path / "nothing-here")

    info = await client.get("/tasks/deepfake/dataset")
    assert info.status_code == 200
    assert info.json()["available"] is False

    # The listing, which promises recordings, is the one that 404s.
    listing = await client.get("/tasks/deepfake/dataset/recordings")
    assert listing.status_code == 404
