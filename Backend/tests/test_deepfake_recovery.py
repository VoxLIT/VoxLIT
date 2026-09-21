"""Failover and recovery tests for /tasks/deepfake (Test Plan 3.1.7).

The premise of every cached route in this task is that the cache is an
OPTIMISATION: `/run` caches a score because a forward pass is expensive, not
because the result is unobtainable without it. So the behaviour worth testing
is what happens when the cache is not there, is broken, or holds something
that cannot be read back.

Two of the tests below are `xfail(strict=True)`. They are not aspirational
padding -- they encode the behaviour the design implies, they fail today, and
the reasons are written up in the test plan's Appendix A. The fix belongs in
`app/core/redis.py`, which this task does not own, so they are recorded here
rather than silently worked around.

The rest of the module covers what the task DOES get right: a model that fails
to load is not cached as a failure, an interrupted evaluation resumes from the
per-clip scores it already has, and `/scores` shares those scores with `/run`.
"""

from importlib import import_module

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.settings import settings

deepfake_router = import_module("app.tasks.deepfake.router")
deepfake_evaluation = import_module("app.tasks.deepfake.evaluation")
deepfake_service = import_module("app.tasks.deepfake.service")

PROTOCOL = {
    "LA_E_7000001": ("-", "bonafide"),
    "LA_E_7000002": ("-", "bonafide"),
    "LA_E_7000003": ("A10", "spoof"),
    "LA_E_7000004": ("A19", "spoof"),
}

SCORES = {
    "LA_E_7000001": 0.03,
    "LA_E_7000002": 0.06,
    "LA_E_7000003": 0.94,
    "LA_E_7000004": 0.89,
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


def _payload(stem: str) -> dict:
    score = SCORES[stem]
    return {
        "model": "xlsr-deepfake",
        "model_label": "wav2vec2 XLS-R (Model A)",
        "decision": "spoof" if score >= 0.5 else "bonafide",
        "threshold": 0.5,
        "threshold_calibrated": False,
        "spoof_probability": score,
        "bonafide_probability": round(1 - score, 6),
        "duration": 3.1,
    }


@pytest.fixture
def stub_detection(monkeypatch):
    calls = []

    def _run_detection(model_key, audio_path):
        from pathlib import Path

        stem = Path(audio_path).stem
        calls.append(stem)
        return _payload(stem)

    monkeypatch.setattr(deepfake_router, "run_detection", _run_detection)
    monkeypatch.setattr(deepfake_evaluation, "run_detection", _run_detection)
    return calls


async def _first_recording_id(client) -> str:
    response = await client.get("/tasks/deepfake/dataset/recordings")
    return response.json()["recordings"][0]["recording_id"]


def _break_cache(monkeypatch, module, error: Exception):
    async def _raise(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(module, "get_result", _raise)
    monkeypatch.setattr(module, "cache_result", _raise)


# ---------------------------------------------------------------------------
# Cache unavailable
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Appendix A finding 1: a Redis outage propagates out of get_result as a "
        "500. The cache is an optimisation, so /run should degrade to an "
        "uncached score. Fix belongs in app/core/redis.py, not owned here."
    ),
)
async def test_run_degrades_to_an_uncached_score_when_redis_is_down(
    client, fake_dataset, stub_detection, monkeypatch
):
    recording_id = await _first_recording_id(client)
    _break_cache(monkeypatch, deepfake_router, RedisConnectionError("connection refused"))

    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "xlsr-deepfake", "recording_id": recording_id},
    )

    assert response.status_code == 200
    assert response.json()["cached"] is False
    assert len(stub_detection) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Appendix A finding 2: an unreadable cache entry raises JSONDecodeError "
        "out of get_result, and the bad value survives until its TTL -- so the "
        "clip stays un-scoreable for a week. Should recompute instead."
    ),
)
async def test_a_corrupt_cache_entry_is_recomputed_not_fatal(
    client, fake_dataset, stub_detection
):
    from app.core import redis as redis_module
    from app.tasks.deepfake.dataset import resolve_recording_path
    from app.tasks.deepfake.service import THRESHOLD_VERSION, file_sha256

    recording_id = await _first_recording_id(client)
    audio_hash = file_sha256(resolve_recording_path(recording_id))
    await redis_module.redis.set(
        f"result:df:xlsr-deepfake:{THRESHOLD_VERSION}:{audio_hash}", "{ truncated json"
    )

    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "xlsr-deepfake", "recording_id": recording_id},
    )

    assert response.status_code == 200
    assert response.json()["spoof_probability"] == SCORES["LA_E_7000001"]


async def test_the_failure_mode_is_a_500_today_not_a_wrong_answer(
    client, fake_dataset, stub_detection, monkeypatch
):
    """Pins the CURRENT behaviour of the two xfails above.

    An outage failing loudly is a bug, but it is a much smaller bug than an
    outage returning a plausible wrong score. This test exists so that if the
    cache path is ever changed, a silent-wrong-answer regression cannot slip
    past under cover of the xfails.
    """
    recording_id = await _first_recording_id(client)
    _break_cache(monkeypatch, deepfake_router, RedisConnectionError("connection refused"))

    with pytest.raises(RedisConnectionError):
        await client.post(
            "/tasks/deepfake/run",
            json={"model": "xlsr-deepfake", "recording_id": recording_id},
        )

    # Nothing was scored, so nothing wrong was reported.
    assert stub_detection == []


# ---------------------------------------------------------------------------
# Model load failure
# ---------------------------------------------------------------------------


async def test_a_failed_model_load_is_not_cached_as_a_failure(monkeypatch):
    """A transient load failure must not poison `_MODEL_CACHE` forever.

    `get_model` only writes the cache after the adapter is constructed, so a
    raise leaves the slot empty and the next request retries. This pins that,
    because the natural "cache the result of the load" refactor would break it
    and the symptom -- a permanently 503'd model until restart -- would look
    like a network problem, not a code problem.
    """
    attempts = []

    class _FlakyAdapter:
        def __init__(self, spec):
            attempts.append(spec.key)
            if len(attempts) == 1:
                raise deepfake_service.DeepfakeModelUnavailable("transient network error")

        def score(self, audio_path, with_embedding=False):
            return {"spoof_probability": 0.5}

    monkeypatch.setattr(deepfake_service, "_HFAudioClassifierAdapter", _FlakyAdapter)
    monkeypatch.setattr(deepfake_service, "_MODEL_CACHE", {})

    with pytest.raises(deepfake_service.DeepfakeModelUnavailable):
        deepfake_service.get_model("xlsr-deepfake")

    # Second call retries rather than replaying the failure.
    adapter = deepfake_service.get_model("xlsr-deepfake")
    assert isinstance(adapter, _FlakyAdapter)
    assert attempts == ["xlsr-deepfake", "xlsr-deepfake"]


async def test_an_unloadable_model_surfaces_as_503_on_every_route(
    client, fake_dataset, monkeypatch
):
    """503, not 500: the model is a dependency, and the caller can retry."""

    def _explode(*_args, **_kwargs):
        raise deepfake_service.DeepfakeModelUnavailable("checkpoint unavailable")

    monkeypatch.setattr(deepfake_router, "run_detection", _explode)
    monkeypatch.setattr(deepfake_evaluation, "run_detection", _explode)
    monkeypatch.setattr(deepfake_router, "run_silence_probe", _explode)
    monkeypatch.setattr(deepfake_router, "generate_saliency", _explode)

    recording_id = await _first_recording_id(client)
    for route in ("/tasks/deepfake/run", "/tasks/deepfake/silence-probe", "/tasks/deepfake/saliency"):
        response = await client.post(
            route, json={"model": "xlsr-deepfake", "recording_id": recording_id}
        )
        assert response.status_code == 503, route

    response = await client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Resuming interrupted work
# ---------------------------------------------------------------------------


async def test_an_interrupted_evaluation_resumes_from_cached_clips(
    client, fake_dataset, stub_detection, monkeypatch
):
    """A /scores run that dies part-way must not re-score from scratch.

    Every clip is cached the moment it is scored, so a resumed run only pays
    for what it did not finish. On the real 200-clip subset that is the
    difference between 33 seconds and nothing at all.
    """
    from pathlib import Path

    original = deepfake_evaluation.run_detection

    def _die_on_the_third_clip(model_key, audio_path):
        if len(stub_detection) >= 2:
            raise RuntimeError("worker killed")
        return original(model_key, audio_path)

    monkeypatch.setattr(deepfake_evaluation, "run_detection", _die_on_the_third_clip)
    with pytest.raises(RuntimeError):
        await client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})

    scored_before_the_crash = list(stub_detection)
    assert len(scored_before_the_crash) == 2

    monkeypatch.setattr(deepfake_evaluation, "run_detection", original)
    response = await client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})

    assert response.status_code == 200
    assert response.json()["scored"] == len(PROTOCOL)
    # The two clips scored before the crash came from the cache this time.
    rescored = stub_detection[len(scored_before_the_crash):]
    assert set(rescored).isdisjoint(set(scored_before_the_crash))
    assert len(rescored) == len(PROTOCOL) - 2


async def test_evaluation_and_run_share_one_cache_entry(
    client, fake_dataset, stub_detection
):
    """Warming either path warms the other -- same key, same payload."""
    evaluation = await client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})
    assert evaluation.status_code == 200
    assert len(stub_detection) == len(PROTOCOL)

    recording_id = await _first_recording_id(client)
    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "xlsr-deepfake", "recording_id": recording_id},
    )

    assert response.status_code == 200
    assert response.json()["cached"] is True
    # No extra forward pass: /run read what /scores left behind.
    assert len(stub_detection) == len(PROTOCOL)
