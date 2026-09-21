"""Load and concurrency tests for /tasks/deepfake (Test Plan 3.1.5).

Load testing a deepfake route against REAL inference would measure the CPU,
not the code: a forward pass is 0.17-0.25 s and there is one model instance
shared by every request. So what is tested here is the concurrency machinery
that exists precisely because inference is expensive and shared:

    service._LOAD_LOCK       one model load, however many requests race
    embeddings._SCORING_LOCKS   one dataset scoring pass per model
    run_in_threadpool        the event loop stays free while a model runs

The first of those guards a failure mode that does not announce itself.
transformers' weight loading is not thread-safe, and a concurrent load
returns CORRUPTED WEIGHTS rather than raising -- a detector that is quietly
wrong, with no error anywhere. The lock is the only thing standing between
this task and that, so it gets a test that actually races.
"""

import asyncio
import statistics
import threading
import time
from importlib import import_module

import pytest

from app.core.settings import settings

deepfake_router = import_module("app.tasks.deepfake.router")
deepfake_service = import_module("app.tasks.deepfake.service")
deepfake_embeddings = import_module("app.tasks.deepfake.embeddings")

CONCURRENT_CALLERS = 5
DATASET_SIZE = 20


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    protocol = []
    for index in range(DATASET_SIZE):
        file_id = f"LA_E_95{index:05d}"
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC" + index.to_bytes(4, "big") * 16)
        label = "bonafide" if index % 2 == 0 else "spoof"
        protocol.append(f"LA_0069 {file_id} - {'-' if index % 2 == 0 else 'A10'} {label}")
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text("\n".join(protocol) + "\n")
    return audio_dir


async def _recording_ids(client) -> list[str]:
    response = await client.get("/tasks/deepfake/dataset/recordings")
    return [r["recording_id"] for r in response.json()["recordings"]]


# ---------------------------------------------------------------------------
# Model loading under a race
# ---------------------------------------------------------------------------


@pytest.mark.performance
def test_concurrent_first_requests_load_the_model_exactly_once(monkeypatch):
    """The race `_LOAD_LOCK` exists for, run for real across threads.

    A barrier releases all five threads into `get_model` at the same instant,
    and the adapter constructor then takes 50 ms -- long enough that an
    unlocked implementation would have every thread inside it at once. With
    the lock, exactly one construction happens and the other four get the
    instance it produced.

    The barrier sits OUTSIDE the constructor deliberately: putting it inside
    would deadlock against the very lock under test, which is a neat
    demonstration that the lock is real but a useless assertion.
    """
    constructions = []
    start_line = threading.Barrier(CONCURRENT_CALLERS)

    class _SlowAdapter:
        def __init__(self, spec):
            time.sleep(0.05)
            constructions.append(spec.key)

        def score(self, audio_path, with_embedding=False):
            return {"spoof_probability": 0.5}

    monkeypatch.setattr(deepfake_service, "_HFAudioClassifierAdapter", _SlowAdapter)
    monkeypatch.setattr(deepfake_service, "_MODEL_CACHE", {})

    adapters = []
    errors = []

    def _call():
        try:
            start_line.wait(timeout=5)
            adapters.append(deepfake_service.get_model("xlsr-deepfake"))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=_call) for _ in range(CONCURRENT_CALLERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == [], errors
    assert len(constructions) == 1, f"model loaded {len(constructions)} times"
    assert len(adapters) == CONCURRENT_CALLERS
    # All five callers hold the SAME adapter -- not five copies of the weights.
    assert len({id(adapter) for adapter in adapters}) == 1


@pytest.mark.performance
async def test_concurrent_run_requests_all_succeed(client, fake_dataset, monkeypatch):
    """Five callers, five correct answers, no interleaving damage."""
    from pathlib import Path

    def _run_detection(model_key, audio_path):
        time.sleep(0.01)  # stand in for a forward pass
        index = int(Path(audio_path).stem[-5:])
        return {"spoof_probability": index / 100.0, "decision": "bonafide"}

    monkeypatch.setattr(deepfake_router, "run_detection", _run_detection)
    ids = (await _recording_ids(client))[:CONCURRENT_CALLERS]

    started = time.perf_counter()
    responses = await asyncio.gather(
        *(
            client.post(
                "/tasks/deepfake/run",
                json={"model": "xlsr-deepfake", "recording_id": recording_id},
            )
            for recording_id in ids
        )
    )
    elapsed = time.perf_counter() - started

    assert all(response.status_code == 200 for response in responses)
    # Each caller got ITS clip's score, not a neighbour's.
    scores = [response.json()["spoof_probability"] for response in responses]
    assert len(set(scores)) == CONCURRENT_CALLERS
    # run_in_threadpool means these overlap rather than queueing end to end.
    assert elapsed < 0.01 * CONCURRENT_CALLERS * 2


# ---------------------------------------------------------------------------
# Dataset scoring under a race
# ---------------------------------------------------------------------------


@pytest.mark.performance
async def test_concurrent_projections_score_the_dataset_once(
    client, fake_dataset, monkeypatch
):
    """Flipping PCA -> UMAP mid-run must not start a second scoring pass.

    `_SCORING_LOCKS` holds the second request until the first has filled the
    cache; it then reads the cache. Without it, a user who changes the method
    during the slow first run pays for the whole dataset twice.
    """
    calls = []

    def _run_embedding(model_key, audio_path):
        calls.append(str(audio_path))
        time.sleep(0.005)
        return {"embedding": [0.1, 0.2, 0.3, 0.4], "spoof_probability": 0.5}

    monkeypatch.setattr(deepfake_embeddings, "run_embedding", _run_embedding)
    # A fresh lock per test: the module-level dict persists across tests.
    monkeypatch.setattr(deepfake_embeddings, "_SCORING_LOCKS", {})

    responses = await asyncio.gather(
        client.post(
            "/tasks/deepfake/embeddings",
            json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2},
        ),
        client.post(
            "/tasks/deepfake/embeddings",
            json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 3},
        ),
    )

    assert all(response.status_code == 200 for response in responses)
    assert len(calls) == DATASET_SIZE, f"scored {len(calls)} times, expected {DATASET_SIZE}"


@pytest.mark.performance
async def test_different_models_are_not_serialised_behind_one_another(
    client, fake_dataset, monkeypatch
):
    """The scoring lock is per model, so Model C does not wait for Model A."""
    monkeypatch.setattr(deepfake_embeddings, "_SCORING_LOCKS", {})
    seen = []

    def _run_embedding(model_key, audio_path):
        seen.append(model_key)
        return {"embedding": [0.1, 0.2, 0.3, 0.4], "spoof_probability": 0.5}

    monkeypatch.setattr(deepfake_embeddings, "run_embedding", _run_embedding)

    responses = await asyncio.gather(
        client.post(
            "/tasks/deepfake/embeddings",
            json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2},
        ),
        client.post(
            "/tasks/deepfake/embeddings",
            json={"model": "xlsr-mamba", "reduction_method": "pca", "n_components": 2},
        ),
    )

    assert all(response.status_code == 200 for response in responses)
    assert set(seen) == {"xlsr-deepfake", "xlsr-mamba"}
    # Each model scored the dataset for itself -- no cross-model cache reuse.
    assert seen.count("xlsr-deepfake") == DATASET_SIZE
    assert seen.count("xlsr-mamba") == DATASET_SIZE


# ---------------------------------------------------------------------------
# Cache throughput
# ---------------------------------------------------------------------------


@pytest.mark.performance
async def test_cache_hit_throughput_under_concurrency(
    client, fake_dataset, monkeypatch, capsys
):
    """Warm reads are the common case once a workbench session is going."""
    monkeypatch.setattr(
        deepfake_router,
        "run_detection",
        lambda model_key, audio_path: {"spoof_probability": 0.42, "decision": "bonafide"},
    )
    ids = await _recording_ids(client)
    for recording_id in ids:
        await client.post(
            "/tasks/deepfake/run",
            json={"model": "xlsr-deepfake", "recording_id": recording_id},
        )

    rounds = 5
    durations = []
    for _ in range(rounds):
        started = time.perf_counter()
        responses = await asyncio.gather(
            *(
                client.post(
                    "/tasks/deepfake/run",
                    json={"model": "xlsr-deepfake", "recording_id": recording_id},
                )
                for recording_id in ids
            )
        )
        durations.append(time.perf_counter() - started)
        assert all(response.json()["cached"] is True for response in responses)

    median = statistics.median(durations)
    with capsys.disabled():
        print(
            f"\n  {DATASET_SIZE} concurrent cache hits: median {median * 1000:.1f} ms "
            f"({DATASET_SIZE / median:.0f} req/s), {rounds} rounds"
        )
    assert median < 1.0
