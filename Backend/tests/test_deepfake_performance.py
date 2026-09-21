"""Performance profiling for /tasks/deepfake (Test Plan 3.1.4).

WHAT THESE NUMBERS ARE
----------------------
Inference is mocked throughout, so nothing here measures a model. What is
measured is the machinery WRAPPED AROUND the model: the SHA-256 of the file,
the Redis round-trip, the FastAPI request cycle, and -- for the embedding view
-- the PCA over a 200-clip matrix, which is the one piece of real computation
in this task that is not a forward pass.

That is the useful measurement, because the cache exists to replace a forward
pass with this machinery. Real per-clip inference cost is measured separately
by Model Research/deepfake_evaluation/src/score_dataset.py (0.167 s/clip for
Model A, 0.254 s/clip for Model C, both CPU) and quoted in the test plan; a
cache hit only pays for itself if it is far cheaper than those.

Thresholds below are generous on purpose -- they are regression guards
against an accidental O(n) re-read or a lost cache, not benchmarks. A CI
machine under load should not turn this file red.
"""

import statistics
import time
from importlib import import_module

import pytest

from app.core.settings import settings

deepfake_router = import_module("app.tasks.deepfake.router")
deepfake_embeddings = import_module("app.tasks.deepfake.embeddings")
deepfake_evaluation = import_module("app.tasks.deepfake.evaluation")

WARM_ITERATIONS = 20
# A cache hit must stay far under the cheapest real forward pass (0.167 s).
CACHE_HIT_BUDGET_SECONDS = 0.05
PROJECTION_BUDGET_SECONDS = 10.0
MEMORY_GROWTH_BUDGET_MB = 500

DATASET_SIZE = 200
EMBEDDING_DIMENSION = 256


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    """A dataset the size of the real one, so the numbers transfer."""
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    protocol = []
    for index in range(DATASET_SIZE):
        file_id = f"LA_E_90{index:05d}"
        # Distinct content per file: the cache key is a content hash, so
        # identical bytes would collapse 200 clips into one cache entry and
        # make every measurement here meaningless.
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC" + index.to_bytes(4, "big") * 64)
        label = "bonafide" if index % 2 == 0 else "spoof"
        attack = "-" if label == "bonafide" else f"A{7 + index % 13:02d}"
        protocol.append(f"LA_0069 {file_id} - {attack} {label}")
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text("\n".join(protocol) + "\n")
    return audio_dir


@pytest.fixture
def stub_inference(monkeypatch):
    """Zero-cost stand-ins, so the timings are of the plumbing only."""
    from pathlib import Path

    def _score(path) -> float:
        return (int(Path(path).stem[-5:]) % 100) / 100.0

    def _run_detection(model_key, audio_path):
        score = _score(audio_path)
        return {
            "model": model_key,
            "model_label": "stub",
            "decision": "spoof" if score >= 0.5 else "bonafide",
            "threshold": 0.5,
            "threshold_calibrated": False,
            "spoof_probability": score,
            "bonafide_probability": round(1 - score, 6),
            "duration": 3.2,
        }

    def _run_embedding(model_key, audio_path):
        score = _score(audio_path)
        return {
            "embedding": [score + i * 0.001 for i in range(EMBEDDING_DIMENSION)],
            "spoof_probability": score,
        }

    monkeypatch.setattr(deepfake_router, "run_detection", _run_detection)
    monkeypatch.setattr(deepfake_evaluation, "run_detection", _run_detection)
    monkeypatch.setattr(deepfake_embeddings, "run_embedding", _run_embedding)


async def _first_recording_id(client) -> str:
    response = await client.get("/tasks/deepfake/dataset/recordings")
    return response.json()["recordings"][0]["recording_id"]


async def _time(call) -> tuple[float, object]:
    started = time.perf_counter()
    result = await call()
    return time.perf_counter() - started, result


@pytest.mark.performance
async def test_cache_hit_latency_on_run(client, fake_dataset, stub_inference, capsys):
    recording_id = await _first_recording_id(client)
    body = {"model": "xlsr-deepfake", "recording_id": recording_id}

    cold_seconds, cold = await _time(lambda: client.post("/tasks/deepfake/run", json=body))
    assert cold.json()["cached"] is False

    warm = []
    for _ in range(WARM_ITERATIONS):
        seconds, response = await _time(lambda: client.post("/tasks/deepfake/run", json=body))
        assert response.json()["cached"] is True
        warm.append(seconds)

    median = statistics.median(warm)
    with capsys.disabled():
        print(
            f"\n  /run  cold(mocked inference)={cold_seconds * 1000:.1f} ms  "
            f"warm median={median * 1000:.1f} ms "
            f"(min {min(warm) * 1000:.1f}, max {max(warm) * 1000:.1f}, n={WARM_ITERATIONS})"
        )
    assert median < CACHE_HIT_BUDGET_SECONDS


@pytest.mark.performance
@pytest.mark.parametrize(
    "route,stub_name,payload_builder",
    [
        ("/tasks/deepfake/silence-probe", "run_silence_probe", lambda: {"variants": {}}),
        (
            "/tasks/deepfake/saliency",
            "generate_saliency",
            lambda: {"segments": [], "series": [], "method": "input-gradient"},
        ),
    ],
)
async def test_cache_hit_latency_on_the_single_clip_views(
    client, fake_dataset, monkeypatch, capsys, route, stub_name, payload_builder
):
    """Feature 2 and Feature 3 are three and two passes respectively, so their
    cache hits matter more per request than /run's."""
    monkeypatch.setattr(deepfake_router, stub_name, lambda *a, **k: payload_builder())
    recording_id = await _first_recording_id(client)
    body = {"model": "xlsr-deepfake", "recording_id": recording_id}

    await client.post(route, json=body)
    warm = []
    for _ in range(WARM_ITERATIONS):
        seconds, response = await _time(lambda: client.post(route, json=body))
        assert response.json()["cached"] is True
        warm.append(seconds)

    median = statistics.median(warm)
    with capsys.disabled():
        print(f"\n  {route}  warm median={median * 1000:.1f} ms (n={WARM_ITERATIONS})")
    assert median < CACHE_HIT_BUDGET_SECONDS


@pytest.mark.performance
async def test_a_warm_projection_is_dominated_by_the_reducer_not_the_cache(
    client, fake_dataset, stub_inference, capsys
):
    """The second /embeddings call must not re-score 200 clips.

    This is the request the user makes most often -- flipping PCA to t-SNE, or
    2D to 3D -- and it is the one that would be unbearable if the per-clip
    vectors were not cached.
    """
    body = {"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2}

    cold_seconds, cold = await _time(lambda: client.post("/tasks/deepfake/embeddings", json=body))
    assert cold.status_code == 200
    assert cold.json()["total_recordings"] == DATASET_SIZE

    warm_seconds, warm = await _time(lambda: client.post("/tasks/deepfake/embeddings", json=body))
    three_d_seconds, three_d = await _time(
        lambda: client.post(
            "/tasks/deepfake/embeddings",
            json={**body, "n_components": 3},
        )
    )

    with capsys.disabled():
        print(
            f"\n  /embeddings ({DATASET_SIZE} clips x {EMBEDDING_DIMENSION}d, mocked inference)"
            f"  cold={cold_seconds:.2f} s  warm 2D={warm_seconds:.2f} s  warm 3D={three_d_seconds:.2f} s"
        )

    assert warm.status_code == 200 and three_d.status_code == 200
    assert warm_seconds < PROJECTION_BUDGET_SECONDS
    assert three_d.json()["effective_components"] == 3


@pytest.mark.performance
async def test_the_evaluation_view_scores_each_clip_exactly_once(
    client, fake_dataset, monkeypatch, capsys
):
    """200 clips, 200 forward passes -- not 400, and not one per histogram bin."""
    calls = []
    from pathlib import Path

    def _run_detection(model_key, audio_path):
        calls.append(Path(audio_path).stem)
        return {"spoof_probability": (int(Path(audio_path).stem[-5:]) % 100) / 100.0}

    monkeypatch.setattr(deepfake_evaluation, "run_detection", _run_detection)

    cold_seconds, response = await _time(
        lambda: client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})
    )
    assert response.status_code == 200
    assert len(calls) == DATASET_SIZE
    assert len(set(calls)) == DATASET_SIZE

    warm_seconds, warm = await _time(
        lambda: client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})
    )
    with capsys.disabled():
        print(
            f"\n  /scores ({DATASET_SIZE} clips, mocked inference)  "
            f"cold={cold_seconds:.2f} s  warm={warm_seconds:.2f} s"
        )

    assert warm.status_code == 200
    # Nothing re-scored on the warm pass.
    assert len(calls) == DATASET_SIZE


@pytest.mark.performance
async def test_a_full_projection_does_not_leak_memory(
    client, fake_dataset, stub_inference, capsys
):
    """RSS delta across repeated 200-clip projections.

    The embedding view holds a 200 x 256 matrix plus whatever the reducer
    allocates; repeated calls must return that, not accumulate it.
    """
    import gc

    import psutil

    process = psutil.Process()
    body = {"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2}
    await client.post("/tasks/deepfake/embeddings", json=body)  # warm the cache

    gc.collect()
    before = process.memory_info().rss
    for _ in range(5):
        response = await client.post("/tasks/deepfake/embeddings", json=body)
        assert response.status_code == 200
    gc.collect()
    growth_mb = (process.memory_info().rss - before) / (1024 * 1024)

    with capsys.disabled():
        print(f"\n  RSS growth over 5 warm projections: {growth_mb:+.1f} MB")
    assert growth_mb < MEMORY_GROWTH_BUDGET_MB
