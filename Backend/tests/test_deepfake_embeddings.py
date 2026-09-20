"""Tests for the deepfake task's 2D/3D embedding view (Feature 4).

`run_embedding` is monkeypatched throughout — these tests must never download
a checkpoint. What is being tested is the head-input capture, the cache-through
behaviour, the request validation and the ground-truth safety of the response.
"""

import threading
from importlib import import_module

import pytest
import torch
from torch import nn

from app.core.settings import settings
from app.tasks.deepfake import service

# See test_deepfake_router.py: the package attribute `router` is the APIRouter,
# not the module.
embeddings_module = import_module("app.tasks.deepfake.embeddings")

FILE_IDS = ["LA_E_2000001", "LA_E_2000002", "LA_E_2000003", "LA_E_2000004"]
EMBEDDING_DIM = 8


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    for index, file_id in enumerate(FILE_IDS):
        # Distinct bytes -> distinct file hashes -> distinct cache entries.
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC-fake-audio-%d" % index)
    # Ground truth lives on disk exactly as in production; the endpoint must
    # never surface it.
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "\n".join(
            f"LA_0069 {file_id} - {'A10' if index % 2 else '-'} "
            f"{'spoof' if index % 2 else 'bonafide'}"
            for index, file_id in enumerate(FILE_IDS)
        )
        + "\n"
    )


@pytest.fixture
def stub_embedding(monkeypatch):
    """Deterministic per-clip vector and score; records every model call."""
    calls = []

    def _fake_run_embedding(model_key, audio_path):
        calls.append((model_key, str(audio_path)))
        seed = int(str(audio_path).rsplit("LA_E_", 1)[1].split(".")[0])
        generator = torch.Generator().manual_seed(seed)
        vector = torch.randn(EMBEDDING_DIM, generator=generator).tolist()
        return {"embedding": vector, "spoof_probability": (seed % 10) / 10}

    monkeypatch.setattr(embeddings_module, "run_embedding", _fake_run_embedding)
    return calls


# --- head-input capture ---------------------------------------------------


def _tiny_classifier():
    body = nn.Linear(4, 3)
    head = nn.Linear(3, 2)
    return body, head


def test_capture_head_input_returns_what_the_head_read():
    body, head = _tiny_classifier()
    features = body(torch.ones(1, 4))

    with service._capture_head_input(head) as captured:
        head(features)

    assert torch.equal(captured["vector"], features)


def test_capture_head_input_removes_its_hook_afterwards():
    _, head = _tiny_classifier()

    with service._capture_head_input(head):
        pass

    assert len(head._forward_pre_hooks) == 0


def test_capture_head_input_without_a_head_is_a_no_op():
    with service._capture_head_input(None) as captured:
        pass

    assert captured == {}


def test_capture_head_input_ignores_other_threads():
    """A plain score() on the shared model must not leak into this capture."""
    _, head = _tiny_classifier()
    mine = torch.zeros(1, 3)
    theirs = torch.ones(1, 3)

    with service._capture_head_input(head) as captured:
        other = threading.Thread(target=lambda: head(theirs))
        other.start()
        other.join()
        head(mine)

    assert torch.equal(captured["vector"], mine)


def test_missing_capture_is_reported_not_returned_as_an_empty_embedding():
    spec = service.get_model_spec("xlsr-deepfake")

    with pytest.raises(service.DeepfakeModelUnavailable):
        service._embedding_from_capture({}, spec)


# --- endpoint --------------------------------------------------------------


async def test_embeddings_returns_aligned_coordinates_and_scores(
    client, fake_dataset, stub_embedding
):
    response = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_recordings"] == len(FILE_IDS)
    assert len(body["recordings"]) == len(FILE_IDS)
    assert len(body["coordinates"]) == len(FILE_IDS)
    assert all(len(row) == 3 for row in body["coordinates"])
    assert body["embedding_dimension"] == EMBEDDING_DIM
    assert body["n_components"] == 3
    assert body["reduction_method_used"] == "pca"

    names = [row["display_filename"] for row in body["recordings"]]
    assert names == sorted(names)  # same order the listing endpoint uses
    for row in body["recordings"]:
        assert set(row) == {
            "recording_id",
            "display_filename",
            "spoof_probability",
            "decision",
        }
        expected = "spoof" if row["spoof_probability"] >= body["threshold"] else "bonafide"
        assert row["decision"] == expected


async def test_embeddings_response_leaks_no_ground_truth(
    client, fake_dataset, stub_embedding
):
    response = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "n_components": 2},
    )

    body = response.json()
    serialized = str(body)
    # No attack ids, no protocol fields; `decision` is the MODEL's opinion and
    # is derived from its own score, not from the protocol's key.
    assert "A10" not in serialized
    assert "protocol" not in serialized
    assert "attack" not in body
    assert all("label" not in row and "key" not in row for row in body["recordings"])
    assert "embedding" not in serialized.replace("embedding_dimension", "")


async def test_embeddings_are_cached_per_clip_across_methods_and_dimensions(
    client, fake_dataset, stub_embedding
):
    first = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2},
    )
    assert first.status_code == 200
    assert len(stub_embedding) == len(FILE_IDS)

    for method, n_components in [("pca", 3), ("tsne", 2), ("umap", 3)]:
        response = await client.post(
            "/tasks/deepfake/embeddings",
            json={
                "model": "xlsr-deepfake",
                "reduction_method": method,
                "n_components": n_components,
            },
        )
        assert response.status_code == 200
        assert all(len(row) == n_components for row in response.json()["coordinates"])

    # Only the first request ever ran the model.
    assert len(stub_embedding) == len(FILE_IDS)


async def test_embeddings_are_cached_separately_per_model(
    client, fake_dataset, stub_embedding
):
    for model in ["xlsr-deepfake", "xlsr-mamba"]:
        response = await client.post(
            "/tasks/deepfake/embeddings", json={"model": model, "n_components": 2}
        )
        assert response.status_code == 200

    assert len(stub_embedding) == 2 * len(FILE_IDS)


async def test_embeddings_rejects_unknown_model(client, fake_dataset, stub_embedding):
    response = await client.post(
        "/tasks/deepfake/embeddings", json={"model": "nope", "n_components": 2}
    )

    assert response.status_code == 400
    assert stub_embedding == []


async def test_embeddings_rejects_unknown_reduction_method(
    client, fake_dataset, stub_embedding
):
    response = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "reduction_method": "isomap"},
    )

    assert response.status_code == 400
    assert stub_embedding == []


@pytest.mark.parametrize("n_components", [1, 4])
async def test_embeddings_rejects_unsupported_dimensionality(
    client, fake_dataset, stub_embedding, n_components
):
    response = await client.post(
        "/tasks/deepfake/embeddings",
        json={"model": "xlsr-deepfake", "n_components": n_components},
    )

    assert response.status_code == 422
    assert stub_embedding == []


async def test_embeddings_404s_when_dataset_absent(
    client, monkeypatch, tmp_path, stub_embedding
):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path / "nowhere")

    response = await client.post(
        "/tasks/deepfake/embeddings", json={"model": "xlsr-deepfake", "n_components": 2}
    )

    assert response.status_code == 404


async def test_embeddings_maps_model_load_failure_to_503(
    client, fake_dataset, monkeypatch
):
    def _unavailable(model_key, audio_path):
        raise service.DeepfakeModelUnavailable("could not load")

    monkeypatch.setattr(embeddings_module, "run_embedding", _unavailable)

    response = await client.post(
        "/tasks/deepfake/embeddings", json={"model": "xlsr-deepfake", "n_components": 2}
    )

    assert response.status_code == 503
    assert "could not load" in response.json()["detail"]


# --- edge cases: tiny and empty datasets ----------------------------------


@pytest.mark.parametrize("method", ["pca", "tsne", "umap"])
@pytest.mark.parametrize("n_components", [2, 3])
def test_projection_of_a_single_recording_never_raises(method, n_components):
    """One clip has no perplexity (t-SNE) or neighbours (UMAP); both fall back to PCA."""
    from app.tasks.deepfake.projection import reduce_embedding_matrix
    import numpy as np

    matrix = np.random.default_rng(0).normal(size=(1, EMBEDDING_DIM))

    coordinates, effective, used = reduce_embedding_matrix(matrix, method, n_components)

    assert coordinates.shape == (1, n_components)
    assert used == "pca"
    assert effective <= n_components


async def test_embeddings_reports_a_dataset_with_no_recordings(
    client, monkeypatch, tmp_path, stub_embedding
):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    (tmp_path / "asvspoof2019_la" / "flac").mkdir(parents=True)

    response = await client.post(
        "/tasks/deepfake/embeddings", json={"model": "xlsr-deepfake", "n_components": 2}
    )

    assert response.status_code == 404
    assert stub_embedding == []
