"""Focused tests for Integrated Gradients saliency in Speaker Verification:
`service.integrated_gradients_saliency`, the
`saliency_method="integrated_gradients"` branch of
`service.compute_saliency_map`, and the matching
`/tasks/verification/explain/saliency` form parameter.

Math-level tests run against synthetic WAV files with a small differentiable
fake adapter, so no real model is loaded. The real-model gradient-path checks
at the bottom are opt-in integration tests (`VOXLIT_RUN_SV_MODEL_TESTS=1`,
locally cached weights only).
"""

import io
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from app.tasks.verification import service


SAMPLE_RATE = 16000
MEL_COUNT = 80


def _voice_like_wav_bytes(seconds: float, *, seed: int = 0) -> bytes:
    """Tones plus a little noise, with a loudness ramp so frames differ."""

    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    audio = sum(
        amplitude * np.sin(2 * np.pi * freq * t)
        for freq, amplitude in {180.0: 0.3, 900.0: 0.2, 2400.0: 0.15, 5200.0: 0.1}.items()
    )
    audio = audio * np.linspace(0.2, 1.0, t.shape[0]) + 0.01 * rng.standard_normal(t.shape[0])
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


class _DifferentiableAdapter(service._BaseAdapter):
    """Small smooth model with the same gradient-path contract as the real
    adapters: log-mel features [time, mel] -> 4-dim embedding."""

    feature_hz_range = (0.0, 8000.0)

    def __init__(self):
        super().__init__(expected_dimension=4)
        self.device = torch.device("cpu")
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=MEL_COUNT
        )
        generator = torch.Generator().manual_seed(7)
        self._projection = torch.randn(MEL_COUNT, 4, generator=generator) / MEL_COUNT
        self._bias = torch.tensor([0.5, -0.25, 0.1, 0.3])

    def features(self, waveform):
        with torch.no_grad():
            return torch.log(self._mel(waveform)[0].T + 1e-6)

    def embed_from_features(self, features):
        batch = features.unsqueeze(0) if features.dim() == 2 else features
        embeddings = torch.tanh(batch @ self._projection).mean(dim=1) + self._bias
        return embeddings[0] if features.dim() == 2 else embeddings

    def extract_embedding(self, audio_path):
        with torch.no_grad():
            embedding = self.embed_from_features(self.features(self.load_audio(audio_path)))
        return self.validate_embedding(embedding)


class _NoGradientPathAdapter:
    def extract_embedding(self, audio_path):
        return torch.tensor([1.0, 0.0, 0.0, 0.0])


_FAKE_SPEC = service.SpeakerModelSpec(
    key="test-model",
    label="Test model",
    model_id="test/model",
    revision="test-revision",
    architecture="test",
    embedding_dimension=4,
    threshold=0.5,
    recommended=True,
)


@pytest.fixture
def adapter(monkeypatch):
    fake = _DifferentiableAdapter()
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: fake)
    return fake


@pytest.fixture
def clip_paths(tmp_path):
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(_voice_like_wav_bytes(1.0, seed=1))
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_voice_like_wav_bytes(3.0, seed=2))
    return reference_path, target_path


def _centroid(adapter, reference_path):
    return adapter.extract_embedding(reference_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sum_pool_keeps_sign_and_total():
    values = np.array([[1.0, -2.0, 3.0, -4.0, 0.5]])

    pooled, boundaries = service._sum_pool(values, 1, 2)

    assert boundaries.tolist() == [0, 2, 5]
    assert pooled.tolist() == [[-1.0, -0.5]]
    assert pooled.sum() == pytest.approx(values.sum())


def test_sum_pool_leaves_small_inputs_untouched():
    values = np.arange(6, dtype=float).reshape(3, 2)

    pooled, boundaries = service._sum_pool(values, 0, 120)

    assert boundaries.tolist() == [0, 1, 2, 3]
    assert np.array_equal(pooled, values)


def test_feature_mel_bin_edges_centre_on_the_filter_peaks():
    edges = service._feature_mel_bin_edges(80, 0.0, 8000.0)

    assert len(edges) == 81
    assert all(low < high for low, high in zip(edges[:-1], edges[1:]))
    assert 0.0 < edges[0] and edges[-1] < 8000.0
    # SpeechBrain's 80-mel, 0-8000 Hz filterbank peaks at 22.12 Hz, 44.94 Hz, ...
    first_centre = service._mel_to_hz(
        (service._hz_to_mel(edges[0]) + service._hz_to_mel(edges[1])) / 2
    )
    assert first_centre == pytest.approx(22.12, abs=0.01)


# ---------------------------------------------------------------------------
# service.integrated_gradients_saliency
# ---------------------------------------------------------------------------


def test_fake_gradient_path_matches_extract_embedding(adapter, clip_paths):
    _, target_path = clip_paths

    waveform = adapter.load_audio(target_path)
    from_features = F.normalize(adapter.embed_from_features(adapter.features(waveform)), dim=0)

    assert service._cosine(from_features, adapter.extract_embedding(target_path)) > 1 - 1e-4


def test_grid_is_downsampled_to_the_limits(adapter, clip_paths):
    reference_path, target_path = clip_paths

    result = service.integrated_gradients_saliency(
        adapter, _centroid(adapter, reference_path), target_path
    )

    grid = np.asarray(result["attributions"])
    # 3 s -> 301 frames x 80 mel bins, pooled to the caps; rows are mel.
    assert grid.shape == (service.IG_MAX_MEL_ROWS, service.IG_MAX_TIME_COLUMNS) == (40, 120)
    assert len(result["time_edges_seconds"]) == 121
    assert len(result["mel_edges_hz"]) == 41
    assert len(result["time_totals"]) == 120
    assert result["time_edges_seconds"][0] == 0.0
    assert result["time_edges_seconds"][-1] == pytest.approx(3.0)
    assert result["audio_duration_seconds"] == pytest.approx(3.0)
    for edges in (result["time_edges_seconds"], result["mel_edges_hz"]):
        assert all(low < high for low, high in zip(edges[:-1], edges[1:]))
    assert result["n_steps"] == 32


def test_short_clip_keeps_one_column_per_frame(adapter, tmp_path):
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(_voice_like_wav_bytes(1.0, seed=1))
    target_path = tmp_path / "short.wav"
    target_path.write_bytes(_voice_like_wav_bytes(0.5, seed=3))

    result = service.integrated_gradients_saliency(
        adapter, _centroid(adapter, reference_path), target_path
    )

    # 0.5 s -> 51 frames, below the 120-column cap.
    assert np.asarray(result["attributions"]).shape == (40, 51)


def test_pooling_preserves_sign_and_totals(adapter, clip_paths):
    reference_path, target_path = clip_paths

    result = service.integrated_gradients_saliency(
        adapter, _centroid(adapter, reference_path), target_path
    )

    grid = np.asarray(result["attributions"])
    assert (grid > 0).any() and (grid < 0).any()
    assert grid.sum() == pytest.approx(result["total_attribution"], abs=1e-9)
    assert np.allclose(grid.sum(axis=0), result["time_totals"])
    assert sum(result["time_totals"]) == pytest.approx(result["total_attribution"], abs=1e-9)


def test_band_totals_use_the_frequency_occlusion_band_edges(adapter, clip_paths):
    reference_path, target_path = clip_paths

    result = service.integrated_gradients_saliency(
        adapter, _centroid(adapter, reference_path), target_path
    )

    edges = service._mel_band_edges(8, SAMPLE_RATE)
    bands = result["bands"]
    assert [band["band_index"] for band in bands] == list(range(1, 9))
    for index, band in enumerate(bands):
        assert set(band) == {"band_index", "low_hz", "high_hz", "label", "total_attribution"}
        assert band["low_hz"] == edges[index]
        assert band["high_hz"] == edges[index + 1]
        assert band["label"] == f"Band {index + 1}"
    # Every feature bin lands in exactly one band or in the out-of-band remainder.
    assert sum(band["total_attribution"] for band in bands) + result[
        "outside_bands_total"
    ] == pytest.approx(result["total_attribution"], abs=1e-9)
    assert any(abs(band["total_attribution"]) > 0 for band in bands)


def test_completeness_holds_on_a_small_differentiable_model(adapter, clip_paths):
    reference_path, target_path = clip_paths
    centroid = _centroid(adapter, reference_path)

    result = service.integrated_gradients_saliency(adapter, centroid, target_path)

    # baseline_similarity is the real clip's score, on the same path as occlusion.
    assert result["baseline_similarity"] == pytest.approx(
        service._cosine(centroid, adapter.extract_embedding(target_path)), abs=1e-5
    )
    assert result["expected_total"] == pytest.approx(
        result["baseline_similarity"] - result["baseline_input_similarity"]
    )
    assert abs(result["expected_total"]) > 1e-3
    assert result["convergence_delta"] == pytest.approx(
        result["total_attribution"] - result["expected_total"], abs=1e-5
    )
    assert result["completeness_ok"] is True
    assert result["total_attribution"] == pytest.approx(
        result["expected_total"], abs=max(0.05 * abs(result["expected_total"]), 0.01)
    )


def test_silent_baseline_is_computed_through_features(adapter, clip_paths):
    reference_path, target_path = clip_paths
    centroid = _centroid(adapter, reference_path)

    result = service.integrated_gradients_saliency(adapter, centroid, target_path)

    silent = adapter.features(torch.zeros(1, 3 * SAMPLE_RATE))
    expected = service._cosine(F.normalize(adapter.embed_from_features(silent), dim=0), centroid)
    assert result["baseline_input_similarity"] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("n_steps", [3, 257])
def test_rejects_out_of_range_step_count(adapter, clip_paths, n_steps):
    reference_path, target_path = clip_paths

    with pytest.raises(ValueError, match="between 4 and 256 steps"):
        service.integrated_gradients_saliency(
            adapter, _centroid(adapter, reference_path), target_path, n_steps=n_steps
        )


def test_rejects_an_adapter_without_a_gradient_path(clip_paths):
    _, target_path = clip_paths

    with pytest.raises(ValueError, match="Integrated Gradients is only available"):
        service.integrated_gradients_saliency(
            _NoGradientPathAdapter(), torch.tensor([1.0, 0.0, 0.0, 0.0]), target_path
        )


# ---------------------------------------------------------------------------
# service.compute_saliency_map(saliency_method=...)
# ---------------------------------------------------------------------------


def test_integrated_gradients_response_shape(adapter, clip_paths):
    reference_path, target_path = clip_paths

    result = service.compute_saliency_map(
        "test-model",
        [reference_path],
        target_path,
        reference_type="cluster",
        cluster_id="Cluster 1",
        target_recording_id="rec_target",
        saliency_method="integrated_gradients",
    )

    assert set(result) == {
        "model",
        "model_label",
        "reference_type",
        "cluster_id",
        "target_recording_id",
        "reference_count",
        "baseline_similarity",
        "threshold",
        "audio_duration_seconds",
        "interpretation",
        "saliency_method",
        "n_steps",
        "baseline_input_similarity",
        "attributions",
        "time_edges_seconds",
        "mel_edges_hz",
        "time_totals",
        "bands",
        "outside_bands_total",
        "convergence_delta",
        "total_attribution",
        "expected_total",
        "completeness_ok",
    }
    assert result["saliency_method"] == "integrated_gradients"
    assert result["cluster_id"] == "Cluster 1"
    assert result["target_recording_id"] == "rec_target"
    assert result["reference_count"] == 1
    assert "occlusion_axis" not in result and "segments" not in result


def test_integrated_gradients_ignores_occlusion_arguments(adapter, clip_paths):
    reference_path, target_path = clip_paths
    kwargs = {"reference_type": "cluster", "saliency_method": "integrated_gradients"}

    plain = service.compute_saliency_map("test-model", [reference_path], target_path, **kwargs)
    with_occlusion_arguments = service.compute_saliency_map(
        "test-model",
        [reference_path],
        target_path,
        segment_count=999,
        occlusion_axis="phase",
        band_count=999,
        **kwargs,
    )

    assert plain == with_occlusion_arguments


def test_compute_saliency_map_rejects_unknown_saliency_method(adapter, clip_paths):
    reference_path, target_path = clip_paths

    with pytest.raises(ValueError, match="saliency_method"):
        service.compute_saliency_map(
            "test-model",
            [reference_path],
            target_path,
            reference_type="cluster",
            saliency_method="shap",
        )


def test_integrated_gradients_requires_a_reference(adapter, clip_paths):
    _, target_path = clip_paths

    with pytest.raises(ValueError, match="At least one reference"):
        service.compute_saliency_map(
            "test-model",
            [],
            target_path,
            reference_type="cluster",
            saliency_method="integrated_gradients",
        )


@pytest.mark.parametrize(
    "occlusion_kwargs",
    [{"segment_count": 4}, {"occlusion_axis": "frequency", "band_count": 6}],
)
def test_occlusion_response_is_unchanged_by_the_explicit_default_method(
    adapter, clip_paths, occlusion_kwargs
):
    reference_path, target_path = clip_paths

    default_result = service.compute_saliency_map(
        "test-model", [reference_path], target_path, reference_type="cluster", **occlusion_kwargs
    )
    explicit_result = service.compute_saliency_map(
        "test-model",
        [reference_path],
        target_path,
        reference_type="cluster",
        saliency_method="occlusion",
        **occlusion_kwargs,
    )

    assert default_result == explicit_result
    assert "saliency_method" not in default_result


# ---------------------------------------------------------------------------
# Endpoint-level
# ---------------------------------------------------------------------------


async def _fake_resolve_audio_source(recording_id: str, sid: str) -> Path:
    return Path("/fake") / f"{recording_id}.wav"


_CLUSTER_FORM = {
    "model": "ecapa-tdnn",
    "reference_type": "cluster",
    "reference_recording_ids": ["rec_a"],
    "target_recording_id": "rec_target",
    "cluster_id": "Cluster 1",
}

_OCCLUSION_KWARGS = {"reference_type", "cluster_id", "target_recording_id", "segment_count"}


@pytest.mark.asyncio
async def test_saliency_endpoint_rejects_unknown_saliency_method(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={**_CLUSTER_FORM, "saliency_method": "shap"},
    )

    assert response.status_code == 422
    assert "saliency_method" in response.json()["detail"]


@pytest.mark.asyncio
async def test_saliency_endpoint_passes_integrated_gradients_through(client):
    expected = {"saliency_method": "integrated_gradients", "attributions": []}
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value=expected) as mock_compute,
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={**_CLUSTER_FORM, "saliency_method": "integrated_gradients"},
        )

    assert response.status_code == 200
    assert response.json() == expected
    assert set(mock_compute.call_args.kwargs) == _OCCLUSION_KWARGS | {"saliency_method"}
    assert mock_compute.call_args.kwargs["saliency_method"] == "integrated_gradients"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra", "expected_kwargs"),
    [
        ({"saliency_method": "occlusion"}, _OCCLUSION_KWARGS),
        ({"saliency_method": "occlusion", "occlusion_axis": "time"}, _OCCLUSION_KWARGS),
        (
            {"saliency_method": "occlusion", "occlusion_axis": "frequency", "band_count": "6"},
            _OCCLUSION_KWARGS | {"occlusion_axis", "band_count"},
        ),
        ({"occlusion_axis": "frequency"}, _OCCLUSION_KWARGS | {"occlusion_axis", "band_count"}),
    ],
)
async def test_saliency_endpoint_occlusion_requests_reach_service_unchanged(
    client, extra, expected_kwargs
):
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value={"ok": True}) as mock_compute,
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={**_CLUSTER_FORM, **extra},
        )

    assert response.status_code == 200
    assert set(mock_compute.call_args.kwargs) == expected_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [
        (service.SpeakerModelUnavailable("weights missing"), 503),
        (ValueError("Integrated Gradients is only available for ECAPA-TDNN"), 422),
    ],
)
async def test_saliency_endpoint_maps_integrated_gradients_errors(client, error, status):
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", side_effect=error),
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={**_CLUSTER_FORM, "saliency_method": "integrated_gradients"},
        )

    assert response.status_code == status
    assert response.json()["detail"] == str(error)


# ---------------------------------------------------------------------------
# Real models (opt-in): the gradient path is the production model
# ---------------------------------------------------------------------------

_RUN_MODEL_TESTS = os.environ.get("VOXLIT_RUN_SV_MODEL_TESTS") == "1"


# ResNet34-LM goes first: pyannote.audio cannot be imported once SpeechBrain's
# lazy modules have been registered by an ECAPA-TDNN load in the same process.
@pytest.mark.integration
@pytest.mark.skipif(
    not _RUN_MODEL_TESTS,
    reason="Loads real pretrained models; set VOXLIT_RUN_SV_MODEL_TESTS=1 to run",
)
@pytest.mark.parametrize("model_key", ["resnet34-lm", "ecapa-tdnn"])
def test_real_model_gradient_path_matches_extract_embedding(model_key, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # cached weights only, never download
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_voice_like_wav_bytes(2.0, seed=4))
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(_voice_like_wav_bytes(2.0, seed=5))

    real_adapter = service.get_model(model_key)
    expected = real_adapter.extract_embedding(target_path)

    features = real_adapter.features(real_adapter.load_audio(target_path))
    assert features.dim() == 2 and features.shape[1] == MEL_COUNT
    probe = features.clone().requires_grad_(True)
    embedding = F.normalize(real_adapter.embed_from_features(probe), p=2, dim=0)
    assert service._cosine(embedding.detach().cpu(), expected) > 1 - 1e-4

    # Gradients reach the features, so both models support Integrated Gradients.
    embedding.sum().backward()
    assert torch.isfinite(probe.grad).all() and float(probe.grad.abs().sum()) > 0

    result = service.integrated_gradients_saliency(
        real_adapter, real_adapter.extract_embedding(reference_path), target_path, n_steps=8
    )
    grid = np.asarray(result["attributions"])
    assert grid.shape == (40, 120) and np.isfinite(grid).all()
    assert result["baseline_similarity"] == pytest.approx(
        service._cosine(real_adapter.extract_embedding(reference_path), expected), abs=1e-4
    )
