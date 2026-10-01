"""Focused tests for frequency-band occlusion saliency in Speaker
Verification: `service._mel_band_edges`, `service._silence_band`,
`service._occlude_bands_and_score`, the `occlusion_axis="frequency"` branch
of `service.compute_saliency_map`, and the matching
`/tasks/verification/explain/saliency` form parameters.

Math-level tests run against real (synthetic) WAV files with a
content-dependent fake adapter, mirroring
`test_speaker_verification_saliency.py`. No real model is loaded.
"""

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from app.tasks.verification import service


SAMPLE_RATE = 16000


def _tone_wav_bytes(
    tones: dict[float, float],
    *,
    sample_rate: int = SAMPLE_RATE,
    seconds: float = 1.0,
) -> bytes:
    """A WAV that is a sum of sine tones, `{frequency_hz: amplitude}`."""

    t = np.arange(int(sample_rate * seconds)) / sample_rate
    audio = sum(amplitude * np.sin(2 * np.pi * freq * t) for freq, amplitude in tones.items())
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


def _audio_upload(name: str, content: bytes):
    return (name, io.BytesIO(content), "audio/wav")


_SPECTRAL_EDGES_HZ = (0.0, 500.0, 1500.0, 3500.0, 8001.0)


class _SpectralEnergyAdapter:
    """Deterministic, content-dependent 4-dim embedding: L2-normalized
    spectral magnitude in four broad frequency ranges. Silencing a frequency
    band therefore measurably changes the embedding. Also records the sample
    count of every band-occluded file it is asked to embed."""

    def __init__(self):
        self.occluded_lengths: list[int] = []

    def extract_embedding(self, audio_path):
        waveform, sample_rate = torchaudio.load(str(audio_path))
        samples = waveform.mean(dim=0)
        if Path(audio_path).name.startswith("band-"):
            self.occluded_lengths.append(int(samples.shape[0]))
        magnitude = torch.fft.rfft(samples).abs()
        freqs = torch.fft.rfftfreq(samples.shape[0], d=1.0 / sample_rate)
        energies = [
            float(magnitude[(freqs >= low) & (freqs < high)].sum())
            for low, high in zip(_SPECTRAL_EDGES_HZ[:-1], _SPECTRAL_EDGES_HZ[1:])
        ]
        vector = torch.tensor(energies, dtype=torch.float32)
        norm = torch.linalg.vector_norm(vector)
        if float(norm) == 0.0:
            return torch.tensor([1.0, 0.0, 0.0, 0.0])
        return vector / norm


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

_TARGET_TONES = {200.0: 0.3, 800.0: 0.2, 2000.0: 0.15, 5000.0: 0.1}
_REFERENCE_TONES = {200.0: 0.1, 800.0: 0.3, 2000.0: 0.1, 5000.0: 0.2}


@pytest.fixture
def fake_model(monkeypatch):
    adapter = _SpectralEnergyAdapter()
    monkeypatch.setattr(service, "get_model_spec", lambda _: _FAKE_SPEC)
    monkeypatch.setattr(service, "get_model", lambda _: adapter)
    return adapter


@pytest.fixture
def tone_paths(tmp_path):
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(_tone_wav_bytes(_REFERENCE_TONES))
    target_path = tmp_path / "target.wav"
    target_path.write_bytes(_tone_wav_bytes(_TARGET_TONES))
    return reference_path, target_path


# ---------------------------------------------------------------------------
# Band edge maths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("band_count", [4, 8, 12])
def test_mel_band_edges_span_50_hz_to_8_khz_at_16_khz(band_count):
    edges = service._mel_band_edges(band_count, 16000)

    assert len(edges) == band_count + 1
    assert edges[0] == 50.0
    assert edges[-1] == 8000.0
    assert all(low < high for low, high in zip(edges[:-1], edges[1:]))


def test_mel_band_edges_are_evenly_spaced_on_the_mel_scale():
    edges = service._mel_band_edges(8, 16000)

    mels = [2595.0 * np.log10(1.0 + hz / 700.0) for hz in edges]
    steps = np.diff(mels)
    assert np.allclose(steps, steps[0], atol=1e-6)
    # Mel spacing means bands get wider in Hz as frequency rises.
    widths = np.diff(edges)
    assert all(a < b for a, b in zip(widths[:-1], widths[1:]))


def test_mel_band_edges_stop_at_nyquist_for_low_sample_rates():
    assert service._mel_band_edges(8, 8000)[-1] == 4000.0


def test_mel_band_edges_never_exceed_8_khz_for_high_sample_rates():
    assert service._mel_band_edges(8, 44100)[-1] == 8000.0


@pytest.mark.parametrize("band_index", [1, 4, 8, 12])
def test_band_label_is_neutral_and_one_based(band_index):
    assert service._band_label(band_index) == f"Band {band_index}"


# ---------------------------------------------------------------------------
# STFT masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("sample_rate", "length"), [(16000, 16001), (8000, 5000), (44100, 12345)])
def test_stft_round_trip_with_no_bins_zeroed_reproduces_the_input(sample_rate, length):
    """The baseline is scored on the original file while every band score is
    scored on reconstructed audio, so reconstruction error must be negligible."""

    generator = torch.Generator().manual_seed(0)
    waveform = (torch.rand(1, length, generator=generator) - 0.5) * 0.8

    # An empty frequency range zeroes no bins.
    reconstructed = service._silence_band(waveform, sample_rate, 0.0, 0.0)

    assert reconstructed.shape == waveform.shape
    assert float((reconstructed - waveform).abs().max()) < 1e-4


def test_silence_band_removes_only_the_tone_inside_the_band():
    t = torch.arange(SAMPLE_RATE) / SAMPLE_RATE
    waveform = (0.3 * torch.sin(2 * torch.pi * 1000.0 * t)).unsqueeze(0)
    original_rms = float(waveform.pow(2).mean().sqrt())

    inside = service._silence_band(waveform, SAMPLE_RATE, 800.0, 1200.0)
    outside = service._silence_band(waveform, SAMPLE_RATE, 3000.0, 4000.0)

    assert inside.shape == waveform.shape
    assert float(inside.pow(2).mean().sqrt()) < 0.05 * original_rms
    assert float(outside.pow(2).mean().sqrt()) == pytest.approx(original_rms, rel=1e-3)


def test_silence_band_rejects_audio_shorter_than_the_stft_window():
    with pytest.raises(ValueError, match="too short"):
        service._silence_band(torch.zeros(1, 100), SAMPLE_RATE, 50.0, 300.0)


# ---------------------------------------------------------------------------
# service.compute_saliency_map(occlusion_axis="frequency")
# ---------------------------------------------------------------------------


_COMMON_KEYS = {
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
}


def test_frequency_mode_response_shape(fake_model, tone_paths):
    reference_path, target_path = tone_paths

    result = service.compute_saliency_map(
        "test-model",
        [reference_path],
        target_path,
        reference_type="cluster",
        cluster_id="Cluster 1",
        target_recording_id="rec_target",
        occlusion_axis="frequency",
        band_count=8,
    )

    assert set(result) == _COMMON_KEYS | {"occlusion_axis", "band_count", "bands"}
    assert result["occlusion_axis"] == "frequency"
    assert result["band_count"] == 8
    assert result["cluster_id"] == "Cluster 1"
    assert result["target_recording_id"] == "rec_target"
    assert result["audio_duration_seconds"] == pytest.approx(1.0)

    bands = result["bands"]
    edges = service._mel_band_edges(8, SAMPLE_RATE)
    assert [band["band_index"] for band in bands] == list(range(1, 9))
    for index, band in enumerate(bands):
        assert set(band) == {
            "band_index",
            "low_hz",
            "high_hz",
            "label",
            "occluded_similarity",
            "similarity_change",
            "influence_strength",
        }
        assert band["low_hz"] == edges[index]
        assert band["high_hz"] == edges[index + 1]
        assert band["label"] == f"Band {index + 1}"
        assert band["similarity_change"] == pytest.approx(
            result["baseline_similarity"] - band["occluded_similarity"]
        )
        assert band["influence_strength"] == pytest.approx(abs(band["similarity_change"]))

    # Silencing bands that hold a tone must actually move the score.
    assert max(band["influence_strength"] for band in bands) > 0.01


def test_frequency_mode_masked_audio_keeps_original_length(fake_model, tmp_path):
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(_tone_wav_bytes(_REFERENCE_TONES))
    target_path = tmp_path / "target.wav"
    # 0.8125 s -> 13000 samples, not a multiple of the STFT hop.
    target_path.write_bytes(_tone_wav_bytes(_TARGET_TONES, seconds=0.8125))
    original_length = sf.info(str(target_path)).frames

    service.compute_saliency_map(
        "test-model",
        [reference_path],
        target_path,
        reference_type="cluster",
        occlusion_axis="frequency",
        band_count=6,
    )

    assert fake_model.occluded_lengths == [original_length] * 6


def test_frequency_mode_is_deterministic(fake_model, tone_paths):
    reference_path, target_path = tone_paths
    kwargs = {"reference_type": "cluster", "occlusion_axis": "frequency", "band_count": 8}

    first = service.compute_saliency_map("test-model", [reference_path], target_path, **kwargs)
    second = service.compute_saliency_map("test-model", [reference_path], target_path, **kwargs)

    assert first == second


@pytest.mark.parametrize("band_count", [3, 13])
def test_frequency_mode_rejects_out_of_range_band_count(fake_model, tone_paths, band_count):
    reference_path, target_path = tone_paths

    with pytest.raises(ValueError):
        service.compute_saliency_map(
            "test-model",
            [reference_path],
            target_path,
            reference_type="cluster",
            occlusion_axis="frequency",
            band_count=band_count,
        )


def test_compute_saliency_map_rejects_unknown_occlusion_axis(fake_model, tone_paths):
    reference_path, target_path = tone_paths

    with pytest.raises(ValueError):
        service.compute_saliency_map(
            "test-model",
            [reference_path],
            target_path,
            reference_type="cluster",
            occlusion_axis="phase",
        )


def test_time_mode_response_is_unchanged(fake_model, tone_paths):
    reference_path, target_path = tone_paths

    default_result = service.compute_saliency_map(
        "test-model", [reference_path], target_path, reference_type="cluster", segment_count=4
    )
    explicit_result = service.compute_saliency_map(
        "test-model",
        [reference_path],
        target_path,
        reference_type="cluster",
        segment_count=4,
        occlusion_axis="time",
        band_count=12,
    )

    assert set(default_result) == _COMMON_KEYS | {"segment_count", "segments"}
    assert default_result == explicit_result
    assert set(default_result["segments"][0]) == {
        "segment_index",
        "start_seconds",
        "end_seconds",
        "occluded_similarity",
        "similarity_change",
        "influence_strength",
    }


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


@pytest.mark.asyncio
async def test_saliency_endpoint_rejects_unknown_occlusion_axis(client):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={**_CLUSTER_FORM, "occlusion_axis": "phase"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("band_count", ["3", "13"])
async def test_saliency_endpoint_rejects_out_of_range_band_count(client, band_count):
    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={**_CLUSTER_FORM, "occlusion_axis": "frequency", "band_count": band_count},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_saliency_endpoint_passes_frequency_axis_through(client):
    expected = {"occlusion_axis": "frequency", "band_count": 6, "bands": []}
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value=expected) as mock_compute,
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={**_CLUSTER_FORM, "occlusion_axis": "frequency", "band_count": "6"},
        )

    assert response.status_code == 200
    assert response.json() == expected
    assert mock_compute.call_args.kwargs["occlusion_axis"] == "frequency"
    assert mock_compute.call_args.kwargs["band_count"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{}, {"occlusion_axis": "time"}])
async def test_saliency_endpoint_time_request_reaches_service_unchanged(client, extra):
    with (
        patch("app.tasks.verification.router._resolve_audio_source", new=_fake_resolve_audio_source),
        patch("app.tasks.verification.router.compute_saliency_map", return_value={"ok": True}) as mock_compute,
    ):
        response = await client.post(
            "/tasks/verification/explain/saliency",
            data={**_CLUSTER_FORM, **extra},
        )

    assert response.status_code == 200
    assert set(mock_compute.call_args.kwargs) == {
        "reference_type",
        "cluster_id",
        "target_recording_id",
        "segment_count",
    }


@pytest.mark.asyncio
async def test_saliency_endpoint_returns_422_when_clip_is_too_short_for_band_occlusion(client, fake_model):
    """The service's "too short" ValueError must surface as a 422, not a 500."""

    files = [
        ("enrollment_files", _audio_upload(f"ref{i}.wav", _tone_wav_bytes(_REFERENCE_TONES)))
        for i in range(3)
    ]
    # 100 samples: shorter than half the 512-sample STFT window.
    files.append(("probe_file", _audio_upload("probe.wav", _tone_wav_bytes(_TARGET_TONES, seconds=100 / SAMPLE_RATE))))

    response = await client.post(
        "/tasks/verification/explain/saliency",
        data={"model": "ecapa-tdnn", "reference_type": "enrollment", "occlusion_axis": "frequency"},
        files=files,
    )

    assert response.status_code == 422
    assert "too short" in response.json()["detail"]
