"""Focused tests for the Speaker Verification perturbation sweep and its
flip-point analysis."""

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from app.core.settings import settings
from app.services import pertubation_service
from app.tasks.verification import dataset, robustness, service, session_assets

ECAPA_SPEC = service.MODEL_SPECS["ecapa-tdnn"]
THRESHOLD = ECAPA_SPEC.threshold
DIM = ECAPA_SPEC.embedding_dimension

NOISE_GRID = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
PITCH_GRID = [-6, -4, -2, -1, 1, 2, 4, 6]
STRETCH_GRID = [0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2.0]


# --------------------------------------------------------------------------
# Fixtures and fakes
# --------------------------------------------------------------------------


def _write_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0, freq: float = 220.0) -> Path:
    t = np.linspace(0, seconds, int(sample_rate * seconds), False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)
    return path


@pytest.fixture
def source(tmp_path):
    return _write_wav(tmp_path / "source.wav", freq=333.0)


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_VERIFICATION_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "vox_indian_demo_92"
    dataset_dir.mkdir(parents=True)
    for index in range(3):
        _write_wav(dataset_dir / f"speaker{index}_clip.wav", freq=180.0 + index * 20)
    return [recording.recording_id for recording in dataset.list_recordings()]


def _embedding_with_similarity(similarity: float) -> torch.Tensor:
    """Unit vector whose cosine similarity with `_ORIGINAL` is `similarity`."""

    values = torch.zeros(DIM)
    values[0] = similarity
    values[1] = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return values


_ORIGINAL = _embedding_with_similarity(1.0)


class _ScheduledAdapter:
    """The source clip maps to `_ORIGINAL`; sweep point `i` (written by
    `sweep_perturbation` as `point-{i}.wav`) maps to an embedding whose
    similarity with the original is `similarities[i]`. An entry that is an
    exception instance is raised instead. Records every requested path."""

    def __init__(self, similarities) -> None:
        self.similarities = similarities
        self.requested: list[Path] = []

    def extract_embedding(self, audio_path):
        path = Path(audio_path)
        self.requested.append(path)
        match = re.fullmatch(r"point-(\d+)\.wav", path.name)
        if match is None:
            return _ORIGINAL
        value = self.similarities[int(match.group(1))]
        if isinstance(value, Exception):
            raise value
        return _embedding_with_similarity(value)


class _HashDerivedAdapter:
    """Embedding is a deterministic function of the audio file's bytes."""

    def extract_embedding(self, audio_path):
        digest = hashlib.sha256(Path(audio_path).read_bytes()).digest()
        values = torch.tensor([digest[i % len(digest)] for i in range(DIM)], dtype=torch.float32)
        return F.normalize(values, p=2, dim=0)


def _install(monkeypatch, adapter) -> list[str]:
    """Patch the model registry; returns the list of `get_model` calls."""

    calls: list[str] = []

    def _get_model(model_key):
        calls.append(model_key)
        return adapter

    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)
    monkeypatch.setattr(service, "get_model", _get_model)
    return calls


def _fast_transforms(monkeypatch) -> None:
    """Replace the subprocess-backed librosa transforms with instant ones
    that still genuinely change the audio."""

    monkeypatch.setattr(
        pertubation_service, "apply_pitch_shift", lambda waveform, _sr, semitones, **_: waveform * (1 + 0.05 * semitones)
    )
    monkeypatch.setattr(
        pertubation_service,
        "apply_time_stretch",
        lambda waveform, factor, **_: waveform[..., : int(waveform.shape[-1] / factor)].repeat(1, 2)[
            ..., : int(waveform.shape[-1] / factor)
        ],
    )


def _above(delta: float) -> float:
    return THRESHOLD + delta


def _flip(result, direction):
    return next(flip for flip in result["flips"] if flip["direction"] == direction)


ALL_SAFE = [_above(0.2)] * 8


# --------------------------------------------------------------------------
# Grids, directions, unsupported types
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation_type,grid,directions",
    [
        ("noise", NOISE_GRID, ["stronger"] * 8),
        ("pitch_shift", PITCH_GRID, ["down"] * 4 + ["up"] * 4),
        ("time_stretch", STRETCH_GRID, ["slower"] * 4 + ["faster"] * 4),
    ],
)
def test_grid_order_and_directions(monkeypatch, source, perturbation_type, grid, directions):
    _fast_transforms(monkeypatch)
    _install(monkeypatch, _ScheduledAdapter(ALL_SAFE))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, perturbation_type)

    assert [point["strength"] for point in result["points"]] == grid
    assert [point["direction"] for point in result["points"]] == directions
    assert all(point["status"] == "ok" for point in result["points"])
    assert result["perturbation_type"] == perturbation_type
    assert result["model"] == ECAPA_SPEC.key
    assert result["model_label"] == ECAPA_SPEC.label
    assert result["threshold"] == THRESHOLD
    assert isinstance(result["summary"], str) and result["summary"].endswith(".")


@pytest.mark.parametrize("perturbation_type", ["time_masking", "frequency_masking", "unknown", ""])
def test_unsupported_type_raises_value_error_without_loading_model(monkeypatch, source, perturbation_type):
    calls = _install(monkeypatch, _ScheduledAdapter(ALL_SAFE))

    with pytest.raises(ValueError):
        robustness.sweep_perturbation("ecapa-tdnn", source, perturbation_type)
    assert calls == []


# --------------------------------------------------------------------------
# Model / original embedding loaded once
# --------------------------------------------------------------------------


def test_model_and_original_embedding_loaded_once(monkeypatch, source):
    adapter = _ScheduledAdapter(ALL_SAFE)
    calls = _install(monkeypatch, adapter)

    robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    assert calls == ["ecapa-tdnn"]
    assert adapter.requested.count(Path(source)) == 1
    assert len(adapter.requested) == 1 + len(NOISE_GRID)


def test_cached_original_embedding_skips_source_extraction(monkeypatch, source):
    adapter = _ScheduledAdapter(ALL_SAFE)
    _install(monkeypatch, adapter)

    result = robustness.sweep_perturbation(
        "ecapa-tdnn", source, "noise", cached_original_embedding=_ORIGINAL
    )

    assert Path(source) not in adapter.requested
    assert result["points"][0]["similarity"] == pytest.approx(_above(0.2), abs=1e-4)


# --------------------------------------------------------------------------
# not_applied points / all-failed
# --------------------------------------------------------------------------


def test_silent_fallback_point_is_not_applied_and_sweep_continues(monkeypatch, source):
    def _pitch_shift(waveform, _sr, semitones, **_):
        return waveform if semitones == 2 else waveform * 0.5  # +2 silently falls back

    monkeypatch.setattr(pertubation_service, "apply_pitch_shift", _pitch_shift)
    adapter = _ScheduledAdapter(ALL_SAFE)
    _install(monkeypatch, adapter)

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "pitch_shift")

    by_strength = {point["strength"]: point for point in result["points"]}
    skipped = by_strength[2]
    assert skipped["status"] == "not_applied"
    assert skipped["reason"]
    assert skipped["similarity"] is None and skipped["same_speaker"] is None
    assert [p["status"] for p in result["points"]].count("ok") == 7
    # The skipped point is never embedded.
    assert all(path.name != "point-5.wav" for path in adapter.requested)


def test_transform_exception_and_embedding_failure_are_not_applied(monkeypatch, source):
    def _pitch_shift(waveform, _sr, semitones, **_):
        if semitones == -6:
            raise OSError("subprocess could not start")
        return waveform * 0.5

    monkeypatch.setattr(pertubation_service, "apply_pitch_shift", _pitch_shift)
    similarities = list(ALL_SAFE)
    similarities[7] = RuntimeError("embedding blew up")
    _install(monkeypatch, _ScheduledAdapter(similarities))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "pitch_shift")

    statuses = [point["status"] for point in result["points"]]
    assert statuses == ["not_applied"] + ["ok"] * 6 + ["not_applied"]
    assert "subprocess could not start" in result["points"][0]["reason"]
    assert "embedding blew up" in result["points"][7]["reason"]


def test_all_points_unchanged_raises_without_loading_model(monkeypatch, source):
    monkeypatch.setattr(pertubation_service, "apply_time_stretch", lambda waveform, *a, **k: waveform)
    calls = _install(monkeypatch, _ScheduledAdapter(ALL_SAFE))

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.sweep_perturbation("ecapa-tdnn", source, "time_stretch")
    assert calls == []


def test_all_embeddings_failing_raises(monkeypatch, source):
    _install(monkeypatch, _ScheduledAdapter([RuntimeError("bad")] * 8))

    with pytest.raises(robustness.PerturbationNotApplied):
        robustness.sweep_perturbation("ecapa-tdnn", source, "noise")


def test_model_unavailable_propagates(monkeypatch, source):
    _install(monkeypatch, _ScheduledAdapter([service.SpeakerModelUnavailable("no weights")] * 8))

    with pytest.raises(service.SpeakerModelUnavailable):
        robustness.sweep_perturbation("ecapa-tdnn", source, "noise")


# --------------------------------------------------------------------------
# Flip-point analysis
# --------------------------------------------------------------------------


def test_flip_strength_is_linearly_interpolated(monkeypatch, source):
    # Safe through 0.02 (+0.1 above threshold), below from 0.05 (-0.3):
    # the crossing sits a quarter of the way from 0.02 to 0.05.
    deltas = [0.3, 0.25, 0.2, 0.15, 0.1, -0.3, -0.4, -0.5]
    _install(monkeypatch, _ScheduledAdapter([_above(delta) for delta in deltas]))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    assert [point["same_speaker"] for point in result["points"]] == [True] * 5 + [False] * 3
    flip = _flip(result, "stronger")
    assert flip["flipped"] is True
    assert flip["already_below_at_weakest"] is False
    assert flip["last_safe_strength"] == 0.02
    assert flip["flip_strength"] == pytest.approx(0.0275, abs=1e-4)
    assert "flips" in result["summary"]


def test_flip_interpolation_skips_not_applied_points(monkeypatch, source):
    deltas = [0.3, 0.25, 0.2, 0.15, 0.1, RuntimeError("bad point"), -0.3, -0.5]
    _install(
        monkeypatch,
        _ScheduledAdapter([d if isinstance(d, Exception) else _above(d) for d in deltas]),
    )

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    flip = _flip(result, "stronger")
    assert flip["last_safe_strength"] == 0.02
    # Interpolated between 0.02 and 0.1, the nearest "ok" neighbours.
    assert flip["flip_strength"] == pytest.approx(0.02 + 0.25 * (0.1 - 0.02), abs=1e-4)


def test_never_flips(monkeypatch, source):
    _install(monkeypatch, _ScheduledAdapter(ALL_SAFE))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    flip = _flip(result, "stronger")
    assert flip["flipped"] is False
    assert flip["flip_strength"] is None
    assert flip["last_safe_strength"] == 0.2
    assert "never flips" in result["summary"]


def test_already_below_threshold_at_weakest_point(monkeypatch, source):
    _install(monkeypatch, _ScheduledAdapter([_above(-0.1)] * 8))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    flip = _flip(result, "stronger")
    assert flip["flipped"] is True
    assert flip["already_below_at_weakest"] is True
    assert flip["flip_strength"] == 0.001
    assert flip["last_safe_strength"] is None
    assert "weakest" in result["summary"]


def test_two_directions_are_analysed_separately_from_the_weakest_outward(monkeypatch, source):
    _fast_transforms(monkeypatch)
    # Grid order: -6, -4, -2, -1, 1, 2, 4, 6. Down flips between -2 and -4
    # (walking outward from -1); up never flips.
    deltas = [-0.4, -0.3, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2]
    _install(monkeypatch, _ScheduledAdapter([_above(delta) for delta in deltas]))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "pitch_shift")

    assert [flip["direction"] for flip in result["flips"]] == ["down", "up"]
    down = _flip(result, "down")
    assert down["flipped"] is True
    assert down["last_safe_strength"] == -2
    assert down["flip_strength"] == pytest.approx(-2.5, abs=1e-3)
    up = _flip(result, "up")
    assert up["flipped"] is False
    assert up["flip_strength"] is None
    assert up["last_safe_strength"] == 6


# --------------------------------------------------------------------------
# SNR, determinism
# --------------------------------------------------------------------------


def test_snr_reported_for_noise_and_decreases_with_strength(monkeypatch, source):
    _install(monkeypatch, _ScheduledAdapter(ALL_SAFE))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    snrs = [point["snr_db"] for point in result["points"]]
    assert all(isinstance(value, float) and math.isfinite(value) for value in snrs)
    assert snrs == sorted(snrs, reverse=True)
    # 0.3-amplitude sine: signal power 0.045; noise power = noise_level**2.
    for point in result["points"]:
        expected = 10 * math.log10(0.045 / point["strength"] ** 2)
        assert point["snr_db"] == pytest.approx(expected, abs=0.2)


@pytest.mark.parametrize("perturbation_type", ["pitch_shift", "time_stretch"])
def test_snr_is_null_for_other_types(monkeypatch, source, perturbation_type):
    _fast_transforms(monkeypatch)
    _install(monkeypatch, _ScheduledAdapter(ALL_SAFE))

    result = robustness.sweep_perturbation("ecapa-tdnn", source, perturbation_type)

    assert all(point["snr_db"] is None for point in result["points"])


def test_noise_sweep_is_deterministic_with_fixed_seed(monkeypatch, source):
    _install(monkeypatch, _HashDerivedAdapter())
    seeds = []
    real_noise = pertubation_service.add_gaussian_noise

    def _spy(waveform, noise_level, seed=None):
        seeds.append(seed)
        return real_noise(waveform, noise_level, seed=seed)

    monkeypatch.setattr(pertubation_service, "add_gaussian_noise", _spy)

    first = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")
    second = robustness.sweep_perturbation("ecapa-tdnn", source, "noise")

    assert first == second
    assert set(seeds) == {robustness.DEFAULT_NOISE_SEED}


# --------------------------------------------------------------------------
# Router: /tasks/verification/perturbation/sweep
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_endpoint_returns_result_and_creates_no_session_asset(
    monkeypatch, client, fake_dataset_dir, tmp_path
):
    storage_root = tmp_path / "session-assets"
    monkeypatch.setattr("app.tasks.verification.session_assets._storage_root", lambda: storage_root)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("a sweep must never allocate a session asset")

    monkeypatch.setattr(session_assets, "begin_asset_write", _forbidden)
    _install(monkeypatch, _HashDerivedAdapter())

    recording_id = fake_dataset_dir[0]
    response = await client.post(
        "/tasks/verification/perturbation/sweep",
        json={"model": "ecapa-tdnn", "recording_id": recording_id, "perturbation_type": "noise"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["perturbation_type"] == "noise"
    assert body["threshold"] == THRESHOLD
    assert body["source_recording_id"] == recording_id
    assert [point["strength"] for point in body["points"]] == NOISE_GRID
    assert [flip["direction"] for flip in body["flips"]] == ["stronger"]
    assert body["summary"]
    assert "original_embedding" not in body
    assert "session_asset" not in body
    assert "vox_indian_demo_92" not in response.text

    listing = await client.get("/tasks/verification/session-assets")
    assert listing.json()["assets"] == []
    assert not storage_root.exists() or not any(storage_root.rglob("*.wav"))


@pytest.mark.asyncio
@pytest.mark.parametrize("perturbation_type", ["time_masking", "nope"])
async def test_sweep_endpoint_unsupported_type_is_422(monkeypatch, client, fake_dataset_dir, perturbation_type):
    _install(monkeypatch, _HashDerivedAdapter())

    response = await client.post(
        "/tasks/verification/perturbation/sweep",
        json={"model": "ecapa-tdnn", "recording_id": fake_dataset_dir[0], "perturbation_type": perturbation_type},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_sweep_endpoint_unsupported_model_is_400(client, fake_dataset_dir):
    response = await client.post(
        "/tasks/verification/perturbation/sweep",
        json={"model": "not-a-real-model", "recording_id": fake_dataset_dir[0], "perturbation_type": "noise"},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_sweep_endpoint_unknown_recording_is_404(client, fake_dataset_dir):
    response = await client.post(
        "/tasks/verification/perturbation/sweep",
        json={"model": "ecapa-tdnn", "recording_id": "rec_0000000000000000", "perturbation_type": "noise"},
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_sweep_endpoint_model_unavailable_is_503(monkeypatch, client, fake_dataset_dir):
    monkeypatch.setattr(service, "get_model_spec", lambda _: ECAPA_SPEC)

    def _unavailable(_model_key):
        raise service.SpeakerModelUnavailable("checkpoint missing")

    monkeypatch.setattr(service, "get_model", _unavailable)

    response = await client.post(
        "/tasks/verification/perturbation/sweep",
        json={"model": "ecapa-tdnn", "recording_id": fake_dataset_dir[0], "perturbation_type": "noise"},
    )

    assert response.status_code == 503
