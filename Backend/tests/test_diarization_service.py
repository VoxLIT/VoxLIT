"""Focused tests for diarization model dispatch, confidence scoring, and the
segment-embedding rules (task_b `service.py`).

**No model is ever loaded.** Two mocking levels are used:

* `_attach_confidence` / `_confidence_bucket` are pure — called directly with
  hand-built unit vectors whose expected margin is computed by hand.
* The `<0.4 s` skip rule lives inside `_Pyannote31Adapter.diarize_and_embed`,
  so that method is exercised on a real adapter instance built with
  `object.__new__` (bypassing the model-loading `__init__`) with a stub
  pipeline and embedding inference attached. That covers the real branch
  rather than a reimplementation of it.
"""

import hashlib
import math
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from app.tasks.task_b import service


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


ALL_MODELS = ["pyannote-3.1", "reverb-v1", "reverb-v2"]


def test_list_models_exposes_all_three_specs():
    models = service.list_models()

    assert [spec["key"] for spec in models] == ALL_MODELS
    by_key = {spec["key"]: spec for spec in models}
    assert by_key["pyannote-3.1"]["pipeline_id"] == "pyannote/speaker-diarization-3.1"
    assert by_key["reverb-v1"]["pipeline_id"] == "Revai/reverb-diarization-v1"
    assert by_key["reverb-v2"]["pipeline_id"] == "Revai/reverb-diarization-v2"
    # Exactly one recommended default, and the UI keeps pyannote as it.
    assert [spec["key"] for spec in models if spec["recommended"]] == ["pyannote-3.1"]


def test_every_model_shares_one_embedding_space():
    """Verified against the real configs: both Rev pipelines cluster with the
    same WeSpeaker weights as 3.1 (their mirror id is a byte-identical
    re-upload). Keeping one embedding_model_id is what makes confidence and
    the similarity matrix comparable across models -- not an approximation."""

    specs = service.MODEL_SPECS.values()
    assert {spec.embedding_model_id for spec in specs} == {
        "pyannote/wespeaker-voxceleb-resnet34-LM"
    }
    assert {spec.embedding_dimension for spec in specs} == {256}


@pytest.mark.parametrize("model_key", ALL_MODELS)
def test_every_spec_carries_a_glass_box_note(model_key):
    """The workbench header renders this; an empty one would silently drop the
    caveat about what the model does and does not change."""

    assert service.MODEL_SPECS[model_key].glass_box_note.strip()


@pytest.mark.parametrize("model_key", ALL_MODELS)
def test_get_model_spec_returns_the_frozen_spec(model_key):
    spec = service.get_model_spec(model_key)
    assert spec is service.MODEL_SPECS[model_key]
    assert spec.key == model_key


@pytest.mark.parametrize(
    "bad_key", ["pyannote-4.0", "", "PYANNOTE-3.1", "ecapa-tdnn", "reverb-v3"]
)
def test_get_model_spec_rejects_unknown_keys(bad_key):
    with pytest.raises(service.UnsupportedDiarizationModel) as error:
        service.get_model_spec(bad_key)
    # The message must name the valid options -- the router surfaces it as a 400.
    for model_key in ALL_MODELS:
        assert model_key in str(error.value)


def test_thresholds_are_the_documented_values():
    assert service.MIN_EMBEDDABLE_SECONDS == 0.4
    assert service.CONFIDENCE_HIGH == 0.5
    assert service.CONFIDENCE_MEDIUM == 0.2


# ---------------------------------------------------------------------------
# Confidence buckets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (1.0, "high"),
        (0.5, "high"),  # boundary is inclusive
        (0.4999, "medium"),
        (0.2, "medium"),  # boundary is inclusive
        (0.1999, "uncertain"),
        (0.0, "uncertain"),
        (-1.0, "uncertain"),
        (None, None),
    ],
)
def test_confidence_bucket_boundaries(confidence, expected):
    assert service._confidence_bucket(confidence) == expected


# ---------------------------------------------------------------------------
# Confidence maths
# ---------------------------------------------------------------------------


def _unit(*components: float) -> list[float]:
    vector = np.asarray(components, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def _segment(segment_id: str, speaker: str, start: float = 0.0, end: float = 1.0) -> dict:
    return {"id": segment_id, "start": start, "end": end, "speaker": speaker}


def test_confidence_is_high_for_two_well_separated_clusters():
    # Orthogonal clusters: each segment sits exactly on its own centroid, so
    # d_own = 0, d_other = 1 - 0 = 1, margin = (1 - 0) / 1 = 1.0.
    segments = [
        _segment("seg_000", "SPEAKER_00"),
        _segment("seg_001", "SPEAKER_00"),
        _segment("seg_002", "SPEAKER_01"),
    ]
    embeddings = {
        "seg_000": _unit(1.0, 0.0),
        "seg_001": _unit(1.0, 0.0),
        "seg_002": _unit(0.0, 1.0),
    }

    service._attach_confidence(segments, embeddings)

    for segment in segments:
        assert segment["confidence"] == pytest.approx(1.0)
        assert segment["confidence_bucket"] == "high"


def test_confidence_is_negative_when_a_segment_sits_on_the_wrong_centroid():
    # seg_002 is labelled SPEAKER_01 but its vector is identical to
    # SPEAKER_00's cluster. SPEAKER_01's centroid is then that same vector
    # (it is the cluster's only member), which would make d_own = 0 -- so give
    # SPEAKER_01 a second, genuinely distinct member to pull its centroid away.
    segments = [
        _segment("seg_000", "SPEAKER_00"),
        _segment("seg_001", "SPEAKER_01"),
        _segment("seg_002", "SPEAKER_01"),
    ]
    embeddings = {
        "seg_000": _unit(1.0, 0.0),
        "seg_001": _unit(0.0, 1.0),
        "seg_002": _unit(1.0, 0.0),  # mislabelled: really looks like SPEAKER_00
    }

    service._attach_confidence(segments, embeddings)

    by_id = {s["id"]: s for s in segments}
    mislabelled = by_id["seg_002"]

    # SPEAKER_01 centroid = normalize(mean([0,1], [1,0])) = [0.7071, 0.7071].
    # For seg_002 = [1, 0]: d_own = 1 - 0.7071 = 0.2929, d_other = 1 - 1 = 0.
    # margin = (0 - 0.2929) / max(0.2929, 0) = -1.0.
    assert mislabelled["confidence"] == pytest.approx(-1.0, abs=1e-3)
    assert mislabelled["confidence_bucket"] == "uncertain"


def test_confidence_margin_matches_the_documented_formula():
    """Spot-check the exact arithmetic on a non-degenerate configuration."""

    segments = [
        _segment("seg_000", "SPEAKER_00"),
        _segment("seg_001", "SPEAKER_01"),
    ]
    # 60 degrees apart. Each speaker has one member, so each centroid is that
    # member. d_own = 0, d_other = 1 - cos(60) = 0.5 -> margin = 1.0.
    embeddings = {
        "seg_000": _unit(1.0, 0.0),
        "seg_001": _unit(math.cos(math.radians(60)), math.sin(math.radians(60))),
    }

    service._attach_confidence(segments, embeddings)

    for segment in segments:
        d_own, d_other = 0.0, 1.0 - math.cos(math.radians(60))
        expected = (d_other - d_own) / max(d_own, d_other)
        assert segment["confidence"] == pytest.approx(round(expected, 3))


def test_confidence_is_none_with_a_single_speaker():
    """Margin is undefined with one centroid -- it must be None, not 0.0."""

    segments = [_segment("seg_000", "SPEAKER_00"), _segment("seg_001", "SPEAKER_00")]
    embeddings = {"seg_000": _unit(1.0, 0.0), "seg_001": _unit(0.9, 0.1)}

    service._attach_confidence(segments, embeddings)

    for segment in segments:
        assert segment["confidence"] is None
        assert segment["confidence_bucket"] is None


def test_confidence_is_none_for_a_segment_without_an_embedding():
    segments = [
        _segment("seg_000", "SPEAKER_00"),
        _segment("seg_001", "SPEAKER_01"),
        _segment("seg_002", "SPEAKER_00"),  # too short to embed
    ]
    embeddings = {"seg_000": _unit(1.0, 0.0), "seg_001": _unit(0.0, 1.0)}

    service._attach_confidence(segments, embeddings)

    by_id = {s["id"]: s for s in segments}
    assert by_id["seg_002"]["confidence"] is None
    assert by_id["seg_002"]["confidence_bucket"] is None
    # ...and it is still present, with its timing intact.
    assert len(segments) == 3
    assert by_id["seg_002"]["speaker"] == "SPEAKER_00"
    # Its neighbours are still scored.
    assert by_id["seg_000"]["confidence"] is not None


def test_speaker_with_no_embeddable_segments_forms_no_centroid():
    """A speaker whose every segment was too short contributes no centroid, so
    the remaining speakers must not be scored against a phantom cluster."""

    segments = [
        _segment("seg_000", "SPEAKER_00"),
        _segment("seg_001", "SPEAKER_01"),  # only member, no embedding
    ]
    embeddings = {"seg_000": _unit(1.0, 0.0)}

    service._attach_confidence(segments, embeddings)

    by_id = {s["id"]: s for s in segments}
    # Only one centroid exists -> no "other" -> undefined margin.
    assert by_id["seg_000"]["confidence"] is None
    assert by_id["seg_001"]["confidence"] is None


def test_attach_confidence_mutates_in_place_and_returns_none():
    segments = [_segment("seg_000", "SPEAKER_00"), _segment("seg_001", "SPEAKER_01")]
    embeddings = {"seg_000": _unit(1.0, 0.0), "seg_001": _unit(0.0, 1.0)}

    assert service._attach_confidence(segments, embeddings) is None
    assert all("confidence" in segment for segment in segments)


def test_attach_confidence_handles_empty_input():
    segments: list[dict] = []
    service._attach_confidence(segments, {})
    assert segments == []


# ---------------------------------------------------------------------------
# The adapter: segment extraction and the <0.4 s embedding rule
# ---------------------------------------------------------------------------


class _FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    """Stands in for a `pyannote.core.Annotation`; only `itertracks` is used."""

    def __init__(self, turns: list[tuple[float, float, str]]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        assert yield_label, "the adapter must request labels"
        for start, end, speaker in self._turns:
            yield _FakeTurn(start, end), "_", speaker


class _FakePipeline:
    def __init__(self, annotation: _FakeAnnotation) -> None:
        self.annotation = annotation
        self.calls: list[dict] = []

    def __call__(self, payload):
        self.calls.append(payload)
        return self.annotation


class _FakeEmbeddingInference:
    """Returns one vector per call, cycling through `vectors`, and records the
    duration of every crop it was asked to embed."""

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.vectors = vectors
        self.crop_durations: list[float] = []

    def __call__(self, payload):
        waveform = payload["waveform"]
        self.crop_durations.append(waveform.shape[1] / payload["sample_rate"])
        return self.vectors[(len(self.crop_durations) - 1) % len(self.vectors)]


def _make_adapter(monkeypatch, turns, vectors, duration_seconds: float = 10.0):
    """A real `_Pyannote31Adapter` with its model-loading __init__ bypassed."""

    adapter = object.__new__(service._Pyannote31Adapter)
    adapter.spec = service.MODEL_SPECS["pyannote-3.1"]
    adapter.pipeline = _FakePipeline(_FakeAnnotation(turns))
    adapter.embedding_inference = _FakeEmbeddingInference(vectors)

    samples = int(duration_seconds * service.TARGET_SAMPLE_RATE)
    waveform = torch.zeros((1, samples), dtype=torch.float32)
    monkeypatch.setattr(
        service, "_load_waveform", lambda path: (waveform, service.TARGET_SAMPLE_RATE)
    )
    return adapter


def test_segments_shorter_than_the_minimum_are_not_embedded(monkeypatch):
    """The documented rule: < 0.4 s is skipped for embedding, and never gets a
    fabricated score. 0.40 s exactly is on the embeddable side of the line."""

    adapter = _make_adapter(
        monkeypatch,
        turns=[
            (0.10, 0.39, "SPEAKER_00"),  # 0.29 s -> skipped
            (1.00, 1.40, "SPEAKER_00"),  # see the float note below -> skipped
            (2.00, 3.00, "SPEAKER_01"),  # comfortably over -> embedded
        ],
        vectors=[
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
    )

    result = adapter.diarize_and_embed("ignored.wav")

    assert [s["id"] for s in result["segments"]] == ["seg_000", "seg_001", "seg_002"]
    assert "seg_000" not in result["embeddings"]
    assert "seg_002" in result["embeddings"]
    # Only the long-enough segment reached the embedding model.
    assert len(adapter.embedding_inference.crop_durations) == 1
    assert adapter.embedding_inference.crop_durations[0] == pytest.approx(1.0, abs=1e-6)


def test_a_segment_exactly_at_the_minimum_is_embedded(monkeypatch):
    """The comparison is `>=`, so a turn measuring exactly 0.4 s is embedded.

    Endpoints chosen so the subtraction is exact in IEEE 754 -- see
    `test_boundary_is_decided_by_the_float_difference` for why that matters.
    """

    adapter = _make_adapter(
        monkeypatch,
        turns=[(0.0, 0.4, "SPEAKER_00")],
        vectors=[np.array([1.0, 0.0], dtype=np.float32)],
    )

    result = adapter.diarize_and_embed("ignored.wav")

    assert "seg_000" in result["embeddings"]
    assert adapter.embedding_inference.crop_durations[0] == pytest.approx(0.4, abs=1e-6)


def test_boundary_is_decided_by_the_float_difference_not_the_decimal_one(monkeypatch):
    """Documents real, slightly surprising behaviour rather than asserting an
    idealised rule: `1.4 - 1.0` is 0.3999999999999999 in IEEE 754, so a turn
    that reads as 0.4 s in decimal falls *under* MIN_EMBEDDABLE_SECONDS while
    `0.4 - 0.0` (exactly 0.4) does not.

    This is a property of float subtraction, not of the threshold; it is
    harmless here (a 1 ms sliver either way of a 0.4 s cut-off changes nothing
    about the science) but it is the reason boundary tests must pick their
    endpoints deliberately.
    """

    assert (1.4 - 1.0) < service.MIN_EMBEDDABLE_SECONDS
    assert (0.4 - 0.0) >= service.MIN_EMBEDDABLE_SECONDS

    adapter = _make_adapter(
        monkeypatch,
        turns=[(1.0, 1.4, "SPEAKER_00"), (2.0, 2.4, "SPEAKER_01")],
        vectors=[np.array([1.0, 0.0], dtype=np.float32)],
    )

    result = adapter.diarize_and_embed("ignored.wav")

    assert result["embeddings"] == {}
    assert len(result["segments"]) == 2


def test_short_segments_end_up_with_null_confidence(monkeypatch):
    """End-to-end through `run_diarization`: skipped-for-embedding must surface
    as `confidence: None`, never a fabricated number."""

    adapter = _make_adapter(
        monkeypatch,
        turns=[
            (0.0, 0.20, "SPEAKER_00"),  # too short
            (1.0, 2.00, "SPEAKER_00"),
            (3.0, 4.00, "SPEAKER_01"),
        ],
        vectors=[
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ],
    )
    monkeypatch.setitem(service._MODEL_CACHE, "pyannote-3.1", adapter)

    result = service.run_diarization("pyannote-3.1", "ignored.wav")

    by_id = {s["id"]: s for s in result["segments"]}
    assert by_id["seg_000"]["confidence"] is None
    assert by_id["seg_000"]["confidence_bucket"] is None
    assert by_id["seg_001"]["confidence"] is not None
    assert by_id["seg_002"]["confidence"] is not None


def test_adapter_l2_normalizes_every_embedding(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        turns=[(0.0, 1.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")],
        vectors=[
            np.array([3.0, 4.0], dtype=np.float32),  # norm 5
            np.array([0.0, 7.0], dtype=np.float32),  # norm 7
        ],
    )

    result = adapter.diarize_and_embed("ignored.wav")

    assert result["embeddings"]["seg_000"] == pytest.approx([0.6, 0.8])
    for vector in result["embeddings"].values():
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([0.0, 0.0], dtype=np.float32),  # zero norm
        np.array([np.nan, 1.0], dtype=np.float32),
        np.array([np.inf, 1.0], dtype=np.float32),
    ],
)
def test_adapter_drops_unusable_embedding_vectors(monkeypatch, bad_vector):
    """A degenerate vector is dropped, not stored -- the segment survives with
    no embedding (and therefore no confidence)."""

    adapter = _make_adapter(
        monkeypatch,
        turns=[(0.0, 1.0, "SPEAKER_00")],
        vectors=[bad_vector],
    )

    result = adapter.diarize_and_embed("ignored.wav")

    assert len(result["segments"]) == 1
    assert result["embeddings"] == {}


def test_adapter_rounds_times_and_reports_duration(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        turns=[(0.123456, 1.987654, "SPEAKER_00")],
        vectors=[np.array([1.0, 0.0], dtype=np.float32)],
        duration_seconds=12.5,
    )

    result = adapter.diarize_and_embed("ignored.wav")

    assert result["segments"][0]["start"] == 0.12
    assert result["segments"][0]["end"] == 1.99
    assert result["duration"] == 12.5


def test_adapter_passes_the_loaded_waveform_to_the_pipeline(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        turns=[(0.0, 1.0, "SPEAKER_00")],
        vectors=[np.array([1.0, 0.0], dtype=np.float32)],
    )

    adapter.diarize_and_embed("ignored.wav")

    assert len(adapter.pipeline.calls) == 1
    payload = adapter.pipeline.calls[0]
    assert payload["sample_rate"] == service.TARGET_SAMPLE_RATE
    assert payload["waveform"].shape[0] == 1


def test_adapter_handles_a_recording_with_no_speech(monkeypatch):
    adapter = _make_adapter(monkeypatch, turns=[], vectors=[np.array([1.0], dtype=np.float32)])

    result = adapter.diarize_and_embed("ignored.wav")

    assert result["segments"] == []
    assert result["embeddings"] == {}


# ---------------------------------------------------------------------------
# run_diarization assembly
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_model_cache(monkeypatch):
    """Never leak a fake adapter into another test's `_MODEL_CACHE`."""

    monkeypatch.setattr(service, "_MODEL_CACHE", {})
    return service._MODEL_CACHE


def test_run_diarization_summarizes_speakers_and_rounds_embeddings(
    monkeypatch, isolated_model_cache
):
    adapter = _make_adapter(
        monkeypatch,
        turns=[
            (0.0, 1.0, "SPEAKER_01"),
            (2.0, 3.0, "SPEAKER_00"),
            (4.0, 5.0, "SPEAKER_01"),
        ],
        vectors=[
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([3.0, 2.0, 1.0], dtype=np.float32),
        ],
    )
    isolated_model_cache["pyannote-3.1"] = adapter

    result = service.run_diarization("pyannote-3.1", "ignored.wav")

    assert result["model"] == "pyannote-3.1"
    assert result["num_speakers"] == 2
    assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]  # sorted
    assert len(result["segments"]) == 3
    for vector in result["embeddings"].values():
        assert all(value == round(value, 5) for value in vector)


def test_run_diarization_rejects_an_unknown_model(isolated_model_cache):
    with pytest.raises(service.UnsupportedDiarizationModel):
        service.run_diarization("not-a-model", "ignored.wav")
    assert isolated_model_cache == {}


def test_get_model_reuses_a_cached_adapter(isolated_model_cache):
    sentinel = object()
    isolated_model_cache["pyannote-3.1"] = sentinel

    assert service.get_model("pyannote-3.1") is sentinel
    assert service.get_model("pyannote-3.1") is sentinel


def test_get_model_rejects_unknown_keys_before_touching_the_cache(isolated_model_cache):
    with pytest.raises(service.UnsupportedDiarizationModel):
        service.get_model("whisper-base")
    assert isolated_model_cache == {}


# ---------------------------------------------------------------------------
# Gated repos -- from_pretrained returns None instead of raising
# ---------------------------------------------------------------------------


@pytest.fixture
def gated_pipeline(monkeypatch):
    """Reproduce what a gated repo really does on the pinned pyannote 3.4.0:
    `Pipeline.from_pretrained` prints a help message and returns None. It does
    NOT raise, so the adapter's `except Exception` never fires."""

    import pyannote.audio

    monkeypatch.setattr(service.settings, "HF_TOKEN", "hf_fake_token")
    monkeypatch.setattr(
        pyannote.audio.Pipeline,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: None),
    )
    # The embedding model is a separate, ungated repo -- stub it so the test
    # never touches real weights.
    monkeypatch.setattr(
        pyannote.audio.Model,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: MagicMock()),
    )
    monkeypatch.setattr(pyannote.audio, "Inference", MagicMock())


@pytest.mark.parametrize("model_key", ALL_MODELS)
def test_a_gated_pipeline_raises_instead_of_building_a_dead_adapter(
    gated_pipeline, model_key
):
    """Without the guard the adapter builds with pipeline=None and dies minutes
    later at inference with "'NoneType' object is not callable" -- a 500 that
    says nothing about gating."""

    with pytest.raises(service.DiarizationModelUnavailable) as error:
        service._Pyannote31Adapter(service.MODEL_SPECS[model_key])

    message = str(error.value)
    assert service.MODEL_SPECS[model_key].pipeline_id in message
    assert "gated" in message.lower()
    assert "HF_TOKEN" in message
    # The outer handler must not have re-wrapped our own message into itself.
    assert message.count("Could not load") == 1


# ---------------------------------------------------------------------------
# Missing / blank HF_TOKEN -- the message has to say where we looked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank", [None, "", "   ", "\t\n"])
def test_a_blank_token_is_treated_as_missing(monkeypatch, blank):
    """A whitespace token must not reach Hugging Face as if it were real --
    that turns a local misconfiguration into an opaque 401 minutes later."""

    import pyannote.audio

    monkeypatch.setattr(service.settings, "HF_TOKEN", blank)
    # Fail loudly if the blank token ever gets past the guard.
    monkeypatch.setattr(
        pyannote.audio.Pipeline,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: pytest.fail("token guard was bypassed")),
    )

    with pytest.raises(service.DiarizationModelUnavailable) as error:
        service._Pyannote31Adapter(service.MODEL_SPECS["pyannote-3.1"])

    assert "HF_TOKEN" in str(error.value)


def test_the_missing_token_message_names_the_env_file_it_read(monkeypatch):
    monkeypatch.setattr(service.settings, "HF_TOKEN", None)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    message = service._missing_token_message()

    # Naming the actual path is the point: it separates "wrong file" from
    # "file missing" from "environment is shadowing it" without a bisect.
    assert str(service.settings.model_config["env_file"]) in message
    assert "not in the process environment" in message
    assert sys.executable in message


def test_the_missing_token_message_calls_out_a_shadowing_environment(monkeypatch):
    """pydantic-settings gives os.environ precedence over env_file, so an
    exported blank value silently beats a correct .env. That is the failure
    mode worth naming explicitly."""

    monkeypatch.setattr(service.settings, "HF_TOKEN", "")
    monkeypatch.setenv("HF_TOKEN", "   ")

    message = service._missing_token_message()

    assert "overrides the .env value" in message
    assert "unset hf_token" in message.lower()


def test_a_failed_load_is_never_cached(gated_pipeline, isolated_model_cache):
    """A dead adapter in `_MODEL_CACHE` would poison every later request until
    a restart, since `get_model` returns the cached object without re-checking."""

    for _ in range(2):
        with pytest.raises(service.DiarizationModelUnavailable):
            service.get_model("reverb-v2")
        assert isolated_model_cache == {}


# ---------------------------------------------------------------------------
# Content hashing -- the cache key behind every /run
# ---------------------------------------------------------------------------


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "clip.wav"
    payload = b"RIFF" + bytes(range(256)) * 4
    path.write_bytes(payload)

    assert service.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_file_sha256_is_chunk_boundary_independent(tmp_path):
    """The reader pulls 1 MiB at a time; a file spanning several chunks must
    hash identically to one-shot hashing."""

    path = tmp_path / "big.wav"
    payload = bytes(range(256)) * ((3 << 20) // 256 + 7)  # ~3 MiB, not chunk-aligned
    path.write_bytes(payload)

    assert service.file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_file_sha256_of_an_empty_file(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    assert service.file_sha256(path) == hashlib.sha256(b"").hexdigest()


def test_identical_bytes_hash_identically_under_different_names(tmp_path):
    """Why an upload of the same audio hits the cache regardless of its id."""

    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    first.write_bytes(b"same-bytes")
    second.write_bytes(b"same-bytes")

    assert service.file_sha256(first) == service.file_sha256(second)
