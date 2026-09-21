"""Speaker Diarization — model loading, inference, and glass-box internals.

Three production pipelines (pyannote/speaker-diarization-3.1 and Rev's two
reverb-diarization checkpoints) behind one small adapter contract — all three
are full pyannote `Pipeline`s, so they differ only by `pipeline_id`. Heavy
imports are deferred until first model use; loading is thread-safe (the
`_load_once` idiom — transformers/torch weight loading is not thread-safe,
commit 0c21127).

Glass-box explainability: per-segment speaker embeddings are re-extracted
with pyannote/wespeaker-voxceleb-resnet34-LM — the SAME embedding model every
one of these pipelines uses internally for clustering — so confidence scores
and the similarity matrix live in the embedding space the pipeline actually
used. The Rev configs reach that checkpoint under their own mirror id,
Revai/pyannote-wespeaker-voxceleb-resnet34-LM, which is a byte-identical
re-upload (verified: same sha256, 218/218 tensors, max abs diff 0.0). So the
three models differ in segmentation only, and their confidences are directly
comparable.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.settings import settings

MIN_EMBEDDABLE_SECONDS = 0.4
CONFIDENCE_HIGH = 0.5
CONFIDENCE_MEDIUM = 0.2
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class DiarizationModelSpec:
    key: str
    label: str
    pipeline_id: str
    embedding_model_id: str
    embedding_dimension: int
    recommended: bool
    # Shown in the workbench header. States what this model actually changes,
    # so the UI never implies more isolation between the three than exists.
    glass_box_note: str


MODEL_SPECS: dict[str, DiarizationModelSpec] = {
    "pyannote-3.1": DiarizationModelSpec(
        key="pyannote-3.1",
        label="pyannote speaker-diarization-3.1",
        pipeline_id="pyannote/speaker-diarization-3.1",
        embedding_model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        embedding_dimension=256,
        recommended=True,
        glass_box_note=(
            "PyanNet segmentation. Embeddings and clustering use WeSpeaker "
            "ResNet34 — the pipeline's own space, so confidence and the "
            "similarity matrix reflect the decision it actually made."
        ),
    ),
    # Rev's two checkpoints fine-tune segmentation only; both configs cluster
    # with the same WeSpeaker weights as 3.1, so we keep pyannote's copy of it
    # for all three and the embedding space stays shared.
    "reverb-v1": DiarizationModelSpec(
        key="reverb-v1",
        label="Rev reverb-diarization-v1",
        pipeline_id="Revai/reverb-diarization-v1",
        embedding_model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        embedding_dimension=256,
        recommended=False,
        glass_box_note=(
            "Rev's PyanNet segmentation, fine-tuned on conversational speech. "
            "Clusters in the same WeSpeaker ResNet34 space as pyannote 3.1 "
            "(Rev mirrors that checkpoint bit-for-bit), so confidence is "
            "directly comparable."
        ),
    ),
    "reverb-v2": DiarizationModelSpec(
        key="reverb-v2",
        label="Rev reverb-diarization-v2",
        pipeline_id="Revai/reverb-diarization-v2",
        embedding_model_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        embedding_dimension=256,
        recommended=False,
        glass_box_note=(
            "Rev's WavLM-based SSeRiouSS segmentation. Same WeSpeaker ResNet34 "
            "embedding space as the others; roughly 2x slower on CPU."
        ),
    ),
}


class UnsupportedDiarizationModel(ValueError):
    """Raised when the API receives a model key outside the task registry."""


class DiarizationModelUnavailable(RuntimeError):
    """Raised when the gated pipeline or its dependencies cannot load."""


class DiarizationAdapter(Protocol):
    def diarize_and_embed(self, audio_path: str | Path) -> dict: ...


def _load_waveform(audio_path: str | Path):
    """Load mono float32 waveform at 16 kHz. Returns (tensor[1, n], sr)."""
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
    waveform = waveform.to(dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio contains invalid values: {audio_path}")
    return waveform, TARGET_SAMPLE_RATE


def _missing_token_message() -> str:
    """Explain a missing HF_TOKEN by reporting where we actually looked.

    The old message just asserted the token was unset, which sends you hunting
    through settings, .env and your shell in turn. Every fact needed to tell
    those apart is available here, so state them:

    * which env_file the settings object is configured with, and whether it is
      really on disk;
    * whether the process environment defines HF_TOKEN -- pydantic-settings
      gives os.environ precedence over env_file, so an exported blank value
      silently shadows a perfectly good .env;
    * which interpreter is running, which is what exposes a server started
      with the wrong uvicorn (bare `uvicorn` is not this project's .venv).
    """
    lines = [
        "HF_TOKEN is not set. The pyannote diarization models are gated: "
        "accept their conditions on Hugging Face and put "
        "HF_TOKEN=<your token> in Backend/.env."
    ]

    env_file = settings.model_config.get("env_file")
    if env_file is None:
        lines.append("Settings is configured with no env_file at all.")
    else:
        exists = Path(env_file).is_file()
        lines.append(
            f"Settings reads env_file {env_file} "
            f"({'found' if exists else 'MISSING -- create it'})."
        )

    environ_value = os.environ.get("HF_TOKEN")
    if environ_value is None:
        lines.append("HF_TOKEN is not in the process environment.")
    elif not environ_value.strip():
        lines.append(
            "HF_TOKEN IS in the process environment but blank, which "
            "overrides the .env value -- the environment wins over env_file. "
            "Unset it (`unset HF_TOKEN`) and restart."
        )
    else:
        lines.append(
            "HF_TOKEN is in the process environment with a value, so this "
            "process is not the one that read it -- check for a stale server."
        )

    lines.append(f"Running interpreter: {sys.executable}")
    return " ".join(lines)


class _Pyannote31Adapter:
    """Wraps the 3.1 pipeline + a standalone copy of its embedding model."""

    def __init__(self, spec: DiarizationModelSpec) -> None:
        self.spec = spec
        try:
            from pyannote.audio import Inference, Model, Pipeline
        except ImportError as error:
            raise DiarizationModelUnavailable(
                "Diarization requires the 'pyannote.audio' package. "
                "Install the backend requirements and restart the API."
            ) from error

        token = (settings.HF_TOKEN or "").strip()
        if not token:
            raise DiarizationModelUnavailable(_missing_token_message())

        # Same torch>=2.6 weights_only workaround as the verification task's
        # _WeSpeakerAdapter: trusted source (official gated pyannote repos).
        previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            pipeline = Pipeline.from_pretrained(
                spec.pipeline_id, use_auth_token=token
            )
            embedding_model = Model.from_pretrained(
                spec.embedding_model_id, use_auth_token=token
            )
        except Exception as error:
            raise DiarizationModelUnavailable(
                f"Could not load {spec.pipeline_id}: {error}"
            ) from error
        finally:
            if previous is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous

        # On a gated repo pyannote prints a help message and returns None
        # rather than raising, so the except clause above never fires. Without
        # this guard the adapter builds with self.pipeline = None and dies at
        # inference time with "'NoneType' object is not callable" -- minutes
        # into a run, as a 500 with no mention of gating. Raising here keeps it
        # a 503 that says what to do. Checked outside the try so the handler
        # above does not re-wrap this message into itself.
        if pipeline is None:
            raise DiarizationModelUnavailable(
                f"Could not load {spec.pipeline_id}: pyannote returned no "
                "pipeline. The model is gated -- accept its conditions on "
                "Hugging Face with the account behind HF_TOKEN."
            )
        self.pipeline = pipeline

        embedding_model.eval()
        self.embedding_inference = Inference(embedding_model, window="whole")

    def diarize_and_embed(self, audio_path: str | Path) -> dict:
        import torch

        waveform, sample_rate = _load_waveform(audio_path)
        duration = waveform.shape[1] / sample_rate

        annotation = self.pipeline(
            {"waveform": waveform, "sample_rate": sample_rate}
        )

        segments: list[dict] = []
        embeddings: dict[str, list[float]] = {}
        for index, (turn, _, speaker) in enumerate(
            annotation.itertracks(yield_label=True)
        ):
            segment_id = f"seg_{index:03d}"
            segment = {
                "id": segment_id,
                "start": round(float(turn.start), 2),
                "end": round(float(turn.end), 2),
                "speaker": str(speaker),
            }
            if (turn.end - turn.start) >= MIN_EMBEDDABLE_SECONDS:
                first = max(0, int(turn.start * sample_rate))
                last = min(waveform.shape[1], int(turn.end * sample_rate))
                crop = waveform[:, first:last]
                with torch.inference_mode():
                    vector = self.embedding_inference(
                        {"waveform": crop, "sample_rate": sample_rate}
                    )
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if np.isfinite(vector).all() and norm > 0.0:
                    embeddings[segment_id] = (vector / norm).tolist()
            segments.append(segment)

        return {
            "duration": round(duration, 2),
            "segments": segments,
            "embeddings": embeddings,
        }


_MODEL_CACHE: dict[str, DiarizationAdapter] = {}
_LOAD_LOCK = threading.Lock()


def list_models() -> list[dict[str, object]]:
    return [asdict(spec) for spec in MODEL_SPECS.values()]


def get_model_spec(model_key: str) -> DiarizationModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        valid = ", ".join(MODEL_SPECS)
        raise UnsupportedDiarizationModel(
            f"Unsupported diarization model '{model_key}'. Valid models: {valid}."
        ) from error


def get_model(model_key: str) -> DiarizationAdapter:
    """Load a selected model once; concurrent first requests cannot race."""
    spec = get_model_spec(model_key)
    with _LOAD_LOCK:
        if model_key not in _MODEL_CACHE:
            _MODEL_CACHE[model_key] = _Pyannote31Adapter(spec)
        return _MODEL_CACHE[model_key]


def _confidence_bucket(confidence: float | None) -> str | None:
    if confidence is None:
        return None
    if confidence >= CONFIDENCE_HIGH:
        return "high"
    if confidence >= CONFIDENCE_MEDIUM:
        return "medium"
    return "uncertain"


def _attach_confidence(segments: list[dict], embeddings: dict[str, list[float]]) -> None:
    """Silhouette-style margin per segment, in place.

    confidence = (d_nearest_other_centroid - d_own_centroid) / max(both)
    with cosine distance d = 1 - cos. Range -1..1; higher = more confidently
    assigned. None for segments with no embedding, or when there is only one
    speaker centroid (margin undefined).
    """
    by_speaker: dict[str, list[np.ndarray]] = {}
    for segment in segments:
        vector = embeddings.get(segment["id"])
        if vector is not None:
            by_speaker.setdefault(segment["speaker"], []).append(
                np.asarray(vector, dtype=np.float32)
            )

    centroids: dict[str, np.ndarray] = {}
    for speaker, vectors in by_speaker.items():
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0.0:
            centroids[speaker] = centroid / norm

    for segment in segments:
        vector = embeddings.get(segment["id"])
        own = centroids.get(segment["speaker"])
        others = [c for s, c in centroids.items() if s != segment["speaker"]]
        if vector is None or own is None or not others:
            segment["confidence"] = None
            segment["confidence_bucket"] = None
            continue
        vector = np.asarray(vector, dtype=np.float32)
        d_own = 1.0 - float(np.dot(vector, own))
        d_other = min(1.0 - float(np.dot(vector, other)) for other in others)
        denominator = max(d_own, d_other)
        confidence = 0.0 if denominator <= 0.0 else (d_other - d_own) / denominator
        segment["confidence"] = round(confidence, 3)
        segment["confidence_bucket"] = _confidence_bucket(confidence)


def run_diarization(model_key: str, audio_path: str | Path) -> dict:
    """Full glass-box run: diarize, embed segments, score confidence."""
    adapter = get_model(model_key)
    result = adapter.diarize_and_embed(audio_path)
    _attach_confidence(result["segments"], result["embeddings"])
    speakers = sorted({s["speaker"] for s in result["segments"]})
    return {
        "model": model_key,
        "duration": result["duration"],
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": result["segments"],
        "embeddings": {
            key: [round(v, 5) for v in vec]
            for key, vec in result["embeddings"].items()
        },
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()