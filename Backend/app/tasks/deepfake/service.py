"""Audio Deepfake Detection — model loading and inference.

Model A is a Tier A checkpoint: it ships `config.json` +
`preprocessor_config.json`, so it loads through the standard transformers
audio-classification interface with no vendored architecture code. Heavy
imports are deferred until first model use; loading is thread-safe (the
`_load_once` idiom -- transformers weight loading is not thread-safe, and
concurrent loads return CORRUPTED WEIGHTS rather than an error; see the
comment in app/services/model_loader_service.py and commit 0c21127).

Which class index means "spoof" is read from the checkpoint's own
`config.json` at load time and never hardcoded -- getting it backwards
produces a detector that is confidently inverted with no error anywhere.
"""

from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

TARGET_SAMPLE_RATE = 16_000

# Defensive cap. ASVspoof LA clips are a few seconds; this only bites on a
# long clip and is reported back so a truncated score is never silent.
MAX_ANALYSIS_SECONDS = 30.0

# No EER calibration exists yet -- Feature 1 (score distribution / DET curve)
# is what will produce one. Until then this is the naive midpoint and every
# response says so. Bump the version when a calibrated threshold replaces it:
# it is part of the cache key, so old results are invalidated.
DEFAULT_THRESHOLD = 0.5
THRESHOLD_VERSION = "deepfake-threshold-v0-uncalibrated"


# ASTFeatureExtractor builds Kaldi-style fbank frames with a fixed 10 ms hop,
# so its `max_length` (in frames) converts to seconds at this rate. Used to
# report the real analysis window instead of the clip's full duration.
FRAMES_PER_SECOND = 100.0


@dataclass(frozen=True, slots=True)
class DeepfakeModelSpec:
    key: str
    label: str
    model_id: str
    revision: str
    sampling_rate: int
    threshold: float
    threshold_calibrated: bool
    recommended: bool
    # Hugging Face repos that require accepting conditions + an HF_TOKEN.
    gated: bool = False
    # "A" = self-describing checkpoint, loads through the shared transformers
    # path. "B" = weights only, needs architecture code we maintain ourselves
    # (see xlsr_mamba.py). Decides which adapter get_model builds.
    tier: str = "A"


MODEL_SPECS: dict[str, DeepfakeModelSpec] = {
    # Deliberately NOT "wav2vec2-xlsr-deepfake": the shared saliency service
    # dispatches on `"wav2vec" in model.lower()`
    # (app/services/saliency_service.py::detect_model_type), so an id carrying
    # that substring would silently route to the EMOTION model's saliency and
    # return emotion attributions labelled as deepfake attributions.
    "xlsr-deepfake": DeepfakeModelSpec(
        key="xlsr-deepfake",
        label="wav2vec2 XLS-R (Model A)",
        model_id="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        revision="main",
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=DEFAULT_THRESHOLD,
        threshold_calibrated=False,
        recommended=True,
        gated=False,
    ),
    # Model B — Audio Spectrogram Transformer. Reads a spectrogram image
    # rather than the waveform, so it fails differently from Model A; that
    # contrast is the point of running both. Same Tier A loading path.
    #
    # Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593 fine-tuned on
    # ASVspoof 2019 LA (public, MIT licence). It is the domain-specific AST
    # baseline in Huang & Hu, "Hybrid Audio Detection Using Fine-Tuned Audio
    # Spectrogram Transformers" (arXiv:2505.15136, 2025), where it scored
    # best of the three AST checkpoints on ASVspoof 2019 LA (89.65%). It
    # replaces WpythonW/ast-fakeaudio-detector, which is gated and whose
    # access request was never approved. The key is kept so the task
    # registry and frontend need no change.
    #
    # Pinned to a commit: a public repo can be re-pushed under "main" and
    # silently change every score. The id2label ({0: Bonafide, 1: Spoof}) is
    # still read at load time, never assumed.
    "ast-fakeaudio": DeepfakeModelSpec(
        key="ast-fakeaudio",
        label="Audio Spectrogram Transformer (Model B)",
        model_id="MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
        revision="afd7436a7205ebd7c9ae22d46bbbc6f4f04f85b5",
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=DEFAULT_THRESHOLD,
        threshold_calibrated=False,
        recommended=False,
        gated=False,
        tier="A",
    ),
    # Model C — XLSR-Mamba (Xiao & Das, IEEE SPL 2025). Tier B: the repo ships
    # weights and an EMPTY config.json, so the architecture lives in our own
    # xlsr_mamba.py. Runs on CPU because the selective scan is reimplemented
    # in pure PyTorch — the official mamba-ssm CUDA kernels cannot be built
    # or run on this platform.
    "xlsr-mamba": DeepfakeModelSpec(
        key="xlsr-mamba",
        label="XLSR-Mamba (Model C)",
        model_id="AustinXiao/XLSR-Mamba-LA",
        revision="main",
        sampling_rate=TARGET_SAMPLE_RATE,
        threshold=DEFAULT_THRESHOLD,
        threshold_calibrated=False,
        recommended=False,
        gated=False,
        tier="B",
    ),
}


class UnsupportedDeepfakeModel(ValueError):
    """Raised when the API receives a model key outside the task registry."""


class DeepfakeModelUnavailable(RuntimeError):
    """Raised when the checkpoint or its dependencies cannot load."""


class DeepfakeAdapter(Protocol):
    def score(self, audio_path: str | Path, with_embedding: bool = False) -> dict: ...


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_SPOOF_TOKENS = {"spoof", "spoofed", "fake", "deepfake", "synthetic", "generated"}
_BONAFIDE_TOKENS = {"bonafide", "real", "genuine", "human", "authentic", "original"}


def _tokenize_label(label: str) -> set[str]:
    normalized = str(label).strip().lower().replace("-", " ").replace("_", " ")
    normalized = normalized.replace("bona fide", "bonafide")
    return set(normalized.split())


def resolve_spoof_index(id2label: dict) -> int:
    """Return the class index meaning "spoof", read from the checkpoint config.

    Raises `DeepfakeModelUnavailable` naming the observed mapping rather than
    guessing: a wrong guess inverts every score the task will ever produce,
    and nothing downstream can detect it.
    """

    normalized = {int(index): label for index, label in id2label.items()}
    if len(normalized) != 2:
        raise DeepfakeModelUnavailable(
            "Expected a binary bona fide/spoof classifier, but the checkpoint "
            f"declares {len(normalized)} classes: {normalized}."
        )

    spoof_indices = {
        index
        for index, label in normalized.items()
        if _tokenize_label(label) & _SPOOF_TOKENS
    }
    bonafide_indices = {
        index
        for index, label in normalized.items()
        if _tokenize_label(label) & _BONAFIDE_TOKENS
    }

    if len(spoof_indices) == 1 and not (spoof_indices & bonafide_indices):
        return spoof_indices.pop()
    if len(bonafide_indices) == 1 and not spoof_indices:
        (bonafide_index,) = bonafide_indices
        return next(index for index in normalized if index != bonafide_index)

    raise DeepfakeModelUnavailable(
        "Could not tell which class means 'spoof' from the checkpoint's own "
        f"labels: {normalized}. Resolve the mapping from the model card and "
        "set it explicitly before trusting any score."
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def _load_waveform(audio_path: str | Path):
    """Load mono float32 waveform at 16 kHz. Returns (tensor[1, n], sr).

    Decoding goes through soundfile (libsndfile), NOT `torchaudio.load`:
    torchaudio >= 2.9 delegates loading to TorchCodec, which is not installed
    here and raises ImportError. libsndfile reads both the WAV uploads and the
    FLAC files ASVspoof ships. Resampling still uses torchaudio.functional,
    which is a pure tensor op and needs no codec.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except Exception as error:
        raise ValueError(f"Could not decode audio: {audio_path} ({error})") from error

    if samples.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    # soundfile returns (frames, channels); torch wants (channels, frames).
    waveform = torch.from_numpy(np.ascontiguousarray(samples.T))
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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


def _analysis_window_seconds(feature_extractor) -> float:
    """How much audio the model actually looks at, in seconds.

    AST-style extractors pad/truncate to a fixed `max_length` in frames
    (1024 -> 10.24 s), and do it SILENTLY. Reporting the clip's full duration
    for those would overstate what the score is based on. Wav2Vec2-style
    extractors have no such limit, so only our own defensive cap applies.
    """
    max_length = getattr(feature_extractor, "max_length", None)
    is_frame_based = getattr(feature_extractor, "num_mel_bins", None) is not None
    if max_length and is_frame_based:
        return min(MAX_ANALYSIS_SECONDS, float(max_length) / FRAMES_PER_SECOND)
    return MAX_ANALYSIS_SECONDS


def _load_failure_message(spec: DeepfakeModelSpec, error: Exception) -> str:
    """Turn a checkpoint load failure into something actionable.

    A gated repo returns a bare 401 that says nothing about what to do. No
    current checkpoint is gated, but repos can become gated after the fact,
    so the message is still detected from the error text.
    """
    text = str(error)
    looks_gated = "gated" in text.lower() or "401" in text or "restricted" in text.lower()
    if spec.gated or looks_gated:
        return (
            f"{spec.model_id} is a gated Hugging Face repo. Accept its "
            f"conditions at https://huggingface.co/{spec.model_id} with the "
            "same account, then put HF_TOKEN=<your token> in Backend/.env "
            "and restart the API. "
            f"(underlying error: {error})"
        )
    return f"Could not load {spec.model_id}: {error}"


@contextmanager
def _capture_head_input(head):
    """Yield a dict that receives the vector `head` is about to classify.

    The classification head's input is the representation the decision is
    actually read from (pooled, and for the transformers checkpoints already
    projected), so it is the natural point for an embedding view: two clips
    that land close together here are ones the detector cannot tell apart.
    Reading it through a forward pre-hook costs no extra pass -- it comes out
    of the same forward pass that produces the score.

    The hook only records calls made by the thread that opened it, so a
    concurrent plain `score()` on the same shared model can never leak its
    clip's vector into this one's. With `head=None` this is a no-op.
    """
    captured: dict = {}
    if head is None:
        yield captured
        return

    owner = threading.get_ident()

    def _record(_module, args):
        if threading.get_ident() == owner:
            captured["vector"] = args[0].detach()

    handle = head.register_forward_pre_hook(_record)
    try:
        yield captured
    finally:
        handle.remove()


def _embedding_from_capture(captured: dict, spec: DeepfakeModelSpec) -> list[float]:
    vector = captured.get("vector")
    if vector is None:
        raise DeepfakeModelUnavailable(
            f"{spec.model_id} exposes no classification head to read an embedding from."
        )
    return [round(float(v), 5) for v in vector.reshape(-1).cpu().tolist()]


class _HFAudioClassifierAdapter:
    """Wraps a standard transformers audio-classification checkpoint.

    Deliberately architecture-agnostic: Model A (wav2vec2 XLS-R, raw
    waveform) and Model B (AST, spectrogram patches) both load and run
    through this one class, because both are Tier A. The only thing that
    differs is the analysis window, which is read from the feature extractor
    rather than assumed.
    """

    def __init__(self, spec: DeepfakeModelSpec) -> None:
        self.spec = spec
        try:
            import torch
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )
        except ImportError as error:
            raise DeepfakeModelUnavailable(
                "Deepfake detection requires 'transformers' and 'torch'. "
                "Install the backend requirements and restart the API."
            ) from error

        from app.core.settings import settings

        # Only gated repos need a token; sending None keeps anonymous access
        # working for the public checkpoints.
        auth = {"token": settings.HF_TOKEN} if settings.HF_TOKEN else {}

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                spec.model_id, revision=spec.revision, **auth
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                spec.model_id, revision=spec.revision, **auth
            )
        except Exception as error:
            raise DeepfakeModelUnavailable(_load_failure_message(spec, error)) from error

        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.id2label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }
        self.spoof_index = resolve_spoof_index(self.id2label)
        self.bonafide_index = next(
            index for index in self.id2label if index != self.spoof_index
        )
        self.analysis_window_seconds = _analysis_window_seconds(self.feature_extractor)

    def score(self, audio_path: str | Path, with_embedding: bool = False) -> dict:
        import torch

        waveform, sample_rate = _load_waveform(audio_path)
        duration = waveform.shape[1] / sample_rate

        window = self.analysis_window_seconds
        max_samples = int(window * sample_rate)
        truncated = waveform.shape[1] > max_samples
        if truncated:
            waveform = waveform[:, :max_samples]

        inputs = self.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}

        head = getattr(self.model, "classifier", None) if with_embedding else None
        with torch.inference_mode(), _capture_head_input(head) as captured:
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

        result = {
            "spoof_probability": round(float(probabilities[self.spoof_index]), 6),
            "bonafide_probability": round(
                float(probabilities[self.bonafide_index]), 6
            ),
            "logits": [round(v, 6) for v in logits.squeeze(0).cpu().tolist()],
            "id2label": self.id2label,
            "spoof_index": self.spoof_index,
            "duration": round(duration, 2),
            "analysed_seconds": round(min(duration, window), 2),
            "analysis_window_seconds": round(window, 2),
            "truncated": truncated,
        }
        if with_embedding:
            result["embedding"] = _embedding_from_capture(captured, self.spec)
        return result


class _XLSRMambaAdapter:
    """Tier B adapter for Model C.

    Same `score()` contract as the Tier A adapter, but everything the
    transformers checkpoint would have described is supplied by us: the
    architecture (xlsr_mamba.XLSRMamba), the preprocessing (tile-pad to
    66800 samples, no normalisation) and the label order.
    """

    def __init__(self, spec: DeepfakeModelSpec) -> None:
        from app.core.settings import settings

        from . import xlsr_mamba

        self.spec = spec
        self._xlsr_mamba = xlsr_mamba
        try:
            self.model = xlsr_mamba.load_xlsr_mamba(
                spec.model_id, revision=spec.revision, token=settings.HF_TOKEN
            )
        except Exception as error:
            raise DeepfakeModelUnavailable(_load_failure_message(spec, error)) from error

        # NOT read from the checkpoint: its config.json is empty. Taken from
        # the reference training code, where genSpoof_list labels bona fide
        # as 1 and produce_evaluation_file scores with batch_out[:, 1].
        self.spoof_index = xlsr_mamba.SPOOF_INDEX
        self.bonafide_index = xlsr_mamba.BONAFIDE_INDEX
        self.id2label = {self.spoof_index: "spoof", self.bonafide_index: "bonafide"}
        self.analysis_window_seconds = (
            xlsr_mamba.EVAL_CUT_SAMPLES / float(spec.sampling_rate)
        )

    def score(self, audio_path: str | Path, with_embedding: bool = False) -> dict:
        import torch

        waveform, sample_rate = _load_waveform(audio_path)
        duration = waveform.shape[1] / sample_rate
        truncated = waveform.shape[1] > self._xlsr_mamba.EVAL_CUT_SAMPLES

        # Tile-repeat short clips rather than zero-padding them — the
        # reference does this, and zero padding scores differently.
        prepared = self._xlsr_mamba.pad_or_tile(waveform.squeeze(0).numpy())
        batch = torch.from_numpy(prepared).float().unsqueeze(0)

        head = self.model.conformer.classifier if with_embedding else None
        with torch.inference_mode(), _capture_head_input(head) as captured:
            logits = self.model(batch)

        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        result = {
            "spoof_probability": round(float(probabilities[self.spoof_index]), 6),
            "bonafide_probability": round(float(probabilities[self.bonafide_index]), 6),
            "logits": [round(v, 6) for v in logits.squeeze(0).tolist()],
            "id2label": self.id2label,
            "spoof_index": self.spoof_index,
            "duration": round(duration, 2),
            "analysed_seconds": round(min(duration, self.analysis_window_seconds), 2),
            "analysis_window_seconds": round(self.analysis_window_seconds, 2),
            "truncated": truncated,
        }
        if with_embedding:
            result["embedding"] = _embedding_from_capture(captured, self.spec)
        return result


# ---------------------------------------------------------------------------
# Cache + public API
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, DeepfakeAdapter] = {}
_LOAD_LOCK = threading.Lock()


def list_models() -> list[dict[str, object]]:
    return [asdict(spec) for spec in MODEL_SPECS.values()]


def get_model_spec(model_key: str) -> DeepfakeModelSpec:
    try:
        return MODEL_SPECS[model_key]
    except KeyError as error:
        valid = ", ".join(MODEL_SPECS)
        raise UnsupportedDeepfakeModel(
            f"Unsupported deepfake model '{model_key}'. Valid models: {valid}."
        ) from error


def get_model(model_key: str) -> DeepfakeAdapter:
    """Load a selected model once; concurrent first requests cannot race."""
    spec = get_model_spec(model_key)
    with _LOAD_LOCK:
        if model_key not in _MODEL_CACHE:
            builder = (
                _XLSRMambaAdapter if spec.tier == "B" else _HFAudioClassifierAdapter
            )
            _MODEL_CACHE[model_key] = builder(spec)
        return _MODEL_CACHE[model_key]


def run_detection(model_key: str, audio_path: str | Path) -> dict:
    """Score one clip and apply the task's operating threshold."""
    spec = get_model_spec(model_key)
    adapter = get_model(model_key)
    result = adapter.score(audio_path)

    spoof_probability = result["spoof_probability"]
    return {
        "model": spec.key,
        "model_label": spec.label,
        "model_id": spec.model_id,
        "decision": "spoof" if spoof_probability >= spec.threshold else "bonafide",
        "threshold": spec.threshold,
        "threshold_calibrated": spec.threshold_calibrated,
        "threshold_version": THRESHOLD_VERSION,
        **result,
    }


def run_embedding(model_key: str, audio_path: str | Path) -> dict:
    """Score one clip and return the vector its classification head read.

    One forward pass. The score comes back with it so the embedding view can
    colour each point by the detector's own opinion without a second pass.
    Returns {"embedding": [...], "spoof_probability": float}.
    """
    result = get_model(model_key).score(audio_path, with_embedding=True)
    return {
        "embedding": result["embedding"],
        "spoof_probability": result["spoof_probability"],
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
