"""Perturbation counterfactuals for Speaker Diarization.

Applies exactly one perturbation to one recording, writes the degraded clip
into the caller's session storage, and hands the path back so the existing
`/run` cache-through path can diarize it. The diff between the two runs lives
in `diff.py`; this module only produces the perturbed audio.

Scope is deliberately two transforms -- `noise` and `time_masking`. Both
preserve length and sample rate, which is what lets `diff.py` compare the two
runs over a single shared evaluation region.

Three correctness properties this module exists to guarantee:

* **Reproducibility is not optional here.** `add_gaussian_noise` is the only
  stochastic transform, and its `seed` drives a request-scoped
  `torch.Generator` (never global RNG state). Identical audio + type + params
  therefore produce byte-identical output, which is the whole reason a repeat
  of the same perturbation can hit the diarization cache instead of paying
  minutes of CPU inference again. This is also why `pertubation_service`'s own
  `perturb_and_save()`/`apply_perturbations()` are NOT used: they call
  `add_gaussian_noise` without a seed (pertubation_service.py:297), so every
  repeat would produce different bytes, a different content hash, and a
  permanent cache miss.
* **A silent no-op must never read as robustness.** Several transforms in
  `pertubation_service` fall back to returning the original waveform on
  internal error or a below-threshold parameter, with no flag anywhere.
  `_waveform_changed` + `PerturbationNotApplied` catch that, so "the timeline
  did not move" can never be an artifact of a transform that never ran.
* **A bad request leaves nothing on disk.** Validate, load, perturb, check,
  and only then write the output file.

The strict-parameter pattern here (`extra="forbid"`, bool rejection,
non-finite rejection) mirrors `tasks/verification/robustness.py`, which is this
repo's reference implementation. It is copied rather than imported because
each task is self-contained.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.services import pertubation_service

from .uploads import PERTURBED_ID_PREFIX, validate_and_canonicalize_sid, _session_dir

DEFAULT_NOISE_SEED = 0


class PerturbationNotApplied(ValueError):
    """The transform ran but produced audio identical to the input -- almost
    always `pertubation_service`'s own silent-fallback path rather than a
    genuinely robust model."""


class PerturbationOutputInvalid(ValueError):
    """The transform produced a non-tensor, empty, non-numeric, or non-finite
    waveform."""


class _StrictParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _reject_bool_and_nonfinite(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid numeric parameter")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("parameter must be finite (no NaN/Infinity)")
        return value


class NoiseParams(_StrictParams):
    noise_level: float = Field(gt=0, le=0.5)
    seed: int = Field(default=DEFAULT_NOISE_SEED, ge=0, le=2**32 - 1)


class TimeMaskingParams(_StrictParams):
    mask_start_percent: float = Field(ge=0, le=100)
    mask_end_percent: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _check_order(self) -> "TimeMaskingParams":
        if self.mask_start_percent >= self.mask_end_percent:
            raise ValueError("mask_start_percent must be less than mask_end_percent")
        return self


PARAM_MODELS: dict[str, type[_StrictParams]] = {
    "noise": NoiseParams,
    "time_masking": TimeMaskingParams,
}


def validate_perturbation_params(perturbation_type: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Exactly the keys applicable to `perturbation_type`; nothing missing,
    nothing extra, no bools-as-numbers, no NaN/Infinity. Returns the
    normalized params actually used -- including the noise seed, so the
    response always carries enough to reproduce the result exactly."""

    model_cls = PARAM_MODELS.get(perturbation_type)
    if model_cls is None:
        valid = ", ".join(PARAM_MODELS)
        raise ValueError(
            f"Unsupported perturbation type '{perturbation_type}'. Valid types: {valid}."
        )
    try:
        parsed = model_cls.model_validate(raw_params)
    except ValidationError as error:
        raise ValueError(str(error)) from error
    return parsed.model_dump()


def _apply_single_perturbation(
    waveform: torch.Tensor, perturbation_type: str, params: dict[str, Any]
) -> torch.Tensor:
    """Dispatch through the `pertubation_service` module (not bound function
    references), so test-time monkeypatching of its silent-fallback path is
    always observed."""

    if perturbation_type == "noise":
        return pertubation_service.add_gaussian_noise(
            waveform, params["noise_level"], seed=params["seed"]
        )
    if perturbation_type == "time_masking":
        return pertubation_service.apply_time_masking(
            waveform, params["mask_start_percent"], params["mask_end_percent"]
        )
    raise ValueError(f"Unsupported perturbation type '{perturbation_type}'.")


def _validate_output_waveform(tensor: object) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise PerturbationOutputInvalid("Perturbation produced a non-tensor output.")
    if tensor.numel() == 0:
        raise PerturbationOutputInvalid("Perturbation produced an empty waveform.")
    if not torch.is_floating_point(tensor):
        raise PerturbationOutputInvalid("Perturbation produced a non-numeric waveform.")
    if not torch.isfinite(tensor).all():
        raise PerturbationOutputInvalid("Perturbation produced a non-finite waveform.")


def _waveform_changed(original: torch.Tensor, perturbed: torch.Tensor) -> bool:
    """Exact content comparison, not a tolerance-based one: a small but
    genuine perturbation (e.g. a very low noise_level) must never be
    misclassified as a no-op because of a loose atol. Shape mismatch always
    counts as changed. Dtype is never downcast for the comparison -- differing
    dtypes are promoted, so precision can only increase, never be erased."""

    a = original.detach().cpu()
    b = perturbed.detach().cpu()
    if a.shape != b.shape:
        return True
    if a.dtype != b.dtype:
        common_dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=common_dtype)
        b = b.to(dtype=common_dtype)
    return not torch.equal(a, b)


def _load_mono_waveform(audio_path: str | Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.numel() == 0:
        raise ValueError("Audio file is empty.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate


def perturbed_id_for(
    source_audio_hash: str, perturbation_type: str, normalized_params: dict[str, Any]
) -> str:
    """Deterministic id for (this exact audio, this exact perturbation).

    Because every transform here is reproducible, the same request always
    names the same file -- so a repeat costs one `stat()` instead of a
    re-perturb, and the diarization behind it is already cached under the
    perturbed file's own content hash. Params are serialized canonically
    (sorted keys, no whitespace) so dict ordering can never split one logical
    perturbation across two ids.
    """

    canonical = json.dumps(
        {"type": perturbation_type, "params": normalized_params},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(f"{source_audio_hash}:{canonical}".encode("utf-8")).hexdigest()
    return f"{PERTURBED_ID_PREFIX}{digest[:32]}"


def perturb_to_session(
    source_path: str | Path,
    sid: str,
    source_audio_hash: str,
    perturbation_type: str,
    raw_params: dict[str, Any],
) -> tuple[str, Path, dict[str, Any]]:
    """Validate, apply one perturbation, and write the result into the
    caller's session directory. Returns `(perturbed_id, path,
    normalized_params)`.

    Sync and blocking (audio IO + transform); callers thread-pool it.

    If the deterministic target path already exists, the audio is neither
    re-perturbed nor rewritten -- the existing file is returned as-is. That is
    safe precisely because the id is derived from the source content hash plus
    the canonical params, so an existing file can only be the byte-identical
    result of this same request.
    """

    validated_sid = validate_and_canonicalize_sid(sid)
    normalized_params = validate_perturbation_params(perturbation_type, raw_params)
    perturbed_id = perturbed_id_for(source_audio_hash, perturbation_type, normalized_params)
    final_path = _session_dir(validated_sid) / f"{perturbed_id}.wav"

    if final_path.exists():
        return perturbed_id, final_path, normalized_params

    waveform, sample_rate = _load_mono_waveform(source_path)
    perturbed_waveform = _apply_single_perturbation(waveform, perturbation_type, normalized_params)
    _validate_output_waveform(perturbed_waveform)
    if not _waveform_changed(waveform, perturbed_waveform):
        raise PerturbationNotApplied(
            f"The requested '{perturbation_type}' perturbation did not change the audio. "
            "This means the transform silently failed, not that the model is robust to it."
        )

    # Write via a temp file in the same directory, then rename: a crash or a
    # concurrent identical request can never leave a half-written file at the
    # deterministic path where the `exists()` check above would trust it.
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".tmp-{perturbed_id}.wav")
    try:
        torchaudio.save(str(temp_path), perturbed_waveform, sample_rate)
        temp_path.replace(final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return perturbed_id, final_path, normalized_params
