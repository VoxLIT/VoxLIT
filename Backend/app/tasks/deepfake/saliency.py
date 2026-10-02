"""Feature 3 — waveform-aligned saliency.

Implements SRS DF-14 (a temporal attribution for the detector's decision,
rendered aligned with the waveform) and DF-15 (the method is named in the
interface, and the shared saliency duration caps apply).

WHAT IT ANSWERS
---------------
Which MOMENTS of the audio pushed the model toward "spoof". If a clip is
called synthetic but the bright regions sit on silence or a microphone click,
the score was never evidence about the voice — which is the same lesson
Feature 2 delivers by ablation, arrived at from the other direction.

METHOD
------
SmoothGrad x input on the decision margin (spoof logit minus bonafide logit),
taken from the loaded model itself: no dataset and no labels.

* Margin, not the spoof logit alone: on a clip the model calls real, the
  spoof logit's gradient shows where the model would look to become MORE
  fake, which is not why it decided. The margin is what the verdict reads.
* x input: a bare gradient is SENSITIVITY ("what if this sample changed"),
  not contribution. Near-silent stretches carry ~1% of the speech's
  amplitude yet the models' gradients there run 40-100% as large, so a bare
  gradient piles heat onto silence that holds no evidence. Multiplying by the
  input weighs each moment by what is actually there.
* SmoothGrad: the gradient is averaged over a few noisy copies of the input,
  because a single raw-waveform gradient is too spiky to read.

Magnitude is reported, because a researcher asks "did this moment matter",
not "which way did it push" -- the sign of a raw-waveform term flips with the
carrier phase, so it carries no interpretable direction here.

Known bias, stated rather than hidden: x input scales with loudness, so it
leans towards speech exactly as a bare gradient leans towards silence. Whether
the SCORE depends on the silence is answered by Feature 2's ablation, not by
this map. Integrated Gradients would be better founded but costs ~32 passes;
on a CPU-only machine Model C alone takes ~9 s per pass, so it is out of
reach. The method actually used is reported in the payload and shown in the
interface (DF-15) rather than being implied.

WHERE IT LIVES
--------------
DF-14 says "reusing the shared saliency service and visualisation". The
shared service (app/services/saliency_service.py) is marked frozen in
STRUCTURE.md, and the Software Architecture Document records that the
deepfake task carries its own preprocessing, loading and inference with no
common adapter -- so it cannot dispatch this task's models. What IS reused:
the shared service's duration cap (same env var, same default) and its exact
response contract, so the payload is interchangeable with the shared
visualisation's data shape.
"""

from __future__ import annotations

import os
from pathlib import Path

from .service import get_model, get_model_spec

# The SAME cap the shared saliency service applies, read from the same
# environment variable with the same default (SRS DF-15). Deliberately read
# rather than imported: importing app.services.saliency_service eagerly loads
# the emotion model as a side effect.
MAX_SALIENCY_SECONDS = int(os.getenv("MAX_SALIENCY_SECONDS", "12"))

# Enough resolution to see structure inside a word, few enough to stay legible
# on a waveform a few hundred pixels wide.
DEFAULT_SEGMENTS = 60

# Noisy passes averaged per clip. Each costs one forward AND backward pass.
SMOOTHGRAD_SAMPLES = int(os.getenv("DEEPFAKE_SMOOTHGRAD_SAMPLES", "8"))
# Noise standard deviation, as a fraction of the input's own standard deviation.
SMOOTHGRAD_NOISE = 0.1
# Fixed, so the same clip always yields the same map (results are cached).
SMOOTHGRAD_SEED = 0

# Bump the method name whenever the attribution's meaning changes: it is part
# of the result cache key.
METHOD = "smoothgrad-x-input-margin"
METHOD_LABEL = (
    f"SmoothGrad × input (|mean ∂margin/∂x · x|, {SMOOTHGRAD_SAMPLES} noisy passes)"
)
TARGET = "decision margin (spoof logit − bonafide logit)"


class SaliencyUnavailable(RuntimeError):
    """Raised when attribution cannot be produced for this model."""


def _smooth(values, window: int):
    """Moving average — raw sample gradients are far too spiky to read."""
    import numpy as np

    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _real_frame_count(samples: int, sample_rate: int) -> int:
    """Kaldi fbank frames a clip produces: 25 ms windows, 10 ms hop, no padding."""
    window = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    return 0 if samples < window else 1 + (samples - window) // hop


def _fold_tiled(values, real_samples: int):
    """Map a tile-padded input's per-sample values back onto the real clip.

    Tier B models tile a short clip until it fills their window, so sample i
    of the clip appears at i, i + n, i + 2n, ... Its total effect is the sum
    over those copies; dropping or stretching the copies would misplace it.
    """
    import numpy as np

    if values.shape[0] <= real_samples:
        return values
    copies = -(-values.shape[0] // real_samples)
    padded = np.zeros(copies * real_samples, dtype=values.dtype)
    padded[: values.shape[0]] = values
    return padded.reshape(copies, real_samples).sum(axis=0)


def _smoothgrad(forward, target, adapter):
    """Gradient of the decision margin, averaged over noisy copies of `target`."""
    import torch

    generator = torch.Generator().manual_seed(SMOOTHGRAD_SEED)
    sigma = SMOOTHGRAD_NOISE * float(target.detach().std())
    total = torch.zeros_like(target)
    for _ in range(max(1, SMOOTHGRAD_SAMPLES)):
        noise = torch.randn(target.shape, generator=generator).to(target.device) * sigma
        noisy = (target.detach() + noise).requires_grad_(True)
        logits = forward(noisy)
        adapter.model.zero_grad(set_to_none=True)
        margin = logits[0, adapter.spoof_index] - logits[0, adapter.bonafide_index]
        margin.backward()
        if noisy.grad is None:
            raise SaliencyUnavailable(
                "No gradient reached the model input, so no attribution can be produced."
            )
        total += noisy.grad.detach()
    return total / max(1, SMOOTHGRAD_SAMPLES)


def _attribution_over_time(adapter, audio_path: str | Path, max_seconds: float):
    """|SmoothGrad x input| of the decision margin, plus its time span.

    Returns (attribution per input position, seconds of audio analysed).
    Handles both tiers: a raw-waveform model gives one value per sample, a
    spectrogram model one per frame. Either way the result is a sequence in
    time order, which is all the caller needs.
    """
    import numpy as np
    import torch

    from .service import _load_waveform

    waveform, sample_rate = _load_waveform(audio_path)

    # Whichever is tighter: the shared saliency cap, or what the model itself
    # will look at (AST truncates to 10.24 s regardless).
    window = min(max_seconds, getattr(adapter, "analysis_window_seconds", max_seconds))
    max_samples = int(window * sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    analysed_seconds = waveform.shape[1] / sample_rate

    feature_extractor = getattr(adapter, "feature_extractor", None)

    if feature_extractor is not None:
        # Tier A — the checkpoint's own preprocessing builds the input.
        inputs = feature_extractor(
            waveform.squeeze(0).numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {name: value.to(adapter.device) for name, value in inputs.items()}
        target = inputs.pop("input_values")

        def forward(values):
            return adapter.model(input_values=values, **inputs).logits

    else:
        # Tier B — our own architecture, fed the raw waveform.
        target = adapter.prepare_input(waveform.squeeze(0).numpy())

        def forward(values):
            return adapter.model(values)

    gradient = _smoothgrad(forward, target, adapter).squeeze(0).cpu().numpy()
    values = target.detach().squeeze(0).cpu().numpy()

    if gradient.ndim > 1:
        # Spectrogram input (frames x mel bins): per-bin contribution, then
        # collapse the frequency axis, leaving one value per time frame.
        attribution = np.abs(gradient * values).sum(axis=-1)
        # AST pads every clip to a fixed frame count (1024 -> 10.24 s). Those
        # padding frames are not audio: keep only the frames the clip filled,
        # or a 3 s clip's attribution gets squeezed into its first second.
        real_frames = _real_frame_count(waveform.shape[1], sample_rate)
        attribution = attribution[: max(1, min(real_frames, attribution.shape[0]))]
    else:
        # A clip shorter than a Tier B window is tiled to fill it: fold every
        # copy's gradient back onto the original sample before weighting it,
        # or the copies get stretched over the real clip's timeline.
        real_samples = waveform.shape[1]
        gradient = _fold_tiled(gradient, real_samples)
        attribution = np.abs(gradient * values[:real_samples])

    if not np.isfinite(attribution).all():
        raise SaliencyUnavailable("Attribution contained non-finite values.")

    return attribution.astype(np.float32), analysed_seconds


def _to_segments(attribution, analysed_seconds: float, segment_count: int):
    """Average the attribution into equal time slices.

    Shape matches the shared saliency service exactly (start_time, end_time,
    saliency, intensity) so the payload is interchangeable with it.
    """
    import numpy as np

    positions = attribution.size
    if positions == 0 or analysed_seconds <= 0:
        return [], []

    # Smooth over roughly one segment's worth of input before slicing.
    attribution = _smooth(attribution, max(1, positions // max(segment_count, 1)))

    edges = np.linspace(0, positions, segment_count + 1, dtype=int)
    means = np.array(
        [
            float(attribution[start:end].mean()) if end > start else 0.0
            for start, end in zip(edges[:-1], edges[1:])
        ],
        dtype=np.float32,
    )

    # Normalise to 0..1 so the colour scale is readable. This is a RANKING
    # across this clip only -- attribution magnitudes are not comparable
    # between clips or between models.
    peak = float(means.max())
    normalised = means / peak if peak > 0 else means

    segment_seconds = analysed_seconds / segment_count
    segments = [
        {
            "start_time": round(index * segment_seconds, 4),
            "end_time": round((index + 1) * segment_seconds, 4),
            "saliency": round(float(normalised[index]), 6),
            "intensity": round(float(normalised[index]), 6),
        }
        for index in range(segment_count)
    ]
    return segments, [round(float(value), 6) for value in normalised]


def _speech_overlay(
    audio_path: str | Path, segments: list[dict], analysed_seconds: float
) -> dict:
    """Where the speech is, and how much attribution lands inside it.

    Reuses Feature 2's segmentation so both views draw the same boundaries.
    This is the number that answers the question the feature exists for: if
    the model says "spoof" but the attribution sits outside the speech, the
    score was never evidence about the voice.
    """
    from .service import _load_waveform
    from .silence_probe import SILENCE_TOP_DB, segment_speech

    waveform, sample_rate = _load_waveform(audio_path)
    segmentation = segment_speech(waveform.squeeze(0).numpy(), sample_rate)

    # Only the speech inside the analysed window: speech after the cut was never
    # attributed, and counting it would inflate the share the view compares to.
    intervals = [
        (start / sample_rate, min(end / sample_rate, analysed_seconds))
        for start, end in segmentation.speech_intervals
        if start / sample_rate < analysed_seconds
    ]

    def inside_speech(segment: dict) -> bool:
        midpoint = (segment["start_time"] + segment["end_time"]) / 2
        return any(start <= midpoint < end for start, end in intervals)

    total = sum(segment["saliency"] for segment in segments)
    in_speech = sum(segment["saliency"] for segment in segments if inside_speech(segment))

    return {
        "speech_intervals": [[round(start, 3), round(end, 3)] for start, end in intervals],
        "silence_top_db": SILENCE_TOP_DB,
        "saliency_in_speech_fraction": round(in_speech / total, 4) if total > 0 else None,
    }


def generate_saliency(
    model_key: str,
    audio_path: str | Path,
    segment_count: int = DEFAULT_SEGMENTS,
) -> dict:
    """Temporal attribution for one clip, in the shared saliency contract."""
    spec = get_model_spec(model_key)
    adapter = get_model(model_key)

    attribution, analysed_seconds = _attribution_over_time(
        adapter, audio_path, MAX_SALIENCY_SECONDS
    )
    segments, series = _to_segments(attribution, analysed_seconds, segment_count)

    if not segments:
        raise SaliencyUnavailable("The clip was too short to attribute.")

    return {
        # --- shared saliency service contract ---
        "model": model_key,
        "method": METHOD,
        "segments": segments,
        "total_duration": round(analysed_seconds, 4),
        "series": series,
        # --- deepfake specifics ---
        "model_label": spec.label,
        # DF-15: the method is named, not implied.
        "method_label": METHOD_LABEL,
        "target": TARGET,
        "smoothgrad_samples": SMOOTHGRAD_SAMPLES,
        "max_saliency_seconds": MAX_SALIENCY_SECONDS,
        "analysis_window_seconds": getattr(adapter, "analysis_window_seconds", None),
        "truncated": analysed_seconds < _clip_seconds(audio_path),
        "normalised": True,
        # Lets the view show whether the heat sits on speech or on silence.
        **_speech_overlay(audio_path, segments, analysed_seconds),
    }


def _clip_seconds(audio_path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.frames / float(info.samplerate)
