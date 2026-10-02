"""Feature 2 — the silence and non-speech probe.

Implements SRS DF-10 (score the same clip three ways and present the three
scores together), DF-11 (identify non-speech by an energy threshold relative
to the clip's OWN level, and report the threshold used) and DF-12 (say the
probe is not applicable rather than reporting an unreliable silence-only
score).

WHY THIS VIEW EXISTS
--------------------
In ASVspoof 2019 LA, genuine clips carry markedly longer silence than most
attacks: TTS systems trim their silences, real recordings do not. That gap is
a property of the CORPUS, not of speech, and a detector can score well by
reading it. Müller et al. trained detectors on nothing but the silence and
they still "worked".

So this is the one view that can invalidate everything else on the screen,
including Feature 1's EER — if trimming the silence collapses the score, or
silence alone still says "spoof", the number was never evidence about the
voice. It is a direct ablation, which returns an unambiguous answer where an
attribution map only hints.

Three forward passes, no gradients. Nothing shared is touched.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .service import TARGET_SAMPLE_RATE, _load_waveform, run_detection

# Relative to the clip's own peak, so it adapts to recording level rather
# than assuming an absolute noise floor (SRS DF-11). 30 dB is the value the
# feature was specified with.
SILENCE_TOP_DB = 30

# Part of the result cache key: bump when the payload's meaning changes.
# v2 scores short variants (flagged unreliable) instead of withholding them.
SILENCE_PROBE_VERSION = "v2"

# Below this a variant is still scored, but flagged as low-reliability: half a
# second is the shortest stretch the probe was specified to trust (SRS DF-12).
# The score is reported rather than withheld so a reader can weigh it.
MIN_NON_SPEECH_SECONDS = 0.5

# The same reliability floor applies to the trimmed version: a clip that is
# almost all silence leaves little speech to score.
MIN_SPEECH_SECONDS = 0.5

# Below this there is no usable signal at all — wav2vec2's feature encoder
# alone needs 400 samples (25 ms) — so the variant cannot be scored and the
# probe says so, with the measured length.
MIN_SCORABLE_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Where the speech is, and how much non-speech surrounds it."""

    speech_intervals: list[tuple[int, int]]
    sample_rate: int
    total_samples: int

    @property
    def speech_samples(self) -> int:
        return sum(end - start for start, end in self.speech_intervals)

    @property
    def non_speech_samples(self) -> int:
        return self.total_samples - self.speech_samples

    def seconds(self, samples: int) -> float:
        return samples / float(self.sample_rate)


def segment_speech(waveform, sample_rate: int) -> Segmentation:
    """Split into speech intervals using an energy threshold (SRS DF-11).

    `librosa.effects.split` measures against the clip's own peak, which is
    what makes the threshold relative rather than absolute.
    """
    import librosa

    intervals = librosa.effects.split(waveform, top_db=SILENCE_TOP_DB)
    return Segmentation(
        speech_intervals=[(int(start), int(end)) for start, end in intervals],
        sample_rate=sample_rate,
        total_samples=int(waveform.shape[0]),
    )


def build_variants(waveform, segmentation: Segmentation):
    """The two ablations, derived from one segmentation so they agree.

    trimmed      — leading and trailing silence removed (internal pauses kept,
                   because removing them would splice the speech itself).
    non_speech   — every non-speech region concatenated: the leading and
                   trailing silence AND the pauses between words. DF-10 asks
                   for "the non-speech portions", not only the outer ones.
    """
    import numpy as np

    if not segmentation.speech_intervals:
        return None, None

    first_start = segmentation.speech_intervals[0][0]
    last_end = segmentation.speech_intervals[-1][1]
    trimmed = waveform[first_start:last_end]

    gaps: list = []
    cursor = 0
    for start, end in segmentation.speech_intervals:
        if start > cursor:
            gaps.append(waveform[cursor:start])
        cursor = end
    if cursor < segmentation.total_samples:
        gaps.append(waveform[cursor:])

    non_speech = np.concatenate(gaps) if gaps else np.empty(0, dtype=waveform.dtype)
    return trimmed, non_speech


def _score_samples(model_key: str, samples, sample_rate: int, directory: Path, name: str) -> dict:
    """Score a raw waveform by writing it out and reusing the normal path.

    Going through a real file keeps every model on exactly the decoding and
    preprocessing it uses in production — Model C's tile-padding included.
    """
    import soundfile as sf

    path = directory / f"{name}.wav"
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return run_detection(model_key, path)


def _variant_payload(result: dict, seconds: float, reliable: bool = True, reason: str | None = None) -> dict:
    payload = {
        "applicable": True,
        "reliable": reliable,
        "seconds": round(seconds, 3),
        "spoof_probability": result["spoof_probability"],
        "decision": result["decision"],
    }
    return {**payload, "reason": reason} if reason else payload


def _not_applicable(seconds: float, reason: str) -> dict:
    return {
        "applicable": False,
        "reliable": False,
        "seconds": round(seconds, 3),
        "spoof_probability": None,
        "decision": None,
        "reason": reason,
    }


def _score_variant(
    model_key: str, samples, seconds: float, reliable_from: float, sample_rate: int,
    workspace: Path, name: str, what: str,
) -> dict:
    """Score one ablation, or explain with the measured length why it can't be."""
    if samples is None or seconds < MIN_SCORABLE_SECONDS:
        return _not_applicable(
            seconds,
            f"{seconds:.3f}s of {what} audio was found; at least {MIN_SCORABLE_SECONDS:.3f}s "
            "is needed to produce any score.",
        )
    result = _score_samples(model_key, samples, sample_rate, workspace, name)
    if seconds < reliable_from:
        return _variant_payload(
            result, seconds, reliable=False,
            reason=(
                f"Scored on only {seconds:.3f}s of {what} audio (below the {reliable_from:.2f}s "
                "reliability floor), so treat this score as indicative."
            ),
        )
    return _variant_payload(result, seconds)


def trimmed_waveform(audio_path: str | Path):
    """(clip with outer silence removed, sample rate) — the evaluation's ablation.

    The same segmentation as the per-clip probe, so the report's
    "silence trimmed" condition and the probe's middle column agree exactly.
    Falls back to the whole clip when no speech is detected.
    """
    waveform_tensor, sample_rate = _load_waveform(audio_path)
    waveform = waveform_tensor.squeeze(0).numpy()
    trimmed, _ = build_variants(waveform, segment_speech(waveform, sample_rate))
    if trimmed is None or len(trimmed) / float(sample_rate) < MIN_SCORABLE_SECONDS:
        trimmed = waveform
    return trimmed, sample_rate


def score_trimmed(model_key: str, audio_path: str | Path) -> dict:
    """run_detection on the silence-trimmed clip. Blocking."""
    trimmed, sample_rate = trimmed_waveform(audio_path)
    with tempfile.TemporaryDirectory(prefix="voxlit-trimmed-") as directory:
        return _score_samples(model_key, trimmed, sample_rate, Path(directory), "trimmed")


def run_silence_probe(model_key: str, audio_path: str | Path) -> dict:
    """Score the clip as submitted, trimmed, and as non-speech only."""
    waveform_tensor, sample_rate = _load_waveform(audio_path)
    waveform = waveform_tensor.squeeze(0).numpy()

    segmentation = segment_speech(waveform, sample_rate)
    trimmed, non_speech = build_variants(waveform, segmentation)

    speech_seconds = segmentation.seconds(segmentation.speech_samples)
    non_speech_seconds = segmentation.seconds(segmentation.non_speech_samples)
    total_seconds = segmentation.seconds(segmentation.total_samples)

    with tempfile.TemporaryDirectory(prefix="voxlit-silence-probe-") as directory:
        workspace = Path(directory)

        original = _score_samples(model_key, waveform, sample_rate, workspace, "original")

        trimmed_seconds = segmentation.seconds(len(trimmed)) if trimmed is not None else 0.0
        trimmed_payload = _score_variant(
            model_key, trimmed, trimmed_seconds, MIN_SPEECH_SECONDS, sample_rate, workspace, "trimmed", "speech"
        )
        non_speech_payload = _score_variant(
            model_key, non_speech, non_speech_seconds, MIN_NON_SPEECH_SECONDS, sample_rate, workspace,
            "non_speech", "non-speech",
        )

    return {
        "model": original["model"],
        "model_label": original["model_label"],
        "threshold": original["threshold"],
        "threshold_calibrated": original["threshold_calibrated"],
        # SRS DF-11 — the threshold that defined "silence" travels with the result.
        "silence_top_db": SILENCE_TOP_DB,
        "min_non_speech_seconds": MIN_NON_SPEECH_SECONDS,
        "min_speech_seconds": MIN_SPEECH_SECONDS,
        "min_scorable_seconds": MIN_SCORABLE_SECONDS,
        "duration": round(total_seconds, 3),
        "speech_seconds": round(speech_seconds, 3),
        "non_speech_seconds": round(non_speech_seconds, 3),
        "non_speech_fraction": (
            round(segmentation.non_speech_samples / segmentation.total_samples, 4)
            if segmentation.total_samples
            else 0.0
        ),
        "speech_intervals": [
            [round(start / sample_rate, 3), round(end / sample_rate, 3)]
            for start, end in segmentation.speech_intervals
        ],
        # DF-10 — the three scores, presented together.
        "variants": {
            "original": _variant_payload(original, total_seconds),
            "trimmed": trimmed_payload,
            "non_speech": non_speech_payload,
        },
        "sample_rate": TARGET_SAMPLE_RATE,
    }
