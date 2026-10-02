"""Feature 1 — batch-score the labelled dataset and assemble the view.

This is the ONLY place in the task that reads the bona fide/spoof answers
(`dataset.load_ground_truth`). It returns aggregates — two distributions, a
DET curve, an EER — and deliberately never returns a per-recording label, so
the workbench's per-clip exercise ("read the score, then check the protocol")
survives. See dataset.py's module docstring.

Cost: one forward pass per clip. Per-clip scores are cached under the SAME
Redis key the `/run` endpoint uses, so an evaluation warms up single-clip
lookups and vice versa, and a re-run after a failure resumes almost free.
"""

from __future__ import annotations

from starlette.concurrency import run_in_threadpool

from app.core.redis import cache_result, get_result

from . import metrics
from .dataset import (
    DATASET_ID,
    list_recordings,
    load_ground_truth,
    resolve_recording_path,
)
from .service import (
    THRESHOLD_VERSION,
    file_sha256,
    get_model_spec,
    run_detection,
)
from .silence_probe import SILENCE_TOP_DB, score_trimmed

# The two evaluation conditions. "silence_trimmed" re-scores every clip with
# its leading and trailing non-speech removed (the silence probe's middle
# column), the ablation Müller et al. (ASVspoof 2021 Workshop, "Speech is Silver,
# Silence is Golden") used to show that ASVspoof 2019 LA detectors can score
# well by reading silence length. Reporting both is what makes the EER
# defensible: the gap between them is the size of the shortcut.
CONDITIONS = {
    "as_distributed": "As distributed (clips unmodified)",
    "silence_trimmed": f"Silence trimmed (outer non-speech below -{SILENCE_TOP_DB} dB removed)",
}

# Demo files are static, so scores stay valid for as long as the model and
# threshold version do.
SCORE_TTL_SECONDS = 7 * 24 * 60 * 60
HISTOGRAM_BINS = 20


def per_clip_cache_key(model_key: str) -> str:
    """Shared with the /run endpoint — same key, same payload."""
    return f"df:{model_key}:{THRESHOLD_VERSION}"


def trimmed_cache_key(model_key: str) -> str:
    return f"df-trimmed:{model_key}:{THRESHOLD_VERSION}:{SILENCE_TOP_DB}"


async def _score_one(model_key: str, path, condition: str = "as_distributed") -> dict:
    audio_hash = await run_in_threadpool(file_sha256, path)
    trimmed = condition == "silence_trimmed"
    cache_key = trimmed_cache_key(model_key) if trimmed else per_clip_cache_key(model_key)

    cached = await get_result(cache_key, audio_hash)
    if cached is not None:
        return cached

    payload = await run_in_threadpool(score_trimmed if trimmed else run_detection, model_key, path)
    await cache_result(cache_key, audio_hash, payload, ttl=SCORE_TTL_SECONDS)
    return payload


def _attack_summary(rows: list[dict]) -> list[dict]:
    """Mean score per spoofing system, plus the genuine clips for contrast.

    The subset is built balanced across A07-A19 precisely so this is
    readable; a detector that collapses on one generator shows up here and
    nowhere else in the view.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        label = "bonafide" if row["label"] == "bonafide" else row["attack"]
        grouped.setdefault(label, []).append(row["score"])

    summary = [
        {
            "attack": name,
            "count": len(scores),
            "mean_score": round(sum(scores) / len(scores), 4),
            "is_spoof": name != "bonafide",
        }
        for name, scores in grouped.items()
    ]
    # Genuine first, then attacks in their catalogue order.
    summary.sort(key=lambda row: (row["is_spoof"], row["attack"]))
    return summary


def builtin_clips() -> tuple[list[tuple[str, object]], dict[str, tuple[str, str]], str]:
    """The ASVspoof subset as (stem, path) pairs, its protocol, and its id."""
    clips = [
        (recording.display_filename.rsplit(".", 1)[0], resolve_recording_path(recording.recording_id))
        for recording in list_recordings()
    ]
    return clips, load_ground_truth(), DATASET_ID


async def evaluate_dataset(
    model_key: str,
    bins: int = HISTOGRAM_BINS,
    condition: str = "as_distributed",
    source: tuple[list[tuple[str, object]], dict[str, tuple[str, str]], str] | None = None,
) -> dict:
    """Score every labelled recording and assemble Feature 1's payload.

    `source` defaults to the built-in subset; a custom dataset passes its own
    clips and label map (see custom_datasets.labelled_clips).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of: {', '.join(CONDITIONS)}.")
    spec = get_model_spec(model_key)
    clips, truth, dataset_id = source if source is not None else builtin_clips()

    rows: list[dict] = []
    for stem, path in clips:
        entry = truth.get(stem) or truth.get(stem.lower())
        if entry is None:
            # A clip with no protocol line cannot be scored against anything.
            continue
        attack, label = entry
        payload = await _score_one(model_key, path, condition)
        rows.append({"label": label, "attack": attack, "score": payload["spoof_probability"]})

    bonafide = [row["score"] for row in rows if row["label"] == "bonafide"]
    spoof = [row["score"] for row in rows if row["label"] != "bonafide"]

    # Raises NotEnoughLabelledData if either class is missing — the router
    # turns that into a 422 rather than reporting a meaningless EER.
    computed = metrics.compute_metrics(bonafide, spoof)

    operating_far, operating_frr = metrics.rates_at(spec.threshold, bonafide, spoof)
    ci_low, ci_high = metrics.bootstrap_eer_interval(bonafide, spoof)
    zero_bound = metrics.zero_error_upper_bound(len(bonafide), len(spoof))

    return {
        "model": model_key,
        "model_label": spec.label,
        "dataset_id": dataset_id,
        "condition": condition,
        "condition_label": CONDITIONS[condition],
        "scored": len(rows),
        "unlabelled_skipped": len(clips) - len(rows),
        "bonafide_count": computed.bonafide_count,
        "spoof_count": computed.spoof_count,
        "eer_percent": computed.eer_percent,
        "eer_threshold": computed.eer_threshold,
        # Uncertainty. With n clips per class each error moves a rate by 1/n,
        # so the EER is only resolved to half of that.
        "eer_ci_percent": [round(ci_low * 100, 3), round(ci_high * 100, 3)],
        "eer_zero_upper_bound_percent": round(zero_bound * 100, 3),
        "eer_resolution_percent": round(50.0 / min(len(bonafide), len(spoof)), 3),
        "confidence_level": metrics.CONFIDENCE,
        "bootstrap_resamples": metrics.BOOTSTRAP_RESAMPLES,
        "roc_auc": round(metrics.roc_auc(bonafide, spoof), 6),
        "score_statistics": {
            "bonafide": metrics.score_summary(bonafide),
            "spoof": metrics.score_summary(spoof),
        },
        "confusion": {
            "operating": metrics.confusion_at(spec.threshold, bonafide, spoof),
            "eer": metrics.confusion_at(computed.eer_threshold, bonafide, spoof),
        },
        # SRS DF-9: a threshold is only meaningful with its dataset attached.
        "threshold_provenance": (
            f"Equal error rate on {dataset_id} "
            f"({computed.bonafide_count} genuine / {computed.spoof_count} spoofed clips, "
            f"{CONDITIONS[condition].lower()}). Thresholds do not transfer between datasets."
        ),
        "distributions": {
            "bonafide": metrics.score_histogram(bonafide, bins=bins),
            "spoof": metrics.score_histogram(spoof, bins=bins),
        },
        "det_curve": [
            {
                "threshold": round(point.threshold, 6),
                "false_acceptance_rate": round(point.false_acceptance_rate, 6),
                "false_rejection_rate": round(point.false_rejection_rate, 6),
            }
            for point in computed.det_curve
        ],
        # Where the shipped threshold actually sits, so the gap between the
        # current operating point and the EER point is visible (SRS DF-2).
        "operating_point": {
            "threshold": spec.threshold,
            "calibrated": spec.threshold_calibrated,
            "false_acceptance_rate": round(operating_far, 6),
            "false_rejection_rate": round(operating_frr, 6),
        },
        "per_attack": _attack_summary(rows),
    }
