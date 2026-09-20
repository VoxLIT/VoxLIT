"""Diff two diarization runs: the original and a perturbed counterfactual.

This is the only genuinely new algorithm in the perturbation feature. The
perturbed audio is produced by `perturbation.py` and diarized by the existing
`/run` cache-through path; everything here operates on the two resulting
payloads, so it is pure and fast (milliseconds) regardless of how long the
inference took.

**The DER computed here is between two runs, not against ground truth.** The
original run is passed as pyannote's "reference" purely to anchor the
comparison; it is not a gold annotation. The AMI RTTM files remain
offline-evaluation-only and are never read by this module or any runtime
endpoint. Every field name and label says "vs. the original run" for that
reason.

Speaker labels are arbitrary per run -- pyannote may call the same person
SPEAKER_00 in one run and SPEAKER_01 in the next -- so nothing can be compared
until the two label sets are aligned. `pyannote.metrics` does that alignment
with the Hungarian algorithm (`optimal_mapping`), maximising total
co-occurrence duration.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

# A boundary has to move by more than this to count as having moved at all.
BOUNDARY_SHIFT_SECONDS = 0.5

# A perturbed label must hold at least this share of an original speaker's
# overlap mass before the change is called a split (and vice versa for a
# merge). Without a floor, a sliver of incidental overlap would be reported as
# a structural change on almost every run.
SPLIT_SHARE_MIN = 0.20

# Resolution of the difference strip. 400 bins is finer than the rendered
# pixel width of the timeline at any realistic viewport, so the strip never
# shows a difference the timeline itself cannot.
DIFF_BINS = 400


def _to_annotation(segments: list[dict], uri: str) -> Annotation:
    """Track name is the segment id, not an auto-assigned index: two segments
    with identical extents (genuinely overlapping speech) would otherwise
    collide on the same track and silently overwrite each other."""

    annotation = Annotation(uri=uri)
    for segment in segments:
        annotation[Segment(segment["start"], segment["end"]), segment["id"]] = segment["speaker"]
    return annotation


def _overlap_matrix(
    original: Annotation, perturbed: Annotation
) -> dict[str, dict[str, float]]:
    """overlap[original_label][perturbed_label] = seconds of temporal overlap.

    `optimal_mapping` alone cannot express a merge or a split, because it is a
    bijection on matched labels. This matrix is what makes those detectable.
    """

    overlap: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (seg_o, track_o), (seg_p, track_p) in original.co_iter(perturbed):
        intersection = seg_o & seg_p
        if intersection.duration <= 0:
            continue
        overlap[original[seg_o, track_o]][perturbed[seg_p, track_p]] += intersection.duration
    return {key: dict(value) for key, value in overlap.items()}


def _detect_splits(overlap: dict[str, dict[str, float]]) -> list[dict]:
    """One original speaker whose speech is now attributed to two or more
    perturbed speakers, each holding a meaningful share."""

    splits = []
    for original_label, by_perturbed in sorted(overlap.items()):
        total = sum(by_perturbed.values())
        if total <= 0:
            continue
        significant = sorted(
            label for label, seconds in by_perturbed.items() if seconds / total >= SPLIT_SHARE_MIN
        )
        if len(significant) >= 2:
            splits.append({"original": original_label, "into": significant})
    return splits


def _detect_merges(overlap: dict[str, dict[str, float]]) -> list[dict]:
    """The mirror image: one perturbed speaker now absorbing speech that two
    or more original speakers used to own."""

    by_perturbed: dict[str, dict[str, float]] = defaultdict(dict)
    for original_label, mapping in overlap.items():
        for perturbed_label, seconds in mapping.items():
            by_perturbed[perturbed_label][original_label] = seconds

    merges = []
    for perturbed_label, by_original in sorted(by_perturbed.items()):
        total = sum(by_original.values())
        if total <= 0:
            continue
        significant = sorted(
            label for label, seconds in by_original.items() if seconds / total >= SPLIT_SHARE_MIN
        )
        if len(significant) >= 2:
            merges.append({"perturbed": perturbed_label, "from": significant})
    return merges


def _boundary_shifts(
    original_segments: list[dict],
    perturbed_segments: list[dict],
    mapping: dict[str, str],
) -> tuple[list[dict], list[str]]:
    """For each original segment, find the perturbed segment that best covers
    it *among those assigned to the same speaker after mapping*, then measure
    how far each of its two boundaries moved.

    An original segment with no counterpart at all is NOT reported as a
    shifted boundary -- it is a lost segment. Counting a vanished segment as a
    boundary shift would overstate how stable the timeline is.
    """

    # Perturbed segments, expressed in the original run's label space.
    translated = [
        {**segment, "mapped_speaker": mapping.get(segment["speaker"])}
        for segment in perturbed_segments
    ]

    shifts: list[dict] = []
    lost: list[str] = []

    for segment in original_segments:
        start, end = segment["start"], segment["end"]
        best = None
        best_overlap = 0.0
        for candidate in translated:
            if candidate["mapped_speaker"] != segment["speaker"]:
                continue
            covered = min(end, candidate["end"]) - max(start, candidate["start"])
            if covered > best_overlap:
                best_overlap = covered
                best = candidate

        if best is None or best_overlap <= 0:
            lost.append(segment["id"])
            continue

        for edge, original_value, perturbed_value in (
            ("start", start, best["start"]),
            ("end", end, best["end"]),
        ):
            delta = perturbed_value - original_value
            if abs(delta) > BOUNDARY_SHIFT_SECONDS:
                shifts.append(
                    {
                        "segment_id": segment["id"],
                        "edge": edge,
                        "original": round(original_value, 2),
                        "perturbed": round(perturbed_value, 2),
                        "delta": round(delta, 2),
                    }
                )

    return shifts, lost


def _diff_regions(
    original: Annotation, perturbed: Annotation, mapping: dict[str, str], duration: float
) -> list[dict]:
    """Time spans where the two runs disagree about who is speaking, after
    label alignment. Computed server-side and returned as merged intervals so
    the UI's difference strip is a straight render with no client-side
    reconstruction of the comparison.
    """

    if duration <= 0:
        return []

    # Rename into the original's label space. An unmatched perturbed label has
    # no counterpart, so it keeps a name that cannot collide with any original
    # label -- it *should* read as a disagreement everywhere it appears.
    translated = perturbed.rename_labels(
        {label: mapping.get(label, f"__unmatched__{label}") for label in perturbed.labels()}
    )

    bin_width = duration / DIFF_BINS
    differing: list[tuple[float, float]] = []
    for index in range(DIFF_BINS):
        start = index * bin_width
        bin_segment = Segment(start, start + bin_width)
        if set(original.crop(bin_segment).labels()) != set(translated.crop(bin_segment).labels()):
            differing.append((start, start + bin_width))

    if not differing:
        return []

    merged = [list(differing[0])]
    for start, end in differing[1:]:
        # Float bin edges are computed identically on both sides, so touching
        # bins compare equal; a tolerance guards against accumulated drift.
        if start - merged[-1][1] <= bin_width * 1e-6:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [{"start": round(start, 2), "end": round(end, 2)} for start, end in merged]


def compare_runs(original_result: dict, perturbed_result: dict) -> dict:
    """Build the delta between an original diarization run and its perturbed
    counterfactual. Pure and fast; safe to call inside a threadpool."""

    original_segments = original_result["segments"]
    perturbed_segments = perturbed_result["segments"]

    original_annotation = _to_annotation(original_segments, uri="original")
    perturbed_annotation = _to_annotation(perturbed_segments, uri="perturbed")

    # Both supported transforms preserve length, so the two runs share one
    # evaluation region. (A future length-changing transform would need
    # min() of the two durations here.) Passing an explicit UEM also stops
    # pyannote approximating one from the union of extents.
    duration = float(original_result["duration"])
    uem = Timeline([Segment(0.0, duration)])

    metric = DiarizationErrorRate()
    # Hungarian assignment maximising co-occurrence: {perturbed -> original}.
    # Only matched labels get an entry, which is exactly what makes the
    # appeared/disappeared sets below fall out of the gaps.
    mapping = dict(metric.optimal_mapping(original_annotation, perturbed_annotation, uem=uem))

    components = metric(original_annotation, perturbed_annotation, uem=uem, detailed=True)

    original_labels = set(original_annotation.labels())
    perturbed_labels = set(perturbed_annotation.labels())
    appeared = sorted(perturbed_labels - set(mapping.keys()))
    disappeared = sorted(original_labels - set(mapping.values()))

    overlap = _overlap_matrix(original_annotation, perturbed_annotation)
    shifts, lost = _boundary_shifts(original_segments, perturbed_segments, mapping)

    return {
        # Named to make it impossible to read as accuracy against ground truth.
        "der": round(float(components["diarization error rate"]), 4),
        "der_is_vs_original_run": True,
        "der_components": {
            "missed_detection": round(float(components["missed detection"]), 2),
            "false_alarm": round(float(components["false alarm"]), 2),
            "confusion": round(float(components["confusion"]), 2),
            "total": round(float(components["total"]), 2),
        },
        "speaker_mapping": mapping,
        "appeared": appeared,
        "disappeared": disappeared,
        "merged": _detect_merges(overlap),
        "split": _detect_splits(overlap),
        "num_speakers_delta": len(perturbed_labels) - len(original_labels),
        "boundary_shifts": {
            "threshold_seconds": BOUNDARY_SHIFT_SECONDS,
            "count": len(shifts),
            "shifts": shifts,
        },
        "lost_segments": lost,
        "diff_regions": _diff_regions(
            original_annotation, perturbed_annotation, mapping, duration
        ),
    }
