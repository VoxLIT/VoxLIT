"""The run-vs-run diff behind the perturbation counterfactuals (task_b
`diff.py`): Hungarian label alignment, DER between two runs, boundary shifts,
and merge/split detection.

Nothing here is mocked. `compare_runs` is pure and operates on two already-
computed diarization payloads, so these tests feed it hand-written synthetic
annotations and assert against real `pyannote.metrics` output — no audio, no
model, no AMI data, milliseconds per test.

**The DER asserted here is between two runs, never against ground truth.** The
original run is passed as pyannote's "reference" purely to anchor the
comparison. The AMI RTTMs are offline-evaluation-only and are not read by this
module or these tests.
"""

import pytest

from app.tasks.task_b import diff


def _segment(segment_id: str, start: float, end: float, speaker: str) -> dict:
    return {
        "id": segment_id,
        "start": start,
        "end": end,
        "speaker": speaker,
        "confidence": 0.8,
        "confidence_bucket": "high",
    }


def _run(segments: list[dict], duration: float = 20.0) -> dict:
    speakers = sorted({s["speaker"] for s in segments})
    return {
        "model": "pyannote-3.1",
        "duration": duration,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "segments": segments,
        "embeddings": {},
    }


# Two speakers alternating, well separated in time.
BASELINE = [
    _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
    _segment("seg_001", 5.0, 9.0, "SPEAKER_01"),
    _segment("seg_002", 10.0, 14.0, "SPEAKER_00"),
    _segment("seg_003", 15.0, 19.0, "SPEAKER_01"),
]


def _relabel(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    return [{**s, "speaker": mapping.get(s["speaker"], s["speaker"])} for s in segments]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_thresholds_are_the_documented_values():
    assert diff.BOUNDARY_SHIFT_SECONDS == 0.5
    assert diff.SPLIT_SHARE_MIN == 0.20
    assert diff.DIFF_BINS == 400


# ---------------------------------------------------------------------------
# Identical runs -- the null case every other assertion is measured against
# ---------------------------------------------------------------------------


def test_identical_runs_produce_a_zero_delta():
    delta = diff.compare_runs(_run(BASELINE), _run(BASELINE))

    assert delta["der"] == 0.0
    assert delta["der_components"]["confusion"] == 0.0
    assert delta["der_components"]["missed_detection"] == 0.0
    assert delta["der_components"]["false_alarm"] == 0.0
    assert delta["appeared"] == []
    assert delta["disappeared"] == []
    assert delta["merged"] == []
    assert delta["split"] == []
    assert delta["num_speakers_delta"] == 0
    assert delta["boundary_shifts"]["count"] == 0
    assert delta["boundary_shifts"]["shifts"] == []
    assert delta["lost_segments"] == []
    assert delta["diff_regions"] == []


def test_identical_runs_map_each_speaker_to_itself():
    delta = diff.compare_runs(_run(BASELINE), _run(BASELINE))

    assert delta["speaker_mapping"] == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}


def test_der_is_always_labelled_as_being_against_the_original_run():
    """Guards the one thing that must never be misread as accuracy."""

    delta = diff.compare_runs(_run(BASELINE), _run(BASELINE))

    assert delta["der_is_vs_original_run"] is True
    assert "der" in delta
    assert "accuracy" not in delta


def test_two_empty_runs_do_not_crash():
    delta = diff.compare_runs(_run([]), _run([]))

    assert delta["speaker_mapping"] == {}
    assert delta["appeared"] == []
    assert delta["disappeared"] == []
    assert delta["boundary_shifts"]["count"] == 0


# ---------------------------------------------------------------------------
# Hungarian label alignment
# ---------------------------------------------------------------------------


def test_swapped_labels_are_realigned_and_cost_nothing():
    """pyannote's labels are arbitrary per run: the same person can be
    SPEAKER_00 in one run and SPEAKER_01 in the next. Without Hungarian
    matching this identical timeline would score a catastrophic DER."""

    swapped = _relabel(BASELINE, {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})

    delta = diff.compare_runs(_run(BASELINE), _run(swapped))

    assert delta["speaker_mapping"] == {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"}
    assert delta["der"] == 0.0
    assert delta["appeared"] == []
    assert delta["disappeared"] == []
    assert delta["diff_regions"] == []
    assert delta["boundary_shifts"]["count"] == 0


def test_alignment_survives_completely_foreign_label_names():
    renamed = _relabel(BASELINE, {"SPEAKER_00": "spk_alpha", "SPEAKER_01": "spk_beta"})

    delta = diff.compare_runs(_run(BASELINE), _run(renamed))

    assert delta["speaker_mapping"] == {"spk_alpha": "SPEAKER_00", "spk_beta": "SPEAKER_01"}
    assert delta["der"] == 0.0
    assert delta["appeared"] == []
    assert delta["disappeared"] == []


def test_mapping_is_keyed_by_perturbed_label():
    """`optimal_mapping` returns {perturbed -> original}; the appeared and
    disappeared sets below are derived from that direction, so it matters."""

    renamed = _relabel(BASELINE, {"SPEAKER_00": "X", "SPEAKER_01": "Y"})

    delta = diff.compare_runs(_run(BASELINE), _run(renamed))

    assert set(delta["speaker_mapping"].keys()) == {"X", "Y"}
    assert set(delta["speaker_mapping"].values()) == {"SPEAKER_00", "SPEAKER_01"}


# ---------------------------------------------------------------------------
# DER between the two runs
# ---------------------------------------------------------------------------


def test_a_dropped_speaker_raises_der_and_is_reported_as_disappeared():
    perturbed = [s for s in BASELINE if s["speaker"] != "SPEAKER_01"]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["disappeared"] == ["SPEAKER_01"]
    assert delta["appeared"] == []
    assert delta["num_speakers_delta"] == -1
    # SPEAKER_01 held 8 s of the 16 s of speech in the original run.
    assert delta["der_components"]["missed_detection"] == pytest.approx(8.0)
    assert delta["der_components"]["total"] == pytest.approx(16.0)
    assert delta["der"] == pytest.approx(0.5)


def test_der_matches_its_own_components():
    """DER is (missed + false alarm + confusion) / total; asserting the
    identity catches a components/rate mismatch in either direction."""

    perturbed = [
        _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
        _segment("seg_001", 5.0, 9.0, "SPEAKER_00"),  # was SPEAKER_01 -> confusion
        _segment("seg_002", 10.0, 14.0, "SPEAKER_00"),
        _segment("seg_003", 15.0, 19.0, "SPEAKER_01"),
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))
    components = delta["der_components"]

    expected = (
        components["missed_detection"] + components["false_alarm"] + components["confusion"]
    ) / components["total"]
    assert delta["der"] == pytest.approx(expected, abs=1e-3)
    assert components["confusion"] > 0


def test_extra_speech_in_the_perturbed_run_is_a_false_alarm():
    perturbed = BASELINE + [_segment("seg_004", 19.2, 19.9, "SPEAKER_00")]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["der_components"]["false_alarm"] == pytest.approx(0.7, abs=1e-6)
    assert delta["der"] > 0


def test_a_brand_new_speaker_is_reported_as_appeared():
    perturbed = BASELINE + [_segment("seg_004", 19.0, 19.9, "SPEAKER_02")]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["appeared"] == ["SPEAKER_02"]
    assert delta["disappeared"] == []
    assert delta["num_speakers_delta"] == 1


# ---------------------------------------------------------------------------
# Boundary shifts
# ---------------------------------------------------------------------------


def test_a_boundary_moving_more_than_the_threshold_is_reported():
    perturbed = [
        _segment("seg_000", 0.0, 4.6, "SPEAKER_00"),  # end +0.6 -> reported
        *BASELINE[1:],
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    shifts = delta["boundary_shifts"]["shifts"]
    assert delta["boundary_shifts"]["count"] == 1
    assert delta["boundary_shifts"]["threshold_seconds"] == 0.5
    assert shifts[0] == {
        "segment_id": "seg_000",
        "edge": "end",
        "original": 4.0,
        "perturbed": 4.6,
        "delta": 0.6,
    }


def test_a_boundary_moving_less_than_the_threshold_is_not_reported():
    perturbed = [_segment("seg_000", 0.0, 4.4, "SPEAKER_00"), *BASELINE[1:]]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["boundary_shifts"]["count"] == 0


def test_a_boundary_moving_exactly_the_threshold_is_not_reported():
    """The comparison is strict (`> 0.5`), so exactly 0.5 s does not count."""

    perturbed = [_segment("seg_000", 0.0, 4.5, "SPEAKER_00"), *BASELINE[1:]]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["boundary_shifts"]["count"] == 0


def test_both_edges_of_one_segment_can_shift():
    perturbed = [_segment("seg_000", 0.8, 4.8, "SPEAKER_00"), *BASELINE[1:]]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    shifts = delta["boundary_shifts"]["shifts"]
    assert delta["boundary_shifts"]["count"] == 2
    assert {s["edge"] for s in shifts} == {"start", "end"}
    assert all(s["segment_id"] == "seg_000" for s in shifts)


def test_a_negative_shift_keeps_its_sign():
    perturbed = [_segment("seg_000", 0.0, 3.2, "SPEAKER_00"), *BASELINE[1:]]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["boundary_shifts"]["shifts"][0]["delta"] == pytest.approx(-0.8)


def test_a_vanished_segment_is_lost_not_a_boundary_shift():
    """Counting a segment that disappeared entirely as a "shifted boundary"
    would overstate how stable the timeline is."""

    perturbed = [s for s in BASELINE if s["id"] != "seg_002"]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["lost_segments"] == ["seg_002"]
    assert delta["boundary_shifts"]["count"] == 0
    assert all(s["segment_id"] != "seg_002" for s in delta["boundary_shifts"]["shifts"])


def test_a_segment_reassigned_to_another_speaker_counts_as_lost():
    """Its counterpart must match on speaker *after* mapping, not just in time."""

    perturbed = [
        {**s, "speaker": "SPEAKER_01"} if s["id"] == "seg_000" else s for s in BASELINE
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert "seg_000" in delta["lost_segments"]


def test_boundary_shifts_are_measured_after_label_alignment():
    """A swapped-label run with one moved boundary must still find the right
    counterpart -- the mapping has to be applied before matching."""

    swapped = _relabel(BASELINE, {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"})
    swapped = [
        {**s, "end": 4.7} if s["id"] == "seg_000" else s for s in swapped
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(swapped))

    assert delta["lost_segments"] == []
    assert delta["boundary_shifts"]["count"] == 1
    assert delta["boundary_shifts"]["shifts"][0]["segment_id"] == "seg_000"


# ---------------------------------------------------------------------------
# Splits and merges
# ---------------------------------------------------------------------------


def test_a_split_speaker_is_detected():
    """SPEAKER_00's two turns are now attributed to two different speakers,
    each holding half the mass."""

    perturbed = [
        _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
        _segment("seg_001", 5.0, 9.0, "SPEAKER_01"),
        _segment("seg_002", 10.0, 14.0, "SPEAKER_02"),  # was SPEAKER_00
        _segment("seg_003", 15.0, 19.0, "SPEAKER_01"),
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["split"] == [{"original": "SPEAKER_00", "into": ["SPEAKER_00", "SPEAKER_02"]}]
    assert delta["num_speakers_delta"] == 1


def test_a_sliver_of_overlap_does_not_count_as_a_split():
    """Below SPLIT_SHARE_MIN (20%), a stray re-attribution is noise, not a
    structural change -- otherwise almost every run would report a split."""

    perturbed = [
        _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
        _segment("seg_001", 5.0, 9.0, "SPEAKER_01"),
        _segment("seg_002", 10.0, 13.5, "SPEAKER_00"),
        _segment("seg_004", 13.5, 14.0, "SPEAKER_02"),  # 0.5 s of 8.0 s = 6%
        _segment("seg_003", 15.0, 19.0, "SPEAKER_01"),
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["split"] == []


def test_a_merge_is_detected():
    """The mirror image: one perturbed speaker absorbs two original ones."""

    perturbed = [{**s, "speaker": "SPEAKER_00"} for s in BASELINE]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["merged"] == [
        {"perturbed": "SPEAKER_00", "from": ["SPEAKER_00", "SPEAKER_01"]}
    ]
    assert delta["num_speakers_delta"] == -1
    assert delta["disappeared"] == ["SPEAKER_01"]


def test_a_sliver_of_overlap_does_not_count_as_a_merge():
    perturbed = [
        _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
        _segment("seg_001", 5.0, 8.5, "SPEAKER_01"),
        _segment("seg_004", 8.5, 9.0, "SPEAKER_00"),  # 0.5 s against 8.0 s of SPEAKER_00
        _segment("seg_002", 10.0, 14.0, "SPEAKER_00"),
        _segment("seg_003", 15.0, 19.0, "SPEAKER_01"),
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["merged"] == []


def test_a_clean_run_reports_neither_merge_nor_split():
    perturbed = [{**s, "end": s["end"] + 0.1} for s in BASELINE]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["merged"] == []
    assert delta["split"] == []


# ---------------------------------------------------------------------------
# Difference regions (the red strip between the stacked timelines)
# ---------------------------------------------------------------------------


def test_diff_regions_cover_only_where_the_runs_disagree():
    perturbed = [s for s in BASELINE if s["id"] != "seg_002"]  # 10.0 - 14.0 removed

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))
    regions = delta["diff_regions"]

    assert len(regions) == 1
    assert regions[0]["start"] == pytest.approx(10.0, abs=0.1)
    assert regions[0]["end"] == pytest.approx(14.0, abs=0.1)


def test_diff_regions_are_merged_ordered_and_within_the_duration():
    perturbed = [s for s in BASELINE if s["id"] not in {"seg_000", "seg_002"}]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))
    regions = delta["diff_regions"]

    assert len(regions) == 2  # two separate disagreements, not 400 bins
    previous_end = 0.0
    for region in regions:
        assert region["start"] < region["end"]
        assert region["start"] >= previous_end  # ordered and non-overlapping
        assert region["end"] <= _run(BASELINE)["duration"] + 0.1
        previous_end = region["end"]


def test_unmatched_perturbed_labels_read_as_disagreement():
    """A perturbed speaker with no counterpart cannot be renamed into the
    original's label space, so every span it occupies must differ."""

    perturbed = [
        _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
        _segment("seg_001", 5.0, 9.0, "SPEAKER_01"),
        _segment("seg_002", 10.0, 14.0, "SPEAKER_00"),
        _segment("seg_003", 15.0, 19.0, "SPEAKER_01"),
        _segment("seg_004", 19.0, 19.9, "SPEAKER_09"),  # appeared, unmatched
    ]

    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    assert delta["appeared"] == ["SPEAKER_09"]
    assert len(delta["diff_regions"]) >= 1
    assert delta["diff_regions"][-1]["end"] == pytest.approx(19.9, abs=0.1)


def test_diff_regions_are_empty_when_a_zero_duration_run_is_compared():
    """Guard against a division by zero in the binning."""

    delta = diff.compare_runs(_run(BASELINE, duration=0.0), _run(BASELINE, duration=0.0))

    assert delta["diff_regions"] == []


# ---------------------------------------------------------------------------
# Overlapping speech -- two segments with identical extents must not collide
# ---------------------------------------------------------------------------


def test_simultaneous_segments_are_both_kept():
    """Track names are segment ids precisely so genuinely overlapping speech
    does not silently overwrite itself on one track."""

    overlapping = [
        _segment("seg_000", 0.0, 4.0, "SPEAKER_00"),
        _segment("seg_001", 0.0, 4.0, "SPEAKER_01"),  # identical extent
    ]

    delta = diff.compare_runs(_run(overlapping), _run(overlapping))

    assert delta["der"] == 0.0
    assert delta["lost_segments"] == []
    assert delta["num_speakers_delta"] == 0


# ---------------------------------------------------------------------------
# Payload shape -- what the frontend delta card actually reads
# ---------------------------------------------------------------------------


def test_delta_payload_has_the_expected_keys():
    delta = diff.compare_runs(_run(BASELINE), _run(BASELINE))

    assert set(delta) == {
        "der",
        "der_is_vs_original_run",
        "der_components",
        "speaker_mapping",
        "appeared",
        "disappeared",
        "merged",
        "split",
        "num_speakers_delta",
        "boundary_shifts",
        "lost_segments",
        "diff_regions",
    }


def test_delta_payload_is_json_serializable():
    """It is cached in Redis as JSON, so no numpy scalars may survive."""

    import json

    perturbed = [s for s in BASELINE if s["id"] != "seg_002"]
    delta = diff.compare_runs(_run(BASELINE), _run(perturbed))

    round_tripped = json.loads(json.dumps(delta))
    assert round_tripped["der"] == delta["der"]
    assert isinstance(delta["der"], float)
    assert isinstance(delta["boundary_shifts"]["count"], int)
