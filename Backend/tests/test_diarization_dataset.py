"""Focused tests for AMI demo-dataset discovery and listing (task_b).

Every test builds a fake `ami_subset`-shaped directory under `tmp_path` and
points `settings.SPEAKER_DIARIZATION_DATASET_ROOT` at it via monkeypatch —
never the real, gitignored AMI data — mirroring the pattern used in
`test_speaker_verification_dataset.py`.

The `rttm/` subfolder is present in every fake dataset on purpose: AMI ground
truth is offline-evaluation-only, so "the runtime never sees it" is a property
worth testing rather than assuming.
"""

import json
from dataclasses import asdict

import pytest

from app.core.settings import settings
from app.tasks.task_b import dataset

REAL_DATASET_DIR = settings.speaker_diarization_ami_dataset_dir

# Deliberately not in alphabetical order: discovery must sort, not echo
# whatever order the filesystem happens to yield.
FAKE_RECORDINGS = ["TS3003a.Mix-Headset.wav", "ES2004a.Mix-Headset.wav", "IS1009a.Mix-Headset.wav"]
RTTM_FILENAME = "ES2004a.rttm"
RTTM_CONTENT = "SPEAKER ES2004a 1 0.00 3.50 <NA> <NA> MEE013 <NA> <NA>\n"


def _build_fake_dataset(root):
    dataset_dir = root / "ami_subset"
    dataset_dir.mkdir(parents=True)
    for filename in FAKE_RECORDINGS:
        (dataset_dir / filename).write_bytes(b"RIFF-fake-audio")
    (dataset_dir / "notes.txt").write_text("not audio")

    rttm_dir = dataset_dir / "rttm"
    rttm_dir.mkdir()
    (rttm_dir / RTTM_FILENAME).write_text(RTTM_CONTENT)
    return dataset_dir


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)
    # A missing monkeypatch would silently read the real AMI data; fail loudly.
    assert settings.speaker_diarization_ami_dataset_dir != REAL_DATASET_DIR
    return _build_fake_dataset(tmp_path)


@pytest.fixture
def missing_dataset_dir(monkeypatch, tmp_path):
    """Root points somewhere real but `ami_subset/` was never created."""

    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)
    assert settings.speaker_diarization_ami_dataset_dir != REAL_DATASET_DIR
    return tmp_path


def _assert_no_ground_truth_leak(payload) -> None:
    """Nothing in a runtime payload may reference the offline-eval RTTMs."""

    serialized = json.dumps(payload)
    assert "rttm" not in serialized.lower()
    assert RTTM_FILENAME not in serialized
    assert "MEE013" not in serialized
    assert "notes.txt" not in serialized


BAD_IDS = [
    "not-a-real-id",
    "../../etc/passwd",
    "../rttm/ES2004a.rttm",
    "ES2004a.Mix-Headset.wav",
    "",
    "rec_0000000000000000",
]


# ---------------------------------------------------------------------------
# Dataset identity and summary
# ---------------------------------------------------------------------------


def test_dataset_id_and_expected_count_are_exact():
    assert dataset.DATASET_ID == "ami-subset"
    assert dataset.EXPECTED_RECORDING_COUNT == 3


def test_get_dataset_info_reports_counts_and_extensions(fake_dataset_dir):
    info = dataset.get_dataset_info()

    assert info["dataset_id"] == "ami-subset"
    assert info["expected_recording_count"] == 3
    assert info["total_recordings"] == len(FAKE_RECORDINGS)
    assert info["available"] is True
    assert info["audio_extensions"] == [".wav"]
    _assert_no_ground_truth_leak(info)


def test_get_dataset_info_reports_unavailable_without_raising(missing_dataset_dir):
    info = dataset.get_dataset_info()

    assert info["available"] is False
    assert info["total_recordings"] == 0
    assert info["dataset_id"] == "ami-subset"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_list_recordings_finds_every_wav(fake_dataset_dir):
    recordings = dataset.list_recordings()

    assert len(recordings) == len(FAKE_RECORDINGS)
    assert {r.display_filename for r in recordings} == set(FAKE_RECORDINGS)
    assert all(r.extension == ".wav" for r in recordings)
    assert all(r.size_bytes == len(b"RIFF-fake-audio") for r in recordings)


def test_list_recordings_excludes_non_audio_and_the_rttm_folder(fake_dataset_dir):
    recordings = dataset.list_recordings()

    names = {r.display_filename for r in recordings}
    assert "notes.txt" not in names
    assert "rttm" not in names
    assert RTTM_FILENAME not in names
    _assert_no_ground_truth_leak([asdict(r) for r in recordings])


def test_rttm_files_are_never_discovered_even_at_the_top_level(monkeypatch, tmp_path):
    """Defence in depth: an RTTM sitting beside the audio is still not audio."""

    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_DATASET_ROOT", tmp_path)
    dataset_dir = _build_fake_dataset(tmp_path)
    (dataset_dir / "ES2004a.rttm").write_text(RTTM_CONTENT)

    recordings = dataset.list_recordings()
    assert len(recordings) == len(FAKE_RECORDINGS)
    _assert_no_ground_truth_leak([asdict(r) for r in recordings])


def test_list_recordings_is_sorted_by_display_filename(fake_dataset_dir):
    recordings = dataset.list_recordings()

    names = [r.display_filename for r in recordings]
    assert names == sorted(FAKE_RECORDINGS)
    # Sanity: the fixture's on-disk creation order is not already sorted, so
    # this assertion is actually testing the sort.
    assert names != FAKE_RECORDINGS


def test_list_recordings_raises_when_dataset_missing(missing_dataset_dir):
    with pytest.raises(dataset.DatasetUnavailable):
        dataset.list_recordings()


# ---------------------------------------------------------------------------
# Opaque ids
# ---------------------------------------------------------------------------


def test_recording_ids_are_opaque_and_reveal_no_filename(fake_dataset_dir):
    for recording in dataset.list_recordings():
        assert recording.recording_id.startswith("rec_")
        digest = recording.recording_id.removeprefix("rec_")
        assert len(digest) == 16
        assert all(character in "0123456789abcdef" for character in digest)
        stem = recording.display_filename.split(".")[0]
        assert stem not in recording.recording_id


def test_recording_ids_are_stable_across_calls(fake_dataset_dir):
    first = {r.recording_id: r.display_filename for r in dataset.list_recordings()}
    second = {r.recording_id: r.display_filename for r in dataset.list_recordings()}
    assert first == second


def test_recording_ids_are_distinct(fake_dataset_dir):
    ids = [r.recording_id for r in dataset.list_recordings()]
    assert len(ids) == len(set(ids))


def test_recording_id_is_derived_from_the_filename(fake_dataset_dir):
    recordings = {r.display_filename: r.recording_id for r in dataset.list_recordings()}
    for filename, recording_id in recordings.items():
        assert recording_id == dataset._recording_id_for(filename)


# ---------------------------------------------------------------------------
# Lookup by id
# ---------------------------------------------------------------------------


def test_get_recording_returns_the_matching_entry(fake_dataset_dir):
    known = dataset.list_recordings()[0]
    assert dataset.get_recording(known.recording_id) == known


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_get_recording_rejects_unknown_and_traversal_ids(fake_dataset_dir, bad_id):
    with pytest.raises(dataset.RecordingNotFound):
        dataset.get_recording(bad_id)


def test_get_recording_raises_unavailable_when_dataset_missing(missing_dataset_dir):
    with pytest.raises(dataset.DatasetUnavailable):
        dataset.get_recording("rec_anything")


# ---------------------------------------------------------------------------
# Path resolution -- the only place an untrusted id meets the filesystem
# ---------------------------------------------------------------------------


def test_resolve_recording_path_returns_the_real_file(fake_dataset_dir):
    known = dataset.list_recordings()[0]
    path = dataset.resolve_recording_path(known.recording_id)

    assert path == fake_dataset_dir / known.display_filename
    assert path.read_bytes() == b"RIFF-fake-audio"


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_resolve_recording_path_rejects_unknown_and_traversal_ids(fake_dataset_dir, bad_id):
    with pytest.raises(dataset.RecordingNotFound):
        dataset.resolve_recording_path(bad_id)


def test_resolve_recording_path_can_never_escape_the_dataset_dir(fake_dataset_dir, tmp_path):
    """The id is compared against ids derived from a directory listing, never
    joined into a path, so no id can name a file outside the dataset dir."""

    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF-not-yours")

    for candidate in ("../secret.wav", "secret.wav", str(outside), dataset._recording_id_for("secret.wav")):
        with pytest.raises(dataset.RecordingNotFound):
            dataset.resolve_recording_path(candidate)


def test_resolve_recording_path_never_reaches_the_rttm_folder(fake_dataset_dir):
    for candidate in ("rttm", f"rttm/{RTTM_FILENAME}", dataset._recording_id_for(RTTM_FILENAME)):
        with pytest.raises(dataset.RecordingNotFound):
            dataset.resolve_recording_path(candidate)


def test_resolve_recording_path_raises_unavailable_when_dataset_missing(missing_dataset_dir):
    with pytest.raises(dataset.DatasetUnavailable):
        dataset.resolve_recording_path("rec_anything")


# ---------------------------------------------------------------------------
# Real dataset (skipped unless the gitignored AMI subset is present)
# ---------------------------------------------------------------------------


AMI_MEETINGS = ["ES2004a", "IS1009a", "TS3003a"]


@pytest.mark.skipif(
    not REAL_DATASET_DIR.is_dir(),
    reason="Real gitignored AMI subset is not present in this environment",
)
def test_real_dataset_contains_the_three_ami_meetings():
    """Subset check, not `len(...) == EXPECTED_RECORDING_COUNT`.

    The real directory currently also holds a stray `audio.wav` that is not
    one of the three AMI meetings, so an exact-count assertion would fail on
    data drift rather than on anything this module does. What actually matters
    at runtime is that all three meetings are discoverable, and that whatever
    else is in there is still surfaced as a well-formed recording.
    """

    recordings = dataset.list_recordings()
    discovered = {r.display_filename for r in recordings}

    for meeting in AMI_MEETINGS:
        assert any(name.startswith(meeting) for name in discovered), meeting
    assert len(recordings) >= dataset.EXPECTED_RECORDING_COUNT
    assert all(r.recording_id.startswith("rec_") and r.size_bytes > 0 for r in recordings)
