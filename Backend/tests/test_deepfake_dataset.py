"""Discovery and listing tests for the ASVspoof 2019 LA demo subset.

Every test builds a fake `asvspoof2019_la`-shaped directory under `tmp_path`
and points `settings.DEEPFAKE_DATASET_ROOT` at it via monkeypatch — never the
real, gitignored dataset — mirroring `test_speaker_verification_dataset.py`.

The tests that matter most here are the ground-truth ones: the bona fide/spoof
key and the attack id are exactly what this task asks a user to judge, so they
must never appear in a listing response.
"""

import json

import pytest

from app.core.settings import settings
from app.tasks.deepfake import dataset

# file_id -> (system_id, key)
FAKE_PROTOCOL = {
    "LA_E_1000001": ("-", "bonafide"),
    "LA_E_1000002": ("A07", "spoof"),
    "LA_E_1000003": ("A13", "spoof"),
    "LA_E_1000004": ("-", "bonafide"),
}


def _build_fake_dataset(root):
    dataset_dir = root / "asvspoof2019_la"
    audio_dir = dataset_dir / "flac"
    audio_dir.mkdir(parents=True)
    for file_id in FAKE_PROTOCOL:
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC-fake-audio")
    # A non-audio file in the audio dir must be ignored.
    (audio_dir / "notes.txt").write_text("not audio")
    (dataset_dir / "protocol.txt").write_text(
        "\n".join(
            f"LA_0069 {file_id} - {system_id} {key}"
            for file_id, (system_id, key) in FAKE_PROTOCOL.items()
        )
        + "\n"
    )
    return dataset_dir


@pytest.fixture
def fake_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    return _build_fake_dataset(tmp_path)


@pytest.fixture
def missing_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path / "nowhere")


def _assert_no_ground_truth_leak(payload) -> None:
    """The label and the attack id are the answer — neither may be serialized.

    The dataset id itself contains "spoof" (asvspoof2019-la), so strip that
    known-safe token first rather than weakening the check.
    """
    serialized = json.dumps(payload).replace(dataset.DATASET_ID, "")
    assert "bonafide" not in serialized
    assert "spoof" not in serialized
    assert "protocol" not in serialized
    for system_id, _key in FAKE_PROTOCOL.values():
        if system_id != "-":
            assert system_id not in serialized


def test_dataset_id_is_exact():
    # Must match the id in both task registries.
    assert dataset.DATASET_ID == "asvspoof2019-la"


def test_get_dataset_info_reports_counts_and_extensions(fake_dataset_dir):
    info = dataset.get_dataset_info()

    assert info["dataset_id"] == "asvspoof2019-la"
    assert info["total_recordings"] == len(FAKE_PROTOCOL)
    assert info["available"] is True
    assert info["audio_extensions"] == [".flac"]
    _assert_no_ground_truth_leak(info)


def test_get_dataset_info_reports_unavailable_instead_of_raising(missing_dataset_dir):
    info = dataset.get_dataset_info()

    assert info["available"] is False
    assert info["total_recordings"] == 0


def test_list_recordings_carries_no_label(fake_dataset_dir):
    from dataclasses import asdict

    recordings = [asdict(r) for r in dataset.list_recordings()]

    assert len(recordings) == len(FAKE_PROTOCOL)
    # Exact field set, so a label field cannot be added without this failing.
    assert set(recordings[0]) == {
        "recording_id",
        "display_filename",
        "extension",
        "size_bytes",
        "duration_seconds",
    }
    _assert_no_ground_truth_leak(recordings)


def test_recording_ids_are_opaque_and_stable(fake_dataset_dir):
    first = {r.display_filename: r.recording_id for r in dataset.list_recordings()}
    second = {r.display_filename: r.recording_id for r in dataset.list_recordings()}

    assert first == second
    for filename, recording_id in first.items():
        assert recording_id.startswith("rec_")
        assert len(recording_id) == len("rec_") + 16
        assert filename.split(".")[0] not in recording_id


def test_list_recordings_is_sorted_by_filename_not_by_label(fake_dataset_dir):
    filenames = [r.display_filename for r in dataset.list_recordings()]

    assert filenames == sorted(filenames)


def test_list_recordings_raises_when_dataset_absent(missing_dataset_dir):
    with pytest.raises(dataset.DatasetUnavailable):
        dataset.list_recordings()


def test_get_recording_round_trips(fake_dataset_dir):
    expected = dataset.list_recordings()[0]

    assert dataset.get_recording(expected.recording_id) == expected


def test_get_recording_rejects_unknown_id(fake_dataset_dir):
    with pytest.raises(dataset.RecordingNotFound):
        dataset.get_recording("rec_deadbeefdeadbeef")


@pytest.mark.parametrize(
    "hostile_id",
    ["../protocol.txt", "../../etc/passwd", "LA_E_1000001.flac", "", "rec_"],
)
def test_resolve_recording_path_rejects_traversal(fake_dataset_dir, hostile_id):
    with pytest.raises(dataset.RecordingNotFound):
        dataset.resolve_recording_path(hostile_id)


def test_resolve_recording_path_returns_a_file_inside_the_dataset(fake_dataset_dir):
    recording = dataset.list_recordings()[0]

    path = dataset.resolve_recording_path(recording.recording_id)

    assert path.is_file()
    assert path.parent == fake_dataset_dir / "flac"


def test_load_ground_truth_parses_the_protocol(fake_dataset_dir):
    truth = dataset.load_ground_truth()

    assert truth == FAKE_PROTOCOL


def test_load_ground_truth_raises_when_protocol_absent(fake_dataset_dir):
    (fake_dataset_dir / "protocol.txt").unlink()

    with pytest.raises(dataset.DatasetUnavailable):
        dataset.load_ground_truth()


# ── The other built-in subsets (ASVspoof 5, In-the-Wild) ─────────────────────

ITW_PROTOCOL = {"0": ("unattributed", "spoof"), "1": ("-", "bonafide")}


def _build_fake_itw(root):
    dataset_dir = root / "in_the_wild"
    (dataset_dir / "wav").mkdir(parents=True)
    for file_id in ITW_PROTOCOL:
        (dataset_dir / "wav" / f"{file_id}.wav").write_bytes(b"RIFF-fake-audio")
    (dataset_dir / "protocol.txt").write_text(
        "\n".join(f"Barack_Obama {f} - {s} {k}" for f, (s, k) in ITW_PROTOCOL.items()) + "\n"
    )


def test_every_builtin_dataset_is_listed_installed_or_not(fake_dataset_dir, tmp_path):
    _build_fake_itw(tmp_path)
    listed = {entry["dataset_id"]: entry for entry in dataset.list_builtin_datasets()}

    assert set(listed) == {"asvspoof2019-la", "asvspoof5", "in-the-wild"}
    assert listed["asvspoof2019-la"]["total_recordings"] == 4
    assert listed["in-the-wild"]["total_recordings"] == 2
    assert listed["in-the-wild"]["audio_extensions"] == [".wav"]
    assert listed["asvspoof5"]["available"] is False


def test_other_datasets_get_namespaced_ids_and_keep_the_original_ones(fake_dataset_dir, tmp_path):
    _build_fake_itw(tmp_path)
    original = dataset.list_recordings()
    itw = dataset.list_recordings("in-the-wild")

    # ASVspoof 2019 LA ids are unchanged, so cached scores stay valid.
    assert original[0].recording_id == dataset._recording_id_for(original[0].display_filename)
    assert {r.display_filename for r in itw} == {"0.wav", "1.wav"}
    assert not {r.recording_id for r in itw} & {r.recording_id for r in original}
    # Any built-in id resolves without naming its dataset.
    assert dataset.resolve_recording_path(itw[0].recording_id).parent.name == "wav"
    assert dataset.audio_media_type(dataset.resolve_recording_path(itw[0].recording_id)) == "audio/wav"


def test_ground_truth_is_read_per_dataset(fake_dataset_dir, tmp_path):
    _build_fake_itw(tmp_path)
    assert dataset.load_ground_truth("in-the-wild") == ITW_PROTOCOL
    assert dataset.load_ground_truth() == FAKE_PROTOCOL


def test_unknown_dataset_ids_are_refused(fake_dataset_dir):
    with pytest.raises(dataset.UnknownDataset):
        dataset.list_recordings("../asvspoof2019_la")
