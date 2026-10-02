"""Custom datasets (Manage Datasets) and the report's statistics.

Covers the researcher-facing additions: named datasets of the visitor's own
audio, an optional label file that makes EER measurable on them, the silence-
trimmed evaluation condition, and the uncertainty figures the report shows.
The label rule from dataset.py carries over: no response may say which clip
carries which label.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from app.tasks.deepfake import custom_datasets, evaluation, metrics, silence_probe
from app.tasks.deepfake import router as deepfake_router


def _wav_bytes(seconds: float = 1.5, sample_rate: int = 16_000, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    buffer = io.BytesIO()
    sf.write(buffer, (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def storage(monkeypatch, tmp_path):
    root = tmp_path / "deepfake_datasets"
    monkeypatch.setattr(custom_datasets, "_root", lambda: root)
    return root


@pytest.fixture
def stub_scores(monkeypatch):
    """Genuine files score low, spoofed ones high, keyed on the display name."""

    def _score(path) -> float:
        name = json.loads(path.with_suffix(".json").read_text())["display_filename"]
        return {"real1": 0.1, "real2": 0.2, "real3": 0.6, "fake1": 0.9, "fake2": 0.4, "fake3": 0.8}[name.split(".")[0]]

    def _detect(model_key, audio_path):
        score = _score(audio_path)
        return {"spoof_probability": score, "decision": "spoof" if score >= 0.5 else "bonafide"}

    monkeypatch.setattr(evaluation, "run_detection", _detect)
    monkeypatch.setattr(evaluation, "score_trimmed", _detect)


async def _make(client, name="My set", files=("real1", "real2", "real3", "fake1", "fake2", "fake3")):
    assert (await client.post("/tasks/deepfake/datasets", data={"dataset_name": name})).status_code == 200
    response = await client.post(
        f"/tasks/deepfake/datasets/{name}/files",
        # Distinct audio per file: scores are cached on the content hash.
        files=[
            ("files", (f"{stem}.wav", _wav_bytes(amplitude=0.1 + 0.05 * index), "audio/wav"))
            for index, stem in enumerate(files)
        ],
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_create_upload_list_and_delete(client):
    uploaded = await _make(client)
    assert len(uploaded["uploaded_files"]) == 6 and uploaded["errors"] == []
    assert all(clip["recording_id"].startswith("cd_") for clip in uploaded["uploaded_files"])

    listing = (await client.get("/tasks/deepfake/datasets")).json()["datasets"]
    assert [(d["dataset_name"], d["total_files"]) for d in listing] == [("My set", 6)]

    clip_id = uploaded["uploaded_files"][0]["recording_id"]
    audio = await client.get(f"/tasks/deepfake/datasets/clips/{clip_id}/audio")
    assert audio.status_code == 200

    assert (await client.delete("/tasks/deepfake/datasets/My set")).status_code == 200
    assert (await client.get("/tasks/deepfake/datasets")).json()["datasets"] == []
    assert (await client.get(f"/tasks/deepfake/datasets/clips/{clip_id}/audio")).status_code == 404


@pytest.mark.parametrize("name", ["", "../escape", "a/b", "x" * 49, ".hidden"])
async def test_unsafe_dataset_names_are_refused(client, name):
    response = await client.post("/tasks/deepfake/datasets", data={"dataset_name": name})
    assert response.status_code in (400, 422)


async def test_duplicate_names_and_bad_files_are_reported(client):
    await _make(client, files=("real1",))
    assert (await client.post("/tasks/deepfake/datasets", data={"dataset_name": "my SET"})).status_code == 400

    response = await client.post(
        "/tasks/deepfake/datasets/My set/files",
        files=[("files", ("ok.wav", _wav_bytes(), "audio/wav")), ("files", ("bad.wav", b"not audio", "audio/wav"))],
    )
    body = response.json()
    assert len(body["uploaded_files"]) == 1
    assert [error["filename"] for error in body["errors"]] == ["bad.wav"]


async def test_label_files_report_counts_never_per_clip_labels(client):
    await _make(client)
    labels = "filename,label\nreal1.wav,bonafide\nreal2.flac,genuine\nreal3,bonafide\nfake1.wav,spoof\nfake2,fake\nfake3,spoof\n"
    response = await client.post(
        "/tasks/deepfake/datasets/My set/labels", files={"file": ("labels.csv", labels.encode(), "text/csv")}
    )
    assert response.status_code == 200, response.text
    assert response.json()["labels"] == {"provided": True, "matched_files": 6, "bonafide": 3, "spoof": 3}

    recordings = (await client.get("/tasks/deepfake/datasets/My set/recordings")).json()["recordings"]
    text = json.dumps(recordings).lower()
    assert "bonafide" not in text and "spoof" not in text and "label" not in text


def test_asvspoof_protocol_lines_are_understood():
    parsed = custom_datasets.parse_label_file("LA_0039 LA_E_1 - A07 spoof\nLA_0040 LA_E_2 - - bonafide\n")
    assert parsed == {"la_e_1": ["A07", "spoof"], "la_e_2": ["-", "bonafide"]}


async def test_evaluation_needs_labels_then_works_in_both_conditions(client, stub_scores):
    await _make(client)
    body = {"model": "xlsr-deepfake", "dataset": "My set"}
    assert (await client.post("/tasks/deepfake/scores", json=body)).status_code == 422

    labels = "real1,bonafide\nreal2,bonafide\nreal3,bonafide\nfake1,spoof\nfake2,spoof\nfake3,spoof\n"
    await client.post("/tasks/deepfake/datasets/My set/labels", files={"file": ("l.csv", labels.encode(), "text/csv")})

    for condition in ("as_distributed", "silence_trimmed"):
        response = await client.post("/tasks/deepfake/scores", json={**body, "condition": condition})
        assert response.status_code == 200, response.text
        report = response.json()
        assert report["dataset_id"] == "custom:My set"
        assert report["condition"] == condition
        assert (report["bonafide_count"], report["spoof_count"]) == (3, 3)
        # real3 (0.6) outranks fake2 (0.4): one error in each direction is unavoidable.
        assert report["eer_percent"] == pytest.approx(33.333, abs=0.01)
        # 8 of the 9 (spoof, genuine) pairs are ordered correctly; only fake2 < real3.
        assert report["roc_auc"] == pytest.approx(8 / 9, abs=1e-6)
        assert report["eer_ci_percent"][0] <= report["eer_percent"] <= report["eer_ci_percent"][1]

    bad = await client.post("/tasks/deepfake/scores", json={**body, "condition": "louder"})
    assert bad.status_code == 400


async def test_another_session_cannot_reach_a_dataset(client, monkeypatch):
    uploaded = await _make(client, files=("real1",))
    clip_id = uploaded["uploaded_files"][0]["recording_id"]
    client.cookies.clear()
    assert (await client.get("/tasks/deepfake/datasets/My set/recordings")).status_code == 404
    assert (await client.get(f"/tasks/deepfake/datasets/clips/{clip_id}/audio")).status_code == 404


# ── statistics ───────────────────────────────────────────────────────────────


def test_fast_eer_matches_the_reference_definition():
    rng = np.random.default_rng(0)
    bonafide = list(rng.normal(0.3, 0.15, 80))
    spoof = list(rng.normal(0.6, 0.15, 90))
    reference, _ = metrics.equal_error_rate(bonafide, spoof)
    assert metrics._eer_fast(bonafide, spoof) == pytest.approx(reference)


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    bonafide, spoof = list(rng.normal(0.3, 0.2, 100)), list(rng.normal(0.7, 0.2, 100))
    eer, _ = metrics.equal_error_rate(bonafide, spoof)
    low, high = metrics.bootstrap_eer_interval(bonafide, spoof)
    assert low <= eer <= high and high - low > 0
    assert metrics.bootstrap_eer_interval(bonafide, spoof) == (low, high)


def test_zero_error_bound_is_the_rule_of_three():
    # 1 - 0.05^(1/100) ≈ 2.95 %, close to 3/n.
    assert metrics.zero_error_upper_bound(100, 100) == pytest.approx(0.02951, abs=1e-4)


def test_auc_and_confusion():
    assert metrics.roc_auc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert metrics.roc_auc([0.5], [0.5]) == 0.5
    confusion = metrics.confusion_at(0.5, [0.1, 0.6], [0.4, 0.9])
    assert (confusion["true_positives"], confusion["false_negatives"]) == (1, 1)
    assert (confusion["false_positives"], confusion["true_negatives"]) == (1, 1)
    assert confusion["false_acceptance_ci"][0] < 0.5 < confusion["false_acceptance_ci"][1]


# ── silence probe: short variants are scored, not withheld ──────────────────


def test_short_variants_are_scored_but_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(silence_probe, "_score_samples", lambda *args: {"spoof_probability": 0.7, "decision": "spoof"})
    samples = np.zeros(4000, dtype=np.float32)  # 0.25 s

    short = silence_probe._score_variant("m", samples, 0.25, 0.5, 16_000, tmp_path, "non_speech", "non-speech")
    assert short["applicable"] is True and short["reliable"] is False
    assert short["spoof_probability"] == 0.7
    assert "0.250s" in short["reason"]

    empty = silence_probe._score_variant("m", samples[:0], 0.0, 0.5, 16_000, tmp_path, "non_speech", "non-speech")
    assert empty["applicable"] is False and "0.000s" in empty["reason"]
