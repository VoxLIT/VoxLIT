"""Security and access-control tests for /tasks/deepfake (Test Plan 3.1.6).

Two properties are worth attacking in this task specifically.

The first is the usual one: recording ids are opaque handles, and a caller
must not be able to bend one into a filesystem path. `resolve_recording_path`
never concatenates the id with a directory -- it hashes each real filename and
compares -- so a traversal id simply misses. These tests hold that design in
place by asserting it across EVERY route that accepts a recording id, not just
the one it was first written for.

The second is specific to this task and matters more: `protocol.txt` holds the
bona fide/spoof answers the workbench exists to let a user judge. If any
runtime response leaks a label, an attack id, or the protocol file itself, the
whole exercise is pointless. `dataset.py` states that as a rule; here it is
enforced as a test, by sweeping every endpoint's serialised body rather than
by checking the fields each response happens to declare.

No model is ever loaded: inference is stubbed throughout.
"""

import json
from importlib import import_module

import pytest

from app.core.settings import settings

deepfake_router = import_module("app.tasks.deepfake.router")
deepfake_evaluation = import_module("app.tasks.deepfake.evaluation")
deepfake_embeddings = import_module("app.tasks.deepfake.embeddings")

# Two genuine, two spoofed, each spoof from a different generator -- enough
# for the ground-truth sweep to have something to leak.
PROTOCOL = {
    "LA_E_6000001": ("-", "bonafide"),
    "LA_E_6000002": ("-", "bonafide"),
    "LA_E_6000003": ("A07", "spoof"),
    "LA_E_6000004": ("A19", "spoof"),
}

SCORES = {
    "LA_E_6000001": 0.04,
    "LA_E_6000002": 0.07,
    "LA_E_6000003": 0.93,
    "LA_E_6000004": 0.88,
}

# Anything in here appearing in ANY runtime body means the protocol leaked.
GROUND_TRUTH_TOKENS = [
    "system_id",
    "protocol",
    "ground_truth",
    "LA_0069",
]

# Attack ids are legitimate in ONE place only: the evaluation view's
# `per_attack`, where each entry is a mean over a group and so says nothing
# about any individual clip. Everywhere else they are a leak.
ATTACK_TOKENS = ["A07", "A19"]
AGGREGATE_ONLY_ROUTES = {"scores"}

TRAVERSAL_IDS = [
    "../protocol.txt",
    "../../protocol.txt",
    "..%2F..%2Fprotocol.txt",
    "....//protocol.txt",
    "/etc/passwd",
    "rec_" + "0" * 16,
    "rec_../../../../etc/passwd",
]

RECORDING_ROUTES = [
    ("/tasks/deepfake/run", "model"),
    ("/tasks/deepfake/silence-probe", "model"),
    ("/tasks/deepfake/saliency", "model"),
]


@pytest.fixture
def fake_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DEEPFAKE_DATASET_ROOT", tmp_path)
    audio_dir = tmp_path / "asvspoof2019_la" / "flac"
    audio_dir.mkdir(parents=True)
    for index, file_id in enumerate(PROTOCOL):
        (audio_dir / f"{file_id}.flac").write_bytes(b"fLaC" + bytes([index]))
    (tmp_path / "asvspoof2019_la" / "protocol.txt").write_text(
        "\n".join(
            f"LA_0069 {file_id} - {system_id} {key}"
            for file_id, (system_id, key) in PROTOCOL.items()
        )
        + "\n"
    )
    return audio_dir


def _detection_for(stem: str) -> dict:
    score = SCORES[stem]
    return {
        "model": "xlsr-deepfake",
        "model_label": "wav2vec2 XLS-R (Model A)",
        "model_id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        "decision": "spoof" if score >= 0.5 else "bonafide",
        "threshold": 0.5,
        "threshold_calibrated": False,
        "threshold_version": "deepfake-threshold-v0-uncalibrated",
        "spoof_probability": score,
        "bonafide_probability": round(1 - score, 6),
        "logits": [-1.0, 1.0],
        "id2label": {0: "bonafide", 1: "spoof"},
        "spoof_index": 1,
        "duration": 3.2,
        "analysed_seconds": 3.2,
        "analysis_window_seconds": 30.0,
        "truncated": False,
    }


@pytest.fixture
def stub_everything(monkeypatch):
    """Stub every inference entry point the routes reach."""
    from pathlib import Path

    def _run_detection(model_key, audio_path):
        return _detection_for(Path(audio_path).stem)

    def _run_embedding(model_key, audio_path):
        stem = Path(audio_path).stem
        score = SCORES[stem]
        return {"embedding": [score, 1 - score, 0.5, 0.25], "spoof_probability": score}

    def _silence_probe(model_key, audio_path):
        payload = _detection_for(Path(audio_path).stem)
        return {
            "model": payload["model"],
            "model_label": payload["model_label"],
            "threshold": 0.5,
            "threshold_calibrated": False,
            "silence_top_db": 30,
            "min_non_speech_seconds": 0.5,
            "duration": 3.2,
            "speech_seconds": 2.0,
            "non_speech_seconds": 1.2,
            "non_speech_fraction": 0.375,
            "speech_intervals": [[0.4, 2.4]],
            "variants": {
                "original": {
                    "applicable": True,
                    "seconds": 3.2,
                    "spoof_probability": payload["spoof_probability"],
                    "decision": payload["decision"],
                },
                "trimmed": {
                    "applicable": True,
                    "seconds": 2.0,
                    "spoof_probability": payload["spoof_probability"],
                    "decision": payload["decision"],
                },
                "non_speech": {
                    "applicable": False,
                    "seconds": 0.3,
                    "spoof_probability": None,
                    "decision": None,
                    "reason": "Only 0.30s of non-speech audio was found.",
                },
            },
            "sample_rate": 16000,
        }

    def _saliency(model_key, audio_path, segment_count=60):
        return {
            "model": model_key,
            "method": "input-gradient",
            "method_label": "Input gradient (|d spoof logit / d input|)",
            "target": "spoof logit",
            "segments": [
                {"start_time": 0.0, "end_time": 1.6, "saliency": 1.0, "intensity": 1.0},
                {"start_time": 1.6, "end_time": 3.2, "saliency": 0.4, "intensity": 0.4},
            ],
            "series": [1.0, 0.4],
            "total_duration": 3.2,
            "model_label": "wav2vec2 XLS-R (Model A)",
            "max_saliency_seconds": 12,
            "analysis_window_seconds": 30.0,
            "truncated": False,
            "normalised": True,
            "speech_intervals": [[0.4, 2.4]],
            "silence_top_db": 30,
            "saliency_in_speech_fraction": 0.72,
        }

    monkeypatch.setattr(deepfake_router, "run_detection", _run_detection)
    monkeypatch.setattr(deepfake_evaluation, "run_detection", _run_detection)
    monkeypatch.setattr(deepfake_embeddings, "run_embedding", _run_embedding)
    monkeypatch.setattr(deepfake_router, "run_silence_probe", _silence_probe)
    monkeypatch.setattr(deepfake_router, "generate_saliency", _saliency)


async def _first_recording_id(client) -> str:
    response = await client.get("/tasks/deepfake/dataset/recordings")
    return response.json()["recordings"][0]["recording_id"]


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize("recording_id", TRAVERSAL_IDS)
@pytest.mark.parametrize("route,_model_field", RECORDING_ROUTES)
async def test_traversal_ids_are_rejected_on_every_recording_route(
    client, fake_dataset, stub_everything, route, _model_field, recording_id
):
    response = await client.post(
        route, json={"model": "xlsr-deepfake", "recording_id": recording_id}
    )
    assert response.status_code == 404, f"{route} accepted {recording_id!r}"
    assert "Unknown recording id" in response.json()["detail"]


@pytest.mark.security
@pytest.mark.parametrize("recording_id", TRAVERSAL_IDS)
async def test_traversal_ids_are_rejected_on_the_audio_route(
    client, fake_dataset, recording_id
):
    response = await client.get(f"/tasks/deepfake/dataset/recordings/{recording_id}/audio")
    # A traversal id either 404s at the handler or never routes at all; what
    # must never happen is a 200 carrying somebody else's bytes.
    assert response.status_code in (404, 405)
    assert b"protocol" not in response.content.lower()


@pytest.mark.security
async def test_the_protocol_file_is_not_reachable_as_a_recording(client, fake_dataset):
    """The answers live one directory above the audio, and are never listed."""
    listing = await client.get("/tasks/deepfake/dataset/recordings")
    filenames = [r["display_filename"] for r in listing.json()["recordings"]]

    assert "protocol.txt" not in filenames
    assert all(name.endswith(".flac") for name in filenames)


@pytest.mark.security
async def test_a_protocol_shaped_id_cannot_be_resolved(client, fake_dataset):
    """Even the id the protocol file WOULD have, if it were a recording."""
    from app.tasks.deepfake.dataset import RecordingNotFound, _recording_id_for, resolve_recording_path

    with pytest.raises(RecordingNotFound):
        resolve_recording_path(_recording_id_for("protocol.txt"))


# ---------------------------------------------------------------------------
# Ground-truth confinement
# ---------------------------------------------------------------------------


@pytest.mark.security
async def test_no_runtime_response_leaks_the_protocol(client, fake_dataset, stub_everything):
    """Sweep every runtime route and search the whole body for the answers.

    Deliberately a text search over the serialised payload, not a field check:
    a future field added to any of these responses is covered automatically.
    """
    recording_id = await _first_recording_id(client)

    bodies = {}
    bodies["recordings"] = (await client.get("/tasks/deepfake/dataset/recordings")).json()
    bodies["recording"] = (
        await client.get(f"/tasks/deepfake/dataset/recordings/{recording_id}")
    ).json()
    bodies["models"] = (await client.get("/tasks/deepfake/models")).json()
    bodies["dataset"] = (await client.get("/tasks/deepfake/dataset")).json()
    bodies["run"] = (
        await client.post(
            "/tasks/deepfake/run",
            json={"model": "xlsr-deepfake", "recording_id": recording_id},
        )
    ).json()
    bodies["scores"] = (
        await client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})
    ).json()
    bodies["embeddings"] = (
        await client.post(
            "/tasks/deepfake/embeddings",
            json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2},
        )
    ).json()
    bodies["silence-probe"] = (
        await client.post(
            "/tasks/deepfake/silence-probe",
            json={"model": "xlsr-deepfake", "recording_id": recording_id},
        )
    ).json()
    bodies["saliency"] = (
        await client.post(
            "/tasks/deepfake/saliency",
            json={"model": "xlsr-deepfake", "recording_id": recording_id},
        )
    ).json()

    for name, body in bodies.items():
        serialised = json.dumps(body)
        for token in GROUND_TRUTH_TOKENS:
            assert token not in serialised, f"{name} leaked {token!r}"
        if name not in AGGREGATE_ONLY_ROUTES:
            for token in ATTACK_TOKENS:
                assert token not in serialised, f"{name} leaked attack {token!r}"


@pytest.mark.security
async def test_the_evaluation_view_reports_attacks_only_in_aggregate(
    client, fake_dataset, stub_everything
):
    """`per_attack` is the one place an attack id legitimately appears.

    It is a mean over a group, so it says nothing about any single clip. This
    test pins that distinction: attack names are present, per-recording rows
    are not.
    """
    payload = (
        await client.post("/tasks/deepfake/scores", json={"model": "xlsr-deepfake"})
    ).json()

    attacks = {row["attack"] for row in payload["per_attack"]}
    assert {"A07", "A19", "bonafide"} <= attacks

    serialised = json.dumps(payload)
    for file_id in PROTOCOL:
        assert file_id not in serialised
    assert "recording_id" not in serialised


@pytest.mark.security
async def test_the_embedding_view_colours_by_score_not_by_label(
    client, fake_dataset, stub_everything
):
    payload = (
        await client.post(
            "/tasks/deepfake/embeddings",
            json={"model": "xlsr-deepfake", "reduction_method": "pca", "n_components": 2},
        )
    ).json()

    for point in payload["recordings"]:
        assert set(point) == {
            "recording_id",
            "display_filename",
            "spoof_probability",
            "decision",
        }
        # `decision` is the MODEL's opinion at its threshold, which is exactly
        # what it is allowed to be -- it must track the score, not the truth.
        expected = "spoof" if point["spoof_probability"] >= payload["threshold"] else "bonafide"
        assert point["decision"] == expected


# ---------------------------------------------------------------------------
# Input handling and secret confinement
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize(
    "model_key",
    ["", "../../etc/passwd", "wav2vec2-xlsr-deepfake", "xlsr-deepfake; rm -rf /", "None"],
)
async def test_unknown_model_keys_are_refused_with_400_not_500(
    client, fake_dataset, stub_everything, model_key
):
    recording_id = await _first_recording_id(client)
    response = await client.post(
        "/tasks/deepfake/run", json={"model": model_key, "recording_id": recording_id}
    )
    assert response.status_code == 400
    assert "Unsupported deepfake model" in response.json()["detail"]


@pytest.mark.security
async def test_malformed_bodies_are_422_not_500(client, fake_dataset, stub_everything):
    for body in ({}, {"model": "xlsr-deepfake"}, {"recording_id": "x"}, {"model": 7}):
        response = await client.post("/tasks/deepfake/run", json=body)
        assert response.status_code == 422, body


@pytest.mark.security
async def test_a_load_failure_never_echoes_the_token(client, fake_dataset, monkeypatch):
    """A 503 must be actionable without printing the secret that failed.

    The gated-repo message names the repo and the env var -- it must not name
    the value.
    """
    from app.tasks.deepfake.service import DeepfakeModelUnavailable, get_model_spec

    monkeypatch.setattr(settings, "HF_TOKEN", "hf_secretvalue_do_not_leak")

    def _explode(model_key, audio_path):
        spec = get_model_spec(model_key)
        from app.tasks.deepfake.service import _load_failure_message

        raise DeepfakeModelUnavailable(
            _load_failure_message(spec, RuntimeError("401 Client Error: gated repo"))
        )

    monkeypatch.setattr(deepfake_router, "run_detection", _explode)
    recording_id = await _first_recording_id(client)

    response = await client.post(
        "/tasks/deepfake/run",
        json={"model": "ast-fakeaudio", "recording_id": recording_id},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "hf_secretvalue_do_not_leak" not in detail
    # Still actionable: it says what to do about it.
    assert "HF_TOKEN" in detail and "huggingface.co" in detail


@pytest.mark.security
async def test_ground_truth_is_never_imported_by_a_route_module():
    """`load_ground_truth` is reachable from evaluation only, by design."""
    import inspect

    source = inspect.getsource(deepfake_router)
    assert "load_ground_truth" not in source

    from app.tasks.deepfake import embeddings, saliency, service, silence_probe

    for module in (embeddings, saliency, service, silence_probe):
        assert "load_ground_truth" not in inspect.getsource(module), module.__name__
