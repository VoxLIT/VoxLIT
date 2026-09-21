"""Score every clip of the ASVspoof 2019 LA demo subset with one detector.

Runs through the PRODUCTION adapters (`app.tasks.deepfake.service`), not a
reimplementation: the offline numbers in the test plan have to be the same
numbers `POST /tasks/deepfake/scores` returns, and the only way to guarantee
that is to call the same code. Preprocessing, the analysis window, the
label-index resolution and the threshold all come from the app.

Ground truth is read here (`dataset.load_ground_truth`) because this is
offline evaluation, which is exactly what that function is reserved for. The
labels it writes stay in this folder and never travel toward a runtime
response -- see the module docstring of Backend/app/tasks/deepfake/dataset.py.

Resumable: a row already present in the output CSV is skipped, so an
interrupted Model C run (~9 s/clip) picks up where it stopped.

Usage, from this folder:
    PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/score_dataset.py --model xlsr-deepfake
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUTS = ROOT / "outputs"

FIELDS = [
    "file_id",
    "recording_id",
    "attack",
    "label",
    "spoof_probability",
    "bonafide_probability",
    "decision",
    "duration",
    "analysed_seconds",
    "analysis_window_seconds",
    "truncated",
    "seconds_elapsed",
]


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["file_id"]: row for row in csv.DictReader(handle)}


def _score_once(adapter, spec, path, with_embedding: bool) -> dict:
    """One forward pass giving the score AND the head-input vector.

    `service.run_detection` would do a second pass for the embedding, which
    doubles a Model C run from ~30 to ~60 minutes. This calls the very same
    `adapter.score()` run_detection calls and applies the very same threshold
    comparison; `--verify` below asserts the two agree exactly on a sample, so
    the shortcut cannot drift from production silently.
    """
    result = adapter.score(path, with_embedding=with_embedding)
    result["decision"] = (
        "spoof" if result["spoof_probability"] >= spec.threshold else "bonafide"
    )
    return result


def _verify_against_production(model_key: str, spec, adapter, paths) -> list[dict]:
    """Confirm the single-pass shortcut equals `service.run_detection`."""
    from app.tasks.deepfake.service import run_detection

    checks = []
    for path in paths:
        production = run_detection(model_key, path)
        shortcut = _score_once(adapter, spec, path, with_embedding=False)
        checks.append(
            {
                "file": Path(path).name,
                "production_score": production["spoof_probability"],
                "shortcut_score": shortcut["spoof_probability"],
                "production_decision": production["decision"],
                "shortcut_decision": shortcut["decision"],
                "agrees": (
                    production["spoof_probability"] == shortcut["spoof_probability"]
                    and production["decision"] == shortcut["decision"]
                ),
            }
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="deepfake model key, e.g. xlsr-deepfake")
    parser.add_argument("--limit", type=int, default=None, help="score at most N clips (smoke runs)")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="skip the embedding capture (one forward pass either way, but the "
        "vectors are large on disk)",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=3,
        help="how many clips to cross-check against service.run_detection (0 disables)",
    )
    args = parser.parse_args()

    from app.tasks.deepfake.dataset import (
        DATASET_ID,
        list_recordings,
        load_ground_truth,
        resolve_recording_path,
    )
    from app.tasks.deepfake.service import get_model, get_model_spec

    spec = get_model_spec(args.model)
    adapter = get_model(args.model)
    truth = load_ground_truth()
    recordings = list_recordings()
    if args.limit:
        recordings = recordings[: args.limit]

    scores_path = OUTPUTS / "scores" / f"{args.model}_scores.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(scores_path)

    embeddings: dict[str, list[float]] = {}
    embeddings_path = OUTPUTS / "embeddings" / f"{args.model}_embeddings.npz"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[{args.model}] {spec.model_id}\n"
        f"  dataset={DATASET_ID} recordings={len(recordings)} "
        f"already scored={len(done)}",
        flush=True,
    )

    if args.verify:
        sample = [resolve_recording_path(r.recording_id) for r in recordings[: args.verify]]
        checks = _verify_against_production(args.model, spec, adapter, sample)
        (OUTPUTS / "scores" / f"{args.model}_production_agreement.json").write_text(
            json.dumps(checks, indent=2), encoding="utf-8"
        )
        disagreed = [check for check in checks if not check["agrees"]]
        print(
            f"  production agreement: {len(checks) - len(disagreed)}/{len(checks)} exact",
            flush=True,
        )
        if disagreed:
            raise SystemExit(f"Offline scoring drifted from run_detection: {disagreed}")

    # Key the header on the file being empty, not on `done`: a run interrupted
    # right after the header leaves a header-only file, and `done` is empty.
    needs_header = not scores_path.exists() or scores_path.stat().st_size == 0
    handle = open(scores_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if needs_header:
        writer.writeheader()

    started = time.perf_counter()
    scored = 0
    try:
        for index, recording in enumerate(recordings, start=1):
            file_id = recording.display_filename.rsplit(".", 1)[0]
            if file_id in done:
                continue
            entry = truth.get(file_id)
            if entry is None:
                # No protocol line: nothing to evaluate this clip against.
                print(f"  ! {file_id} has no protocol line, skipped", flush=True)
                continue
            attack, label = entry

            path = resolve_recording_path(recording.recording_id)
            clip_started = time.perf_counter()
            payload = _score_once(adapter, spec, path, with_embedding=not args.no_embeddings)
            if not args.no_embeddings:
                embeddings[file_id] = payload.pop("embedding")
            elapsed = time.perf_counter() - clip_started

            writer.writerow(
                {
                    "file_id": file_id,
                    "recording_id": recording.recording_id,
                    "attack": attack,
                    "label": label,
                    "spoof_probability": payload["spoof_probability"],
                    "bonafide_probability": payload["bonafide_probability"],
                    "decision": payload["decision"],
                    "duration": payload["duration"],
                    "analysed_seconds": payload["analysed_seconds"],
                    "analysis_window_seconds": payload["analysis_window_seconds"],
                    "truncated": payload["truncated"],
                    "seconds_elapsed": round(elapsed, 3),
                }
            )
            handle.flush()
            scored += 1
            if index % 10 == 0 or index == len(recordings):
                rate = (time.perf_counter() - started) / max(scored, 1)
                print(
                    f"  {index}/{len(recordings)}  {file_id} "
                    f"score={payload['spoof_probability']:.4f} "
                    f"({rate:.2f}s/clip)",
                    flush=True,
                )
    finally:
        handle.close()
        if embeddings:
            existing = {}
            if embeddings_path.is_file():
                with np.load(embeddings_path) as archive:
                    existing = {name: archive[name] for name in archive.files}
            existing.update(
                {name: np.asarray(vector, dtype=np.float32) for name, vector in embeddings.items()}
            )
            np.savez_compressed(embeddings_path, **existing)

    total = time.perf_counter() - started
    summary = {
        "model": args.model,
        "model_id": spec.model_id,
        "dataset_id": DATASET_ID,
        "clips_scored_this_run": scored,
        "clips_in_csv": len(_load_existing(scores_path)),
        "seconds_total": round(total, 2),
        "seconds_per_clip": round(total / scored, 3) if scored else None,
        "embeddings_captured": len(embeddings),
    }
    print(json.dumps(summary, indent=2), flush=True)
    (OUTPUTS / "scores" / f"{args.model}_scoring_run.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
