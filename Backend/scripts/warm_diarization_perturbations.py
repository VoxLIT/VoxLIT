"""Pre-compute diarization perturbation results so a live demo is a cache hit.

A cold perturbation is a full re-run of the pipeline: seconds on a short clip,
5-10 minutes on a real AMI meeting. Because every layer of this feature is
content-addressed and every transform is seeded, warming the cache *is* the
canned result -- there is no separate fixture format to keep in sync. Whatever
this script stores is byte-for-byte what a real request would have produced.

Usage (from Backend/, with the venv active):

    python -m scripts.warm_diarization_perturbations \
        --recording audio.wav \
        --preset noise:0.01 --preset mask:20-40

    # every demo recording, default presets
    python -m scripts.warm_diarization_perturbations --all

`--recording` takes the display filename of a demo recording (see
`GET /tasks/task-b/dataset/recordings`). Presets are `noise:<level>` or
`mask:<start>-<end>` (percentages).

The script drives the app in-process through the router's own coroutines
rather than reimplementing the flow, so a warmed entry cannot drift from what
the endpoint computes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_PRESETS = ["noise:0.01", "mask:20-40"]
DEFAULT_MODEL = "pyannote-3.1"


def _parse_preset(raw: str) -> tuple[str, dict]:
    """`noise:0.01` -> ("noise", {...});  `mask:20-40` -> ("time_masking", {...})."""

    kind, _, value = raw.partition(":")
    if kind == "noise":
        return "noise", {"noise_level": float(value)}
    if kind == "mask":
        start, _, end = value.partition("-")
        return "time_masking", {
            "mask_start_percent": float(start),
            "mask_end_percent": float(end),
        }
    raise SystemExit(f"Unrecognized preset '{raw}'. Use noise:<level> or mask:<start>-<end>.")


async def _warm(recordings: list[str] | None, presets: list[str], model: str) -> int:
    from app.tasks.task_b import dataset
    from app.tasks.task_b.router import PerturbationRequest, PerturbationSpec, run_perturbation

    class _FakeRequest:
        """The endpoint only ever reads `request.state.sid`; a warming run is
        its own session so it never touches a real user's storage."""

        def __init__(self, sid: str) -> None:
            self.state = type("S", (), {"sid": sid})()

    sid = uuid.uuid4().hex
    request = _FakeRequest(sid)

    try:
        available = dataset.list_recordings()
    except dataset.DatasetUnavailable as error:
        raise SystemExit(f"{error}\nThe AMI demo data is not present; nothing to warm.")

    by_name = {item.display_filename: item.recording_id for item in available}
    targets = recordings if recordings else list(by_name)

    unknown = [name for name in targets if name not in by_name]
    if unknown:
        raise SystemExit(
            f"Unknown recording(s): {', '.join(unknown)}\nAvailable: {', '.join(by_name)}"
        )

    warmed = 0
    for name in targets:
        for preset in presets:
            perturbation_type, params = _parse_preset(preset)
            started = time.monotonic()
            payload = PerturbationRequest(
                model=model,
                recording_id=by_name[name],
                perturbation=PerturbationSpec(type=perturbation_type, params=params),
            )
            result = await run_perturbation(request, payload)
            elapsed = time.monotonic() - started
            delta = result["delta"]
            print(
                f"  {name:28s} {preset:14s} "
                f"DER {delta['der']:.4f}  "
                f"speakers {result['original']['num_speakers']}->{result['perturbed']['num_speakers']}  "
                f"shifts {delta['boundary_shifts']['count']}  "
                f"({'cached' if result['cached'] else f'{elapsed:.1f}s'})"
            )
            warmed += 1

    # Close the pool while the loop is still running. Otherwise redis-py's
    # connection __del__ fires after asyncio.run() has closed the loop and
    # prints a "Event loop is closed" traceback over the script's output.
    from app.core import redis as redis_module

    await redis_module.redis.aclose()
    return warmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording",
        action="append",
        dest="recordings",
        help="Display filename of a demo recording. Repeatable. Omit with --all for every one.",
    )
    parser.add_argument("--all", action="store_true", help="Warm every demo recording.")
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help=f"noise:<level> or mask:<start>-<end>. Repeatable. Default: {' '.join(DEFAULT_PRESETS)}",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.recordings and not args.all:
        parser.error("pass --recording <name> (repeatable) or --all")

    presets = args.presets or DEFAULT_PRESETS
    print(f"Warming {args.model}: presets {', '.join(presets)}")
    warmed = asyncio.run(_warm(args.recordings, presets, args.model))
    print(f"Done - {warmed} perturbation result(s) in cache.")


if __name__ == "__main__":
    main()
