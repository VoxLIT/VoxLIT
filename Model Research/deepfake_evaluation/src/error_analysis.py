"""Where one detector fails on the ASVspoof 2019 LA demo subset, and why.

The metrics in evaluate_model.py say HOW OFTEN a detector is wrong. This says
WHICH clips, WHICH attacks, how CONFIDENT it was while being wrong, and --
most importantly for an interpretability tool -- whether the score it got
right was even evidence about the voice.

That last question is the one the task's own Feature 2 exists to ask, so the
ablation here calls the shipped `run_silence_probe`: score the clip as
submitted, with its leading/trailing silence trimmed, and with nothing but the
non-speech left. If trimming collapses the score, or silence alone still reads
"spoof", the headline EER is measuring the corpus, not the detector.

Usage, from this folder:
    PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/error_analysis.py --model xlsr-deepfake
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_model import load_scores  # noqa: E402

CONFIDENT_HIGH = 0.9
CONFIDENT_LOW = 0.1


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r, written out rather than pulled in from scipy."""
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = (
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    ) ** 0.5
    return round(numerator / denominator, 4) if denominator else None


def failure_inventory(rows: list[dict], threshold: float) -> list[dict]:
    """Every clip the detector gets wrong at this threshold."""
    failures = []
    for row in rows:
        is_spoof = row["label"] != "bonafide"
        called_spoof = row["spoof_probability"] >= threshold
        if is_spoof == called_spoof:
            continue
        failures.append(
            {
                "file_id": row["file_id"],
                "recording_id": row["recording_id"],
                "truth": row["label"],
                "attack": row["attack"],
                "spoof_probability": row["spoof_probability"],
                # ASVspoof naming: "acceptance" is acceptance as genuine.
                "error_type": "false_acceptance" if is_spoof else "false_rejection",
                "confidently_wrong": (
                    row["spoof_probability"] < CONFIDENT_LOW
                    if is_spoof
                    else row["spoof_probability"] > CONFIDENT_HIGH
                ),
                "duration": row["duration"],
                "truncated": row["truncated"],
            }
        )
    failures.sort(key=lambda row: row["spoof_probability"])
    return failures


def margin_analysis(rows: list[dict], threshold: float) -> dict:
    """How close the correct answers came to being wrong.

    A detector with no failures is not necessarily a safe one -- if its
    nearest-miss sits 0.005 from the threshold, the next clip flips it.
    """
    margins = [
        (row["spoof_probability"] - threshold)
        * (1 if row["label"] != "bonafide" else -1)
        for row in rows
    ]
    closest = min(margins)
    closest_row = rows[margins.index(closest)]
    return {
        "threshold": threshold,
        "minimum_correct_margin": round(min(m for m in margins if m >= 0), 6)
        if any(m >= 0 for m in margins)
        else None,
        "closest_clip": {
            "file_id": closest_row["file_id"],
            "truth": closest_row["label"],
            "attack": closest_row["attack"],
            "spoof_probability": closest_row["spoof_probability"],
            "signed_margin": round(closest, 6),
        },
        "clips_within_0.05_of_threshold": sum(
            1 for row in rows if abs(row["spoof_probability"] - threshold) < 0.05
        ),
        "clips_within_0.10_of_threshold": sum(
            1 for row in rows if abs(row["spoof_probability"] - threshold) < 0.10
        ),
    }


def threshold_sensitivity(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """What each candidate operating point costs, in both error directions."""
    from evaluate_model import confusion_at

    return [confusion_at(threshold, rows) for threshold in thresholds]


def duration_and_truncation(rows: list[dict]) -> dict:
    bonafide = [r for r in rows if r["label"] == "bonafide"]
    spoof = [r for r in rows if r["label"] != "bonafide"]
    return {
        "score_vs_duration_pearson_r": {
            "bonafide": _correlation(
                [r["duration"] for r in bonafide], [r["spoof_probability"] for r in bonafide]
            ),
            "spoof": _correlation(
                [r["duration"] for r in spoof], [r["spoof_probability"] for r in spoof]
            ),
            "all": _correlation(
                [r["duration"] for r in rows], [r["spoof_probability"] for r in rows]
            ),
        },
        "duration_seconds": {
            "bonafide": {
                "mean": round(statistics.mean([r["duration"] for r in bonafide]), 3),
                "min": min(r["duration"] for r in bonafide),
                "max": max(r["duration"] for r in bonafide),
            },
            "spoof": {
                "mean": round(statistics.mean([r["duration"] for r in spoof]), 3),
                "min": min(r["duration"] for r in spoof),
                "max": max(r["duration"] for r in spoof),
            },
        },
        "truncated_clips": sum(1 for r in rows if r["truncated"]),
        "note": (
            "A duration gap between the two classes is a property of the corpus. "
            "If it is large, a detector can score well by reading clip length "
            "instead of the voice -- which is what the silence ablation tests."
        ),
    }


def silence_ablation(model: str, rows: list[dict], per_class: int) -> dict:
    """Run the shipped Feature 2 probe over a stratified sample."""
    from app.tasks.deepfake.dataset import resolve_recording_path
    from app.tasks.deepfake.silence_probe import run_silence_probe

    bonafide = [r for r in rows if r["label"] == "bonafide"][:per_class]
    spoof = [r for r in rows if r["label"] != "bonafide"][:per_class]

    results = []
    for row in bonafide + spoof:
        path = resolve_recording_path(row["recording_id"])
        probe = run_silence_probe(model, path)
        variants = probe["variants"]
        results.append(
            {
                "file_id": row["file_id"],
                "truth": row["label"],
                "attack": row["attack"],
                "duration": probe["duration"],
                "speech_seconds": probe["speech_seconds"],
                "non_speech_seconds": probe["non_speech_seconds"],
                "non_speech_fraction": probe["non_speech_fraction"],
                "original_score": variants["original"]["spoof_probability"],
                "trimmed_score": variants["trimmed"]["spoof_probability"],
                "trimmed_applicable": variants["trimmed"]["applicable"],
                "non_speech_score": variants["non_speech"]["spoof_probability"],
                "non_speech_applicable": variants["non_speech"]["applicable"],
                "non_speech_reason": variants["non_speech"].get("reason"),
            }
        )
        print(
            f"  probe {row['file_id']} ({row['label']}): "
            f"original={variants['original']['spoof_probability']} "
            f"trimmed={variants['trimmed']['spoof_probability']} "
            f"silence={variants['non_speech']['spoof_probability']}",
            flush=True,
        )

    def _delta(key: str, subset: list[dict]) -> float | None:
        pairs = [(r["original_score"], r[key]) for r in subset if r[key] is not None]
        if not pairs:
            return None
        return round(statistics.mean(after - before for before, after in pairs), 4)

    genuine = [r for r in results if r["truth"] == "bonafide"]
    fake = [r for r in results if r["truth"] != "bonafide"]
    silence_only = [r for r in results if r["non_speech_score"] is not None]

    return {
        "sample_size": len(results),
        "per_class": per_class,
        "mean_score_change_when_trimmed": {
            "bonafide": _delta("trimmed_score", genuine),
            "spoof": _delta("trimmed_score", fake),
        },
        "mean_score_change_silence_only": {
            "bonafide": _delta("non_speech_score", genuine),
            "spoof": _delta("non_speech_score", fake),
        },
        "silence_only_scored_clips": len(silence_only),
        "silence_only_still_reads_spoof": sum(
            1 for r in silence_only if r["non_speech_score"] >= 0.5
        ),
        "non_speech_fraction": {
            "bonafide": round(
                statistics.mean([r["non_speech_fraction"] for r in genuine]), 4
            )
            if genuine
            else None,
            "spoof": round(statistics.mean([r["non_speech_fraction"] for r in fake]), 4)
            if fake
            else None,
        },
        "rows": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--probe-sample",
        type=int,
        default=10,
        help="clips per class for the silence ablation (0 skips it)",
    )
    args = parser.parse_args()

    from app.tasks.deepfake.service import get_model_spec

    spec = get_model_spec(args.model)
    rows = load_scores(args.model)
    metrics_path = OUTPUTS / "metrics" / f"{args.model}_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"Run src/evaluate_model.py --model {args.model} first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    eer_threshold = metrics["eer_threshold"]

    failures = failure_inventory(rows, spec.threshold)
    failures_at_eer = failure_inventory(rows, eer_threshold)

    out_dir = OUTPUTS / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if failures:
        with open(out_dir / f"{args.model}_failures.csv", "w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(failures[0]))
            writer.writeheader()
            writer.writerows(failures)

    report = {
        "model": args.model,
        "model_label": spec.label,
        "shipped_threshold": spec.threshold,
        "eer_threshold": eer_threshold,
        "failures_at_shipped_threshold": {
            "total": len(failures),
            "false_acceptances_spoof_missed": sum(
                1 for f in failures if f["error_type"] == "false_acceptance"
            ),
            "false_rejections_genuine_flagged": sum(
                1 for f in failures if f["error_type"] == "false_rejection"
            ),
            "confidently_wrong": sum(1 for f in failures if f["confidently_wrong"]),
            "by_attack": {
                attack: sum(1 for f in failures if f["attack"] == attack)
                for attack in sorted({f["attack"] for f in failures})
            },
            "clips": failures,
        },
        "failures_at_eer_threshold": {
            "total": len(failures_at_eer),
            "clips": failures_at_eer,
        },
        "margin_analysis": margin_analysis(rows, spec.threshold),
        "threshold_sensitivity": threshold_sensitivity(
            rows, sorted({0.1, 0.25, spec.threshold, round(eer_threshold, 6), 0.75, 0.9})
        ),
        "duration_and_truncation": duration_and_truncation(rows),
        "per_attack_weakest": sorted(
            [a for a in metrics["per_attack"] if a["is_spoof"]],
            key=lambda a: a["mean_score"],
        )[:3],
    }

    if args.probe_sample:
        print(f"Running the silence ablation on {args.probe_sample} clips per class...", flush=True)
        report["silence_ablation"] = silence_ablation(args.model, rows, args.probe_sample)
        ablation_rows = report["silence_ablation"]["rows"]
        with open(
            out_dir / f"{args.model}_silence_ablation.csv", "w", newline="", encoding="utf-8"
        ) as h:
            writer = csv.DictWriter(h, fieldnames=list(ablation_rows[0]))
            writer.writeheader()
            writer.writerows(ablation_rows)

    out = out_dir / f"{args.model}_error_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "silence_ablation"}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
