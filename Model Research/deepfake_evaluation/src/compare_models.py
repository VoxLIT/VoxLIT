"""Do two detectors fail on the same clips, or on different ones?

Model A reads the waveform (wav2vec2 XLS-R); Model C reads the same waveform
through a bidirectional Mamba stack with a much shorter analysis window. If
their errors are uncorrelated, the pair is genuinely informative and the
workbench's model switch teaches something. If they fail on the same clips,
running both only costs time.

Usage, from this folder:
    PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/compare_models.py --models xlsr-deepfake xlsr-mamba
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


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = float(position)
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation -- the two models' score SCALES are not comparable
    (Model C saturates at 1e-4 and 0.999), only their orderings are."""
    rx, ry = _rank(xs), _rank(ys)
    mean_x, mean_y = statistics.mean(rx), statistics.mean(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = (
        sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry)
    ) ** 0.5
    return round(numerator / denominator, 4) if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs=2, required=True)
    args = parser.parse_args()

    from app.tasks.deepfake.service import get_model_spec

    left, right = args.models
    left_rows = {row["file_id"]: row for row in load_scores(left)}
    right_rows = {row["file_id"]: row for row in load_scores(right)}
    shared = sorted(set(left_rows) & set(right_rows))
    if not shared:
        raise SystemExit("The two models have no clips in common.")

    left_threshold = get_model_spec(left).threshold
    right_threshold = get_model_spec(right).threshold

    disagreements = []
    both_wrong = []
    agree = 0
    for file_id in shared:
        a, c = left_rows[file_id], right_rows[file_id]
        truth_is_spoof = a["label"] != "bonafide"
        a_says_spoof = a["spoof_probability"] >= left_threshold
        c_says_spoof = c["spoof_probability"] >= right_threshold

        if a_says_spoof == c_says_spoof:
            agree += 1
            if a_says_spoof != truth_is_spoof:
                both_wrong.append(file_id)
        else:
            disagreements.append(
                {
                    "file_id": file_id,
                    "truth": a["label"],
                    "attack": a["attack"],
                    f"{left}_score": a["spoof_probability"],
                    f"{right}_score": c["spoof_probability"],
                    "correct_model": right if c_says_spoof == truth_is_spoof else left,
                }
            )

    left_errors = {
        file_id
        for file_id in shared
        if (left_rows[file_id]["spoof_probability"] >= left_threshold)
        != (left_rows[file_id]["label"] != "bonafide")
    }
    right_errors = {
        file_id
        for file_id in shared
        if (right_rows[file_id]["spoof_probability"] >= right_threshold)
        != (right_rows[file_id]["label"] != "bonafide")
    }

    report = {
        "models": [left, right],
        "clips_compared": len(shared),
        "decision_agreement_rate": round(agree / len(shared), 4),
        "spearman_rank_correlation_of_scores": _spearman(
            [left_rows[f]["spoof_probability"] for f in shared],
            [right_rows[f]["spoof_probability"] for f in shared],
        ),
        f"{left}_errors": sorted(left_errors),
        f"{right}_errors": sorted(right_errors),
        "errors_in_common": sorted(left_errors & right_errors),
        "errors_unique_to_one_model": sorted(left_errors ^ right_errors),
        "both_models_wrong_and_agreeing": both_wrong,
        "disagreements": disagreements,
        "analysis_window_seconds": {
            left: left_rows[shared[0]]["analysis_window_seconds"],
            right: right_rows[shared[0]]["analysis_window_seconds"],
        },
        "clips_truncated": {
            left: sum(1 for f in shared if left_rows[f]["truncated"]),
            right: sum(1 for f in shared if right_rows[f]["truncated"]),
        },
        "interpretation": (
            "Errors sitting on different clips means the two detectors fail "
            "independently, so a disagreement between them is a real signal "
            "for the user. Errors in common would mean the corpus, not the "
            "architecture, decides what gets missed."
        ),
    }

    out_dir = OUTPUTS / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{left}_vs_{right}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if disagreements:
        with open(out_dir / f"{left}_vs_{right}_disagreements.csv", "w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(disagreements[0]))
            writer.writeheader()
            writer.writerows(disagreements)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
