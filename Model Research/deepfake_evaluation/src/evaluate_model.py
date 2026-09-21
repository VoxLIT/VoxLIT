"""Turn one model's per-clip scores into the evaluation metrics.

EER, the DET curve and the error rates at a threshold come from the SHIPPED
code (`app.tasks.deepfake.metrics`), not from a second implementation. That is
deliberate: this script is as much a check on the code the app runs as it is a
measurement of the model, and a parallel implementation here would only prove
that two of my own functions agree with each other. Scikit-learn is used for
ROC-AUC alone, which the app does not compute.

Usage, from this folder:
    PYTHONPATH=../../Backend ../../Backend/.venv/bin/python src/evaluate_model.py --model xlsr-deepfake
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


def load_scores(model: str) -> list[dict]:
    path = OUTPUTS / "scores" / f"{model}_scores.csv"
    if not path.is_file():
        raise SystemExit(f"No scores for {model}. Run src/score_dataset.py first ({path}).")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["spoof_probability"] = float(row["spoof_probability"])
            row["duration"] = float(row["duration"])
            row["analysed_seconds"] = float(row["analysed_seconds"])
            row["truncated"] = row["truncated"].lower() == "true"
            rows.append(row)
    return rows


def confusion_at(threshold: float, rows: list[dict]) -> dict:
    """Counts with "positive" = spoof, the class the detector is built to flag.

    Note this is the OPPOSITE orientation to `metrics.rates_at`, which follows
    the ASVspoof convention where "accept" means accept as genuine. Both are
    reported, because reading a confusion matrix with the ASVspoof naming is
    the single easiest way to misread these results.
    """
    tp = sum(1 for r in rows if r["label"] != "bonafide" and r["spoof_probability"] >= threshold)
    fn = sum(1 for r in rows if r["label"] != "bonafide" and r["spoof_probability"] < threshold)
    tn = sum(1 for r in rows if r["label"] == "bonafide" and r["spoof_probability"] < threshold)
    fp = sum(1 for r in rows if r["label"] == "bonafide" and r["spoof_probability"] >= threshold)
    total = tp + fn + tn + fp
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": round(threshold, 6),
        "true_positives_spoof_caught": tp,
        "false_negatives_spoof_missed": fn,
        "true_negatives_genuine_passed": tn,
        "false_positives_genuine_flagged": fp,
        "accuracy": round((tp + tn) / total, 6) if total else 0.0,
        "balanced_accuracy": round((recall + specificity) / 2, 6),
        "precision": round(precision, 6),
        "recall_spoof_detection_rate": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1_score": round(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0, 6
        ),
        # ASVspoof naming, same numbers seen from the verification system's side.
        "false_acceptance_rate": round(fn / (tp + fn), 6) if tp + fn else 0.0,
        "false_rejection_rate": round(fp / (tn + fp), 6) if tn + fp else 0.0,
    }


def score_statistics(scores: list[float]) -> dict:
    return {
        "count": len(scores),
        "mean": round(statistics.mean(scores), 6),
        "median": round(statistics.median(scores), 6),
        "standard_deviation": round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0,
        "minimum": round(min(scores), 6),
        "maximum": round(max(scores), 6),
    }


def per_attack(rows: list[dict], threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        name = "bonafide" if row["label"] == "bonafide" else row["attack"]
        grouped.setdefault(name, []).append(row)

    summary = []
    for name, group in grouped.items():
        scores = [row["spoof_probability"] for row in group]
        is_spoof = name != "bonafide"
        flagged = sum(1 for score in scores if score >= threshold)
        summary.append(
            {
                "attack": name,
                "is_spoof": is_spoof,
                "count": len(group),
                "mean_score": round(statistics.mean(scores), 4),
                "median_score": round(statistics.median(scores), 4),
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
                # For an attack: the share this detector catches. For the
                # genuine row: the share it wrongly flags.
                "flagged_as_spoof": flagged,
                "detection_rate" if is_spoof else "false_alarm_rate": round(
                    flagged / len(group), 4
                ),
            }
        )
    summary.sort(key=lambda row: (row["is_spoof"], row["attack"]))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    from app.tasks.deepfake import metrics as shipped_metrics
    from app.tasks.deepfake.dataset import DATASET_ID
    from app.tasks.deepfake.service import THRESHOLD_VERSION, get_model_spec

    spec = get_model_spec(args.model)
    rows = load_scores(args.model)

    bonafide = [r["spoof_probability"] for r in rows if r["label"] == "bonafide"]
    spoof = [r["spoof_probability"] for r in rows if r["label"] != "bonafide"]

    computed = shipped_metrics.compute_metrics(bonafide, spoof)
    operating_far, operating_frr = shipped_metrics.rates_at(spec.threshold, bonafide, spoof)

    try:
        from sklearn.metrics import roc_auc_score

        auc = round(
            float(
                roc_auc_score(
                    [0 if r["label"] == "bonafide" else 1 for r in rows],
                    [r["spoof_probability"] for r in rows],
                )
            ),
            6,
        )
    except Exception as error:  # pragma: no cover - reported, not fatal
        auc = None
        print(f"  ! ROC-AUC unavailable: {error}", flush=True)

    det_path = OUTPUTS / "metrics" / f"{args.model}_det_curve.csv"
    det_path.parent.mkdir(parents=True, exist_ok=True)
    with open(det_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "false_acceptance_rate", "false_rejection_rate"])
        for point in computed.det_curve:
            writer.writerow(
                [
                    round(point.threshold, 6),
                    round(point.false_acceptance_rate, 6),
                    round(point.false_rejection_rate, 6),
                ]
            )

    payload = {
        "model": args.model,
        "model_label": spec.label,
        "model_id": spec.model_id,
        "model_tier": spec.tier,
        "dataset_id": DATASET_ID,
        "threshold_version": THRESHOLD_VERSION,
        "clips_scored": len(rows),
        "bonafide_count": computed.bonafide_count,
        "spoof_count": computed.spoof_count,
        "eer_percent": computed.eer_percent,
        "eer_threshold": computed.eer_threshold,
        "roc_auc": auc,
        "threshold_provenance": (
            f"Equal error rate on {DATASET_ID} ({computed.bonafide_count} genuine / "
            f"{computed.spoof_count} spoofed clips). Thresholds do not transfer "
            "between datasets."
        ),
        "shipped_operating_point": {
            "threshold": spec.threshold,
            "calibrated": spec.threshold_calibrated,
            "false_acceptance_rate": round(operating_far, 6),
            "false_rejection_rate": round(operating_frr, 6),
            **confusion_at(spec.threshold, rows),
        },
        "eer_operating_point": confusion_at(computed.eer_threshold, rows),
        "score_statistics": {
            "bonafide": score_statistics(bonafide),
            "spoof": score_statistics(spoof),
        },
        "separation": round(
            statistics.mean(spoof) - statistics.mean(bonafide), 6
        ),
        "truncated_clips": sum(1 for r in rows if r["truncated"]),
        "analysis_window_seconds": rows[0]["analysis_window_seconds"] if rows else None,
        "per_attack": per_attack(rows, spec.threshold),
        "det_curve_points": len(computed.det_curve),
        "det_curve_file": str(det_path.relative_to(ROOT)),
    }

    out = OUTPUTS / "metrics" / f"{args.model}_metrics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)} and {det_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
