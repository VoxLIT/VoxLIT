"""Detection metrics for Feature 1 — score distributions, DET curve, EER.

Implements SRS DF-6 (two score distributions on a common axis), DF-7 (the
Detection Error Tradeoff curve across all thresholds) and DF-8 (equal error
rate and the threshold at which it occurs).

SCORE CONVENTION
----------------
Every function here takes scores where HIGHER MEANS MORE LIKELY SPOOF, which
is what the detectors' `spoof_probability` already is. A clip is called spoof
when `score >= threshold`. From that:

    false acceptance — a spoofed clip let through as genuine (score < t)
    false rejection  — a genuine clip flagged as spoofed  (score >= t)

Note these are named from the verification system's point of view, which is
the ASVspoof convention: "accept" means "accept as genuine". Getting them the
wrong way round mirrors the DET curve and moves the EER, so the direction is
asserted in the tests.

The scores are RANKING scores, not calibrated probabilities — a threshold
derived here applies only to the dataset it came from (SRS DF-9).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetPoint:
    threshold: float
    false_acceptance_rate: float
    false_rejection_rate: float


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    eer_percent: float
    eer_threshold: float
    det_curve: list[DetPoint]
    bonafide_count: int
    spoof_count: int


class NotEnoughLabelledData(ValueError):
    """Raised when either class is missing, which makes an EER meaningless."""


def rates_at(threshold: float, bonafide: list[float], spoof: list[float]) -> tuple[float, float]:
    """(false acceptance rate, false rejection rate) at one threshold.

    Public because Feature 1 also reports the rates at the model's CURRENT
    operating threshold, not only at the EER point.
    """
    # A spoof clip scoring BELOW the threshold is wrongly accepted as genuine.
    false_acceptance = sum(1 for score in spoof if score < threshold) / len(spoof)
    # A genuine clip scoring AT OR ABOVE the threshold is wrongly rejected.
    false_rejection = sum(1 for score in bonafide if score >= threshold) / len(bonafide)
    return false_acceptance, false_rejection


def _candidate_thresholds(bonafide: list[float], spoof: list[float]) -> list[float]:
    """Every threshold at which the decision can change, plus the two ends.

    Sweeping the observed scores themselves is what makes this exact rather
    than an approximation over an arbitrary grid.
    """
    observed = sorted(set(bonafide) | set(spoof))
    # Below the lowest score everything is called genuine; above the highest,
    # everything is called spoof. Both ends are needed for a complete curve.
    lowest, highest = observed[0], observed[-1]
    span = max(highest - lowest, 1e-9)
    return [lowest - span * 1e-6, *observed, highest + span * 1e-6]


def det_curve(bonafide: list[float], spoof: list[float]) -> list[DetPoint]:
    """The full error tradeoff, one point per distinguishable threshold."""
    if not bonafide or not spoof:
        raise NotEnoughLabelledData(
            "A DET curve needs at least one genuine and one spoofed clip; "
            f"got {len(bonafide)} genuine and {len(spoof)} spoofed."
        )
    points = []
    for threshold in _candidate_thresholds(bonafide, spoof):
        far, frr = rates_at(threshold, bonafide, spoof)
        points.append(DetPoint(threshold, far, frr))
    return points


def equal_error_rate(bonafide: list[float], spoof: list[float]) -> tuple[float, float]:
    """(EER as a fraction, threshold where it occurs).

    The EER is where false acceptance and false rejection cross. With finite
    samples they rarely land exactly equal, so this takes the threshold that
    minimises the gap and reports the mean of the two rates there — the
    standard treatment.
    """
    points = det_curve(bonafide, spoof)
    best = min(points, key=lambda p: abs(p.false_acceptance_rate - p.false_rejection_rate))
    rate = (best.false_acceptance_rate + best.false_rejection_rate) / 2.0
    return rate, best.threshold


def score_histogram(scores: list[float], bins: int = 20) -> list[dict[str, float]]:
    """Counts over [0, 1], the range detector scores already live in.

    A fixed range (rather than one fitted to the data) is what lets the two
    distributions share a common axis, which is the point of DF-6.
    """
    width = 1.0 / bins
    counts = [0] * bins
    for score in scores:
        index = min(int(score / width), bins - 1)
        counts[index] += 1
    return [
        {
            "bin_start": round(index * width, 4),
            "bin_end": round((index + 1) * width, 4),
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def compute_metrics(bonafide: list[float], spoof: list[float]) -> DetectionMetrics:
    rate, threshold = equal_error_rate(bonafide, spoof)
    return DetectionMetrics(
        eer_percent=round(rate * 100, 3),
        eer_threshold=round(threshold, 6),
        det_curve=det_curve(bonafide, spoof),
        bonafide_count=len(bonafide),
        spoof_count=len(spoof),
    )


# ── Uncertainty and threshold-free summaries ─────────────────────────────────
#
# A point EER from a few hundred clips says little on its own: with 100 clips
# per class each error moves a rate by a whole percentage point, and an EER of
# exactly zero only bounds the true rate from above. These give the report
# the error bars a reader trained in statistics will ask for.

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20190901  # fixed, so a report is reproducible run to run
CONFIDENCE = 0.95


def _eer_fast(bonafide, spoof) -> float:
    """EER as a fraction, vectorised for the bootstrap.

    Same definition as `equal_error_rate` (thresholds at every observed score,
    pick the smallest |FAR - FRR|, report their mean); numpy only so a
    thousand resamples take well under a second.
    """
    import numpy as np

    bonafide = np.sort(np.asarray(bonafide, dtype=np.float64))
    spoof = np.sort(np.asarray(spoof, dtype=np.float64))
    observed = np.unique(np.concatenate([bonafide, spoof]))
    span = max(float(observed[-1] - observed[0]), 1e-9)
    thresholds = np.concatenate([[observed[0] - span * 1e-6], observed, [observed[-1] + span * 1e-6]])
    # spoof below t -> accepted as genuine; genuine at/above t -> rejected.
    far = np.searchsorted(spoof, thresholds, side="left") / spoof.size
    frr = 1.0 - np.searchsorted(bonafide, thresholds, side="left") / bonafide.size
    best = int(np.argmin(np.abs(far - frr)))
    return float((far[best] + frr[best]) / 2.0)


def bootstrap_eer_interval(
    bonafide: list[float],
    spoof: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the EER, as fractions.

    Stratified: each class is resampled with replacement at its own size, so
    every replicate keeps the evaluation's class balance. When the sample EER
    is 0 the replicates are all 0 too, which is why the report pairs this with
    `zero_error_upper_bound`.
    """
    import numpy as np

    if not bonafide or not spoof:
        raise NotEnoughLabelledData("A confidence interval needs both classes.")
    rng = np.random.default_rng(seed)
    genuine = np.asarray(bonafide, dtype=np.float64)
    attacks = np.asarray(spoof, dtype=np.float64)
    replicates = np.empty(resamples)
    for index in range(resamples):
        replicates[index] = _eer_fast(
            rng.choice(genuine, genuine.size, replace=True),
            rng.choice(attacks, attacks.size, replace=True),
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(replicates, [tail, 1.0 - tail])
    return float(low), float(high)


def clopper_pearson(errors: int, trials: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Exact binomial interval for an error rate of errors/trials."""
    from scipy.stats import beta

    if trials <= 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    low = 0.0 if errors == 0 else float(beta.ppf(alpha / 2, errors, trials - errors + 1))
    high = 1.0 if errors == trials else float(beta.ppf(1 - alpha / 2, errors + 1, trials - errors))
    return low, high


def zero_error_upper_bound(bonafide_count: int, spoof_count: int, confidence: float = CONFIDENCE) -> float:
    """Upper confidence bound on the EER when no clip of either class is wrong.

    The EER is the mean of two rates; each is bounded by its own one-sided
    exact binomial bound (1 - (1 - c)^(1/n), the "rule of three" ≈ 3/n), and
    the mean of the bounds bounds the mean.
    """
    alpha = 1.0 - confidence

    def one_sided(n: int) -> float:
        return 1.0 - alpha ** (1.0 / n) if n > 0 else 1.0

    return (one_sided(bonafide_count) + one_sided(spoof_count)) / 2.0


def roc_auc(bonafide: list[float], spoof: list[float]) -> float:
    """Area under the ROC curve, spoof as the positive class.

    The Mann-Whitney form: the probability that a random spoof clip outscores
    a random genuine one, ties counting half. Threshold-free, so it separates
    "the ranking is good" from "the cut is in the right place".
    """
    import numpy as np

    genuine = np.asarray(bonafide, dtype=np.float64)
    attacks = np.asarray(spoof, dtype=np.float64)
    greater = (attacks[:, None] > genuine[None, :]).sum()
    ties = (attacks[:, None] == genuine[None, :]).sum()
    return float((greater + 0.5 * ties) / (attacks.size * genuine.size))


def confusion_at(threshold: float, bonafide: list[float], spoof: list[float]) -> dict[str, float | int]:
    """Counts and the usual derived rates at one threshold (spoof = positive)."""
    tp = sum(1 for score in spoof if score >= threshold)
    fn = len(spoof) - tp
    fp = sum(1 for score in bonafide if score >= threshold)
    tn = len(bonafide) - fp
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    far_low, far_high = clopper_pearson(fn, len(spoof))
    frr_low, frr_high = clopper_pearson(fp, len(bonafide))
    return {
        "threshold": round(threshold, 6),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "accuracy": round((tp + tn) / total, 6) if total else 0.0,
        "balanced_accuracy": round((recall + specificity) / 2, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
        "false_acceptance_rate": round(fn / len(spoof), 6) if spoof else 0.0,
        "false_acceptance_ci": [round(far_low, 6), round(far_high, 6)],
        "false_rejection_rate": round(fp / len(bonafide), 6) if bonafide else 0.0,
        "false_rejection_ci": [round(frr_low, 6), round(frr_high, 6)],
    }


def score_summary(scores: list[float]) -> dict[str, float | int]:
    import numpy as np

    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "standard_deviation": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 6),
        "minimum": round(float(values.min()), 6),
        "maximum": round(float(values.max()), 6),
    }
