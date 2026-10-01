import { describe, expect, it } from "vitest";
import {
  buildPairs,
  changedPairs,
  equalErrorRate,
  errorCurve,
  metricsAt,
  type ScoredPair,
} from "../thresholdMetrics";

// Recordings 0-2 are speaker A, recording 3 is speaker B.
// Same-speaker similarities: 0.9, 0.8, 0.4. Different-speaker: 0.6, 0.2, 0.1.
const similarity_matrix = [
  [1.0, 0.9, 0.8, 0.6],
  [0.9, 1.0, 0.4, 0.2],
  [0.8, 0.4, 1.0, 0.1],
  [0.6, 0.2, 0.1, 1.0],
];
const labelled = {
  similarity_matrix,
  ground_truth_groups: ["A", "A", "A", "B"],
  ground_truth_available: true,
};
const unlabelled = { similarity_matrix, ground_truth_groups: null, ground_truth_available: false };

const pair = (similarity: number, sameSpeaker: boolean | null, i = 0, j = 1): ScoredPair => ({
  i,
  j,
  similarity,
  sameSpeaker,
});

describe("buildPairs", () => {
  it("uses each unique pair once, i < j, never the diagonal", () => {
    const pairs = buildPairs(labelled);

    expect(pairs).toHaveLength(6);
    expect(pairs.every((p) => p.i < p.j)).toBe(true);
    expect(pairs.map((p) => `${p.i}-${p.j}`)).toEqual(["0-1", "0-2", "0-3", "1-2", "1-3", "2-3"]);
    expect(pairs.map((p) => p.similarity)).toEqual([0.9, 0.8, 0.6, 0.4, 0.2, 0.1]);
  });

  it("marks pairs same-speaker only when both ground-truth groups match", () => {
    expect(buildPairs(labelled).map((p) => p.sameSpeaker)).toEqual([true, true, false, true, false, false]);
  });

  it("leaves sameSpeaker null without ground truth", () => {
    expect(buildPairs(unlabelled).every((p) => p.sameSpeaker === null)).toBe(true);
    // Groups present but flagged unavailable are ignored too.
    expect(
      buildPairs({ ...labelled, ground_truth_available: false }).every((p) => p.sameSpeaker === null)
    ).toBe(true);
  });

  it("returns no pairs for fewer than two recordings", () => {
    expect(buildPairs({ similarity_matrix: [[1]], ground_truth_groups: ["A"], ground_truth_available: true })).toEqual(
      []
    );
  });
});

describe("metricsAt", () => {
  it("computes accepted count, FAR, FRR and accuracy on a hand-made matrix", () => {
    const metrics = metricsAt(buildPairs(labelled), 0.5);

    expect(metrics.totalPairs).toBe(6);
    expect(metrics.accepted).toBe(3); // 0.9, 0.8, 0.6
    expect(metrics.samePairs).toBe(3);
    expect(metrics.differentPairs).toBe(3);
    expect(metrics.falseAccepts).toBe(1); // 0.6
    expect(metrics.falseRejects).toBe(1); // 0.4
    expect(metrics.far).toBeCloseTo(1 / 3);
    expect(metrics.frr).toBeCloseTo(1 / 3);
    expect(metrics.accuracy).toBeCloseTo(4 / 6);
  });

  it("computes balanced accuracy on a hand-made unbalanced case", () => {
    // 1 same-speaker pair (accepted), 4 different-speaker pairs (1 accepted).
    const pairs = [
      pair(0.9, true),
      pair(0.7, false, 0, 2),
      pair(0.3, false, 0, 3),
      pair(0.2, false, 1, 2),
      pair(0.1, false, 1, 3),
    ];
    const metrics = metricsAt(pairs, 0.5);

    expect(metrics.far).toBe(0.25);
    expect(metrics.frr).toBe(0);
    expect(metrics.balancedAccuracy).toBeCloseTo(0.875); // ((1 - 0.25) + (1 - 0)) / 2
    expect(metrics.accuracy).toBeCloseTo(0.8);
    expect(metricsAt(buildPairs(labelled), 0.5).balancedAccuracy).toBeCloseTo(2 / 3);
  });

  it("gives a balanced accuracy of 0.5 when everything is rejected or everything is accepted", () => {
    const pairs = buildPairs(labelled);

    const rejectAll = metricsAt(pairs, 0.95);
    expect(rejectAll.accepted).toBe(0);
    expect(rejectAll.balancedAccuracy).toBe(0.5);

    const acceptAll = metricsAt(pairs, 0.05);
    expect(acceptAll.accepted).toBe(6);
    expect(acceptAll.balancedAccuracy).toBe(0.5);
  });

  it("returns a null balanced accuracy when either rate is null", () => {
    expect(metricsAt([pair(0.7, false)], 0.5).balancedAccuracy).toBeNull();
    expect(metricsAt([pair(0.7, true)], 0.5).balancedAccuracy).toBeNull();
    expect(metricsAt(buildPairs(unlabelled), 0.5).balancedAccuracy).toBeNull();
  });

  it("accepts a pair whose similarity equals the threshold", () => {
    const metrics = metricsAt(buildPairs(labelled), 0.6);
    expect(metrics.accepted).toBe(3);
    expect(metrics.falseAccepts).toBe(1);
  });

  it("returns only the accepted count without ground truth", () => {
    const metrics = metricsAt(buildPairs(unlabelled), 0.5);

    expect(metrics.accepted).toBe(3);
    expect(metrics.totalPairs).toBe(6);
    expect(metrics.samePairs).toBeNull();
    expect(metrics.differentPairs).toBeNull();
    expect(metrics.falseAccepts).toBeNull();
    expect(metrics.falseRejects).toBeNull();
    expect(metrics.far).toBeNull();
    expect(metrics.frr).toBeNull();
    expect(metrics.accuracy).toBeNull();
  });

  it("returns a null FRR, never NaN, when there are no same-speaker pairs", () => {
    const metrics = metricsAt([pair(0.7, false), pair(0.2, false, 0, 2)], 0.5);

    expect(metrics.samePairs).toBe(0);
    expect(metrics.frr).toBeNull();
    expect(metrics.far).toBe(0.5);
    expect(metrics.accuracy).toBe(0.5);
  });

  it("returns a null FAR, never NaN, when there are no different-speaker pairs", () => {
    const metrics = metricsAt([pair(0.7, true), pair(0.2, true, 0, 2)], 0.5);

    expect(metrics.differentPairs).toBe(0);
    expect(metrics.far).toBeNull();
    expect(metrics.frr).toBe(0.5);
    expect(metrics.accuracy).toBe(0.5);
  });

  it("handles an empty pair list", () => {
    const metrics = metricsAt([], 0.5);
    expect(metrics.accepted).toBe(0);
    expect(metrics.far).toBeNull();
    expect(metrics.accuracy).toBeNull();
  });
});

describe("errorCurve", () => {
  it("spans the lowest to the highest similarity with matching FAR and FRR", () => {
    const pairs = buildPairs(labelled);
    const curve = errorCurve(pairs, 9);

    expect(curve).toHaveLength(9);
    expect(curve[0].threshold).toBe(0.1);
    expect(curve[8].threshold).toBe(0.9);
    // Lowest threshold accepts everything; the highest only the top pair.
    expect(curve[0]).toMatchObject({ far: 1, frr: 0 });
    expect(curve[8].far).toBe(0);
    expect(curve[8].frr).toBeCloseTo(2 / 3);
    for (const point of curve) {
      const metrics = metricsAt(pairs, point.threshold);
      expect(point.far).toBeCloseTo(metrics.far as number);
      expect(point.frr).toBeCloseTo(metrics.frr as number);
    }
  });

  it("returns null rates for an empty class and nothing for no pairs", () => {
    const curve = errorCurve([pair(0.7, false), pair(0.2, false, 0, 2)], 5);
    expect(curve.every((point) => point.frr === null && point.far !== null)).toBe(true);
    expect(errorCurve([], 5)).toEqual([]);
  });

  it("returns a single point when every similarity is identical", () => {
    expect(errorCurve([pair(0.5, true), pair(0.5, false, 0, 2)], 10)).toHaveLength(1);
  });
});

describe("equalErrorRate", () => {
  it("finds the threshold where FAR and FRR meet on a known case", () => {
    const eer = equalErrorRate(buildPairs(labelled));

    expect(eer).not.toBeNull();
    expect(eer?.threshold).toBe(0.6);
    expect(eer?.far).toBeCloseTo(1 / 3);
    expect(eer?.frr).toBeCloseTo(1 / 3);
    expect(eer?.eer).toBeCloseTo(1 / 3);
  });

  it("is zero for perfectly separated scores", () => {
    const eer = equalErrorRate([pair(0.9, true), pair(0.8, true, 0, 2), pair(0.2, false, 0, 3), pair(0.1, false, 1, 2)]);

    expect(eer?.eer).toBe(0);
    expect(eer?.threshold).toBe(0.8);
  });

  it("averages FAR and FRR when they never cross exactly", () => {
    // At 0.5: FAR 1/2, FRR 1/3 (closest). At 0.7: FAR 0, FRR 1/3.
    const eer = equalErrorRate([
      pair(0.9, true),
      pair(0.7, true, 0, 2),
      pair(0.4, true, 0, 3),
      pair(0.5, false, 1, 2),
      pair(0.3, false, 1, 3),
    ]);

    expect(eer?.threshold).toBe(0.5);
    expect(eer?.eer).toBeCloseTo(5 / 12);
  });

  it("is null without ground truth or with an empty class", () => {
    expect(equalErrorRate(buildPairs(unlabelled))).toBeNull();
    expect(equalErrorRate([pair(0.7, true)])).toBeNull();
    expect(equalErrorRate([pair(0.7, false)])).toBeNull();
    expect(equalErrorRate([])).toBeNull();
  });
});

describe("changedPairs", () => {
  const pairs = buildPairs(labelled);

  it("returns nothing when both thresholds are the same", () => {
    expect(changedPairs(pairs, 0.5, 0.5)).toEqual([]);
  });

  it("lists pairs newly accepted when the threshold is lowered", () => {
    const changed = changedPairs(pairs, 0.5, 0.3);

    expect(changed).toHaveLength(1);
    expect(changed[0]).toMatchObject({
      i: 1,
      j: 2,
      similarity: 0.4,
      acceptedAtCalibrated: false,
      acceptedAtWhatIf: true,
    });
  });

  it("lists pairs newly rejected when the threshold is raised, nearest the calibrated value first", () => {
    const changed = changedPairs(pairs, 0.5, 0.85);

    expect(changed.map((p) => p.similarity)).toEqual([0.6, 0.8]);
    expect(changed.every((p) => p.acceptedAtCalibrated && !p.acceptedAtWhatIf)).toBe(true);
  });

  it("works without ground truth", () => {
    expect(changedPairs(buildPairs(unlabelled), 0.5, 0.3)).toHaveLength(1);
  });
});
