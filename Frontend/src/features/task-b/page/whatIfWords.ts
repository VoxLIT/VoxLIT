/** Plain-words reading of a perturbation comparison (Step 5), kept free of UI
 *  so it can be tested. Every number comes from the backend's `delta`; the
 *  client never recomputes the comparison. */
import type { DiarizationDelta, PerturbationRunSummary, PerturbationSpec } from "../types";
import { formatClock } from "./segmentWords";

/** Perturbed-run labels are arbitrary per run, so colour them by the original
 *  speaker the backend matched them to. A speaker with no match gets the next
 *  colour after the original run's, so it can never pass for someone else.
 *
 *  `speakers` is the list to hand to `speakerColor`; `keyOf` turns a perturbed
 *  label into its entry in that list. */
export function perturbedColourKey(
  delta: DiarizationDelta,
  originalSpeakers: string[],
  perturbedSpeakers: string[],
): { speakers: string[]; keyOf: (perturbedLabel: string) => string } {
  const unmatchedKey = (label: string) => `+${label}`;
  const speakers = [
    ...originalSpeakers,
    ...perturbedSpeakers.filter((label) => !(label in delta.speaker_mapping)).map(unmatchedKey),
  ];
  return {
    speakers,
    keyOf: (label) => delta.speaker_mapping[label] ?? unmatchedKey(label),
  };
}

/** A share (0..1) as a friendly percentage: "6%", "under 1%", "0%". */
export const formatShare = (share: number): string => {
  if (share <= 0) return "0%";
  if (share < 0.01) return "under 1%";
  return `${Math.round(share * 100)}%`;
};

/** Share of the original run's speech that the perturbed run credited to a
 *  different speaker — the backend's confusion component over reference speech. */
export const changedHandsShare = (delta: DiarizationDelta): number => {
  const { confusion, total } = delta.der_components;
  return total > 0 ? confusion / total : 0;
};

export const missedShare = (delta: DiarizationDelta): number => {
  const { missed_detection, total } = delta.der_components;
  return total > 0 ? missed_detection / total : 0;
};

/** The perturbation in words: "a little noise", "00:12.0–00:24.0 cut out". */
export function perturbationWords(spec: PerturbationSpec, duration: number): string {
  if (spec.type === "noise") {
    // noise_level is on the backend's 0..0.5 scale; the slider showed percent of that.
    const percent = ((spec.params.noise_level ?? 0) / 0.5) * 100;
    return percent < 10 ? "a little noise" : percent < 40 ? "some noise" : "a lot of noise";
  }
  const start = ((spec.params.mask_start_percent ?? 0) / 100) * duration;
  const end = ((spec.params.mask_end_percent ?? 0) / 100) * duration;
  return `${formatClock(start)}–${formatClock(end)} cut out`;
}

const structuralChanges = (delta: DiarizationDelta) =>
  delta.appeared.length + delta.disappeared.length + delta.merged.length + delta.split.length;

export interface WhatIfFinding {
  title: string;
  detail: string;
  alarming: boolean;
}

export function perturbationFinding(
  delta: DiarizationDelta,
  original: PerturbationRunSummary,
  perturbed: PerturbationRunSummary,
  spec: PerturbationSpec,
): WhatIfFinding {
  const before = original.num_speakers;
  const after = perturbed.num_speakers;
  const plural = (count: number) => `${count} speaker${count === 1 ? "" : "s"}`;

  let speakers: string;
  if (after === before) {
    const all = before === 1 ? "the one speaker" : before === 2 ? "both speakers" : `all ${plural(before)}`;
    speakers = structuralChanges(delta) > 0 ? `found ${plural(after)}, but regrouped some voices` : `still found ${all}`;
  } else if (after < before) {
    speakers = `found only ${after} of the ${plural(before)}`;
  } else {
    speakers = `found ${plural(after)} instead of ${before}`;
  }

  const changed = changedHandsShare(delta);
  const hands = changed > 0 ? `${formatShare(changed)} of the talk time changed hands` : "no talk time changed hands";
  const title = `With ${perturbationWords(spec, original.duration)}, the system ${speakers}; ${hands}.`;

  const missed = missedShare(delta);
  const details: string[] = [];
  if (missed >= 0.01) details.push(`${formatShare(missed)} of the speech was not picked up at all.`);
  if (delta.boundary_shifts.count > 0) {
    details.push(
      `${delta.boundary_shifts.count} turn boundar${delta.boundary_shifts.count === 1 ? "y" : "ies"} moved by more than ${delta.boundary_shifts.threshold_seconds} s.`,
    );
  }
  if (details.length === 0) details.push("The two timelines line up almost exactly.");

  return {
    title,
    detail: details.join(" "),
    alarming: after !== before || structuralChanges(delta) > 0 || changed >= 0.05 || missed >= 0.05,
  };
}

export type BadgeKind = "appeared" | "disappeared" | "merged" | "split";

export interface ChangeBadge {
  kind: BadgeKind;
  key: string;
  text: string;
}

const joinNames = (names: string[]) =>
  names.length <= 1 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;

/** One friendly badge per structural change. Merged and split speakers are
 *  named by their ORIGINAL labels, the ones the user saw in Steps 2–4. */
export function changeBadges(delta: DiarizationDelta): ChangeBadge[] {
  return [
    ...delta.merged.map((item) => ({
      kind: "merged" as const,
      key: `merged-${item.perturbed}`,
      text: `${joinNames(item.from)} were merged into one voice`,
    })),
    ...delta.split.map((item) => ({
      kind: "split" as const,
      key: `split-${item.original}`,
      text: `${item.original} was split into ${item.into.length} voices`,
    })),
    ...delta.disappeared.map((speaker) => ({
      kind: "disappeared" as const,
      key: `disappeared-${speaker}`,
      text: `${speaker} disappeared`,
    })),
    ...delta.appeared.map((speaker) => ({
      kind: "appeared" as const,
      key: `appeared-${speaker}`,
      text: `A new voice appeared (${speaker})`,
    })),
  ];
}
