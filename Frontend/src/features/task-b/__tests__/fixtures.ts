import type { DiarizationResult, PerturbationResult, ProjectionResult } from "../types";

/**
 * A small run built so that the map and the model disagree: on the 2D map
 * `seg_2` sits right next to `seg_1`, but in embedding space `seg_1`'s closest
 * match is `seg_4`, which belongs to the OTHER speaker. `seg_5` is too short
 * to embed, so it has no embedding, no dot and no confidence.
 */
export const result: DiarizationResult = {
  model: "pyannote-3.1",
  recording_id: "rec_demo",
  duration: 20,
  num_speakers: 2,
  speakers: ["SPEAKER_00", "SPEAKER_01"],
  segments: [
    { id: "seg_1", start: 0, end: 3, speaker: "SPEAKER_00", confidence: 0.7, confidence_bucket: "high" },
    { id: "seg_2", start: 3.5, end: 6, speaker: "SPEAKER_00", confidence: 0.3, confidence_bucket: "medium" },
    { id: "seg_3", start: 7, end: 10, speaker: "SPEAKER_01", confidence: 0.6, confidence_bucket: "high" },
    { id: "seg_4", start: 11, end: 14, speaker: "SPEAKER_01", confidence: 0.1, confidence_bucket: "uncertain" },
    { id: "seg_5", start: 15, end: 15.3, speaker: "SPEAKER_00", confidence: null, confidence_bucket: null },
  ],
  embeddings: {
    seg_1: [1, 0, 0],
    seg_2: [0.9, 0.1, 0],
    seg_3: [0, 1, 0],
    seg_4: [1, 0, 0.01],
  },
  cached: true,
};

export const projection: ProjectionResult = {
  model: "pyannote-3.1",
  recording_id: "rec_demo",
  points: [
    { id: "seg_1", x: 0, y: 0, speaker: "SPEAKER_00", confidence: 0.7 },
    { id: "seg_2", x: 0.1, y: 0, speaker: "SPEAKER_00", confidence: 0.3 },
    { id: "seg_3", x: -4, y: 3, speaker: "SPEAKER_01", confidence: 0.6 },
    { id: "seg_4", x: 5, y: 5, speaker: "SPEAKER_01", confidence: 0.1 },
  ],
};

/** The same four segments with a third PCA axis. */
export const projection3d: ProjectionResult = {
  model: "pyannote-3.1",
  recording_id: "rec_demo",
  dims: 3,
  explained_variance: [0.5, 0.3, 0.12],
  points: projection.points.map((point, index) => ({ ...point, z: index - 1.5 })),
};

/**
 * A noise run over `result` in which the perturbed run swapped the two labels
 * (its SPEAKER_01 is the original SPEAKER_00 and vice versa) and found a third
 * voice. Colouring by the perturbed run's own labels would swap the colours.
 */
export const perturbation: PerturbationResult = {
  model: "pyannote-3.1",
  perturbation: { type: "noise", params: { noise_level: 0.02, seed: 1234 } },
  original: {
    recording_id: "rec_demo",
    duration: result.duration,
    num_speakers: result.num_speakers,
    speakers: result.speakers,
    segments: result.segments,
  },
  perturbed: {
    recording_id: "prt_demo",
    duration: 20,
    num_speakers: 3,
    speakers: ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
    segments: [
      { id: "seg_1", start: 0, end: 3, speaker: "SPEAKER_01", confidence: 0.7, confidence_bucket: "high" },
      { id: "seg_2", start: 7, end: 10, speaker: "SPEAKER_00", confidence: 0.6, confidence_bucket: "high" },
      { id: "seg_3", start: 16, end: 18, speaker: "SPEAKER_02", confidence: 0.1, confidence_bucket: "uncertain" },
    ],
  },
  delta: {
    der: 0.3,
    der_is_vs_original_run: true,
    der_components: { missed_detection: 1, false_alarm: 2, confusion: 0.6, total: 12 },
    speaker_mapping: { SPEAKER_01: "SPEAKER_00", SPEAKER_00: "SPEAKER_01" },
    appeared: ["SPEAKER_02"],
    disappeared: [],
    merged: [],
    split: [],
    num_speakers_delta: 1,
    boundary_shifts: { threshold_seconds: 0.5, count: 0, shifts: [] },
    lost_segments: ["seg_4"],
    diff_regions: [{ start: 11, end: 14 }],
  },
  cached: false,
};
