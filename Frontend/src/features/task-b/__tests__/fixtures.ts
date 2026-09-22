import type { DiarizationResult, ProjectionResult } from "../types";

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
