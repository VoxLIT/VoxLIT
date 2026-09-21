/** Response shapes from /tasks/task-b (see Backend/app/tasks/task_b/router.py). */

/** Demo recordings and session uploads share this shape deliberately — they
 *  differ only in the `recording_id` prefix (`rec_` for AMI, `upl_` for an
 *  upload), which selects the audio endpoint and nothing else. */
export interface RecordingInfo {
  recording_id: string;
  display_filename: string;
  extension: string;
  size_bytes: number;
}

/** One entry from GET /tasks/task-b/models — the backend `DiarizationModelSpec`.
 *  `glass_box_note` says what the selected model actually changes, so the
 *  header never implies more separation between the three than exists. */
export interface DiarizationModelInfo {
  key: string;
  label: string;
  pipeline_id: string;
  embedding_model_id: string;
  embedding_dimension: number;
  recommended: boolean;
  glass_box_note: string;
}

export type ConfidenceBucket = "high" | "medium" | "uncertain" | null;

export interface DiarizationSegment {
  id: string;
  start: number;
  end: number;
  speaker: string;
  confidence: number | null;
  confidence_bucket: ConfidenceBucket;
}

export interface DiarizationResult {
  model: string;
  recording_id: string;
  duration: number;
  num_speakers: number;
  speakers: string[];
  segments: DiarizationSegment[];
  embeddings: Record<string, number[]>;
  cached: boolean;
}

export interface ProjectionPoint {
  id: string;
  x: number;
  y: number;
  speaker: string;
  confidence: number | null;
}

export interface ProjectionResult {
  model: string;
  recording_id: string;
  points: ProjectionPoint[];
}

/* --- Perturbation counterfactuals (see Backend/app/tasks/task_b/diff.py) --- */

export type PerturbationType = "noise" | "time_masking";

export interface PerturbationSpec {
  type: PerturbationType;
  /** Normalized by the backend — noise carries the `seed` it actually used. */
  params: Record<string, number>;
}

/** One diarization run, as the comparison view needs it (no embeddings). */
export interface PerturbationRunSummary {
  recording_id: string;
  duration: number;
  num_speakers: number;
  speakers: string[];
  segments: DiarizationSegment[];
}

export interface BoundaryShift {
  segment_id: string;
  edge: "start" | "end";
  original: number;
  perturbed: number;
  delta: number;
}

export interface DiarizationDelta {
  /** DER against the ORIGINAL RUN, not ground truth. Never label it accuracy. */
  der: number;
  der_is_vs_original_run: true;
  /** All in seconds. */
  der_components: {
    missed_detection: number;
    false_alarm: number;
    confusion: number;
    total: number;
  };
  /** perturbed label → original label, matched labels only. */
  speaker_mapping: Record<string, string>;
  appeared: string[];
  disappeared: string[];
  merged: { perturbed: string; from: string[] }[];
  split: { original: string; into: string[] }[];
  num_speakers_delta: number;
  boundary_shifts: {
    threshold_seconds: number;
    count: number;
    shifts: BoundaryShift[];
  };
  lost_segments: string[];
  /** Spans where the runs disagree after label alignment — drives the red strip. */
  diff_regions: { start: number; end: number }[];
}

export interface PerturbationResult {
  model: string;
  perturbation: PerturbationSpec;
  original: PerturbationRunSummary;
  perturbed: PerturbationRunSummary;
  delta: DiarizationDelta;
  cached: boolean;
}

/** Stable palette: speaker → color by index in result.speakers. */
export const SPEAKER_COLORS = [
  "#2563eb", // blue
  "#dc2626", // red
  "#16a34a", // green
  "#d97706", // amber
  "#9333ea", // purple
  "#0891b2", // cyan
  "#db2777", // pink
  "#65a30d", // lime
];

export const speakerColor = (speakers: string[], speaker: string): string =>
  SPEAKER_COLORS[Math.max(0, speakers.indexOf(speaker)) % SPEAKER_COLORS.length];