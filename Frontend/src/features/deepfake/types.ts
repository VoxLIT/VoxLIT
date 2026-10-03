/** A recording in the ASVspoof 2019 LA demo subset, as returned by
 *  GET /tasks/deepfake/dataset/recordings. Carries no bona fide/spoof label —
 *  the ground truth is deliberately server-side only (see the backend's
 *  app/tasks/deepfake/dataset.py module docstring). */
export interface RecordingInfo {
  recording_id: string;
  display_filename: string;
  extension: string;
  size_bytes: number;
  /** Read from the file header; null when it could not be read. */
  duration_seconds?: number | null;
}

/** One of the visitor's own clips (POST/GET /tasks/deepfake/uploads). Same
 *  shape as a demo recording, so every per-clip view accepts either; the id
 *  starts with `up_`. Stored for 24 h, visible only to the session that made it. */
export interface UserClip extends RecordingInfo {
  source: "upload" | "recording";
  /** Unix seconds. */
  created_at: number;
}

/** Response shape of POST /tasks/deepfake/run. */
export interface DeepfakeResult {
  model: string;
  model_label: string;
  model_id: string;
  recording_id: string;

  decision: "spoof" | "bonafide";
  spoof_probability: number;
  bonafide_probability: number;
  logits: number[];

  /** The operating point the decision was taken at. */
  threshold: number;
  /** False until Feature 1 produces an EER-calibrated threshold. */
  threshold_calibrated: boolean;
  threshold_version: string;

  /** Which class index the checkpoint's own config calls "spoof". */
  id2label: Record<string, string>;
  spoof_index: number;

  duration: number;
  analysed_seconds: number;
  /** How much audio this model looks at at all. AST is fixed at 10.24s;
   *  wav2vec2 XLS-R has no architectural limit, so this is our own cap. */
  analysis_window_seconds: number;
  truncated: boolean;
  cached: boolean;
}

/** One bar of a score distribution (SRS DF-6). Bins always span 0..1 so the
 *  genuine and synthetic distributions share a common axis. */
export interface ScoreBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

/** One threshold's worth of the Detection Error Tradeoff curve (SRS DF-7).
 *  "Acceptance" is acceptance as GENUINE, the ASVspoof convention. */
export interface DetPoint {
  threshold: number;
  false_acceptance_rate: number;
  false_rejection_rate: number;
}

/** Mean score for one spoofing system, or for the genuine clips. */
export interface AttackSummary {
  attack: string;
  count: number;
  mean_score: number;
  is_spoof: boolean;
}

/** Response shape of POST /tasks/deepfake/scores — Feature 1.
 *  Aggregates only: no per-recording label is ever returned. */
export interface DeepfakeEvaluation {
  model: string;
  model_label: string;
  dataset_id: string;
  scored: number;
  bonafide_count: number;
  spoof_count: number;

  eer_percent: number;
  eer_threshold: number;
  /** "as_distributed" or "silence_trimmed" — see evaluation.CONDITIONS. */
  condition?: EvaluationCondition;
  condition_label?: string;
  unlabelled_skipped?: number;
  /** Stratified percentile bootstrap interval for the EER, in percent. */
  eer_ci_percent?: [number, number];
  /** Exact one-sided bound on the EER when no clip is wrong, in percent. */
  eer_zero_upper_bound_percent?: number;
  /** Half of 1/n: the smallest EER step this many clips can resolve. */
  eer_resolution_percent?: number;
  confidence_level?: number;
  bootstrap_resamples?: number;
  roc_auc?: number;
  score_statistics?: { bonafide: ScoreStatistics; spoof: ScoreStatistics };
  confusion?: { operating: Confusion; eer: Confusion };
  /** SRS DF-9 — a threshold means nothing without the dataset it came from. */
  threshold_provenance: string;

  distributions: { bonafide: ScoreBin[]; spoof: ScoreBin[] };
  det_curve: DetPoint[];

  /** Where the model's shipped threshold currently sits on that curve. */
  operating_point: {
    threshold: number;
    calibrated: boolean;
    false_acceptance_rate: number;
    false_rejection_rate: number;
  };

  per_attack: AttackSummary[];
}

export type EvaluationCondition = "as_distributed" | "silence_trimmed";

export interface ScoreStatistics {
  count: number;
  mean: number;
  median: number;
  standard_deviation: number;
  minimum: number;
  maximum: number;
}

/** Counts at one threshold, spoof as the positive class. */
export interface Confusion {
  threshold: number;
  true_positives: number;
  false_negatives: number;
  false_positives: number;
  true_negatives: number;
  accuracy: number;
  balanced_accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  false_acceptance_rate: number;
  false_acceptance_ci: [number, number];
  false_rejection_rate: number;
  false_rejection_ci: [number, number];
}

/** A researcher's own dataset (Manage Datasets). Label COUNTS only. */
/** One of the task's built-in labelled subsets. Counts only, never labels. */
export interface BuiltinDataset {
  dataset_id: string;
  label: string;
  description: string;
  citation: string;
  license: string;
  total_recordings: number;
  available: boolean;
}

export interface CustomDataset {
  dataset_name: string;
  created_at: number | null;
  updated_at: number | null;
  total_files: number;
  total_duration_seconds: number;
  labels: { provided: boolean; matched_files: number; bonafide: number; spoof: number };
}

/** One of the three scorings in the silence probe (SRS DF-10).
 *  `applicable: false` means there was too little audio of that kind to
 *  score meaningfully — DF-12 requires saying so rather than guessing. */
export interface ProbeVariant {
  applicable: boolean;
  /** false when scored on less audio than the reliability floor (0.5 s). */
  reliable?: boolean;
  seconds: number;
  spoof_probability: number | null;
  decision: "spoof" | "bonafide" | null;
  reason?: string;
}

/** Response shape of POST /tasks/deepfake/silence-probe — Feature 2. */
export interface SilenceProbeResult {
  model: string;
  model_label: string;
  recording_id: string;

  threshold: number;
  threshold_calibrated: boolean;
  /** SRS DF-11 — the energy threshold that defined "silence", reported with
   *  the result. Relative to the clip's own peak, not an absolute floor. */
  silence_top_db: number;
  min_non_speech_seconds: number;
  min_speech_seconds?: number;
  min_scorable_seconds?: number;

  duration: number;
  speech_seconds: number;
  non_speech_seconds: number;
  non_speech_fraction: number;
  /** [start, end] seconds of each detected speech region. */
  speech_intervals: [number, number][];

  variants: {
    original: ProbeVariant;
    trimmed: ProbeVariant;
    non_speech: ProbeVariant;
  };

  sample_rate: number;
  cached: boolean;
}

/** One time slice of the attribution. Shape matches the shared saliency
 *  service's segments exactly, so the payload is interchangeable with it. */
export interface SaliencySegment {
  start_time: number;
  end_time: number;
  saliency: number;
  intensity: number;
}

/** Response shape of POST /tasks/deepfake/saliency — Feature 3. */
export interface DeepfakeSaliency {
  model: string;
  model_label: string;
  recording_id: string;

  /** Machine name of the attribution method (shared-service contract). */
  method: string;
  /** SRS DF-15 — human-readable method name, shown in the interface. */
  method_label: string;
  /** What the attribution explains (the spoof − bonafide logit margin). */
  target: string;

  segments: SaliencySegment[];
  /** Normalised 0..1 attribution, one value per segment. */
  series: number[];
  total_duration: number;

  /** The shared saliency service's duration cap, applied here too (DF-15). */
  max_saliency_seconds: number;
  analysis_window_seconds: number | null;
  truncated: boolean;
  /** Attribution is ranked within this clip only — never comparable across
   *  clips or models. */
  normalised: boolean;

  /** Speech regions from Feature 2's segmentation, so the heat can be read
   *  against where the voice actually is. */
  speech_intervals: [number, number][];
  silence_top_db: number;
  /** Share of total attribution falling inside speech. Low means the model
   *  reacted to something other than the voice. */
  saliency_in_speech_fraction: number | null;

  cached: boolean;
}

/** One point of the embedding view. Carries the detector's own score, never
 *  the dataset's bona fide/spoof label. */
export interface EmbeddingRecording {
  recording_id: string;
  display_filename: string;
  spoof_probability: number;
  /** The model's decision at its shipped threshold. */
  decision: "spoof" | "bonafide";
  /** Present (true) only on the visitor's own clips, placed on the dataset map. */
  uploaded?: boolean;
}

/** Response shape of POST /tasks/deepfake/embeddings — Feature 4.
 *  `coordinates` is index-aligned with `recordings`. */
export interface DeepfakeEmbeddingProjection {
  model: string;
  model_label: string;
  reduction_method: string;
  /** Differs from `reduction_method` only when the reducer had to fall back. */
  reduction_method_used: string;
  n_components: 2 | 3;
  effective_components: number;
  /** Width of the vector the detector's classification head reads. */
  embedding_dimension: number;
  total_recordings: number;
  threshold: number;
  threshold_calibrated: boolean;
  recordings: EmbeddingRecording[];
  coordinates: number[][];
}
