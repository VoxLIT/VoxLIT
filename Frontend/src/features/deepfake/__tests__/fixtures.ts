/**
 * Response fixtures for the deepfake component tests.
 *
 * Importing this file also registers the mock clean up these tests need. It is
 * done here and not in a shared setup file on purpose: the runner is shared with
 * other tasks, and resetting every mock after every test would be a surprise for
 * their tests.
 *
 * The numbers are the REAL ones measured on the ASVspoof 2019 LA demo subset
 * (see Model Research/deepfake_evaluation/outputs), not invented values. That
 * matters for the formatting assertions: a fixture with a tidy 0.5 would not
 * catch a component that rounds to two decimals where the spec says three.
 */
import { afterEach, vi } from "vitest";
import type {
  DeepfakeEmbeddingProjection,
  DeepfakeEvaluation,
  DeepfakeResult,
  DeepfakeSaliency,
  SilenceProbeResult,
} from "../types";

export const detection: DeepfakeResult = {
  model: "xlsr-deepfake",
  model_label: "wav2vec2 XLS-R (Model A)",
  model_id: "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
  recording_id: "rec_e3b439fcd7c6906d",
  decision: "bonafide",
  spoof_probability: 0.142884,
  bonafide_probability: 0.857116,
  logits: [1.42, -0.37],
  threshold: 0.5,
  threshold_calibrated: false,
  threshold_version: "deepfake-threshold-v0-uncalibrated",
  id2label: { "0": "bonafide", "1": "spoof" },
  spoof_index: 1,
  duration: 3.53,
  analysed_seconds: 3.53,
  analysis_window_seconds: 30,
  truncated: false,
  cached: false,
};

/** Model A's measured evaluation, trimmed to a readable number of bins. */
export const evaluation: DeepfakeEvaluation = {
  model: "xlsr-deepfake",
  model_label: "wav2vec2 XLS-R (Model A)",
  dataset_id: "asvspoof2019-la",
  scored: 200,
  bonafide_count: 100,
  spoof_count: 100,
  eer_percent: 0,
  eer_threshold: 0.142884,
  threshold_provenance:
    "Equal error rate on asvspoof2019-la (100 genuine / 100 spoofed clips). " +
    "Thresholds do not transfer between datasets.",
  distributions: {
    bonafide: [
      { bin_start: 0.0, bin_end: 0.05, count: 0 },
      { bin_start: 0.05, bin_end: 0.1, count: 100 },
    ],
    spoof: [
      { bin_start: 0.1, bin_end: 0.15, count: 1 },
      { bin_start: 0.85, bin_end: 0.9, count: 30 },
      { bin_start: 0.9, bin_end: 0.95, count: 69 },
    ],
  },
  det_curve: [
    { threshold: 0.058764, false_acceptance_rate: 0.0, false_rejection_rate: 1.0 },
    { threshold: 0.142884, false_acceptance_rate: 0.0, false_rejection_rate: 0.0 },
    { threshold: 0.92563, false_acceptance_rate: 0.99, false_rejection_rate: 0.0 },
  ],
  operating_point: {
    threshold: 0.5,
    calibrated: false,
    false_acceptance_rate: 0.02,
    false_rejection_rate: 0.0,
  },
  per_attack: [
    { attack: "bonafide", count: 100, mean_score: 0.0693, is_spoof: false },
    { attack: "A07", count: 8, mean_score: 0.9197, is_spoof: true },
    { attack: "A19", count: 7, mean_score: 0.5945, is_spoof: true },
  ],
};

/** LA_E_1070252: genuine, and its score triples once the silence is trimmed. */
export const silenceProbe: SilenceProbeResult = {
  model: "xlsr-deepfake",
  model_label: "wav2vec2 XLS-R (Model A)",
  recording_id: "rec_1070252",
  threshold: 0.5,
  threshold_calibrated: false,
  silence_top_db: 30,
  min_non_speech_seconds: 0.5,
  duration: 3.42,
  speech_seconds: 2.29,
  non_speech_seconds: 1.13,
  non_speech_fraction: 0.33,
  speech_intervals: [[0.55, 2.84]],
  variants: {
    original: { applicable: true, seconds: 3.42, spoof_probability: 0.063, decision: "bonafide" },
    trimmed: { applicable: true, seconds: 2.29, spoof_probability: 0.428, decision: "bonafide" },
    non_speech: {
      applicable: true,
      seconds: 1.13,
      spoof_probability: 0.868,
      decision: "spoof",
    },
  },
  sample_rate: 16000,
  cached: false,
};

export const saliency: DeepfakeSaliency = {
  model: "xlsr-deepfake",
  model_label: "wav2vec2 XLS-R (Model A)",
  recording_id: "rec_1070252",
  method: "smoothgrad-x-input-margin",
  method_label: "SmoothGrad × input (|mean ∂margin/∂x · x|, 8 noisy passes)",
  target: "decision margin (spoof logit − bonafide logit)",
  segments: [
    { start_time: 0.0, end_time: 1.14, saliency: 1.0, intensity: 1.0 },
    { start_time: 1.14, end_time: 2.28, saliency: 0.31, intensity: 0.31 },
    { start_time: 2.28, end_time: 3.42, saliency: 0.12, intensity: 0.12 },
  ],
  series: [1.0, 0.31, 0.12],
  total_duration: 3.42,
  max_saliency_seconds: 12,
  analysis_window_seconds: 30,
  truncated: false,
  normalised: true,
  speech_intervals: [[0.55, 2.84]],
  silence_top_db: 30,
  saliency_in_speech_fraction: 0.24,
  cached: false,
};

export const projection: DeepfakeEmbeddingProjection = {
  model: "xlsr-deepfake",
  model_label: "wav2vec2 XLS-R (Model A)",
  reduction_method: "pca",
  reduction_method_used: "pca",
  n_components: 2,
  effective_components: 2,
  embedding_dimension: 1024,
  total_recordings: 3,
  threshold: 0.5,
  threshold_calibrated: false,
  recordings: [
    {
      recording_id: "rec_a",
      display_filename: "LA_E_1070252.flac",
      spoof_probability: 0.0633,
      decision: "bonafide",
    },
    {
      recording_id: "rec_b",
      display_filename: "LA_E_3986097.flac",
      spoof_probability: 0.142884,
      decision: "bonafide",
    },
    {
      recording_id: "rec_c",
      display_filename: "LA_E_9938640.flac",
      spoof_probability: 0.923,
      decision: "spoof",
    },
  ],
  coordinates: [
    [-4.2, 1.1],
    [-3.8, 0.4],
    [5.9, -1.6],
  ],
};

export const recordings = {
  dataset_id: "asvspoof2019-la",
  total_recordings: 2,
  recordings: [
    {
      recording_id: "rec_a",
      display_filename: "LA_E_1070252.flac",
      extension: ".flac",
      size_bytes: 108_544,
      duration_seconds: 3.42,
    },
    {
      recording_id: "rec_c",
      display_filename: "LA_E_9938640.flac",
      extension: ".flac",
      size_bytes: 96_128,
      duration_seconds: 3.01,
    },
  ],
};

/** Stub `fetch` with one JSON response, and hand back the mock to inspect. */
export const stubFetch = (
  body: unknown,
  init: { ok?: boolean; status?: number } = {},
) => {
  const fetchMock = vi.fn(async () => ({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    json: async () => body,
    arrayBuffer: async () => new ArrayBuffer(8),
  }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
