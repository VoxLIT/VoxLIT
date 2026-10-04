export type SaliencyReferenceType = "cluster" | "enrollment";

// "integrated_gradients" is a saliency method rather than an occlusion axis,
// but it is one more view of the same toggle and is stored the same way.
export type SaliencyAxis = "time" | "frequency" | "integrated_gradients";

export const DEFAULT_SALIENCY_BAND_COUNT = 8;

export interface SaliencySegment {
  segment_index: number;
  start_seconds: number;
  end_seconds: number;
  occluded_similarity: number;
  similarity_change: number;
  influence_strength: number;
}

export interface SaliencyBand {
  band_index: number;
  low_hz: number;
  high_hz: number;
  label: string;
  occluded_similarity: number;
  similarity_change: number;
  influence_strength: number;
}

interface SaliencyMapBase {
  model: string;
  model_label: string;
  reference_type: SaliencyReferenceType;
  cluster_id: string | null;
  target_recording_id: string | null;
  reference_count: number;
  baseline_similarity: number;
  threshold: number;
  audio_duration_seconds: number;
  interpretation: string;
}

// Time-axis responses carry no `occlusion_axis` key.
export interface SaliencyMapResponse extends SaliencyMapBase {
  occlusion_axis?: "time";
  segment_count: number;
  segments: SaliencySegment[];
}

export interface FrequencySaliencyMapResponse extends SaliencyMapBase {
  occlusion_axis: "frequency";
  band_count: number;
  bands: SaliencyBand[];
}

export interface IntegratedGradientsBand {
  band_index: number;
  low_hz: number;
  high_hz: number;
  label: string;
  total_attribution: number;
}

export interface IntegratedGradientsSaliencyResponse extends SaliencyMapBase {
  saliency_method: "integrated_gradients";
  n_steps: number;
  /** Similarity of the all-silent baseline clip to the reference centroid. */
  baseline_input_similarity: number;
  /** Rows are mel bands (low to high); each row runs over time. */
  attributions: number[][];
  time_edges_seconds: number[];
  mel_edges_hz: number[];
  time_totals: number[];
  /** Same mel bands as the frequency-occlusion view. */
  bands: IntegratedGradientsBand[];
  outside_bands_total: number;
  convergence_delta: number;
  total_attribution: number;
  expected_total: number;
  completeness_ok: boolean;
}

export type AnySaliencyMapResponse =
  | SaliencyMapResponse
  | FrequencySaliencyMapResponse
  | IntegratedGradientsSaliencyResponse;

export type SaliencyResultsByAxis = {
  time: SaliencyMapResponse | null;
  frequency: FrequencySaliencyMapResponse | null;
  integrated_gradients: IntegratedGradientsSaliencyResponse | null;
};

export const EMPTY_SALIENCY_RESULTS: SaliencyResultsByAxis = {
  time: null,
  frequency: null,
  integrated_gradients: null,
};

export const isFrequencySaliency = (
  result: AnySaliencyMapResponse | null
): result is FrequencySaliencyMapResponse =>
  !!result && "occlusion_axis" in result && result.occlusion_axis === "frequency";

export const isIntegratedGradientsSaliency = (
  result: AnySaliencyMapResponse | null
): result is IntegratedGradientsSaliencyResponse =>
  !!result && "saliency_method" in result && result.saliency_method === "integrated_gradients";

/** Backend 422 detail when a model has no differentiable feature path. */
export const isIntegratedGradientsUnsupportedError = (error: string | null): boolean =>
  !!error && error.includes("Integrated Gradients is only available");
