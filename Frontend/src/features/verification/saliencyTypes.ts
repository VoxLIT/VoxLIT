export type SaliencyReferenceType = "cluster" | "enrollment";

export type SaliencyAxis = "time" | "frequency";

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

export type AnySaliencyMapResponse = SaliencyMapResponse | FrequencySaliencyMapResponse;

export type SaliencyResultsByAxis = {
  time: SaliencyMapResponse | null;
  frequency: FrequencySaliencyMapResponse | null;
};

export const EMPTY_SALIENCY_RESULTS: SaliencyResultsByAxis = { time: null, frequency: null };

export const isFrequencySaliency = (
  result: AnySaliencyMapResponse | null
): result is FrequencySaliencyMapResponse => result?.occlusion_axis === "frequency";
