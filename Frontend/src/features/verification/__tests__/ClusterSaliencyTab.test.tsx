import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import { ClusterSaliencyTab } from "../ClusterSaliencyTab";
import type { BatchAnalysisResponse } from "../batchTypes";
import type { UploadedFile } from "@/tasks/types";

vi.mock("@/components/audio/WaveformViewer", () => ({
  WaveformViewer: ({ timelineBelow }: { timelineBelow?: React.ReactNode }) => (
    <div data-testid="mock-waveform-viewer">
      <div>Waveform</div>
      {timelineBelow}
    </div>
  ),
}));

describe("ClusterSaliencyTab", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  const mockFile1: UploadedFile = {
    file_id: "rec_001",
    filename: "clip1.wav",
    file_path: "datasets/demo/clip1.wav",
    message: "Selected from dataset",
  };

  const mockFile2: UploadedFile = {
    file_id: "rec_002",
    filename: "clip2.wav",
    file_path: "datasets/demo/clip2.wav",
    message: "Selected from dataset",
  };

  const mockSingletonFile: UploadedFile = {
    file_id: "rec_003",
    filename: "clip3.wav",
    file_path: "datasets/demo/clip3.wav",
    message: "Selected from dataset",
  };

  const mockNotInBatchFile: UploadedFile = {
    file_id: "rec_999",
    filename: "clip999.wav",
    file_path: "datasets/demo/clip999.wav",
    message: "Selected from dataset",
  };

  const mockBatchResult: BatchAnalysisResponse = {
    model: "ecapa-tdnn",
    model_label: "ECAPA-TDNN",
    threshold: 0.65,
    recording_count: 3,
    embedding_dimension: 192,
    labels: ["clip1.wav", "clip2.wav", "clip3.wav"],
    ground_truth_groups: ["spk1", "spk1", "spk2"],
    ground_truth_available: true,
    embeddings: [[0.1], [0.2], [0.3]],
    similarity_matrix: [
      [1.0, 0.88, 0.3],
      [0.88, 1.0, 0.25],
      [0.3, 0.25, 1.0],
    ],
    decision_matrix: [
      [true, true, false],
      [true, true, false],
      [false, false, true],
    ],
    clustering_distance_threshold: 0.65,
    clustering_threshold_version: "1.0",
    cluster_labels: ["Cluster 1", "Cluster 1", "Cluster 2"],
    cluster_fit_scores: [0.92, 0.92, 1.0],
    cluster_count: 2,
    cluster_summaries: [
      {
        cluster_id: "Cluster 1",
        member_count: 2,
        member_indices: [0, 1],
        member_labels: ["clip1.wav", "clip2.wav"],
        mean_intra_cluster_similarity: 0.88,
        min_intra_cluster_similarity: 0.88,
        representative_index: 0,
        representative_label: "clip1.wav",
        mean_fit_score: 0.92,
      },
      {
        cluster_id: "Cluster 2",
        member_count: 1,
        member_indices: [2],
        member_labels: ["clip3.wav"],
        mean_intra_cluster_similarity: 1.0,
        min_intra_cluster_similarity: 1.0,
        representative_index: 2,
        representative_label: "clip3.wav",
        mean_fit_score: 1.0,
      },
    ],
    recording_cluster_stats: [
      {
        cluster_id: "Cluster 1",
        mean_similarity_to_cluster: 0.88,
        min_similarity_to_cluster: 0.88,
        nearest_index: 1,
        nearest_label: "clip2.wav",
        nearest_similarity: 0.88,
        nearest_in_same_cluster: true,
      },
      {
        cluster_id: "Cluster 1",
        mean_similarity_to_cluster: 0.88,
        min_similarity_to_cluster: 0.88,
        nearest_index: 0,
        nearest_label: "clip1.wav",
        nearest_similarity: 0.88,
        nearest_in_same_cluster: true,
      },
      {
        cluster_id: "Cluster 2",
        mean_similarity_to_cluster: 1.0,
        min_similarity_to_cluster: 1.0,
        nearest_index: 0,
        nearest_label: "clip1.wav",
        nearest_similarity: 0.3,
        nearest_in_same_cluster: false,
      },
    ],
    evaluation_metrics: null,
    true_speaker_count: null,
  };

  const mockSubmittedIds = ["rec_001", "rec_002", "rec_003"];
  const mockColorMap = {
    "Cluster 1": "#3b82f6",
    "Cluster 2": "#10b981",
  };

  it("renders notice when batch analysis has not been run", () => {
    render(
      <ClusterSaliencyTab
        model="ecapa-tdnn"
        selectedFile={mockFile1}
        batchResult={null}
        submittedIds={[]}
        clusterColorMap={{}}
      />
    );

    expect(screen.getByText(/Batch analysis has not been run yet/i)).toBeInTheDocument();
    expect(screen.queryByText(/Cluster saliency —/i)).not.toBeInTheDocument();
  });

  it("renders prompt when no recording is selected", () => {
    render(
      <ClusterSaliencyTab
        model="ecapa-tdnn"
        selectedFile={null}
        batchResult={mockBatchResult}
        submittedIds={mockSubmittedIds}
        clusterColorMap={mockColorMap}
      />
    );

    expect(screen.getByText(/none selected — click a row in the Audio Dataset table/i)).toBeInTheDocument();
    expect(screen.queryByText(/Cluster saliency —/i)).not.toBeInTheDocument();
  });

  it("renders notice when selected file is not in batch", () => {
    render(
      <ClusterSaliencyTab
        model="ecapa-tdnn"
        selectedFile={mockNotInBatchFile}
        batchResult={mockBatchResult}
        submittedIds={mockSubmittedIds}
        clusterColorMap={mockColorMap}
      />
    );

    expect(
      screen.getByText(/The selected recording was not included in the last batch analysis run/i)
    ).toBeInTheDocument();
    expect(screen.queryByText(/Cluster saliency —/i)).not.toBeInTheDocument();
  });

  it("renders singleton cluster message when selected file is alone in cluster", () => {
    render(
      <ClusterSaliencyTab
        model="ecapa-tdnn"
        selectedFile={mockSingletonFile}
        batchResult={mockBatchResult}
        submittedIds={mockSubmittedIds}
        clusterColorMap={mockColorMap}
      />
    );

    expect(screen.getByText("Cluster saliency — clip3.wav")).toBeInTheDocument();
    expect(
      screen.getByText(/This recording's predicted cluster has no other members/i)
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Generate saliency map/i })).not.toBeInTheDocument();
  });

  it("renders interactive saliency map and calls API on generate", async () => {
    const mockSaliencyResponse = {
      model: "ecapa-tdnn",
      model_label: "ECAPA-TDNN",
      reference_type: "cluster",
      cluster_id: "Cluster 1",
      target_recording_id: "rec_001",
      reference_count: 1,
      baseline_similarity: 0.88,
      threshold: 0.65,
      segment_count: 8,
      audio_duration_seconds: 3.0,
      interpretation: "Temporal occlusion analysis against cluster centroid.",
      segments: [
        {
          segment_index: 1,
          start_seconds: 0.0,
          end_seconds: 0.375,
          occluded_similarity: 0.8,
          similarity_change: -0.08,
          influence_strength: 0.08,
        },
      ],
    };

    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => mockSaliencyResponse,
    } as Response);

    render(
      <ClusterSaliencyTab
        model="ecapa-tdnn"
        selectedFile={mockFile1}
        batchResult={mockBatchResult}
        submittedIds={mockSubmittedIds}
        clusterColorMap={mockColorMap}
      />
    );

    expect(screen.getByText("Cluster saliency — clip1.wav")).toBeInTheDocument();
    const generateBtn = screen.getByRole("button", { name: /Generate saliency map/i });
    expect(generateBtn).toBeInTheDocument();

    fireEvent.click(generateBtn);

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledTimes(1);
    });

    const [calledUrl, calledOptions] = fetchSpy.mock.calls[0];
    expect(calledUrl).toContain("/tasks/verification/explain/saliency");
    expect(calledOptions?.method).toBe("POST");

    // Form data validation
    const body = calledOptions?.body as FormData;
    expect(body.get("model")).toBe("ecapa-tdnn");
    expect(body.get("reference_type")).toBe("cluster");
    expect(body.get("target_recording_id")).toBe("rec_001");
    expect(body.get("cluster_id")).toBe("Cluster 1");
    expect(body.getAll("reference_recording_ids")).toEqual(["rec_002"]);

    // After success, waveform viewer should be present
    await waitFor(() => {
      expect(screen.getByTestId("mock-waveform-viewer")).toBeInTheDocument();
    });
  });
});
