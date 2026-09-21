import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { ClusterAssignmentResults } from "../ClusterAssignmentResults";
import { clusterAssignmentStore, type ClusterAssignmentSnapshot } from "../clusterAssignmentStore";
import type { AudioFile } from "@/tasks/types";

describe("ClusterAssignmentResults", () => {
  beforeEach(() => {
    clusterAssignmentStore.publish(null);
  });

  const mockFile: AudioFile = {
    file_id: "rec_001",
    filename: "clip1.wav",
    label: "Speaker 1 Clip 1",
    duration: 3.5,
    sample_rate: 16000,
  };

  const mockSnapshot: ClusterAssignmentSnapshot = {
    fileId: "rec_001",
    stats: {
      recording_id: "rec_001",
      label: "Speaker 1 Clip 1",
      cluster_id: "Cluster 1",
      silhouette_score: 0.85,
      mean_similarity_to_cluster: 0.92,
      min_similarity_to_cluster: 0.88,
      nearest_recording_id: "rec_002",
      nearest_label: "Speaker 1 Clip 2",
      nearest_similarity: 0.94,
      nearest_in_same_cluster: true,
      cluster_fit_margin: 0.15,
      intra_cluster_spread: 0.05,
    },
    clusterSize: 3,
    modelLabel: "ECAPA-TDNN",
    clusteringDistanceThreshold: 0.65,
    clusteringThresholdVersion: "1.0",
    groundTruthGroup: "spk_1",
    groundTruthAvailable: true,
  };

  it("renders prompt when no file is selected", () => {
    render(<ClusterAssignmentResults selectedFile={null} />);
    expect(
      screen.getByText("Select a recording to see its cluster assignment.")
    ).toBeInTheDocument();
  });

  it("renders batch run notice when no batch snapshot is published", () => {
    render(<ClusterAssignmentResults selectedFile={mockFile} />);
    expect(
      screen.getByText("Run a batch analysis to see cluster assignment results.")
    ).toBeInTheDocument();
  });

  it("renders mismatch notice when selected file does not match batch snapshot", () => {
    clusterAssignmentStore.publish({
      ...mockSnapshot,
      fileId: "other_file_id",
    });

    render(<ClusterAssignmentResults selectedFile={mockFile} />);
    expect(
      screen.getByText("This recording is not part of the current batch results.")
    ).toBeInTheDocument();
  });

  it("renders full cluster assignment details when snapshot matches selected file", () => {
    clusterAssignmentStore.publish(mockSnapshot);

    render(<ClusterAssignmentResults selectedFile={mockFile} />);

    // Header
    expect(screen.getByText("Cluster assignment results")).toBeInTheDocument();

    // Predicted cluster
    expect(screen.getByText("Cluster 1")).toBeInTheDocument();

    // Ground truth group is not rendered
    expect(screen.queryByText("Ground-truth speaker group")).not.toBeInTheDocument();
    expect(screen.queryByText("spk_1")).not.toBeInTheDocument();

    // Cluster size
    expect(screen.getByText("3 recordings")).toBeInTheDocument();

    // Average similarity
    expect(screen.getByText("0.9200")).toBeInTheDocument();

    // Nearest neighbour
    expect(screen.getByText("Speaker 1 Clip 2")).toBeInTheDocument();
    expect(screen.getByText("0.9400")).toBeInTheDocument();

    // Model & threshold
    expect(screen.getByText("ECAPA-TDNN")).toBeInTheDocument();
    expect(screen.getByText("0.6500")).toBeInTheDocument();
  });

  it("displays 'different cluster' badge when nearest neighbour is in a different cluster", () => {
    clusterAssignmentStore.publish({
      ...mockSnapshot,
      stats: {
        ...mockSnapshot.stats,
        nearest_in_same_cluster: false,
      },
    });

    render(<ClusterAssignmentResults selectedFile={mockFile} />);
    expect(screen.getByText("different cluster")).toBeInTheDocument();
  });
});
