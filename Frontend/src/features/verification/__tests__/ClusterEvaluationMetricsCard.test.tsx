import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { ClusterEvaluationMetricsCard } from "../ClusterEvaluationMetricsCard";
import type { EvaluationMetrics } from "../batchTypes";

const mockMetrics: EvaluationMetrics = {
  adjusted_rand_index: 0.8524,
  normalized_mutual_information: 0.7812,
  cluster_purity: 0.9145,
  pairwise_precision: 0.885,
  pairwise_recall: 0.823,
  pairwise_f1_score: 0.8529,
  pairwise_accuracy: 0.9312,
  total_unique_pairs: 1000,
  true_positive_pairs: 250,
  true_negative_pairs: 681,
  false_positive_pairs: 32,
  false_negative_pairs: 37,
};

describe("ClusterEvaluationMetricsCard", () => {
  it("renders notice when ground truth is unavailable or metrics are null", () => {
    const { rerender } = render(
      <ClusterEvaluationMetricsCard
        evaluationMetrics={null}
        groundTruthAvailable={false}
        predictedClusterCount={4}
        trueSpeakerCount={null}
      />
    );

    expect(
      screen.getByText(/Ground-truth speaker groups were not provided for this batch/i)
    ).toBeInTheDocument();

    rerender(
      <ClusterEvaluationMetricsCard
        evaluationMetrics={null}
        groundTruthAvailable={true}
        predictedClusterCount={4}
        trueSpeakerCount={5}
      />
    );

    expect(
      screen.getByText(/Ground-truth speaker groups were not provided for this batch/i)
    ).toBeInTheDocument();
  });

  it("renders partition agreement metrics formatted to 4 decimals when available", () => {
    render(
      <ClusterEvaluationMetricsCard
        evaluationMetrics={mockMetrics}
        groundTruthAvailable={true}
        predictedClusterCount={4}
        trueSpeakerCount={4}
      />
    );

    expect(screen.getByText("ARI: 0.8524")).toBeInTheDocument();
    expect(screen.getByText("NMI: 0.7812")).toBeInTheDocument();
    expect(screen.getByText("Purity: 0.9145")).toBeInTheDocument();
  });

  it("renders pairwise rates and accuracy", () => {
    render(
      <ClusterEvaluationMetricsCard
        evaluationMetrics={mockMetrics}
        groundTruthAvailable={true}
        predictedClusterCount={4}
        trueSpeakerCount={4}
      />
    );

    expect(screen.getByText("Precision: 0.8850")).toBeInTheDocument();
    expect(screen.getByText("Recall: 0.8230")).toBeInTheDocument();
    expect(screen.getByText("F1 score: 0.8529")).toBeInTheDocument();
    expect(screen.getByText("Accuracy: 0.9312")).toBeInTheDocument();
  });

  it("renders formatted pair counts and speaker counts", () => {
    render(
      <ClusterEvaluationMetricsCard
        evaluationMetrics={mockMetrics}
        groundTruthAvailable={true}
        predictedClusterCount={5}
        trueSpeakerCount={4}
      />
    );

    expect(screen.getByText("Total: 1,000")).toBeInTheDocument();
    expect(screen.getByText("True positive: 250")).toBeInTheDocument();
    expect(screen.getByText("True speaker groups: 4")).toBeInTheDocument();
    expect(screen.getByText("Predicted clusters: 5")).toBeInTheDocument();
  });
});
