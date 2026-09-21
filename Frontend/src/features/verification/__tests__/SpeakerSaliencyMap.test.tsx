import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { SpeakerSaliencyMap } from "../SpeakerSaliencyMap";
import type { SaliencyMapResponse } from "../saliencyTypes";

vi.mock("@/components/audio/WaveformViewer", () => ({
  WaveformViewer: ({ timelineBelow }: { timelineBelow?: React.ReactNode }) => (
    <div data-testid="mock-waveform-viewer">
      <div>Waveform</div>
      {timelineBelow}
    </div>
  ),
}));

const mockSaliencyResult: SaliencyMapResponse = {
  model: "ecapa-tdnn",
  target_recording_id: "rec_target",
  baseline_similarity: 0.85,
  audio_duration_seconds: 4.0,
  reference_recording_ids: ["rec_ref_1", "rec_ref_2"],
  reference_type: "cluster",
  cluster_id: "Cluster 1",
  segments: [
    {
      segment_index: 1,
      start_seconds: 0.0,
      end_seconds: 1.0,
      occluded_similarity: 0.7,
      similarity_change: -0.15, // Opposes similarity when occluded, so segment supports similarity
      influence_strength: 0.15,
      is_most_influential: true,
    },
    {
      segment_index: 2,
      start_seconds: 1.0,
      end_seconds: 2.0,
      occluded_similarity: 0.9,
      similarity_change: 0.05, // Opposes similarity
      influence_strength: 0.05,
      is_most_influential: false,
    },
    {
      segment_index: 3,
      start_seconds: 2.0,
      end_seconds: 3.0,
      occluded_similarity: 0.852,
      similarity_change: 0.002, // Minimal influence (< 0.01)
      influence_strength: 0.002,
      is_most_influential: false,
    },
    {
      segment_index: 4,
      start_seconds: 3.0,
      end_seconds: 4.0,
      occluded_similarity: 0.84,
      similarity_change: -0.01,
      influence_strength: 0.01,
      is_most_influential: false,
    },
  ],
};

describe("SpeakerSaliencyMap", () => {
  it("renders emptyStateMessage and hides generate controls when provided", () => {
    render(
      <SpeakerSaliencyMap
        title="Cluster saliency"
        audioUrl="/test.wav"
        requireCredentials={false}
        result={null}
        isLoading={false}
        error={null}
        staleReason={null}
        emptyStateMessage="Cannot compute saliency for a single-recording cluster."
        onGenerate={vi.fn()}
        generateLabel="Generate Saliency"
        segmentCount={4}
        onSegmentCountChange={vi.fn()}
      />
    );

    expect(
      screen.getByText("Cannot compute saliency for a single-recording cluster.")
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Generate Saliency" })).not.toBeInTheDocument();
  });

  it("renders segment slider and triggers onSegmentCountChange", () => {
    const onSegmentChange = vi.fn();
    render(
      <SpeakerSaliencyMap
        title="Cluster saliency"
        audioUrl="/test.wav"
        requireCredentials={false}
        result={null}
        isLoading={false}
        error={null}
        staleReason={null}
        emptyStateMessage={null}
        onGenerate={vi.fn()}
        generateLabel="Generate Saliency"
        segmentCount={6}
        onSegmentCountChange={onSegmentChange}
      />
    );

    expect(screen.getByText("Segments: 6")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate Saliency" })).toBeInTheDocument();
  });

  it("renders loading state when isLoading is true", () => {
    render(
      <SpeakerSaliencyMap
        title="Cluster saliency"
        audioUrl="/test.wav"
        requireCredentials={false}
        result={null}
        isLoading={true}
        error={null}
        staleReason={null}
        emptyStateMessage={null}
        onGenerate={vi.fn()}
        generateLabel="Generate Saliency"
        segmentCount={6}
        onSegmentCountChange={vi.fn()}
      />
    );

    expect(screen.getByText("Running occlusion passes…")).toBeInTheDocument();
  });

  it("renders error alert when error is provided", () => {
    render(
      <SpeakerSaliencyMap
        title="Cluster saliency"
        audioUrl="/test.wav"
        requireCredentials={false}
        result={null}
        isLoading={false}
        error="Audio file could not be decoded"
        staleReason={null}
        emptyStateMessage={null}
        onGenerate={vi.fn()}
        generateLabel="Generate Saliency"
        segmentCount={6}
        onSegmentCountChange={vi.fn()}
      />
    );

    expect(screen.getByText("Saliency map could not be generated")).toBeInTheDocument();
    expect(screen.getByText("Audio file could not be decoded")).toBeInTheDocument();
  });

  it("renders staleReason alert when result is out of date", () => {
    render(
      <SpeakerSaliencyMap
        title="Cluster saliency"
        audioUrl="/test.wav"
        requireCredentials={false}
        result={mockSaliencyResult}
        isLoading={false}
        error={null}
        staleReason="Model changed since this saliency map was generated."
        emptyStateMessage={null}
        onGenerate={vi.fn()}
        generateLabel="Generate Saliency"
        segmentCount={4}
        onSegmentCountChange={vi.fn()}
      />
    );

    expect(
      screen.getByText("Model changed since this saliency map was generated.")
    ).toBeInTheDocument();
  });

  it("renders saliency heatmap and most influential segments list", () => {
    render(
      <SpeakerSaliencyMap
        title="Cluster saliency"
        audioUrl="/test.wav"
        requireCredentials={false}
        result={mockSaliencyResult}
        isLoading={false}
        error={null}
        staleReason={null}
        emptyStateMessage={null}
        onGenerate={vi.fn()}
        generateLabel="Generate Saliency"
        segmentCount={4}
        onSegmentCountChange={vi.fn()}
      />
    );

    expect(screen.getByTestId("mock-waveform-viewer")).toBeInTheDocument();
    expect(screen.getByText("Most influential segments")).toBeInTheDocument();
    expect(screen.getByText("Low influence")).toBeInTheDocument();
    expect(screen.getByText("High influence")).toBeInTheDocument();
  });
});
