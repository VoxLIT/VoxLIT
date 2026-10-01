import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { SpeakerSaliencyMap } from "../SpeakerSaliencyMap";
import type { FrequencySaliencyMapResponse, SaliencyMapResponse } from "../saliencyTypes";

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

  describe("occlusion axis toggle", () => {
    const frequencyResult: FrequencySaliencyMapResponse = {
      model: "ecapa-tdnn",
      model_label: "ECAPA-TDNN",
      reference_type: "cluster",
      cluster_id: "Cluster 1",
      target_recording_id: "rec_target",
      reference_count: 2,
      baseline_similarity: 0.85,
      threshold: 0.36,
      occlusion_axis: "frequency",
      band_count: 4,
      audio_duration_seconds: 4.0,
      interpretation: "irrelevant for this test",
      bands: [
        {
          band_index: 1,
          low_hz: 50,
          high_hz: 421.6,
          label: "Pitch",
          occluded_similarity: 0.7,
          similarity_change: 0.15, // removing it lowered the score -> supports
          influence_strength: 0.15,
        },
        {
          band_index: 2,
          low_hz: 421.6,
          high_hz: 1319.2,
          label: "Vowel body",
          occluded_similarity: 0.9,
          similarity_change: -0.05, // opposes
          influence_strength: 0.05,
        },
        {
          band_index: 3,
          low_hz: 1319.2,
          high_hz: 3487.5,
          label: "Vowel shape",
          occluded_similarity: 0.848,
          similarity_change: 0.002, // below MIN_DISPLAY_INFLUENCE_DELTA -> minimal
          influence_strength: 0.002,
        },
        {
          band_index: 4,
          low_hz: 3487.5,
          high_hz: 8000,
          label: "Hiss sounds (s/sh)",
          occluded_similarity: 0.82,
          similarity_change: 0.03,
          influence_strength: 0.03,
        },
      ],
    };

    const renderMap = (overrides: Partial<React.ComponentProps<typeof SpeakerSaliencyMap>> = {}) =>
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
          onSegmentCountChange={vi.fn()}
          {...overrides}
        />
      );

    it("renders a Time | Frequency toggle that defaults to the time view", () => {
      renderMap();

      expect(screen.getByRole("radio", { name: "Time" })).toHaveAttribute("aria-checked", "true");
      expect(screen.getByRole("radio", { name: "Frequency" })).toHaveAttribute("aria-checked", "false");
      expect(screen.getByText("Segments: 6")).toBeInTheDocument();
      expect(screen.queryByText(/Each band was silenced once/)).not.toBeInTheDocument();
    });

    it("reports the chosen axis and hides the segment slider in the frequency view", () => {
      const onAxisChange = vi.fn();
      renderMap({ onOcclusionAxisChange: onAxisChange });

      fireEvent.click(screen.getByRole("radio", { name: "Frequency" }));

      expect(onAxisChange).toHaveBeenCalledWith("frequency");
      expect(screen.queryByText("Segments: 6")).not.toBeInTheDocument();
      expect(
        screen.getByText(
          "Each band was silenced once. A green bar means removing it lowered the match score, so it supported the match."
        )
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Generate Saliency" })).toBeInTheDocument();
    });

    it("renders one labelled bar per band, low to high, with the shared classification", () => {
      renderMap({ occlusionAxis: "frequency", result: frequencyResult });

      expect(screen.queryByTestId("mock-waveform-viewer")).not.toBeInTheDocument();
      expect(screen.queryByText("Most influential segments")).not.toBeInTheDocument();

      const rows = Array.from(screen.getByTestId("saliency-band-list").children);
      expect(rows.map((row) => row.querySelector(".font-medium")?.textContent)).toEqual([
        "Pitch",
        "Vowel body",
        "Vowel shape",
        "Hiss sounds (s/sh)",
      ]);
      expect(screen.getByText("50–422 Hz")).toBeInTheDocument();
      expect(screen.getByText("3488–8000 Hz")).toBeInTheDocument();

      const bar = (index: number) => screen.getByTestId(`saliency-band-bar-${index}`);
      expect(bar(1)).toHaveAttribute("data-classification", "supports");
      expect(bar(2)).toHaveAttribute("data-classification", "opposes");
      expect(bar(3)).toHaveAttribute("data-classification", "minimal");
      // Widths are relative to the strongest band.
      expect(bar(1).style.width).toBe("100%");
      expect(parseFloat(bar(4).style.width)).toBeCloseTo(20, 5);
    });

    it("does not render a time result in the frequency view or vice versa", () => {
      const { rerender } = renderMap({ occlusionAxis: "frequency", result: mockSaliencyResult });
      expect(screen.queryByTestId("mock-waveform-viewer")).not.toBeInTheDocument();
      expect(screen.queryByTestId("saliency-band-list")).not.toBeInTheDocument();

      rerender(
        <SpeakerSaliencyMap
          title="Cluster saliency"
          audioUrl="/test.wav"
          requireCredentials={false}
          result={frequencyResult}
          isLoading={false}
          error={null}
          staleReason={null}
          emptyStateMessage={null}
          onGenerate={vi.fn()}
          generateLabel="Generate Saliency"
          segmentCount={6}
          onSegmentCountChange={vi.fn()}
          occlusionAxis="time"
        />
      );
      expect(screen.queryByTestId("mock-waveform-viewer")).not.toBeInTheDocument();
      expect(screen.queryByTestId("saliency-band-list")).not.toBeInTheDocument();
    });
  });
});
