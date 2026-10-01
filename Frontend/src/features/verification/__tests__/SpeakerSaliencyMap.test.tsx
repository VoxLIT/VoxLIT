import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { SpeakerSaliencyMap } from "../SpeakerSaliencyMap";
import type {
  FrequencySaliencyMapResponse,
  IntegratedGradientsSaliencyResponse,
  SaliencyMapResponse,
} from "../saliencyTypes";

// Plotly needs a real layout engine; what matters here is the data handed to it.
const plotProps = vi.hoisted(() => [] as { data: Record<string, any>[]; layout: Record<string, any> }[]);
vi.mock("react-plotly.js", () => ({
  default: (props: { data: Record<string, any>[]; layout: Record<string, any> }) => {
    plotProps.push(props);
    return <div data-testid="plot" />;
  },
}));

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
          label: "Band 1",
          occluded_similarity: 0.7,
          similarity_change: 0.15, // removing it lowered the score -> supports
          influence_strength: 0.15,
        },
        {
          band_index: 2,
          low_hz: 421.6,
          high_hz: 1319.2,
          label: "Band 2",
          occluded_similarity: 0.9,
          similarity_change: -0.05, // opposes
          influence_strength: 0.05,
        },
        {
          band_index: 3,
          low_hz: 1319.2,
          high_hz: 3487.5,
          label: "Band 3",
          occluded_similarity: 0.848,
          similarity_change: 0.002, // below MIN_DISPLAY_INFLUENCE_DELTA -> minimal
          influence_strength: 0.002,
        },
        {
          band_index: 4,
          low_hz: 3487.5,
          high_hz: 8000,
          label: "Band 4",
          occluded_similarity: 0.82,
          similarity_change: 0.03,
          influence_strength: 0.03,
        },
      ],
    };

    const gradientBandResult: IntegratedGradientsSaliencyResponse = {
      model: "ecapa-tdnn",
      model_label: "ECAPA-TDNN",
      reference_type: "cluster",
      cluster_id: "Cluster 1",
      target_recording_id: "rec_target",
      reference_count: 2,
      baseline_similarity: 0.6,
      threshold: 0.36,
      audio_duration_seconds: 1.0,
      interpretation: "irrelevant for this test",
      saliency_method: "integrated_gradients",
      n_steps: 32,
      baseline_input_similarity: 0,
      attributions: [[0.6]],
      time_edges_seconds: [0, 1],
      mel_edges_hz: [50, 8000],
      time_totals: [0.6],
      bands: [{ band_index: 1, low_hz: 50, high_hz: 8000, label: "Band 1", total_attribution: 0.6 }],
      outside_bands_total: 0,
      convergence_delta: 0,
      total_attribution: 0.6,
      expected_total: 0.6,
      completeness_ok: true,
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
        "Band 1 · 50–422 Hz",
        "Band 2 · 422–1319 Hz",
        "Band 3 · 1319–3488 Hz",
        "Band 4 · 3488–8000 Hz",
      ]);
      expect(screen.getByTestId("saliency-band-list")).not.toHaveTextContent(/pitch|vowel|hiss|clarity/i);

      const bar = (index: number) => screen.getByTestId(`saliency-band-bar-${index}`);
      expect(bar(1)).toHaveAttribute("data-classification", "supports");
      expect(bar(2)).toHaveAttribute("data-classification", "opposes");
      expect(bar(3)).toHaveAttribute("data-classification", "minimal");
      // Widths are relative to the strongest band.
      expect(bar(1).style.width).toBe("100%");
      expect(parseFloat(bar(4).style.width)).toBeCloseTo(20, 5);
    });

    it.each([
      ["frequency", "Similarity change per frequency band"],
      ["integrated_gradients", "Total per frequency band"],
    ] as const)("%s view: offers the approximate speech-content guide next to its band title", async (axis, title) => {
      renderMap({ occlusionAxis: axis, result: axis === "frequency" ? frequencyResult : gradientBandResult });

      expect(screen.getByText(title)).toBeInTheDocument();
      fireEvent.focus(screen.getByLabelText("About Typical speech content by frequency (approximate)"));

      const tooltip = await screen.findByRole("tooltip");
      expect(tooltip).toHaveTextContent(
        "Bands are mel-spaced and carry no fixed meaning. As rough, overlapping guides from the speech literature: adult F0 commonly lies around 85–255 Hz (Titze, 1994); the first formant typically around 250–1000 Hz and the second around 800–2500 Hz (Peterson & Barney, 1952); sibilant fricatives such as /s/ and /ʃ/ concentrate energy above ~4 kHz (Jongman et al., 2000). Speaker-specific information has been reported in both low and ~4–5 kHz regions (Lu & Dang, 2008). Ranges vary by speaker, sex and vowel."
      );
    });

    it("shows band changes with an explicit sign and three decimals", () => {
      renderMap({ occlusionAxis: "frequency", result: frequencyResult });

      const list = screen.getByTestId("saliency-band-list");
      expect(list).toHaveTextContent("+0.150");
      expect(list).toHaveTextContent("-0.050");
    });

    describe("baseline similarity card", () => {
      const clusterSummary =
        "How similar this clip is to the average voice of the other 2 clips in this speaker group (this clip excluded).";
      const enrollmentSummary =
        "How similar this clip is to the reference speaker's average voice, before anything is silenced.";
      const timeResult: SaliencyMapResponse = {
        model: "ecapa-tdnn",
        model_label: "ECAPA-TDNN",
        reference_type: "enrollment",
        cluster_id: null,
        target_recording_id: null,
        reference_count: 3,
        baseline_similarity: 0.85,
        threshold: 0.36,
        segment_count: 1,
        audio_duration_seconds: 1.0,
        interpretation: "irrelevant for this test",
        segments: [
          {
            segment_index: 1,
            start_seconds: 0,
            end_seconds: 1,
            occluded_similarity: 0.8,
            similarity_change: 0.05,
            influence_strength: 0.05,
          },
        ],
      };

      it("cluster mode: describes the group and never compares against the pair threshold", () => {
        renderMap({ occlusionAxis: "frequency", result: frequencyResult });

        const card = screen.getByTestId("baseline-similarity-card");
        expect(card).toHaveTextContent("Baseline similarity");
        expect(card).toHaveTextContent("0.8500");
        expect(card).toHaveTextContent(clusterSummary);
        expect(card).not.toHaveTextContent(/threshold/i);
        expect(card).not.toHaveTextContent(enrollmentSummary);
      });

      it("enrollment mode: shows the summary and an above-threshold verdict (time view)", () => {
        renderMap({ result: timeResult });

        const card = screen.getByTestId("baseline-similarity-card");
        expect(card).toHaveTextContent("0.8500");
        expect(card).toHaveTextContent(enrollmentSummary);
        expect(card).toHaveTextContent("Above threshold 0.36 → same speaker");
        expect(card).not.toHaveTextContent("speaker group");
      });

      it("enrollment mode: shows a below-threshold verdict (frequency view)", () => {
        renderMap({
          occlusionAxis: "frequency",
          result: { ...frequencyResult, reference_type: "enrollment", cluster_id: null, baseline_similarity: 0.2 },
        });

        const card = screen.getByTestId("baseline-similarity-card");
        expect(card).toHaveTextContent("0.2000");
        expect(card).toHaveTextContent(enrollmentSummary);
        expect(card).toHaveTextContent("Below threshold 0.36 → different speakers");
      });

      it("explains the number in its tooltip", async () => {
        renderMap({ occlusionAxis: "frequency", result: frequencyResult });

        fireEvent.focus(screen.getByLabelText("About Baseline similarity"));

        const tooltip = await screen.findByRole("tooltip");
        expect(tooltip).toHaveTextContent(
          "The model turns each reference clip into a voice fingerprint and averages them into one 'centroid'. This number is the cosine similarity between this clip and that centroid."
        );
        expect(tooltip).toHaveTextContent(
          "Each bar below shows how much this number drops when that part is silenced."
        );
      });
    });

    describe("Gradient (IG) view", () => {
      const helpText =
        "Integrated Gradients traces how the score changes as the clip fades in from silence. Green regions pushed the clip towards the reference voice; red pushed it away.";
      const gradientResult: IntegratedGradientsSaliencyResponse = {
        model: "ecapa-tdnn",
        model_label: "ECAPA-TDNN",
        reference_type: "cluster",
        cluster_id: "Cluster 1",
        target_recording_id: "rec_target",
        reference_count: 2,
        baseline_similarity: 0.6,
        threshold: 0.36,
        audio_duration_seconds: 3.0,
        interpretation: "irrelevant for this test",
        saliency_method: "integrated_gradients",
        n_steps: 32,
        baseline_input_similarity: -0.05,
        // 2 mel rows x 3 time columns.
        attributions: [
          [0.2, -0.1, 0.0],
          [0.3, 0.05, -0.4],
        ],
        time_edges_seconds: [0, 1, 2, 3],
        mel_edges_hz: [50, 1000, 8000],
        time_totals: [0.5, -0.05, -0.4],
        bands: [
          { band_index: 1, low_hz: 50, high_hz: 421.6, label: "Band 1", total_attribution: 0.4 },
          { band_index: 2, low_hz: 421.6, high_hz: 8000, label: "Band 2", total_attribution: -0.1 },
        ],
        outside_bands_total: 0,
        convergence_delta: 0.0,
        total_attribution: 0.65,
        expected_total: 0.65,
        completeness_ok: true,
      };

      it("offers Time | Frequency | Gradient (IG)", () => {
        renderMap();

        expect(screen.getAllByRole("radio").map((radio) => radio.textContent)).toEqual([
          "Time",
          "Frequency",
          "Gradient (IG)",
        ]);
      });

      it("reports the chosen view, hides the segment slider, and shows the help text", () => {
        const onAxisChange = vi.fn();
        renderMap({ onOcclusionAxisChange: onAxisChange });
        expect(screen.queryByText(helpText)).not.toBeInTheDocument();

        fireEvent.click(screen.getByRole("radio", { name: "Gradient (IG)" }));

        expect(onAxisChange).toHaveBeenCalledWith("integrated_gradients");
        expect(screen.queryByText("Segments: 6")).not.toBeInTheDocument();
        expect(screen.getByText(helpText)).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Generate Saliency" })).toBeInTheDocument();
      });

      it("renders a diverging heatmap symmetric around zero", () => {
        plotProps.length = 0;
        renderMap({ occlusionAxis: "integrated_gradients", result: gradientResult });

        expect(screen.getByTestId("ig-heatmap")).toBeInTheDocument();
        expect(screen.getByTestId("plot")).toBeInTheDocument();
        expect(screen.queryByTestId("mock-waveform-viewer")).not.toBeInTheDocument();
        expect(screen.queryByTestId("saliency-band-list")).not.toBeInTheDocument();

        const { data, layout } = plotProps[plotProps.length - 1];
        const trace = data[0];
        expect(trace.type).toBe("heatmap");
        expect(trace.z).toEqual(gradientResult.attributions);
        expect(trace.x).toEqual([0.5, 1.5, 2.5]);
        expect(trace.zmin).toBeCloseTo(-0.4);
        expect(trace.zmax).toBeCloseTo(0.4);
        expect(trace.zmid).toBe(0);
        // red (opposes) -> white (neutral) -> green (supports)
        expect(trace.colorscale.map(([, color]: [number, string]) => color)).toEqual([
          "#f43f5e",
          "#ffffff",
          "#10b981",
        ]);
        expect(layout.xaxis.title.text).toBe("Time (s)");
        expect(layout.yaxis.title.text).toBe("Frequency (Hz)");
        expect(layout.yaxis.ticktext).toEqual(["525", "4500"]);
      });

      it("renders per-time and per-band total strips coloured by direction", () => {
        renderMap({ occlusionAxis: "integrated_gradients", result: gradientResult });

        expect(screen.getByTestId("ig-time-strip")).toBeInTheDocument();
        const timeBar = (index: number) => screen.getByTestId(`ig-time-bar-${index}`);
        expect(timeBar(0)).toHaveAttribute("data-direction", "supports");
        expect(timeBar(0).style.height).toBe("50%");
        expect(timeBar(1)).toHaveAttribute("data-direction", "opposes");
        expect(timeBar(2)).toHaveAttribute("data-direction", "opposes");
        expect(parseFloat(timeBar(2).style.height)).toBeCloseTo(40, 5);

        const rows = Array.from(screen.getByTestId("ig-band-list").children);
        expect(rows.map((row) => row.querySelector(".font-medium")?.textContent)).toEqual([
          "Band 1 · 50–422 Hz",
          "Band 2 · 422–8000 Hz",
        ]);
        expect(screen.getByTestId("ig-band-bar-1")).toHaveAttribute("data-direction", "supports");
        expect(screen.getByTestId("ig-band-bar-1").style.width).toBe("100%");
        expect(screen.getByTestId("ig-band-bar-2")).toHaveAttribute("data-direction", "opposes");
        expect(parseFloat(screen.getByTestId("ig-band-bar-2").style.width)).toBeCloseTo(25, 5);
        expect(screen.getByTestId("ig-band-list")).toHaveTextContent("+0.400");
        expect(screen.getByTestId("ig-band-list")).toHaveTextContent("-0.100");
      });

      it("shows a positive completeness badge when the attributions add up", () => {
        renderMap({ occlusionAxis: "integrated_gradients", result: gradientResult });

        const badge = screen.getByTestId("ig-completeness-badge");
        expect(badge).toHaveAttribute("data-completeness", "ok");
        expect(badge).toHaveTextContent("Attributions add up to the score change");
        expect(badge).not.toHaveTextContent("Approximation is rough");
      });

      it("warns when the approximation is rough", () => {
        renderMap({
          occlusionAxis: "integrated_gradients",
          result: { ...gradientResult, completeness_ok: false, total_attribution: 0.5, convergence_delta: -0.15 },
        });

        const badge = screen.getByTestId("ig-completeness-badge");
        expect(badge).toHaveAttribute("data-completeness", "rough");
        expect(badge).toHaveTextContent("Approximation is rough (n_steps too low)");
        expect(badge).not.toHaveTextContent("Attributions add up");
      });

      it("shows the baseline card with a gradient-specific tooltip", async () => {
        renderMap({ occlusionAxis: "integrated_gradients", result: gradientResult });

        expect(screen.getByTestId("baseline-similarity-card")).toHaveTextContent("0.6000");
        fireEvent.focus(screen.getByLabelText("About Baseline similarity"));
        const tooltip = await screen.findByRole("tooltip");
        expect(tooltip).toHaveTextContent("The map below splits the gap between this number and a silent clip's score");
        expect(tooltip).not.toHaveTextContent("silenced");
      });

      it("shows the ECAPA-only 422 as a notice, not a failure", () => {
        const message = "Integrated Gradients is only available for ECAPA-TDNN";
        const { unmount } = renderMap({ occlusionAxis: "integrated_gradients", error: message });

        expect(screen.getByTestId("ig-unsupported-notice")).toHaveTextContent(message);
        expect(screen.queryByText("Saliency map could not be generated")).not.toBeInTheDocument();
        expect(screen.queryByTestId("ig-heatmap")).not.toBeInTheDocument();
        unmount();

        // The limitation says nothing about the occlusion views.
        renderMap({ occlusionAxis: "time", error: message });
        expect(screen.queryByTestId("ig-unsupported-notice")).not.toBeInTheDocument();
        expect(screen.queryByText("Saliency map could not be generated")).not.toBeInTheDocument();
      });

      it("still shows other gradient errors as failures", () => {
        renderMap({ occlusionAxis: "integrated_gradients", error: "Audio file could not be decoded" });

        expect(screen.getByText("Saliency map could not be generated")).toBeInTheDocument();
        expect(screen.queryByTestId("ig-unsupported-notice")).not.toBeInTheDocument();
      });

      it("labels the loading state and never renders a gradient result in another view", () => {
        const { unmount } = renderMap({ occlusionAxis: "integrated_gradients", isLoading: true });
        expect(screen.getByText("Computing gradients…")).toBeInTheDocument();
        unmount();

        renderMap({ occlusionAxis: "time", result: gradientResult });
        renderMap({ occlusionAxis: "frequency", result: gradientResult });
        expect(screen.queryByTestId("ig-heatmap")).not.toBeInTheDocument();
        expect(screen.queryByTestId("mock-waveform-viewer")).not.toBeInTheDocument();
      });
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
