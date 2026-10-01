import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";

// Plotly needs a real layout engine; what matters here is the data handed to it.
const plotProps: { data: Record<string, any>[]; layout: Record<string, any> }[] = [];
vi.mock("react-plotly.js", () => ({
  default: (props: { data: Record<string, any>[]; layout: Record<string, any> }) => {
    plotProps.push(props);
    return <div data-testid="plot" />;
  },
}));

const { PerturbationSweepCard } = await import("../PerturbationSweepCard");
type SweepResponse = import("../PerturbationSweepCard").PerturbationSweepResponse;

const pitchResult: SweepResponse = {
  model: "ecapa-tdnn",
  model_label: "ECAPA-TDNN",
  threshold: 0.4,
  perturbation_type: "pitch_shift",
  source_recording_id: "rec_001",
  summary: "For ECAPA-TDNN, shifting pitch down flips the decision; shifting pitch up never flips the decision.",
  points: [
    { strength: -6, direction: "down", status: "ok", reason: null, similarity: 0.1, same_speaker: false, snr_db: null },
    { strength: -4, direction: "down", status: "ok", reason: null, similarity: 0.2, same_speaker: false, snr_db: null },
    { strength: -2, direction: "down", status: "ok", reason: null, similarity: 0.6, same_speaker: true, snr_db: null },
    { strength: -1, direction: "down", status: "ok", reason: null, similarity: 0.8, same_speaker: true, snr_db: null },
    { strength: 1, direction: "up", status: "ok", reason: null, similarity: 0.8, same_speaker: true, snr_db: null },
    {
      strength: 2,
      direction: "up",
      status: "not_applied",
      reason: "The perturbation did not change the audio.",
      similarity: null,
      same_speaker: null,
      snr_db: null,
    },
    { strength: 4, direction: "up", status: "ok", reason: null, similarity: 0.6, same_speaker: true, snr_db: null },
    { strength: 6, direction: "up", status: "ok", reason: null, similarity: 0.5, same_speaker: true, snr_db: null },
  ],
  flips: [
    { direction: "down", flipped: true, flip_strength: -3, last_safe_strength: -2, already_below_at_weakest: false },
    { direction: "up", flipped: false, flip_strength: null, last_safe_strength: 6, already_below_at_weakest: false },
  ],
};

const stubFetch = (payload: unknown, ok = true, status = 200) => {
  const fetchMock = vi.fn().mockResolvedValue({ ok, status, json: async () => payload });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const renderCard = (props: Partial<React.ComponentProps<typeof PerturbationSweepCard>> = {}) =>
  render(<PerturbationSweepCard model="ecapa-tdnn" recordingId="rec_001" recordingLabel="clip1.wav" {...props} />);

const runSweep = async () => {
  fireEvent.click(screen.getByRole("button", { name: "Run sweep" }));
  await screen.findByTestId("sweep-summary");
};

const lastTrace = (name: string) => {
  const trace = plotProps[plotProps.length - 1].data.find((item) => item.name === name);
  if (!trace) throw new Error(`No trace named ${name}`);
  return trace;
};

describe("PerturbationSweepCard", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    plotProps.length = 0;
  });

  it("disables Run sweep when no recording is selected", () => {
    const fetchMock = stubFetch(pitchResult);
    renderCard({ recordingId: null, recordingLabel: null });

    const button = screen.getByRole("button", { name: "Run sweep" });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByText("Select a recording to run a sweep.")).toBeInTheDocument();
  });

  it("shows the help line before any sweep has run", () => {
    renderCard();

    expect(
      screen.getByText(
        "Each point is the same clip, perturbed more strongly, compared with its original. Below the dashed line the model would no longer say it is the same speaker."
      )
    ).toBeInTheDocument();
    expect(screen.queryByTestId("plot")).not.toBeInTheDocument();
  });

  it("posts the selected recording, model and type to the sweep endpoint", async () => {
    const fetchMock = stubFetch(pitchResult);
    renderCard();

    fireEvent.click(screen.getByRole("radio", { name: "Pitch shift" }));
    await runSweep();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/tasks\/verification\/perturbation\/sweep$/);
    expect(JSON.parse(init.body)).toEqual({
      model: "ecapa-tdnn",
      recording_id: "rec_001",
      perturbation_type: "pitch_shift",
    });
  });

  it("renders the backend summary and per-direction flip text", async () => {
    stubFetch(pitchResult);
    renderCard();
    await runSweep();

    const summary = screen.getByTestId("sweep-summary");
    expect(summary).toHaveTextContent(pitchResult.summary);
    expect(summary).toHaveTextContent("Pitch down");
    expect(summary).toHaveTextContent("Safe up to -2 semitones");
    expect(summary).toHaveTextContent("Flips at about -3 semitones");
    expect(summary).toHaveTextContent("Pitch up");
    expect(summary).toHaveTextContent("Safe up to +6 semitones");
    expect(summary).toHaveTextContent("Never flips in this range");
  });

  it("says so when the clip is already below the threshold at the weakest strength", async () => {
    stubFetch({
      ...pitchResult,
      flips: [
        { direction: "down", flipped: true, flip_strength: -1, last_safe_strength: null, already_below_at_weakest: true },
        pitchResult.flips[1],
      ],
    });
    renderCard();
    await runSweep();

    expect(screen.getByTestId("sweep-summary")).toHaveTextContent(
      "Flips at about -1 semitones (already below the threshold at the weakest tested strength)"
    );
  });

  it("colours points by verdict, draws a dashed threshold and marks the flip point", async () => {
    stubFetch(pitchResult);
    renderCard();
    await runSweep();

    expect(lastTrace("Same speaker").x).toEqual([-2, -1, 1, 4, 6]);
    expect(lastTrace("Same speaker").marker.color).toBe("#10b981");
    expect(lastTrace("Different speaker").x).toEqual([-6, -4]);
    expect(lastTrace("Different speaker").marker.color).toBe("#f43f5e");

    const threshold = lastTrace("Threshold");
    expect(threshold.y).toEqual([0.4, 0.4]);
    expect(threshold.line.dash).toBe("dash");

    const flip = lastTrace("Flip point");
    expect(flip.x).toEqual([-3]);
    expect(flip.y).toEqual([0.4]);
  });

  it("draws not_applied points as grey hollow markers and leaves them off the line", async () => {
    stubFetch(pitchResult);
    renderCard();
    await runSweep();

    const skipped = lastTrace("Not applied");
    expect(skipped.x).toEqual([2]);
    expect(skipped.marker.symbol).toBe("circle-open");
    expect(skipped.marker.color).toBe("#9ca3af");
    expect(skipped.text[0]).toContain("Not applied: The perturbation did not change the audio.");
    expect(lastTrace("Pitch up").x).not.toContain(2);
    expect(screen.getByTestId("sweep-summary")).toHaveTextContent("1 point could not be applied");
  });

  it("renders a display-only anchor for the unchanged clip", async () => {
    const fetchMock = stubFetch(pitchResult);
    renderCard();
    await runSweep();

    const anchor = lastTrace("Original (unchanged)");
    expect(anchor.x).toEqual([0]);
    expect(anchor.y).toEqual([1]);
    expect(anchor.text).toEqual(["Original (unchanged)"]);
    expect(anchor.marker.symbol).toBe("circle-open");
    expect(anchor.marker.color).toBe("#9ca3af");

    // Never sent to the backend, never part of the verdict markers or flip text.
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).not.toHaveProperty("anchor");
    expect(lastTrace("Same speaker").x).not.toContain(0);
    expect(screen.getByTestId("sweep-summary")).not.toHaveTextContent("Original (unchanged)");
    expect(screen.getByTestId("sweep-summary")).toHaveTextContent("Safe up to -2 semitones");
  });

  it("draws each direction as its own line from the anchor, never joined across it", async () => {
    stubFetch(pitchResult);
    renderCard();
    await runSweep();

    const down = lastTrace("Pitch down");
    const up = lastTrace("Pitch up");
    // Each walks outward from the anchor at 0 / similarity 1.0.
    expect(down.x).toEqual([0, -1, -2, -4, -6]);
    expect(down.y).toEqual([1, 0.8, 0.6, 0.2, 0.1]);
    expect(up.x).toEqual([0, 1, 4, 6]);
    expect(up.y).toEqual([1, 0.8, 0.6, 0.5]);

    // No line trace holds points from both sides of the anchor.
    const lines = plotProps[plotProps.length - 1].data.filter(
      (trace) => trace.mode === "lines" && trace.name !== "Threshold"
    );
    expect(lines).toHaveLength(2);
    for (const line of lines) {
      const xs = line.x as number[];
      expect(xs.some((x) => x < 0) && xs.some((x) => x > 0)).toBe(false);
    }
    expect(
      screen.getByText(
        "Even small pitch shifts lower the score, partly because the pitch-shift algorithm also adds slight processing artifacts."
      )
    ).toBeInTheDocument();
  });

  it("anchors time stretch at factor 1.0 with separate slower and faster lines", async () => {
    stubFetch({
      ...pitchResult,
      perturbation_type: "time_stretch",
      points: [
        { strength: 0.7, direction: "slower", status: "ok", reason: null, similarity: 0.5, same_speaker: true, snr_db: null },
        { strength: 0.95, direction: "slower", status: "ok", reason: null, similarity: 0.9, same_speaker: true, snr_db: null },
        { strength: 1.05, direction: "faster", status: "ok", reason: null, similarity: 0.9, same_speaker: true, snr_db: null },
        { strength: 1.5, direction: "faster", status: "ok", reason: null, similarity: 0.6, same_speaker: true, snr_db: null },
      ],
      flips: [
        { direction: "slower", flipped: false, flip_strength: null, last_safe_strength: 0.7, already_below_at_weakest: false },
        { direction: "faster", flipped: false, flip_strength: null, last_safe_strength: 1.5, already_below_at_weakest: false },
      ],
    });
    renderCard();
    await runSweep();

    expect(lastTrace("Original (unchanged)").x).toEqual([1]);
    expect(lastTrace("Slower").x).toEqual([1, 0.95, 0.7]);
    expect(lastTrace("Faster").x).toEqual([1, 1.05, 1.5]);
    expect(screen.queryByText(/Even small pitch shifts/)).not.toBeInTheDocument();
  });

  it("shows SNR on the axis for a noise sweep", async () => {
    stubFetch({
      ...pitchResult,
      perturbation_type: "noise",
      points: [
        { strength: 0.01, direction: "stronger", status: "ok", reason: null, similarity: 0.9, same_speaker: true, snr_db: 26.5 },
        { strength: 0.1, direction: "stronger", status: "ok", reason: null, similarity: 0.3, same_speaker: false, snr_db: 6.5 },
      ],
      flips: [
        { direction: "stronger", flipped: true, flip_strength: 0.085, last_safe_strength: 0.01, already_below_at_weakest: false },
      ],
    });
    renderCard();
    await runSweep();

    const { xaxis } = plotProps[plotProps.length - 1].layout;
    expect(xaxis.type).toBe("log");
    expect(xaxis.title.text).toContain("SNR");
    expect(xaxis.ticktext).toEqual(["No noise", "0.01<br>27 dB", "0.1<br>7 dB"]);

    // The noise anchor sits at the left edge, and the single line starts from it.
    const anchor = lastTrace("No noise");
    expect(anchor.x).toEqual([0.005]);
    expect(anchor.y).toEqual([1]);
    expect(lastTrace("Added noise").x).toEqual([0.005, 0.01, 0.1]);
    expect(lastTrace("Threshold").x).toEqual([0.005, 0.1]);
    expect(screen.getByTestId("sweep-summary")).toHaveTextContent("Flips at about noise level 0.085");
  });

  it("shows the backend error detail when the sweep fails", async () => {
    stubFetch({ detail: "No 'pitch_shift' sweep point changed the audio." }, false, 422);
    renderCard();

    fireEvent.click(screen.getByRole("button", { name: "Run sweep" }));

    expect(await screen.findByText("No 'pitch_shift' sweep point changed the audio.")).toBeInTheDocument();
    expect(screen.queryByTestId("sweep-summary")).not.toBeInTheDocument();
  });

  it("clears the result when the recording changes", async () => {
    stubFetch(pitchResult);
    const { rerender } = renderCard();
    await runSweep();

    rerender(<PerturbationSweepCard model="ecapa-tdnn" recordingId="rec_002" recordingLabel="clip2.wav" />);

    await waitFor(() => expect(screen.queryByTestId("sweep-summary")).not.toBeInTheDocument());
    expect(screen.queryByTestId("plot")).not.toBeInTheDocument();
  });

  it("clears the result when the model changes", async () => {
    stubFetch(pitchResult);
    const { rerender } = renderCard();
    await runSweep();

    rerender(<PerturbationSweepCard model="resnet34-lm" recordingId="rec_001" recordingLabel="clip1.wav" />);

    await waitFor(() => expect(screen.queryByTestId("sweep-summary")).not.toBeInTheDocument());
  });

  it("aborts an in-flight sweep when the recording changes and can run again", async () => {
    const signals: AbortSignal[] = [];
    const fetchMock = vi.fn().mockImplementation((_url: string, init: RequestInit) => {
      signals.push(init.signal as AbortSignal);
      if (signals.length === 1) return new Promise(() => {}); // first request never settles
      return Promise.resolve({ ok: true, status: 200, json: async () => pitchResult });
    });
    vi.stubGlobal("fetch", fetchMock);

    const { rerender } = renderCard();
    fireEvent.click(screen.getByRole("button", { name: "Run sweep" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    // Changing recording aborts the in-flight sweep and re-enables the button.
    rerender(<PerturbationSweepCard model="ecapa-tdnn" recordingId="rec_002" recordingLabel="clip2.wav" />);
    await waitFor(() => expect(signals[0].aborted).toBe(true));

    await runSweep();
    expect(signals[1].aborted).toBe(false);
  });
});
