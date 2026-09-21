/**
 * SaliencyPanel — Feature 3 (SRS DF-14, DF-15) in the interface.
 *
 * DF-15 is a requirement about honesty rather than about pixels: the method
 * must be NAMED, the caps stated, and the normalisation disclosed, because an
 * attribution map is the easiest thing in this workbench to over-read. Those
 * are the assertions here, alongside the strip being operable by keyboard --
 * it is a seek control, not only a picture.
 *
 * jsdom has no Web Audio, so the waveform decode always fails; the component
 * is built to render the heat strip without it, and this suite silently
 * depends on that being true.
 */
import { beforeAll, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SaliencyPanel } from "../SaliencyPanel";
import { saliency, stubFetch } from "./fixtures";

// jsdom ships no media pipeline, so HTMLMediaElement.play() returns undefined
// where every real browser returns a Promise. The seek handler calls .catch()
// on it, which would throw an error unrelated to the code under test.
beforeAll(() => {
  window.HTMLMediaElement.prototype.play = () => Promise.resolve();
  window.HTMLMediaElement.prototype.pause = () => {};
});

const renderPanel = (recordingId = "rec_1070252") =>
  render(
    <SaliencyPanel
      model="xlsr-deepfake"
      modelLabel="wav2vec2 XLS-R (Model A)"
      recordingId={recordingId}
    />,
  );

const run = async () => {
  await userEvent.click(screen.getByRole("button", { name: /show saliency/i }));
};

describe("SaliencyPanel", () => {
  it("names the attribution method rather than implying one (DF-15)", async () => {
    stubFetch(saliency);
    renderPanel();
    await run();

    expect(
      await screen.findByText(/Input gradient \(\|d spoof logit \/ d input\|\)/),
    ).toBeInTheDocument();
    expect(screen.getByText(/taken on the spoof logit/)).toBeInTheDocument();
  });

  it("discloses that attribution is ranked within the clip only", async () => {
    stubFetch(saliency);
    renderPanel();
    await run();

    expect(
      await screen.findByText(/never compares across clips or models/),
    ).toBeInTheDocument();
  });

  it("states the shared duration cap (DF-15)", async () => {
    stubFetch(saliency);
    renderPanel();
    await run();

    expect(await screen.findByText(/capped at 12s/)).toBeInTheDocument();
    expect(
      screen.getByText(/the same cap the shared saliency service applies/),
    ).toBeInTheDocument();
  });

  it("says so when the clip was truncated to fit the cap", async () => {
    stubFetch({ ...saliency, truncated: true, total_duration: 12 });
    renderPanel();
    await run();

    expect(await screen.findByText(/this clip was truncated to fit/)).toBeInTheDocument();
  });

  it("draws the attribution aligned with the waveform", async () => {
    stubFetch(saliency);
    renderPanel();
    await run();

    expect(
      await screen.findByRole("img", { name: /saliency aligned with the waveform/i }),
    ).toBeInTheDocument();
  });

  it("flags attribution that lands outside the speech", async () => {
    // Speech occupies 67% of the fixture clip but takes only 24% of the
    // attribution, so the verdict must be the alarming one. It is judged
    // against the clip's own speech share, not a flat 50%, which is what
    // keeps a mostly-silent clip from being condemned automatically.
    stubFetch(saliency);
    renderPanel();
    await run();

    expect(
      await screen.findByText("Only 24.0% of the attribution falls on speech"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Speech occupies 67\.0% of this clip/)).toBeInTheDocument();
  });

  it("does not flag attribution that sits where the voice is", async () => {
    stubFetch({ ...saliency, saliency_in_speech_fraction: 0.81 });
    renderPanel();
    await run();

    expect(
      await screen.findByText("81.0% of the attribution falls on speech"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/reacting to the speech rather than to the silence/),
    ).toBeInTheDocument();
  });

  it("exposes the strip as a keyboard-operable seek control", async () => {
    stubFetch(saliency);
    renderPanel();
    await run();

    const slider = await screen.findByRole("slider", { name: /seek within the clip/i });
    expect(slider).toHaveAttribute("aria-valuemax", "3.42");
    expect(slider).toHaveAttribute("aria-valuenow", "0");

    slider.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(slider).toHaveAttribute("aria-valuenow", "0.25");

    await userEvent.keyboard("{End}");
    expect(slider).toHaveAttribute("aria-valuenow", "3.37");
  });

  it("does not seek past the ends of the clip", async () => {
    stubFetch(saliency);
    renderPanel();
    await run();

    const slider = await screen.findByRole("slider", { name: /seek within the clip/i });
    slider.focus();
    await userEvent.keyboard("{ArrowLeft}");
    expect(slider).toHaveAttribute("aria-valuenow", "0");
  });

  it("clears the attribution when the model changes", async () => {
    stubFetch(saliency);
    const { rerender } = renderPanel();
    await run();
    await screen.findByRole("slider", { name: /seek within the clip/i });

    rerender(
      <SaliencyPanel
        model="xlsr-mamba"
        modelLabel="XLSR-Mamba (Model C)"
        recordingId="rec_1070252"
      />,
    );

    expect(
      screen.queryByRole("slider", { name: /seek within the clip/i }),
    ).not.toBeInTheDocument();
  });

  it("surfaces a 422 from the backend", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        String(url).includes("/saliency")
          ? {
              ok: false,
              status: 422,
              json: async () => ({ detail: "The clip was too short to attribute." }),
            }
          : { ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(8) },
      ),
    );
    renderPanel();
    await run();

    expect(await screen.findByText("The clip was too short to attribute.")).toBeInTheDocument();
  });
});
