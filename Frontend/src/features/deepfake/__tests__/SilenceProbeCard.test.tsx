/**
 * SilenceProbeCard — Feature 2 (SRS DF-10..DF-12) in the interface.
 *
 * This is the view that can invalidate every other number on the screen, so
 * the tests are written around the three things it must never soften: the
 * three scores appear TOGETHER, the energy threshold that defined "silence"
 * travels with them, and an inapplicable leg says so instead of showing a
 * number nobody should trust.
 *
 * The fixture is the measured behaviour of a real genuine clip
 * (LA_E_1070252): 0.063 as submitted, 0.428 trimmed, 0.868 on the silence
 * alone -- which is precisely the finding the card exists to surface.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SilenceProbeCard } from "../SilenceProbeCard";
import { silenceProbe, stubFetch } from "./fixtures";

const renderCard = (recordingId = "rec_1070252") =>
  render(
    <SilenceProbeCard
      model="xlsr-deepfake"
      modelLabel="wav2vec2 XLS-R (Model A)"
      recordingId={recordingId}
    />,
  );

const run = async () => {
  await userEvent.click(screen.getByRole("button", { name: /run the silence probe/i }));
};

describe("SilenceProbeCard", () => {
  it("shows all three scorings together (DF-10)", async () => {
    stubFetch(silenceProbe);
    renderCard();
    await run();

    expect(await screen.findByText("As submitted")).toBeInTheDocument();
    expect(screen.getByText("Silence trimmed")).toBeInTheDocument();
    expect(screen.getByText("Non-speech only")).toBeInTheDocument();
    expect(screen.getByText("0.063")).toBeInTheDocument();
    expect(screen.getByText("0.428")).toBeInTheDocument();
    expect(screen.getByText("0.868")).toBeInTheDocument();
  });

  it("shows each ablation's change against the original score", async () => {
    stubFetch(silenceProbe);
    renderCard();
    await run();

    expect(await screen.findByText("+0.365")).toBeInTheDocument(); // trimmed
    expect(screen.getByText("+0.805")).toBeInTheDocument(); // silence only
  });

  it("raises the alarm when the silence alone reads as spoof", async () => {
    stubFetch(silenceProbe);
    renderCard();
    await run();

    expect(await screen.findByText("The silence alone reads as spoof")).toBeInTheDocument();
    expect(
      screen.getByText(/in the recording conditions, not the voice/),
    ).toBeInTheDocument();
  });

  it("raises the alarm when trimming flips the decision", async () => {
    stubFetch({
      ...silenceProbe,
      variants: {
        ...silenceProbe.variants,
        trimmed: { applicable: true, seconds: 2.29, spoof_probability: 0.88, decision: "spoof" },
        non_speech: {
          applicable: true,
          seconds: 1.13,
          spoof_probability: 0.1,
          decision: "bonafide",
        },
      },
    });
    renderCard();
    await run();

    expect(
      await screen.findByText("Trimming the silence flipped the decision"),
    ).toBeInTheDocument();
  });

  it("reports a clean result only when both legs actually pass", async () => {
    stubFetch({
      ...silenceProbe,
      variants: {
        original: { applicable: true, seconds: 3.42, spoof_probability: 0.92, decision: "spoof" },
        trimmed: { applicable: true, seconds: 2.29, spoof_probability: 0.91, decision: "spoof" },
        non_speech: {
          applicable: true,
          seconds: 1.13,
          spoof_probability: 0.08,
          decision: "bonafide",
        },
      },
    });
    renderCard();
    await run();

    expect(await screen.findByText("The score survives the ablation")).toBeInTheDocument();
  });

  it("says the stronger half is untested when the silence leg cannot run (DF-12)", async () => {
    // Trimming has to leave the score alone here, otherwise the earlier
    // "trimming moved the score" verdict fires first and this branch is
    // never reached. A near-silence-free spoof clip is the realistic case:
    // LA_E_1119893 has 0.00 non-speech fraction in the measured ablation.
    stubFetch({
      ...silenceProbe,
      variants: {
        original: { applicable: true, seconds: 3.05, spoof_probability: 0.919, decision: "spoof" },
        trimmed: { applicable: true, seconds: 3.04, spoof_probability: 0.919, decision: "spoof" },
        non_speech: {
          applicable: false,
          seconds: 0.31,
          spoof_probability: null,
          decision: null,
          reason:
            "Only 0.31s of non-speech audio was found (at least 0.50s is needed). " +
            "The probe cannot say anything reliable about this clip.",
        },
      },
    });
    renderCard();
    await run();

    expect(await screen.findByText("Not applicable")).toBeInTheDocument();
    expect(screen.getByText(/Only 0\.31s of non-speech audio was found/)).toBeInTheDocument();
    expect(
      screen.getByText(/the silence check could not run/),
    ).toBeInTheDocument();
  });

  it("reports the energy threshold that defined silence (DF-11)", async () => {
    stubFetch(silenceProbe);
    renderCard();
    await run();

    expect(
      await screen.findByText(/energy threshold 30 dB below this clip/),
    ).toBeInTheDocument();
    expect(screen.getByText(/not an absolute noise floor/)).toBeInTheDocument();
  });

  it("shows how much of the clip is speech versus non-speech", async () => {
    stubFetch(silenceProbe);
    renderCard();
    await run();

    expect(await screen.findByText("2.29s / 1.13s")).toBeInTheDocument();
  });

  it("clears a previous clip's probe when the selection changes", async () => {
    stubFetch(silenceProbe);
    const { rerender } = renderCard();
    await run();
    await screen.findByText("0.063");

    rerender(
      <SilenceProbeCard
        model="xlsr-deepfake"
        modelLabel="wav2vec2 XLS-R (Model A)"
        recordingId="rec_another"
      />,
    );

    // A stale probe would attribute one clip's behaviour to a different clip.
    expect(screen.queryByText("0.063")).not.toBeInTheDocument();
  });

  it("surfaces a backend failure instead of an empty card", async () => {
    stubFetch({ detail: "checkpoint unavailable" }, { ok: false, status: 503 });
    renderCard();
    await run();

    expect(await screen.findByText("checkpoint unavailable")).toBeInTheDocument();
  });
});
