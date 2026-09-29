import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { WhatIf } from "../page/WhatIf";
import { changeBadges, perturbationFinding, perturbedColourKey } from "../page/whatIfWords";
import type { DiarizationDelta, PerturbationResult } from "../types";
import { perturbation, result } from "./fixtures";

const renderWhatIf = (
  overrides: Partial<{
    perturbation: PerturbationResult | null;
    perturbedAudioUrl: string | undefined;
    isPreviewing: boolean;
    previewPerturbation: () => void;
  }> = {},
) =>
  render(
    <WhatIf
      result={result}
      perturbationType="noise"
      setPerturbationType={vi.fn()}
      noisePercent={4}
      setNoisePercent={vi.fn()}
      maskRange={[20, 40]}
      setMaskRange={vi.fn()}
      perturbation={perturbation}
      isPerturbing={false}
      perturbationError={null}
      runPerturbation={vi.fn()}
      isPreviewing={false}
      previewPerturbation={vi.fn()}
      canRun
      selectedId={null}
      onSelectOriginal={vi.fn()}
      onSelectPerturbed={vi.fn()}
      perturbedAudioRef={createRef<HTMLAudioElement>()}
      perturbedAudioUrl="/audio/prt_demo"
      {...overrides}
    />,
  );

describe("whatIfWords", () => {
  it("writes the finding from the backend's numbers", () => {
    const finding = perturbationFinding(perturbation.delta, perturbation.original, perturbation.perturbed, perturbation.perturbation);
    // confusion 0.6 s of 12 s reference speech = 5%
    expect(finding.title).toBe("With a little noise, the system found 3 speakers instead of 2; 5% of the talk time changed hands.");
    expect(finding.alarming).toBe(true);
  });

  it("says when every speaker survived", () => {
    const calm: DiarizationDelta = {
      ...perturbation.delta,
      appeared: [],
      num_speakers_delta: 0,
      der_components: { missed_detection: 0, false_alarm: 0, confusion: 0, total: 12 },
    };
    const finding = perturbationFinding(calm, perturbation.original, perturbation.original, perturbation.perturbation);
    expect(finding.title).toBe("With a little noise, the system still found both speakers; no talk time changed hands.");
    expect(finding.alarming).toBe(false);
  });

  it("colours perturbed speakers by their matched original speaker", () => {
    const key = perturbedColourKey(perturbation.delta, result.speakers, perturbation.perturbed.speakers);
    expect(key.keyOf("SPEAKER_01")).toBe("SPEAKER_00");
    expect(key.keyOf("SPEAKER_00")).toBe("SPEAKER_01");
    // an unmatched voice takes a slot after the original speakers
    expect(key.speakers.indexOf(key.keyOf("SPEAKER_02"))).toBe(2);
  });

  it("describes merges and splits by original labels", () => {
    const badges = changeBadges({
      ...perturbation.delta,
      appeared: [],
      merged: [{ perturbed: "SPEAKER_00", from: ["SPEAKER_00", "SPEAKER_01"] }],
      split: [{ original: "SPEAKER_02", into: ["SPEAKER_03", "SPEAKER_04"] }],
    });
    expect(badges.map((b) => b.text)).toEqual([
      "SPEAKER_00 and SPEAKER_01 were merged into one voice",
      "SPEAKER_02 was split into 2 voices",
    ]);
  });
});

describe("WhatIf", () => {
  it("labels DER as change from the original run and never as accuracy", () => {
    const { container } = renderWhatIf();
    fireEvent.click(screen.getByRole("button", { name: /technical details/i }));
    expect(screen.getByText("Change from the original run")).toBeInTheDocument();
    expect(screen.getByTestId("der-value")).toHaveTextContent("30.0%");
    expect(container.textContent).not.toMatch(/accuracy/i);
  });

  it("keeps a voice's colour across the before and after timelines", () => {
    renderWhatIf();
    const before = within(screen.getByTestId("lane-before"));
    const after = within(screen.getByTestId("lane-after"));
    // After-run SPEAKER_01 was matched to original SPEAKER_00, so it is labelled and coloured as SPEAKER_00.
    const originalBar = before.getByRole("button", { name: /^SPEAKER_00, 00:00\.0 to 00:03\.0/ });
    const perturbedBar = after.getByRole("button", { name: /^SPEAKER_00, 00:00\.0 to 00:03\.0/ });
    expect(perturbedBar.style.background).toBe(originalBar.style.background);
  });

  it("shows an animated badge for the voice that appeared", () => {
    renderWhatIf();
    expect(screen.getAllByTestId("change-badge").map((b) => b.textContent)).toEqual(["A new voice appeared (SPEAKER_02)"]);
  });

  it("lets the changed audio be built and played before the second run", () => {
    const previewPerturbation = vi.fn();
    const { container, rerender } = renderWhatIf({ perturbation: null, perturbedAudioUrl: undefined, previewPerturbation });
    expect(container.querySelector("audio")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Hear the changed audio" }));
    expect(previewPerturbation).toHaveBeenCalledOnce();

    // The preview landed: a player for the clip, no comparison yet.
    rerender(
      <WhatIf
        result={result}
        perturbationType="noise"
        setPerturbationType={vi.fn()}
        noisePercent={4}
        setNoisePercent={vi.fn()}
        maskRange={[20, 40]}
        setMaskRange={vi.fn()}
        perturbation={null}
        isPerturbing={false}
        perturbationError={null}
        runPerturbation={vi.fn()}
        isPreviewing={false}
        previewPerturbation={previewPerturbation}
        canRun
        selectedId={null}
        onSelectOriginal={vi.fn()}
        onSelectPerturbed={vi.fn()}
        perturbedAudioRef={createRef<HTMLAudioElement>()}
        perturbedAudioUrl="/audio/prt_demo"
      />,
    );
    expect(container.querySelector("audio")).toHaveAttribute("src", "/audio/prt_demo");
    expect(screen.queryByTestId("lane-after")).toBeNull();
  });

  it("keeps one player for the clip once the comparison has run", () => {
    const { container } = renderWhatIf();
    expect(container.querySelectorAll("audio")).toHaveLength(1);
    expect(screen.getByText("Hear what the second run heard")).toBeInTheDocument();
  });

  it("locks the settings and the run while the clip is being built", () => {
    renderWhatIf({ perturbation: null, isPreviewing: true });
    expect(screen.getByRole("button", { name: /changing the audio/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /see what changes/i })).toBeDisabled();
    expect(screen.getByRole("radio", { name: /background noise/i })).toBeDisabled();
  });
});
