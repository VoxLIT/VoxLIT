/**
 * ScoreBar — the score and the cut that turns it into a decision.
 *
 * The component exists to stop a user reading "SPOOF" as a fact. So the
 * things worth testing are exactly the ones that would let it drift back into
 * a verdict badge: the number must be shown to three decimals, the threshold
 * must be drawn and labelled, and an uncalibrated threshold must say so.
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ScoreBar } from "../ScoreBar";

describe("ScoreBar", () => {
  it("shows the score itself, not only the decision", () => {
    render(<ScoreBar spoofProbability={0.142884} threshold={0.5} calibrated={false} />);
    // Three decimals: enough to separate two clips, few enough to read.
    expect(screen.getByText("0.143")).toBeInTheDocument();
    expect(screen.getByText("spoof score")).toBeInTheDocument();
  });

  it("says when the threshold is uncalibrated", () => {
    render(<ScoreBar spoofProbability={0.92} threshold={0.5} calibrated={false} />);
    expect(screen.getByText(/threshold 0\.50, uncalibrated/)).toBeInTheDocument();
  });

  it("drops the caveat once the threshold is calibrated", () => {
    // The EER threshold measured for Model A on the demo subset.
    render(<ScoreBar spoofProbability={0.92} threshold={0.142884} calibrated={true} />);
    expect(screen.getByText(/threshold 0\.14$/)).toBeInTheDocument();
    expect(screen.queryByText(/uncalibrated/)).not.toBeInTheDocument();
  });

  it("draws the bar to the score and the marker to the threshold", () => {
    const { container } = render(
      <ScoreBar spoofProbability={0.75} threshold={0.5} calibrated={false} />,
    );
    const [bar, marker] = Array.from(container.querySelectorAll<HTMLElement>("div[style]"));
    expect(bar.style.width).toBe("75%");
    expect(marker.style.left).toBe("50%");
  });

  it("clamps a score outside 0..1 instead of overflowing the track", () => {
    const { container } = render(
      <ScoreBar spoofProbability={1.4} threshold={0.5} calibrated={false} />,
    );
    const [bar] = Array.from(container.querySelectorAll<HTMLElement>("div[style]"));
    expect(bar.style.width).toBe("100%");
  });

  it("colours the bar by which side of the threshold the score falls", () => {
    const { container: spoofSide } = render(
      <ScoreBar spoofProbability={0.51} threshold={0.5} calibrated={false} />,
    );
    expect(spoofSide.querySelector(".bg-destructive\\/70")).not.toBeNull();

    const { container: genuineSide } = render(
      <ScoreBar spoofProbability={0.49} threshold={0.5} calibrated={false} />,
    );
    expect(genuineSide.querySelector(".bg-primary\\/60")).not.toBeNull();
  });

  it("treats a score exactly at the threshold as spoof, matching the backend", () => {
    // service.run_detection: `score >= threshold` -> "spoof". The UI must not
    // disagree with the decision the API already took.
    const { container } = render(
      <ScoreBar spoofProbability={0.5} threshold={0.5} calibrated={false} />,
    );
    expect(container.querySelector(".bg-destructive\\/70")).not.toBeNull();
  });

  it("labels both ends of the axis so the direction is unambiguous", () => {
    render(<ScoreBar spoofProbability={0.5} threshold={0.5} calibrated={false} />);
    expect(screen.getByText(/0\.0 — bona fide/)).toBeInTheDocument();
    expect(screen.getByText(/spoof — 1\.0/)).toBeInTheDocument();
  });
});
