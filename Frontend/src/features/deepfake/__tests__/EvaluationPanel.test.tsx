/**
 * EvaluationPanel — Feature 1 (SRS DF-6..DF-9) in the interface.
 *
 * The panel's job is to make three things unmissable: how separable the two
 * populations are, what the threshold in force costs, and that the threshold
 * belongs to one dataset. A component that showed only "EER 0.00%" would pass
 * a naive test and fail the requirement, so the provenance line and the
 * operating-point cost are asserted explicitly.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EvaluationPanel } from "../EvaluationPanel";
import { evaluation, stubFetch } from "./fixtures";

const renderPanel = (props: Partial<React.ComponentProps<typeof EvaluationPanel>> = {}) =>
  render(
    <EvaluationPanel
      model="xlsr-deepfake"
      modelLabel="wav2vec2 XLS-R (Model A)"
      datasetAvailable
      {...props}
    />,
  );

describe("EvaluationPanel", () => {
  it("refuses to run without the labelled dataset on disk", () => {
    renderPanel({ datasetAvailable: false });
    expect(screen.getByText("No labelled dataset")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /evaluate/i })).toBeDisabled();
  });

  it("reports the equal error rate and the clip counts it was measured over", async () => {
    stubFetch(evaluation);
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    // "0.00%" appears twice by design: once as the headline stat and once in
    // the DET curve's legend, where it labels the marked crossing point.
    expect(await screen.findAllByText("0.00%")).toHaveLength(2);
    expect(screen.getByText("0.1429")).toBeInTheDocument();
    expect(screen.getByText("Genuine clips")).toBeInTheDocument();
    expect(screen.getAllByText("100")).toHaveLength(2);
  });

  it("carries the threshold's provenance, so it is never quoted bare (DF-9)", async () => {
    stubFetch(evaluation);
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    expect(
      await screen.findByText(/Thresholds do not transfer between datasets/),
    ).toBeInTheDocument();
  });

  it("shows what the threshold in force costs, in both error directions", async () => {
    stubFetch(evaluation);
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    // 0.02 false acceptance at the shipped 0.5 -- two A19 clips let through.
    expect(await screen.findByText("2.0%")).toBeInTheDocument();
    expect(screen.getByText("0.0%")).toBeInTheDocument();
    expect(screen.getByText(/spoofed clips let through as genuine/)).toBeInTheDocument();
    expect(screen.getByText(/At the threshold in force \(0\.50, uncalibrated\)/)).toBeInTheDocument();
  });

  it("breaks the scores down per spoofing system", async () => {
    stubFetch(evaluation);
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    expect(await screen.findByText("A19")).toBeInTheDocument();
    // A19 is where Model A is weakest on this subset: mean 0.5945.
    expect(screen.getByText("0.595")).toBeInTheDocument();
    expect(screen.getByText("0.920")).toBeInTheDocument();
    // "bonafide" is relabelled in the table, so the row never reads as a
    // per-clip answer. (The distribution legend also says "genuine", hence
    // the scoped query.)
    expect(screen.getByRole("table")).toHaveTextContent("genuine");
    expect(screen.getByRole("table")).toHaveTextContent("n=7");
  });

  it("renders both distributions and the DET curve", async () => {
    stubFetch(evaluation);
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    expect(
      await screen.findByRole("img", {
        name: /score distributions for genuine and synthetic clips/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /detection error tradeoff curve/i }),
    ).toBeInTheDocument();
  });

  it("surfaces a 422 from the backend instead of showing a blank panel", async () => {
    stubFetch(
      { detail: "A DET curve needs at least one genuine and one spoofed clip; got 4 genuine and 0 spoofed." },
      { ok: false, status: 422 },
    );
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    expect(await screen.findByText(/at least one genuine and one spoofed clip/)).toBeInTheDocument();
  });

  it("warns rather than silently showing another model's results", async () => {
    stubFetch(evaluation);
    const { rerender } = renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));
    await screen.findAllByText("0.00%");

    rerender(
      <EvaluationPanel
        model="xlsr-mamba"
        modelLabel="XLSR-Mamba (Model C)"
        datasetAvailable
      />,
    );

    expect(await screen.findByText("Model changed")).toBeInTheDocument();
    expect(screen.queryAllByText("0.00%")).toHaveLength(0);
  });

  it("asks the backend for the currently selected model", async () => {
    const fetchMock = stubFetch(evaluation);
    renderPanel({ model: "xlsr-mamba", modelLabel: "XLSR-Mamba (Model C)" });
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/tasks\/deepfake\/scores$/);
    expect(JSON.parse(init.body as string)).toEqual({ model: "xlsr-mamba" });
  });

  it("tells the user the first run is slow before it finishes", async () => {
    let resolve!: (value: unknown) => void;
    const pending = new Promise((r) => {
      resolve = r;
    });
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        await pending;
        return { ok: true, status: 200, json: async () => evaluation };
      }),
    );
    renderPanel();
    await userEvent.click(screen.getByRole("button", { name: /evaluate/i }));

    expect(await screen.findByText(/One forward pass per clip/)).toBeInTheDocument();
    resolve(null);
    expect(await screen.findAllByText("0.00%")).toHaveLength(2);
  });
});
