/**
 * DeepfakeWorkbench — the centre panel, end to end against a stubbed API.
 *
 * The workbench's premise is an exercise: listen, read the score, and only
 * then look up the answer. That premise breaks the moment a label appears on
 * screen, so the last test here sweeps the whole rendered DOM for one. The
 * rest cover the path a user actually walks: the list loads, a clip is
 * chosen, detection runs, and the result is presented as a score with a
 * stated -- and currently uncalibrated -- threshold.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DeepfakeWorkbench } from "../DeepfakeWorkbench";
import { detection, recordings } from "./fixtures";

/** Route the two endpoints the workbench uses. */
const stubApi = (detectionBody: unknown = detection, ok = true, status = 200) => {
  const fetchMock = vi.fn(async (url: string) => {
    if (String(url).includes("/dataset/recordings")) {
      return { ok: true, status: 200, json: async () => recordings };
    }
    return { ok, status, json: async () => detectionBody };
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const renderWorkbench = (
  props: Partial<React.ComponentProps<typeof DeepfakeWorkbench>> = {},
) =>
  render(
    <DeepfakeWorkbench
      model="xlsr-deepfake"
      modelLabel="wav2vec2 XLS-R (Model A)"
      selectedFile={null}
      onFileSelect={() => {}}
      {...props}
    />,
  );

describe("DeepfakeWorkbench", () => {
  it("lists the dataset's recordings by filename", async () => {
    stubApi();
    renderWorkbench();

    expect(await screen.findByRole("option", { name: "LA_E_1070252.flac" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "LA_E_9938640.flac" })).toBeInTheDocument();
  });

  it("cannot run until a recording is chosen", async () => {
    stubApi();
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });

    expect(screen.getByRole("button", { name: /run detection/i })).toBeDisabled();
  });

  it("publishes the choice so the other panels follow it", async () => {
    stubApi();
    const onFileSelect = vi.fn();
    renderWorkbench({ onFileSelect });
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });

    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");

    expect(onFileSelect).toHaveBeenCalledWith(
      expect.objectContaining({ file_id: "rec_a", filename: "LA_E_1070252.flac" }),
    );
  });

  it("scores the chosen clip and shows the score with its threshold", async () => {
    const fetchMock = stubApi();
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });

    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    expect(await screen.findByText("0.143")).toBeInTheDocument();
    expect(screen.getByText(/threshold 0\.50, uncalibrated/)).toBeInTheDocument();
    expect(screen.getByText("BONA FIDE")).toBeInTheDocument();

    const [url, init] = fetchMock.mock.calls.at(-1) as unknown as [string, RequestInit];
    expect(url).toMatch(/\/tasks\/deepfake\/run$/);
    expect(JSON.parse(init.body as string)).toEqual({
      model: "xlsr-deepfake",
      recording_id: "rec_a",
    });
  });

  it("warns that the decision rests on an uncalibrated cut", async () => {
    stubApi();
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });
    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    expect(await screen.findByText("This threshold is not calibrated")).toBeInTheDocument();
    expect(screen.getByText(/naive midpoint/)).toBeInTheDocument();
  });

  it("says when only part of the clip was scored", async () => {
    stubApi({ ...detection, truncated: true, duration: 18.4, analysed_seconds: 10.24 });
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });
    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    expect(await screen.findByText("Clip truncated")).toBeInTheDocument();
    expect(screen.getByText(/Scored the first 10\.24s of a 18\.4s clip/)).toBeInTheDocument();
  });

  it("shows which class index the checkpoint itself calls spoof", async () => {
    stubApi();
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });
    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    // Read from the model's config, never hardcoded -- an inverted mapping
    // would otherwise be invisible.
    expect(await screen.findByText("1 · spoof")).toBeInTheDocument();
  });

  it("surfaces a detection failure", async () => {
    stubApi({ detail: "Could not load the checkpoint" }, false, 503);
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });
    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));

    expect(await screen.findByText("Could not load the checkpoint")).toBeInTheDocument();
  });

  it("drops a previous clip's score when the selection changes", async () => {
    stubApi();
    renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });
    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await screen.findByText("0.143");

    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_c");

    expect(screen.queryByText("0.143")).not.toBeInTheDocument();
  });

  it("never shows the dataset's bona fide/spoof answer", async () => {
    stubApi();
    const { container } = renderWorkbench();
    await screen.findByRole("option", { name: "LA_E_1070252.flac" });
    await userEvent.selectOptions(screen.getByRole("combobox"), "rec_a");
    await userEvent.click(screen.getByRole("button", { name: /run detection/i }));
    await screen.findByText("0.143");

    const text = container.textContent ?? "";
    // What must not appear is the protocol's DATA: an attack id, a system id,
    // or a speaker id. Prose about the method ("lines the scores up against
    // the ground truth") is a description of what happens offline and is not
    // a leak, so it is deliberately not in this list.
    for (const token of ["A07", "A16", "A19", "system_id", "LA_0069"]) {
      expect(text).not.toContain(token);
    }
    // The only verdict on screen is the model's own, taken at its threshold.
    expect(text).toContain("BONA FIDE");
    expect(detection.spoof_probability < detection.threshold).toBe(true);
    // And the UI says out loud that the labels are withheld on purpose.
    expect(
      screen.getByText(/labels are deliberately not shown/),
    ).toBeInTheDocument();
  });

  it("reports a listing failure instead of rendering an empty picker", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 404,
        json: async () => ({ detail: "Deepfake demo dataset not found: asvspoof2019-la" }),
      })),
    );
    renderWorkbench();

    await waitFor(() =>
      expect(screen.getByText(/Deepfake demo dataset not found/)).toBeInTheDocument(),
    );
  });
});
