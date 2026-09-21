/**
 * DeepfakeEmbeddingPanel — Feature 4 in the interface.
 *
 * The one property this view must never lose is that points are coloured by
 * the DETECTOR's score and never by the dataset's labels. The backend
 * enforces that by not sending labels at all; the front end has to not invent
 * them, so the plot's own props are inspected here rather than a screenshot.
 *
 * react-plotly.js is replaced with a probe component: Plotly needs a real
 * layout engine, and what matters for these tests is the data handed to it.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const plotProps: Record<string, unknown>[] = [];
vi.mock("react-plotly.js", () => ({
  default: (props: Record<string, unknown>) => {
    plotProps.push(props);
    return <div data-testid="plot" />;
  },
}));

const { DeepfakeEmbeddingPanel } = await import("../DeepfakeEmbeddingPanel");
const { projection, stubFetch } = await import("./fixtures");
const { BONAFIDE_COLOR, SPOOF_COLOR } = await import("../embeddingColors");

const renderPanel = (
  props: Partial<React.ComponentProps<typeof DeepfakeEmbeddingPanel>> = {},
) =>
  render(
    <DeepfakeEmbeddingPanel
      model="xlsr-deepfake"
      modelLabel="wav2vec2 XLS-R (Model A)"
      availableFiles={["a", "b", "c"]}
      selectedFile={null}
      onFileSelect={() => {}}
      {...props}
    />,
  );

const start = async () => {
  await userEvent.click(screen.getByRole("button", { name: /show the dataset in 2d \/ 3d/i }));
};

describe("DeepfakeEmbeddingPanel", () => {
  it("stays idle until asked, because the first run scores the whole dataset", () => {
    const fetchMock = stubFetch(projection);
    renderPanel();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(
      screen.getByText(/coloured by this detector's own score/i),
    ).toBeInTheDocument();
  });

  it("waits for the recording list before offering to run", () => {
    stubFetch(projection);
    renderPanel({ availableFiles: [] });

    expect(
      screen.getByRole("button", { name: /waiting for the recording list/i }),
    ).toBeDisabled();
  });

  it("colours every point by the detector's score, never by a label", async () => {
    plotProps.length = 0;
    stubFetch(projection);
    renderPanel();
    await start();

    await screen.findByTestId("plot");
    const { data } = plotProps.at(-1) as { data: Record<string, any>[] };
    const [points] = data;

    expect(points.marker.color).toEqual([0.0633, 0.142884, 0.923]);
    expect(points.marker.colorscale).toEqual([
      [0, BONAFIDE_COLOR],
      [1, SPOOF_COLOR],
    ]);
    // The scale is pinned to 0..1 rather than fitted to the batch, so two
    // models' plots are read on the same colour axis.
    expect(points.marker.cmin).toBe(0);
    expect(points.marker.cmax).toBe(1);
  });

  it("requests 3 components when the 3D switch is on", async () => {
    const fetchMock = stubFetch(projection);
    renderPanel();
    await start();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    await userEvent.click(screen.getByRole("switch"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const [, init] = fetchMock.mock.calls[1] as unknown as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({
      model: "xlsr-deepfake",
      reduction_method: "pca",
      n_components: 3,
    });
  });

  it("says so when the reducer had to fall back to another method", async () => {
    stubFetch({
      ...projection,
      reduction_method: "umap",
      reduction_method_used: "pca",
    });
    renderPanel();
    await start();

    expect(
      await screen.findByText(/UMAP could not run on this data — showing PCA instead/),
    ).toBeInTheDocument();
  });

  it("reports the vector width and the reduction it used", async () => {
    stubFetch(projection);
    renderPanel();
    await start();

    expect(
      await screen.findByText(/1024-dimensional vectors reduced to 2D with PCA/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/distances on screen are not distances inside the model/),
    ).toBeInTheDocument();
  });

  it("surfaces a 503 rather than an empty plot", async () => {
    stubFetch({ detail: "AustinXiao/XLSR-Mamba-LA could not be loaded" }, { ok: false, status: 503 });
    renderPanel();
    await start();

    expect(
      await screen.findByText(/XLSR-Mamba-LA could not be loaded/),
    ).toBeInTheDocument();
  });

  it("does not show one model's points under another model's name", async () => {
    stubFetch(projection); // payload says model: xlsr-deepfake
    renderPanel({ model: "xlsr-mamba", modelLabel: "XLSR-Mamba (Model C)" });
    await userEvent.click(
      screen.getByRole("button", { name: /show the dataset in 2d \/ 3d/i }),
    );

    await waitFor(() => expect(screen.queryByTestId("plot")).not.toBeInTheDocument());
  });

  it("shows the selected recording's score, from the model and not the protocol", async () => {
    stubFetch(projection);
    renderPanel({
      selectedFile: {
        file_id: "rec_c",
        filename: "LA_E_9938640.flac",
        file_path: "LA_E_9938640.flac",
        message: "",
      },
    });
    await start();

    expect(
      await screen.findByText(/LA_E_9938640\.flac · spoof score 0\.923/),
    ).toBeInTheDocument();
  });
});
