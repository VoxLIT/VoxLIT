/**
 * DeepfakePage — the task's own full-page layout.
 *
 * Covers what a first-time visitor does (pick, guess, ask), the hover card
 * that replaced the Datapoint Editor, the rule that a selected point is shown
 * by SIZE and never by a new colour, second-level details staying hidden until
 * asked for, and the threshold slider in the detector report.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

const { DeepfakePage } = await import("../page/DeepfakePage");
const { scoreColor } = await import("../page/palette");
const { detection, evaluation, projection, recordings } = await import("./fixtures");
import type { TaskDefinition } from "@/tasks/types";

// The shared registry imports every task's components (and Plotly with them),
// which jsdom cannot load, so the page gets the deepfake entry's shape here.
const task: TaskDefinition = {
  id: "deepfake",
  route: "/tasks/deepfake",
  name: "Audio Deepfake Detection",
  shortDescription: "",
  status: "active",
  models: [
    { id: "xlsr-deepfake", label: "wav2vec2 XLS-R (Model A)", available: true },
    { id: "ast-fakeaudio", label: "Audio Spectrogram Transformer (Model B)", available: true },
  ],
  defaultModel: "xlsr-deepfake",
  datasets: [{ id: "asvspoof2019-la", label: "ASVspoof 2019 LA (subset)", available: true }],
  defaultDataset: "asvspoof2019-la",
  allowCustomDatasets: false,
  capabilities: { saliency: true, attention: false, perturbation: false, resultKind: null, batchAnalysis: null },
};

// fixtures.ts unstubs globals after every test, so this is re-stubbed per test.
beforeEach(() => {
  // motion's whileInView needs it; jsdom has none. Report everything as visible.
  class Observer {
    constructor(private callback: IntersectionObserverCallback) {}
    observe(target: Element) {
      this.callback([{ isIntersecting: true, target } as IntersectionObserverEntry], this as never);
    }
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  }
  vi.stubGlobal("IntersectionObserver", Observer);
  window.scrollTo = () => {};
  Element.prototype.scrollIntoView = () => {};
});

/** Answer each deepfake endpoint with its fixture. */
const routeFetch = (overrides: Record<string, unknown> = {}) => {
  const bodies: Record<string, unknown> = {
    "dataset/recordings": recordings,
    embeddings: projection,
    run: detection,
    scores: evaluation,
    ...overrides,
  };
  const fetchMock = vi.fn(async (url: string, _init?: RequestInit) => {
    const key = Object.keys(bodies).find((path) => url.endsWith(`/tasks/deepfake/${path}`));
    return {
      ok: key !== undefined,
      status: key ? 200 : 404,
      json: async () => (key ? bodies[key] : { detail: "not found" }),
      arrayBuffer: async () => new ArrayBuffer(8),
    };
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const renderPage = () =>
  render(
    <MemoryRouter>
      <DeepfakePage task={task} />
    </MemoryRouter>,
  );

const drawMap = async () => {
  await userEvent.click(await screen.findByRole("button", { name: /map the voices/i }));
  await waitFor(() => expect(screen.getAllByTestId("map-point")).toHaveLength(projection.recordings.length));
};

const pointFor = (recordingId: string) =>
  screen.getAllByTestId("map-point").find((node) => node.getAttribute("data-recording-id") === recordingId)!;

describe("DeepfakePage", () => {
  it("opens on the study and its three-step protocol, not on a control panel", async () => {
    routeFetch();
    renderPage();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/audio deepfake detection/i);
    expect(screen.getByText(/real voice,/i)).toHaveTextContent(/or a machine\?/i);
    for (const step of [/1\. Listen/, /2\. Guess/, /3\. Compare/]) {
      expect(screen.getByText(step)).toBeInTheDocument();
    }
    expect(screen.getByRole("button", { name: /start listening/i })).toBeInTheDocument();
  });

  it("lays out every clip in the library, with no inner scroll box", async () => {
    routeFetch();
    renderPage();
    const library = document.getElementById("library")!;
    for (const recording of recordings.recordings) {
      expect(await within(library).findByText(recording.display_filename)).toBeInTheDocument();
    }
    expect(library.querySelector(".overflow-y-auto, .overflow-auto")).toBeNull();
  });

  it("opens a hover card for a point and selects it from there", async () => {
    routeFetch();
    renderPage();
    await drawMap();
    const target = projection.recordings[0];

    fireEvent.mouseEnter(pointFor(target.recording_id).querySelector("circle.cursor-pointer")!);
    const card = await screen.findByRole("dialog", { name: new RegExp(target.display_filename) });
    expect(within(card).getByText(target.spoof_probability.toFixed(3))).toBeInTheDocument();
    expect(within(card).getByText("3.42s")).toBeInTheDocument();

    await userEvent.click(within(card).getByRole("button", { name: /study this/i }));
    expect(pointFor(target.recording_id)).toHaveAttribute("data-selected", "true");
    expect(within(screen.getByRole("complementary", { name: "Verdict" })).getByText(target.display_filename)).toBeInTheDocument();
  });

  it("marks the selected point by size and ripples, keeping its score colour", async () => {
    routeFetch();
    renderPage();
    await drawMap();
    const target = projection.recordings[2];
    await userEvent.click(pointFor(target.recording_id).querySelector("circle.cursor-pointer")!);

    const point = pointFor(target.recording_id);
    expect(point).toHaveAttribute("data-selected", "true");
    expect(point.querySelectorAll(".df-ripple")).toHaveLength(2);
    const fills = [...point.querySelectorAll("circle")].map((circle) => circle.getAttribute("fill"));
    expect(fills).toContain(scoreColor(target.spoof_probability));
    expect(fills).not.toContain("#FFD700");
  });

  it("lets a visitor guess, then compares the guess with the detector — not with a label", async () => {
    const fetchMock = routeFetch();
    renderPage();
    const library = document.getElementById("library")!;
    await userEvent.click(await within(library).findByText(recordings.recordings[0].display_filename));

    const verdict = screen.getByRole("complementary", { name: "Verdict" });
    await userEvent.click(within(verdict).getByRole("radio", { name: /machine/i }));
    await userEvent.click(within(verdict).getByRole("button", { name: /reveal the detector/i }));

    expect(await within(verdict).findByText(/you and the detector disagree/i)).toBeInTheDocument();
    expect(within(verdict).getByText("Sounds real")).toBeInTheDocument();
    const runCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/run"));
    expect(JSON.parse(String((runCall![1] as RequestInit).body))).toEqual({
      model: "xlsr-deepfake",
      recording_id: recordings.recordings[0].recording_id,
    });
  });

  it("keeps the numbers one click deeper", async () => {
    routeFetch();
    renderPage();
    const library = document.getElementById("library")!;
    await userEvent.click(await within(library).findByText(recordings.recordings[0].display_filename));
    const verdict = screen.getByRole("complementary", { name: "Verdict" });
    await userEvent.click(within(verdict).getByRole("button", { name: /ask the detector/i }));
    await within(verdict).findByText("Sounds real");

    expect(within(verdict).queryByText("[1.42, -0.37]")).toBeNull();
    await userEvent.click(within(verdict).getByRole("button", { name: /technical details/i }));
    expect(await within(verdict).findByText("[1.42, -0.37]")).toBeInTheDocument();
  });

  it("turns the evaluation into a threshold you can drag", async () => {
    routeFetch();
    renderPage();
    const report = document.getElementById("report")!;
    await waitFor(() => expect(within(report).getByRole("button", { name: /test wav2vec2/i })).toBeEnabled());
    await userEvent.click(within(report).getByRole("button", { name: /test wav2vec2/i }));

    const slider = await within(report).findByRole("slider", { name: /decision threshold/i });
    expect(slider).toHaveValue(String(evaluation.operating_point.threshold));
    // At the shipped 0.5 cut, 2% of 100 fakes get through.
    expect(within(report).getByText(/fakes slip through/i).parentElement).toHaveTextContent(/of 100/);

    fireEvent.change(slider, { target: { value: "0.95" } });
    expect(slider).toHaveValue("0.95");
    await userEvent.click(within(report).getByRole("button", { name: /τ\s*EER/i }));
    expect(slider).toHaveValue(String(evaluation.eer_threshold));
  });

  it("sorts the dataset table by the detector's score, highest first", async () => {
    routeFetch();
    renderPage();
    await drawMap();
    const library = document.getElementById("library")!;

    await userEvent.click(within(library).getByRole("button", { name: /spoof score/i }));
    const names = within(library)
      .getAllByRole("button", { pressed: false })
      .map((button) => button.textContent ?? "")
      .filter((text) => text.endsWith(".flac"));
    const expected = [...projection.recordings]
      .sort((a, b) => b.spoof_probability - a.spoof_probability)
      .map((recording) => recording.display_filename)
      .filter((filename) => recordings.recordings.some((row) => row.display_filename === filename));
    expect(names).toEqual(expected);
  });

  it("never renders a per-clip ground-truth label", async () => {
    routeFetch();
    renderPage();
    await drawMap();
    const library = document.getElementById("library")!;
    expect(library.textContent).not.toMatch(/bona ?fide|\bspoof\b|\bA[01]\d\b/i);
  });
});

/** Like routeFetch, but the /run reply waits until the returned function is called. */
const holdRun = () => {
  const fetchMock = routeFetch();
  const answer = fetchMock.getMockImplementation()!;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => (release = resolve));
  fetchMock.mockImplementation(async (url, init) => {
    if (String(url).endsWith("/run")) await gate;
    return answer(url, init);
  });
  return async () => {
    await act(async () => {
      release();
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
  };
};

describe("DeepfakePage review fixes", () => {
  it("drops a detection that comes back after the visitor picked another clip", async () => {
    const release = holdRun();
    renderPage();
    const library = document.getElementById("library")!;
    await userEvent.click(await within(library).findByText(recordings.recordings[0].display_filename));
    const verdict = screen.getByRole("complementary", { name: "Verdict" });
    await userEvent.click(within(verdict).getByRole("button", { name: /ask the detector/i }));

    // while the first clip is still being listened to, choose another
    await userEvent.click(await within(library).findByText(recordings.recordings[1].display_filename));
    await release();

    expect(within(verdict).queryByText("Sounds real")).toBeNull();
    expect(within(verdict).getByRole("button", { name: /ask the detector/i })).toBeEnabled();
  });

  it("lets a keyboard user reach, preview and select a map point", async () => {
    routeFetch();
    renderPage();
    await drawMap();
    const target = projection.recordings[1];
    const point = pointFor(target.recording_id);
    const hit = within(point).getByRole("button", { name: new RegExp(`${target.display_filename}.*detector score`) });
    expect(hit).toHaveAttribute("tabindex", "0");

    fireEvent.focus(hit);
    expect(await screen.findByRole("dialog", { name: new RegExp(target.display_filename) })).toBeInTheDocument();

    fireEvent.keyDown(hit, { key: "Enter" });
    expect(point).toHaveAttribute("data-selected", "true");
    expect(hit).toHaveAttribute("aria-pressed", "true");
  });
});
