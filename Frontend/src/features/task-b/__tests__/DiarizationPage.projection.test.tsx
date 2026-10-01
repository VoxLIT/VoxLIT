/**
 * A failed /projection must not take the run down with it: the timeline
 * still shows, no "run failed" banner appears, and the segment map says why
 * it could not be drawn.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { TaskDefinition } from "@/tasks/types";
import { DiarizationPage } from "../page/DiarizationPage";
import { projection, projection3d, result } from "./fixtures";

// jsdom has no WebGL; the page only needs to know the 3D map was handed its points.
vi.mock("../page/SegmentMap3D", () => ({
  default: ({ points }: { points: unknown[] }) => <div data-testid="segment-map-3d" data-count={points.length} />,
}));

// The shared registry imports every task's components (and Plotly with them),
// which jsdom cannot load, so the page gets the task-b entry's shape here.
const task: TaskDefinition = {
  id: "task-b",
  route: "/tasks/task-b",
  name: "Speaker Diarization",
  shortDescription: "",
  status: "active",
  models: [{ id: "pyannote-3.1", label: "pyannote 3.1", available: true }],
  defaultModel: "pyannote-3.1",
  datasets: [],
  defaultDataset: null,
  allowCustomDatasets: false,
  capabilities: { saliency: false, attention: false, perturbation: false, resultKind: null, batchAnalysis: null },
};

const recordings = [{ recording_id: "rec_demo", display_filename: "ES2004a.wav", extension: "wav", size_bytes: 1024 }];

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
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, value),
    removeItem: (key: string) => void store.delete(key),
  });
});

/** Answer each task-b endpoint; `/projection` answers with `projectionReply`. */
const routeFetch = (projectionReply: { ok: boolean; status: number; json: () => Promise<unknown> }) => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const reply = (ok: boolean, status: number, body: unknown) => ({ ok, status, json: async () => body });
      if (url.includes("/tasks/task-b/projection") && url.includes("dims=3")) {
        return reply(true, 200, projection3d);
      }
      if (url.includes("/tasks/task-b/projection")) {
        return projectionReply;
      }
      if (url.endsWith("/tasks/task-b/dataset/recordings")) return reply(true, 200, { recordings });
      if (url.endsWith("/tasks/task-b/uploads")) return reply(true, 200, { uploads: [] });
      if (url.endsWith("/tasks/task-b/models")) return reply(true, 200, { models: [] });
      if (url.endsWith("/tasks/task-b/run")) return reply(true, 200, result);
      return reply(false, 404, { detail: "not found" });
    }),
  );
};

const runDemo = async () => {
  render(
    <MemoryRouter>
      <DiarizationPage task={task} />
    </MemoryRouter>,
  );
  const library = document.getElementById("meetings")!;
  await within(library).findByText("ES2004a.wav");
  await userEvent.click(within(library).getByRole("button", { name: "Select" }));
  await userEvent.click(screen.getByRole("button", { name: /find the speakers/i }));
};

describe("DiarizationPage — projection failure", () => {
  it("still shows the timeline when /projection fails", async () => {
    routeFetch({
      ok: false,
      status: 422,
      json: async () => ({ detail: "Not enough embeddable segments for a 2D projection." }),
    });
    await runDemo();

    expect(await screen.findByText(/couldn.t lay out the map/i)).toHaveTextContent("Not enough embeddable segments");
    expect(screen.getByText("Speaker timeline")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("still shows the timeline when /projection returns a non-JSON error", async () => {
    routeFetch({ ok: false, status: 500, json: () => Promise.reject(new SyntaxError("Unexpected token <")) });
    await runDemo();

    expect(await screen.findByText(/couldn.t lay out the map/i)).toHaveTextContent("Projection failed (500)");
    expect(screen.getByText("Speaker timeline")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("draws the map when /projection succeeds", async () => {
    routeFetch({ ok: true, status: 200, json: async () => projection });
    await runDemo();
    expect(await screen.findAllByTestId("map-point")).toHaveLength(projection.points.length);
  });

  it("fetches the 3D layout only when 3D is picked, then draws it", async () => {
    routeFetch({ ok: true, status: 200, json: async () => projection });
    await runDemo();
    await screen.findAllByTestId("map-point");
    const projectionCalls = () =>
      vi.mocked(fetch).mock.calls.map(([url]) => String(url)).filter((url) => url.includes("/projection"));
    expect(projectionCalls()).toEqual([expect.stringContaining("dims=2")]);

    await userEvent.click(screen.getByRole("radio", { name: /3D/ }));

    expect(await screen.findByTestId("segment-map-3d")).toHaveAttribute("data-count", String(projection3d.points.length));
    expect(projectionCalls()).toHaveLength(2);
    expect(projectionCalls()[1]).toContain("dims=3");
  });
});
