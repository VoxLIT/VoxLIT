/**
 * The visitor's own clips: upload/record them, then score them with every
 * detector through every per-clip feature.
 *
 * Audio decoding is stubbed (jsdom has no Web Audio), so what is tested is the
 * flow: conversion before upload, the `up_` id reaching every endpoint, all
 * three detectors being asked, and user clips joining the voice map request.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { TaskDefinition } from "@/tasks/types";
import type { UserClip } from "../types";

vi.mock("../page/wav", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../page/wav")>()),
  convertToWav: vi.fn(async () => ({ wav: new Blob(["RIFF"], { type: "audio/wav" }), duration: 2.4 })),
}));

const { DeepfakePage } = await import("../page/DeepfakePage");
const { AllDetectors } = await import("../page/AllDetectors");
const { audioUrlFor, isUserClip } = await import("../page/api");
const { encodeWav, wavName } = await import("../page/wav");
const { detection, projection, recordings, saliency, silenceProbe } = await import("./fixtures");

const MODELS = [
  { id: "xlsr-deepfake", label: "wav2vec2 XLS-R (Model A)", available: true },
  { id: "ast-fakeaudio", label: "Audio Spectrogram Transformer (Model B)", available: true },
  { id: "xlsr-mamba", label: "XLSR-Mamba (Model C)", available: true },
];

const task: TaskDefinition = {
  id: "deepfake",
  route: "/tasks/deepfake",
  name: "Audio Deepfake Detection",
  shortDescription: "",
  status: "active",
  models: MODELS,
  defaultModel: "xlsr-deepfake",
  datasets: [{ id: "asvspoof2019-la", label: "ASVspoof 2019 LA (subset)", available: true }],
  defaultDataset: "asvspoof2019-la",
  allowCustomDatasets: false,
  capabilities: { saliency: true, attention: false, perturbation: false, resultKind: null, batchAnalysis: null },
};

const clip: UserClip = {
  recording_id: "up_0123456789abcdef",
  display_filename: "my voice.wav",
  extension: ".wav",
  size_bytes: 76_844,
  duration_seconds: 2.4,
  source: "upload",
  created_at: 1_790_000_000,
};

beforeEach(() => {
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

type Init = RequestInit | undefined;

const routeFetch = (initialClips: UserClip[] = []) => {
  const bodies: Record<string, (init: Init) => unknown> = {
    "dataset/recordings": () => recordings,
    uploads: (init) => (init?.method === "POST" ? clip : { recordings: initialClips }),
    embeddings: (init) => {
      const body = JSON.parse(String(init?.body));
      const extra = (body.extra_recording_ids ?? []).map((id: string) => ({
        recording_id: id,
        display_filename: clip.display_filename,
        spoof_probability: 0.73,
        decision: "spoof",
        uploaded: true,
      }));
      return {
        ...projection,
        model: body.model,
        recordings: [...projection.recordings, ...extra],
        coordinates: [...projection.coordinates, ...extra.map(() => [0.1, 0.2])],
      };
    },
    run: (init) => ({ ...detection, ...JSON.parse(String(init?.body)) }),
    "silence-probe": (init) => ({ ...silenceProbe, ...JSON.parse(String(init?.body)) }),
    saliency: (init) => ({ ...saliency, ...JSON.parse(String(init?.body)) }),
  };
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const key = Object.keys(bodies).find((path) => url.endsWith(`/tasks/deepfake/${path}`));
    return {
      ok: key !== undefined,
      status: key ? 200 : 404,
      json: async () => (key ? bodies[key](init) : { detail: "not found" }),
      arrayBuffer: async () => new ArrayBuffer(8),
    };
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
};

const bodiesFor = (fetchMock: ReturnType<typeof routeFetch>, path: string) =>
  fetchMock.mock.calls
    .filter(([url]) => String(url).endsWith(`/tasks/deepfake/${path}`))
    .map(([, init]) => (init as RequestInit | undefined)?.body);

describe("user clip helpers", () => {
  it("routes user clip audio to the uploads endpoint", () => {
    expect(isUserClip(clip.recording_id)).toBe(true);
    expect(audioUrlFor(clip.recording_id)).toMatch(/\/tasks\/deepfake\/uploads\/up_0123456789abcdef\/audio$/);
    expect(audioUrlFor("rec_abc")).toMatch(/\/dataset\/recordings\/rec_abc\/audio$/);
  });

  it("writes a valid 16 kHz mono 16-bit WAV header", async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -1, 1]));
    // jsdom's Blob has no arrayBuffer(); FileReader works everywhere.
    const bytes = await new Promise<ArrayBuffer>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.readAsArrayBuffer(blob);
    });
    const view = new DataView(bytes);
    const text = (offset: number) => String.fromCharCode(...[0, 1, 2, 3].map((i) => view.getUint8(offset + i)));
    expect(text(0)).toBe("RIFF");
    expect(text(8)).toBe("WAVE");
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16_000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(8);
    expect(view.getInt16(48, true)).toBe(-32768);
  });

  it("renames converted files to .wav", () => {
    expect(wavName("voice note.m4a")).toBe("voice note.wav");
    expect(wavName("recording.webm")).toBe("recording.wav");
  });
});

describe("Your voice section", () => {
  it("uploads a converted clip, lists it, and studies it", async () => {
    const fetchMock = routeFetch();
    render(
      <MemoryRouter>
        <DeepfakePage task={task} />
      </MemoryRouter>,
    );
    const section = document.getElementById("your-voice")!;
    const input = within(section).getByLabelText(/upload an audio clip/i) as HTMLInputElement;
    await userEvent.upload(input, new File(["abc"], "voice note.m4a", { type: "audio/mp4" }));

    expect(await within(section).findByText("my voice.wav")).toBeInTheDocument();
    const [upload] = fetchMock.mock.calls.filter(
      ([url, init]) => String(url).endsWith("/uploads") && (init as RequestInit)?.method === "POST",
    );
    const form = (upload[1] as RequestInit).body as FormData;
    expect((form.get("file") as File).name).toBe("voice note.wav");
    expect(form.get("source")).toBe("upload");

    // The new clip becomes the one under study.
    const verdict = screen.getByRole("complementary", { name: "Verdict" });
    expect(within(verdict).getByText("my voice.wav")).toBeInTheDocument();
  });

  it("puts the visitor's clips on the voice map, ringed", async () => {
    const fetchMock = routeFetch([clip]);
    render(
      <MemoryRouter>
        <DeepfakePage task={task} />
      </MemoryRouter>,
    );
    expect(await within(document.getElementById("your-voice")!).findByText("my voice.wav")).toBeInTheDocument();
    await userEvent.click(await screen.findByRole("button", { name: /map the voices/i }));
    await waitFor(() => expect(screen.getAllByTestId("user-clip-marker")).toHaveLength(1));

    const [body] = bodiesFor(fetchMock, "embeddings").slice(-1);
    expect(JSON.parse(String(body)).extra_recording_ids).toEqual([clip.recording_id]);
  });
});

describe("AllDetectors", () => {
  it("runs every feature on every detector for a user clip, one at a time", async () => {
    const fetchMock = routeFetch();
    render(<AllDetectors models={MODELS} recording={clip} />);

    expect(screen.getByText("Your clip")).toBeInTheDocument();
    expect(screen.getAllByTestId("detector-card")).toHaveLength(3);
    await userEvent.click(screen.getByRole("button", { name: /run everything on all 3/i }));

    await waitFor(() => expect(screen.getAllByText("Sounds real")).toHaveLength(3));
    await screen.findByText(/they all agree/i);

    for (const path of ["run", "silence-probe", "saliency"]) {
      const models = bodiesFor(fetchMock, path).map((body) => JSON.parse(String(body)));
      expect(models.map((body) => body.model).sort()).toEqual(MODELS.map((model) => model.id).sort());
      expect(models.every((body) => body.recording_id === clip.recording_id)).toBe(true);
    }
    // Detector by detector: all of Model A's features before Model B's.
    const order = fetchMock.mock.calls
      .map(([, init]) => (init as RequestInit | undefined)?.body)
      .filter(Boolean)
      .map((body) => JSON.parse(String(body)).model);
    expect(order).toEqual(MODELS.flatMap((model) => [model.id, model.id, model.id]));
    expect(screen.queryByRole("button", { name: /run everything/i })).toBeNull();
  });

  it("stops asking a detector that fails, and carries on with the others", async () => {
    const fetchMock = routeFetch();
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
      if (init?.body && JSON.parse(String(init.body)).model === "ast-fakeaudio") {
        return { ok: false, status: 503, json: async () => ({ detail: "gated repo" }), arrayBuffer: async () => new ArrayBuffer(0) };
      }
      return base(url, init);
    });
    render(<AllDetectors models={MODELS} recording={clip} />);
    await userEvent.click(screen.getByRole("button", { name: /run everything on all 3/i }));

    const [, cardB] = screen.getAllByTestId("detector-card");
    expect(await within(cardB).findByText("gated repo")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("Sounds real")).toHaveLength(2));
    const askedB = fetchMock.mock.calls.filter(([, init]) => (init as RequestInit)?.body && JSON.parse(String((init as RequestInit).body)).model === "ast-fakeaudio");
    expect(askedB).toHaveLength(1);
    expect(await screen.findByText(/2 of 2|0 of 2/)).toBeInTheDocument();
  });

  it("runs a single cell on its own", async () => {
    const fetchMock = routeFetch();
    render(<AllDetectors models={MODELS} recording={clip} />);
    const [, cardB] = screen.getAllByTestId("detector-card");
    await userEvent.click(within(cardB).getByRole("button", { name: /run the silence test/i }));

    expect(await within(cardB).findByText("Whole clip")).toBeInTheDocument();
    const [body] = bodiesFor(fetchMock, "silence-probe");
    expect(JSON.parse(String(body))).toEqual({ model: "ast-fakeaudio", recording_id: clip.recording_id });
  });
});
