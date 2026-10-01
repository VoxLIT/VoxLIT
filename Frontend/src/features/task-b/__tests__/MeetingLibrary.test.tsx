import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MeetingLibrary } from "../page/MeetingLibrary";
import type { RecordingInfo } from "../types";

const recording: RecordingInfo = {
  recording_id: "rec_demo",
  display_filename: "ES2004a.wav",
  extension: "wav",
  size_bytes: 1024,
};

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
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const renderLibrary = (recordings: RecordingInfo[], isLoading: boolean) =>
  render(
    <MeetingLibrary
      recordings={recordings}
      uploads={[]}
      selectedId=""
      onSelect={vi.fn()}
      onUpload={vi.fn()}
      isUploading={false}
      disabled={false}
      isLoading={isLoading}
    />,
  );

describe("MeetingLibrary", () => {
  it("says so when there are no recordings or uploads", () => {
    renderLibrary([], false);
    expect(screen.getByText("No recordings available")).toBeInTheDocument();
    // The uploader stays usable — it's the way out of the empty state.
    expect(screen.getByRole("button", { name: "Upload your own recording" })).toBeInTheDocument();
  });

  it("shows loading, not the empty message, before the listing returns", () => {
    renderLibrary([], true);
    expect(screen.getByText("Loading recordings…")).toBeInTheDocument();
    expect(screen.queryByText("No recordings available")).not.toBeInTheDocument();
  });

  it("lists recordings without an empty message", () => {
    renderLibrary([recording], false);
    expect(screen.getByText("ES2004a.wav")).toBeInTheDocument();
    expect(screen.queryByText("No recordings available")).not.toBeInTheDocument();
  });
});
