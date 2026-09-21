import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

/**
 * Shared setup for the deepfake component tests.
 *
 * Every panel in this feature fetches from the backend on mount, so `fetch` is
 * the seam each test drives. It is reset between tests rather than being
 * mocked per-file, which keeps a forgotten stub in one test from silently
 * feeding the next one.
 *
 * jsdom implements no layout, so the SVG/canvas views (DetCurve,
 * ScoreDistribution, EmbeddingScatter) are asserted on their structure and
 * text, never on pixel geometry.
 */
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

// ResizeObserver is not implemented in jsdom; the scatter plot measures its
// container with it.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", ResizeObserverStub);

if (!window.HTMLElement.prototype.scrollIntoView) {
  window.HTMLElement.prototype.scrollIntoView = () => {};
}

// jsdom ships no media pipeline: HTMLMediaElement.play() is unimplemented and
// returns undefined, where every real browser returns a Promise. The saliency
// strip's seek handler calls .catch() on it, so without this stub a keyboard
// seek throws an unhandled TypeError that has nothing to do with the code
// under test.
window.HTMLMediaElement.prototype.play = () => Promise.resolve();
window.HTMLMediaElement.prototype.pause = () => {};
