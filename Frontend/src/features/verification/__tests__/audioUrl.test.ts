import { describe, it, expect } from "vitest";
import { verificationAudioUrl } from "../audioUrl";
import { API_BASE } from "@/lib/api";

describe("verificationAudioUrl", () => {
  it("resolves session asset ids (asset_*) to the session-assets audio endpoint", () => {
    const url = verificationAudioUrl("asset_abc123");
    expect(url).toBe(`${API_BASE}/tasks/verification/session-assets/asset_abc123/audio`);
  });

  it("resolves custom recording ids (crec_*) to the custom-recordings audio endpoint", () => {
    const url = verificationAudioUrl("crec_xyz789");
    expect(url).toBe(`${API_BASE}/tasks/verification/custom-recordings/crec_xyz789/audio`);
  });

  it("resolves demo dataset recording ids (rec_* or standard names) to dataset recordings audio endpoint", () => {
    const url = verificationAudioUrl("rec_demo_clip_01");
    expect(url).toBe(`${API_BASE}/tasks/verification/dataset/recordings/rec_demo_clip_01/audio`);
  });

  it("safely URL-encodes special characters in recording IDs", () => {
    const url = verificationAudioUrl("asset_clip with spaces&symbols#1");
    expect(url).toBe(
      `${API_BASE}/tasks/verification/session-assets/asset_clip%20with%20spaces%26symbols%231/audio`
    );
  });
});
