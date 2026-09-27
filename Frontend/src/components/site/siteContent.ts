import { TaskId } from "@/tasks/types";

export const GITHUB_URL = "https://github.com/VoxLIT/VoxLIT";
export const LIT_URL = "https://pair-code.github.io/lit/";

/** One-line hook for each task on the home page. */
export const TASK_SHOWCASE: Record<TaskId, { tagline: string }> = {
  transcription: {
    tagline: "See which sounds Whisper relied on for every word.",
  },
  emotion: {
    tagline: "Trace an emotion prediction back to the moments that drove it.",
  },
  verification: {
    tagline: "Same speaker or not? Inspect the embeddings behind the call.",
  },
  "task-b": {
    tagline: "Who spoke when, with the uncertainty drawn on the timeline.",
  },
  deepfake: {
    tagline: "Real or synthetic? Three detectors, three different ways to fail.",
  },
};
