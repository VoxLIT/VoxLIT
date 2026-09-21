/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/**
 * Test runner for the Audio Deepfake Detection feature — OWNER: Chanupa Gurusinghe.
 *
 * `include` is scoped to src/features/deepfake on purpose. This task owns its
 * own feature folder and nothing else, and the repository's other frontend
 * test file (src/tests/ui-components.test.tsx) imports `@jest/globals`, which
 * is not a dependency here — pulling it into this run would fail the suite on
 * somebody else's file. When the other tasks adopt Vitest, widening this glob
 * is a one-line change.
 *
 * The `@` alias mirrors vite.config.ts so components resolve their shared
 * imports (`@/lib/api`, `@/components/ui/...`) exactly as they do in the app.
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/features/deepfake/__tests__/setup.ts"],
    include: ["src/features/deepfake/**/*.test.{ts,tsx}"],
    css: false,
    coverage: {
      provider: "v8",
      include: ["src/features/deepfake/**/*.{ts,tsx}"],
      exclude: ["src/features/deepfake/__tests__/**", "src/features/deepfake/index.ts"],
      reporter: ["text-summary"],
    },
  },
});
