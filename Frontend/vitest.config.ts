import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react-swc";
import path from "path";

/**
 * One test runner for every task's frontend tests.
 *
 * `npm test` runs everything under src/. To run one task's tests only, pass its
 * folder, for example `npm run test:deepfake`.
 *
 * Shared setup lives in src/test/setup.ts. Task specific setup stays inside the
 * task's own tests, so one task's mocks can never leak into another task's.
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
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    coverage: {
      // Only takes effect with --coverage. Scoped to the deepfake feature,
      // which is where the reported coverage figure comes from.
      provider: "v8",
      include: ["src/features/deepfake/**/*.{ts,tsx}"],
      exclude: ["src/features/deepfake/__tests__/**", "src/features/deepfake/index.ts"],
      reporter: ["text-summary"],
    },
  },
});
