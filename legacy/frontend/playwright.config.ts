import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright E2E test configuration for RAFTcorr.
 *
 * Expects the dev server (Flask + Vite) to be running before tests.
 * Run with: npx playwright test
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // DIC tests share server state — run sequentially
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1, // Single worker — shared Flask backend
  reporter: process.env.CI ? "github" : "html",
  timeout: 60_000, // DIC processing can take time

  use: {
    baseURL: "http://127.0.0.1:5000",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
