import { test, expect } from "@playwright/test";

test.describe("App Loading & Navigation", () => {
  test.beforeEach(async ({ page }) => {
    // Clear onboarding flag so each test starts fresh unless stated otherwise
    await page.addInitScript(() => {
      localStorage.removeItem("dic_onboarding_done");
    });
  });

  test("homepage loads and shows RAFTcorr branding", async ({ page }) => {
    // Dismiss onboarding for this test so we can inspect the page beneath
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
    await page.goto("/");

    // Header should contain the app name and version badge
    const header = page.locator("header");
    await expect(header).toBeVisible();
    await expect(header.getByText("RAFTcorr")).toBeVisible();
    await expect(header.getByText("v2.0")).toBeVisible();
  });

  test("sidebar is visible with workflow steps", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
    await page.goto("/");

    // Sidebar (aside element) should be present
    const sidebar = page.locator("aside");
    await expect(sidebar).toBeVisible();

    // Workflow stepper in the header should show all three steps
    await expect(page.getByRole("button", { name: /ROI Selection/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Displacement/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Post-Processing/i })).toBeVisible();
  });

  test("navigation to /displacement shows displacement page", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
    await page.goto("/displacement");

    // The displacement page should render the sidebar with visualization settings
    const sidebar = page.locator("aside");
    await expect(sidebar).toBeVisible();

    // The Displacement stepper button should be visible (even if disabled — it
    // still renders). The page content area should exist.
    await expect(page.getByRole("button", { name: /Displacement/i })).toBeVisible();
  });

  test("navigation to /post-processing shows post-processing page", async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
    await page.goto("/post-processing");

    // Post-processing page should have its sidebar controls
    const sidebar = page.locator("aside");
    await expect(sidebar).toBeVisible();

    await expect(page.getByRole("button", { name: /Post-Processing/i })).toBeVisible();
  });

  test("onboarding overlay appears on first visit", async ({ page }) => {
    // localStorage is cleared in beforeEach, so onboarding should appear
    await page.goto("/");

    // The overlay should contain the welcome heading
    const welcomeHeading = page.getByText("Welcome to RAFTcorr");
    await expect(welcomeHeading).toBeVisible();

    // The first step should be visible
    await expect(page.getByText("1. Load Images")).toBeVisible();

    // Skip and Next buttons should be present
    await expect(page.getByRole("button", { name: /Skip/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Next/i })).toBeVisible();
  });

  test("onboarding can be dismissed via Skip button", async ({ page }) => {
    await page.goto("/");

    // Onboarding should be visible
    await expect(page.getByText("Welcome to RAFTcorr")).toBeVisible();

    // Click Skip
    await page.getByRole("button", { name: /Skip/i }).click();

    // Overlay should disappear
    await expect(page.getByText("Welcome to RAFTcorr")).not.toBeVisible();

    // The main app content (ROI page) should now be accessible
    await expect(page.getByText("No images loaded")).toBeVisible();
  });

  test("after dismissal, refresh does not show onboarding again", async ({ page }) => {
    await page.goto("/");

    // Dismiss onboarding
    await expect(page.getByText("Welcome to RAFTcorr")).toBeVisible();
    await page.getByRole("button", { name: /Skip/i }).click();
    await expect(page.getByText("Welcome to RAFTcorr")).not.toBeVisible();

    // Reload the page — onboarding should stay dismissed
    await page.reload();
    await page.waitForLoadState("domcontentloaded");

    // Give the app a moment to hydrate
    await expect(page.getByText("RAFTcorr")).toBeVisible();

    // Onboarding overlay should NOT appear
    await expect(page.getByText("Welcome to RAFTcorr")).not.toBeVisible();
  });

  test("onboarding can be navigated step by step", async ({ page }) => {
    await page.goto("/");

    // Step 1 is shown initially
    await expect(page.getByText("1. Load Images")).toBeVisible();

    // Click Next to advance through all steps
    await page.getByRole("button", { name: /Next/i }).click();
    await expect(page.getByText("2. Draw ROI")).toBeVisible();

    await page.getByRole("button", { name: /Next/i }).click();
    await expect(page.getByText("3. Run DIC Processing")).toBeVisible();

    await page.getByRole("button", { name: /Next/i }).click();
    await expect(page.getByText("4. Analyze Results")).toBeVisible();

    // On the last step, the button text should be "Get Started"
    await expect(page.getByRole("button", { name: /Get Started/i })).toBeVisible();

    // Click "Get Started" to dismiss
    await page.getByRole("button", { name: /Get Started/i }).click();
    await expect(page.getByText("Welcome to RAFTcorr")).not.toBeVisible();
  });
});
