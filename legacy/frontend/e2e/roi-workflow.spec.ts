import { test, expect } from "@playwright/test";

test.describe("ROI Page Workflow", () => {
  test.beforeEach(async ({ page }) => {
    // Skip onboarding for all ROI tests
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
    await page.goto("/");
    // Wait for the app to fully render
    await expect(page.getByText("RAFTcorr")).toBeVisible();
  });

  test("ROI page shows empty canvas placeholder when no images loaded", async ({ page }) => {
    // The canvas area should show the empty state
    await expect(page.getByText("No images loaded")).toBeVisible();
    await expect(
      page.getByText("Set input directory and load images to begin")
    ).toBeVisible();
  });

  test("Path Settings section is visible with input controls", async ({ page }) => {
    // The Path Settings collapsible section should be open by default
    await expect(page.getByText("Path Settings")).toBeVisible();

    // Image Directory label
    await expect(page.getByText("Image Directory")).toBeVisible();

    // The directory input field should exist
    const dirInput = page.locator('input[placeholder="/path/to/images"]');
    await expect(dirInput).toBeVisible();

    // Browse button (folder search icon) should be present
    const browseBtn = page.locator('button[title="Browse"]');
    await expect(browseBtn).toBeVisible();

    // Confirm button for loading images
    await expect(page.getByRole("button", { name: /Confirm/i })).toBeVisible();
  });

  test("Natural sort checkbox is present and checked by default", async ({ page }) => {
    // Natural sort label
    await expect(page.getByText(/Natural sort/i)).toBeVisible();

    // The checkbox should be checked by default
    const checkbox = page.locator('input[type="checkbox"]').first();
    await expect(checkbox).toBeChecked();
  });

  test("Processing mode selector shows Accumulative and Incremental", async ({ page }) => {
    // Processing Mode section
    await expect(page.getByText("Processing Mode")).toBeVisible();

    // Segmented control buttons
    const accumulativeBtn = page.getByRole("button", { name: "Accumulative" });
    const incrementalBtn = page.getByRole("button", { name: "Incremental" });

    await expect(accumulativeBtn).toBeVisible();
    await expect(incrementalBtn).toBeVisible();
  });

  test("switching to Incremental mode shows additional settings", async ({ page }) => {
    // Click Incremental
    await page.getByRole("button", { name: "Incremental" }).click();

    // Incremental Settings section should appear
    await expect(page.getByText("Incremental Settings")).toBeVisible();

    // Reference Update selector should be visible
    await expect(page.getByText(/Ref\. Update/i)).toBeVisible();

    // Mask selector should be visible
    await expect(page.getByText("Mask")).toBeVisible();

    // Median Filter toggle should be visible
    await expect(page.getByText(/Median Filter/i)).toBeVisible();
  });

  test("switching back to Accumulative hides Incremental Settings", async ({ page }) => {
    // Switch to Incremental first
    await page.getByRole("button", { name: "Incremental" }).click();
    await expect(page.getByText("Incremental Settings")).toBeVisible();

    // Switch back to Accumulative
    await page.getByRole("button", { name: "Accumulative" }).click();

    // Incremental Settings should disappear
    await expect(page.getByText("Incremental Settings")).not.toBeVisible();
  });

  test("Run button is present but disabled without ROI confirmed", async ({ page }) => {
    // The Run button should exist at the bottom of the sidebar
    const runButton = page.getByRole("button", { name: /Run/i }).first();
    await expect(runButton).toBeVisible();

    // It should be disabled since no ROI is confirmed
    await expect(runButton).toBeDisabled();
  });

  test("ROI toolbar shows tool buttons", async ({ page }) => {
    // The ROI Tools label in the bottom toolbar
    await expect(page.getByText("ROI Tools")).toBeVisible();

    // Add and Cut tool groups
    await expect(page.getByRole("button", { name: /Add/i }).first()).toBeVisible();
    await expect(page.getByRole("button", { name: /Cut/i }).first()).toBeVisible();

    // Action buttons: Import, Save, Invert, Clear
    await expect(page.getByRole("button", { name: /Import/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Save/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Invert/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Clear/i })).toBeVisible();
  });

  test("ROI tools are disabled when no images are loaded", async ({ page }) => {
    // The Clear button should be disabled without images
    const clearBtn = page.getByRole("button", { name: /Clear/i });
    await expect(clearBtn).toBeDisabled();

    // Import button should also be disabled
    const importBtn = page.getByRole("button", { name: /Import/i });
    await expect(importBtn).toBeDisabled();
  });

  test("Processing Mode section is collapsible", async ({ page }) => {
    // Section should be open by default — Accumulative button visible
    await expect(page.getByRole("button", { name: "Accumulative" })).toBeVisible();

    // Click the Processing Mode header to collapse
    await page.getByText("Processing Mode").click();

    // Accumulative button should now be hidden
    await expect(page.getByRole("button", { name: "Accumulative" })).not.toBeVisible();

    // Click again to expand
    await page.getByText("Processing Mode").click();
    await expect(page.getByRole("button", { name: "Accumulative" })).toBeVisible();
  });

  test("status indicator shows Ready when idle", async ({ page }) => {
    // The RunControl at bottom of sidebar should show "Ready"
    await expect(page.getByText("Ready")).toBeVisible();

    // Frame counter should show 0 / 0
    await expect(page.getByText("Frame: 0 / 0")).toBeVisible();
  });

  test("Session Save and Load buttons are present", async ({ page }) => {
    // Session controls at top of sidebar
    await expect(page.getByRole("button", { name: /Save/i }).first()).toBeVisible();
    await expect(page.getByRole("button", { name: /Load/i }).first()).toBeVisible();
  });
});
