import { test, expect } from "@playwright/test";

test.describe("API Responses & Error Handling", () => {
  test("GET /api/processing/status returns valid JSON", async ({ request }) => {
    const response = await request.get("/api/processing/status");
    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    // The status endpoint should return an object (not null or array)
    expect(typeof body).toBe("object");
    expect(body).not.toBeNull();
  });

  test("GET /api/export/images/status returns valid JSON when no export active", async ({
    request,
  }) => {
    const response = await request.get("/api/export/images/status");
    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(typeof body).toBe("object");
    expect(body).not.toBeNull();
  });

  test("GET /api/strain/status returns valid JSON", async ({ request }) => {
    const response = await request.get("/api/strain/status");
    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(typeof body).toBe("object");
    expect(body).not.toBeNull();
  });

  test("non-existent /api/ endpoint returns JSON 404 (not HTML)", async ({
    request,
  }) => {
    const response = await request.get("/api/this-route-does-not-exist");
    expect(response.status()).toBe(404);

    const contentType = response.headers()["content-type"] ?? "";
    // Should be JSON, not text/html (the SPA catch-all guard)
    expect(contentType).toContain("json");

    const body = await response.json();
    expect(body).toHaveProperty("error");
  });
});

test.describe("SPA Routing & Fallback", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
  });

  test("navigating to unknown client route shows the SPA (not a 404 page)", async ({
    page,
  }) => {
    await page.goto("/some/nonexistent/route");

    // The SPA shell (header with RAFTcorr) should still render
    await expect(page.getByText("RAFTcorr")).toBeVisible();

    // Since this is not a known route, React Router will render nothing in
    // the Outlet, but the layout (header + stepper) should still be visible
    await expect(
      page.getByRole("button", { name: /ROI Selection/i })
    ).toBeVisible();
  });
});

test.describe("UI Error States", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem("dic_onboarding_done", "true");
    });
  });

  test("loading images with empty path shows error toast", async ({ page }) => {
    await page.goto("/");

    // Click Confirm without entering any path
    await page.getByRole("button", { name: /Confirm/i }).click();

    // An inline error message should appear in the sidebar
    await expect(page.getByText("Please enter a directory path")).toBeVisible();
  });

  test("loading images with non-existent path shows error feedback", async ({
    page,
  }) => {
    await page.goto("/");

    // Type a path that does not exist on disk
    const dirInput = page.locator('input[placeholder="/path/to/images"]');
    await dirInput.fill("/nonexistent/test/path/12345");

    // Click Confirm
    await page.getByRole("button", { name: /Confirm/i }).click();

    // Wait for the error to appear — either inline error text or a toast
    // The exact message depends on the backend, but some error should surface
    await expect(
      page
        .locator("text=/Failed to load|not found|does not exist|No images|error/i")
        .first()
    ).toBeVisible({ timeout: 10_000 });
  });

  test("Displacement stepper button is disabled when no ROI is confirmed", async ({
    page,
  }) => {
    await page.goto("/");

    const displacementBtn = page.getByRole("button", {
      name: /Displacement/i,
    });
    await expect(displacementBtn).toBeVisible();
    await expect(displacementBtn).toBeDisabled();
  });

  test("Post-Processing stepper button is disabled when no results exist", async ({
    page,
  }) => {
    await page.goto("/");

    const postBtn = page.getByRole("button", {
      name: /Post-Processing/i,
    });
    await expect(postBtn).toBeVisible();
    await expect(postBtn).toBeDisabled();
  });
});
