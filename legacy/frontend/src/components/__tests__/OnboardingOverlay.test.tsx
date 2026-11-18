import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { OnboardingOverlay, resetOnboarding } from "@/components/shared/OnboardingOverlay";

// ---- localStorage mock ------------------------------------------------
const mockStorage = new Map<string, string>();

beforeEach(() => {
  mockStorage.clear();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => mockStorage.get(key) ?? null,
    setItem: (key: string, value: string) => mockStorage.set(key, value),
    removeItem: (key: string) => mockStorage.delete(key),
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---- Tests ------------------------------------------------------------

describe("OnboardingOverlay", () => {
  it("renders when localStorage has no value", () => {
    render(<OnboardingOverlay />);
    expect(screen.getByText("Welcome to RAFTcorr")).toBeInTheDocument();
  });

  it("does NOT render when localStorage has 'true'", () => {
    mockStorage.set("dic_onboarding_done", "true");
    const { container } = render(<OnboardingOverlay />);
    expect(container.innerHTML).toBe("");
  });

  it("shows step 1 content initially ('Load Images')", () => {
    render(<OnboardingOverlay />);
    expect(screen.getByText("1. Load Images")).toBeInTheDocument();
    expect(
      screen.getByText(/Click 'Browse' in the sidebar/)
    ).toBeInTheDocument();
  });

  it("Next button advances to step 2 ('Draw ROI')", async () => {
    const user = userEvent.setup();
    render(<OnboardingOverlay />);

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("2. Draw ROI")).toBeInTheDocument();
  });

  it("Back button goes to previous step", async () => {
    const user = userEvent.setup();
    render(<OnboardingOverlay />);

    // Advance to step 2
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("2. Draw ROI")).toBeInTheDocument();

    // Go back to step 1
    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(screen.getByText("1. Load Images")).toBeInTheDocument();
  });

  it("Skip button dismisses and sets localStorage", async () => {
    const user = userEvent.setup();
    const { container } = render(<OnboardingOverlay />);

    await user.click(screen.getByRole("button", { name: "Skip" }));
    expect(container.innerHTML).toBe("");
    expect(mockStorage.get("dic_onboarding_done")).toBe("true");
  });

  it("'Get Started' on last step dismisses", async () => {
    const user = userEvent.setup();
    const { container } = render(<OnboardingOverlay />);

    // Navigate to the last step (step 4)
    await user.click(screen.getByRole("button", { name: "Next" })); // → step 2
    await user.click(screen.getByRole("button", { name: "Next" })); // → step 3
    await user.click(screen.getByRole("button", { name: "Next" })); // → step 4

    expect(screen.getByText("4. Analyze Results")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Get Started" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Get Started" }));
    expect(container.innerHTML).toBe("");
    expect(mockStorage.get("dic_onboarding_done")).toBe("true");
  });

  it("step dots are clickable and navigate", async () => {
    const user = userEvent.setup();
    render(<OnboardingOverlay />);

    // There are 4 step dot buttons (small, no accessible name)
    // They appear as buttons in the step indicator area
    const allButtons = screen.getAllByRole("button");
    // Dot buttons are the ones that are not Skip/Next/Back/Get Started
    const dotButtons = allButtons.filter(
      (btn) =>
        !["Skip", "Next", "Back", "Get Started"].includes(
          btn.textContent?.trim() ?? ""
        )
    );
    expect(dotButtons).toHaveLength(4);

    // Click the 3rd dot (index 2) to jump to step 3
    await user.click(dotButtons[2]);
    expect(screen.getByText("3. Run DIC Processing")).toBeInTheDocument();

    // Click the 1st dot to jump back to step 1
    await user.click(dotButtons[0]);
    expect(screen.getByText("1. Load Images")).toBeInTheDocument();
  });

  it("resetOnboarding() removes localStorage key", () => {
    mockStorage.set("dic_onboarding_done", "true");
    expect(mockStorage.get("dic_onboarding_done")).toBe("true");

    resetOnboarding();
    expect(mockStorage.get("dic_onboarding_done")).toBeUndefined();
  });
});
