import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrainControls } from "@/components/postprocessing/StrainControls";
import { useAppStore } from "@/stores/appStore";
import { calculateStrain } from "@/api/strain";

// ---- Mocks ------------------------------------------------------------

vi.mock("@/stores/appStore", () => ({
  useAppStore: vi.fn(),
}));

vi.mock("@/api/strain", () => ({
  calculateStrain: vi.fn().mockResolvedValue(undefined),
}));

// Mock ParamTooltip to avoid pulling in paramDescriptions
vi.mock("@/components/shared/ParamTooltip", () => ({
  ParamTooltip: () => null,
}));

// ---- Helper to configure store mock -----------------------------------

function mockStoreValues(overrides = {}) {
  const defaults = {
    hasStrain: false,
    strainComputing: false,
    strainProgress: 0,
    setStrainComputing: vi.fn(),
    visSettings: { fps: 1 },
  };
  const values = { ...defaults, ...overrides };
  (useAppStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    (selector: (s: typeof values) => unknown) => selector(values)
  );
  return values;
}

// ---- Setup / Teardown -------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  mockStoreValues();
});

// ---- Tests ------------------------------------------------------------

describe("StrainControls", () => {
  it("renders with default values (Calculate Strain button)", () => {
    render(<StrainControls />);
    expect(
      screen.getByRole("button", { name: "Calculate Strain" })
    ).toBeInTheDocument();
  });

  it("button shows 'Computing...' when strainComputing=true", () => {
    mockStoreValues({ strainComputing: true, strainProgress: 42 });
    render(<StrainControls />);
    expect(screen.getByRole("button", { name: /Computing.*42%/ })).toBeInTheDocument();
  });

  it("button shows 'Recalculate Strain' when hasStrain=true", () => {
    mockStoreValues({ hasStrain: true });
    render(<StrainControls />);
    expect(
      screen.getByRole("button", { name: "Recalculate Strain" })
    ).toBeInTheDocument();
  });

  it("VSG size auto-corrects to odd number >= 9 on blur", async () => {
    const user = userEvent.setup();
    render(<StrainControls />);

    // Find the VSG input by its placeholder
    const vsgInput = screen.getByPlaceholderText("odd ≥ 9");
    expect(vsgInput).toBeInTheDocument();

    // Clear and type an even number
    await user.clear(vsgInput);
    await user.type(vsgInput, "10");
    await user.tab(); // trigger blur
    // Even 10 → next odd 11
    expect(vsgInput).toHaveValue("11");

    // Clear and type a value below 9
    await user.clear(vsgInput);
    await user.type(vsgInput, "4");
    await user.tab();
    // Below 9 → 9
    expect(vsgInput).toHaveValue("9");
  });

  it("Calculate button calls calculateStrain with correct params", async () => {
    const mockSetStrainComputing = vi.fn();
    mockStoreValues({ setStrainComputing: mockSetStrainComputing });
    const user = userEvent.setup();
    render(<StrainControls />);

    await user.click(
      screen.getByRole("button", { name: "Calculate Strain" })
    );

    expect(mockSetStrainComputing).toHaveBeenCalledWith(true);
    expect(calculateStrain).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "green_lagrange",
        vsg_size: 31,
        step: 15,
        poly_order: 1,
        weighting: "uniform",
        spatial_sigma: 0,
        temporal_sigma: 0,
        fps: 1,
        cond_threshold: 1000,
        boundary_erosion: -1, // erosionAuto=true → -1
      })
    );
  });

  it("button is disabled while computing", () => {
    mockStoreValues({ strainComputing: true, strainProgress: 50 });
    render(<StrainControls />);

    const button = screen.getByRole("button", { name: /Computing/ });
    expect(button).toBeDisabled();
  });

  it("Gaussian sigma field only visible when weighting=gaussian", async () => {
    const user = userEvent.setup();
    render(<StrainControls />);

    // With default weighting "uniform", Gauss sigma should not be visible
    expect(screen.queryByPlaceholderText("auto")).not.toBeInTheDocument();

    // Change weighting to "gaussian"
    const weightingSelect = screen.getByDisplayValue("Uniform");
    await user.selectOptions(weightingSelect, "gaussian");

    // Now the Gauss sigma input should appear
    expect(screen.getByPlaceholderText("auto")).toBeInTheDocument();
  });

  it("progress percentage shown during computation", () => {
    mockStoreValues({ strainComputing: true, strainProgress: 73.6 });
    render(<StrainControls />);

    // strainProgress.toFixed(0) → "74"
    expect(screen.getByText(/74%/)).toBeInTheDocument();
  });
});
