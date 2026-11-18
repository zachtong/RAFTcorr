import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ExportDialog } from "@/components/postprocessing/ExportDialog";
import { useAppStore } from "@/stores/appStore";
import {
  exportScientific,
  exportImages,
  cancelExport,
  exportAnimation,
  exportReport,
} from "@/api/export";

// ---- Mocks ---------------------------------------------------------------

vi.mock("@/stores/appStore", () => ({
  useAppStore: vi.fn(),
}));

vi.mock("@/api/export", () => ({
  exportScientific: vi.fn().mockResolvedValue({ ok: true, path: "/out.mat" }),
  exportImages: vi.fn().mockResolvedValue(undefined),
  cancelExport: vi.fn().mockResolvedValue(undefined),
  exportAnimation: vi.fn().mockResolvedValue(undefined),
  exportReport: vi.fn().mockResolvedValue({ ok: true, path: "/report.html" }),
}));

// Mock CollapsibleSection to always render children (defaultOpen=false in ExportDialog)
vi.mock("@/components/shared/CollapsibleSection", () => ({
  CollapsibleSection: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div data-testid={`section-${title}`}>{children}</div>
  ),
}));

// Mock SmallInput as a simple controlled input
vi.mock("@/components/shared/SmallInput", () => ({
  SmallInput: ({
    value,
    onChange,
    className,
  }: {
    value: string | number;
    onChange?: (v: string) => void;
    className?: string;
  }) => (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      className={className}
      data-testid="small-input"
    />
  ),
}));

// Mock FileBrowser
vi.mock("@/components/shared/FileBrowser", () => ({
  FileBrowser: ({ open, onSelect }: { open: boolean; onSelect: (p: string) => void }) =>
    open ? (
      <div data-testid="file-browser">
        <button onClick={() => onSelect("/browsed/path")}>Select</button>
      </div>
    ) : null,
}));

const mockToast = vi.fn();
vi.mock("@/components/shared/Toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

// ---- Helper to configure store mock -------------------------------------

function mockStore(overrides = {}) {
  const defaults = {
    imageDir: "/test/images",
    numFrames: 10,
    exportActive: false,
    exportProgress: 0,
    displayComponent: "u",
    hasStrain: false,
    strainComponents: [] as string[],
    visSettings: {
      colormap: "turbo",
      alpha: 0.7,
      background: "reference" as const,
      logScale: false,
      physicalEnabled: false,
      physicalRatio: 1.0,
      physicalUnit: "px",
      fps: 1.0,
      fixedRange: false,
      vminU: "",
      vmaxU: "",
      vminV: "",
      vmaxV: "",
    },
    arrowSettings: {
      showQuiver: false,
      showStreamlines: false,
      showPrincipalDirs: false,
      spacing: 30,
      scale: 20,
      color: "white",
      lineWidth: 1.5,
      streamQuality: 4,
    },
    colorbarSettings: {
      labelText: "",
      labelFontSize: 12,
      tickCount: 0,
      tickFontSize: 10,
      barThickness: 1.0,
      hideOutline: false,
      discreteLevels: 0,
    },
  };
  const store = { ...defaults, ...overrides };
  (useAppStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    (selector: (s: typeof store) => unknown) => selector(store)
  );
  return store;
}

// ---- Setup / Teardown ----------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  mockStore();
});

// ---- Tests ---------------------------------------------------------------

describe("ExportDialog", () => {
  // --- Rendering ---

  it("renders all export section headings", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Scientific Data")).toBeInTheDocument();
    expect(screen.getByText("Visualization Images")).toBeInTheDocument();
    expect(screen.getByText("Animation")).toBeInTheDocument();
    expect(screen.getByText("Report")).toBeInTheDocument();
  });

  it("renders scientific export path input with default value from imageDir", () => {
    render(<ExportDialog />);

    const input = screen.getByPlaceholderText("Export path (.h5)");
    expect(input).toBeInTheDocument();
    expect(input).toHaveValue("/test/images/results.h5");
  });

  it("renders Export .h5 button (default format)", () => {
    render(<ExportDialog />);

    expect(screen.getByText(/Export \.h5/)).toBeInTheDocument();
  });

  it("renders Export Images button", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Export Images")).toBeInTheDocument();
  });

  it("renders Export Animation button", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Export Animation")).toBeInTheDocument();
  });

  it("renders Generate Report button", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Generate Report")).toBeInTheDocument();
  });

  // --- DPI selector ---

  it("renders DPI selector with predefined options", () => {
    render(<ExportDialog />);

    const dpiSelect = screen.getByDisplayValue("150");
    expect(dpiSelect).toBeInTheDocument();

    const options = dpiSelect.querySelectorAll("option");
    const values = Array.from(options).map((o) => o.getAttribute("value"));
    expect(values).toEqual(["100", "150", "300", "600"]);
  });

  it("DPI can be changed via select", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    const dpiSelect = screen.getByDisplayValue("150");
    await user.selectOptions(dpiSelect, "300");
    expect(dpiSelect).toHaveValue("300");
  });

  // --- Component checkboxes ---

  it("renders displacement component checkboxes", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Components:")).toBeInTheDocument();
    // These labels appear in multiple places (image export, animation select, key frames)
    // Use getAllByText to verify they exist at least once
    expect(screen.getAllByText("U (horizontal)").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("V (vertical)").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Magnitude").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Velocity").length).toBeGreaterThanOrEqual(1);
  });

  it("shows strain components when hasStrain is true", () => {
    mockStore({
      hasStrain: true,
      strainComponents: ["exx", "eyy", "exy"],
    });
    render(<ExportDialog />);

    // Strain labels appear in image export, animation select, and key frames
    expect(screen.getAllByText(/εxx/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/εyy/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/εxy/).length).toBeGreaterThanOrEqual(1);
  });

  it("does not show strain components when hasStrain is false", () => {
    mockStore({ hasStrain: false, strainComponents: [] });
    render(<ExportDialog />);

    expect(screen.queryByText(/εxx/)).not.toBeInTheDocument();
  });

  it("component checkbox can be toggled", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    // "U (horizontal)" is the image-export label (the sci-var picker uses "U").
    const uCheckboxes = screen.getAllByRole("checkbox", { name: "U (horizontal)" });
    expect(uCheckboxes[0]).toBeChecked();

    // "Magnitude" appears in the sci-var picker (default ON), image export
    // (default OFF), and key frame (default ON). The image-export one is the
    // only Magnitude checkbox that is initially unchecked.
    const magCheckboxes = screen.getAllByRole("checkbox", { name: "Magnitude" });
    const imageExportMag = magCheckboxes.find(
      (cb) => !(cb as HTMLInputElement).checked,
    ) as HTMLInputElement;
    expect(imageExportMag).toBeDefined();

    await user.click(imageExportMag);
    expect(imageExportMag).toBeChecked();
  });

  // --- Scientific export ---

  it("scientific export button calls exportScientific API", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText(/Export \.h5/));

    await waitFor(() => {
      expect(exportScientific).toHaveBeenCalledWith(
        expect.objectContaining({
          file_path: expect.stringContaining("results.h5"),
          upsample_strain: true,
        })
      );
    });
  });

  it("scientific export sends selected variables list", async () => {
    mockStore({ hasStrain: true, strainComponents: ["exx", "eyy", "exy"] });
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText(/Export \.h5/));

    await waitFor(() => {
      const call = (exportScientific as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(call.variables).toEqual(
        expect.arrayContaining([
          "X_ref", "Y_ref", "ROI_mask",
          "U", "V", "Magnitude", "Velocity",
          "exx", "eyy", "exy", "e1", "e2",
          "max_shear", "von_mises", "rotation",
        ]),
      );
    });
  });

  it("export button is disabled when no variables selected", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByRole("button", { name: /^None$/ }));

    const btn = screen.getByText(/Export \.h5/).closest("button");
    expect(btn).toBeDisabled();
  });

  it("shows success toast after scientific export", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText(/Export \.h5/));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        "success",
        "Scientific data exported successfully"
      );
    });
  });

  it("shows error toast when scientific export fails", async () => {
    const user = userEvent.setup();
    (exportScientific as ReturnType<typeof vi.fn>).mockRejectedValue({
      response: { status: 500, data: { error: "No results to export" } },
    });

    render(<ExportDialog />);

    await user.click(screen.getByText(/Export \.h5/));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        "error",
        "Scientific export failed: No results to export"
      );
    });
  });

  // --- Image export ---

  it("image export button calls exportImages API", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText("Export Images"));

    await waitFor(() => {
      expect(exportImages).toHaveBeenCalledWith(
        expect.objectContaining({
          output_dir: expect.stringContaining("/export/"),
          components: expect.objectContaining({ u: expect.any(Object) }),
          frame_range: [1, 10],
          settings: expect.objectContaining({
            colormap: "turbo",
            dpi: 150,
          }),
        })
      );
    });
  });

  it("shows error when no components selected for image export", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    // Uncheck the default "u" component — first occurrence is in image export
    const uCheckboxes = screen.getAllByRole("checkbox", { name: "U (horizontal)" });
    await user.click(uCheckboxes[0]);

    await user.click(screen.getByText("Export Images"));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        "error",
        "Select at least one component to export"
      );
    });

    expect(exportImages).not.toHaveBeenCalled();
  });

  // --- Export progress ---
  // Note: Progress bar visibility depends on internal activeExportType state.
  // We test what is externally observable: buttons disabled during export,
  // and progress rendering after triggering export with exportActive=false initially.

  it("Export Images button is disabled during active export", () => {
    mockStore({ exportActive: true, exportProgress: 30 });
    render(<ExportDialog />);

    // Export Images button should be disabled when exportActive is true
    const imgButton = screen.getByText("Export Images");
    expect(imgButton.closest("button")).toBeDisabled();
  });

  it("Export Animation button is disabled during active export", () => {
    mockStore({ exportActive: true, exportProgress: 30 });
    render(<ExportDialog />);

    const animButton = screen.getByText("Export Animation");
    expect(animButton.closest("button")).toBeDisabled();
  });

  it("scientific export button is disabled when path is empty", () => {
    mockStore({ imageDir: "" });
    render(<ExportDialog />);

    const sciButton = screen.getByText(/Export \.h5/);
    expect(sciButton.closest("button")).toBeDisabled();
  });

  // --- Animation export ---

  it("renders animation format selector with GIF and MP4 options", () => {
    render(<ExportDialog />);

    const formatSelect = screen.getByDisplayValue("GIF");
    expect(formatSelect).toBeInTheDocument();

    const options = formatSelect.querySelectorAll("option");
    const values = Array.from(options).map((o) => o.getAttribute("value"));
    expect(values).toEqual(["gif", "mp4"]);
  });

  it("animation format can be changed", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    const formatSelect = screen.getByDisplayValue("GIF");
    await user.selectOptions(formatSelect, "mp4");
    expect(formatSelect).toHaveValue("mp4");
  });

  it("animation component selector shows displacement components", () => {
    render(<ExportDialog />);

    // The animation component select defaults to "u" which shows "U (horizontal)"
    const componentSelect = screen.getByDisplayValue("U (horizontal)");
    expect(componentSelect).toBeInTheDocument();
  });

  it("renders Colorbar and Timestamp checkboxes for animation", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Colorbar")).toBeInTheDocument();
    expect(screen.getByText("Timestamp")).toBeInTheDocument();
  });

  it("Colorbar checkbox defaults to checked", () => {
    render(<ExportDialog />);

    const checkbox = screen.getByRole("checkbox", { name: "Colorbar" });
    expect(checkbox).toBeChecked();
  });

  it("Timestamp checkbox defaults to unchecked", () => {
    render(<ExportDialog />);

    const checkbox = screen.getByRole("checkbox", { name: "Timestamp" });
    expect(checkbox).not.toBeChecked();
  });

  it("animation export button calls exportAnimation API", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText("Export Animation"));

    await waitFor(() => {
      expect(exportAnimation).toHaveBeenCalledWith(
        expect.objectContaining({
          format: "gif",
          component: "u",
          fps: 10,
          include_colorbar: true,
          timestamp_overlay: false,
        })
      );
    });
  });

  // --- Report export ---

  it("renders report title input with default value", () => {
    render(<ExportDialog />);

    const titleInput = screen.getByDisplayValue("RAFT-DIC Analysis Report");
    expect(titleInput).toBeInTheDocument();
  });

  it("renders author and notes inputs", () => {
    render(<ExportDialog />);

    expect(screen.getByPlaceholderText("Author / Institution")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Analysis notes...")).toBeInTheDocument();
  });

  it("renders report section checkboxes", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Sections:")).toBeInTheDocument();
    expect(screen.getByText("Header")).toBeInTheDocument();
    expect(screen.getByText("Experiment")).toBeInTheDocument();
    expect(screen.getByText("Parameters")).toBeInTheDocument();
    expect(screen.getByText("ROI")).toBeInTheDocument();
    expect(screen.getByText("Key Frames")).toBeInTheDocument();
    expect(screen.getByText("Statistics")).toBeInTheDocument();
    expect(screen.getByText("Point Probes")).toBeInTheDocument();
    expect(screen.getByText("Line Probes")).toBeInTheDocument();
    expect(screen.getByText("Area Probes")).toBeInTheDocument();
  });

  it("report section checkboxes default to all checked", () => {
    render(<ExportDialog />);

    // Use getByRole with accessible name to find specific section checkboxes
    const sectionNames = [
      "Header", "Experiment", "Parameters", "ROI",
      "Key Frames", "Statistics", "Point Probes", "Line Probes", "Area Probes",
    ];
    for (const name of sectionNames) {
      const checkbox = screen.getByRole("checkbox", { name });
      expect(checkbox).toBeChecked();
    }
  });

  it("report section checkbox can be unchecked", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    const checkbox = screen.getByRole("checkbox", { name: "Header" });
    expect(checkbox).toBeChecked();

    await user.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });

  it("renders theme selector with options", () => {
    render(<ExportDialog />);

    const themeSelect = screen.getByDisplayValue("Light");
    expect(themeSelect).toBeInTheDocument();

    const options = themeSelect.querySelectorAll("option");
    const values = Array.from(options).map((o) => o.getAttribute("value"));
    expect(values).toEqual(["light", "dark", "academic", "minimal"]);
  });

  it("renders format selector with HTML, PDF, Both", () => {
    render(<ExportDialog />);

    const formatSelect = screen.getByDisplayValue("HTML");
    expect(formatSelect).toBeInTheDocument();

    const options = formatSelect.querySelectorAll("option");
    const values = Array.from(options).map((o) => o.getAttribute("value"));
    expect(values).toEqual(["html", "pdf", "both"]);
  });

  it("Generate Report calls exportReport API", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText("Generate Report"));

    await waitFor(() => {
      expect(exportReport).toHaveBeenCalledWith(
        expect.objectContaining({
          format: "html",
          theme: "light",
          custom_title: "RAFT-DIC Analysis Report",
        })
      );
    });
  });

  it("shows success toast after report generation", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    await user.click(screen.getByText("Generate Report"));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        "success",
        expect.stringContaining("Report generated")
      );
    });
  });

  it("shows error toast when report generation fails", async () => {
    const user = userEvent.setup();
    (exportReport as ReturnType<typeof vi.fn>).mockRejectedValue({
      response: { data: { error: "No data available" } },
    });

    render(<ExportDialog />);

    await user.click(screen.getByText("Generate Report"));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith("error", "No data available");
    });
  });

  it("Generate Report button shows 'Generating...' while busy", async () => {
    (exportReport as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {}));
    const user = userEvent.setup();

    render(<ExportDialog />);
    await user.click(screen.getByText("Generate Report"));

    await waitFor(() => {
      expect(screen.getByText("Generating...")).toBeInTheDocument();
    });
  });

  // --- Key Frame Components ---

  it("shows key frame component checkboxes when Key Frames section is checked", () => {
    render(<ExportDialog />);

    expect(screen.getByText("Key Frame Components:")).toBeInTheDocument();
    // Magnitude appears in image export components AND key frame components
    const magTexts = screen.getAllByText("Magnitude");
    expect(magTexts.length).toBeGreaterThanOrEqual(2);
  });

  // --- Browse buttons ---

  it("scientific export browse button opens FileBrowser", async () => {
    const user = userEvent.setup();
    render(<ExportDialog />);

    const browseButtons = screen.getAllByTitle("Browse");
    expect(browseButtons.length).toBeGreaterThanOrEqual(1);

    await user.click(browseButtons[0]);
    expect(screen.getByTestId("file-browser")).toBeInTheDocument();
  });

  // --- WYSIWYG note ---

  it("shows WYSIWYG settings note", () => {
    render(<ExportDialog />);

    expect(
      screen.getByText(/Uses current visualization settings/)
    ).toBeInTheDocument();
  });

  // --- Animation scale options ---

  it("renders animation scale selector", () => {
    render(<ExportDialog />);

    const scaleSelect = screen.getByDisplayValue("100%");
    expect(scaleSelect).toBeInTheDocument();

    const options = scaleSelect.querySelectorAll("option");
    const values = Array.from(options).map((o) => o.getAttribute("value"));
    expect(values).toEqual(["1.0", "0.5", "0.25"]);
  });

  // --- Output directory defaults ---

  it("image export directory defaults based on imageDir", () => {
    render(<ExportDialog />);

    const dirInput = screen.getByPlaceholderText("Output directory");
    expect(dirInput).toHaveValue("/test/images/export/");
  });

  it("animation output path defaults based on imageDir and component", () => {
    render(<ExportDialog />);

    const animInput = screen.getByPlaceholderText("Output file (.gif / .mp4)") as HTMLInputElement;
    expect(animInput.value).toContain("/test/images/animation_u_");
    expect(animInput.value).toMatch(/\.gif$/);
  });
});
