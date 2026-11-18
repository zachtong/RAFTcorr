import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PathSettings } from "@/components/roi/PathSettings";
import { useAppStore } from "@/stores/appStore";
import { useRoiStore } from "@/stores/roiStore";
import { loadImages } from "@/api/images";
import { clearRoi } from "@/api/roi";
import { clearProbes } from "@/api/probes";
import { listModels } from "@/api/models";

// ---- Mocks ---------------------------------------------------------------

vi.mock("@/stores/appStore", () => ({
  useAppStore: vi.fn(),
}));

vi.mock("@/stores/roiStore", () => ({
  useRoiStore: vi.fn(),
}));

vi.mock("@/api/images", () => ({
  loadImages: vi.fn(),
}));

vi.mock("@/api/roi", () => ({
  clearRoi: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/api/probes", () => ({
  clearProbes: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/api/models", () => ({
  listModels: vi.fn().mockResolvedValue([]),
  describeModel: vi.fn().mockResolvedValue({}),
  selectModel: vi.fn().mockResolvedValue({}),
}));

// Mock FileBrowser to avoid rendering the heavy component
vi.mock("@/components/shared/FileBrowser", () => ({
  FileBrowser: ({ open, onSelect }: { open: boolean; onSelect: (p: string) => void }) =>
    open ? (
      <div data-testid="file-browser">
        <button onClick={() => onSelect("/selected/path")}>Select Directory</button>
      </div>
    ) : null,
}));

// Mock CollapsibleSection to always render children without collapse behavior
vi.mock("@/components/shared/CollapsibleSection", () => ({
  CollapsibleSection: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div data-testid={`section-${title}`}>{children}</div>
  ),
}));

// Mock SelectField
vi.mock("@/components/shared/SelectField", () => ({
  SelectField: () => <select data-testid="model-select" />,
}));

// Mock FieldRow
vi.mock("@/components/shared/FieldRow", () => ({
  FieldRow: ({ children, label }: { children: React.ReactNode; label: string }) => (
    <div data-testid={`field-${label}`}>{children}</div>
  ),
}));

const mockToast = vi.fn();
vi.mock("@/components/shared/Toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

// ---- Helper to configure store mocks ------------------------------------

const mockSetImageDir = vi.fn();
const mockSetImageFiles = vi.fn();
const mockResetSession = vi.fn();
const mockSetModel = vi.fn();
const mockResetRoi = vi.fn();
const mockClearDrawing = vi.fn();

function mockStores(appOverrides = {}, roiOverrides = {}) {
  const appDefaults = {
    imageDir: "",
    setImageDir: mockSetImageDir,
    setImageFiles: mockSetImageFiles,
    resetSession: mockResetSession,
    imageFiles: [] as string[],
    selectedModel: "",
    setModel: mockSetModel,
    modelMetadata: null,
  };
  const appStore = { ...appDefaults, ...appOverrides };
  (useAppStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    (selector: (s: typeof appStore) => unknown) => selector(appStore)
  );

  const roiDefaults = {
    setMaskUrl: mockResetRoi,
    setDrawingMode: mockClearDrawing,
  };
  const roiStore = { ...roiDefaults, ...roiOverrides };
  (useRoiStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    (selector: (s: typeof roiStore) => unknown) => selector(roiStore)
  );

  return { appStore, roiStore };
}

// ---- Setup / Teardown ----------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  mockStores();
});

// ---- Tests ---------------------------------------------------------------

describe("PathSettings", () => {
  it("renders directory input and confirm button", () => {
    render(<PathSettings />);

    expect(screen.getByPlaceholderText("/path/to/images")).toBeInTheDocument();
    expect(screen.getByText("Confirm")).toBeInTheDocument();
  });

  it("renders natural sort checkbox (default checked)", () => {
    render(<PathSettings />);

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeChecked();
  });

  it("renders Browse button that opens FileBrowser", async () => {
    const user = userEvent.setup();
    render(<PathSettings />);

    // FileBrowser should not be visible initially
    expect(screen.queryByTestId("file-browser")).not.toBeInTheDocument();

    // Click the Browse button (has title="Browse")
    const browseButton = screen.getByTitle("Browse");
    await user.click(browseButton);

    // FileBrowser should now be visible
    expect(screen.getByTestId("file-browser")).toBeInTheDocument();
  });

  it("calls loadImages API when Confirm button is clicked", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["img1.png", "img2.png"],
      count: 2,
      width: 512,
      height: 512,
      sort_suggestion: null,
    });

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/test/images");
    await user.click(screen.getByText("Confirm"));

    await waitFor(() => {
      expect(loadImages).toHaveBeenCalledWith("/test/images", true);
    });
  });

  it("shows success toast and image count after successful load", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["a.png", "b.png", "c.png"],
      count: 3,
      width: 640,
      height: 480,
      sort_suggestion: null,
    });

    mockStores({ imageFiles: ["a.png", "b.png", "c.png"] });
    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/test/dir");
    await user.click(screen.getByText("Confirm"));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        "success",
        "Loaded 3 images (640x480)"
      );
    });
  });

  it("shows error toast and error text when load fails", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockRejectedValue({
      response: { data: { error: "Directory not found" } },
    });

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/bad/path");
    await user.click(screen.getByText("Confirm"));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith("error", "Directory not found");
    });

    // Error message also displayed in the component
    expect(screen.getByText("Directory not found")).toBeInTheDocument();
  });

  it("shows validation message when path is empty", async () => {
    const user = userEvent.setup();
    render(<PathSettings />);

    // Click Confirm without entering a path
    await user.click(screen.getByText("Confirm"));

    expect(mockToast).toHaveBeenCalledWith("error", "Please enter a directory path");
    expect(screen.getByText("Please enter a directory path")).toBeInTheDocument();
    expect(loadImages).not.toHaveBeenCalled();
  });

  it("natural sort toggle persists to localStorage", async () => {
    const user = userEvent.setup();
    render(<PathSettings />);

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).toBeChecked();

    // Uncheck
    await user.click(checkbox);
    expect(localStorage.getItem("dic_natural_sort")).toBe("false");

    // Check again
    await user.click(checkbox);
    expect(localStorage.getItem("dic_natural_sort")).toBe("true");
  });

  it("reads natural sort preference from localStorage on mount", () => {
    localStorage.setItem("dic_natural_sort", "false");
    render(<PathSettings />);

    const checkbox = screen.getByRole("checkbox");
    expect(checkbox).not.toBeChecked();
  });

  it("displays image count when images are loaded", () => {
    mockStores({ imageFiles: ["frame_001.png", "frame_002.png", "frame_003.png"] });
    render(<PathSettings />);

    expect(screen.getByText(/3 images loaded/)).toBeInTheDocument();
  });

  it("shows sort_suggestion toast when backend recommends natural sort", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["img1.png", "img2.png"],
      count: 2,
      width: 256,
      height: 256,
      sort_suggestion: "natural",
    });

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/test/dir");
    await user.click(screen.getByText("Confirm"));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        "info",
        expect.stringContaining("Natural Sort")
      );
    });
  });

  it("clears backend ROI and probes on successful load", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["a.png"],
      count: 1,
      width: 100,
      height: 100,
      sort_suggestion: null,
    });

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/test");
    await user.click(screen.getByText("Confirm"));

    await waitFor(() => {
      expect(clearRoi).toHaveBeenCalled();
      expect(clearProbes).toHaveBeenCalled();
      expect(mockResetSession).toHaveBeenCalled();
      expect(mockResetRoi).toHaveBeenCalledWith(null);
      expect(mockClearDrawing).toHaveBeenCalledWith(null);
    });
  });

  it("FileBrowser selection triggers load with selected path", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["x.png"],
      count: 1,
      width: 64,
      height: 64,
      sort_suggestion: null,
    });

    render(<PathSettings />);

    // Open file browser
    await user.click(screen.getByTitle("Browse"));
    expect(screen.getByTestId("file-browser")).toBeInTheDocument();

    // Select a directory
    await user.click(screen.getByText("Select Directory"));

    await waitFor(() => {
      expect(loadImages).toHaveBeenCalledWith("/selected/path", true);
    });
  });

  it("Enter key in directory input triggers load", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["a.png"],
      count: 1,
      width: 100,
      height: 100,
      sort_suggestion: null,
    });

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/enter/test{Enter}");

    await waitFor(() => {
      expect(loadImages).toHaveBeenCalledWith("/enter/test", true);
    });
  });

  it("shows loading state ('...') while loading", async () => {
    // Make loadImages never resolve to keep loading state
    (loadImages as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {}));
    const user = userEvent.setup();

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/test");
    await user.click(screen.getByText("Confirm"));

    // Button should show loading indicator
    await waitFor(() => {
      expect(screen.getByText("...")).toBeInTheDocument();
    });
  });

  it("displays model metadata badges when modelMetadata is present", () => {
    mockStores({
      modelMetadata: {
        small: false,
        full_resolution: true,
        corr_levels: 4,
      },
    });
    render(<PathSettings />);

    expect(screen.getByText("RAFT-Large")).toBeInTheDocument();
    expect(screen.getByText("Full res")).toBeInTheDocument();
    expect(screen.getByText("4-level pyramid")).toBeInTheDocument();
  });

  it("fetches models after successful image load", async () => {
    const user = userEvent.setup();
    (loadImages as ReturnType<typeof vi.fn>).mockResolvedValue({
      files: ["a.png"],
      count: 1,
      width: 100,
      height: 100,
      sort_suggestion: null,
    });
    (listModels as ReturnType<typeof vi.fn>).mockResolvedValue([
      { path: "/models/raft.pth", label: "RAFT-Large" },
    ]);

    render(<PathSettings />);

    const input = screen.getByPlaceholderText("/path/to/images");
    await user.type(input, "/test");
    await user.click(screen.getByText("Confirm"));

    await waitFor(() => {
      expect(listModels).toHaveBeenCalled();
    });
  });
});
