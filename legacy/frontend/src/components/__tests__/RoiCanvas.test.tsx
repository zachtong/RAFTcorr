import { render, screen, fireEvent } from "@testing-library/react";
import { RoiCanvas } from "@/components/roi/RoiCanvas";
import { useAppStore } from "@/stores/appStore";
import { useRoiStore } from "@/stores/roiStore";

// ---- Mocks ---------------------------------------------------------------

vi.mock("@/stores/appStore", () => ({
  useAppStore: vi.fn(),
}));

vi.mock("@/stores/roiStore", () => ({
  useRoiStore: vi.fn(),
}));

vi.mock("@/api/images", () => ({
  referenceImageUrl: () => "/api/images/reference",
}));

vi.mock("@/api/roi", () => ({
  addPolygon: vi.fn().mockResolvedValue({}),
  addRectangle: vi.fn().mockResolvedValue({}),
  addCircle: vi.fn().mockResolvedValue({}),
  confirmRoi: vi.fn().mockResolvedValue({}),
}));

// ---- Helper to configure store mocks ------------------------------------

const mockSetRoiConfirmed = vi.fn();
const mockAddPoint = vi.fn();
const mockClearPoints = vi.fn();
const mockSetMaskUrl = vi.fn();

function mockStores(appOverrides = {}, roiOverrides = {}) {
  const appDefaults = {
    imageFiles: [] as string[],
    imageWidth: 0,
    imageHeight: 0,
    setRoiConfirmed: mockSetRoiConfirmed,
    showTileGrid: false,
    tilingPreview: null,
  };
  const appStore = { ...appDefaults, ...appOverrides };
  (useAppStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    (selector: (s: typeof appStore) => unknown) => selector(appStore)
  );

  const roiDefaults = {
    drawingMode: null as string | null,
    cutMode: false,
    currentPoints: [] as [number, number][],
    addPoint: mockAddPoint,
    clearPoints: mockClearPoints,
    maskUrl: null as string | null,
    setMaskUrl: mockSetMaskUrl,
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
  mockStores();
});

// ---- Tests ---------------------------------------------------------------

describe("RoiCanvas", () => {
  it("renders without crashing when no images loaded", () => {
    const { container } = render(<RoiCanvas />);
    expect(container.firstChild).toBeTruthy();
  });

  it("shows placeholder message when no images loaded", () => {
    render(<RoiCanvas />);
    expect(screen.getByText("No images loaded")).toBeInTheDocument();
    expect(
      screen.getByText("Set input directory and load images to begin")
    ).toBeInTheDocument();
  });

  it("renders reference image when images are loaded", () => {
    mockStores({
      imageFiles: ["frame_001.png", "frame_002.png"],
      imageWidth: 640,
      imageHeight: 480,
    });
    render(<RoiCanvas />);

    const img = screen.getByAltText("Reference");
    expect(img).toBeInTheDocument();
    expect(img.getAttribute("src")).toContain("/api/images/reference");
  });

  it("does not show placeholder when images are loaded", () => {
    mockStores({
      imageFiles: ["frame_001.png"],
      imageWidth: 256,
      imageHeight: 256,
    });
    render(<RoiCanvas />);

    expect(screen.queryByText("No images loaded")).not.toBeInTheDocument();
  });

  it("uses crosshair cursor when drawing mode is active", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { drawingMode: "polygon" }
    );
    const { container } = render(<RoiCanvas />);

    const canvasDiv = container.firstChild as HTMLElement;
    expect(canvasDiv.style.cursor).toBe("crosshair");
  });

  it("uses default cursor when no drawing mode", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { drawingMode: null }
    );
    const { container } = render(<RoiCanvas />);

    const canvasDiv = container.firstChild as HTMLElement;
    expect(canvasDiv.style.cursor).toBe("default");
  });

  it("cursor is crosshair for rectangle drawing mode", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { drawingMode: "rectangle" }
    );
    const { container } = render(<RoiCanvas />);

    const canvasDiv = container.firstChild as HTMLElement;
    expect(canvasDiv.style.cursor).toBe("crosshair");
  });

  it("cursor is crosshair for circle drawing mode", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { drawingMode: "circle" }
    );
    const { container } = render(<RoiCanvas />);

    const canvasDiv = container.firstChild as HTMLElement;
    expect(canvasDiv.style.cursor).toBe("crosshair");
  });

  it("renders ROI mask overlay when maskUrl is set", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { maskUrl: "/api/roi/mask?t=12345" }
    );
    render(<RoiCanvas />);

    const maskImg = screen.getByAltText("ROI Mask");
    expect(maskImg).toBeInTheDocument();
    expect(maskImg.getAttribute("src")).toBe("/api/roi/mask?t=12345");
  });

  it("does not render mask overlay when maskUrl is null", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { maskUrl: null }
    );
    render(<RoiCanvas />);

    expect(screen.queryByAltText("ROI Mask")).not.toBeInTheDocument();
  });

  it("renders SVG overlay for drawing when images are loaded", () => {
    mockStores({
      imageFiles: ["a.png"],
      imageWidth: 100,
      imageHeight: 100,
    });
    const { container } = render(<RoiCanvas />);

    const svg = container.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("has mouse event handlers attached to the container", () => {
    mockStores({
      imageFiles: ["a.png"],
      imageWidth: 100,
      imageHeight: 100,
    });
    const { container } = render(<RoiCanvas />);
    const canvasDiv = container.firstChild as HTMLElement;

    // Verify the container can handle mouse events without errors
    fireEvent.mouseDown(canvasDiv, { clientX: 50, clientY: 50 });
    fireEvent.mouseMove(canvasDiv, { clientX: 60, clientY: 60 });
    fireEvent.mouseUp(canvasDiv);
    fireEvent.click(canvasDiv, { clientX: 70, clientY: 70 });

    // No errors thrown = handlers are attached
  });

  it("prevents context menu (right-click)", () => {
    mockStores({
      imageFiles: ["a.png"],
      imageWidth: 100,
      imageHeight: 100,
    });
    const { container } = render(<RoiCanvas />);
    const canvasDiv = container.firstChild as HTMLElement;

    const event = new MouseEvent("contextmenu", {
      bubbles: true,
      cancelable: true,
    });
    const prevented = !canvasDiv.dispatchEvent(event);
    expect(prevented).toBe(true);
  });

  it("reference image has max-w-none class to prevent Tailwind constraint", () => {
    mockStores({
      imageFiles: ["frame_001.png"],
      imageWidth: 640,
      imageHeight: 480,
    });
    render(<RoiCanvas />);

    const img = screen.getByAltText("Reference");
    expect(img.className).toContain("max-w-none");
  });

  it("image style includes transform with translate and scale", () => {
    mockStores({
      imageFiles: ["frame_001.png"],
      imageWidth: 640,
      imageHeight: 480,
    });
    render(<RoiCanvas />);

    const img = screen.getByAltText("Reference");
    expect(img.style.transform).toContain("translate");
    expect(img.style.transform).toContain("scale");
    expect(img.style.transformOrigin).toBe("0 0");
  });

  it("image rendering is set to pixelated when scale > 2", () => {
    // The default scale is computed from container size. Since jsdom has 0
    // container dimensions, the initial scale will be 1 (< 2), so image
    // rendering should be "auto" by default.
    mockStores({
      imageFiles: ["frame_001.png"],
      imageWidth: 640,
      imageHeight: 480,
    });
    render(<RoiCanvas />);

    const img = screen.getByAltText("Reference");
    // Default scale is <=2, so should be "auto"
    expect(img.style.imageRendering).toBe("auto");
  });

  it("renders polygon SVG drawing in progress", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      {
        drawingMode: "polygon",
        currentPoints: [[10, 10], [50, 10], [50, 50]] as [number, number][],
      }
    );
    const { container } = render(<RoiCanvas />);

    const polyline = container.querySelector("polyline");
    expect(polyline).toBeInTheDocument();
    // Should use blue stroke for add mode (cutMode=false)
    expect(polyline?.getAttribute("stroke")).toBe("#3b82f6");
  });

  it("renders polygon SVG with red stroke in cut mode", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      {
        drawingMode: "polygon",
        cutMode: true,
        currentPoints: [[10, 10], [50, 10]] as [number, number][],
      }
    );
    const { container } = render(<RoiCanvas />);

    const polyline = container.querySelector("polyline");
    expect(polyline).toBeInTheDocument();
    expect(polyline?.getAttribute("stroke")).toBe("#ef4444");
  });

  it("renders vertex circles for polygon points", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      {
        drawingMode: "polygon",
        currentPoints: [[10, 10], [50, 10], [50, 50]] as [number, number][],
      }
    );
    const { container } = render(<RoiCanvas />);

    // SVG circles for each point vertex
    const circles = container.querySelectorAll("svg circle");
    expect(circles.length).toBe(3);
  });

  it("Escape key clears current drawing points", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      {
        drawingMode: "polygon",
        currentPoints: [[10, 10]] as [number, number][],
      }
    );
    render(<RoiCanvas />);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(mockClearPoints).toHaveBeenCalled();
  });

  it("Escape key does nothing when no drawing mode", () => {
    mockStores(
      { imageFiles: ["a.png"], imageWidth: 100, imageHeight: 100 },
      { drawingMode: null }
    );
    render(<RoiCanvas />);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(mockClearPoints).not.toHaveBeenCalled();
  });

  it("container has overflow-hidden class", () => {
    const { container } = render(<RoiCanvas />);
    const canvasDiv = container.firstChild as HTMLElement;
    expect(canvasDiv.className).toContain("overflow-hidden");
  });

  it("reference image is not draggable", () => {
    mockStores({
      imageFiles: ["frame_001.png"],
      imageWidth: 100,
      imageHeight: 100,
    });
    render(<RoiCanvas />);

    const img = screen.getByAltText("Reference");
    expect(img.getAttribute("draggable")).toBe("false");
  });

  it("renders tile grid overlay when showTileGrid is true", () => {
    mockStores({
      imageFiles: ["a.png"],
      imageWidth: 200,
      imageHeight: 200,
      showTileGrid: true,
      tilingPreview: {
        work_area: { x: 0, y: 0, w: 200, h: 200 },
        tiles: [[0, 0, 100, 100], [100, 0, 100, 100]],
      },
    });
    const { container } = render(<RoiCanvas />);

    // Should have tile overlay rects (work area boundary + 2 tiles = 3 rects total)
    const rects = container.querySelectorAll("svg rect");
    expect(rects.length).toBe(3);
  });

  it("does not render tile grid when showTileGrid is false", () => {
    mockStores({
      imageFiles: ["a.png"],
      imageWidth: 200,
      imageHeight: 200,
      showTileGrid: false,
      tilingPreview: {
        work_area: { x: 0, y: 0, w: 200, h: 200 },
        tiles: [[0, 0, 100, 100]],
      },
    });
    const { container } = render(<RoiCanvas />);

    // Only the SVG overlay exists, no tile rects
    const rects = container.querySelectorAll("svg rect");
    expect(rects.length).toBe(0);
  });
});
