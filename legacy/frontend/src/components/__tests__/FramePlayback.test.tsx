import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FramePlayback } from "@/components/displacement/FramePlayback";
import { useAppStore } from "@/stores/appStore";

// ---- Mocks ------------------------------------------------------------

const mockNext = vi.fn();
const mockPrev = vi.fn();
const mockFirst = vi.fn();
const mockLast = vi.fn();
const mockGoTo = vi.fn();
const mockPlay = vi.fn();
const mockPause = vi.fn();
const mockSetFps = vi.fn();

vi.mock("@/hooks/useFrameNav", () => ({
  useFrameNav: vi.fn(() => ({
    currentFrame: 0,
    numFrames: 10,
    next: mockNext,
    prev: mockPrev,
    first: mockFirst,
    last: mockLast,
    goTo: mockGoTo,
    play: mockPlay,
    pause: mockPause,
    isPlaying: false,
    setFps: mockSetFps,
  })),
}));

vi.mock("@/stores/appStore", () => ({
  useAppStore: vi.fn(),
}));

const mockZoomIn = vi.fn();
const mockZoomOut = vi.fn();
const mockZoomReset = vi.fn();

function mockStore() {
  (useAppStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
    (selector: (s: Record<string, unknown>) => unknown) =>
      selector({
        zoomIn: mockZoomIn,
        zoomOut: mockZoomOut,
        zoomReset: mockZoomReset,
      })
  );
}

// ---- Setup / Teardown -------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
  mockStore();
});

// Need a reference to re-import for overrides
import { useFrameNav } from "@/hooks/useFrameNav";

function mockFrameNav(overrides: Record<string, unknown> = {}) {
  (useFrameNav as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
    currentFrame: 0,
    numFrames: 10,
    next: mockNext,
    prev: mockPrev,
    first: mockFirst,
    last: mockLast,
    goTo: mockGoTo,
    play: mockPlay,
    pause: mockPause,
    isPlaying: false,
    setFps: mockSetFps,
    ...overrides,
  });
}

// ---- Tests ------------------------------------------------------------

describe("FramePlayback", () => {
  it("renders nothing when numFrames is 0", () => {
    mockFrameNav({ numFrames: 0 });
    const { container } = render(<FramePlayback />);
    expect(container.innerHTML).toBe("");
  });

  it("renders playback bar when numFrames > 0", () => {
    mockFrameNav({ numFrames: 10, currentFrame: 0 });
    render(<FramePlayback />);

    // Frame counter (slider label) shows 1-indexed "Frame 1 / 10"
    expect(screen.getByText(/Frame\s+1\s*\/\s*10/)).toBeInTheDocument();
    // Speed label
    expect(screen.getByText("Speed")).toBeInTheDocument();
    // Go button
    expect(screen.getByRole("button", { name: "Go" })).toBeInTheDocument();
  });

  it("displays 1-indexed frame number", () => {
    mockFrameNav({ currentFrame: 4, numFrames: 20 });
    render(<FramePlayback />);

    // The number input shows currentFrame + 1 = 5 (spinbutton role is the
    // Goto <input type="number"> — the slider is type="range").
    const frameInput = screen.getByRole("spinbutton");
    expect(frameInput).toHaveValue(5);
    expect(screen.getByText(/Frame\s+5\s*\/\s*20/)).toBeInTheDocument();
  });

  it("calls first() when skip-back button clicked", async () => {
    const user = userEvent.setup();
    mockFrameNav();
    render(<FramePlayback />);

    // First button is the SkipBack icon button (first in the nav group)
    const buttons = screen.getAllByRole("button");
    // SkipBack is the first navigation button
    await user.click(buttons[0]);
    expect(mockFirst).toHaveBeenCalledOnce();
  });

  it("calls prev() when chevron-left button clicked", async () => {
    const user = userEvent.setup();
    mockFrameNav();
    render(<FramePlayback />);

    const buttons = screen.getAllByRole("button");
    // ChevronLeft is the second navigation button
    await user.click(buttons[1]);
    expect(mockPrev).toHaveBeenCalledOnce();
  });

  it("calls play() when play button clicked", async () => {
    const user = userEvent.setup();
    mockFrameNav({ isPlaying: false });
    render(<FramePlayback />);

    const buttons = screen.getAllByRole("button");
    // Play/Pause is the third navigation button
    await user.click(buttons[2]);
    expect(mockPlay).toHaveBeenCalledOnce();
  });

  it("calls pause() when pause button clicked during playback", async () => {
    const user = userEvent.setup();
    mockFrameNav({ isPlaying: true });
    render(<FramePlayback />);

    const buttons = screen.getAllByRole("button");
    // Play/Pause is the third navigation button
    await user.click(buttons[2]);
    expect(mockPause).toHaveBeenCalledOnce();
  });

  it("calls next() when chevron-right button clicked", async () => {
    const user = userEvent.setup();
    mockFrameNav();
    render(<FramePlayback />);

    const buttons = screen.getAllByRole("button");
    // ChevronRight is the fourth navigation button
    await user.click(buttons[3]);
    expect(mockNext).toHaveBeenCalledOnce();
  });

  it("calls last() when skip-forward button clicked", async () => {
    const user = userEvent.setup();
    mockFrameNav();
    render(<FramePlayback />);

    const buttons = screen.getAllByRole("button");
    // SkipForward is the fifth navigation button
    await user.click(buttons[4]);
    expect(mockLast).toHaveBeenCalledOnce();
  });

  it("calls zoomIn, zoomOut, zoomReset when zoom buttons clicked", async () => {
    const user = userEvent.setup();
    mockFrameNav();
    render(<FramePlayback />);

    const buttons = screen.getAllByRole("button");
    // After 5 nav buttons: ZoomIn(5), ZoomOut(6), ZoomReset(7)
    await user.click(buttons[5]);
    expect(mockZoomIn).toHaveBeenCalledOnce();

    await user.click(buttons[6]);
    expect(mockZoomOut).toHaveBeenCalledOnce();

    await user.click(buttons[7]);
    expect(mockZoomReset).toHaveBeenCalledOnce();
  });

  it("speed select changes fps", async () => {
    const user = userEvent.setup();
    mockFrameNav();
    render(<FramePlayback />);

    const speedSelect = screen.getByDisplayValue("5x");
    await user.selectOptions(speedSelect, "2");

    expect(mockSetFps).toHaveBeenCalledWith(2);
  });

  it("Go button navigates to user-entered frame (1-indexed → 0-indexed)", async () => {
    const user = userEvent.setup();
    mockFrameNav({ currentFrame: 0, numFrames: 10 });
    render(<FramePlayback />);

    const frameInput = screen.getByRole("spinbutton");
    // Use fireEvent.change to set value directly (type="number" inputs
    // don't support setSelectionRange in jsdom, so user.clear+type fails)
    fireEvent.change(frameInput, { target: { value: "7" } });
    await user.click(screen.getByRole("button", { name: "Go" }));

    // goTo receives 0-indexed: 7 - 1 = 6
    expect(mockGoTo).toHaveBeenCalledWith(6);
  });

  it("Enter key in frame input triggers go-to navigation", async () => {
    mockFrameNav({ currentFrame: 0, numFrames: 10 });
    render(<FramePlayback />);

    const frameInput = screen.getByRole("spinbutton");
    // Set value via fireEvent, then fire Enter keydown
    fireEvent.change(frameInput, { target: { value: "3" } });
    fireEvent.keyDown(frameInput, { key: "Enter" });

    expect(mockGoTo).toHaveBeenCalledWith(2); // 3 - 1 = 2
  });

  it("shows pre-render progress bar when preRenderCache is active", () => {
    mockFrameNav();
    render(
      <FramePlayback
        preRenderCache={{
          isPreRendering: true,
          progress: 60,
          cachedCount: 6,
          totalFrames: 10,
          isFrameReady: () => true,
        }}
      />
    );

    // The progress bar has a style attribute with width 60%
    const progressBar = document.querySelector('[style*="width: 60%"]');
    expect(progressBar).toBeTruthy();
  });

  it("does NOT show pre-render progress bar when not pre-rendering", () => {
    mockFrameNav();
    render(
      <FramePlayback
        preRenderCache={{
          isPreRendering: false,
          progress: 100,
          cachedCount: 10,
          totalFrames: 10,
          isFrameReady: () => true,
        }}
      />
    );

    const progressBar = document.querySelector('[style*="width:"]');
    expect(progressBar).toBeNull();
  });
});
