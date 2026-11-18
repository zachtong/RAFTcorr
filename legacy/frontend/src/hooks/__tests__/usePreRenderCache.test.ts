import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePreRenderCache } from "@/hooks/usePreRenderCache";
import { useAppStore } from "@/stores/appStore";

// Mock the colormap modules — they depend on OffscreenCanvas, fetch, createImageBitmap
// Note: ImageData is not available in jsdom, so use a plain object as stand-in
vi.mock("@/lib/applyColormap", () => ({
  decodeDataTexture: vi.fn().mockResolvedValue({
    imageData: { width: 1, height: 1, data: new Uint8ClampedArray(4) },
    dataMin: 0,
    dataMax: 1,
  }),
  colorizeDataTexture: vi.fn().mockResolvedValue("blob:mock-url"),
}));

vi.mock("@/lib/colormapLUT", () => ({
  getColormapLUT: vi.fn().mockResolvedValue(new Uint8Array(1024)),
  prefetchColormapLUT: vi.fn(),
}));

function setupStore(overrides: Record<string, unknown> = {}) {
  useAppStore.setState({
    numFrames: 0,
    hasResults: false,
    displayComponent: "u",
    referenceFrame: 0,
    resultVersion: 1,
    visSettings: {
      colormap: "turbo",
      alpha: 0.7,
      background: "reference",
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
    ...overrides,
  });
}

describe("usePreRenderCache", () => {
  beforeEach(() => {
    setupStore();
    vi.clearAllMocks();
    // Mock URL.revokeObjectURL (not available in jsdom)
    vi.stubGlobal("URL", {
      ...globalThis.URL,
      createObjectURL: vi.fn(() => "blob:mock-created"),
      revokeObjectURL: vi.fn(),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("initial state", () => {
    it("should not be pre-rendering initially", () => {
      const { result } = renderHook(() => usePreRenderCache());
      expect(result.current.isPreRendering).toBe(false);
    });

    it("should have 0 progress initially", () => {
      const { result } = renderHook(() => usePreRenderCache());
      expect(result.current.progress).toBe(0);
    });

    it("should have 0 cached count initially", () => {
      const { result } = renderHook(() => usePreRenderCache());
      expect(result.current.cachedCount).toBe(0);
    });

    it("should report totalFrames from store", () => {
      setupStore({ numFrames: 15, hasResults: true });
      const { result } = renderHook(() => usePreRenderCache());
      expect(result.current.totalFrames).toBe(15);
    });
  });

  describe("getFrame", () => {
    it("should return undefined for uncached frame", () => {
      setupStore({ numFrames: 10, hasResults: true });
      const { result } = renderHook(() => usePreRenderCache());
      expect(result.current.getFrame(0)).toBeUndefined();
    });
  });

  describe("startPreRender", () => {
    it("should not start when there are no results", () => {
      setupStore({ numFrames: 0, hasResults: false });
      const { result } = renderHook(() => usePreRenderCache());

      act(() => {
        result.current.startPreRender();
      });

      // Should remain not pre-rendering because hasResults is false
      expect(result.current.isPreRendering).toBe(false);
    });

    it("should start pre-rendering when results exist", async () => {
      // Mock fetch for data texture fetching
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({
          "X-Data-Min": "0",
          "X-Data-Max": "1",
        }),
        blob: () => Promise.resolve(new Blob(["fake"])),
      });
      vi.stubGlobal("fetch", mockFetch);

      setupStore({ numFrames: 3, hasResults: true });
      const { result } = renderHook(() => usePreRenderCache());

      act(() => {
        result.current.startPreRender();
      });

      expect(result.current.isPreRendering).toBe(true);
    });

    it("should fetch with correct URL pattern for displacement", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({
          "X-Data-Min": "-2",
          "X-Data-Max": "2",
        }),
        blob: () => Promise.resolve(new Blob(["fake"])),
      });
      vi.stubGlobal("fetch", mockFetch);

      setupStore({ numFrames: 1, hasResults: true, displayComponent: "u" });
      const { result } = renderHook(() => usePreRenderCache());

      act(() => {
        result.current.startPreRender();
      });

      // Wait for the async fetch to be called
      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });

      const callUrl = mockFetch.mock.calls[0][0] as string;
      expect(callUrl).toContain("/api/displacement/render/0");
      expect(callUrl).toContain("component=u");
      expect(callUrl).toContain("render_mode=data");
    });

    it("should fetch with strain URL for strain components", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({
          "X-Data-Min": "0",
          "X-Data-Max": "0.01",
        }),
        blob: () => Promise.resolve(new Blob(["fake"])),
      });
      vi.stubGlobal("fetch", mockFetch);

      setupStore({ numFrames: 1, hasResults: true, displayComponent: "exx" });
      const { result } = renderHook(() => usePreRenderCache("exx"));

      act(() => {
        result.current.startPreRender();
      });

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });

      const callUrl = mockFetch.mock.calls[0][0] as string;
      expect(callUrl).toContain("/api/strain/render/0");
      expect(callUrl).toContain("component=exx");
    });
  });

  describe("cancelPreRender", () => {
    it("should set isPreRendering to false when cancelled", () => {
      const mockFetch = vi.fn().mockImplementation(
        () => new Promise(() => {}), // never resolves — simulates in-flight
      );
      vi.stubGlobal("fetch", mockFetch);

      setupStore({ numFrames: 5, hasResults: true });
      const { result } = renderHook(() => usePreRenderCache());

      act(() => {
        result.current.startPreRender();
      });
      expect(result.current.isPreRendering).toBe(true);

      act(() => {
        result.current.cancelPreRender();
      });
      expect(result.current.isPreRendering).toBe(false);
    });
  });

  describe("invalidate", () => {
    it("should reset cached count and progress", () => {
      const { result } = renderHook(() => usePreRenderCache());

      act(() => {
        result.current.invalidate();
      });

      expect(result.current.cachedCount).toBe(0);
      expect(result.current.progress).toBe(0);
    });
  });

  describe("concurrency", () => {
    it("should limit concurrent fetches to 3 workers", async () => {
      let activeFetches = 0;
      let maxActiveFetches = 0;

      const mockFetch = vi.fn().mockImplementation(async () => {
        activeFetches++;
        maxActiveFetches = Math.max(maxActiveFetches, activeFetches);
        // Simulate some latency
        await new Promise((r) => setTimeout(r, 10));
        activeFetches--;
        return {
          ok: true,
          headers: new Headers({
            "X-Data-Min": "0",
            "X-Data-Max": "1",
          }),
          blob: () => Promise.resolve(new Blob(["fake"])),
        };
      });
      vi.stubGlobal("fetch", mockFetch);

      setupStore({ numFrames: 10, hasResults: true });
      const { result } = renderHook(() => usePreRenderCache());

      await act(async () => {
        result.current.startPreRender();
        // Wait for all fetches to complete
        await new Promise((r) => setTimeout(r, 200));
      });

      // CONCURRENCY constant in the hook is 3
      expect(maxActiveFetches).toBeLessThanOrEqual(3);
    });
  });

  describe("componentOverride", () => {
    it("should use componentOverride instead of store component", async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        headers: new Headers({ "X-Data-Min": "0", "X-Data-Max": "1" }),
        blob: () => Promise.resolve(new Blob(["fake"])),
      });
      vi.stubGlobal("fetch", mockFetch);

      setupStore({ numFrames: 1, hasResults: true, displayComponent: "u" });
      const { result } = renderHook(() => usePreRenderCache("v"));

      act(() => {
        result.current.startPreRender();
      });

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });

      const callUrl = mockFetch.mock.calls[0][0] as string;
      expect(callUrl).toContain("component=v");
    });
  });
});
