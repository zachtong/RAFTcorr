import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useFrameNav } from "@/hooks/useFrameNav";
import { useAppStore } from "@/stores/appStore";

function getState() {
  return useAppStore.getState();
}

/**
 * Prepare the store with a known number of frames and a starting frame.
 */
function setupFrames(numFrames: number, currentFrame = 0) {
  useAppStore.setState({ numFrames, currentFrame, hasResults: numFrames > 0 });
}

describe("useFrameNav", () => {
  beforeEach(() => {
    setupFrames(10, 0);
    // Mock requestAnimationFrame / cancelAnimationFrame for play/pause tests
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((cb) => {
      return setTimeout(() => cb(performance.now()), 0) as unknown as number;
    });
    vi.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      clearTimeout(id);
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("frame navigation", () => {
    it("should return current frame and total frames from store", () => {
      const { result } = renderHook(() => useFrameNav());
      expect(result.current.currentFrame).toBe(0);
      expect(result.current.numFrames).toBe(10);
    });

    it("next() should advance frame by 1", () => {
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.next();
      });
      expect(getState().currentFrame).toBe(1);
    });

    it("prev() should go back frame by 1", () => {
      setupFrames(10, 5);
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.prev();
      });
      expect(getState().currentFrame).toBe(4);
    });

    it("first() should go to frame 0", () => {
      setupFrames(10, 7);
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.first();
      });
      expect(getState().currentFrame).toBe(0);
    });

    it("last() should go to the last frame", () => {
      setupFrames(10, 0);
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.last();
      });
      expect(getState().currentFrame).toBe(9);
    });

    it("goTo() should jump to a specific frame", () => {
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.goTo(5);
      });
      expect(getState().currentFrame).toBe(5);
    });
  });

  describe("boundary clamping", () => {
    it("next() should not exceed numFrames - 1", () => {
      setupFrames(10, 9); // at last frame
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.next();
      });
      expect(getState().currentFrame).toBe(9);
    });

    it("prev() should not go below 0", () => {
      setupFrames(10, 0); // at first frame
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.prev();
      });
      expect(getState().currentFrame).toBe(0);
    });

    it("goTo() should clamp to valid range (upper bound)", () => {
      setupFrames(10, 0);
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.goTo(100);
      });
      expect(getState().currentFrame).toBe(9);
    });

    it("goTo() should clamp to valid range (lower bound)", () => {
      setupFrames(10, 5);
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.goTo(-5);
      });
      expect(getState().currentFrame).toBe(0);
    });
  });

  describe("play/pause", () => {
    it("should start in non-playing state", () => {
      const { result } = renderHook(() => useFrameNav());
      expect(result.current.isPlaying).toBe(false);
    });

    it("play() should set isPlaying to true", () => {
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.play();
      });
      expect(result.current.isPlaying).toBe(true);
    });

    it("pause() should set isPlaying to false", () => {
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.play();
      });
      expect(result.current.isPlaying).toBe(true);

      act(() => {
        result.current.pause();
      });
      expect(result.current.isPlaying).toBe(false);
    });

    it("play() should wrap to frame 0 when already at last frame", () => {
      setupFrames(10, 9); // at last frame
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.play();
      });
      // Should have reset to 0 before starting playback
      expect(getState().currentFrame).toBe(0);
    });

    it("calling play() twice should not double-start (idempotent)", () => {
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.play();
      });
      const rafCallCount = (window.requestAnimationFrame as ReturnType<typeof vi.fn>).mock.calls.length;
      act(() => {
        result.current.play();
      });
      // No additional rAF calls because play() early-returns if already playing
      expect(
        (window.requestAnimationFrame as ReturnType<typeof vi.fn>).mock.calls.length
      ).toBe(rafCallCount);
    });
  });

  describe("setFps", () => {
    it("should allow changing FPS without errors", () => {
      const { result } = renderHook(() => useFrameNav());
      act(() => {
        result.current.setFps(30);
      });
      // No error thrown — FPS is stored in a ref and read dynamically
    });
  });

  describe("cleanup on unmount", () => {
    it("should cancel animation frame on unmount", () => {
      const { result, unmount } = renderHook(() => useFrameNav());
      act(() => {
        result.current.play();
      });
      unmount();
      // cancelAnimationFrame should have been called during cleanup
      expect(window.cancelAnimationFrame).toHaveBeenCalled();
    });
  });
});
