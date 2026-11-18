import { describe, it, expect, beforeEach } from "vitest";
import { useRoiStore } from "@/stores/roiStore";

function getState() {
  return useRoiStore.getState();
}

function resetStore() {
  useRoiStore.setState({
    drawingMode: null,
    cutMode: false,
    currentPoints: [],
    maskUrl: null,
    showImportDialog: false,
  });
}

describe("roiStore", () => {
  beforeEach(() => {
    resetStore();
  });

  describe("initial state", () => {
    it("should have no drawing mode active", () => {
      expect(getState().drawingMode).toBeNull();
    });

    it("should have cutMode disabled", () => {
      expect(getState().cutMode).toBe(false);
    });

    it("should have empty currentPoints", () => {
      expect(getState().currentPoints).toEqual([]);
    });

    it("should have no mask URL", () => {
      expect(getState().maskUrl).toBeNull();
    });
  });

  describe("activateTool", () => {
    it("should set polygon drawing mode", () => {
      getState().activateTool("polygon", false);
      expect(getState().drawingMode).toBe("polygon");
      expect(getState().cutMode).toBe(false);
    });

    it("should set rectangle drawing mode with cut", () => {
      getState().activateTool("rectangle", true);
      expect(getState().drawingMode).toBe("rectangle");
      expect(getState().cutMode).toBe(true);
    });

    it("should set circle drawing mode", () => {
      getState().activateTool("circle", false);
      expect(getState().drawingMode).toBe("circle");
      expect(getState().cutMode).toBe(false);
    });

    it("should toggle off when same tool re-clicked (same shape + same cut)", () => {
      getState().activateTool("polygon", false);
      expect(getState().drawingMode).toBe("polygon");

      // Click same tool again — should toggle off
      getState().activateTool("polygon", false);
      expect(getState().drawingMode).toBeNull();
    });

    it("should NOT toggle off if shape same but cut mode differs", () => {
      getState().activateTool("polygon", false);
      // Same shape, different cut mode — should switch, not toggle off
      getState().activateTool("polygon", true);
      expect(getState().drawingMode).toBe("polygon");
      expect(getState().cutMode).toBe(true);
    });

    it("should switch to new tool when different shape clicked", () => {
      getState().activateTool("polygon", false);
      getState().activateTool("rectangle", false);
      expect(getState().drawingMode).toBe("rectangle");
    });

    it("should clear currentPoints when activating a tool", () => {
      getState().addPoint(10, 20);
      getState().addPoint(30, 40);
      expect(getState().currentPoints).toHaveLength(2);

      getState().activateTool("rectangle", false);
      expect(getState().currentPoints).toEqual([]);
    });

    it("should clear currentPoints when toggling off", () => {
      getState().activateTool("polygon", false);
      getState().addPoint(10, 20);
      // Toggle off
      getState().activateTool("polygon", false);
      expect(getState().currentPoints).toEqual([]);
    });
  });

  describe("setDrawingMode", () => {
    it("should set drawing mode directly and clear points", () => {
      getState().addPoint(10, 20);
      getState().setDrawingMode("rectangle");
      expect(getState().drawingMode).toBe("rectangle");
      expect(getState().currentPoints).toEqual([]);
    });

    it("should allow setting to null (deactivate)", () => {
      getState().setDrawingMode("polygon");
      getState().setDrawingMode(null);
      expect(getState().drawingMode).toBeNull();
    });
  });

  describe("addPoint / clearPoints", () => {
    it("should accumulate points in order", () => {
      getState().addPoint(10, 20);
      getState().addPoint(30, 40);
      getState().addPoint(50, 60);

      expect(getState().currentPoints).toEqual([
        [10, 20],
        [30, 40],
        [50, 60],
      ]);
    });

    it("should not mutate previous points array (immutability)", () => {
      getState().addPoint(10, 20);
      const first = getState().currentPoints;
      getState().addPoint(30, 40);
      const second = getState().currentPoints;

      // References should be different (new array each time)
      expect(first).not.toBe(second);
      // Original array unchanged
      expect(first).toEqual([[10, 20]]);
    });

    it("should clear all points", () => {
      getState().addPoint(1, 2);
      getState().addPoint(3, 4);
      getState().clearPoints();
      expect(getState().currentPoints).toEqual([]);
    });
  });

  describe("setCutMode", () => {
    it("should set cut mode independently", () => {
      getState().setCutMode(true);
      expect(getState().cutMode).toBe(true);
    });

    it("should not affect drawing mode", () => {
      getState().activateTool("polygon", false);
      getState().setCutMode(true);
      expect(getState().drawingMode).toBe("polygon");
    });
  });

  describe("setMaskUrl", () => {
    it("should set mask URL", () => {
      getState().setMaskUrl("blob:http://localhost/abc123");
      expect(getState().maskUrl).toBe("blob:http://localhost/abc123");
    });

    it("should clear mask URL when set to null", () => {
      getState().setMaskUrl("blob:http://localhost/abc123");
      getState().setMaskUrl(null);
      expect(getState().maskUrl).toBeNull();
    });
  });

  describe("setShowImportDialog", () => {
    it("should toggle import dialog visibility", () => {
      getState().setShowImportDialog(true);
      expect(getState().showImportDialog).toBe(true);

      getState().setShowImportDialog(false);
      expect(getState().showImportDialog).toBe(false);
    });
  });

  describe("drawing mode transitions", () => {
    it("should handle full workflow: activate → draw points → switch tool → points cleared", () => {
      getState().activateTool("polygon", false);
      getState().addPoint(10, 20);
      getState().addPoint(30, 40);
      expect(getState().currentPoints).toHaveLength(2);

      // Switch to rectangle
      getState().activateTool("rectangle", false);
      expect(getState().drawingMode).toBe("rectangle");
      expect(getState().currentPoints).toEqual([]);
    });

    it("should handle activate → draw → deactivate → reactivate cycle", () => {
      getState().activateTool("circle", false);
      getState().addPoint(100, 100);

      // Toggle off
      getState().activateTool("circle", false);
      expect(getState().drawingMode).toBeNull();
      expect(getState().currentPoints).toEqual([]);

      // Reactivate
      getState().activateTool("circle", false);
      expect(getState().drawingMode).toBe("circle");
      expect(getState().currentPoints).toEqual([]);
    });
  });
});
