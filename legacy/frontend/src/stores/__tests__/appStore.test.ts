import { describe, it, expect, beforeEach } from "vitest";
import { useAppStore } from "@/stores/appStore";

/**
 * Helper to get current store state (non-reactive snapshot).
 */
function getState() {
  return useAppStore.getState();
}

/**
 * Reset store to initial defaults before each test.
 */
function resetStore() {
  // Trigger built-in resetSession to clear most fields
  getState().resetSession();
  // Also reset fields that resetSession intentionally preserves
  useAppStore.setState({
    imageDir: "",
    imageFiles: [],
    selectedModel: "",
    modelMetadata: null,
    mode: "accumulative",
    resultVersion: 0,
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
  });
}

describe("appStore", () => {
  beforeEach(() => {
    resetStore();
  });

  describe("initial state", () => {
    it("should have currentFrame = 0", () => {
      expect(getState().currentFrame).toBe(0);
    });

    it("should have imageWidth = 0 and imageHeight = 0", () => {
      expect(getState().imageWidth).toBe(0);
      expect(getState().imageHeight).toBe(0);
    });

    it("should have no results", () => {
      expect(getState().hasResults).toBe(false);
      expect(getState().numFrames).toBe(0);
    });

    it("should have processing inactive", () => {
      expect(getState().processingActive).toBe(false);
      expect(getState().processingPaused).toBe(false);
      expect(getState().processingProgress).toBe(0);
    });

    it("should have no strain data", () => {
      expect(getState().hasStrain).toBe(false);
      expect(getState().strainComputing).toBe(false);
      expect(getState().strainComponents).toEqual([]);
    });

    it("should have default visSettings", () => {
      const vis = getState().visSettings;
      expect(vis.colormap).toBe("turbo");
      expect(vis.alpha).toBe(0.7);
      expect(vis.background).toBe("reference");
      expect(vis.logScale).toBe(false);
    });

    it("should have default display component 'u'", () => {
      expect(getState().displayComponent).toBe("u");
    });

    it("should have empty probes array", () => {
      expect(getState().probes).toEqual([]);
    });

    it("should have default zoom = 1", () => {
      expect(getState().viewZoom).toBe(1);
      expect(getState().viewOffset).toEqual({ x: 0, y: 0 });
    });
  });

  describe("setCurrentFrame", () => {
    it("should update the current frame index", () => {
      getState().setCurrentFrame(5);
      expect(getState().currentFrame).toBe(5);
    });

    it("should allow setting to 0", () => {
      getState().setCurrentFrame(10);
      getState().setCurrentFrame(0);
      expect(getState().currentFrame).toBe(0);
    });
  });

  describe("resetSession", () => {
    it("should clear processing state", () => {
      getState().setProcessing(true, 50, 3, 10);
      getState().resetSession();

      expect(getState().processingActive).toBe(false);
      expect(getState().processingPaused).toBe(false);
      expect(getState().processingProgress).toBe(0);
      expect(getState().processingCurrent).toBe(0);
      expect(getState().processingTotal).toBe(0);
    });

    it("should clear result state", () => {
      getState().setResults(20);
      getState().setCurrentFrame(15);
      getState().resetSession();

      expect(getState().hasResults).toBe(false);
      expect(getState().numFrames).toBe(0);
      expect(getState().currentFrame).toBe(0);
    });

    it("should clear strain state", () => {
      getState().setStrain(true, ["exx", "eyy"]);
      getState().resetSession();

      expect(getState().hasStrain).toBe(false);
      expect(getState().strainComponents).toEqual([]);
    });

    it("should reset display component to 'u'", () => {
      getState().setDisplayComponent("exx");
      getState().resetSession();
      expect(getState().displayComponent).toBe("u");
    });

    it("should clear probes", () => {
      getState().setProbes([
        { id: 1, type: "point", coords: [100, 200], color: "#ff0000", label: "P1" },
      ]);
      getState().resetSession();
      expect(getState().probes).toEqual([]);
    });

    it("should reset zoom and offset", () => {
      getState().zoomIn();
      getState().setViewOffset({ x: 50, y: 50 });
      getState().resetSession();

      expect(getState().viewZoom).toBe(1);
      expect(getState().viewOffset).toEqual({ x: 0, y: 0 });
    });

    it("should clear image dimensions but preserve imageDir", () => {
      getState().setImageDir("/test/images");
      getState().setImageFiles(["a.png", "b.png"], 640, 480);
      getState().resetSession();

      // imageDir is intentionally preserved by resetSession
      expect(getState().imageDir).toBe("/test/images");
      // image dimensions are cleared
      expect(getState().imageWidth).toBe(0);
      expect(getState().imageHeight).toBe(0);
    });

    it("should reset export state", () => {
      getState().setExport(true, 75);
      getState().resetSession();

      expect(getState().exportActive).toBe(false);
      expect(getState().exportProgress).toBe(0);
    });

    it("should reset incremental settings to defaults", () => {
      getState().setKeyFrameMode("custom");
      getState().setKeyFrameInterval(50);
      getState().setUseMedianFilter(true);
      getState().resetSession();

      expect(getState().keyFrameMode).toBe("every_frame");
      expect(getState().keyFrameInterval).toBe(10);
      expect(getState().useMedianFilter).toBe(false);
    });
  });

  describe("visSettings updates", () => {
    it("should update colormap while preserving other settings", () => {
      getState().updateVisSettings({ colormap: "jet" });
      const vis = getState().visSettings;
      expect(vis.colormap).toBe("jet");
      // other fields unchanged
      expect(vis.alpha).toBe(0.7);
      expect(vis.background).toBe("reference");
    });

    it("should update alpha", () => {
      getState().updateVisSettings({ alpha: 0.5 });
      expect(getState().visSettings.alpha).toBe(0.5);
    });

    it("should support updating multiple fields at once", () => {
      getState().updateVisSettings({
        colormap: "viridis",
        logScale: true,
        fps: 10,
      });
      const vis = getState().visSettings;
      expect(vis.colormap).toBe("viridis");
      expect(vis.logScale).toBe(true);
      expect(vis.fps).toBe(10);
    });

    it("should update fixed range and vmin/vmax values", () => {
      getState().updateVisSettings({
        fixedRange: true,
        vminU: "-5",
        vmaxU: "5",
      });
      const vis = getState().visSettings;
      expect(vis.fixedRange).toBe(true);
      expect(vis.vminU).toBe("-5");
      expect(vis.vmaxU).toBe("5");
    });
  });

  describe("processing state", () => {
    it("should activate processing with progress info", () => {
      getState().setProcessing(true, 25, 5, 20);

      expect(getState().processingActive).toBe(true);
      expect(getState().processingPaused).toBe(false);
      expect(getState().processingProgress).toBe(25);
      expect(getState().processingCurrent).toBe(5);
      expect(getState().processingTotal).toBe(20);
    });

    it("should deactivate processing and reset progress", () => {
      getState().setProcessing(true, 50, 10, 20);
      getState().setProcessing(false);

      expect(getState().processingActive).toBe(false);
      expect(getState().processingProgress).toBe(0);
    });

    it("should toggle paused state independently", () => {
      getState().setProcessing(true, 50, 10, 20);
      getState().setProcessingPaused(true);

      expect(getState().processingActive).toBe(true);
      expect(getState().processingPaused).toBe(true);
    });

    it("should reset paused when setProcessing is called", () => {
      getState().setProcessingPaused(true);
      getState().setProcessing(true, 0, 0, 10);
      expect(getState().processingPaused).toBe(false);
    });
  });

  describe("setResults", () => {
    it("should set hasResults when numFrames > 0", () => {
      getState().setResults(10);
      expect(getState().hasResults).toBe(true);
      expect(getState().numFrames).toBe(10);
      expect(getState().currentFrame).toBe(0);
    });

    it("should clear hasResults when numFrames = 0", () => {
      getState().setResults(10);
      getState().setResults(0);
      expect(getState().hasResults).toBe(false);
      expect(getState().numFrames).toBe(0);
    });

    it("should increment resultVersion", () => {
      const before = getState().resultVersion;
      getState().setResults(5);
      expect(getState().resultVersion).toBe(before + 1);
    });

    it("should reset currentFrame to 0", () => {
      getState().setCurrentFrame(7);
      getState().setResults(10);
      expect(getState().currentFrame).toBe(0);
    });
  });

  describe("setStrain", () => {
    it("should set strain with components", () => {
      getState().setStrain(true, ["exx", "eyy", "exy"]);
      expect(getState().hasStrain).toBe(true);
      expect(getState().strainComponents).toEqual(["exx", "eyy", "exy"]);
    });

    it("should clear strainComputing flag", () => {
      getState().setStrainComputing(true, 50);
      getState().setStrain(true, ["exx"]);
      expect(getState().strainComputing).toBe(false);
    });

    it("should increment resultVersion", () => {
      const before = getState().resultVersion;
      getState().setStrain(true, ["exx"]);
      expect(getState().resultVersion).toBe(before + 1);
    });
  });

  describe("setRoiConfirmed", () => {
    it("should flip the confirmed flag without touching roiVersion", () => {
      const before = getState().roiVersion;
      getState().setRoiConfirmed(true);
      expect(getState().roiConfirmed).toBe(true);
      expect(getState().roiVersion).toBe(before);
      getState().setRoiConfirmed(false);
      expect(getState().roiConfirmed).toBe(false);
      expect(getState().roiVersion).toBe(before);
    });
  });

  describe("bumpRoiVersion", () => {
    it("increments roiVersion on every call", () => {
      const before = getState().roiVersion;
      getState().bumpRoiVersion();
      expect(getState().roiVersion).toBe(before + 1);
      getState().bumpRoiVersion();
      expect(getState().roiVersion).toBe(before + 2);
    });
  });

  describe("zoom actions", () => {
    it("should increase zoom on zoomIn", () => {
      getState().zoomIn();
      expect(getState().viewZoom).toBeCloseTo(1.25);
    });

    it("should decrease zoom on zoomOut", () => {
      getState().zoomOut();
      expect(getState().viewZoom).toBeCloseTo(0.8);
    });

    it("should cap zoom at max 5", () => {
      // Zoom in many times
      for (let i = 0; i < 20; i++) getState().zoomIn();
      expect(getState().viewZoom).toBeLessThanOrEqual(5);
    });

    it("should cap zoom at min 0.2", () => {
      // Zoom out many times
      for (let i = 0; i < 20; i++) getState().zoomOut();
      expect(getState().viewZoom).toBeGreaterThanOrEqual(0.2);
    });

    it("should reset zoom and offset on zoomReset", () => {
      getState().zoomIn();
      getState().zoomIn();
      getState().setViewOffset({ x: 100, y: 200 });
      getState().zoomReset();

      expect(getState().viewZoom).toBe(1);
      expect(getState().viewOffset).toEqual({ x: 0, y: 0 });
    });
  });

  describe("probe actions", () => {
    it("should set probe placing mode and clear related state", () => {
      getState().setProbePlacingFirst([100, 200]);
      getState().addAreaPolyPoint(10, 20);
      getState().setProbePlacingMode("line");

      expect(getState().probePlacingMode).toBe("line");
      expect(getState().probePlacingFirst).toBeNull();
      expect(getState().areaPolyPoints).toEqual([]);
    });

    it("should accumulate area polygon points", () => {
      getState().addAreaPolyPoint(10, 20);
      getState().addAreaPolyPoint(30, 40);
      getState().addAreaPolyPoint(50, 60);

      expect(getState().areaPolyPoints).toEqual([
        [10, 20],
        [30, 40],
        [50, 60],
      ]);
    });

    it("should clear area polygon points", () => {
      getState().addAreaPolyPoint(10, 20);
      getState().clearAreaPolyPoints();
      expect(getState().areaPolyPoints).toEqual([]);
    });
  });

  describe("arrowSettings updates", () => {
    it("should update arrow settings while preserving others", () => {
      getState().updateArrowSettings({ showQuiver: true, spacing: 20 });
      const arrows = getState().arrowSettings;
      expect(arrows.showQuiver).toBe(true);
      expect(arrows.spacing).toBe(20);
      // others unchanged
      expect(arrows.color).toBe("white");
      expect(arrows.lineWidth).toBe(1.5);
    });
  });

  describe("setImageFiles", () => {
    it("should set files and dimensions together", () => {
      getState().setImageFiles(["img001.png", "img002.png"], 1920, 1080);
      expect(getState().imageFiles).toEqual(["img001.png", "img002.png"]);
      expect(getState().imageWidth).toBe(1920);
      expect(getState().imageHeight).toBe(1080);
    });
  });

  describe("restoreFromSession", () => {
    it("should restore state from a session summary", () => {
      const before = getState().resultVersion;
      getState().restoreFromSession({
        num_displacement_frames: 25,
        num_strain_frames: 25,
        strain_components: ["exx", "eyy"],
        has_roi: true,
        roi_confirmed: true,
        image_dir: "/data/experiment",
        image_dir_exists: true,
        image_width: 800,
        image_height: 600,
      });

      expect(getState().imageDir).toBe("/data/experiment");
      expect(getState().imageWidth).toBe(800);
      expect(getState().imageHeight).toBe(600);
      expect(getState().hasResults).toBe(true);
      expect(getState().numFrames).toBe(25);
      expect(getState().currentFrame).toBe(0);
      expect(getState().roiConfirmed).toBe(true);
      expect(getState().hasStrain).toBe(true);
      expect(getState().strainComponents).toEqual(["exx", "eyy"]);
      expect(getState().displayComponent).toBe("u");
      expect(getState().resultVersion).toBe(before + 1);
    });

    it("should set hasResults false when no displacement frames", () => {
      getState().restoreFromSession({
        num_displacement_frames: 0,
        num_strain_frames: 0,
        strain_components: [],
        has_roi: false,
        roi_confirmed: false,
        image_dir: "/empty",
        image_dir_exists: true,
        image_width: 0,
        image_height: 0,
      });
      expect(getState().hasResults).toBe(false);
      expect(getState().hasStrain).toBe(false);
    });
  });

  describe("export state", () => {
    it("should set export active with progress", () => {
      getState().setExport(true, 42);
      expect(getState().exportActive).toBe(true);
      expect(getState().exportProgress).toBe(42);
    });

    it("should default progress to 0 when not specified", () => {
      getState().setExport(true);
      expect(getState().exportProgress).toBe(0);
    });
  });

  describe("colorbar settings", () => {
    it("should update colorbar settings partially", () => {
      getState().updateColorbarSettings({ labelText: "Strain (mm)", tickCount: 5 });
      const cb = getState().colorbarSettings;
      expect(cb.labelText).toBe("Strain (mm)");
      expect(cb.tickCount).toBe(5);
      // others unchanged
      expect(cb.labelFontSize).toBe(12);
      expect(cb.barThickness).toBe(1.0);
    });
  });
});
