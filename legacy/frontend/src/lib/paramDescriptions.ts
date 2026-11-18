/**
 * Human-readable parameter descriptions for tooltip display.
 *
 * Keys match parameter names used in ProcessingParams and StrainControls.
 */
export const PARAM_DESCRIPTIONS: Record<string, string> = {
  // --- Processing parameters ---
  memoryBudget:
    "Maximum pixels per tile (controls GPU memory usage).\n" +
    "Larger = fewer tiles & faster, but may cause out-of-memory (OOM).\n" +
    "Recommended: use model default.",
  blendQuality:
    "Tile stitching quality.\n" +
    "Fast: minimal overlap, fastest processing\n" +
    "Balanced: moderate overlap, recommended default\n" +
    "High: maximum overlap, fewest seams but slower",
  tileOverlap:
    "Overlap in pixels between adjacent tiles.\n" +
    "More overlap = smoother seam transitions, but more computation.\n" +
    "Recommended: 32-128 px",
  roiMargin:
    "Extra pixels beyond the ROI boundary (context region).\n" +
    "Provides boundary context for the RAFT optical flow network.\n" +
    "Recommended: 64 px",
  smoothingRadius:
    "Standard deviation of Gaussian smoothing kernel (pixels).\n" +
    "Larger = smoother displacement field, but loses detail.\n" +
    "0 = no smoothing. Recommended: 0-3",
  mode:
    "Processing mode:\n" +
    "• Accumulative: displacement relative to the first frame (suited for small deformations)\n" +
    "• Incremental: frame-to-frame displacement accumulation (suited for large deformations, crack propagation)",

  // --- Incremental mode parameters ---
  keyFrameMode:
    "Reference frame update strategy:\n" +
    "• Every Frame: update reference every frame\n" +
    "• Every N: update every N frames\n" +
    "• Custom: manually specify key frames",
  keyFrameInterval:
    "Reference frame update interval (in frames).\n" +
    "Smaller interval = more accurate but faster error accumulation.\n" +
    "Recommended: 5-20",
  maskSource:
    "Crack mask source:\n" +
    "• Auto: automatically warp the ROI mask\n" +
    "• Custom: load pre-made masks from a folder",
  useMedianFilter:
    "Apply median filter to accumulated displacement.\n" +
    "Reduces error accumulation in incremental mode, but modifies raw output.",

  // --- Strain parameters ---
  strainMethod:
    "Strain computation method:\n" +
    "• Green-Lagrange: rigorous formula accounting for large-rotation deformation\n" +
    "• Engineering: simplified small-deformation approximation",
  vsgSize:
    "Virtual Strain Gauge (VSG) window size in pixels.\n" +
    "Must be odd. Larger = smoother but lower spatial resolution.\n" +
    "Recommended: 9-51",
  step:
    "VSG step size in pixels.\n" +
    "Output strain map spatial resolution = VSG / step.\n" +
    "Larger = faster computation but lower resolution.",
  polyOrder:
    "Polynomial order for fitting within the VSG.\n" +
    "1 = linear (recommended), 2 = quadratic (more accurate but needs larger VSG)",
  weighting:
    "Weight distribution for data points within the VSG:\n" +
    "• Gaussian: higher weight at center, lower at edges (recommended)\n" +
    "• Uniform: equal weight everywhere",
  gaussianSigma:
    "Gaussian weight standard deviation within the VSG.\n" +
    "Leave empty = auto (VSG/6). Smaller values concentrate on the center.",
  spatialSigma:
    "Gaussian smoothing of the displacement field before strain computation.\n" +
    "Reduces displacement noise impact on strain.\n" +
    "0 = no smoothing. Recommended: 0-5 px",
  temporalSigma:
    "Gaussian smoothing of strain in the temporal direction.\n" +
    "Reduces frame-to-frame strain fluctuations.\n" +
    "0 = no smoothing. Recommended: 0-5 frames",
  condThreshold:
    "Condition number threshold (numerical stability filter).\n" +
    "Points where the VSG condition number exceeds this value are set to NaN.\n" +
    "Recommended: 100-10000",
  erosion:
    "Boundary erosion distance in pixels.\n" +
    "Removes unreliable strain values near the ROI boundary.\n" +
    "Auto = VSG/2",
};
