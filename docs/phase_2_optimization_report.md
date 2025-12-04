# RAFT-DIC Development Report - Phase 2
**Date:** 2025-12-01
**Phase:** 2 - Core Algorithm Optimization & Memory Management
**Previous Phase:** Phase 1 - Initial GUI Refactoring (Tabbed Interface)

## 1. Executive Summary
This phase focused on "hardening" the application against large image datasets and varying hardware constraints. We moved from a naive "one-size-fits-all" processing approach to a dynamic, resource-aware architecture. Key achievements include solving persistent OOM errors, implementing adaptive tiling, and stabilizing the visualization pipeline.

## 2. Key Technical Achievements

### 2.1 Intelligent Memory Management
*   **Safety Factor Mechanism**: Introduced a tunable `safety_factor` (0.2 - 1.0) to control VRAM aggressiveness.
*   **Shared Memory Offloading**: Implemented a "Shared Memory" mode (Safety Factor > 1.0) that allows the system to utilize host RAM for larger tiles when VRAM is insufficient.
*   **Active VRAM Cleanup**: Added forced garbage collection (`gc.collect()` + `torch.cuda.empty_cache()`) before critical estimation steps to ensure model switching doesn't lead to false OOM detections.

### 2.2 Dynamic Tiling Engine
*   **ROI-Aware Tiling**: The tiling engine now calculates grid dimensions based on the **ROI size** rather than the full image.
    *   *Impact*: Small ROIs on 4K images now run in "Direct Processing" mode (instant) instead of triggering unnecessary tiling.
*   **Adaptive Grid**: Automatically calculates `nx` x `ny` tiles based on `safe_pmax` (derived from model metadata) and `overlap`.

### 2.3 Visualization Stability
*   **Layout Refactoring**: Solved the "shrinking image" bug by using `mpl_toolkits.axes_grid1.make_axes_locatable`. This enforces a strict layout where colorbars have dedicated space and do not encroach on the data plot.
*   **Robust State Handling**: Added error recovery for Colorbar updates to prevent crashes when toggling visibility or changing parameters rapidly.

## 3. Technical Deep Dive (Algorithm Logic)

### 3.1 VRAM Estimation Formula
The system estimates the maximum safe pixel count ($P_{max}$) to avoid OOM errors using the following logic:

$$ P_{max} = \frac{VRAM_{free} \times SafetyFactor}{BytesPerPixel} $$

*   **$VRAM_{free}$**: Obtained via `torch.cuda.mem_get_info(0)`. Crucially, we run `torch.cuda.empty_cache()` before this check to clear fragmented memory from previous models.
*   **$SafetyFactor$**: User-configurable (0.2 - 1.0).
    *   *Note*: RAFT-Fine models use a stricter internal multiplier (0.45x) on top of this because they maintain higher-resolution feature maps (1/8th resolution vs 1/8th + 1/16th).
*   **$BytesPerPixel$**: Empirically determined constant (~14,000 bytes/pixel for FP32 inference including gradients and optimizer states, though inference-only is lower, we stay conservative).

### 3.2 Adaptive Tiling Logic
If the ROI area exceeds $P_{max}$, the image is split into tiles. The grid dimensions ($N_x, N_y$) are calculated as:

$$ N_x = \lceil \frac{W_{roi} - Overlap}{Stride} \rceil $$

Where:
*   $Stride = \sqrt{P_{max}} - Overlap$
*   $W_{roi}$: Width of the selected Region of Interest (not full image).
*   $Overlap$: User-defined context overlap (default 32px) to ensure continuity at boundaries.

### 3.3 OOM Recovery Flow
The inference pipeline (`dic_over_roi_with_tiling`) implements a self-healing mechanism:
1.  **Attempt 1**: Run inference on the calculated tile grid.
2.  **Catch `RuntimeError`**: If CUDA OOM occurs:
    *   Trigger `torch.cuda.empty_cache()`.
    *   **Retry**: Re-run the failed tile/batch with a reduced batch size or strictly sequential processing.
    *   *Future*: If sequential fails, dynamically increase tile count (reduce tile size) and recurse (not yet fully implemented, currently relies on user increasing Safety Factor).

## 4. UI/UX Refinements
*   **Safety Factor Dropdown**: Replaced obscure config files with a direct UI dropdown for memory tuning.
*   **Real-time Feedback**: "Auto-tiling Active" status now updates instantly when ROI or Model changes.
*   **Simplified Controls**: Removed deprecated widgets ("Pixel Budget") to reduce cognitive load.

---

## 4. Next Phase Preview (Phase 3)
**Focus:** Processing Mode & Advanced Visualization Refactoring

The next development cycle will focus on architectural flexibility and data interpretation tools:

### 4.1 Processing Mode Refactoring
*   **Objective**: Clearly distinguish and implement **Incremental** vs **Accumulative** displacement modes.
*   **Plan**:
    *   Refactor `DICProcessor` to handle stateful tracking (Incremental) vs reference-based tracking (Accumulative).
    *   Ensure the UI clearly reflects which mode is active and how it affects the result (e.g., drift correction in Incremental mode).

### 4.2 Visualization Configuration
*   **Objective**: Allow flexible configuration of Reference and Deformed images.
*   **Plan**:
    *   Implement a "Reference Frame" selector (currently defaults to Frame 0).
    *   Allow users to visualize "Current vs Reference" or "Current vs Previous" (Lagrangian vs Eulerian perspectives).
    *   Refactor the data pipeline to support these dynamic reference pairs without re-running the heavy optical flow inference (where possible).

---

## 5. File Map (Phase 2 Changes)
*   `main_GUI.py`: VRAM cleanup, ROI-aware logic.
*   `raft_dic_gui/model.py`: Metadata & Safety calculations.
*   `raft_dic_gui/processing.py`: Tiling engine & OOM retry.
*   `raft_dic_gui/views/control_panel.py`: Safety UI.
*   `raft_dic_gui/views/preview_panel.py`: Visualization layout.
