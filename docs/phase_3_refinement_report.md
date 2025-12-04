# RAFTcorr Development Report - Phase 3
**Date:** 2025-12-01
**Phase:** 3 - Production Readiness & Robustness
**Previous Phase:** Phase 2 - Core Algorithm Optimization

## 1. Executive Summary
Phase 3 focused on transforming the optimized core (from Phase 2) into a **production-ready application**. The primary goals were reliability, data integrity, and user experience. We addressed critical issues regarding execution control (Stop/Resume), model consistency, and UI responsiveness. The application is now rebranded as **RAFTcorr** and features a robust "Smart Resume" system that guarantees data validity across sessions.

## 2. Key Technical Achievements

### 2.1 Robust Execution Pipeline ("Smart Resume")
*   **Problem**: Users needed to stop long-running jobs and resume them later. Simply checking for existing files was dangerous because parameters (or models) might have changed between runs, leading to mixed/invalid results.
*   **Solution**: Implemented a strict **Configuration Consistency Check**.
    *   **Run Config**: On every run, the system saves a `run_config.json` containing critical parameters (`model_path`, `tile_overlap`, `sigma`, etc.).
    *   **Validation**: When resuming, the current UI state is compared against the saved `run_config.json`.
    *   **Auto-Correction**:
        *   **Match**: The system skips existing frames (Resume).
        *   **Mismatch**: The system triggers a **Force Re-run**, automatically deleting *all* stale `.npy` files to prevent data contamination.

### 2.2 Model Selection Integrity
*   **Problem**: A subtle bug caused the application to sometimes use a "default" model instead of the one selected in the UI, or to persist a model selection across restarts incorrectly.
*   **Solution**:
    *   **UI-Logic Sync**: Enforced explicit synchronization between the UI dropdown (`Combobox`) and the internal configuration object (`DICConfig`) at startup and before every run.
    *   **Traceability**: Added granular debug logging (temporarily) to verify the flow of model selection from UI -> Config -> Processor.

### 2.3 UI Responsiveness & UX
*   **Event Loop Protection**: Implemented **Rate Limiting** (10Hz) for progress bar updates. Previously, rapid updates on small images would flood the main thread, causing the UI to freeze ("Not Responding").
*   **Visual Feedback**:
    *   **ROI Status**: Updated to show "ROI: W x H" (e.g., "ROI: 450 x 800") instead of just input image size, aiding in tile estimation.
    *   **Tile Overlay**: Added a "Show Tiles" checkbox to visualize the exact grid layout on the ROI, helping users understand the `p_max_pixels` impact.
*   **Control Logic**: Fixed the "Stop" button state management to ensure it is always clickable during execution.

## 3. Technical Deep Dive

### 3.1 Smart Resume Logic Flow
The `DICProcessor.run` method now follows this strict logic:

1.  **Load Config**: Read `run_config.json` from output directory.
2.  **Compare**: Check `current_config` vs `saved_config`.
    *   *Critical Keys*: `model_path`, `tile_overlap`, `context_padding`, `safety_factor`, `sigma`, `mode`.
3.  **Decision**:
    *   If `Changed`: Set `force_rerun = True`.
    *   If `Same`: Set `force_rerun = False`.
4.  **Cleanup (Crucial Step)**:
    *   If `force_rerun` is True: **Delete** `displacement_results_npy/*.npy`. This ensures that we never mix Frame 1-10 (Model A) with Frame 11-20 (Model B).
5.  **Execution Loop**:
    *   Iterate frames.
    *   If `not force_rerun` AND file exists: **Skip** (Load from disk).
    *   Else: **Compute** & **Save Immediately**.

### 3.2 Rebranding
The project has been officially renamed to **RAFTcorr** (RAFT-based Digital Image Correlation).
*   Updated `README.md`, `USER_MANUAL.md`, and `setup.py`.
*   Reflects the tool's specialized nature for DIC rather than just generic optical flow.

## 4. File Map (Phase 3 Changes)
*   `raft_dic_gui/controller.py`: Smart Resume logic, Stale file cleanup, Loop rate limiting.
*   `raft_dic_gui/views/control_panel.py`: Model selection sync, Tile Overlay toggle.
*   `main_GUI.py`: Configuration validation, Thread management.
*   `task.md`: Comprehensive tracking of the stabilization process.

## 5. Future Outlook
With the core engine optimized (Phase 2) and the execution pipeline stabilized (Phase 3), the application is ready for broader testing. Future work may focus on:
*   **Batch Processing**: Queueing multiple folders.
*   **Advanced Post-Processing**: Strain calculation (Green-Lagrange), Virtual Extensometers.
*   **Export Formats**: VTK/Paraview support.
