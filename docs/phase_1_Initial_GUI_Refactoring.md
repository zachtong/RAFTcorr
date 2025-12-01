# RAFTcorr Development Status

**Last Updated:** 2025-11-30
**Current Version:** Refactored GUI with Tabbed Interface & Interactive Visualization

## 1. Project Overview
RAFTcorr is a GUI application for 2D Digital Image Correlation (DIC) using the RAFT optical flow model. It allows users to load image sequences, select ROIs, run displacement analysis, and visualize results.

## 2. Architecture (Refactored)
The project follows a modular MVC-like pattern:
- **Entry Point**: `main_GUI.py` - Orchestrates the application, manages global state (`current_image`, `results`), and assembles views.
- **Controller**: `raft_dic_gui/controller.py` (`DICProcessor`) - Handles the core RAFT processing logic in a background thread.
- **Model/Config**: `raft_dic_gui/config.py` (`DICConfig`) - Dataclass for all processing parameters.
- **Views**:
    - `raft_dic_gui/views/control_panel.py` (`ControlPanel`) - Left sidebar for parameter inputs. Uses `ttk` widgets.
    - `raft_dic_gui/views/preview_panel.py` (`PreviewPanel`) - Right area with Tabs:
        - **Tab 1: ROI Selection**: Canvas for drawing/editing ROI.
        - **Tab 2: Displacement Result**: Interactive Matplotlib canvas (`FigureCanvasTkAgg`) for visualizing U/V fields.
- **Processing Logic**: `raft_dic_gui/processing.py` - Contains visualization generation logic (returns Matplotlib Figures).

## 3. Key Features & Recent Changes
- **Tabbed Interface**: ROI selection and Results are now in separate tabs. Auto-switches to Results after processing.
- **Interactive Visualization**: Results use Matplotlib toolbar for Zoom/Pan.
- **Robustness**:
    - Fixed "boolean index error" on folder change (implemented `reset_state`).
    - Fixed "Fixed Range" default (now defaults to False/Adaptive).
    - Fixed flickering by reusing the Matplotlib Figure/Canvas instance.
- **UI Polish**:
    - Global font size increased to 15 (approx 1.5x).
    - Control Panel width increased to 550px.
    - Application renamed to "RAFTcorr".

## 4. User Preferences & Rules
- **Language**: Always respond in **Chinese**.
- **Code Comments**: English.
- **UI Style**: Native `ttk` with `clam` theme. Fonts explicitly configured for readability.
- **Visualization**:
    - "Fixed Range" unchecked = Adaptive (auto-updates UI entries).
    - "Fixed Range" checked = Uses UI entry values.

## 5. Next Steps / Pending Ideas
- [ ] Implement "Batch Processing" for multiple folders?
- [ ] Add "Export Results" to CSV/Excel?
- [ ] Add "Strain Calculation" (derived from displacement)?
- [ ] Further optimize RAFT model loading (cache model)?

## 6. How to Resume
In a new session, the AI should:
1.  Read this file (`docs/development_status.md`).
2.  Scan `main_GUI.py` and `raft_dic_gui/` to understand the structure.
3.  The AI will be immediately up to speed.

## 7. File Structure & Descriptions

### Root Directory
- **`main_GUI.py`**: The application entry point.
    - Initializes the main Tkinter window (`RAFTDICGUI` class).
    - Instantiates `DICConfig`, `DICProcessor`, `ControlPanel`, and `PreviewPanel`.
    - Manages global callbacks and high-level logic (e.g., `run`, `reset_state`, `update_image_info`).
    - Handles theme application (`apply_theme`).
- **`verify_installation.py`**: A standalone script to check if all required dependencies (PyTorch, RAFT, Tkinter, etc.) are correctly installed and functional.

### `raft_dic_gui/` (Package)
- **`config.py`**: Defines the `DICConfig` dataclass.
    - Centralizes all application parameters (paths, model settings, crop settings, visualization options).
    - Includes validation logic (`validate()` method) and serialization helpers (`to_dict`, `from_dict`).
- **`controller.py`**: Defines the `DICProcessor` class.
    - Runs the heavy RAFT processing in a background thread to keep the UI responsive.
    - Handles model loading (`load_model`) and image processing loop (`run`).
    - Communicates progress back to the UI via callbacks.
- **`model.py`**: Contains the RAFT model definition and loading logic.
    - Wraps the underlying PyTorch RAFT implementation.
    - Provides `load_model` helper to safely load checkpoints.
- **`processing.py`**: Handles data processing and visualization generation.
    - `prepare_visualization_data`: Extracts U/V fields, masks, and boundary points from raw results.
    - `update_displacement_plot`: Updates the Matplotlib Axes with new data (efficiently, without recreating figures).
    - Contains helper functions for image conversion and resizing.
- **`ui_components.py`**: Reusable custom UI widgets.
    - `CTkCollapsibleFrame`: A collapsible frame widget (though currently `main_GUI` might use `ttk` adapter).
    - `Tooltip`: A simple tooltip implementation for UI elements.

### `raft_dic_gui/views/` (UI Components)
- **`control_panel.py`**: The left-hand sidebar (`ControlPanel` class).
    - Contains all input fields (Entry, Checkbox, Radiobutton) for configuring `DICConfig`.
    - Manages user interaction for parameters.
    - Triggers callbacks (e.g., `update_preview`) when parameters change.
- **`preview_panel.py`**: The right-hand display area (`PreviewPanel` class).
    - Implements a Tabbed interface (`ttk.Notebook`):
        - **ROI Selection Tab**: Uses a `tk.Canvas` for drawing/editing ROI polygons.
        - **Displacement Result Tab**: Embeds a Matplotlib figure (`FigureCanvasTkAgg`) for interactive results.
    - Manages the visualization state (Figure, Axes, Colorbars).
    - Handles playback of result sequences.

### Other Directories
- **`core/`**: The core RAFT model implementation (PyTorch).
    - Contains the neural network architecture (`raft.py`, `extractor.py`, `update.py`) and correlation layer logic (`corr.py`).
    - This is the "engine" that `raft_dic_gui.model` wraps.
- **`assets/`**: Application resources.
    - `app_config.json`: Default application configuration.
    - `icons/`: UI icons.
    - `themes/`: UI themes.
- **`models/`**: Pre-trained RAFT model checkpoints (`.pth` files).
    - Users select these models from the Control Panel.
- **`000_strain_display/`**: MATLAB scripts for strain calculation and visualization.
    - Likely serves as a reference implementation or a separate tool for post-processing strain analysis.
- **`alt_cuda_corr/`**: C++ and CUDA source code for an alternative correlation kernel implementation.
- **`legacy_archive/`**: Deprecated or old code files.
- **`docs/`**: Project documentation, including this file (`development_status.md`).
