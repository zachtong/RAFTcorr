"""
RAFT-DIC GUI Application
------------------------
This module implements a graphical user interface for the RAFT-DIC displacement field calculator.
It provides tools for:
- Image loading and ROI selection
- Displacement field calculation using RAFT
- Result visualization and analysis
- Interactive parameter adjustment

Author: Zixiang Tong @ UT-Austin, Lehu Bu @ UT-Austin
Date: 2025-12-01
Version: 1.0

Dependencies:
- tkinter
- OpenCV
- PIL
- NumPy
- Matplotlib
- SciPy
- RAFT-DIC core modules

Usage:
    See README.md for details.
"""
import os
import sys
import threading
import tkinter as tk
import tkinter.ttk as tk_ttk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

import io
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import torch
import raft_dic_gui.model as mdl
import math
import gc

# ---------------------------------------------------------------------------
# CustomTkinter integration shims (visual-only refactor, logic unchanged)
# ---------------------------------------------------------------------------
import customtkinter as ctk
import raft_dic_gui.processing as proc
from raft_dic_gui.ui_components import (
    CTkCollapsibleFrame,
    _TTKAdapter,
    Tooltip
)
from raft_dic_gui.config import DICConfig
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.views.control_panel import ControlPanel
from raft_dic_gui.views.preview_panel import PreviewPanel

# Override future uses of CollapsibleFrame and ttk with CTk variants
CollapsibleFrame = CTkCollapsibleFrame
ttk = _TTKAdapter()

class RAFTDICGUI:
    def _enable_confirm(self):
        try:
            self.confirm_roi_btn.configure(state="normal")
            self.confirm_roi_btn.grid()
        except Exception:
            pass

    def _ui_call(self, func, *args, wait: bool = False, **kwargs):
        """Execute a callable on the Tk main thread."""
        if threading.current_thread() is threading.main_thread():
            return func(*args, **kwargs)

        result = {}
        event = threading.Event()

        def _wrapper():
            try:
                print(f"[DEBUG] Executing _ui_call wrapper for {func.__name__}", flush=True) 
                result['value'] = func(*args, **kwargs)
            except Exception as e:
                print(f"[ERROR] Exception in _ui_call wrapper: {e}", flush=True)
            finally:
                event.set()

        print(f"[DEBUG] Scheduling _ui_call for {func.__name__}", flush=True)
        self.root.after(0, _wrapper)
        if wait:
            event.wait()
            return result.get('value')

    def _set_running_state(self, running: bool):
        """Toggle global UI state while processing runs in background."""
        def _toggle():
            try:
                self.root.configure(cursor="watch" if running else "")
            except Exception:
                pass

            state = 'disabled' if running else 'normal'
            print(f"[DEBUG] Setting running state to {running} (state={state})", flush=True)
            
            # Recursively disable/enable all buttons/entries
            def _set_state_recursive(widget):
                try:
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton, ctk.CTkButton, ctk.CTkEntry)):
                        # Skip run/stop buttons here, handle them explicitly below
                        if widget is not self.control_panel.run_button and widget is not self.control_panel.stop_button:
                            widget.configure(state=state)
                except Exception:
                    pass
                
                for child in widget.winfo_children():
                    _set_state_recursive(child)

            _set_state_recursive(self.root)
            
            # Explicitly handle Run/Stop buttons
            try:
                self.control_panel.run_button.configure(state='disabled' if running else 'normal')
                self.control_panel.stop_button.configure(state='normal' if running else 'disabled')
            except Exception as e:
                print(f"[ERROR] Failed to set Run/Stop button state: {e}", flush=True)

        self._ui_call(_toggle, wait=True)

    def _update_progress(self, percent: float = None, current: int = None, total: int = None):
        """Update progress bar and text safely from any thread."""
        def _apply():
            if percent is not None:
                try:
                    self.control_panel.progress.configure(value=percent)
                except Exception as e:
                    print(f"[ERROR] Failed to update progress bar: {e}", flush=True)
            if current is not None and total is not None:
                try:
                    self.control_panel.progress_text.configure(text=f"{current}/{total}")
                except Exception as e:
                    print(f"[ERROR] Failed to update progress text: {e}", flush=True)

        self._ui_call(_apply)

    def _update_results_ui(self, total_frames: int):
        """Refresh playback controls after new results arrive."""
        def _apply():
            try:
                self.frame_slider.configure(to=total_frames)
                self.frame_slider.set(1 if total_frames > 0 else 0)
            except Exception:
                pass
            try:
                self.total_frames_label.configure(text=f"/{total_frames}")
            except Exception:
                pass
            try:
                self.frame_entry.delete(0, tk.END)
                self.frame_entry.insert(0, "1" if total_frames > 0 else "0")
            except Exception:
                pass
            try:
                self.update_displacement_preview()
            except Exception:
                pass

        self._ui_call(_apply)

    def __init__(self, root):
        self.root = root
        self.root.title("RAFTcorr: 2D displacement tracking")
        
        # Initialize configuration and controller
        self.config = DICConfig()
        self.processor = DICProcessor()
        
        # Configure grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main paned window
        self.main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        self.main_paned.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create Control Panel (Left)
        self.control_panel = ControlPanel(self.main_paned, config=self.config)
        self.main_paned.add(self.control_panel, minsize=550)
        
        # Create Preview Panel (Right)
        self.preview_panel = PreviewPanel(self.main_paned, config=self.config, root=self.root)
        self.main_paned.add(self.preview_panel, minsize=600)
        
        # Wire up callbacks
        self._setup_callbacks()
        # Initialize state
        self.displacement_results = []
        self.current_image = None
        self.image_files = []
        self.roi_mask = None
        self.roi_rect = None
        
        # Apply theme
        self.apply_theme()

    def _setup_callbacks(self):
        # Control Panel callbacks
        self.control_panel.callbacks['run'] = self.run
        self.control_panel.callbacks['stop'] = self.request_stop
        self.control_panel.callbacks['browse_input'] = self.browse_input
        self.control_panel.callbacks['browse_output'] = self.browse_output
        self.control_panel.callbacks['update_preview'] = self.preview_panel.update_displacement_preview
        self.control_panel.callbacks['on_param_change'] = self.preview_panel.update_displacement_preview
        self.control_panel.callbacks['on_preview_scale_change'] = lambda v: self.preview_panel.update_displacement_preview()
        self.control_panel.callbacks['update_roi_label'] = self.preview_panel.update_roi_label
        self.control_panel.callbacks['set_fixed_colorbar'] = self.set_fixed_colorbar_from_frame
        self.control_panel.callbacks['set_fixed_colorbar'] = self.set_fixed_colorbar_from_frame
        self.control_panel.callbacks['on_model_selected'] = self._update_roi_status_text
        self.control_panel.callbacks['on_safety_factor_change'] = lambda: (self._update_roi_status_text(), self.preview_panel.draw_roi())
        self.control_panel.callbacks['on_overlap_change'] = self.preview_panel.draw_roi
        self.control_panel.callbacks['on_show_tiles_change'] = self.preview_panel.draw_roi
        
        # Preview Panel callbacks
        self.preview_panel.control_panel = self.control_panel
        self.preview_panel.callbacks['on_roi_confirmed'] = self._on_roi_confirmed
        self.preview_panel.callbacks['on_tab_changed'] = self._on_preview_tab_changed
        
        # Post-Processing callbacks
        self.control_panel.post_processing_panel.callbacks['calculate_strain'] = self.calculate_strain
        self.control_panel.post_processing_panel.callbacks['update_post_preview'] = self.update_post_preview

    def calculate_strain(self):
        """Handle strain calculation request."""
        print("[DEBUG] Calculate Strain requested.")
        
        if not self.displacement_results:
            messagebox.showwarning("Warning", "No displacement results available. Please run analysis first.")
            return

        # Get parameters
        pp = self.control_panel.post_processing_panel
        method = pp.strain_method.get()
        
        # VSG Parameters
        try:
            vsg_size = int(pp.vsg_size.get())
            if vsg_size < 9 or vsg_size > 101 or vsg_size % 2 == 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Parameter", "VSG Size must be an odd integer between 9 and 101.")
            return
            
        try:
            step = int(pp.strain_step.get())
            if step < 1:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Invalid Parameter", "Step must be an integer >= 1.")
            return
            
        poly_order = int(pp.poly_order.get())
        weighting = pp.weighting.get()
            
        print(f"[DEBUG] Strain Params: Method={method}, Size={vsg_size}, Order={poly_order}, Weight={weighting}, Step={step}")
        
        # Check selected components
        comps = {k: v.get() for k, v in pp.strain_components.items()}
        if not any(comps.values()):
            messagebox.showwarning("Warning", "Please select at least one strain component to calculate.")
            return

        # Run calculation (Batch)
        self._set_running_state(True)
        
        # Use existing progress bar
        progress = self.control_panel.progress
        progress_text = self.control_panel.progress_text
        total = len(self.displacement_results)
        
        def _run():
            try:
                count = 0
                self.strain_results = [] # Clear previous results
                
                # Update available components in UI
                active_comps = [k for k, v in comps.items() if v]
                self._ui_call(pp.vis_comp_combo.configure, values=active_comps)
                if active_comps:
                    self._ui_call(pp.vis_component.set, active_comps[0])
                
                for i, item in enumerate(self.displacement_results):
                    # Update progress
                    perc = (i / total) * 100
                    self._ui_call(progress.configure, value=perc)
                    self._ui_call(progress_text.configure, text=f"Calculating Strain: {i+1}/{total}")
                    
                    # Load displacement
                    if isinstance(item, str):
                        disp = np.load(item)
                    else:
                        disp = item
                        
                    # Calculate
                    strain = proc.calculate_strain_field(disp, method=method, 
                                                       vsg_size=vsg_size, 
                                                       poly_order=poly_order, 
                                                       weighting=weighting,
                                                       step=step)
                    self.strain_results.append(strain)
                    count += 1
                
                self._ui_call(progress.configure, value=100)
                self._ui_call(progress_text.configure, text="Strain Calculation Complete")
                print(f"[DEBUG] Calculated strain for {count} frames.")
                
                self._ui_call(messagebox.showinfo, "Success", f"Strain calculation complete for {count} frames.")
                
                # Trigger visualization update
                self._ui_call(self.update_post_preview)
                
            except Exception as e:
                print(f"[ERROR] Strain calculation failed: {e}")
                self._ui_call(messagebox.showerror, "Error", f"Strain calculation failed: {e}")
            finally:
                self._set_running_state(False)
                
        import threading
        threading.Thread(target=_run, daemon=True).start()

    def update_post_preview(self, *args):
        """Update the post-processing visualization (Strain/Probes)."""
        print("[DEBUG] update_post_preview called")
        if not hasattr(self, 'strain_results') or not self.strain_results:
            print("[DEBUG] No strain results found.")
            return
            
        try:
            # Get current frame index from Post-Processing slider
            try:
                if hasattr(self.preview_panel, 'post_frame_slider'):
                    val = self.preview_panel.post_frame_slider.get()
                    current_frame = int(float(val)) - 1
                else:
                    current_frame = 0
            except Exception:
                current_frame = 0
            
            # Ensure valid frame index
            if current_frame < 0:
                current_frame = 0
            
            print(f"[DEBUG] Current strain frame: {current_frame}")
            
            if current_frame < 0 or current_frame >= len(self.strain_results):
                print(f"[DEBUG] Frame index out of range (0-{len(self.strain_results)-1})")
                return
                
            strain_data = self.strain_results[current_frame]
            if strain_data is None:
                print("[DEBUG] Strain data is None for this frame")
                return
                
            # Get display settings
            pp = self.control_panel.post_processing_panel
            comp_name = pp.vis_component.get()
            cmap = pp.vis_colormap.get()
            
            try:
                alpha = float(pp.vis_alpha.get())
                alpha = max(0.0, min(1.0, alpha))
            except ValueError:
                alpha = 0.7
            
            if comp_name not in strain_data:
                return
                
            data_map = strain_data[comp_name]
            
            # Color Range Logic
            vmin, vmax = None, None
            is_fixed = pp.post_fixed_range.get()
            
            if is_fixed:
                # Try to get from entries
                try:
                    vmin = float(pp.post_vmin.get())
                    vmax = float(pp.post_vmax.get())
                    # Update stored range
                    pp.component_ranges[comp_name] = (vmin, vmax)
                except ValueError:
                    # If invalid, fallback to auto or stored
                    if comp_name in pp.component_ranges:
                        vmin, vmax = pp.component_ranges[comp_name]
                        pp.post_vmin.set(f"{vmin:.4f}")
                        pp.post_vmax.set(f"{vmax:.4f}")
            else:
                # Auto range
                vmin = np.nanmin(data_map)
                vmax = np.nanmax(data_map)
                # Update UI entries
                pp.post_vmin.set(f"{vmin:.4f}")
                pp.post_vmax.set(f"{vmax:.4f}")
                # Clear stored range for this component if switching to auto? 
                # Or keep it? Let's keep it but update it to current auto for convenience if they switch back?
                # No, if they switch to fixed, they probably want the last fixed value or the current auto value.
                # Let's just update the entries.
            
            # Load background image (Reference Image)
            bg_img = None
            if self.image_files:
                # Cache reference image if not already cached
                if not hasattr(self, '_cached_ref_img') or self._cached_ref_img is None:
                    try:
                        input_dir = self.control_panel.input_path.get()
                        img_path = os.path.join(input_dir, self.image_files[0])
                        self._cached_ref_img = proc.load_and_convert_image(img_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to load reference image: {e}")
                        self._cached_ref_img = None
                bg_img = self._cached_ref_img
            
            # Update PreviewPanel
            self.preview_panel.plot_post_data(data_map, comp_name, cmap, alpha, 
                                            background_image=bg_img, 
                                            roi_rect=self.roi_rect,
                                            vmin=vmin, vmax=vmax)
            
            # Update Post-Processing Playback Controls
            # Use a flag or check value to prevent recursion loop
            total_frames = len(self.strain_results)
            if hasattr(self.preview_panel, 'post_frame_slider'):
                self.preview_panel.post_frame_slider.configure(to=total_frames)
                # Only set if different to avoid recursion loop
                current_val = float(self.preview_panel.post_frame_slider.get())
                if int(current_val) != (current_frame + 1):
                    self.preview_panel.post_frame_slider.set(current_frame + 1)
                
            if hasattr(self.preview_panel, 'post_frame_entry'):
                self.preview_panel.post_frame_entry.delete(0, tk.END)
                self.preview_panel.post_frame_entry.insert(0, str(current_frame + 1))
                
            if hasattr(self.preview_panel, 'post_total_frames_label'):
                self.preview_panel.post_total_frames_label.configure(text=f"/{total_frames}")
                
            if hasattr(self.preview_panel, 'post_current_image_name'):
                if current_frame < len(self.image_files):
                    self.preview_panel.post_current_image_name.configure(text=self.image_files[current_frame])
            
        except Exception as e:
            print(f"[ERROR] Update post preview failed: {e}")

    def _on_preview_tab_changed(self, index):
        """Synchronize Control Panel tabs with Preview Panel tabs."""
        try:
            # Map Preview Panel tabs to Control Panel tabs
            # 0 (ROI) -> 0 (DIC Params)
            # 1 (Displacement) -> 0 (DIC Params)
            # 2 (Post-Processing) -> 1 (Post-Processing)
            
            target_index = 0
            if index == 2:
                target_index = 1
                
            self.control_panel.select_tab(target_index)
        except Exception as e:
            print(f"Error syncing tabs: {e}")

    def _on_roi_confirmed(self, mask, rect):
        self.roi_mask = mask
        self.roi_rect = rect
        self._update_roi_status_text()

    def _update_roi_status_text(self):
        """Update the status text in PreviewPanel with VRAM and Tiling info."""
        # Force cleanup to get accurate free memory reading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        try:
            # 1. Get GPU Info
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                free_gb = free_mem / (1024**3)
                total_gb = total_mem / (1024**3)
                gpu_str = f"GPU: {gpu_name} ({total_gb:.1f} GB)\nAvailable: {free_gb:.1f} GB"
            else:
                gpu_str = "GPU: Not available (CPU mode)"
                free_mem = 0

            # 2. Get Model Info
            model_label = self.control_panel.selected_model.get()
            model_entry = self.control_panel.model_lookup.get(model_label)
            if model_entry:
                try:
                    metadata = mdl.describe_checkpoint(model_entry.path)
                    model_type = "RAFT-Fine (Full Res)" if metadata.full_resolution else "RAFT-Large (Standard)"
                    
                    # 3. Calculate Safe Tile
                    # Get safety factor from UI
                    try:
                        sf = float(self.control_panel.safety_factor.get())
                    except Exception:
                        sf = 0.55
                        
                    safe_pmax = mdl.estimate_safe_pmax(metadata, safety_factor=sf)
                    
                    # CRITICAL FIX: Update BOTH config and UI entry so it persists!
                    self.config.p_max_pixels = safe_pmax
                    self.control_panel.p_max_pixels.set(f"{safe_pmax}")
                    
                    safe_tile_dim = int(math.sqrt(safe_pmax))
                    tile_str = f"Max Safe Tile: {safe_tile_dim} x {safe_tile_dim} px"
                except Exception:
                    model_type = "Unknown Model"
                    tile_str = "Max Safe Tile: N/A"
                    safe_pmax = 500*500
                    self.config.p_max_pixels = safe_pmax
                    self.control_panel.p_max_pixels.set(f"{safe_pmax}")
            else:
                model_type = "None"
                tile_str = "Max Safe Tile: N/A"
                safe_pmax = 500*500
                self.config.p_max_pixels = safe_pmax
                self.control_panel.p_max_pixels.set(f"{safe_pmax}")

            # 4. Get Image Info & Strategy
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                
                # Use ROI size if available, otherwise full image
                calc_w, calc_h = w, h
                img_str = f"Input: {w} x {h}" # Default
                
                if self.roi_rect:
                    x0, y0, x1, y1 = self.roi_rect
                    rw, rh = x1 - x0, y1 - y0
                    if rw > 0 and rh > 0:
                        calc_w, calc_h = rw, rh
                        img_str = f"ROI: {rw} x {rh}"

                if (calc_w * calc_h) > safe_pmax:
                    # Tiling needed
                    # Estimate tiles with overlap
                    try:
                        overlap = int(self.control_panel.tile_overlap.get())
                    except Exception:
                        overlap = 32
                        
                    stride = max(1, safe_tile_dim - overlap)
                    nx = math.ceil((calc_w - overlap) / stride)
                    ny = math.ceil((calc_h - overlap) / stride)
                    # Ensure at least 1 tile
                    nx = max(1, nx)
                    ny = max(1, ny)
                    
                    strategy_str = f"-> Auto-tiling Active ({nx}x{ny} tiles)"
                else:
                    strategy_str = "-> Direct Processing (Optimal)"
            else:
                img_str = "Input: None"
                strategy_str = ""

            # Combine
            full_text = f"{gpu_str}\nModel: {model_type}\n{tile_str}\n{img_str} {strategy_str}"
            
            self.preview_panel.update_info_text(full_text)
            
        except Exception as e:
            print(f"Error updating status text: {e}")
        
    def apply_theme(self):
        """Apply a modern theme to ttk widgets"""
        style = tk_ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = "#f0f0f0"
        fg_color = "#333333"
        accent_color = "#007acc"
        
        base_font = ('Segoe UI', 15)
        
        style.configure(".", background=bg_color, foreground=fg_color, font=base_font)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color, font=base_font)
        style.configure("TButton", padding=6, relief="flat", background="#e1e1e1", font=base_font)
        style.configure("TEntry", font=base_font)
        style.configure("TCombobox", font=base_font)
        style.configure("TCheckbutton", font=base_font, background=bg_color)
        style.configure("TRadiobutton", font=base_font, background=bg_color)
        
        style.map("TButton", background=[('active', '#d1d1d1')])
        
        # Accent button style
        style.configure("Accent.TButton", background=accent_color, foreground="white")
        style.map("Accent.TButton", background=[('active', '#005999')])

    def browse_input(self, directory=None):
        """Handle input directory selection."""
        if directory is None:
            directory = filedialog.askdirectory()
        
        if directory:
            self.control_panel.input_path.set(directory)
            try:
                self.update_image_info(directory)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load images: {e}")

    def browse_output(self, directory=None):
        """Handle output directory selection."""
        if directory is None:
            directory = filedialog.askdirectory()
        
        if directory:
            self.control_panel.output_path.set(directory)

    def update_image_info(self, directory):
        """Load first image and update preview."""
        # Get image file list
        self.image_files = sorted([f for f in os.listdir(directory) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        
        if not self.image_files:
            raise ValueError("No valid image files found in the directory")
        
        # Read first image
        img_path = os.path.join(directory, self.image_files[0])
        img = proc.load_and_convert_image(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Store current image
        self.current_image = img
        
        # Reset state for new dataset
        self.roi_mask = None
        self.roi_rect = None
        self.displacement_results = []
        self.preview_panel.reset_state()

        # Update PreviewPanel
        self.preview_panel.set_image(img, img_path)
        
        # Update crop size in ControlPanel
        h, w = img.shape[:2]
        self.control_panel.crop_size_w.set(str(w))
        self.control_panel.crop_size_h.set(str(h))
        
        # Update Status Text
        self._update_roi_status_text()

    def validate_inputs(self):
        """Validate run parameters."""
        if not self.control_panel.input_path.get():
            messagebox.showwarning("Input Missing", "Please select an input directory.")
            return False
        if not self.roi_mask is not None:
            messagebox.showwarning("ROI Missing", "Please select and confirm an ROI.")
            return False
        return True

    def run(self):
        """Start the DIC processing pipeline."""
        # Update config from UI
        self.control_panel.update_config(self.config)
        
        # Validate config
        valid, msg = self.config.validate()
        if not valid:
            messagebox.showerror("Configuration Error", msg)
            return

        if not self.validate_inputs():
            return

        # Reset stop flag
        self._stop_requested = False
        
        # Update processor callbacks
        self.processor.update_progress = self._update_progress
        self.processor.check_stop = lambda: self._stop_requested

        # Start processing in a separate thread
        self._set_running_state(True)
        threading.Thread(target=self._run_process, daemon=True).start()

    def request_stop(self):
        """Signal the processor to stop."""
        self._stop_requested = True
        self.control_panel.stop_button.configure(state='disabled')

    def _run_process(self):
        """Background processing task."""
        try:
            # Run processing
            results = self.processor.run(
                self.config,
                self.roi_mask,
                self.roi_rect
            )
            
            # Store results
            self.displacement_results = results
            
            # Update UI with results
            self._ui_call(self._on_processing_complete, len(results))
            
        except Exception as e:
            self._ui_call(messagebox.showerror, "Processing Error", str(e))
        finally:
            self._set_running_state(False)

    def _on_processing_complete(self, total_frames):
        """Handle successful completion of processing."""
        messagebox.showinfo("Done", "Processing complete!")
        
        # Pass results to preview panel
        self.preview_panel.set_results(self.displacement_results, self.image_files)
        self.preview_panel.show_results_tab()
        
        # Enable Post-Processing tab
        try:
            self.preview_panel.notebook.tab(2, state="normal")
        except Exception:
            pass
        
        self._update_results_ui(total_frames)
        # Auto-set ranges if enabled or first run
        self.auto_set_colorbar_ranges()

    def auto_set_colorbar_ranges(self):
        """Compute colorbar ranges from the first displacement result."""
        try:
            if not self.displacement_results:
                return
            
            # Use the first result (or middle one for better representative range?)
            # Using first for now as per legacy behavior
            item = self.displacement_results[0]
            disp = np.load(item) if isinstance(item, str) else item
            
            u = disp[:, :, 0]
            v = disp[:, :, 1]
            
            self.control_panel.colorbar_u_min.set(f"{np.nanmin(u):.3f}")
            self.control_panel.colorbar_u_max.set(f"{np.nanmax(u):.3f}")
            self.control_panel.colorbar_v_min.set(f"{np.nanmin(v):.3f}")
            self.control_panel.colorbar_v_max.set(f"{np.nanmax(v):.3f}")
            
            self.control_panel.use_fixed_colorbar.set(True)
            self.preview_panel.update_displacement_preview()
            
        except Exception as e:
            print(f"Failed to auto-set ranges: {e}")

    def set_fixed_colorbar_from_frame(self):
        """Set fixed ranges from a specific frame."""
        try:
            if not self.displacement_results:
                return
                
            try:
                idx = int(self.control_panel.fixed_range_frame.get()) - 1
            except ValueError:
                idx = 0
                
            idx = max(0, min(idx, len(self.displacement_results) - 1))
            
            item = self.displacement_results[idx]
            disp = np.load(item) if isinstance(item, str) else item
            
            u = disp[:, :, 0]
            v = disp[:, :, 1]
            
            self.control_panel.colorbar_u_min.set(f"{np.nanmin(u):.3f}")
            self.control_panel.colorbar_u_max.set(f"{np.nanmax(u):.3f}")
            self.control_panel.colorbar_v_min.set(f"{np.nanmin(v):.3f}")
            self.control_panel.colorbar_v_max.set(f"{np.nanmax(v):.3f}")
            
            self.preview_panel.update_displacement_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set range: {e}")

    def reset_visualization_settings(self):
        """Reset colormap and ranges to defaults and refresh preview."""
        self.control_panel.colormap.set('viridis')
        self.control_panel.use_fixed_colorbar.set(False)
        self.control_panel.colorbar_u_min.set("-1")
        self.control_panel.colorbar_u_max.set("1")
        self.control_panel.colorbar_v_min.set("-1")
        self.control_panel.colorbar_v_max.set("1")
        self.preview_panel.update_displacement_preview()

    def update_crop_size(self):
        """Update crop size entries with image dimensions"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.control_panel.crop_size_w.set(str(w))
            self.control_panel.crop_size_h.set(str(h))


def main():
    """Main entry point for the RAFT-DIC GUI application."""
    # Load optional external app config
    def _load_app_config():
        cfg_paths = [
            os.path.join('assets', 'app_config.json'),
            os.path.join('assets', 'config', 'app_config.json')
        ]
        for p in cfg_paths:
            try:
                if os.path.isfile(p):
                    import json
                    with open(p, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception:
                pass
        return {}

    root = ctk.CTk()
    # Apply configurable branding and theme without hardcoding
    try:
        cfg = _load_app_config()
        title = cfg.get('app_title') or 'RAFT-DIC GUI'
        root.title(title)
        # Appearance and color theme
        if 'appearance_mode' in cfg:
            try: ctk.set_appearance_mode(cfg['appearance_mode'])
            except Exception: pass
        if 'color_theme' in cfg:
            try:
                ct = cfg['color_theme']
                # If a JSON path is provided, treat it as an override on top of a base theme
                if isinstance(ct, str) and ct.lower().endswith('.json') and os.path.isfile(ct):
                    import json
                    # Start from a known-good base theme
                    ctk.set_default_color_theme('blue')
                    try:
                        with open(ct, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        overrides = data.get('overrides', data)
                        def _deep_update(dst, src):
                            for k, v in src.items():
                                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                                    _deep_update(dst[k], v)
                                else:
                                    dst[k] = v
                        # Apply overrides into the live theme dict before widgets are created
                        _deep_update(ctk.ThemeManager.theme, overrides)
                    except Exception:
                        # Fall back to base theme if override fails
                        pass
                else:
                    # Use built-in or file-based theme directly
                    ctk.set_default_color_theme(ct)
                # Validate essential keys exist in theme; if not, fallback to 'blue'
                try:
                    th = ctk.ThemeManager.theme
                    required = [
                        ('CTkFrame', None),
                        ('CTkScrollbar', 'fg_color'),
                        ('CTkScrollbar', 'border_spacing'),
                        ('CTkButton', 'text_color_disabled')
                    ]
                    for key, sub in required:
                        if key not in th:
                            raise KeyError(key)
                        if sub and sub not in th[key]:
                            raise KeyError(f"{key}.{sub}")
                except Exception:
                    ctk.set_default_color_theme('blue')
            except Exception:
                # If loading fails entirely, keep default
                pass
        # Window icon (png/ico)
        try:
            icon_png = cfg.get('icon_png')
            icon_ico = cfg.get('icon_ico')
            # Auto-generate simple default icons if missing
            def _ensure_icon(path_png, path_ico):
                try:
                    os.makedirs(os.path.dirname(path_png), exist_ok=True)
                except Exception:
                    pass
                if not os.path.isfile(path_png):
                    try:
                        img = Image.new('RGBA', (256, 256), (17, 94, 164, 255))
                        drw = ImageDraw.Draw(img)
                        drw.ellipse((40, 40, 216, 216), fill=(255, 180, 0, 255))
                        img.save(path_png, format='PNG')
                    except Exception:
                        pass
                if path_ico and not os.path.isfile(path_ico):
                    try:
                        img = Image.open(path_png).convert('RGBA')
                        img.save(path_ico, sizes=[(256,256)])
                    except Exception:
                        pass
            if icon_png:
                _ensure_icon(icon_png, icon_ico)
            if icon_png and os.path.isfile(icon_png):
                root.iconphoto(True, tk.PhotoImage(file=icon_png))
            if icon_ico and os.path.isfile(icon_ico):
                try:
                    root.iconbitmap(icon_ico)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    app = RAFTDICGUI(root)
    try:
        # Adaptive window size and centering
        root.update_idletasks()
        w, h = 1680, 990
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        root.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass
    root.mainloop()

if __name__ == '__main__':
    main() 







