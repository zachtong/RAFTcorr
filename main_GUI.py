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
        self.preview_panel.set_control_panel(self.control_panel)
        self.main_paned.add(self.preview_panel, minsize=600)
        
        # Wire up callbacks
        self._setup_callbacks()
        # Initialize state
        self.displacement_results = []
        self.current_image = None
        self.image_files = []
        self.roi_mask = None
        
        # Probe Manager
        from raft_dic_gui.probe_manager import ProbeManager
        self.probe_manager = ProbeManager()
        self.roi_rect = None
        self.current_probe_mode = 'point'
        
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
        self.control_panel.callbacks['on_model_selected'] = self._update_roi_status_text
        
        self.control_panel.callbacks['on_safety_factor_change'] = lambda: (self._update_roi_status_text(), self.preview_panel.draw_roi())
        self.control_panel.callbacks['on_overlap_change'] = self.preview_panel.draw_roi
        self.control_panel.callbacks['on_show_tiles_change'] = self.preview_panel.draw_roi
        
        # Preview Panel callbacks
        self.preview_panel.callbacks['on_roi_confirmed'] = self._on_roi_confirmed
        self.preview_panel.callbacks['on_tab_changed'] = self._on_preview_tab_changed
        self.preview_panel.callbacks['on_add_point'] = self._on_add_point_click
        self.preview_panel.callbacks['on_add_line'] = self._on_add_line
        self.preview_panel.callbacks['on_tool_finished'] = lambda: print("[DEBUG] Tool finished")
        self.preview_panel.callbacks['on_area_added'] = self._on_area_added
        
        # Probe callbacks
        self.control_panel.post_processing_panel.callbacks['add_point_probe'] = self._on_add_point_probe
        self.control_panel.post_processing_panel.callbacks['add_line_probe'] = self._on_add_line_probe
        self.control_panel.post_processing_panel.callbacks['add_area_probe'] = self._on_add_area_probe
        self.control_panel.post_processing_panel.callbacks['remove_probe'] = self._on_remove_probe
        self.control_panel.post_processing_panel.callbacks['clear_probes'] = self._on_clear_probes
        self.control_panel.post_processing_panel.callbacks['select_probe'] = self._on_select_probe
        self.control_panel.post_processing_panel.callbacks['probe_mode_changed'] = self._on_probe_mode_changed
        self.control_panel.post_processing_panel.callbacks['calculate_strain'] = self.calculate_strain
        self.control_panel.post_processing_panel.callbacks['update_post_preview'] = self.update_post_preview

    def _on_probe_mode_changed(self, mode):
        """Handle probe mode change (point/line/area)."""
        print(f"[DEBUG] Probe mode changed to: {mode}")
        self.current_probe_mode = mode
        
        # Stop any active tools from other modes
        if mode != 'point':
            self.preview_panel._stop_point_tool()
        if mode != 'line':
            self.preview_panel._stop_line_tool()
            
        self._update_probe_ui()
        self.update_post_preview()

    def calculate_strain(self):
        """Handle strain calculation request."""
        print("[DEBUG] Calculate Strain requested.")
        
        if not self.displacement_results:
            messagebox.showwarning("Warning", "No displacement results available. Please run analysis first.")
            return

        # Get parameters
        pp = self.control_panel.post_processing_panel
        method_display = pp.strain_method.get()
        # Map display string to internal key
        if "Green-Lagrange" in method_display:
            method = 'green_lagrange'
        elif "Engineering" in method_display:
            method = 'engineering'
        else:
            method = 'green_lagrange' # Default
        
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
                
                # Update component list
                if self.strain_results:
                    keys = list(self.strain_results[0].keys())
                    self._ui_call(pp.update_component_list, keys)
                
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
        # print("[DEBUG] update_post_preview called")
        
        has_strain = hasattr(self, 'strain_results') and self.strain_results
        has_disp = hasattr(self, 'displacement_results') and self.displacement_results
        
        if not has_strain and not has_disp:
            # print("[DEBUG] No results found.")
            return
            
        # Determine total frames based on what's available
        if has_strain:
            total_frames = len(self.strain_results)
        else:
            total_frames = len(self.displacement_results)
            
        # Recursion guard
        if getattr(self, '_updating_ui', False):
            return
        self._updating_ui = True
        
        try:
            # Get current settings
            pp = self.control_panel.post_processing_panel
            comp_name = pp.vis_component.get()
            
            # Get current frame from slider or entry
            if hasattr(self.preview_panel, 'post_frame_slider'):
                current_frame = int(float(self.preview_panel.post_frame_slider.get())) - 1
            else:
                current_frame = 0
                
            # Bounds check
            if current_frame < 0: current_frame = 0
            if current_frame >= total_frames: current_frame = total_frames - 1
            
            # Get data for this frame
            data_map = None
            if comp_name in ['u', 'v']:
                # Displacement
                idx = 0 if comp_name == 'u' else 1
                if current_frame < len(self.displacement_results):
                    d = self.displacement_results[current_frame]
                    if isinstance(d, str):
                        d = np.load(d)
                    data_map = d[..., idx]
            else:
                # Strain
                if current_frame < len(self.strain_results):
                    s = self.strain_results[current_frame]
                    if comp_name in s:
                        data_map = s[comp_name]
                        
            if data_map is None:
                self._updating_ui = False
                return
                
            # Get visualization params
            cmap = pp.vis_colormap.get()
            try:
                alpha = float(pp.vis_alpha.get())
            except ValueError:
                alpha = 0.7
                
            # Range handling
            vmin, vmax = None, None
            is_fixed = pp.post_fixed_range.get()
            
            if is_fixed:
                try:
                    vmin = float(pp.post_vmin.get())
                    vmax = float(pp.post_vmax.get())
                    pp.component_ranges[comp_name] = (vmin, vmax)
                except ValueError:
                    if comp_name in pp.component_ranges:
                        vmin, vmax = pp.component_ranges[comp_name]
                        pp.post_vmin.set(f"{vmin:.4f}")
                        pp.post_vmax.set(f"{vmax:.4f}")
            else:
                vmin = np.nanmin(data_map)
                vmax = np.nanmax(data_map)
                pp.post_vmin.set(f"{vmin:.4f}")
                pp.post_vmax.set(f"{vmax:.4f}")
            
            # Load background image
            bg_img = None
            if self.image_files:
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
            
            # Update Controls
            if hasattr(self.preview_panel, 'post_frame_slider'):
                self.preview_panel.post_frame_slider.configure(to=total_frames)
                # Don't set slider value here to avoid recursion
                
            if hasattr(self.preview_panel, 'post_frame_entry'):
                self.preview_panel.post_frame_entry.delete(0, tk.END)
                self.preview_panel.post_frame_entry.insert(0, str(current_frame + 1))
                
            if hasattr(self.preview_panel, 'post_total_frames_label'):
                self.preview_panel.post_total_frames_label.configure(text=f"/{total_frames}")

            # Update Plots
            self._update_plots(comp_name)
            
            if hasattr(self.preview_panel, 'post_current_image_name'):
                if current_frame < len(self.image_files):
                    self.preview_panel.post_current_image_name.configure(text=self.image_files[current_frame])
            
            # Redraw probes
            self._update_probe_ui()
            
        except Exception as e:
            print(f"[ERROR] Error in update_post_preview: {e}")
        finally:
            self._updating_ui = False

    def _on_add_point_probe(self):
        """Start point probe tool."""
        self.preview_panel.start_point_tool()

    def _on_add_point_click(self, x, y):
        """Handle point added from canvas."""
        self.probe_manager.add_point(x, y)
        self._update_probe_ui()
        self.update_post_preview() # Update graph
        
    def _on_remove_probe(self):
        """Remove selected probe."""
        pp = self.control_panel.post_processing_panel
        # Check current mode to know which list to check
        if self.current_probe_mode == 'point':
            selection = pp.probe_list.selection()
            if not selection: return
            for item in selection:
                vals = pp.probe_list.item(item)['values']
                pid = vals[0]
                self.probe_manager.remove_probe_by_id_type(pid, 'point')
        elif self.current_probe_mode == 'line':
            if hasattr(pp, 'line_list'):
                selection = pp.line_list.selection()
                if not selection: return
                for item in selection:
                    vals = pp.line_list.item(item)['values']
                    pid = vals[0]
                    self.probe_manager.remove_probe_by_id_type(pid, 'line')
        elif self.current_probe_mode == 'area':
            if hasattr(pp, 'area_list'):
                selection = pp.area_list.selection()
                if not selection: return
                for item in selection:
                    vals = pp.area_list.item(item)['values']
                    pid = vals[0]
                    self.probe_manager.remove_probe_by_id_type(pid, 'area')
            
        self._update_probe_ui()
        self.update_post_preview()
        
    def _on_clear_probes(self):
        """Clear probes of the current type."""
        self.probe_manager.clear_by_type(self.current_probe_mode)
        self._update_probe_ui()
        self.update_post_preview()
        
    def _on_add_line_probe(self):
        """Start line probe tool."""
        self.preview_panel.start_line_tool()
        
    def _on_add_line(self, p1, p2):
        """Handle line added from canvas."""
        self.probe_manager.add_line(p1, p2)
        self._update_probe_ui()
        self.update_post_preview()
        
    def _on_select_probe(self):
        """Handle probe selection in list."""
        # Trigger update to switch between Point Graph and Kymograph
        self.update_post_preview() # This calls _update_plots

    def _update_probe_ui(self):
        """Update probe list and canvas."""
        pp = self.control_panel.post_processing_panel
        pp.update_probe_list(self.probe_manager.probes)
        self.preview_panel.draw_probes(self.probe_manager.probes, mode=self.current_probe_mode)

    def _update_plots(self, comp_name):
        """Update the time-series plot for probes."""
        # The graph is in PreviewPanel, not PostProcessingPanel
        if not hasattr(self.preview_panel, 'post_graph_ax'):
            return
            
        ax = self.preview_panel.post_graph_ax
        canvas = self.preview_panel.post_graph_canvas
        
        # Clear axes completely to prevent shrinking/overlay issues
        ax.clear()
        
        # Remove existing colorbar if present
        if hasattr(self, 'kymo_cb') and self.kymo_cb:
            try:
                self.kymo_cb.remove()
            except Exception:
                pass
            self.kymo_cb = None
        
        # Get data source based on component
        data_list = []
        if comp_name in ['u', 'v']:
            # Extract displacement component
            idx = 0 if comp_name == 'u' else 1
            for d in self.displacement_results:
                if isinstance(d, str):
                    d = np.load(d)
                data_list.append(d[..., idx])
        else:
            # Extract strain component
            for s in self.strain_results:
                if comp_name in s:
                    data_list.append(s[comp_name])
                else:
                    data_list.append(None)
                    
        # Calculate scale factors and offset
        scale_factors = (1.0, 1.0)
        offset = (0, 0)
        
        if data_list and data_list[0] is not None:
            data_h, data_w = data_list[0].shape
            
            # Get ROI info
            if hasattr(self, 'roi_rect') and self.roi_rect is not None:
                xmin, ymin, xmax, ymax = self.roi_rect
                roi_w = xmax - xmin
                roi_h = ymax - ymin
                offset = (xmin, ymin)
            else:
                # Fallback to full image if no ROI (shouldn't happen usually)
                roi_w, roi_h = data_w, data_h
                if hasattr(self, '_cached_ref_img') and self._cached_ref_img is not None:
                    roi_h, roi_w = self._cached_ref_img.shape[:2]
            
            # Avoid division by zero
            sy = data_h / roi_h if roi_h > 0 else 1.0
            sx = data_w / roi_w if roi_w > 0 else 1.0
            scale_factors = (sy, sx)

        # Extract probe data based on mode
        if self.current_probe_mode == 'point':
            results = self.probe_manager.extract_time_series(data_list, scale_factors=scale_factors, offset=offset)
            
            # Plot Point Probes
            frames = range(len(data_list))
            for pid, values in results.items():
                # Find probe color
                color = next((p.color for p in self.probe_manager.probes if p.id == pid), 'black')
                ax.plot(frames, values, label=f"P{pid}", color=color, marker='o', markersize=3)
                
            ax.set_xlabel("Frame")
            ax.set_ylabel(comp_name.upper())
            ax.grid(True, alpha=0.3)
            if results:
                ax.legend()
                
        elif self.current_probe_mode == 'line':
            # Check if a Line Probe is selected
            pp = self.control_panel.post_processing_panel
            selected_line_id = None
            if hasattr(pp, 'line_list'):
                sel = pp.line_list.selection()
                if sel:
                    selected_line_id = pp.line_list.item(sel[0])['values'][0]
            
            # Get Metric
            metric = 'avg'
            if hasattr(pp, 'line_metric_var'):
                val = pp.line_metric_var.get()
                if val == 'Maximum': metric = 'max'
                elif val == 'Minimum': metric = 'min'
            
            results = {}
            for p in self.probe_manager.probes:
                if p.type == 'line':
                    # Extract series
                    vals = self.probe_manager.extract_line_series(data_list, p.id, metric=metric, scale_factors=scale_factors, offset=offset)
                    if vals is not None:
                        results[p.id] = vals
            
            frames = range(len(data_list))
            for pid, values in results.items():
                # Find probe color
                color = next((p.color for p in self.probe_manager.probes if p.id == pid), 'black')
                # Highlight selected line
                linewidth = 3 if pid == selected_line_id else 1.5
                alpha = 1.0 if pid == selected_line_id else 0.7
                
                label = f"L{pid} ({metric.capitalize()})"
                ax.plot(frames, values, label=label, color=color, linewidth=linewidth, alpha=alpha)
                
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{metric.capitalize()} {comp_name.upper()}")
            ax.grid(True, alpha=0.3)
            if results:
                ax.legend()
                
        elif self.current_probe_mode == 'area':
            # Check if an Area Probe is selected
            pp = self.control_panel.post_processing_panel
            selected_area_id = None
            if hasattr(pp, 'area_list'):
                sel = pp.area_list.selection()
                if sel:
                    selected_area_id = pp.area_list.item(sel[0])['values'][0]
            
            # Get Metric
            metric = 'avg'
            if hasattr(pp, 'area_metric_var'):
                val = pp.area_metric_var.get()
                if val == 'Maximum': metric = 'max'
                elif val == 'Minimum': metric = 'min'
            
            results = {}
            for p in self.probe_manager.probes:
                if p.type == 'area':
                    # Extract series
                    vals = self.probe_manager.extract_area_series(data_list, p.id, metric=metric, scale_factors=scale_factors, offset=offset)
                    if vals is not None:
                        results[p.id] = vals
            
            frames = range(len(data_list))
            for pid, values in results.items():
                # Find probe color
                color = next((p.color for p in self.probe_manager.probes if p.id == pid), 'black')
                # Highlight selected area
                linewidth = 3 if pid == selected_area_id else 1.5
                alpha = 1.0 if pid == selected_area_id else 0.7
                
                label = f"A{pid} ({metric.capitalize()})"
                ax.plot(frames, values, label=label, color=color, linewidth=linewidth, alpha=alpha)
                
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{metric.capitalize()} {comp_name.upper()}")
            ax.grid(True, alpha=0.3)
            if results:
                ax.legend()
                
        canvas.draw()

    def _on_add_area_probe(self, shape_type):
        """Start area probe tool."""
        self.preview_panel.start_area_tool(shape_type)
        
    def _on_area_added(self, shape_type, coords):
        """Handle area added from canvas."""
        self.probe_manager.add_area(shape_type, coords)
        self._update_probe_ui()
        self.update_post_preview()

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
        print(f"[DEBUG] _on_roi_confirmed called. Mask shape: {mask.shape if mask is not None else 'None'}")
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
        
        print(f"[DEBUG] validate_inputs: roi_mask is {self.roi_mask is not None}")
        if self.roi_mask is None:
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







