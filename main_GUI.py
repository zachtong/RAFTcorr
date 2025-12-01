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
Date: 2025-06-10
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
                result['value'] = func(*args, **kwargs)
            finally:
                event.set()

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
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        try:
                            if widget is self.stop_button:
                                widget.configure(state='normal' if running else 'disabled')
                            else:
                                widget.configure(state=state)
                        except Exception:
                            pass

            try:
                self.run_button.configure(state='disabled' if running else 'normal')
            except Exception:
                pass
            try:
                self.stop_button.configure(state='normal' if running else 'disabled')
            except Exception:
                pass

        self._ui_call(_toggle, wait=True)

    def _update_progress(self, percent: float = None, current: int = None, total: int = None):
        """Update progress bar and text safely from any thread."""
        def _apply():
            if percent is not None:
                try:
                    self.progress.configure(value=percent)
                except Exception:
                    pass
            if current is not None and total is not None:
                try:
                    self.progress_text.configure(text=f"{current}/{total}")
                except Exception:
                    pass

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
        
        # Preview Panel callbacks
        self.preview_panel.control_panel = self.control_panel
        self.preview_panel.callbacks['on_roi_confirmed'] = self._on_roi_confirmed

    def _on_roi_confirmed(self, mask, rect):
        self.roi_mask = mask
        self.roi_rect = rect
        
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
        self.stop_button.configure(state='disabled')

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







