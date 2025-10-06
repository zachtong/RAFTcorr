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
Run this script directly to launch the GUI application.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import cv2
from PIL import Image, ImageTk
import helpFunctions as hf
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import visualization as vis
import io
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

class CollapsibleFrame(ttk.LabelFrame):
    """A collapsible frame widget that can be expanded or collapsed."""
    def __init__(self, parent, text="", **kwargs):
        super().__init__(parent, text=text, **kwargs)
        
        # Create a container frame for the header
        self.header_frame = ttk.Frame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        # Add toggle button to header
        self.toggle_button = ttk.Button(self.header_frame, text="▼", width=2,
                                      command=self.toggle)
        self.toggle_button.grid(row=0, column=0, padx=(0,5))
        
        # Create content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=1)
        
        self.is_expanded = True
        
    def toggle(self):
        """Toggle the visibility of the content frame."""
        if self.is_expanded:
            self.content_frame.grid_remove()
            self.toggle_button.configure(text="▶")
        else:
            self.content_frame.grid()
            self.toggle_button.configure(text="▼")
        self.is_expanded = not self.is_expanded
        
    def get_content_frame(self):
        """Return the content frame for adding widgets."""
        return self.content_frame

class Tooltip:
    """Create a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Create top level window
        self.tooltip = tk.Toplevel(self.widget)
        # Remove window decorations
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        # Create tooltip content
        label = ttk.Label(self.tooltip, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief="solid", borderwidth=1)
        label.pack()
    
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class RAFTDICGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAFT-DIC Displacement Field Calculator")
        
        # Set minimum window size
        self.root.minsize(1200, 800)
        
        # Initialize variables
        self.init_variables()
        
        # Create main container with padding
        main_container = ttk.Frame(root, padding="5")
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure root grid
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # Create main PanedWindow for left-right split
        self.main_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.main_paned.grid(row=0, column=0, sticky="nsew")
        
        # Configure main container grid
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Create left control panel with fixed minimum width
        self.left_frame = ttk.Frame(self.main_paned, width=300)
        self.left_frame.grid_propagate(False)  # Prevent frame from shrinking
        self.main_paned.add(self.left_frame, weight=0)
        
        # Add vertical separator
        separator = ttk.Separator(main_container, orient="vertical")
        separator.grid(row=0, column=1, sticky="ns")
        
        # Create right preview panel
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=1)
        
        # Create control panel in left frame
        self.create_control_panel(self.left_frame)
        
        # Create preview panel in right frame
        self.create_preview_panel(self.right_frame)
        
        # Set initial paned window position
        self.root.update()
        self.main_paned.sashpos(0, 300)
        
        self.displacement_results = []
        self.current_displacement = None
        self.displacement_cache = {}
        self.cache_size = 5
        
    def init_variables(self):
        """Initialize variables"""
        # Path variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        
        # Processing parameter variables
        self.mode = tk.StringVar(value="accumulative")
        self.use_crop = tk.BooleanVar(value=False)  # New variable for crop control
        self.crop_size_w = tk.StringVar(value="0")  # Changed order to W first
        self.crop_size_h = tk.StringVar(value="0")  # Changed order to H second
        self.shift_size = tk.StringVar(value="256")  # Renamed from stride
        self.max_displacement = tk.StringVar(value="5")
        
        # Add colorbar range variables
        self.use_fixed_colorbar = tk.BooleanVar(value=False)
        self.colorbar_u_min = tk.StringVar(value="-1")
        self.colorbar_u_max = tk.StringVar(value="1")
        self.colorbar_v_min = tk.StringVar(value="-1")
        self.colorbar_v_max = tk.StringVar(value="1")
        
        # Add colormap variable
        self.colormap = tk.StringVar(value="viridis")
        
        # Modify smoothing processing related variables
        self.use_smooth = tk.BooleanVar(value=True)
        self.sigma = tk.StringVar(value="2.0")
        
        # Help tooltips text
        self.tooltips = {
            "path": "Select input image directory and output directory for results",
            "mode": "Accumulative: Calculate displacement relative to first frame\nIncremental: Calculate displacement relative to previous frame",
            "crop": "Enable/disable image cropping for processing",
            "crop_size": "Size of the cropping window (Width × Height)",
            "shift": "Step size for moving the cropping window",
            "max_disp": "Expected maximum displacement value",
            "smooth": "Apply Gaussian filter to smooth the results",
            "vis": "Visualization settings including colorbar range and colormap",
            "run": "Start processing the image sequence"
        }
        
        # ROI related variables
        self.roi_points = []          # Store ROI polygon vertices
        self.drawing_roi = False      # ROI drawing state flag
        self.roi_mask = None         # ROI binary mask
        self.roi_rect = None         # ROI bounding rectangle coordinates
        self.roi_scale_factor = 1.0  # ROI display scaling factor
        self.display_size = (400, 400)
        self.is_cutting_mode = False  # Whether in cutting mode
        
        # Image and result related variables
        self.current_image = None
        self.displacement_results = []
        
        # Add scaling related variables
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.current_photo = None  # Save current displayed PhotoImage
        
        # Add playback related variables
        self.is_playing = False
        self.play_after_id = None
        self.play_interval = 100  # Playback interval (ms)
        
        # Add overlay related variables
        self.overlay_type = tk.StringVar(value="quiver")
        self.overlay_density = tk.StringVar(value="20")
        self.overlay_alpha = tk.StringVar(value="0.5")  # Changed default to 0.5
        
        # Add background image mode variables for deformed image visualization
        self.background_mode = tk.StringVar(value="reference")  # reference or deformed
        self.use_smooth_interpolation = tk.BooleanVar(value=True)  # for deformed mode interpolation
        
    def create_control_panel(self, parent):
        """Create control panel with collapsible sections"""
        # Main control frame with scrollbar
        control_canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)
        
        # Configure scrolling
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout for scrollbar and canvas
        control_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure parent grid
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        # Create window in canvas
        canvas_frame = control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        
        # Configure control frame
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Path selection section
        path_frame = CollapsibleFrame(control_frame, text="Path Settings")
        path_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Add help button to header
        help_btn = self.create_help_button(path_frame.header_frame, "path")
        help_btn.grid(row=0, column=1, padx=2)
        
        path_content = path_frame.get_content_frame()
        path_content.grid_columnconfigure(1, weight=1)  # Make entry expand
        
        # Input path
        ttk.Label(path_content, text="Input Directory:").grid(row=0, column=0, sticky="w", padx=5)
        input_entry = ttk.Entry(path_content, textvariable=self.input_path)
        input_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(path_content, text="Browse", command=self.browse_input, width=8).grid(row=0, column=2, padx=5)
        
        # Output path
        ttk.Label(path_content, text="Output Directory:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        output_entry = ttk.Entry(path_content, textvariable=self.output_path)
        output_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(path_content, text="Browse", command=self.browse_output, width=8).grid(row=1, column=2, padx=5, pady=5)
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=5)
        
        # Processing mode section
        mode_frame = CollapsibleFrame(control_frame, text="Processing Mode")
        mode_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Add help button to header
        help_btn = self.create_help_button(mode_frame.header_frame, "mode")
        help_btn.grid(row=0, column=1, padx=2)
        
        mode_content = mode_frame.get_content_frame()
        mode_content.grid_columnconfigure(0, weight=1)
        
        # Mode radio buttons with padding
        ttk.Radiobutton(mode_content, text="Accumulative", variable=self.mode, 
                       value="accumulative").grid(row=0, column=0, sticky="w", padx=20, pady=2)
        ttk.Radiobutton(mode_content, text="Incremental", variable=self.mode, 
                       value="incremental").grid(row=1, column=0, sticky="w", padx=20, pady=2)
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=5)
        
        # Parameters section
        param_frame = CollapsibleFrame(control_frame, text="Parameters")
        param_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        param_content = param_frame.get_content_frame()
        param_content.grid_columnconfigure(0, weight=1)
        
        # Basic parameters
        basic_frame = ttk.LabelFrame(param_content, text="Basic Settings", padding=5)
        basic_frame.grid(row=0, column=0, sticky="ew", pady=5)
        basic_frame.grid_columnconfigure(1, weight=1)
        
        # Add help button
        help_btn = self.create_help_button(basic_frame, "crop")
        help_btn.grid(row=0, column=2, padx=2)
        
        # Add crop checkbox
        ttk.Checkbutton(basic_frame, text="Enable Crop", 
                       variable=self.use_crop,
                       command=self.update_crop_state).grid(row=0, column=0, 
                                                          columnspan=2, sticky="w", padx=5)
        
        # Crop Size settings with W×H order
        ttk.Label(basic_frame, text="Crop Size (W×H):").grid(row=1, column=0, sticky="w", padx=5)
        size_frame = ttk.Frame(basic_frame)
        size_frame.grid(row=1, column=1, sticky="w", pady=2)
        
        self.crop_w_entry = ttk.Entry(size_frame, textvariable=self.crop_size_w, width=5, state="disabled")
        self.crop_w_entry.grid(row=0, column=0)
        ttk.Label(size_frame, text="×").grid(row=0, column=1, padx=2)
        self.crop_h_entry = ttk.Entry(size_frame, textvariable=self.crop_size_h, width=5, state="disabled")
        self.crop_h_entry.grid(row=0, column=2)
        
        # Help button for crop size
        help_btn = self.create_help_button(size_frame, "crop_size")
        help_btn.grid(row=0, column=3, padx=2)
        
        # Shift size settings
        ttk.Label(basic_frame, text="Shift Size:").grid(row=2, column=0, sticky="w", padx=5)
        self.shift_entry = ttk.Entry(basic_frame, textvariable=self.shift_size, 
                                   width=10, state="disabled")
        self.shift_entry.grid(row=2, column=1, sticky="w", pady=2)
        
        # Help button for shift size
        help_btn = self.create_help_button(basic_frame, "shift")
        help_btn.grid(row=2, column=2, padx=2)
        
        # Advanced parameters
        advanced_frame = ttk.LabelFrame(param_content, text="Advanced Settings", padding=5)
        advanced_frame.grid(row=1, column=0, sticky="ew", pady=5)
        advanced_frame.grid_columnconfigure(1, weight=1)
        
        # Max Displacement settings
        ttk.Label(advanced_frame, text="Max Displacement:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(advanced_frame, textvariable=self.max_displacement, width=10).grid(row=0, column=1, sticky="w", pady=2)
        
        # Smoothing options
        smooth_frame = ttk.Frame(advanced_frame)
        smooth_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=2)
        
        ttk.Checkbutton(smooth_frame, text="Use Smoothing", 
                       variable=self.use_smooth).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(smooth_frame, text="Sigma:").grid(row=0, column=1, sticky="w", padx=(10,0))
        ttk.Entry(smooth_frame, textvariable=self.sigma, width=5).grid(row=0, column=2, padx=5)
        ttk.Label(smooth_frame, text="(0.5-5.0)").grid(row=0, column=3, sticky="w")
        
        # Visualization settings
        vis_frame = ttk.LabelFrame(param_content, text="Visualization Settings", padding=5)
        vis_frame.grid(row=2, column=0, sticky="ew", pady=5)
        vis_frame.grid_columnconfigure(0, weight=1)
        
        # Colorbar settings
        ttk.Checkbutton(vis_frame, text="Use Fixed Range", 
                       variable=self.use_fixed_colorbar).grid(row=0, column=0, sticky="w", padx=5)
        
        range_frame = ttk.Frame(vis_frame)
        range_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)
        range_frame.grid_columnconfigure(1, weight=1)
        range_frame.grid_columnconfigure(3, weight=1)
        
        # U Range
        ttk.Label(range_frame, text="U Range:").grid(row=0, column=0, sticky="w")
        ttk.Entry(range_frame, textvariable=self.colorbar_u_min, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(range_frame, text="to").grid(row=0, column=2, padx=2)
        ttk.Entry(range_frame, textvariable=self.colorbar_u_max, width=8).grid(row=0, column=3, padx=2)
        
        # V Range
        ttk.Label(range_frame, text="V Range:").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(range_frame, textvariable=self.colorbar_v_min, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(range_frame, text="to").grid(row=1, column=2, padx=2)
        ttk.Entry(range_frame, textvariable=self.colorbar_v_max, width=8).grid(row=1, column=3, padx=2)
        
        # Colormap selection
        ttk.Label(vis_frame, text="Colormap:").grid(row=2, column=0, sticky="w", padx=5, pady=(5,0))
        colormap_combo = ttk.Combobox(vis_frame, 
                                    textvariable=self.colormap,
                                    values=["viridis", "jet", "magma", "plasma", "inferno", 
                                           "cividis", "RdYlBu", "coolwarm"],
                                    width=15,
                                    state="readonly")
        colormap_combo.grid(row=3, column=0, sticky="w", padx=5, pady=(0,5))
        
        # Background Image Mode selection
        bg_mode_frame = ttk.LabelFrame(vis_frame, text="Background Image Mode", padding=5)
        bg_mode_frame.grid(row=4, column=0, sticky="ew", pady=5)
        bg_mode_frame.grid_columnconfigure(0, weight=1)
        
        # Mode radio buttons
        ttk.Radiobutton(bg_mode_frame, text="Reference Image", variable=self.background_mode, 
                       value="reference", command=self.update_displacement_preview).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(bg_mode_frame, text="Deformed Image", variable=self.background_mode, 
                       value="deformed", command=self.update_displacement_preview).grid(row=1, column=0, sticky="w", padx=5, pady=2)
        
        # Deformed mode options
        deformed_options_frame = ttk.Frame(bg_mode_frame)
        deformed_options_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        
        ttk.Checkbutton(deformed_options_frame, text="Smooth Interpolation", 
                       variable=self.use_smooth_interpolation,
                       command=self.update_displacement_preview).grid(row=0, column=0, sticky="w", padx=20)
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=5, column=0, sticky="ew", pady=5)
        
        # Run section
        run_frame = CollapsibleFrame(control_frame, text="Run Control")
        run_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        
        run_content = run_frame.get_content_frame()
        run_content.grid_columnconfigure(0, weight=1)
        
        ttk.Button(run_content, text="Run", command=self.run, width=15).grid(row=0, column=0, pady=5)
        self.progress = ttk.Progressbar(run_content, length=200, mode='determinate')
        self.progress.grid(row=1, column=0, pady=5)
        
        # Configure canvas scrolling
        def configure_scroll_region(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        
        def configure_canvas_width(event):
            control_canvas.itemconfig(canvas_frame, width=event.width)
        
        control_frame.bind("<Configure>", configure_scroll_region)
        control_canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mouse wheel
        def on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        control_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Bind events for parameter changes
        self.crop_h_entry.bind('<Return>', self.on_param_change)
        self.crop_h_entry.bind('<FocusOut>', self.on_param_change)
        self.crop_w_entry.bind('<Return>', self.on_param_change)
        self.crop_w_entry.bind('<FocusOut>', self.on_param_change)
        self.shift_entry.bind('<Return>', self.on_param_change)
        self.shift_entry.bind('<FocusOut>', self.on_param_change)

    def create_preview_panel(self, parent):
        """Create preview panel"""
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding="5")
        preview_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(2, weight=1)
        
        # ROI selection area (left)
        roi_frame = ttk.LabelFrame(preview_frame, text="ROI Selection", padding="5")
        roi_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        roi_frame.grid_rowconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(0, weight=1)
        
        # Create Canvas for ROI selection
        self.roi_canvas = tk.Canvas(roi_frame, width=400, height=400)
        self.roi_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbars
        x_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.HORIZONTAL, command=self.roi_canvas.xview)
        y_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.VERTICAL, command=self.roi_canvas.yview)
        x_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        y_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.roi_canvas.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        
        # Bind mouse wheel events
        self.roi_canvas.bind('<MouseWheel>', self.on_mousewheel)  # Windows
        self.roi_canvas.bind('<Button-4>', self.on_mousewheel)    # Linux up scroll
        self.roi_canvas.bind('<Button-5>', self.on_mousewheel)    # Linux down scroll
        
        # Add scaling control buttons
        zoom_frame = ttk.Frame(roi_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(zoom_frame, text="Zoom in", command=lambda: self.zoom(1.2)).grid(row=0, column=0, padx=2)
        ttk.Button(zoom_frame, text="Zomm out", command=lambda: self.zoom(0.8)).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.reset_zoom).grid(row=0, column=2, padx=2)
        
        # ROI control buttons
        self.roi_controls = ttk.Frame(roi_frame)
        self.roi_controls.grid(row=3, column=0, columnspan=2, pady=5)
        
        ttk.Button(self.roi_controls, text="Draw ROI", 
                   command=self.start_roi_drawing).grid(row=0, column=0, padx=2)
        ttk.Button(self.roi_controls, text="Cut ROI", 
                   command=self.start_cutting_roi).grid(row=0, column=1, padx=2)
        ttk.Button(self.roi_controls, text="Clear ROI", 
                   command=self.clear_roi).grid(row=0, column=2, padx=2)
        
        # Confirm button is created at initialization but not displayed
        self.confirm_roi_btn = ttk.Button(self.roi_controls, text="Confirm ROI", 
                                         command=self.confirm_roi)
        self.confirm_roi_btn.grid(row=0, column=3, padx=2)
        self.confirm_roi_btn.grid_remove()  # Initially hidden
        
        # Crop preview area (middle)
        crop_frame = ttk.LabelFrame(preview_frame, text="Crop Windows Preview", padding="5")
        crop_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        crop_frame.grid_rowconfigure(0, weight=1)
        crop_frame.grid_columnconfigure(0, weight=1)
        
        # Create Canvas instead of Label to display preview
        self.preview_canvas = tk.Canvas(crop_frame)
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind size change events
        self.preview_canvas.bind('<Configure>', self.on_preview_canvas_resize)
        
        # Displacement overlay area (right)
        disp_frame = ttk.LabelFrame(preview_frame, text="Displacement Field Overlay", padding="5")
        disp_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        disp_frame.grid_rowconfigure(0, weight=1)
        disp_frame.grid_columnconfigure(0, weight=1)
        
        # Add displacement overlay display
        self.displacement_label = ttk.Label(disp_frame)
        self.displacement_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add control panel at the bottom
        control_frame = ttk.Frame(disp_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add playback control buttons
        play_control = ttk.Frame(control_frame)
        play_control.grid(row=0, column=0, padx=5)
        
        # Add play/pause button
        self.play_icon = "▶"  # Play icon
        self.pause_icon = "⏸"  # Pause icon
        self.is_playing = False
        self.play_button = ttk.Button(play_control, 
                                    text=self.play_icon,
                                    width=3,
                                    command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=2)
        
        # Add frame control
        frame_control = ttk.Frame(control_frame)
        frame_control.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Add Previous button
        ttk.Button(frame_control, text="◀", width=3,
                  command=self.previous_frame).grid(row=0, column=0, padx=2)
        
        ttk.Label(frame_control, text="Frame:").grid(row=0, column=1, padx=5)
        
        # Add frame number input box
        self.frame_entry = ttk.Entry(frame_control, width=5)
        self.frame_entry.grid(row=0, column=2, padx=2)
        
        # Add total frame number display
        self.total_frames_label = ttk.Label(frame_control, text="/1")
        self.total_frames_label.grid(row=0, column=3, padx=2)
        
        # Add Go button
        ttk.Button(frame_control, text="Go", 
                  command=self.jump_to_frame).grid(row=0, column=4, padx=5)
        
        # Add Next button
        ttk.Button(frame_control, text="▶", width=3,
                  command=self.next_frame).grid(row=0, column=5, padx=2)
        
        # Add current image name display
        self.current_image_name = ttk.Label(frame_control, text="")
        self.current_image_name.grid(row=0, column=6, padx=5)
        
        # Add playback speed control
        speed_frame = ttk.Frame(play_control)
        speed_frame.grid(row=0, column=1, padx=5)
        
        ttk.Label(speed_frame, text="Speed:").grid(row=0, column=0)
        self.speed_var = tk.StringVar(value="1x")
        speed_menu = ttk.OptionMenu(speed_frame, self.speed_var, "1x",
                                  "0.25x", "0.5x", "1x", "2x", "4x",
                                  command=self.change_play_speed)
        speed_menu.grid(row=0, column=1)
        
        # Add transparency control
        alpha_frame = ttk.Frame(control_frame)
        alpha_frame.grid(row=0, column=2, padx=5)
        
        ttk.Label(alpha_frame, text="Transparency:").grid(row=0, column=0, padx=2)
        self.overlay_alpha = tk.StringVar(value="0.5")
        alpha_entry = ttk.Entry(alpha_frame,
                              textvariable=self.overlay_alpha,
                              width=5)
        alpha_entry.grid(row=0, column=1, padx=2)
        
        # Add update button
        ttk.Button(alpha_frame, 
                  text="Update",
                  command=self.update_displacement_preview).grid(row=0, column=2, padx=5)
        
        # Slider
        self.frame_slider = ttk.Scale(disp_frame, 
                                    from_=1, 
                                    to=1,
                                    orient=tk.HORIZONTAL,
                                    command=self.update_displacement_preview)
        self.frame_slider.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Image information display
        self.image_info = ttk.Label(preview_frame, text="")
        self.image_info.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
    def browse_input(self):
        """Browse input directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.input_path.set(directory)
            # Get image file list
            self.image_files = sorted([f for f in os.listdir(directory) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
            # Clear previous ROI and preview
            self.clear_roi()
            # Update image information
            self.update_image_info(directory)
            
    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_path.set(directory)
    
    def update_image_info(self, directory):
        """Update image information"""
        try:
            # Get image file list
            image_files = sorted([f for f in os.listdir(directory) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
            
            if not image_files:
                messagebox.showerror("Error", "No valid image files found in the directory")
                return
            
            # Save image file list for later use
            self.image_files = image_files
            
            # Read first image, use function from helpFunctions
            img_path = os.path.join(directory, image_files[0])
            img = hf.load_and_convert_image(img_path)
            if img is None:
                raise Exception(f"Failed to load image: {img_path}")
            
            # Store current image
            self.current_image = img
            
            # Update crop size with image dimensions
            self.update_crop_size()
            
            # Update ROI preview
            self.update_roi_label(img)
            
            # Update information display
            h, w = img.shape[:2]
            info_text = f"Image size: {w}x{h}\nNumber of images: {len(image_files)}"
            self.image_info.config(text=info_text)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def on_preview_canvas_resize(self, event):
        """Respond to preview canvas size change"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            self.update_preview()

    def update_preview(self, *args):
        """Update preview"""
        if self.current_image is None:
            return
        
        try:
            # Get ROI rectangle dimensions
            if self.roi_rect:
                xmin, ymin, xmax, ymax = self.roi_rect
                roi_h, roi_w = ymax - ymin, xmax - xmin
                
                # Get user input parameters, but do not force them within ROI dimensions
                try:
                    crop_h = int(self.crop_size_h.get() or "128")
                    crop_w = int(self.crop_size_w.get() or "128")
                    shift = int(self.shift_size.get() or "64")
                except ValueError:
                    crop_h, crop_w = 128, 128
                    shift = 64
                
                # Only used for preview display when adjusting size
                preview_crop_h = min(crop_h, roi_h)
                preview_crop_w = min(crop_w, roi_w)
                preview_shift = min(shift, min(preview_crop_h, preview_crop_w))
                
                # Update preview
                preview_image = self.current_image[ymin:ymax, xmin:xmax].copy()
                vis.update_preview(
                    self.preview_canvas,
                    preview_image,
                    crop_size=(preview_crop_h, preview_crop_w),
                    shift=preview_shift
                )
        except Exception as e:
            print(f"Error in update_preview: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def validate_inputs(self):
        """Validate input parameters and adjust to valid values"""
        try:
            if not os.path.exists(self.input_path.get()):
                raise ValueError("Input directory does not exist")
            
            # Get ROI dimensions
            if self.roi_rect is None:
                raise ValueError("Please select ROI first")
            
            xmin, ymin, xmax, ymax = self.roi_rect
            roi_h, roi_w = ymax - ymin, xmax - xmin
            
            if self.use_crop.get():
                # Only validate crop and shift parameters if crop is enabled
                try:
                    crop_h = int(self.crop_size_h.get())
                    crop_w = int(self.crop_size_w.get())
                    shift = int(self.shift_size.get())
                except ValueError:
                    raise ValueError("Invalid crop size or shift value")
                
                # Adjust crop size
                if crop_h < 128 or crop_w < 128:
                    print("Warning: Crop size too small, using minimum size of 128")
                    crop_h = max(128, crop_h)
                    crop_w = max(128, crop_w)
                    self.crop_size_h.set(str(crop_h))
                    self.crop_size_w.set(str(crop_w))
                
                if crop_h > roi_h or crop_w > roi_w:
                    print("Warning: Crop size larger than ROI, using ROI dimensions")
                    crop_h = min(crop_h, roi_h)
                    crop_w = min(crop_w, roi_w)
                    self.crop_size_h.set(str(crop_h))
                    self.crop_size_w.set(str(crop_w))
                
                # Adjust shift
                if shift < 32:
                    print("Warning: Shift too small, using minimum value of 32")
                    shift = 32
                    self.shift_size.set(str(shift))
                
                if shift > min(crop_h, crop_w):
                    print("Warning: Shift larger than crop size, using crop size")
                    shift = min(crop_h, crop_w)
                    self.shift_size.set(str(shift))
            
            # Verify max displacement
            max_displacement = int(self.max_displacement.get())
            if max_displacement < 0:
                raise ValueError("Max displacement must be non-negative")
            
            # Verify smoothing processing parameters
            try:
                sigma = float(self.sigma.get())
                if not 0.5 <= sigma <= 5.0:
                    print("Warning: Sigma value should be between 0.5 and 5.0")
                    sigma = np.clip(sigma, 0.5, 5.0)
                    self.sigma.set(f"{sigma:.2f}")
            except ValueError:
                print("Warning: Invalid sigma value, using default 2.0")
                self.sigma.set("2.0")
            
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False
    
    def process_images(self, args):
        """Process all images"""
        # Get image file list
        image_files = sorted([f for f in os.listdir(args.img_dir) 
                          if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))])

        if len(image_files) < 2:
            raise Exception("At least 2 images are needed")

        # Ensure ROI is set
        if self.roi_rect is None:
            raise Exception("Please select ROI area first")

        # Get ROI rectangle area and mask
        xmin, ymin, xmax, ymax = self.roi_rect

        # Create results directory
        results_dir = os.path.join(args.project_root, "displacement_results_npy")
        os.makedirs(results_dir, exist_ok=True)

        # Load reference image
        ref_img_path = os.path.join(args.img_dir, image_files[0])
        ref_image = hf.load_and_convert_image(ref_img_path)
        
        # Extract ROI area
        ref_roi = ref_image[ymin:ymax, xmin:xmax]

        # Process each image pair
        total_pairs = len(image_files) - 1
        for i in range(1, len(image_files)):
            # Update progress bar
            self.progress['value'] = (i / total_pairs) * 100
            self.root.update()

            # Load deformed image
            def_img_path = os.path.join(args.img_dir, image_files[i])
            def_image = hf.load_and_convert_image(def_img_path)
            def_roi = def_image[ymin:ymax, xmin:xmax]

            # Extract ROI mask corresponding area
            roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None

            # Process image pair
            if self.use_crop.get():
                # Use crop mode with specified parameters
                displacement_field, _ = hf.cut_image_pair_with_flow(
                    ref_roi, def_roi,
                    args.project_root,
                    args.model,
                    args.device,
                    crop_size=args.crop_size,
                    shift=args.shift,
                    maxDisplacement=args.max_displacement,
                    plot_windows=(i == 1),
                    roi_mask=roi_mask_crop,
                    use_smooth=args.use_smooth,
                    sigma=args.sigma
                )
            else:
                # Process entire ROI without cropping
                displacement_field, _ = hf.process_image_pair(
                    ref_roi, def_roi,
                    args.project_root,
                    args.model,
                    args.device,
                    maxDisplacement=args.max_displacement,
                    roi_mask=roi_mask_crop,
                    use_smooth=args.use_smooth,
                    sigma=args.sigma
                )

            # Save displacement field results with coordinate information
            hf.save_displacement_results(displacement_field, args.project_root, i, 
                                       roi_rect=self.roi_rect, roi_mask=self.roi_mask)

            # If incremental mode, update reference image
            if args.mode == "incremental":
                ref_roi = def_roi.copy()

        # Save results path
        self.displacement_results = [
            os.path.join(results_dir, f"displacement_field_{i}.npy")
            for i in range(1, len(image_files))
        ]
        
        # Update slider range
        if self.displacement_results:
            self.frame_slider.configure(to=len(self.displacement_results))
            self.frame_slider.set(1)
            self.total_frames_label.configure(text=f"/{len(self.displacement_results)}")
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, "1")
            self.update_displacement_preview()

    def update_displacement_preview(self, *args):
        """Update displacement field preview with support for both reference and deformed image modes"""
        if not self.displacement_results or not hasattr(self, 'current_image'):
            return
        
        try:
            # Get current frame index
            current_frame = int(self.frame_slider.get()) - 1
            if current_frame < 0 or current_frame >= len(self.displacement_results):
                return
            
            # Update frame number input box
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(current_frame + 1))
            
            # Update current image name display
            if hasattr(self, 'image_files') and len(self.image_files) > current_frame + 1:
                self.current_image_name.configure(text=self.image_files[current_frame + 1])
            
            # Load displacement field
            displacement = np.load(self.displacement_results[current_frame])
            
            if self.roi_rect is None:
                return
            
            # Get displacement components for colorbar range calculation
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            
            try:
                alpha = float(self.overlay_alpha.get())
            except ValueError:
                alpha = 0.5
            
            # Ensure valid values
            alpha = max(0.1, min(1.0, alpha))   # Limit between 0.1 and 1.0
            
            # Get current colormap
            current_colormap = self.colormap.get()
            
            # Determine colorbar ranges
            if self.use_fixed_colorbar.get():
                try:
                    vmin_u = float(self.colorbar_u_min.get())
                    vmax_u = float(self.colorbar_u_max.get())
                    vmin_v = float(self.colorbar_v_min.get())
                    vmax_v = float(self.colorbar_v_max.get())
                except ValueError:
                    vmin_u = np.nanmin(u)
                    vmax_u = np.nanmax(u)
                    vmin_v = np.nanmin(v)
                    vmax_v = np.nanmax(v)
            else:
                vmin_u = np.nanmin(u)
                vmax_u = np.nanmax(u)
                vmin_v = np.nanmin(v)
                vmax_v = np.nanmax(v)
            
            # Create visualization based on selected background mode
            if self.background_mode.get() == "deformed":
                fig = self.create_deformed_displacement_visualization(
                    displacement, current_frame, alpha, current_colormap, 
                    vmin_u, vmax_u, vmin_v, vmax_v)
            else:  # reference mode (default)
                fig = self.create_reference_displacement_visualization(
                    displacement, self.current_image, alpha, current_colormap, 
                    vmin_u, vmax_u, vmin_v, vmax_v)
            
            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       pad_inches=0)
            plt.close(fig)
            buf.seek(0)
            
            # Update preview
            preview = Image.open(buf)
            w, h = preview.size
            max_size = (800, 400)  # Increased size for better visibility
            scale = min(max_size[0]/w, max_size[1]/h)
            display_size = (int(w*scale), int(h*scale))
            preview = preview.resize(display_size, Image.LANCZOS)
            
            # Update label
            photo = ImageTk.PhotoImage(preview)
            self.displacement_label.configure(image=photo)
            self.displacement_label.image = photo
            
            # If last frame and playing, reset to first frame
            if self.is_playing and current_frame == len(self.displacement_results) - 1:
                self.frame_slider.set(1)
            
        except Exception as e:
            print(f"Error in update_displacement_preview: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_deformed_displacement_visualization(self, displacement, current_frame, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
        """
        Create displacement field visualization on deformed image background
        
        The algorithm:
        1. Start with the original complex polygon ROI (not just bounding box)
        2. For each point within the original ROI polygon, apply displacement field
        3. Create deformed ROI shape by mapping original polygon points to new locations
        4. Display displacement field on deformed image within the deformed polygon ROI
        
        Args:
            displacement: displacement field array
            current_frame: current frame index
            alpha: transparency value
            current_colormap: colormap name
            vmin_u, vmax_u: U component range
            vmin_v, vmax_v: V component range
            
        Returns:
            matplotlib figure
        """
        try:
            # Load current deformed image
            if hasattr(self, 'image_files') and len(self.image_files) > current_frame + 1:
                deformed_img_path = os.path.join(self.input_path.get(), self.image_files[current_frame + 1])
                deformed_image = hf.load_and_convert_image(deformed_img_path)
                if deformed_image is None:
                    # Fallback to reference image if loading fails
                    deformed_image = self.current_image
            else:
                # Fallback to reference image
                deformed_image = self.current_image
                
            # Get ROI bounding box coordinates (for displacement field indexing)
            xmin, ymin, xmax, ymax = self.roi_rect
            
            # Get displacement components
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            
            # Get the original ROI polygon mask within the bounding box
            roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None
            
            if roi_mask_crop is None:
                # Fallback to reference visualization if no ROI mask
                return self.create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)
            
            # Create coordinate grids for the ROI bounding box area
            roi_h, roi_w = u.shape
            y_coords, x_coords = np.mgrid[0:roi_h, 0:roi_w]
            
            # Transform coordinates to global image coordinates
            x_global = x_coords + xmin
            y_global = y_coords + ymin
            
            # Apply displacement field to get deformed coordinates
            x_deformed = x_global + u
            y_deformed = y_global + v
            
            # Initialize full-size arrays for the entire image
            h, w = deformed_image.shape[:2]
            u_full = np.full((h, w), np.nan)
            v_full = np.full((h, w), np.nan)
            deformed_mask = np.zeros((h, w), dtype=bool)
            
            # Only process points that are within the original polygon ROI
            valid_roi_points = 0
            for i in range(roi_h):
                for j in range(roi_w):
                    # Check if this point is within the original polygon ROI
                    if not roi_mask_crop[i, j]:
                        continue
                        
                    # Skip invalid displacements
                    if np.isnan(u[i, j]) or np.isnan(v[i, j]):
                        continue
                    
                    # Get deformed coordinates (rounded to nearest pixel)
                    x_def = int(np.round(x_deformed[i, j]))
                    y_def = int(np.round(y_deformed[i, j]))
                    
                    # Check if deformed coordinates are within image bounds
                    if 0 <= x_def < w and 0 <= y_def < h:
                        u_full[y_def, x_def] = u[i, j]
                        v_full[y_def, x_def] = v[i, j]
                        deformed_mask[y_def, x_def] = True
                        valid_roi_points += 1
            
            print(f"Processed {valid_roi_points} points within original polygon ROI")
            
            # If we have too few valid points, try interpolation approach
            if valid_roi_points < 10:
                return self._create_deformed_visualization_with_interpolation(
                    displacement, deformed_image, alpha, current_colormap, 
                    vmin_u, vmax_u, vmin_v, vmax_v)
            
            # Apply smoothing if requested to fill gaps and smooth the field
            if self.use_smooth_interpolation.get():
                # Create a denser field using interpolation
                u_full, v_full, deformed_mask = self._interpolate_sparse_displacement_field(
                    u_full, v_full, deformed_mask, h, w)
            
            # Create figure with two subplots
            fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Display U component on deformed image
            ax_u.imshow(deformed_image, cmap='gray')
            
            # Create masked array for U component
            u_masked = np.ma.array(u_full, mask=~deformed_mask)
            
            im_u = ax_u.imshow(u_masked, cmap=current_colormap, 
                             alpha=alpha, vmin=vmin_u, vmax=vmax_u)
            ax_u.set_title("U Component on Deformed Image (Polygon ROI)")
            fig.colorbar(im_u, ax=ax_u)
            
            # Display V component on deformed image
            ax_v.imshow(deformed_image, cmap='gray')
            
            # Create masked array for V component
            v_masked = np.ma.array(v_full, mask=~deformed_mask)
            
            im_v = ax_v.imshow(v_masked, cmap=current_colormap, 
                             alpha=alpha, vmin=vmin_v, vmax=vmax_v)
            ax_v.set_title("V Component on Deformed Image (Polygon ROI)")
            fig.colorbar(im_v, ax=ax_v)
            
            # Remove axes and set aspect ratio
            ax_u.set_axis_off()
            ax_v.set_axis_off()
            ax_u.set_aspect('equal')
            ax_v.set_aspect('equal')
            
            return fig
            
        except Exception as e:
            print(f"Error in deformed visualization: {str(e)}, falling back to reference mode")
            import traceback
            traceback.print_exc()
            # Fallback to reference image visualization
            return self.create_reference_displacement_visualization(displacement, self.current_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)
    
    def _interpolate_sparse_displacement_field(self, u_full, v_full, mask, h, w):
        """
        Interpolate sparse displacement field to create a denser representation
        
        Args:
            u_full: sparse U displacement field
            v_full: sparse V displacement field  
            mask: mask indicating valid displacement locations
            h, w: image dimensions
            
        Returns:
            tuple: (interpolated_u, interpolated_v, new_mask)
        """
        try:
            # Get coordinates of valid points
            valid_coords = np.column_stack(np.where(mask))
            if len(valid_coords) < 4:
                return u_full, v_full, mask
                
            # Get displacement values at valid points
            u_values = u_full[mask]
            v_values = v_full[mask]
            
            # Create coordinate grid for interpolation
            Y_grid, X_grid = np.mgrid[0:h, 0:w]
            
            # Interpolate U and V components
            u_interp = griddata(valid_coords[:, [1, 0]], u_values, (X_grid, Y_grid), 
                              method='linear', fill_value=np.nan)
            v_interp = griddata(valid_coords[:, [1, 0]], v_values, (X_grid, Y_grid), 
                              method='linear', fill_value=np.nan)
            
            # Apply Gaussian smoothing to reduce artifacts
            u_temp = np.nan_to_num(u_interp, nan=0.0)
            v_temp = np.nan_to_num(v_interp, nan=0.0)
            
            u_smoothed = gaussian_filter(u_temp, sigma=2.0)
            v_smoothed = gaussian_filter(v_temp, sigma=2.0)
            
            # Create new mask (only where interpolation was successful)
            new_mask = ~np.isnan(u_interp) & ~np.isnan(v_interp)
            
            # Restore original values at known points
            u_smoothed[mask] = u_full[mask]
            v_smoothed[mask] = v_full[mask]
            
            # Combine masks
            final_mask = new_mask | mask
            
            return u_smoothed, v_smoothed, final_mask
            
        except Exception as e:
            print(f"Interpolation failed: {e}")
            return u_full, v_full, mask
    
    def _create_deformed_visualization_with_interpolation(self, displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
        """
        Fallback method using interpolation when direct mapping produces too few points
        This method also respects the complex polygon ROI
        """
        try:
            # Get ROI coordinates
            xmin, ymin, xmax, ymax = self.roi_rect
            
            # Get displacement components
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            
            # Get the original ROI polygon mask within the bounding box
            roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None
            
            # Create coordinate grids for the ROI area
            roi_h, roi_w = u.shape
            y_coords, x_coords = np.mgrid[0:roi_h, 0:roi_w]
            
            # Transform to global coordinates and apply displacement
            x_global = x_coords + xmin
            y_global = y_coords + ymin
            x_deformed = x_global + u
            y_deformed = y_global + v
            
            # Create full image grid
            h, w = deformed_image.shape[:2]
            Y_full, X_full = np.mgrid[0:h, 0:w]
            
            # Only use points within the original polygon ROI for interpolation
            if roi_mask_crop is not None:
                # Combine ROI mask with validity mask
                valid_mask = (~np.isnan(u) & ~np.isnan(v) & roi_mask_crop)
            else:
                # Fallback to just validity mask if no ROI mask available
                valid_mask = ~np.isnan(u) & ~np.isnan(v)
                
            if np.sum(valid_mask) < 4:
                return self.create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)
            
            points_deformed = np.column_stack((x_deformed[valid_mask], y_deformed[valid_mask]))
            u_values = u[valid_mask]
            v_values = v[valid_mask]
            
            print(f"Interpolation fallback using {len(u_values)} points from polygon ROI")
            
            # Interpolate onto full image grid
            u_full = griddata(points_deformed, u_values, (X_full, Y_full), 
                            method='linear', fill_value=np.nan)
            v_full = griddata(points_deformed, v_values, (X_full, Y_full), 
                            method='linear', fill_value=np.nan)
            
            # Create figure
            fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Display U component
            ax_u.imshow(deformed_image, cmap='gray')
            mask_u = ~np.isnan(u_full)
            u_masked = np.ma.array(u_full, mask=~mask_u)
            
            im_u = ax_u.imshow(u_masked, cmap=current_colormap, 
                             alpha=alpha, vmin=vmin_u, vmax=vmax_u)
            ax_u.set_title("U Component on Deformed Image (Polygon ROI, Interpolated)")
            fig.colorbar(im_u, ax=ax_u)
            
            # Display V component
            ax_v.imshow(deformed_image, cmap='gray')
            mask_v = ~np.isnan(v_full)
            v_masked = np.ma.array(v_full, mask=~mask_v)
            
            im_v = ax_v.imshow(v_masked, cmap=current_colormap, 
                             alpha=alpha, vmin=vmin_v, vmax=vmax_v)
            ax_v.set_title("V Component on Deformed Image (Polygon ROI, Interpolated)")
            fig.colorbar(im_v, ax=ax_v)
            
            # Remove axes and set aspect ratio
            ax_u.set_axis_off()
            ax_v.set_axis_off()
            ax_u.set_aspect('equal')
            ax_v.set_aspect('equal')
            
            return fig
            
        except Exception as e:
            print(f"Interpolation fallback failed: {e}")
            return self.create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)

    def create_reference_displacement_visualization(self, displacement, background_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
        """
        Create displacement field visualization on reference image background (original method)
        
        Args:
            displacement: displacement field array
            background_image: background image to display
            alpha: transparency value
            current_colormap: colormap name
            vmin_u, vmax_u: U component range
            vmin_v, vmax_v: V component range
            
        Returns:
            matplotlib figure
        """
        # Get ROI coordinates
        xmin, ymin, xmax, ymax = self.roi_rect
        
        # Create figure with two subplots
        fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get displacement components
        u = displacement[:, :, 0]
        v = displacement[:, :, 1]
        
        # Create full-size displacement fields initialized with NaN
        h, w = background_image.shape[:2]
        u_full = np.full((h, w), np.nan)
        v_full = np.full((h, w), np.nan)
        
        # Place displacement fields in their correct position
        u_full[ymin:ymax, xmin:xmax] = u
        v_full[ymin:ymax, xmin:xmax] = v
        
        # Display U component
        ax_u.imshow(background_image, cmap='gray')
        mask_u = ~np.isnan(u_full)
        u_masked = np.ma.array(u_full, mask=~mask_u)
        
        im_u = ax_u.imshow(u_masked, cmap=current_colormap, 
                         alpha=alpha * mask_u, vmin=vmin_u, vmax=vmax_u)
        ax_u.set_title("U Component on Reference Image")
        fig.colorbar(im_u, ax=ax_u)
        
        # Display V component
        ax_v.imshow(background_image, cmap='gray')
        mask_v = ~np.isnan(v_full)
        v_masked = np.ma.array(v_full, mask=~mask_v)
        
        im_v = ax_v.imshow(v_masked, cmap=current_colormap, 
                         alpha=alpha * mask_v, vmin=vmin_v, vmax=vmax_v)
        ax_v.set_title("V Component on Reference Image")
        fig.colorbar(im_v, ax=ax_v)
        
        # Remove axes and set aspect ratio
        ax_u.set_axis_off()
        ax_v.set_axis_off()
        ax_u.set_aspect('equal')
        ax_v.set_aspect('equal')
        
        return fig

    def fig2img(self, fig):
        """Convert matplotlib image to PhotoImage"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        img = Image.open(buf)
        # Set uniform display size
        display_size = (400, 400)  # Keep consistent with original image preview
        img = img.resize(display_size, Image.LANCZOS)
        
        return ImageTk.PhotoImage(img)
    
    def run(self):
        """Run processing program"""
        if not self.validate_inputs():
            return
        
        # Create parameter object
        class Args:
            pass
        
        args = Args()
        args.img_dir = self.input_path.get()
        args.project_root = self.output_path.get()
        args.mode = self.mode.get()
        args.scale_factor = 1.0  # Fixed at 1.0, no longer use variable scaling
        args.crop_size = (int(self.crop_size_h.get()), int(self.crop_size_w.get()))
        args.shift = int(self.shift_size.get())
        args.max_displacement = int(self.max_displacement.get())
        
        # Add smoothing parameters
        args.use_smooth = self.use_smooth.get()
        args.sigma = float(self.sigma.get())
        
        # Load model
        model_args = hf.Args()
        args.model = hf.load_model("models/raft-dic_v1.pth", args=model_args)
        args.device = 'cuda'
        
        # Ensure output directory exists
        os.makedirs(args.project_root, exist_ok=True)
        
        try:
            # Disable interface
            self.root.config(cursor="watch")
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        widget.configure(state='disabled')
            
            # Reset progress bar
            self.progress['value'] = 0
            
            # Run processing
            self.process_images(args)
            
            # Update slider and frame number display
            total_frames = len(self.displacement_results)
            self.frame_slider.configure(to=total_frames)
            self.frame_slider.set(1)
            self.total_frames_label.configure(text=f"/{total_frames}")
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, "1")
            self.update_displacement_preview()
            
            messagebox.showinfo("Success", "Processing completed!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            print(f"Error details: {str(e)}")  # Add detailed error output
        finally:
            # Restore interface
            self.root.config(cursor="")
            for child in self.root.winfo_children():
                for widget in child.winfo_children():
                    if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                        widget.configure(state='normal')
            self.progress['value'] = 0

    def start_cutting_roi(self):
        """Start ROI cutting mode"""
        if self.roi_mask is None:
            messagebox.showerror("Error", "Please draw initial ROI first")
            return
        
        self.is_cutting_mode = True
        self.drawing_roi = True
        self.roi_points = []
        
        # Bind mouse events
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)
        
        # Keep existing ROI display
        preview = self.current_image.copy()
        overlay = np.zeros_like(preview)
        overlay[self.roi_mask] = [0, 255, 0]  # Green indicates selected area
        preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        
        # Update display
        self.update_roi_label(preview)
        
        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def start_roi_drawing(self):
        """Start ROI drawing"""
        if self.current_image is None:
            messagebox.showerror("Error", "Please select input images first")
            return
        
        self.drawing_roi = True
        self.roi_points = []
        
        # If in cutting mode, keep existing ROI display
        if self.is_cutting_mode and self.roi_mask is not None:
            preview = self.current_image.copy()
            overlay = np.zeros_like(preview)
            overlay[self.roi_mask] = [0, 255, 0]  # Green indicates selected area
            preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
            self.update_roi_label(preview)
        else:
            # Normal mode, display original image
            self.update_roi_label(self.current_image)
        
        # Bind mouse events
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)
        
        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def add_roi_point(self, event):
        """Add ROI point"""
        if not self.drawing_roi:
            return
        
        # Get actual coordinates on canvas (considering scroll position)
        canvas_x = self.roi_canvas.canvasx(event.x)
        canvas_y = self.roi_canvas.canvasy(event.y)
        
        # Convert canvas coordinates to image coordinates (considering scaling)
        x = canvas_x / self.zoom_factor
        y = canvas_y / self.zoom_factor
        
        # Add point to list
        self.roi_points.append((x, y))
        
        # Draw point on canvas
        self.roi_canvas.create_oval(
            canvas_x-2, canvas_y-2,
            canvas_x+2, canvas_y+2,
            fill='red'
        )
        
        # If not first point, draw connecting line
        if len(self.roi_points) > 1:
            prev_x = self.roi_points[-2][0] * self.zoom_factor
            prev_y = self.roi_points[-2][1] * self.zoom_factor
            self.roi_canvas.create_line(
                prev_x, prev_y,
                canvas_x, canvas_y,
                fill='green', width=2
            )

    def update_roi_preview(self, event):
        """Update ROI preview"""
        if not self.drawing_roi or len(self.roi_points) == 0:
            return
        
        # Get coordinates of last point
        last_x = int(self.roi_points[-1][0] * self.roi_scale_factor)
        last_y = int(self.roi_points[-1][1] * self.roi_scale_factor)
        
        # Delete previous preview line
        self.roi_canvas.delete('preview_line')
        
        # Draw new preview line
        self.roi_canvas.create_line(last_x, last_y, event.x, event.y,
                                  fill='yellow', width=2, tags='preview_line')

    def finish_roi_drawing(self, event):
        """Finish ROI drawing"""
        if len(self.roi_points) < 3:
            messagebox.showerror("Error", "Please select at least 3 points")
            return
        
        self.drawing_roi = False
        
        # Draw closed line segment
        first_x = int(self.roi_points[0][0] * self.roi_scale_factor)
        first_y = int(self.roi_points[0][1] * self.roi_scale_factor)
        last_x = int(self.roi_points[-1][0] * self.roi_scale_factor)
        last_y = int(self.roi_points[-1][1] * self.roi_scale_factor)
        self.roi_canvas.create_line(last_x, last_y, first_x, first_y, 
                                  fill='green', width=2)
        
        # Unbind mouse events
        self.roi_canvas.unbind('<Button-1>')
        self.roi_canvas.unbind('<Motion>')
        self.roi_canvas.unbind('<Double-Button-1>')
        
        # Create or update ROI mask
        if self.is_cutting_mode and self.roi_mask is not None:
            # In cutting mode, subtract new drawn area from existing ROI
            new_mask = np.zeros_like(self.roi_mask, dtype=bool)
            points = np.array(self.roi_points, dtype=np.int32)
            cv2.fillPoly(new_mask.view(np.uint8), [points], 1)
            self.roi_mask = self.roi_mask & ~new_mask
        else:
            # Normal mode create new ROI
            self.create_roi_mask()
        
        # Reset cutting mode
        self.is_cutting_mode = False
        
        # Update display, including semi-transparent mask
        self.update_roi_display()
        
        # Show confirm button and enable
        self.confirm_roi_btn.grid()
        self.confirm_roi_btn.configure(state='normal')

    def create_roi_mask(self):
        """Create ROI mask"""
        if not self.roi_points or len(self.roi_points) < 3:
            return
        
        # Get image dimensions
        h, w = self.current_image.shape[:2]
        
        # Create mask
        if self.roi_mask is None:
            self.roi_mask = np.zeros((h, w), dtype=bool)
        
        # Convert point coordinates to integer array
        points = np.array(self.roi_points, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(self.roi_mask.view(np.uint8), [points], 1)

    def extract_roi_rectangle(self):
        """Extract smallest rectangle containing ROI"""
        if self.roi_mask is None:
            return
        
        # Find non-zero point coordinates
        rows, cols = np.nonzero(self.roi_mask)
        if len(rows) == 0 or len(cols) == 0:
            return
        
        # Get boundaries
        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()
        
        # Store rectangle coordinates
        self.roi_rect = (xmin, ymin, xmax+1, ymax+1)

    def update_roi_display(self):
        """Update ROI display, including semi-transparent mask"""
        if self.current_image is None or self.roi_mask is None:
            return
        
        # Create preview image with semi-transparent mask
        preview = self.current_image.copy()
        overlay = np.zeros_like(preview)
        overlay[self.roi_mask] = [0, 255, 0]  # Green indicates selected area
        preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        
        # Update ROI preview
        self.update_roi_label(preview)

    def clear_roi(self):
        """Clear ROI selection"""
        self.roi_points = []
        self.roi_mask = None
        self.roi_rect = None
        self.is_cutting_mode = False
        
        # Reset preview
        if self.current_image is not None:
            self.update_roi_label(self.current_image)
        
        # Reset main preview area
        self.update_preview()
        
        # Hide confirm button
        self.confirm_roi_btn.grid_remove()
        
        # Stop playback
        self.is_playing = False
        if hasattr(self, 'play_button'):
            self.play_button.configure(text=self.play_icon)
        if self.play_after_id:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def confirm_roi(self):
        """Confirm ROI selection and update preview"""
        if self.roi_mask is None:
            messagebox.showerror("Error", "Please draw ROI first")
            return
        
        # Extract largest rectangle containing ROI
        self.extract_roi_rectangle()
        
        # Get ROI rectangle area
        if self.roi_rect:
            xmin, ymin, xmax, ymax = self.roi_rect
            
            # Update crop area preview
            self.update_preview()
            
            # Update information display
            roi_info = f"\nROI size: {xmax-xmin}x{ymax-ymin}"
            self.image_info.config(text=self.image_info.cget("text") + roi_info)

    def update_roi_label(self, image):
        """Update ROI preview label"""
        if image is None:
            return
        
        # Get original image dimensions
        h, w = image.shape[:2]
        
        # Calculate display size
        display_w = int(w * self.zoom_factor)
        display_h = int(h * self.zoom_factor)
        
        # Adjust image size
        resized = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB format
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        elif resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Create PhotoImage
        image_pil = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(image_pil)
        
        # Clear canvas and display new image
        self.roi_canvas.delete("all")
        self.roi_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        
        # Redraw existing ROI points and lines
        if self.roi_points:
            for i, point in enumerate(self.roi_points):
                x, y = point
                canvas_x = x * self.zoom_factor
                canvas_y = y * self.zoom_factor
                
                # Draw point
                self.roi_canvas.create_oval(
                    canvas_x-2, canvas_y-2,
                    canvas_x+2, canvas_y+2,
                    fill='red'
                )
                
                # Draw line
                if i > 0:
                    prev_x = self.roi_points[i-1][0] * self.zoom_factor
                    prev_y = self.roi_points[i-1][1] * self.zoom_factor
                    self.roi_canvas.create_line(
                        prev_x, prev_y,
                        canvas_x, canvas_y,
                        fill='green', width=2
                    )
        
        # Update scroll area
        self.roi_canvas.configure(scrollregion=(0, 0, display_w, display_h))

    def on_mousewheel(self, event):
        """Handle mouse wheel events"""
        if self.current_image is None:
            return
            
        # Get current mouse position (relative to canvas)
        x = self.roi_canvas.canvasx(event.x)
        y = self.roi_canvas.canvasy(event.y)
        
        # Windows platform
        if event.delta:
            if event.delta > 0:
                self.zoom(1.1, x, y)
            else:
                self.zoom(0.9, x, y)
        # Linux/Mac platform
        else:
            if event.num == 4:
                self.zoom(1.1, x, y)
            elif event.num == 5:
                self.zoom(0.9, x, y)

    def zoom(self, factor, x=None, y=None):
        """Scale image"""
        if self.current_image is None:
            return
            
        # Update scaling factor
        old_zoom = self.zoom_factor
        self.zoom_factor *= factor
        
        # Limit scaling range
        min_zoom = 0.1
        max_zoom = 5.0
        self.zoom_factor = max(min_zoom, min(max_zoom, self.zoom_factor))
        
        # If scaling factor did not change, return immediately
        if self.zoom_factor == old_zoom:
            return
        
        # Save current scroll position
        old_x = self.roi_canvas.canvasx(0)
        old_y = self.roi_canvas.canvasy(0)
        
        # Update display
        self.update_roi_label(self.current_image)
        
        # If mouse position provided, adjust scroll position to keep point under mouse
        if x is not None and y is not None:
            # Calculate the relative position change
            canvas_width = self.roi_canvas.winfo_width()
            canvas_height = self.roi_canvas.winfo_height()
            
            if canvas_width > 0 and canvas_height > 0:
                # Calculate new scroll position to keep the point under the cursor
                new_x = (x * self.zoom_factor - canvas_width / 2) / (self.roi_canvas.winfo_reqwidth() * self.zoom_factor)
                new_y = (y * self.zoom_factor - canvas_height / 2) / (self.roi_canvas.winfo_reqheight() * self.zoom_factor)
                
                # Clamp values between 0 and 1
                new_x = max(0, min(1, new_x))
                new_y = max(0, min(1, new_y))
                
                # Set new scroll position
                self.roi_canvas.xview_moveto(new_x)
                self.roi_canvas.yview_moveto(new_y)

    def reset_zoom(self):
        """Reset scaling"""
        if self.current_image is None:
            return
            
        self.zoom_factor = 1.0
        self.update_roi_label(self.current_image)
        
        # Reset scrollbar position
        self.roi_canvas.xview_moveto(0)
        self.roi_canvas.yview_moveto(0)

    def jump_to_frame(self):
        """Jump to specified frame"""
        try:
            frame_num = int(self.frame_entry.get())
            total_frames = len(self.displacement_results)
            if 1 <= frame_num <= total_frames:
                self.frame_slider.set(frame_num)
                self.update_displacement_preview()
            else:
                messagebox.showerror("Error", f"Frame number must be between 1 and {total_frames}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame number")

    def toggle_play(self):
        """Toggle play/pause state"""
        if not self.displacement_results:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            # Update button display to pause icon
            self.play_button.configure(text=self.pause_icon)
            # Start playback
            self.play_next_frame()
        else:
            # Update button display to play icon
            self.play_button.configure(text=self.play_icon)
            # Stop playback
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None

    def play_next_frame(self):
        """Play next frame"""
        if not self.is_playing:
            return
        
        current_frame = int(self.frame_slider.get())
        total_frames = len(self.displacement_results)
        
        # Calculate next frame
        next_frame = current_frame + 1
        if next_frame > total_frames:
            next_frame = 1  # Loop playback
        
        # Update slider position
        self.frame_slider.set(next_frame)
        
        # Schedule next frame playback
        self.play_after_id = self.root.after(self.play_interval, self.play_next_frame)

    def change_play_speed(self, *args):
        """Change playback speed"""
        speed_map = {
            "0.25x": 400,
            "0.5x": 200,
            "1x": 100,
            "2x": 50,
            "4x": 25
        }
        self.play_interval = speed_map[self.speed_var.get()]
        
        # If playing, restart playback to apply new speed
        if self.is_playing:
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
            self.play_next_frame()

    def on_param_change(self, event=None):
        """Update preview when parameter input is complete"""
        self.update_preview()

    def previous_frame(self):
        """Go to previous frame"""
        if not self.displacement_results:
            return
        current_frame = int(self.frame_slider.get())
        if current_frame > 1:
            self.frame_slider.set(current_frame - 1)
            self.update_displacement_preview()
    
    def next_frame(self):
        """Go to next frame"""
        if not self.displacement_results:
            return
        current_frame = int(self.frame_slider.get())
        if current_frame < len(self.displacement_results):
            self.frame_slider.set(current_frame + 1)
            self.update_displacement_preview()

    def create_help_button(self, parent, tooltip_key):
        """Create a help button with tooltip"""
        help_btn = ttk.Label(parent, text="?", cursor="hand2")
        help_btn.configure(background="lightgray", width=2)
        Tooltip(help_btn, self.tooltips[tooltip_key])
        return help_btn

    def update_crop_state(self, *args):
        """Update the state of crop-related widgets based on checkbox"""
        state = "normal" if self.use_crop.get() else "disabled"
        self.crop_w_entry.configure(state=state)
        self.crop_h_entry.configure(state=state)
        self.shift_entry.configure(state=state)

    def update_crop_size(self):
        """Update crop size entries with image dimensions"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.crop_size_w.set(str(w))
            self.crop_size_h.set(str(h))


def main():
    """Main entry point for the RAFT-DIC GUI application."""
    root = tk.Tk()
    app = RAFTDICGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main() 