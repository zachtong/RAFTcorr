import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, Rbf

import raft_dic_gui.processing as proc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PreviewPanel(ttk.Frame):
    def __init__(self, parent, control_panel=None, root=None, callbacks=None, config=None):
        super().__init__(parent)
        self.config = config
        self.control_panel = control_panel
        self.root = root
        self.callbacks = callbacks or {}
        
        # Internal layout: Notebook (Tabs) for ROI and Vis
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.roi_pane = ttk.Frame(self.notebook)
        self.vis_pane = ttk.Frame(self.notebook)
        
        self.notebook.add(self.roi_pane, text="ROI Selection")
        self.notebook.add(self.vis_pane, text="Displacement Result")
        
        # Post-Processing Tab
        self.post_pane = ttk.Frame(self.notebook)
        self.notebook.add(self.post_pane, text="Post-Processing")
        self.notebook.tab(2, state="disabled") # Enabled only after results exist
        
        # State variables
        self.roi_points = []
        self.drawing_roi = False
        self.roi_mask = None
        self.roi_rect = None
        self.roi_scale_factor = 1.0
        self.display_size = (400, 400)
        self._roi_cache = None
        self.is_cutting_mode = False
        self.current_image = None
        self.current_image_path = None
        self.displacement_results = []
        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.current_photo = None
        self.is_playing = False
        self.play_after_id = None
        self.play_interval = 100
        self.image_files = [] # List of image filenames for playback
        self.canvas = None
        self.toolbar = None
        self.fig = None
        self.ax_u = None
        self.ax_v = None
        self.cb_u = None
        self.cb_u = None
        self.cb_v = None
        self.post_cid = None # For Matplotlib event connection
        self.line_preview = None
        
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        self.create_widgets()

    def set_control_panel(self, control_panel):
        """Set the control panel reference."""
        self.control_panel = control_panel

    def _on_tab_changed(self, event):
        if self.callbacks.get('on_tab_changed'):
            try:
                self.callbacks['on_tab_changed'](self.notebook.index("current"))
            except Exception:
                pass

    def create_widgets(self):
        # ROI Selection Panel
        self._create_roi_panel()
        # Displacement Overlay Panel
        self._create_vis_panel()
        # Post-Processing Panel
        self._create_post_panel()

    def _create_post_panel(self):
        post_pane = self.post_pane
        
        # Split Post-Processing Area: Top (Canvas) / Bottom (Graph)
        self.post_paned = tk.PanedWindow(post_pane, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=4)
        self.post_paned.pack(fill="both", expand=True)
        
        # Top: Matplotlib Canvas for Strain/Extensometer Map
        self.post_canvas_frame = ttk.Frame(self.post_paned)
        self.post_paned.add(self.post_canvas_frame, minsize=300, stretch="always")
        
        self.post_fig, self.post_ax = plt.subplots(figsize=(5, 4))
        self.post_canvas = FigureCanvasTkAgg(self.post_fig, master=self.post_canvas_frame)
        self.post_canvas.draw()
        self.post_canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        self.post_toolbar = NavigationToolbar2Tk(self.post_canvas, self.post_canvas_frame)
        self.post_toolbar.update()
        self.post_toolbar.pack(side="top", fill="x")
        
        # Add Playback Controls for Post-Processing
        self._create_playback_controls(self.post_canvas_frame, 'post', use_pack=True)
        
        # Bottom: Graph Area for Time-Series
        self.post_graph_frame = ttk.Frame(self.post_paned)
        self.post_paned.add(self.post_graph_frame, minsize=150, stretch="always")
        
        self.post_graph_fig, self.post_graph_ax = plt.subplots(figsize=(5, 2))
        self.post_graph_canvas = FigureCanvasTkAgg(self.post_graph_fig, master=self.post_graph_frame)
        self.post_graph_canvas.draw()
        self.post_graph_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _create_roi_panel(self):
        roi_pane = self.roi_pane
        # Ensure panes expand
        try:
            roi_pane.grid_rowconfigure(0, weight=1)
            roi_pane.grid_columnconfigure(0, weight=1)
        except Exception:
            pass
        
        roi_frame = ttk.LabelFrame(roi_pane, text="ROI Selection", padding="5")
        roi_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        roi_frame.grid_rowconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(0, weight=1)
        
        self.roi_canvas = tk.Canvas(roi_frame, width=420, height=420, background='#0f0f14', highlightthickness=1, highlightbackground='#2a2a34')
        self.roi_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        x_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.HORIZONTAL, command=self.roi_canvas.xview)
        y_scrollbar = ttk.Scrollbar(roi_frame, orient=tk.VERTICAL, command=self.roi_canvas.yview)
        x_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        y_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.roi_canvas.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        
        # Bind events
        self.roi_canvas.bind('<MouseWheel>', self.on_mousewheel)
        self.roi_canvas.bind('<Button-4>', self.on_mousewheel)
        self.roi_canvas.bind('<Button-5>', self.on_mousewheel)
        self._bind_polygon_events()
        
        # Pan bindings (Middle mouse or Shift+Drag)
        self.roi_canvas.bind('<Button-2>', self.start_pan)
        self.roi_canvas.bind('<B2-Motion>', self.pan)
        self.roi_canvas.bind('<ButtonRelease-2>', self.end_pan)
        
        # Zoom toolbar
        zoom_frame = ttk.Frame(roi_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.zoom(1.2)).grid(row=0, column=0, padx=2)
        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.zoom(0.8)).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="100%", width=5, command=self.reset_zoom).grid(row=0, column=2, padx=2)
        
        # ROI Tools
        self.roi_controls = ttk.LabelFrame(roi_frame, text="ROI Tools", padding=5)
        self.roi_controls.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        for c in range(6):
            self.roi_controls.grid_columnconfigure(c, weight=1)

        ttk.Button(self.roi_controls, text="Draw Poly", width=9,
                   command=lambda: self.start_polygon_tool(cut=False)).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Cut Poly", width=9,
                   command=lambda: self.start_polygon_tool(cut=True)).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Draw Rect", width=9,
                   command=lambda: self.start_shape_tool('rect', cut=False)).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Cut Rect", width=9,
                   command=lambda: self.start_shape_tool('rect', cut=True)).grid(row=0, column=3, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Draw Circle", width=9,
                   command=lambda: self.start_shape_tool('circle', cut=False)).grid(row=0, column=4, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Cut Circle", width=9,
                   command=lambda: self.start_shape_tool('circle', cut=True)).grid(row=0, column=5, padx=2, pady=2)

        ttk.Button(self.roi_controls, text="Import Mask...", width=12,
                   command=self.import_roi_mask).grid(row=1, column=0, padx=2, pady=2, columnspan=2, sticky='ew')
        ttk.Button(self.roi_controls, text="Invert ROI", width=9,
                   command=self.invert_roi).grid(row=1, column=2, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Clean Small", width=9,
                   command=self.clean_small_regions).grid(row=1, column=3, padx=2, pady=2)
        ttk.Button(self.roi_controls, text="Clear ROI", width=10,
                   command=self.clear_roi).grid(row=1, column=4, padx=2, pady=2)
        self.confirm_roi_btn = ttk.Button(self.roi_controls, text="Confirm ROI", width=12,
                                         command=self.confirm_roi)
        self.confirm_roi_btn.grid(row=1, column=5, padx=2, pady=2)
        # self.confirm_roi_btn.grid_remove() # Keep persistent as requested
        
        # Info labels
        self.info_font = ctk.CTkFont(family="Segoe UI", size=11)
        self.image_info = ttk.Label(roi_frame, text="", font=self.info_font)
        self.image_info.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        self.metrics_font = ("Consolas", 11)
        self.roi_metrics_text = tk.Text(roi_frame, height=6, wrap=tk.WORD, font=self.metrics_font)
        self.roi_metrics_text.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0,5))
        self.roi_metrics_text.configure(state='disabled')

    def _create_vis_panel(self):
        vis_pane = self.vis_pane
        try:
            vis_pane.grid_rowconfigure(0, weight=1)
            vis_pane.grid_columnconfigure(0, weight=1)
        except Exception:
            pass
            
        disp_frame = ttk.LabelFrame(vis_pane, text="Displacement Field Overlay", padding="5")
        disp_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        disp_frame.grid_rowconfigure(0, weight=1)
        disp_frame.grid_columnconfigure(0, weight=1)
        self.disp_frame = disp_frame
        
        self.disp_frame = disp_frame
        
        # Canvas Frame (replaces label)
        self.canvas_frame = ttk.Frame(disp_frame)
        self.canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize Figure and Canvas once
        self.fig, (self.ax_u, self.ax_v) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2)
        self.ax_u.set_axis_off()
        self.ax_v.set_axis_off()
        
        self.ax_u.set_title("U Component")
        self.ax_v.set_title("V Component")
        
        # Create dividers for stable colorbar layout
        self.div_u = make_axes_locatable(self.ax_u)
        self.div_v = make_axes_locatable(self.ax_v)
        self.cax_u = None
        self.cax_v = None
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create Playback Controls
        self._create_playback_controls(disp_frame, 'disp')
        
        # Refresh binding
        try:
            def _schedule_vis_refresh(event=None):
                try:
                    if hasattr(self, '_vis_refresh_after') and self._vis_refresh_after:
                        self.root.after_cancel(self._vis_refresh_after)
                except Exception:
                    pass
                try:
                    self._vis_refresh_after = self.root.after(50, self.update_displacement_preview)
                except Exception:
                    pass
            self.disp_frame.bind('<Configure>', _schedule_vis_refresh)
            vis_pane.bind('<Configure>', _schedule_vis_refresh)
        except Exception:
            pass

    def _create_playback_controls(self, parent, prefix, use_pack=False):
        """Create reusable playback controls."""
        control_frame = ttk.Frame(parent)
        if use_pack:
            control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        else:
            control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Playback controls
        play_control = ttk.Frame(control_frame)
        play_control.grid(row=0, column=0, padx=5)
        
        play_btn = ttk.Button(play_control, text="Play", width=5, command=self.toggle_play)
        play_btn.grid(row=0, column=0, padx=2)
        setattr(self, f'{prefix}_play_button', play_btn)
        
        frame_control = ttk.Frame(control_frame)
        frame_control.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(frame_control, text="<", width=3, command=self.previous_frame).grid(row=0, column=0, padx=2)
        ttk.Label(frame_control, text="Frame:").grid(row=0, column=1, padx=5)
        
        frame_entry = ttk.Entry(frame_control, width=5)
        frame_entry.grid(row=0, column=2, padx=2)
        setattr(self, f'{prefix}_frame_entry', frame_entry)
        
        total_lbl = ttk.Label(frame_control, text="/1")
        total_lbl.grid(row=0, column=3, padx=2)
        setattr(self, f'{prefix}_total_frames_label', total_lbl)
        
        ttk.Button(frame_control, text="Go", command=self.jump_to_frame).grid(row=0, column=4, padx=5)
        ttk.Button(frame_control, text=">", width=3, command=self.next_frame).grid(row=0, column=5, padx=2)
        
        img_name_lbl = ttk.Label(frame_control, text="")
        img_name_lbl.grid(row=0, column=6, padx=5)
        setattr(self, f'{prefix}_current_image_name', img_name_lbl)
        
        speed_frame = ttk.Frame(play_control)
        speed_frame.grid(row=0, column=1, padx=5)
        ttk.Label(speed_frame, text="Speed:").grid(row=0, column=0)
        
        if not hasattr(self, 'speed_var'):
            self.speed_var = tk.StringVar(value="1x")
            
        speed_menu = ttk.OptionMenu(speed_frame, self.speed_var, "1x", "0.25x", "0.5x", "1x", "2x", "4x", command=self.change_play_speed)
        speed_menu.grid(row=0, column=1)
        
        slider = ttk.Scale(parent, from_=1, to=1, orient=tk.HORIZONTAL, command=self._on_slider_change)
        if use_pack:
            slider.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        else:
            slider.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        setattr(self, f'{prefix}_frame_slider', slider)
        
        # Alias for backward compatibility if prefix is 'disp'
        if prefix == 'disp':
            self.play_button = play_btn
            self.frame_entry = frame_entry
            self.total_frames_label = total_lbl
            self.current_image_name = img_name_lbl
            self.frame_slider = slider

    def _on_slider_change(self, val):
        # Determine which tab is active and update accordingly
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 1: # Displacement
            self.update_displacement_preview()
        elif current_tab == 2: # Post-Processing
            if self.control_panel and hasattr(self.control_panel, 'post_processing_panel'):
                 if self.control_panel.post_processing_panel.callbacks.get('update_post_preview'):
                     self.control_panel.post_processing_panel.callbacks['update_post_preview']()

    # --- ROI Methods ---
    def set_image(self, image, image_path=None):
        self.current_image = image
        self.current_image_path = image_path
        self._roi_cache = None
        self.draw_roi()
        self.update_roi_label(image)

    def draw_roi(self):
        self.roi_canvas.delete("all")
        if self.current_image is None:
            return

        # Convert to PIL and resize
        h, w = self.current_image.shape[:2]
        new_w = int(w * self.zoom_factor)
        new_h = int(h * self.zoom_factor)
        
        pil_img = Image.fromarray(self.current_image)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(pil_img)
        
        self.roi_canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.roi_canvas.create_image(0, 0, image=self.current_photo, anchor=tk.NW)
        
        # Draw ROI points/shapes
        # ... (Simplified for brevity, would need full implementation from main_GUI.py)
        # For now, just drawing points if any
        if self.roi_points:
            scaled_points = [(x * self.zoom_factor, y * self.zoom_factor) for x, y in self.roi_points]
            if len(scaled_points) > 1:
                self.roi_canvas.create_line(scaled_points, fill='yellow', width=2)
            for x, y in scaled_points:
                self.roi_canvas.create_oval(x-3, y-3, x+3, y+3, fill='red', outline='yellow')

        # Draw mask overlay if exists
        if self.roi_mask is not None:
            # Create overlay
            mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            mask_overlay[self.roi_mask] = [255, 255, 0, 64] # Yellow transparent
            pil_mask = Image.fromarray(mask_overlay)
            pil_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)
            self.mask_photo = ImageTk.PhotoImage(pil_mask)
            self.roi_canvas.create_image(0, 0, image=self.mask_photo, anchor=tk.NW)

        self.draw_tiles_overlay()
        self.draw_rulers(new_w, new_h)

    def draw_tiles_overlay(self):
        """Draw tile grid overlay if enabled."""
        if not self.control_panel or not self.control_panel.show_tiles.get():
            return
            
        if self.current_image is None:
            return

        try:
            # Get parameters
            p_max = self.config.p_max_pixels
            if isinstance(p_max, str):
                if '*' in p_max:
                    w, h = map(int, p_max.split('*'))
                    p_max = w * h
                else:
                    p_max = int(p_max)
            
            tile_size = int(np.sqrt(p_max))
            overlap = int(self.control_panel.tile_overlap.get())
            stride = max(1, tile_size - overlap)
            
            # Determine ROI bounds
            h, w = self.current_image.shape[:2]
            x0, y0, x1, y1 = 0, 0, w, h
            if self.roi_rect:
                x0, y0, x1, y1 = self.roi_rect
                
            rw, rh = x1 - x0, y1 - y0
            if rw <= 0 or rh <= 0:
                return

            # Calculate positions
            xs = proc.calculate_window_positions(rw, tile_size, stride)
            ys = proc.calculate_window_positions(rh, tile_size, stride)
            
            # Draw tiles
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    # Tile bounds in ROI coords
                    tx0 = x0 + x
                    ty0 = y0 + y
                    tx1 = min(tx0 + tile_size, x0 + rw)
                    ty1 = min(ty0 + tile_size, y0 + rh)
                    
                    # Convert to canvas coords
                    cx0 = tx0 * self.zoom_factor
                    cy0 = ty0 * self.zoom_factor
                    cx1 = tx1 * self.zoom_factor
                    cy1 = ty1 * self.zoom_factor
                    
                    # Draw tile (Blue)
                    self.roi_canvas.create_rectangle(cx0, cy0, cx1, cy1, 
                                                   outline='#00BFFF', width=1, tags='tile_overlay')
                    
                    # Draw valid core (Orange) - approximate visualization
                    # Core is roughly the center area. 
                    # For visualization, just showing the tile boundary is often enough to see overlap.
                    # But let's add a small inner box to indicate the "core" if requested?
                    # User asked for "how they overlap", so the dense grid of blue boxes is perfect.
                    # Let's add a text label for tile index to help order
                    if len(xs) * len(ys) < 100: # Only if not too many tiles
                         self.roi_canvas.create_text((cx0+cx1)/2, (cy0+cy1)/2, 
                                                   text=f"{i},{j}", fill='#00BFFF', font=("Arial", 8), tags='tile_overlay')

        except Exception as e:
            print(f"Error drawing tiles: {e}")

    def draw_rulers(self, w, h):
        """Draw simple rulers/ticks around the image."""
        # Draw border
        self.roi_canvas.create_rectangle(0, 0, w, h, outline='#444444', width=1)
        
        # Draw ticks every 50 pixels (scaled)
        step = 50 * self.zoom_factor
        if step < 20: step = 100 * self.zoom_factor
        
        # X ticks
        for x in np.arange(0, w, step):
            self.roi_canvas.create_line(x, 0, x, 5, fill='#666666')
            self.roi_canvas.create_line(x, h, x, h-5, fill='#666666')
            
        # Y ticks
        for y in np.arange(0, h, step):
            self.roi_canvas.create_line(0, y, 5, y, fill='#666666')
            self.roi_canvas.create_line(w, y, w-5, y, fill='#666666')

    def on_mousewheel(self, event):
        if event.state & 0x0004: # Control key -> Zoom
            factor = 1.1 if event.delta > 0 else 0.9
            self.zoom(factor)
        elif event.state & 0x0001: # Shift key -> Horizontal Scroll
            self.roi_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        else: # Normal -> Vertical Scroll
            self.roi_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def start_pan(self, event):
        self.roi_canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        self.roi_canvas.scan_dragto(event.x, event.y, gain=1)

    def end_pan(self, event):
        pass

    def zoom(self, factor):
        self.zoom_factor *= factor
        self.draw_roi()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.draw_roi()

    def start_shape_tool(self, shape='rect', cut=False):
        """Start ROI shape tool (rect/circle)."""
        print(f"[DEBUG] start_shape_tool called. shape={shape}, cut={cut}")
        self.drawing_roi = True
        self.roi_points = []
        self.current_shape = shape
        self.is_cutting_mode = cut
        self.shape_start = None
        
        self.roi_canvas.config(cursor='crosshair')
        self.roi_canvas.bind("<Button-1>", self._on_shape_down)
        self.roi_canvas.bind("<B1-Motion>", self._on_shape_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self._on_shape_up)
        self.confirm_roi_btn.grid()
        
    def _on_shape_down(self, event):
        x = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        self.shape_start = (x, y)
        
    def _on_shape_drag(self, event):
        if not self.shape_start: return
        x0, y0 = self.shape_start
        x1 = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y1 = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        
        self.roi_canvas.delete('preview_shape')
        
        cx0 = x0 * self.zoom_factor
        cy0 = y0 * self.zoom_factor
        cx1 = x1 * self.zoom_factor
        cy1 = y1 * self.zoom_factor
        
        if self.current_shape == 'rect':
            self.roi_canvas.create_rectangle(cx0, cy0, cx1, cy1, outline='yellow', tags='preview_shape')
        elif self.current_shape == 'circle':
            r = ((x1-x0)**2 + (y1-y0)**2)**0.5
            cr = r * self.zoom_factor
            self.roi_canvas.create_oval(cx0-cr, cy0-cr, cx0+cr, cy0+cr, outline='yellow', tags='preview_shape')
            
    def _on_shape_up(self, event):
        if not self.shape_start: return
        x0, y0 = self.shape_start
        x1 = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y1 = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        self.shape_start = None
        self.roi_canvas.delete('preview_shape')
        
        # Create mask
        h, w = self.current_image.shape[:2]
        new_mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.current_shape == 'rect':
            tx0, tx1 = int(min(x0, x1)), int(max(x0, x1))
            ty0, ty1 = int(min(y0, y1)), int(max(y0, y1))
            cv2.rectangle(new_mask, (tx0, ty0), (tx1, ty1), 1, -1)
            
        elif self.current_shape == 'circle':
            r = int(((x1-x0)**2 + (y1-y0)**2)**0.5)
            cv2.circle(new_mask, (int(x0), int(y0)), r, 1, -1)
            
        new_mask_bool = new_mask.astype(bool)
        
        if self.roi_mask is None:
            self.roi_mask = np.zeros((h, w), dtype=bool)
            
        if self.is_cutting_mode:
            self.roi_mask = self.roi_mask & ~new_mask_bool
        else:
            self.roi_mask = self.roi_mask | new_mask_bool
            
        # Update rect
        if self.roi_mask is not None:
            y_indices, x_indices = np.where(self.roi_mask)
            if len(x_indices) > 0:
                xmin, xmax = x_indices.min(), x_indices.max()
                ymin, ymax = y_indices.min(), y_indices.max()
                self.roi_rect = (xmin, ymin, xmax+1, ymax+1)
            else:
                self.roi_rect = (0, 0, w, h)
                
        self.draw_roi()
        
        # Notify controller
        if self.callbacks.get('on_roi_confirmed'):
            self.callbacks['on_roi_confirmed'](self.roi_mask, self.roi_rect)

    def start_polygon_tool(self, cut=False):
        print(f"[DEBUG] start_polygon_tool called. cut={cut}")
        self.drawing_roi = True
        self.roi_points = []
        self.is_cutting_mode = cut
        self._bind_polygon_events()
        self.roi_canvas.config(cursor='crosshair')
        self.draw_roi()
        self.confirm_roi_btn.grid()

    def _bind_polygon_events(self):
        self.roi_canvas.bind("<Button-1>", self.on_canvas_click)
        self.roi_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.roi_canvas.bind("<Motion>", self.on_mouse_move)
        # Shortcuts to close polygon
        self.roi_canvas.bind("<Button-3>", lambda e: self.confirm_roi()) # Right click
        self.roi_canvas.bind("<Double-Button-1>", lambda e: self.confirm_roi()) # Double click

    def start_point_tool(self):
        """Start the point probe tool."""
        print("[DEBUG] start_point_tool called")
        
        # Check active tab
        current_tab = self.notebook.index("current")
        if current_tab == 2: # Post-Processing
            print("[DEBUG] Starting point tool on Post-Processing Canvas")
            self.post_canvas.get_tk_widget().config(cursor='crosshair')
            if self.post_cid is None:
                self.post_cid = self.post_canvas.mpl_connect('button_press_event', self._on_post_click)
        else:
            # ROI Canvas (fallback)
            if self.current_image is None:
                tk.messagebox.showerror("Error", "Please select input images first")
                return
            self.roi_canvas.config(cursor='crosshair')
            self.roi_canvas.bind('<Button-1>', self._point_mouse_click)
            self.roi_canvas.bind('<Button-3>', self._stop_point_tool)
        
    def _stop_point_tool(self, event=None):
        # Unbind ROI canvas
        self.roi_canvas.unbind('<Button-1>')
        self.roi_canvas.unbind('<Button-3>')
        self.roi_canvas.config(cursor='')
        
        # Unbind Post Canvas
        if self.post_cid is not None:
            self.post_canvas.mpl_disconnect(self.post_cid)
            self.post_cid = None
        self.post_canvas.get_tk_widget().config(cursor='')
        
        if self.callbacks.get('on_tool_finished'):
            self.callbacks['on_tool_finished']()

    def _on_post_click(self, event):
        """Handle click on Matplotlib canvas."""
        if event.inaxes != self.post_ax:
            return
        
        # event.xdata, event.ydata are in image coordinates
        x, y = event.xdata, event.ydata
        print(f"[DEBUG] Post-Processing Click: {x}, {y}")
        
        if self.callbacks.get('on_add_point'):
            self.callbacks['on_add_point'](x, y)

    def _point_mouse_click(self, event):
        x = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        
        # Notify controller to add probe
        if self.callbacks.get('on_add_point'):
            self.callbacks['on_add_point'](x, y)

    def start_line_tool(self):
        """Start the line probe tool."""
        print("[DEBUG] start_line_tool called")
        current_tab = self.notebook.index("current")
        if current_tab == 2: # Post-Processing
            self.post_canvas.get_tk_widget().config(cursor='crosshair')
            # Disconnect point tool if active
            if self.post_cid is not None:
                self.post_canvas.mpl_disconnect(self.post_cid)
            
            # Connect line tool events
            self.line_start = None
            self.line_preview = None
            self.post_cid_press = self.post_canvas.mpl_connect('button_press_event', self._on_post_line_click)
            self.post_cid_move = self.post_canvas.mpl_connect('motion_notify_event', self._on_post_line_drag)
            self.post_cid_release = self.post_canvas.mpl_connect('button_release_event', self._on_post_line_release)
        else:
            tk.messagebox.showinfo("Info", "Line tool is only available in Post-Processing tab for now.")

    def _stop_line_tool(self):
        if hasattr(self, 'post_cid_press'):
            self.post_canvas.mpl_disconnect(self.post_cid_press)
            self.post_canvas.mpl_disconnect(self.post_cid_move)
            self.post_canvas.mpl_disconnect(self.post_cid_release)
            self.post_cid_press = None
        
        if self.line_preview:
            try:
                self.line_preview.remove()
            except:
                pass
            self.line_preview = None
        self.post_canvas.draw_idle()
        self.post_canvas.get_tk_widget().config(cursor='')
        
        if self.callbacks.get('on_tool_finished'):
            self.callbacks['on_tool_finished']()

    def _on_post_line_click(self, event):
        if event.inaxes != self.post_ax:
            return
        self.line_start = (event.xdata, event.ydata)
        
    def _on_post_line_drag(self, event):
        if self.line_start is None or event.inaxes != self.post_ax:
            return
        
        x0, y0 = self.line_start
        x1, y1 = event.xdata, event.ydata
        
        if self.line_preview:
            self.line_preview.set_data([x0, x1], [y0, y1])
        else:
            self.line_preview, = self.post_ax.plot([x0, x1], [y0, y1], 'r--', linewidth=2)
        self.post_canvas.draw_idle()
        
    def _on_post_line_release(self, event):
        if self.line_start is None or event.inaxes != self.post_ax:
            self.line_start = None
            return
            
        x0, y0 = self.line_start
        x1, y1 = event.xdata, event.ydata
        self.line_start = None
        
        # Remove preview
        if self.line_preview:
            try:
                self.line_preview.remove()
            except:
                pass
            self.line_preview = None
            
        # Add line probe
        if self.callbacks.get('on_add_line'):
            self.callbacks['on_add_line']((x0, y0), (x1, y1))

    def draw_probes(self, probes, mode='point'):
        """Draw probes on the canvas (ROI or Post)."""
        # 1. Draw on ROI Canvas (Disabled as requested)
        self.roi_canvas.delete('probe')

        # 2. Draw on Post-Processing Canvas (Matplotlib)
        # Clear previous probes from plot
        if hasattr(self, 'probe_artists'):
            for art in self.probe_artists:
                try:
                    art.remove()
                except Exception:
                    pass # Ignore if already removed or not supported
        self.probe_artists = []
        
        current_tab = self.notebook.index("current")
        
        if current_tab == 2: # Only if visible
            for p in probes:
                # Filter by mode
                if mode == 'point' and p.type != 'point':
                    continue
                if mode == 'line' and p.type != 'line':
                    continue
                if mode == 'area' and p.type != 'area':
                    continue
                    
                if p.type == 'point':
                    # Plot point
                    pt, = self.post_ax.plot(p.coords[0], p.coords[1], marker='+', color=p.color, markersize=10, markeredgewidth=2)
                    txt = self.post_ax.text(p.coords[0]+5, p.coords[1]-5, str(p.id), color=p.color, fontsize=10, fontweight='bold')
                    self.probe_artists.extend([pt, txt])
                elif p.type == 'line':
                    # Plot line
                    p1, p2 = p.coords
                    ln, = self.post_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=p.color, linewidth=2, marker='o', markersize=4)
                    # Label at midpoint
                    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
                    txt = self.post_ax.text(mx, my, str(p.id), color=p.color, fontsize=10, fontweight='bold')
                    self.probe_artists.extend([ln, txt])
                elif p.type == 'area':
                    shape = p.coords['shape']
                    data = p.coords['data']
                    
                    if shape == 'rect':
                        x0, y0, x1, y1 = data
                        w, h = abs(x1-x0), abs(y1-y0)
                        rect = plt.Rectangle((min(x0, x1), min(y0, y1)), w, h, fill=False, edgecolor=p.color, linewidth=2)
                        self.post_ax.add_patch(rect)
                        self.probe_artists.append(rect)
                        # Label
                        txt = self.post_ax.text(min(x0,x1), min(y0,y1)-5, str(p.id), color=p.color, fontsize=10, fontweight='bold')
                        self.probe_artists.append(txt)
                        
                    elif shape == 'circle':
                        cx, cy, r = data
                        circ = plt.Circle((cx, cy), r, fill=False, edgecolor=p.color, linewidth=2)
                        self.post_ax.add_patch(circ)
                        self.probe_artists.append(circ)
                        txt = self.post_ax.text(cx, cy, str(p.id), color=p.color, fontsize=10, fontweight='bold', ha='center', va='center')
                        self.probe_artists.append(txt)
                        
                    elif shape == 'poly':
                        pts = np.array(data)
                        poly = plt.Polygon(pts, fill=False, edgecolor=p.color, linewidth=2)
                        self.post_ax.add_patch(poly)
                        self.probe_artists.append(poly)
                        # Label at centroid
                        cx, cy = np.mean(pts, axis=0)
                        txt = self.post_ax.text(cx, cy, str(p.id), color=p.color, fontsize=10, fontweight='bold', ha='center', va='center')
                        self.probe_artists.append(txt)
                    
        self.post_canvas.draw_idle()

    def _unbind_post_events(self):
        widget = self.post_canvas.get_tk_widget()
        widget.unbind('<Button-1>')
        widget.unbind('<B1-Motion>')
        widget.unbind('<ButtonRelease-1>')
        widget.unbind('<Double-Button-1>')
        widget.unbind('<Motion>')
        widget.config(cursor='')

    def start_area_tool(self, shape_type):
        if self.current_image is None: return
        self.current_shape = shape_type
        self.shape_start = None
        self.roi_points = []
        if self.post_cid:
            self.post_canvas.mpl_disconnect(self.post_cid)
            
        if shape_type == 'poly':
            self.post_cid = self.post_canvas.mpl_connect('button_press_event', self._on_mpl_poly_click)
            # We also need motion for preview line
            self.post_cid_motion = self.post_canvas.mpl_connect('motion_notify_event', self._on_mpl_poly_move)
        else:
            self.post_cid = self.post_canvas.mpl_connect('button_press_event', self._on_mpl_area_down)
            self.post_cid_release = self.post_canvas.mpl_connect('button_release_event', self._on_mpl_area_up)
            self.post_cid_motion = self.post_canvas.mpl_connect('motion_notify_event', self._on_mpl_area_drag)
            
    def _on_mpl_area_down(self, event):
        if event.inaxes != self.post_ax: return
        self.shape_start = (event.xdata, event.ydata)
        
    def _on_mpl_area_drag(self, event):
        if not self.shape_start or event.inaxes != self.post_ax: return
        x0, y0 = self.shape_start
        x1, y1 = event.xdata, event.ydata
        
        # Draw preview
        # Remove previous preview
        for art in getattr(self, 'preview_artists', []):
            art.remove()
        self.preview_artists = []
        
        if self.current_shape == 'rect':
            rect = plt.Rectangle((min(x0, x1), min(y0, y1)), abs(x1-x0), abs(y1-y0), 
                               fill=False, edgecolor='yellow', linewidth=2, linestyle='--')
            self.post_ax.add_patch(rect)
            self.preview_artists.append(rect)
            
        elif self.current_shape == 'circle':
            r = ((x1-x0)**2 + (y1-y0)**2)**0.5
            circ = plt.Circle((x0, y0), r, fill=False, edgecolor='yellow', linewidth=2, linestyle='--')
            self.post_ax.add_patch(circ)
            self.preview_artists.append(circ)
            
        self.post_canvas.draw_idle()
        
    def _on_mpl_area_up(self, event):
        if not self.shape_start or event.inaxes != self.post_ax: return
        x0, y0 = self.shape_start
        x1, y1 = event.xdata, event.ydata
        self.shape_start = None
        
        # Clear preview
        for art in getattr(self, 'preview_artists', []):
            art.remove()
        self.preview_artists = []
        self.post_canvas.draw_idle()
        
        # Disconnect events
        self.post_canvas.mpl_disconnect(self.post_cid)
        self.post_canvas.mpl_disconnect(self.post_cid_release)
        self.post_canvas.mpl_disconnect(self.post_cid_motion)
        
        # Notify callback
        if self.callbacks.get('on_area_added'):
            if self.current_shape == 'rect':
                coords = [x0, y0, x1, y1]
            elif self.current_shape == 'circle':
                r = ((x1-x0)**2 + (y1-y0)**2)**0.5
                coords = [x0, y0, r]
            self.callbacks['on_area_added'](self.current_shape, coords)

    def _on_mpl_poly_click(self, event):
        if event.inaxes != self.post_ax: return
        if event.button == 1: # Left click
            self.roi_points.append([event.xdata, event.ydata])
            # Draw point
            pt, = self.post_ax.plot(event.xdata, event.ydata, 'yo', markersize=4)
            if not hasattr(self, 'preview_artists'): self.preview_artists = []
            self.preview_artists.append(pt)
            
            if len(self.roi_points) > 1:
                # Draw line from prev
                p1 = self.roi_points[-2]
                p2 = self.roi_points[-1]
                ln, = self.post_ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'y--')
                self.preview_artists.append(ln)
                
            self.post_canvas.draw_idle()
            
        elif event.button == 3 or (event.dblclick): # Right click or double click to finish
            self._finish_poly()
            
    def _on_mpl_poly_move(self, event):
        if not self.roi_points or event.inaxes != self.post_ax: return
        # Draw rubber band
        # (Implementation omitted for brevity, can be added if needed)
        pass

    def _finish_poly(self):
        if len(self.roi_points) < 3: return
        
        # Clear preview
        for art in getattr(self, 'preview_artists', []):
            art.remove()
        self.preview_artists = []
        self.post_canvas.draw_idle()
        
        # Disconnect
        self.post_canvas.mpl_disconnect(self.post_cid)
        self.post_canvas.mpl_disconnect(self.post_cid_motion)
        
        if self.callbacks.get('on_area_added'):
            self.callbacks['on_area_added']('poly', self.roi_points)
            
        # Update rect
        y_indices, x_indices = np.where(self.roi_mask)
        if len(x_indices) > 0:
            xmin, xmax = x_indices.min(), x_indices.max()
            ymin, ymax = y_indices.min(), y_indices.max()
            self.roi_rect = (xmin, ymin, xmax+1, ymax+1)
        else:
            self.roi_rect = (0, 0, w, h)

    def on_canvas_click(self, event):
        if not self.drawing_roi:
            return
        x = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        
        # Check if closing polygon (click near start)
        if len(self.roi_points) > 2:
            x0, y0 = self.roi_points[0]
            dist = ((x - x0)**2 + (y - y0)**2)**0.5
            if dist < 10 / self.zoom_factor:
                self.confirm_roi()
                return

        self.roi_points.append((x, y))
        self.draw_roi()

    def on_canvas_drag(self, event):
        pass

    def on_canvas_release(self, event):
        pass

    def on_mouse_move(self, event):
        pass

    def confirm_roi(self):
        print(f"[DEBUG] confirm_roi called. Points: {len(self.roi_points)}, Mask: {self.roi_mask is not None}")
        if not self.roi_points and self.roi_mask is None:
            print("[DEBUG] confirm_roi returning early (no points, no mask)")
            return
        
        if self.roi_points:
            # Create mask from polygon
            h, w = self.current_image.shape[:2]
            new_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.roi_points, np.int32)
            cv2.fillPoly(new_mask, [pts], 1)
            new_mask_bool = new_mask.astype(bool)
            
            if self.roi_mask is None:
                self.roi_mask = np.zeros((h, w), dtype=bool)
                
            if self.is_cutting_mode:
                self.roi_mask = self.roi_mask & ~new_mask_bool
            else:
                self.roi_mask = self.roi_mask | new_mask_bool
                
            self.roi_points = []
            self.drawing_roi = False
        
        # Calculate rect
        if self.roi_mask is not None:
            y_indices, x_indices = np.where(self.roi_mask)
            if len(x_indices) > 0:
                xmin, xmax = x_indices.min(), x_indices.max()
                ymin, ymax = y_indices.min(), y_indices.max()
                self.roi_rect = (xmin, ymin, xmax+1, ymax+1)
            else:
                self.roi_rect = (0, 0, w, h)
        
        self.draw_roi()
        # self.confirm_roi_btn.grid_remove() # Keep persistent
        
        # Notify controller
        if self.callbacks.get('on_roi_confirmed'):
            print("[DEBUG] Calling on_roi_confirmed callback")
            self.callbacks['on_roi_confirmed'](self.roi_mask, self.roi_rect)
        else:
            print("[DEBUG] No on_roi_confirmed callback found!")

    def import_roi_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.bmp;*.tif;*.tiff")])
        if path:
            mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if self.current_image is not None:
                    h, w = self.current_image.shape[:2]
                    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                self.roi_mask = mask_img > 127
                self.confirm_roi()

    def invert_roi(self):
        if self.roi_mask is not None:
            self.roi_mask = ~self.roi_mask
            self.confirm_roi()

    def clean_small_regions(self):
        if self.roi_mask is not None:
            # Simple cleanup
            kernel = np.ones((5,5), np.uint8)
            mask_uint = self.roi_mask.astype(np.uint8)
            mask_uint = cv2.morphologyEx(mask_uint, cv2.MORPH_OPEN, kernel)
            self.roi_mask = mask_uint.astype(bool)
            self.confirm_roi()

    def clear_roi(self):
        self.roi_mask = None
        self.roi_points = []
        self.roi_rect = None
        self.draw_roi()
        self.confirm_roi_btn.grid()

    def update_roi_label(self, image):
        if image is None: return
        h, w = image.shape[:2]
        self.image_info.configure(text=f"Image Size: {w} x {h}")
        # Metrics text is now updated via update_info_text from main controller
        
    def update_info_text(self, text: str):
        """Update the detailed info text area below ROI selection."""
        self.roi_metrics_text.configure(state='normal')
        self.roi_metrics_text.delete(1.0, tk.END)
        self.roi_metrics_text.insert(tk.END, text)
        self.roi_metrics_text.configure(state='disabled')

    # --- Visualization Methods ---
    def set_results(self, results, image_files):
        self.displacement_results = results
        self.image_files = image_files
        if results:
            self.frame_slider.configure(to=len(results))
            self.total_frames_label.configure(text=f"/{len(results)}")
            self.frame_slider.set(1)
            self.update_displacement_preview()

    def _ensure_roi_cache(self):
        return None

    # --- Playback Methods ---
    def _get_active_controls(self):
        """Get the controls for the currently active tab."""
        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 1:
            return 'disp'
        elif current_tab == 2:
            return 'post'
        return None

    def toggle_play(self):
        prefix = self._get_active_controls()
        if not prefix: return
        
        play_btn = getattr(self, f'{prefix}_play_button')
        
        if self.is_playing:
            self.is_playing = False
            play_btn.configure(text="Play")
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None
        else:
            self.is_playing = True
            play_btn.configure(text="Pause")
            self.play_next_frame()

    def play_next_frame(self):
        if not self.is_playing: return
        
        prefix = self._get_active_controls()
        if not prefix: return
        
        slider = getattr(self, f'{prefix}_frame_slider')
        
        # Determine total frames
        total = 0
        if prefix == 'disp':
            total = len(self.displacement_results)
        elif prefix == 'post':
            # Need to check strain results length, but it's not stored here directly
            # We can infer from slider range
            total = int(slider.cget('to'))
            
        current = int(float(slider.get()))
        if current >= total:
            current = 0
        
        slider.set(current + 1)
        # Slider callback handles update
        
        self.play_after_id = self.root.after(self.play_interval, self.play_next_frame)

    def previous_frame(self):
        prefix = self._get_active_controls()
        if not prefix: return
        
        slider = getattr(self, f'{prefix}_frame_slider')
        val = int(float(slider.get()))
        slider.set(max(1, val - 1))

    def next_frame(self):
        prefix = self._get_active_controls()
        if not prefix: return
        
        slider = getattr(self, f'{prefix}_frame_slider')
        total = int(slider.cget('to'))
        val = int(float(slider.get()))
        slider.set(min(total, val + 1))

    def jump_to_frame(self):
        prefix = self._get_active_controls()
        if not prefix: return
        
        entry = getattr(self, f'{prefix}_frame_entry')
        slider = getattr(self, f'{prefix}_frame_slider')
        
        try:
            val = int(entry.get())
            total = int(slider.cget('to'))
            if 1 <= val <= total:
                slider.set(val)
        except ValueError:
            pass

    def reset_state(self):
        """Reset all state, clear results, and switch to ROI tab."""
        self.displacement_results = []
        self.roi_mask = None
        self.roi_rect = None
        self.roi_points = []
        self.roi_history = []
        self.is_playing = False
        
        # Clear ROI canvas
        self.roi_canvas.delete("all")
        
        # Clear Visualization Canvas
        if self.ax_u: self.ax_u.clear()
        if self.ax_v: self.ax_v.clear()
        if self.ax_u: self.ax_u.set_axis_off()
        if self.ax_v: self.ax_v.set_axis_off()
        if self.cb_u: 
            self.cb_u.remove()
            self.cb_u = None
        if self.cb_v: 
            self.cb_v.remove()
            self.cb_v = None
        if self.canvas: self.canvas.draw()
        
        # Switch to ROI tab
        self.show_roi_tab()

    def show_roi_tab(self):
        """Switch to ROI Selection tab."""
        self.notebook.select(0)

    def show_results_tab(self):
        """Switch to Displacement Result tab."""
        self.notebook.select(1)
        
    def change_play_speed(self, val):
        speed_map = {"0.25x": 400, "0.5x": 200, "1x": 100, "2x": 50, "4x": 25}
        self.play_interval = speed_map.get(val, 100)

    def update_displacement_preview(self, *args):
        """Update displacement field preview with support for both reference and deformed image modes"""
        print(f"[DEBUG] update_displacement_preview called. Results: {len(self.displacement_results)}, Image: {hasattr(self, 'current_image')}")
        if not self.displacement_results or not hasattr(self, 'current_image'):
            return
        
        try:
            # Get current frame index
            try:
                current_frame = int(self.frame_slider.get()) - 1
            except Exception:
                current_frame = 0
                
            if current_frame < 0 or current_frame >= len(self.displacement_results):
                print(f"[DEBUG] Invalid frame index: {current_frame}")
                return
            
            # Update frame number input box
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(current_frame + 1))
            
            # Get result for this frame
            result = self.displacement_results[current_frame]
            print(f"[DEBUG] Processing frame {current_frame}. Result shape: {result.shape}")
            
            # Update current image name
            if current_frame < len(self.image_files):
                self.current_image_name.configure(text=self.image_files[current_frame])
            
            # Get visualization parameters from ControlPanel
            cp = self.control_panel
            
            # Load deformed image if needed
            deformed_image = self.current_image
            if cp.background_mode.get() == 'deformed':
                if hasattr(self, 'image_files') and len(self.image_files) > current_frame + 1:
                    try:
                        deformed_img_path = os.path.join(cp.input_path.get(), self.image_files[current_frame + 1])
                        deformed_image = proc.load_and_convert_image(deformed_img_path)
                    except Exception:
                        pass

            # Prepare data
            print("[DEBUG] Calling prepare_visualization_data...")
            data = proc.prepare_visualization_data(
                result, 
                self.current_image, 
                deformed_image,
                self.roi_rect,
                self.roi_mask,
                float(cp.preview_scale.get()),
                int(cp.interp_sample_step.get()),
                cp.background_mode.get(),
                cp.deform_display_mode.get(),
                cp.deform_interp.get(),
                cp.show_deformed_boundary.get(),
                int(cp.quiver_step.get() or "10")
            )
            
            # Determine color range
            if cp.use_fixed_colorbar.get():
                vmin_u = float(cp.colorbar_u_min.get() or 0)
                vmax_u = float(cp.colorbar_u_max.get() or 0)
                vmin_v = float(cp.colorbar_v_min.get() or 0)
                vmax_v = float(cp.colorbar_v_max.get() or 0)
            else:
                # Calculate adaptive range from current frame
                u = result[:, :, 0]
                v = result[:, :, 1]
                vmin_u, vmax_u = np.nanmin(u), np.nanmax(u)
                vmin_v, vmax_v = np.nanmin(v), np.nanmax(v)
                
                # Update ControlPanel entries to reflect adaptive range
                cp.colorbar_u_min.set(f"{vmin_u:.3f}")
                cp.colorbar_u_max.set(f"{vmax_u:.3f}")
                cp.colorbar_v_min.set(f"{vmin_v:.3f}")
                cp.colorbar_v_max.set(f"{vmax_v:.3f}")

            # Update plot
            im_u, im_v = proc.update_displacement_plot(
                self.ax_u, 
                self.ax_v, 
                data,
                cp.colormap.get(),
                float(cp.overlay_alpha.get()),
                vmin_u, vmax_u,
                vmin_v, vmax_v
            )
            
            # Update colorbars
            if cp.show_colorbars.get():
                # U Component
                if im_u:
                    if self.cax_u is None:
                        self.cax_u = self.div_u.append_axes("right", size="5%", pad=0.05)
                    else:
                        self.cax_u.clear()
                    self.cb_u = self.fig.colorbar(im_u, cax=self.cax_u)
                
                # V Component
                if im_v:
                    if self.cax_v is None:
                        self.cax_v = self.div_v.append_axes("right", size="5%", pad=0.05)
                    else:
                        self.cax_v.clear()
                    self.cb_v = self.fig.colorbar(im_v, cax=self.cax_v)
            else:
                # Hide colorbars if they exist
                if self.cax_u is not None:
                    self.cax_u.remove()
                    self.cax_u = None
                    self.cb_u = None
                if self.cax_v is not None:
                    self.cax_v.remove()
                    self.cax_v = None
                    self.cb_v = None

            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating preview: {e}")

    def plot_post_data(self, data_map, title, colormap, alpha, background_image=None, roi_rect=None, vmin=None, vmax=None):
        """Plot post-processing data (e.g., strain) on the post_canvas."""
        if self.post_ax is None:
            return
            
        self.post_ax.clear()
        
        # Plot background image if provided
        if background_image is not None:
            self.post_ax.imshow(background_image, cmap='gray')
            
        # Determine extent for overlay
        extent = None
        if roi_rect is not None:
            # roi_rect is (xmin, ymin, xmax, ymax)
            # imshow extent is (left, right, bottom, top) for origin='upper'
            # In image coords, bottom is ymax, top is ymin
            xmin, ymin, xmax, ymax = roi_rect
            extent = [xmin, xmax, ymax, ymin]
        
        # Plot data
        im = self.post_ax.imshow(data_map, cmap=colormap, alpha=alpha, extent=extent, vmin=vmin, vmax=vmax)
        self.post_ax.set_title(title.upper())
        self.post_ax.set_axis_off()
        
        # Force full image view if background is present
        if background_image is not None:
            h, w = background_image.shape[:2]
            self.post_ax.set_xlim(0, w)
            self.post_ax.set_ylim(h, 0)
        
        # Colorbar
        if not hasattr(self, 'post_div'):
            self.post_div = make_axes_locatable(self.post_ax)
            self.post_cax = None
            
        if self.post_cax is None:
            self.post_cax = self.post_div.append_axes("right", size="5%", pad=0.05)
        else:
            self.post_cax.clear()
            
        self.post_fig.colorbar(im, cax=self.post_cax)
        
        self.post_canvas.draw()

