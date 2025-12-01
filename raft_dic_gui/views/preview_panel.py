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
        self.cb_v = None
        
        self.create_widgets()

    def create_widgets(self):
        # ROI Selection Panel
        self._create_roi_panel()
        # Displacement Overlay Panel
        self._create_vis_panel()

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
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(disp_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        self.disp_control_frame = control_frame
        
        # Playback controls
        play_control = ttk.Frame(control_frame)
        play_control.grid(row=0, column=0, padx=5)
        
        self.play_button = ttk.Button(play_control, text="Play", width=5, command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=2)
        
        frame_control = ttk.Frame(control_frame)
        frame_control.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(frame_control, text="<", width=3, command=self.previous_frame).grid(row=0, column=0, padx=2)
        ttk.Label(frame_control, text="Frame:").grid(row=0, column=1, padx=5)
        self.frame_entry = ttk.Entry(frame_control, width=5)
        self.frame_entry.grid(row=0, column=2, padx=2)
        self.total_frames_label = ttk.Label(frame_control, text="/1")
        self.total_frames_label.grid(row=0, column=3, padx=2)
        ttk.Button(frame_control, text="Go", command=self.jump_to_frame).grid(row=0, column=4, padx=5)
        ttk.Button(frame_control, text=">", width=3, command=self.next_frame).grid(row=0, column=5, padx=2)
        
        self.current_image_name = ttk.Label(frame_control, text="")
        self.current_image_name.grid(row=0, column=6, padx=5)
        
        speed_frame = ttk.Frame(play_control)
        speed_frame.grid(row=0, column=1, padx=5)
        ttk.Label(speed_frame, text="Speed:").grid(row=0, column=0)
        self.speed_var = tk.StringVar(value="1x")
        speed_menu = ttk.OptionMenu(speed_frame, self.speed_var, "1x", "0.25x", "0.5x", "1x", "2x", "4x", command=self.change_play_speed)
        speed_menu.grid(row=0, column=1)
        
        self.frame_slider = ttk.Scale(disp_frame, from_=1, to=1, orient=tk.HORIZONTAL, command=self.update_displacement_preview)
        self.frame_slider.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
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

        self.draw_rulers(new_w, new_h)

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

    def start_polygon_tool(self, cut=False):
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

    def start_shape_tool(self, shape, cut=False):
        """Start a shape-drawing tool (rect/square/circle/triangle), with click-drag to define."""
        if self.current_image is None:
            tk.messagebox.showerror("Error", "Please select input images first")
            return
        if cut and self.roi_mask is None:
            tk.messagebox.showerror("Error", "Please draw initial ROI first")
            return
        self.is_cutting_mode = bool(cut)
        self.current_shape = shape
        self.shape_start = None
        self.roi_canvas.config(cursor='crosshair')

        # Overlay existing ROI if any
        self.draw_roi()

        # Bind dragging events
        self.roi_canvas.bind('<Button-1>', self._shape_mouse_down)
        self.roi_canvas.bind('<B1-Motion>', self._shape_mouse_move)
        self.roi_canvas.bind('<ButtonRelease-1>', self._shape_mouse_up)

    def _shape_mouse_down(self, event):
        self.roi_canvas.delete('shape_preview')
        x = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        self.shape_start = (x, y)

    def _shape_mouse_move(self, event):
        if not self.shape_start:
            return
        self.roi_canvas.delete('shape_preview')
        x0, y0 = self.shape_start
        x1 = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y1 = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        # Convert to canvas coords for preview
        cx0, cy0 = x0 * self.zoom_factor, y0 * self.zoom_factor
        cx1, cy1 = x1 * self.zoom_factor, y1 * self.zoom_factor
        if self.current_shape in ('rect',):
            self.roi_canvas.create_rectangle(cx0, cy0, cx1, cy1, outline='yellow', width=2, tags='shape_preview')
        elif self.current_shape == 'circle':
            # Circle within bounding box (use min side)
            dx, dy = cx1 - cx0, cy1 - cy0
            r = min(abs(dx), abs(dy)) / 2
            cx = cx0 + (dx/2)
            cy = cy0 + (dy/2)
            self.roi_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline='yellow', width=2, tags='shape_preview')

    def _shape_mouse_up(self, event):
        if not self.shape_start:
            return
        x0, y0 = self.shape_start
        x1 = self.roi_canvas.canvasx(event.x) / self.zoom_factor
        y1 = self.roi_canvas.canvasy(event.y) / self.zoom_factor
        self.shape_start = None
        self.roi_canvas.delete('shape_preview')

        # Apply shape to mask
        self._apply_shape_to_mask(self.current_shape, x0, y0, x1, y1, cut=self.is_cutting_mode)

        # Unbind shape events and reset cursor
        self.roi_canvas.unbind('<Button-1>')
        self.roi_canvas.unbind('<B1-Motion>')
        self.roi_canvas.unbind('<ButtonRelease-1>')
        self.roi_canvas.config(cursor='')

        # Update display, show confirm
        self.draw_roi()
        self.confirm_roi_btn.grid()
        self.confirm_roi_btn.configure(state='normal')
        
        # Notify controller
        if self.callbacks.get('on_roi_confirmed'):
            self.callbacks['on_roi_confirmed'](self.roi_mask, self.roi_rect)

    def _apply_shape_to_mask(self, shape: str, x0: float, y0: float, x1: float, y1: float, cut: bool):
        h, w = self.current_image.shape[:2]
        x0i, y0i = int(round(x0)), int(round(y0))
        x1i, y1i = int(round(x1)), int(round(y1))
        xmin, xmax = max(0, min(x0i, x1i)), min(w - 1, max(x0i, x1i))
        ymin, ymax = max(0, min(y0i, y1i)), min(h - 1, max(y0i, y1i))

        if self.roi_mask is None:
            self.roi_mask = np.zeros((h, w), dtype=bool)

        new_mask = np.zeros((h, w), dtype=np.uint8)
        if shape in ('rect',):
            cv2.rectangle(new_mask, (xmin, ymin), (xmax, ymax), color=1, thickness=-1)
        elif shape == 'circle':
            cx = int(round((xmin + xmax) / 2))
            cy = int(round((ymin + ymax) / 2))
            r = int(round(min(xmax - xmin, ymax - ymin) / 2))
            cv2.circle(new_mask, (cx, cy), r, color=1, thickness=-1)

        new_mask_bool = new_mask.astype(bool)
        if cut:
            self.roi_mask = self.roi_mask & ~new_mask_bool
        else:
            self.roi_mask = self.roi_mask | new_mask_bool
            
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
        if not self.roi_points and self.roi_mask is None:
            return
        
        if self.roi_points:
            # Create mask from polygon
            h, w = self.current_image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.roi_points, np.int32)
            cv2.fillPoly(mask, [pts], 1)
            self.roi_mask = mask.astype(bool)
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
            self.callbacks['on_roi_confirmed'](self.roi_mask, self.roi_rect)

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
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.play_button.configure(text="Play")
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None
        else:
            self.is_playing = True
            self.play_button.configure(text="Pause")
            self.play_next_frame()

    def play_next_frame(self):
        if not self.is_playing: return
        
        current = int(float(self.frame_slider.get()))
        total = len(self.displacement_results)
        if current >= total:
            current = 0
        
        self.frame_slider.set(current + 1)
        self.update_displacement_preview()
        
        self.play_after_id = self.root.after(self.play_interval, self.play_next_frame)

    def previous_frame(self):
        val = int(float(self.frame_slider.get()))
        self.frame_slider.set(max(1, val - 1))
        self.update_displacement_preview()

    def next_frame(self):
        val = int(float(self.frame_slider.get()))
        self.frame_slider.set(min(len(self.displacement_results), val + 1))
        self.update_displacement_preview()

    def jump_to_frame(self):
        try:
            val = int(self.frame_entry.get())
            total = len(self.displacement_results)
            if 1 <= val <= total:
                self.frame_slider.set(val)
                self.update_displacement_preview()
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
        if not self.displacement_results or not hasattr(self, 'current_image'):
            return
        
        try:
            # Get current frame index
            try:
                current_frame = int(self.frame_slider.get()) - 1
            except Exception:
                current_frame = 0
                
            if current_frame < 0 or current_frame >= len(self.displacement_results):
                return
            
            # Update frame number input box
            self.frame_entry.delete(0, tk.END)
            self.frame_entry.insert(0, str(current_frame + 1))
            
            # Get result for this frame
            result = self.displacement_results[current_frame]
            
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

