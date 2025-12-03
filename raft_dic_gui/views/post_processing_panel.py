import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from raft_dic_gui.ui_components import CTkCollapsibleFrame as CollapsibleFrame, Tooltip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os

class PostProcessingPanel(ttk.Frame):
    def __init__(self, parent, config=None, callbacks=None):
        super().__init__(parent)
        self.config = config
        self.callbacks = callbacks or {}
        
        # State variables
        self.strain_method = tk.StringVar(value="green_lagrange")
        self.strain_sigma = tk.StringVar(value="0.0")
        self.strain_components = {
            'exx': tk.BooleanVar(value=True),
            'eyy': tk.BooleanVar(value=True),
            'exy': tk.BooleanVar(value=True),
            'e1': tk.BooleanVar(value=False),
            'e2': tk.BooleanVar(value=False),
            'max_shear': tk.BooleanVar(value=False),
            'von_mises': tk.BooleanVar(value=False),
        }
        
        self.create_widgets()

    def _trigger(self, key, *args):
        if self.callbacks.get(key):
            self.callbacks[key](*args)

    def create_widgets(self):
        # Main Layout: Just the Sidebar content
        # We don't need a PanedWindow anymore since this panel is ONLY the sidebar
        
        # Scrollable Sidebar
        sidebar_canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=sidebar_canvas.yview)
        self.sidebar_content = ttk.Frame(sidebar_canvas)
        
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        sidebar_canvas.create_window((0, 0), window=self.sidebar_content, anchor="nw", tags="self.sidebar_content")
        
        def on_canvas_configure(event):
            sidebar_canvas.itemconfig("self.sidebar_content", width=event.width)
            
        sidebar_canvas.bind("<Configure>", on_canvas_configure)
        self.sidebar_content.bind("<Configure>", lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        
        # 1. Strain Calculation Section
        self._create_strain_section()
        
        # 2. Visualization Controls Section
        self._create_vis_section()
        
        # 3. Virtual Extensometers Section
        self._create_probes_section()
        
        # 4. Data Export Section
        self._create_export_section()

    def _create_strain_section(self):
        frame = CollapsibleFrame(self.sidebar_content, text="Strain Calculation")
        frame.pack(fill="x", padx=5, pady=5)
        content = frame.get_content_frame()
        
        r = 0
        # Method
        ttk.Label(content, text="Method:").grid(row=r, column=0, sticky="w")
        ttk.Combobox(content, textvariable=self.strain_method, values=["green_lagrange", "engineering"]).grid(row=r, column=1, sticky="ew")
        r += 1

        # Smoothing
        ttk.Label(content, text="Smoothing Sigma (px):").grid(row=r, column=0, sticky="w")
        ttk.Entry(content, textvariable=self.strain_sigma, width=6).grid(row=r, column=1, sticky="w")
        r += 1
        
        ttk.Separator(content, orient="horizontal").grid(row=r, column=0, columnspan=2, sticky="ew", pady=5)
        r += 1
        
        # Components
        lbl = ttk.Label(content, text="Components to Calculate:")
        lbl.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 2))
        r += 1
        
        for key, var in self.strain_components.items():
            cb = ttk.Checkbutton(content, text=key.upper().replace('_', ' '), variable=var)
            cb.grid(row=r, column=0, columnspan=2, sticky="w", padx=10)
            r += 1
        
        # Calculate Button
        btn = ttk.Button(content, text="Calculate Strain", command=lambda: self._trigger('calculate_strain'))
        btn.grid(row=r, column=0, columnspan=2, pady=10, sticky="ew")

    def _create_vis_section(self):
        frame = CollapsibleFrame(self.sidebar_content, text="Visualization Controls")
        frame.pack(fill="x", padx=5, pady=5)
        content = frame.get_content_frame()
        
        # Display Component Selector
        ttk.Label(content, text="Display Component:").grid(row=0, column=0, sticky="w")
        self.vis_component = tk.StringVar(value="exx")
        self.vis_comp_combo = ttk.Combobox(content, textvariable=self.vis_component, values=[], state="readonly")
        self.vis_comp_combo.grid(row=0, column=1, sticky="ew")
        self.vis_comp_combo.bind("<<ComboboxSelected>>", lambda e: self._trigger('update_post_preview'))
        
        ttk.Label(content, text="Colormap:").grid(row=1, column=0, sticky="w")
        self.vis_colormap = tk.StringVar(value="turbo")
        cb = ttk.Combobox(content, textvariable=self.vis_colormap, values=["turbo", "viridis", "jet", "magma", "plasma", "inferno", "cividis", "RdYlBu", "coolwarm"])
        cb.grid(row=1, column=1, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self._trigger('update_post_preview'))
        
        ttk.Label(content, text="Transparency (0-1):").grid(row=2, column=0, sticky="w")
        self.vis_alpha = tk.StringVar(value="0.7")
        alpha_entry = ttk.Entry(content, textvariable=self.vis_alpha, width=6)
        alpha_entry.grid(row=2, column=1, sticky="w")
        alpha_entry.bind('<Return>', lambda e: self._trigger('update_post_preview'))
        alpha_entry.bind('<FocusOut>', lambda e: self._trigger('update_post_preview'))
        
        # Color Range Controls
        ttk.Separator(content, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.post_fixed_range = tk.BooleanVar(value=False)
        self.post_vmin = tk.StringVar(value="")
        self.post_vmax = tk.StringVar(value="")
        
        # Store ranges per component: {comp_name: (vmin, vmax)}
        self.component_ranges = {}
        
        cb_fixed = ttk.Checkbutton(content, text="Fixed Color Range", variable=self.post_fixed_range, 
                                 command=lambda: self._trigger('update_post_preview'))
        cb_fixed.grid(row=4, column=0, columnspan=2, sticky="w")
        
        range_frame = ttk.Frame(content)
        range_frame.grid(row=5, column=0, columnspan=2, sticky="ew")
        
        ttk.Label(range_frame, text="Min:").pack(side="left", padx=2)
        v_min_entry = ttk.Entry(range_frame, textvariable=self.post_vmin, width=8)
        v_min_entry.pack(side="left", padx=2)
        v_min_entry.bind('<Return>', lambda e: self._trigger('update_post_preview'))
        v_min_entry.bind('<FocusOut>', lambda e: self._trigger('update_post_preview'))
        
        ttk.Label(range_frame, text="Max:").pack(side="left", padx=2)
        v_max_entry = ttk.Entry(range_frame, textvariable=self.post_vmax, width=8)
        v_max_entry.pack(side="left", padx=2)
        v_max_entry.bind('<Return>', lambda e: self._trigger('update_post_preview'))
        v_max_entry.bind('<FocusOut>', lambda e: self._trigger('update_post_preview'))

    def _create_probes_section(self):
        frame = CollapsibleFrame(self.sidebar_content, text="Virtual Extensometers")
        frame.pack(fill="x", padx=5, pady=5)
        content = frame.get_content_frame()
        
        # Tabs for Point / Line / Area
        tabs = ttk.Notebook(content)
        tabs.pack(fill="x", expand=True)
        
        # Point Tab
        p_tab = ttk.Frame(tabs)
        tabs.add(p_tab, text="Point")
        ttk.Button(p_tab, text="Add Point").pack(fill="x", pady=2)
        ttk.Button(p_tab, text="Remove Point").pack(fill="x", pady=2)
        
        # Line Tab
        l_tab = ttk.Frame(tabs)
        tabs.add(l_tab, text="Line")
        ttk.Button(l_tab, text="Add Line").pack(fill="x", pady=2)
        
        # Area Tab
        a_tab = ttk.Frame(tabs)
        tabs.add(a_tab, text="Area")
        ttk.Button(a_tab, text="Add Rect").pack(fill="x", pady=2)

    def _create_export_section(self):
        frame = CollapsibleFrame(self.sidebar_content, text="Data Export")
        frame.pack(fill="x", padx=5, pady=5)
        content = frame.get_content_frame()
        
        ttk.Button(content, text="Export Strain Data...").pack(fill="x", pady=2)
        ttk.Button(content, text="Export Probe Data...").pack(fill="x", pady=2)
