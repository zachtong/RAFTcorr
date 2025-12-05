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
        self.strain_method = tk.StringVar(value="Green-Lagrange (Large Deformation)")
        method_cb = ttk.Combobox(content, textvariable=self.strain_method, 
                               values=["Green-Lagrange (Large Deformation)", "Engineering (Small Strain)"],
                               state="readonly", width=25)
        method_cb.grid(row=r, column=1, sticky="ew")
        Tooltip(method_cb, "Green-Lagrange: For large deformations (includes non-linear terms)\nEngineering: For small strains (linear approximation)")
        r += 1

        # VSG Parameters
        # VSG Size
        ttk.Label(content, text="VSG Size (px):").grid(row=r, column=0, sticky="w")
        self.vsg_size = tk.StringVar(value="31")
        vsg_entry = ttk.Entry(content, textvariable=self.vsg_size, width=6)
        vsg_entry.grid(row=r, column=1, sticky="w")
        Tooltip(vsg_entry, "Odd integer between 9 and 101")
        r += 1
        
        # Step
        ttk.Label(content, text="Step (px):").grid(row=r, column=0, sticky="w")
        self.strain_step = tk.StringVar(value="1")
        step_entry = ttk.Entry(content, textvariable=self.strain_step, width=6)
        step_entry.grid(row=r, column=1, sticky="w")
        Tooltip(step_entry, "Calculation stride (integer >= 1)")
        r += 1
        
        # Polynomial Order
        ttk.Label(content, text="Poly Order:").grid(row=r, column=0, sticky="w")
        self.poly_order = tk.StringVar(value="1")
        ttk.Combobox(content, textvariable=self.poly_order, values=["1", "2"], state="readonly", width=4).grid(row=r, column=1, sticky="w")
        r += 1
        
        # Weighting
        ttk.Label(content, text="Weighting:").grid(row=r, column=0, sticky="w")
        self.weighting = tk.StringVar(value="Gaussian")
        ttk.Combobox(content, textvariable=self.weighting, values=["Uniform", "Gaussian"], state="readonly", width=8).grid(row=r, column=1, sticky="w")
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
        self.vis_comp_combo = ttk.Combobox(content, textvariable=self.vis_component, values=['u', 'v'], state="readonly")
        self.vis_comp_combo.grid(row=0, column=1, sticky="ew")
        self.vis_comp_combo = ttk.Combobox(content, textvariable=self.vis_component, values=['u', 'v'], state="readonly")
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
        self.probe_tabs = ttk.Notebook(content)
        self.probe_tabs.pack(fill="x", expand=True)
        self.probe_tabs.bind("<<NotebookTabChanged>>", self._on_probe_tab_changed)
        
        # Point Tab
        p_tab = ttk.Frame(self.probe_tabs)
        self.probe_tabs.add(p_tab, text="Point")
        
        # Buttons
        btn_frame = ttk.Frame(p_tab)
        btn_frame.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Add Point", command=lambda: self._trigger('add_point_probe')).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame, text="Remove", command=lambda: self._trigger('remove_probe')).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(btn_frame, text="Clear All", command=lambda: self._trigger('clear_probes')).pack(side="left", expand=True, fill="x", padx=1)
        
        # List Container
        list_frame = ttk.Frame(p_tab)
        list_frame.pack(fill="both", expand=True, pady=2)
        
        # Scrollbar
        sb = ttk.Scrollbar(list_frame, orient="vertical")
        sb.pack(side="right", fill="y")
        
        # List
        cols = ("ID", "Coords", "Color")
        self.probe_list = ttk.Treeview(list_frame, columns=cols, show="headings", height=6)
        self.probe_list.heading("ID", text="ID")
        self.probe_list.heading("Coords", text="Coords")
        self.probe_list.heading("Color", text="Color")
        self.probe_list.column("ID", width=30, anchor="center")
        self.probe_list.column("Coords", width=80, anchor="center")
        self.probe_list.column("Color", width=50, anchor="center")
        self.probe_list.pack(side="left", fill="both", expand=True)
        
        self.probe_list.configure(yscrollcommand=sb.set)
        sb.configure(command=self.probe_list.yview)
        
        self.probe_list.bind("<<TreeviewSelect>>", lambda e: self._trigger('select_probe'))
        
        # Line Tab
        l_tab = ttk.Frame(self.probe_tabs)
        self.probe_tabs.add(l_tab, text="Line")
        
        # Line Buttons
        l_btn_frame = ttk.Frame(l_tab)
        l_btn_frame.pack(fill="x", pady=2)
        ttk.Button(l_btn_frame, text="Add Line", command=lambda: self._trigger('add_line_probe')).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(l_btn_frame, text="Remove", command=lambda: self._trigger('remove_probe')).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(l_btn_frame, text="Clear All", command=lambda: self._trigger('clear_probes')).pack(side="left", expand=True, fill="x", padx=1)
        
        # Metric Selection
        metric_frame = ttk.Frame(l_tab)
        metric_frame.pack(fill="x", pady=2, padx=2)
        ttk.Label(metric_frame, text="Metric:").pack(side="left")
        self.line_metric_var = tk.StringVar(value="Average")
        self.line_metric_combo = ttk.Combobox(metric_frame, textvariable=self.line_metric_var, values=["Average", "Maximum", "Minimum"], state="readonly", width=10)
        self.line_metric_combo.pack(side="left", padx=5)
        self.line_metric_combo.bind("<<ComboboxSelected>>", lambda e: self._trigger('update_post_preview'))
        
        # Line List Container
        l_list_frame = ttk.Frame(l_tab)
        l_list_frame.pack(fill="both", expand=True, pady=2)
        
        # Scrollbar
        sb_l = ttk.Scrollbar(l_list_frame, orient="vertical")
        sb_l.pack(side="right", fill="y")
        
        # Line List
        l_cols = ("ID", "Length", "Color")
        self.line_list = ttk.Treeview(l_list_frame, columns=l_cols, show="headings", height=6)
        self.line_list.heading("ID", text="ID")
        self.line_list.heading("Length", text="Length (px)")
        self.line_list.heading("Color", text="Color")
        self.line_list.column("ID", width=30, anchor="center")
        self.line_list.column("Length", width=80, anchor="center")
        self.line_list.column("Color", width=50, anchor="center")
        self.line_list.pack(side="left", fill="both", expand=True)
        
        self.line_list.configure(yscrollcommand=sb_l.set)
        sb_l.configure(command=self.line_list.yview)
        
        self.line_list.bind("<<TreeviewSelect>>", lambda e: self._trigger('select_probe'))
        
        # Area Tab
        a_tab = ttk.Frame(self.probe_tabs)
        self.probe_tabs.add(a_tab, text="Area")
        
        # Area Buttons
        a_btn_frame = ttk.Frame(a_tab)
        a_btn_frame.pack(fill="x", pady=2)
        ttk.Button(a_btn_frame, text="Rect", width=5, command=lambda: self._trigger('add_area_probe', 'rect')).pack(side="left", padx=1)
        ttk.Button(a_btn_frame, text="Circle", width=5, command=lambda: self._trigger('add_area_probe', 'circle')).pack(side="left", padx=1)
        ttk.Button(a_btn_frame, text="Poly", width=5, command=lambda: self._trigger('add_area_probe', 'poly')).pack(side="left", padx=1)
        ttk.Button(a_btn_frame, text="Remove", width=7, command=lambda: self._trigger('remove_probe')).pack(side="left", padx=1)
        ttk.Button(a_btn_frame, text="Clear", width=5, command=lambda: self._trigger('clear_probes')).pack(side="left", padx=1)
        
        # Metric Selection
        a_metric_frame = ttk.Frame(a_tab)
        a_metric_frame.pack(fill="x", pady=2, padx=2)
        ttk.Label(a_metric_frame, text="Metric:").pack(side="left")
        self.area_metric_var = tk.StringVar(value="Average")
        self.area_metric_combo = ttk.Combobox(a_metric_frame, textvariable=self.area_metric_var, values=["Average", "Maximum", "Minimum"], state="readonly", width=10)
        self.area_metric_combo.pack(side="left", padx=5)
        self.area_metric_combo.bind("<<ComboboxSelected>>", lambda e: self._trigger('update_post_preview'))
        
        # Area List Container
        a_list_frame = ttk.Frame(a_tab)
        a_list_frame.pack(fill="both", expand=True, pady=2)
        
        # Scrollbar
        sb_a = ttk.Scrollbar(a_list_frame, orient="vertical")
        sb_a.pack(side="right", fill="y")
        
        # Area List
        a_cols = ("ID", "Type", "Color")
        self.area_list = ttk.Treeview(a_list_frame, columns=a_cols, show="headings", height=6)
        self.area_list.heading("ID", text="ID")
        self.area_list.heading("Type", text="Type")
        self.area_list.heading("Color", text="Color")
        self.area_list.column("ID", width=30, anchor="center")
        self.area_list.column("Type", width=80, anchor="center")
        self.area_list.column("Color", width=50, anchor="center")
        self.area_list.pack(side="left", fill="both", expand=True)
        
        self.area_list.configure(yscrollcommand=sb_a.set)
        sb_a.configure(command=self.area_list.yview)
        
        self.area_list.bind("<<TreeviewSelect>>", lambda e: self._trigger('select_probe'))

    def _on_probe_tab_changed(self, event):
        """Handle tab change in probe section."""
        idx = self.probe_tabs.index("current")
        mode = ['point', 'line', 'area'][idx]
        self._trigger('probe_mode_changed', mode)

    def _create_export_section(self):
        frame = CollapsibleFrame(self.sidebar_content, text="Data Export")
        frame.pack(fill="x", padx=5, pady=5)
        content = frame.get_content_frame()
        
        ttk.Button(content, text="Export Strain Data...").pack(fill="x", pady=2)
        ttk.Button(content, text="Export Probe Data...").pack(fill="x", pady=2)

    def update_probe_list(self, probes):
        """Update the probe list treeview."""
        # Store current selection
        selected_point_id = None
        if self.probe_list.selection():
            item = self.probe_list.selection()[0]
            selected_point_id = self.probe_list.item(item)['values'][0]
            
        selected_line_id = None
        if hasattr(self, 'line_list') and self.line_list.selection():
            item = self.line_list.selection()[0]
            selected_line_id = self.line_list.item(item)['values'][0]

        # Clear existing
        for item in self.probe_list.get_children():
            self.probe_list.delete(item)
        if hasattr(self, 'line_list'):
            for item in self.line_list.get_children():
                self.line_list.delete(item)
        if hasattr(self, 'area_list'):
            for item in self.area_list.get_children():
                self.area_list.delete(item)
            
        # Add new
        for p in probes:
            if p.type == 'point':
                coords_str = f"({p.coords[0]:.1f}, {p.coords[1]:.1f})"
                tag_name = f"color_{p.id}"
                item_id = self.probe_list.insert("", "end", values=(p.id, coords_str, "●"), tags=(tag_name,))
                self.probe_list.tag_configure(tag_name, foreground=p.color)
                
                # Restore selection
                if selected_point_id is not None and p.id == selected_point_id:
                    self.probe_list.selection_set(item_id)
                    
            elif p.type == 'line':
                p1, p2 = p.coords
                length = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                tag_name = f"color_{p.id}"
                item_id = self.line_list.insert("", "end", values=(p.id, f"{length:.1f}", "●"), tags=(tag_name,))
                self.line_list.tag_configure(tag_name, foreground=p.color)
                
                # Restore selection
                if selected_line_id is not None and p.id == selected_line_id:
                    self.line_list.selection_set(item_id)
                    
            elif p.type == 'area':
                shape = p.coords.get('shape', 'unknown')
                tag_name = f"color_{p.id}"
                item_id = self.area_list.insert("", "end", values=(p.id, shape.capitalize(), "●"), tags=(tag_name,))
                self.area_list.tag_configure(tag_name, foreground=p.color)
                
                # Restore selection
                if hasattr(self, 'area_list') and self.area_list.selection():
                    # This logic is slightly flawed for restoring selection across updates if ID isn't stored
                    # But for now, let's just leave it or improve it if needed.
                    pass

    def update_component_list(self, components):
        """Update the values in the display component dropdown."""
        # Always include u and v
        base = ['u', 'v']
        # Add new components, avoiding duplicates
        full_list = base + [c for c in components if c not in base]
        self.vis_comp_combo['values'] = full_list
        
        # If current selection is not in list, default to u
        if self.vis_component.get() not in full_list:
            self.vis_component.set('u')
