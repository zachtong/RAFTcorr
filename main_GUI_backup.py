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
from tkinter import ttk, filedialog, messagebox, simpledialog
import customtkinter as ctk
import tkinter.font as tkfont
import os
import cv2
from PIL import Image, ImageTk
from raft_dic_gui import processing as proc
from raft_dic_gui import model as mdl
from raft_dic_gui import preview as vis
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import io
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# CustomTkinter integration shims (visual-only refactor, logic unchanged)
# ---------------------------------------------------------------------------
# We introduce CTk-based replacements while preserving names and callbacks.
# A CTk-based collapsible container and a lightweight ttk adapter are provided
# to modernize the appearance without touching the functional logic.

class CTkCollapsibleFrame(ctk.CTkFrame):
    """A collapsible frame with header and content area (CTk-styled)."""
    def __init__(self, parent, text="", **kwargs):
        super().__init__(parent, **kwargs)
        # Header
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=6, pady=(4,2))
        self.header_frame.grid_columnconfigure(2, weight=1)

        self.toggle_button = ctk.CTkButton(
            self.header_frame, text="−", width=28, command=self.toggle)
        self.toggle_button.grid(row=0, column=0, padx=(0,8))

        self.title_label = ctk.CTkLabel(
            self.header_frame, text=text, font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"))
        self.title_label.grid(row=0, column=1, sticky="w")

        # Content
        self._content = ctk.CTkFrame(self, corner_radius=8)
        self._content.grid(row=1, column=0, sticky="nsew", padx=6, pady=(2,6))
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._collapsed = False

    def toggle(self):
        if self._collapsed:
            self._content.grid()
            self.toggle_button.configure(text="−")
        else:
            self._content.grid_remove()
            self.toggle_button.configure(text="+")
        self._collapsed = not self._collapsed

    def get_content_frame(self):
        return self._content


class _ConfigAliasMixin:
    def config(self, *args, **kwargs):
        return self.configure(*args, **kwargs)

class _SeparatorShim(ctk.CTkFrame, _ConfigAliasMixin):
    def __init__(self, parent, orient="horizontal", **kwargs):
        color = kwargs.pop("fg_color", ("#e5e7eb", "#2b2b2b"))
        if orient == "horizontal":
            super().__init__(parent, height=1, fg_color=color)
        else:
            super().__init__(parent, width=1, fg_color=color)

class _PanedWindowShim(tk.PanedWindow, _ConfigAliasMixin):
    def __init__(self, parent, orient=tk.HORIZONTAL, **kwargs):
        kwargs.pop('style', None)
        super().__init__(parent, orient=orient, sashrelief=tk.FLAT, bd=0)
        try:
            self.configure(sashwidth=6)
        except Exception:
            pass

    def add(self, widget, weight=1):
        try:
            super().add(widget)
        except Exception:
            super().add(widget)

    def sashpos(self, index, pos):
        try:
            if self.cget('orient') in (tk.HORIZONTAL, 'horizontal'):
                self.sash_place(index, pos, 0)
            else:
                self.sash_place(index, 0, pos)
        except Exception:
            pass

class _ProgressbarShim(ctk.CTkProgressBar, _ConfigAliasMixin):
    def __init__(self, parent, length=None, mode='determinate', **kwargs):
        super().__init__(parent, **kwargs)
        if length:
            try:
                self.configure(width=length)
            except Exception:
                pass
        if mode == 'indeterminate':
            try:
                self.configure(mode='indeterminate')
            except Exception:
                pass

    def configure(self, *args, **kwargs):
        value = kwargs.pop('value', None)
        if value is not None:
            try:
                self.set(float(value)/100.0)
            except Exception:
                pass
        return super().configure(*args, **kwargs)

class _TTKAdapter:
    class Frame(ctk.CTkFrame, _ConfigAliasMixin):
        def __init__(self, parent, padding=None, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('padding', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class LabelFrame(ctk.CTkFrame, _ConfigAliasMixin):
        def __init__(self, parent, text=None, padding=None, **kwargs):
            # Ignore unsupported ttk LabelFrame args; render as CTkFrame
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('padding', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Label(ctk.CTkLabel, _ConfigAliasMixin):
        def __init__(self, parent, **kwargs):
            # Map ttk args to CTkLabel
            fg = kwargs.pop('foreground', None)
            bg = kwargs.pop('background', None)
            if fg:
                kwargs['text_color'] = fg
            if bg:
                kwargs['fg_color'] = bg
            kwargs.pop('padding', None)
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            super().__init__(parent, **kwargs)

        def configure(self, require_redraw=False, **kwargs):
            # Intercept unsupported ttk args on reconfigure
            fg = kwargs.pop('foreground', None)
            bg = kwargs.pop('background', None)
            if fg is not None:
                kwargs['text_color'] = fg
            if bg is not None:
                kwargs['fg_color'] = bg
            kwargs.pop('padding', None)
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(require_redraw=require_redraw, **kwargs)

    class Button(ctk.CTkButton, _ConfigAliasMixin):
        def __init__(self, parent, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('padding', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            cmd = kwargs.get('command', None)
            txt = kwargs.get('text', None)
            def _is_garbled(t):
                if not isinstance(t, str):
                    return False
                try:
                    t.encode('ascii')
                    return False
                except Exception:
                    return True
            if cmd and (txt is None or _is_garbled(txt)):
                try:
                    name = getattr(cmd, '__name__', '')
                    if name == 'previous_frame':
                        kwargs['text'] = '<'
                    elif name == 'next_frame':
                        kwargs['text'] = '>'
                    elif name == 'toggle_play':
                        kwargs['text'] = 'Play'
                except Exception:
                    pass
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('padding', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Entry(ctk.CTkEntry, _ConfigAliasMixin):
        def __init__(self, parent, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            try:
                w = kwargs.get('width', None)
                if w is None or (isinstance(w, (int, float)) and w < 120):
                    kwargs['width'] = 140
            except Exception:
                pass
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Text(ctk.CTkTextbox, _ConfigAliasMixin):
        def __init__(self, parent, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Combobox(ctk.CTkComboBox, _ConfigAliasMixin):
        def __init__(self, parent, textvariable=None, values=(), state=None, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            try:
                w = kwargs.get('width', None)
                if w is None or (isinstance(w, (int, float)) and w < 120):
                    kwargs['width'] = 160
            except Exception:
                pass
            super().__init__(parent, variable=textvariable, values=list(values), **kwargs)
            self._sel_handler = None
            # state 'readonly' is ignored; CTkComboBox has no edit by default
        def bind(self, sequence=None, func=None, add=None):
            if sequence == '<<ComboboxSelected>>' and func is not None:
                self._sel_handler = func
                try:
                    super().configure(command=lambda v: self._sel_handler(None))
                except Exception:
                    pass
            else:
                try:
                    return super().bind(sequence, func, add)
                except Exception:
                    return None
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class OptionMenu(ctk.CTkOptionMenu, _ConfigAliasMixin):
        def __init__(self, parent, variable, default, *values, command=None, **kwargs):
            all_values = (default,) + values
            super().__init__(parent, variable=variable, values=list(all_values), command=command, **kwargs)
            try:
                self.set(default)
            except Exception:
                pass
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Checkbutton(ctk.CTkCheckBox, _ConfigAliasMixin):
        def __init__(self, parent, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Radiobutton(ctk.CTkRadioButton, _ConfigAliasMixin):
        def __init__(self, parent, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            super().__init__(parent, **kwargs)
        def configure(self, *args, **kwargs):
            kwargs.pop('style', None)
            kwargs.pop('relief', None)
            kwargs.pop('borderwidth', None)
            return super().configure(*args, **kwargs)

    class Scale(ctk.CTkSlider, _ConfigAliasMixin):
        def __init__(self, parent, from_=0, to=1, orient=None, length=None, command=None, **kwargs):
            super().__init__(parent, from_=from_, to=to, command=command, **kwargs)
            if length:
                try:
                    self.configure(width=length)
                except Exception:
                    pass
    Progressbar = _ProgressbarShim
    Separator = _SeparatorShim
    PanedWindow = _PanedWindowShim
    class Scrollbar(ctk.CTkScrollbar, _ConfigAliasMixin):
        def __init__(self, parent, orient="vertical", command=None, **kwargs):
            # Map ttk 'orient' to CTk 'orientation'
            orientation = "vertical"
            try:
                if isinstance(orient, str):
                    if orient.lower().startswith('h'):
                        orientation = "horizontal"
                else:
                    # Fallback: any other value keeps default vertical
                    pass
            except Exception:
                pass
            super().__init__(parent, orientation=orientation, command=command, **kwargs)
    Style = object

# Override future uses of CollapsibleFrame and ttk with CTk variants
CollapsibleFrame = CTkCollapsibleFrame
ttk = _TTKAdapter()

class LegacyCollapsibleFrame(ttk.LabelFrame):
    """A collapsible frame widget that can be expanded or collapsed."""
    def __init__(self, parent, text="", **kwargs):
        super().__init__(parent, text=text, **kwargs)
        
        # Create a container frame for the header
        self.header_frame = ttk.Frame(self)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        # Add toggle button to header
        self.toggle_button = ttk.Button(self.header_frame, text="ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼", width=2,
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
            self.toggle_button.configure(text="ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶")
        else:
            self.content_frame.grid()
            self.toggle_button.configure(text="ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼")
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
    def _enable_confirm(self):
        try:
            self.confirm_roi_btn.configure(state="normal")
            self.confirm_roi_btn.grid()
        except Exception:
            pass
        
    def __init__(self, root):
        self.root = root
        # Ensure CTk adapters are active at runtime
        try:
            global CollapsibleFrame, ttk
            CollapsibleFrame = CTkCollapsibleFrame
            ttk = _TTKAdapter()
        except Exception:
            pass
        self.root.title("RAFT-DIC Displacement Field Calculator")
        # Apply a consistent ttk theme and styles for a more modern look
        self.apply_theme()
        # Set default and minimum window size (taller to reveal Run button)
        try:
            self.root.geometry('1960x990')
        except Exception:
            pass
        # Maximize window on launch
        try:
            self.root.state("zoomed")
        except Exception:
            try:
                self.root.attributes("-zoomed", True)
            except Exception:
                pass
        self.root.minsize(1200, 900)
        
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
        
        # Create left control panel with fixed minimum width (wider for visibility)
        self.left_frame = ttk.Frame(self.main_paned, width=420)
        self.left_frame.grid_propagate(False)  # Prevent frame from shrinking
        self.main_paned.add(self.left_frame, weight=0)
        
        # Note: no extra separator needed; the PanedWindow sash serves as divider
        
        # Create ROI and Visualization panes directly on the main splitter
        roi_pane = tk.Frame(self.main_paned)
        vis_pane = tk.Frame(self.main_paned)
        self.roi_pane = roi_pane
        self.vis_pane = vis_pane
        try:
            self.main_paned.add(roi_pane, weight=1)
            self.main_paned.add(vis_pane, weight=1)
        except Exception:
            pass
        # Track dragging on the outer sash to update ratios
        try:
            self.main_paned.bind('<ButtonPress-1>', self._on_sash_press)
            self.main_paned.bind('<ButtonRelease-1>', self._on_sash_release)
        except Exception:
            pass
        
        # Create control panel in left frame
        self.create_control_panel(self.left_frame)
        
        # Create ROI and Visualization panels in their respective panes
        self.create_preview_panel(roi_pane, vis_pane)
        
        # Store desired split ratios r1:left, r2:roi, r3:vis (sum=1)
        self._ratios = {'r1': 0.3, 'r2': 0.3, 'r3': 0.4}
        self._sash_dragging = False
        self._applying_layout = False
        self._resize_after = None
        self.root.update_idletasks()
        try:
            total_w = max(1, self.main_paned.winfo_width())
            s0 = int(total_w * self._ratios['r1'])
            s1 = int(total_w * (self._ratios['r1'] + self._ratios['r2']))
            self.main_paned.sashpos(0, s0)
            self.main_paned.sashpos(1, s1)
        except Exception:
            pass
        # Keep ratios on window resize
        try:
            self.root.bind('<Configure>', self._on_root_resize)
        except Exception:
            pass
        
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
        # Default shift size per spec; used when cropping large images
        self.shift_size = tk.StringVar(value="1800")  # default shift for 2000 crops
        # Removed: max_displacement UI/logic (fusion now uses smooth weighting)
        
        # Add colorbar range variables
        self.use_fixed_colorbar = tk.BooleanVar(value=False)
        self.colorbar_u_min = tk.StringVar(value="-1")
        self.colorbar_u_max = tk.StringVar(value="1")
        self.colorbar_v_min = tk.StringVar(value="-1")
        self.colorbar_v_max = tk.StringVar(value="1")
        self.symmetric_colorbar = tk.BooleanVar(value=False)
        
        # Add colormap variable (default to turbo for better contrast)
        self.colormap = tk.StringVar(value="turbo")

        # Modify smoothing processing related variables
        self.use_smooth = tk.BooleanVar(value=True)
        self.sigma = tk.StringVar(value="2.0")

        # Visualization extras
        self.symmetric_colorbar = tk.BooleanVar(value=False) if not hasattr(self, 'symmetric_colorbar') else self.symmetric_colorbar
        self.deform_interp = tk.StringVar(value="linear")      # 'linear', 'nearest', 'rbf'
        self.show_deformed_boundary = tk.BooleanVar(value=True)
        self.deform_display_mode = tk.StringVar(value="heatmap")  # 'heatmap' or 'quiver'
        self.quiver_step = tk.StringVar(value="20")
        
        # Performance prefs
        self.fast_preview = tk.BooleanVar(value=True)
        self.show_colorbars = tk.BooleanVar(value=False)
        self.preview_scale = tk.StringVar(value="0.5")

        # Parameters for memory-safe tiled ROI processing
        # Default to "Small" preset (~100 px) rather than legacy 32
        self.D_global = tk.StringVar(value="100")
        self.g_tile = tk.StringVar(value="100")
        self.overlap_ratio = tk.StringVar(value="0.10")
        # Displacement presets UI (maps to D_global & g_tile internally)
        self.disp_preset = tk.StringVar(value="small")  # 'small','large','custom'
        self.disp_custom = tk.StringVar(value="100")     # used when preset == custom
        # Run/stop state
        self.stop_requested = False
        self.roi_confirmed = False
        self.interp_sample_step = tk.StringVar(value="2")
        
        # Additional tiling controls
        self.p_max_pixels = tk.StringVar(value="1100*1100")
        self.prefer_square = tk.BooleanVar(value=False)
        self.show_tiles_overlay = tk.BooleanVar(value=False)
        # Help tooltips text
        self.tooltips = {
            "path": "Select input image directory and output directory for results",
            "mode": "Accumulative: Calculate displacement relative to first frame\nIncremental: Calculate displacement relative to previous frame",
            "crop": "Enable/disable image cropping for processing",
            "crop_size": "Size of the cropping window (Width ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Height)",
            "shift": "Step size for moving the cropping window",
            "max_disp": "Expected maximum displacement value",
            "smooth": "Gaussian smoothing (sigma in pixels). Larger sigma = smoother, less detail. Typical 0.5ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“5.0.",
            "vis": "Visualization settings including colorbar range and colormap",
            "run": "Start processing the image sequence"
        }
        # Extend tooltips for tiling parameters
        self.tooltips.update({
            "tiling": "Tiled ROI DIC: D_global = context padding (px) around ROI bbox; g_tile = per-tile guard band to avoid edge artifacts; overlap_ratio = fraction of valid interior overlap for feathered fusion (default 0.10).",
            "pmax": "Pixel budget per RAFT call (tile area). Default ~1100ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â½1100. Lower to reduce VRAM usage; higher risks OOM.",
            "prefer_square": "If enabled, use square tiles up to the pixel budget; otherwise match context aspect ratio to reduce tile count.",
            "tiles_overlay": "Draw tiles and their valid interiors over ROI preview for debugging."
        })
        
        # ROI related variables
        self.roi_points = []
        # ROI modified -> re-enable confirm button
        self._enable_confirm()
        self.drawing_roi = False      # ROI drawing state flag
        self.roi_mask = None         # ROI binary mask
        self.roi_rect = None         # ROI bounding rectangle coordinates
        self.roi_scale_factor = 1.0  # ROI display scaling factor
        self.display_size = (400, 400)
        # ROI cache
        self._roi_cache = None
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
        
        # Mode radio buttons laid out horizontally
        ttk.Radiobutton(mode_content, text="Accumulative", variable=self.mode,
                       value="accumulative").grid(row=0, column=0, sticky="w", padx=20, pady=2)
        ttk.Radiobutton(mode_content, text="Incremental", variable=self.mode,
                       value="incremental").grid(row=0, column=1, sticky="w", padx=10, pady=2)
        
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
        
        # Crop Size settings with WÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂH order
        ttk.Label(basic_frame, text="Crop Size (WÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂH):").grid(row=1, column=0, sticky="w", padx=5)
        size_frame = ttk.Frame(basic_frame)
        size_frame.grid(row=1, column=1, sticky="w", pady=2)
        
        self.crop_w_entry = ttk.Entry(size_frame, textvariable=self.crop_size_w, width=5, state="disabled")
        self.crop_w_entry.grid(row=0, column=0)
        ttk.Label(size_frame, text="ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â").grid(row=0, column=1, padx=2)
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
        
        # Hide entire Basic Settings (crop-related) per spec
        try:
            basic_frame.grid_remove()
        except Exception:
            pass

        # Advanced parameters (collapsible, hidden by default)
        advanced_cf = CollapsibleFrame(param_content, text="Advanced Parameters")
        advanced_cf.grid(row=1, column=0, sticky="ew", pady=5)
        adv_content = advanced_cf.get_content_frame()
        adv_content.grid_columnconfigure(1, weight=1)
        try:
            # start collapsed
            advanced_cf.toggle()
        except Exception:
            pass

        # [MAX DISPLACEMENT REMOVED]
        # ttk.Label(advanced_frame, text="Max Displacement:").grid(row=0, column=0, sticky="w", padx=5)
        # self.max_disp_entry = ttk.Entry(advanced_frame, textvariable=self.max_displacement, width=10)
        # self.max_disp_entry.grid(row=0, column=1, sticky="w", pady=2)
        # self.max_disp_hint = ttk.Label(advanced_frame, text="")
        # self.max_disp_hint.grid(row=1, column=1, columnspan=2, sticky="w", padx=5)

        # Tiled ROI processing parameters
        tile_frame = ttk.Frame(adv_content)
        tile_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=4)
        tile_frame.grid_columnconfigure(1, weight=1)

        # Max estimated displacement dropdown with custom
        ttk.Label(tile_frame, text="Max estimated displacement:").grid(row=0, column=0, sticky="w", padx=5)
        self.disp_preset_display = tk.StringVar(value="Small (~100 px)")
        disp_combo = ttk.Combobox(tile_frame, textvariable=self.disp_preset_display,
                                  values=("Small (~100 px)", "Large (~300 px)", "Customized"), width=20)
        disp_combo.grid(row=0, column=1, sticky="w")
        # Custom value entry (only when Customized)
        self.disp_custom_entry = ttk.Entry(tile_frame, textvariable=self.disp_custom, width=10)
        self.disp_custom_entry.grid(row=0, column=2, sticky="w", padx=(6,0))
        try:
            self.disp_custom_entry.grid_remove()
        except Exception:
            pass
        def _map_disp_selection(*_):
            try:
                sel = (self.disp_preset_display.get() or "").lower()
                if sel.startswith("small"):
                    self.disp_preset.set("small")
                    try: self.disp_custom_entry.grid_remove()
                    except Exception: pass
                elif sel.startswith("large"):
                    self.disp_preset.set("large")
                    try: self.disp_custom_entry.grid_remove()
                    except Exception: pass
                else:
                    self.disp_preset.set("custom")
                    try: self.disp_custom_entry.grid()
                    except Exception: pass
                self.on_disp_preset_change()
                try:
                    # Refresh ROI preview/metrics to reflect new D/g
                    self.update_roi_label(self.current_image)
                except Exception:
                    pass
            except Exception:
                pass
        try:
            disp_combo.bind('<<ComboboxSelected>>', lambda e: _map_disp_selection())
        except Exception:
            pass

        # Place overlap ratio under Pixel Budget to avoid overlap in the grid
        ttk.Label(tile_frame, text="Overlap Ratio:").grid(row=3, column=0, sticky="w", padx=(5,5))
        self.overlap_entry = ttk.Entry(tile_frame, textvariable=self.overlap_ratio, width=8)
        self.overlap_entry.grid(row=3, column=1, sticky="w")
        try:
            self.overlap_entry.bind('<Return>', lambda e: self.update_roi_label(self.current_image))
            self.overlap_entry.bind('<FocusOut>', lambda e: self.update_roi_label(self.current_image))
        except Exception:
            pass
        # Pixel budget and options
        ttk.Label(tile_frame, text="Pixel Budget (e.g., 1100*1100):").grid(row=1, column=0, sticky="w", padx=5, pady=(4,0))
        self.pmax_entry = ttk.Entry(tile_frame, textvariable=self.p_max_pixels, width=14)
        self.pmax_entry.grid(row=1, column=1, sticky="w", pady=(4,0))
        try:
            self.pmax_entry.bind('<Return>', lambda e: self.update_roi_label(self.current_image))
            self.pmax_entry.bind('<FocusOut>', lambda e: self.update_roi_label(self.current_image))
        except Exception:
            pass
        # Help button for pixel budget
        try:
            hp = self.create_help_button(tile_frame, "pmax")
            hp.grid(row=2, column=4, padx=6, pady=(4,0))
        except Exception:
            pass
        ttk.Checkbutton(tile_frame, text="Show Tiles Overlay", variable=self.show_tiles_overlay,
                        command=lambda: self.update_roi_label(self.current_image)).grid(row=2, column=3, sticky="w", padx=(15,5), pady=(4,0))
        # Help button for tiling
        help_btn = self.create_help_button(tile_frame, "tiling")
        help_btn.grid(row=2, column=4, padx=6, pady=(4,0))

        # Smoothing options
        smooth_frame = ttk.Frame(adv_content)
        smooth_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        
        ttk.Checkbutton(smooth_frame, text="Use Smoothing",
                        variable=self.use_smooth,
                        command=self.update_smoothing_state).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(smooth_frame, text="Sigma (px):").grid(row=0, column=1, sticky="w", padx=(10,0))
        self.sigma_entry = ttk.Entry(smooth_frame, textvariable=self.sigma, width=5)
        self.sigma_entry.grid(row=0, column=2, padx=5)
        self.sigma_hint = ttk.Label(smooth_frame, text="0.5ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“5.0 (Gaussian blur strength)")
        self.sigma_hint.grid(row=0, column=3, sticky="w")
        # Slider for sigma (0.5ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“5.0); maps to StringVar with rounding
        self.sigma_scale = ttk.Scale(smooth_frame, from_=0.5, to=5.0, orient=tk.HORIZONTAL,
                                     command=self.on_sigma_scale_change, length=160)
        self.sigma_scale.grid(row=0, column=4, padx=10)
        # Help button
        help_btn = self.create_help_button(smooth_frame, "smooth")
        help_btn.grid(row=0, column=5, padx=2)
        
        # Visualization settings
        vis_frame = ttk.LabelFrame(param_content, text="Visualization Settings", padding=5)
        vis_frame.grid(row=2, column=0, sticky="ew", pady=5)
        vis_frame.grid_columnconfigure(0, weight=1)
        
        # Colorbar settings
        ttk.Checkbutton(vis_frame, text="Use Fixed Range", 
                       variable=self.use_fixed_colorbar,
                       command=self.update_displacement_preview).grid(row=0, column=0, sticky="w", padx=5)
        
        range_frame = ttk.Frame(vis_frame)
        range_frame.grid(row=1, column=0, sticky="ew", pady=5, padx=5)
        range_frame.grid_columnconfigure(1, weight=1)
        range_frame.grid_columnconfigure(3, weight=1)
        
        # U Range
        ttk.Label(range_frame, text="U Range:").grid(row=0, column=0, sticky="w")
        umin_entry = ttk.Entry(range_frame, textvariable=self.colorbar_u_min, width=8)
        umin_entry.grid(row=0, column=1, padx=2)
        ttk.Label(range_frame, text="to").grid(row=0, column=2, padx=2)
        umax_entry = ttk.Entry(range_frame, textvariable=self.colorbar_u_max, width=8)
        umax_entry.grid(row=0, column=3, padx=2)
        
        # V Range
        ttk.Label(range_frame, text="V Range:").grid(row=1, column=0, sticky="w", pady=2)
        vmin_entry = ttk.Entry(range_frame, textvariable=self.colorbar_v_min, width=8)
        vmin_entry.grid(row=1, column=1, padx=2)
        ttk.Label(range_frame, text="to").grid(row=1, column=2, padx=2)
        vmax_entry = ttk.Entry(range_frame, textvariable=self.colorbar_v_max, width=8)
        vmax_entry.grid(row=1, column=3, padx=2)

        # Bind range entries to update preview
        for ent in (umin_entry, umax_entry, vmin_entry, vmax_entry):
            try:
                ent.bind('<FocusOut>', self.on_param_change)
                ent.bind('<Return>', self.on_param_change)
            except Exception:
                pass
        
        # Colormap selection
        ttk.Label(vis_frame, text="Colormap:").grid(row=2, column=0, sticky="w", padx=5, pady=(5,0))
        colormap_combo = ttk.Combobox(vis_frame, 
                                    textvariable=self.colormap,
                                    values=["turbo", "viridis", "jet", "magma", "plasma", "inferno", 
                                           "cividis", "RdYlBu", "coolwarm"],
                                    width=15,
                                    state="readonly")
        colormap_combo.grid(row=3, column=0, sticky="w", padx=5, pady=(0,5))
        colormap_combo.bind('<<ComboboxSelected>>', self.on_param_change)

        # Transparency control and Update button (moved here from viz pane)
        alpha_row = ttk.Frame(vis_frame)
        alpha_row.grid(row=4, column=0, sticky='w', padx=5, pady=(0,5))
        ttk.Label(alpha_row, text="Transparency:").grid(row=0, column=0, padx=(0,6))
        alpha_entry = ttk.Entry(alpha_row, textvariable=self.overlay_alpha, width=6)
        alpha_entry.grid(row=0, column=1, padx=(0,8))
        ttk.Button(alpha_row, text="Update", command=self.update_displacement_preview).grid(row=0, column=2)

        # Fixed range from specific frame
        fixed_from_frame = ttk.Frame(vis_frame)
        fixed_from_frame.grid(row=5, column=0, sticky='w', padx=5, pady=(0,5))
        ttk.Label(fixed_from_frame, text="Set fixed range from Frame #").grid(row=0, column=0, padx=(0,6))
        self.fixed_range_frame = tk.StringVar(value="1")
        ttk.Entry(fixed_from_frame, textvariable=self.fixed_range_frame, width=6).grid(row=0, column=1, padx=(0,8))
        ttk.Button(fixed_from_frame, text="Apply", command=self.set_fixed_colorbar_from_frame).grid(row=0, column=2)

        # Performance toggles
        perf = ttk.LabelFrame(vis_frame, text="Performance", padding=5)
        perf.grid(row=7, column=0, sticky='ew', padx=5, pady=5)
        ttk.Checkbutton(perf, text="Fast Preview (no colorbars)", variable=self.fast_preview,
                        command=self.update_displacement_preview).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(perf, text="Show Colorbars (slower)", variable=self.show_colorbars,
                        command=self.update_displacement_preview).grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(perf, text="Preview scale:").grid(row=1, column=0, sticky='w', pady=(4,0))
        scale = ttk.Scale(perf, from_=0.25, to=1.0, orient=tk.HORIZONTAL,
                          command=lambda v: self.on_preview_scale_change(v), length=160)
        # Initialize slider with current value
        try:
            scale.set(float(self.preview_scale.get()))
        except Exception:
            scale.set(0.5)
        scale.grid(row=1, column=1, sticky='w', padx=10, pady=(4,0))
        ttk.Label(perf, text="Interp sample step:").grid(row=2, column=0, sticky='w', pady=(4,0))
        step_combo = ttk.Combobox(perf, textvariable=self.interp_sample_step,
                                  values=["1","2","4","8"], width=5, state='readonly')
        step_combo.grid(row=2, column=1, sticky='w', padx=10, pady=(4,0))
        step_combo.bind('<<ComboboxSelected>>', self.on_param_change)
        
        # Background Image Mode selection
        bg_mode_frame = ttk.LabelFrame(vis_frame, text="Background Image Mode", padding=5)
        bg_mode_frame.grid(row=6, column=0, sticky="ew", pady=5)
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

        # Deformed visualization method options
        ttk.Label(deformed_options_frame, text="Deformed Mode:").grid(row=1, column=0, sticky="w", padx=(20,4))
        ttk.Radiobutton(deformed_options_frame, text="Heatmap", value="heatmap",
                        variable=self.deform_display_mode,
                        command=self.update_displacement_preview).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(deformed_options_frame, text="Quiver", value="quiver",
                        variable=self.deform_display_mode,
                        command=self.update_displacement_preview).grid(row=1, column=2, sticky="w")

        ttk.Label(deformed_options_frame, text="Interpolation:").grid(row=2, column=0, sticky="w", padx=(20,4))
        interp_combo = ttk.Combobox(deformed_options_frame, textvariable=self.deform_interp,
                                    values=["linear", "nearest", "rbf"], width=8, state='readonly')
        interp_combo.grid(row=2, column=1, sticky="w")
        interp_combo.bind('<<ComboboxSelected>>', self.on_param_change)

        # Show deformed-options only when 'Deformed Image' is selected
        def _update_deformed_options(*_):
            try:
                if self.background_mode.get() == 'deformed':
                    deformed_options_frame.grid()
                else:
                    deformed_options_frame.grid_remove()
            except Exception:
                pass
        try:
            _update_deformed_options()
            self.background_mode.trace_add('write', lambda *args: _update_deformed_options())
        except Exception:
            pass

        ttk.Checkbutton(deformed_options_frame, text="Show Deformed Boundary",
                        variable=self.show_deformed_boundary,
                        command=self.update_displacement_preview).grid(row=3, column=0, columnspan=2, sticky="w", padx=(20,0))

        ttk.Label(deformed_options_frame, text="Quiver Step:").grid(row=4, column=0, sticky="w", padx=(20,4))
        quiver_entry = ttk.Entry(deformed_options_frame, textvariable=self.quiver_step, width=6)
        quiver_entry.grid(row=4, column=1, sticky="w")
        quiver_entry.bind('<Return>', self.on_param_change)
        quiver_entry.bind('<FocusOut>', self.on_param_change)
        
        # Add separator
        ttk.Separator(control_frame, orient="horizontal").grid(row=5, column=0, sticky="ew", pady=5)
        
        # Run section
        run_frame = CollapsibleFrame(control_frame, text="Run Control")
        run_frame.grid(row=6, column=0, sticky="ew", padx=5, pady=5)
        
        run_content = run_frame.get_content_frame()
        run_content.grid_columnconfigure(0, weight=1)
        
        # Run/Stop controls and progress
        ttk.Button(run_content, text="Run", command=self.run, width=12).grid(row=0, column=0, pady=5, padx=(0,6))
        ttk.Button(run_content, text="Stop", command=self.request_stop, width=12).grid(row=0, column=1, pady=5)
        self.progress = ttk.Progressbar(run_content, length=220, mode='determinate')
        self.progress.grid(row=1, column=0, columnspan=2, pady=5, sticky='ew')
        self.progress_text = ttk.Label(run_content, text="0/0")
        self.progress_text.grid(row=2, column=0, columnspan=2)
        
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
        
        # Bind mouse wheel only to this canvas to avoid affecting other panes
        control_canvas.bind("<MouseWheel>", on_mousewheel)

        # Bind events for parameter changes
        self.crop_h_entry.bind('<Return>', self.on_param_change)
        self.crop_h_entry.bind('<FocusOut>', self.on_param_change)
        self.crop_w_entry.bind('<Return>', self.on_param_change)
        self.crop_w_entry.bind('<FocusOut>', self.on_param_change)
        self.shift_entry.bind('<Return>', self.on_param_change)
        self.shift_entry.bind('<FocusOut>', self.on_param_change)
        # Bind sigma entry to sync with scale
        self.sigma_entry.bind('<Return>', self.on_sigma_entry_change)
        self.sigma_entry.bind('<FocusOut>', self.on_sigma_entry_change)

        # Initialize max displacement UI state
        self.update_max_disp_hint()
        # Initialize smoothing UI state
        self.update_smoothing_state()

    def create_preview_panel(self, roi_pane, vis_pane):
        """Create ROI and Visualization panels in provided panes"""
        # Ensure panes expand
        try:
            roi_pane.grid_rowconfigure(0, weight=1)
            roi_pane.grid_columnconfigure(0, weight=1)
            vis_pane.grid_rowconfigure(0, weight=1)
            vis_pane.grid_columnconfigure(0, weight=1)
        except Exception:
            pass
        
        # ROI selection area (left)
        roi_frame = ttk.LabelFrame(roi_pane, text="ROI Selection", padding="5")
        roi_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        roi_frame.grid_rowconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(0, weight=1)
        roi_pane.grid_rowconfigure(0, weight=1)
        roi_pane.grid_columnconfigure(0, weight=1)
        
        # Create Canvas for ROI selection
        self.roi_canvas = tk.Canvas(roi_frame, width=420, height=420, background='#0f0f14', highlightthickness=1, highlightbackground='#2a2a34')
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
        
        # Add scaling control toolbar (compact)
        zoom_frame = ttk.Frame(roi_frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(zoom_frame, text="+", width=3, command=lambda: self.zoom(1.2)).grid(row=0, column=0, padx=2)
        ttk.Button(zoom_frame, text="-", width=3, command=lambda: self.zoom(0.8)).grid(row=0, column=1, padx=2)
        ttk.Button(zoom_frame, text="100%", width=5, command=self.reset_zoom).grid(row=0, column=2, padx=2)
        
        # ROI control buttons (compact styles)
        self.roi_controls = ttk.LabelFrame(roi_frame, text="ROI Tools", padding=5)
        self.roi_controls.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        for c in range(6):
            self.roi_controls.grid_columnconfigure(c, weight=1)

        # Compact single-row primary tools
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

        # Secondary row for import/cleanup and actions
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
        self.confirm_roi_btn.grid_remove()
        
        # [CROP PREVIEW HIDDEN]
        
        # Displacement overlay area (right)
        disp_frame = ttk.LabelFrame(vis_pane, text="Displacement Field Overlay", padding="5")
        disp_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        disp_frame.grid_rowconfigure(0, weight=1)
        disp_frame.grid_columnconfigure(0, weight=1)
        vis_pane.grid_rowconfigure(0, weight=1)
        vis_pane.grid_columnconfigure(0, weight=1)
        # Keep a reference for dynamic sizing of the visualization
        self.disp_frame = disp_frame
        
        # Add displacement overlay display (use Tk label to avoid CTk image warnings)
        self.displacement_label = tk.Label(disp_frame)
        self.displacement_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add control panel at the bottom
        control_frame = ttk.Frame(disp_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        # Reference to compute available height for the image (frame height minus controls)
        self.disp_control_frame = control_frame

        # Refresh the visualization when the pane/frame geometry changes
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
        
        # Add playback control buttons
        play_control = ttk.Frame(control_frame)
        play_control.grid(row=0, column=0, padx=5)
        
        # Add play/pause button
        self.play_icon = 'Play'
        self.pause_icon = 'Pause'
        self.play_button = ttk.Button(play_control, text=self.play_icon, width=5, command=self.toggle_play)
        self.play_button.grid(row=0, column=0, padx=2)
        # Override any corrupted icon text with safe ASCII labels
        try:
            self.play_icon = "Play"
            self.pause_icon = "Pause"
            self.play_button.configure(text=self.play_icon)
        except Exception:
            pass
        
        # Add frame control
        frame_control = ttk.Frame(control_frame)
        frame_control.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Add Previous button
        ttk.Button(frame_control, text="ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬", width=3,
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
        ttk.Button(frame_control, text="ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¶", width=3,
                  command=self.next_frame).grid(row=0, column=5, padx=2)
        
        # Add current image name display
        self.current_image_name = ttk.Label(frame_control, text="")
        self.current_image_name.grid(row=0, column=6, padx=5)
        # Add clean previous/next buttons to override any garbled icons
        try:
            ttk.Button(frame_control, text="<", width=3, command=self.previous_frame).grid(row=0, column=0, padx=2)
            ttk.Button(frame_control, text=">", width=3, command=self.next_frame).grid(row=0, column=5, padx=2)
        except Exception:
            pass
        
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
        
        # Slider
        self.frame_slider = ttk.Scale(disp_frame, 
                                    from_=1, 
                                    to=1,
                                    orient=tk.HORIZONTAL,
                                    command=self.update_displacement_preview)
        self.frame_slider.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Image information display (fixed under ROI panel)
        self.image_info = ttk.Label(roi_frame, text="")
        self.image_info.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # ROI metrics text area (live tiling/budget info)
        self.roi_metrics_text = tk.Text(roi_frame, height=6, wrap=tk.WORD)
        self.roi_metrics_text.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0,5))
        try:
            # Make it read-only by default
            self.roi_metrics_text.configure(state='disabled')
        except Exception:
            pass

        # Set initial split (~3:4 of the right area)
        # No inner splitter: ROI and Visualization are direct panes on main_paned
        
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
            
            # Read first image using processing helper
            img_path = os.path.join(directory, image_files[0])
            img = proc.load_and_convert_image(img_path)
            if img is None:
                raise Exception(f"Failed to load image: {img_path}")
            
            # Store current image
            self.current_image = img

            # Cropping is deprecated; large-image handling occurs after ROI selection
            self.suggest_crop_for_large_image(img)
            
            # Update crop size with image dimensions
            self.update_crop_size()
            
            # Update ROI preview
            self.update_roi_label(img)
            
            # Update information display (replace, do not append)
            h, w = img.shape[:2]
            info_text = f"Image size: {w}x{h}\nNumber of images: {len(image_files)}"
            if self.roi_rect:
                xmin, ymin, xmax, ymax = self.roi_rect
                info_text += f"\nROI size: {xmax-xmin}x{ymax-ymin}"
            self.image_info.config(text=info_text)
            
        except Exception as e:
            # Auto-fit image to ROI canvas
            try:
                cw = self.roi_canvas.winfo_width() or 420
                ch = self.roi_canvas.winfo_height() or 420
                ih, iw = img.shape[:2]
                scale = min(cw/max(1, iw), ch/max(1, ih))
                self.zoom_factor = max(0.1, min(5.0, float(scale)))
            except Exception:
                self.zoom_factor = 1.0
            messagebox.showerror("Error", str(e))

    def suggest_crop_for_large_image(self, img):
        """No-op: cropping is not used; keep crop UI disabled."""
        try:
            self.use_crop.set(False)
        except Exception:
            pass
    
    def on_disp_preset_change(self):
        """Map user-facing displacement preset to internal D_global and g_tile."""
        try:
            mode = self.disp_preset.get()
        except Exception:
            mode = "small"
        if mode == "small":
            val = 100
            self.disp_custom_entry.configure(state="disabled")
        elif mode == "large":
            val = 300
            self.disp_custom_entry.configure(state="disabled")
        else:
            # Customized: use user entry
            self.disp_custom_entry.configure(state="normal")
            try:
                val = int(float(self.disp_custom.get()))
            except Exception:
                val = 100
                self.disp_custom.set(str(val))
        # Set both D_global and g_tile internally
        self.D_global.set(str(val))
        self.g_tile.set(str(val))
    def on_preview_canvas_resize(self, event):
        """[CROP PREVIEW HIDDEN] No-op for resize"""
        return

    def update_preview(self, *args):
        """[CROP PREVIEW HIDDEN] No-op for preview updates"""
        return
    
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
            
            # Cropping validation removed: we no longer use tiling/cropping
            
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

        # Ensure output directory exists
        os.makedirs(args.project_root, exist_ok=True)

        # Load reference image
        ref_img_path = os.path.join(args.img_dir, image_files[0])
        ref_image = proc.load_and_convert_image(ref_img_path)
        
        # Extract ROI area
        ref_roi = ref_image[ymin:ymax, xmin:xmax]

                # Determine processing canvas
        H_ref, W_ref = ref_image.shape[:2]
        use_full_frame = (H_ref <= 2500 and W_ref <= 2500)

        # Precompute ROI bbox and expanded context C using user D_global
        try:
            Dg = int(float(self.D_global.get()))
        except Exception:
            # Fallback to Small preset default
            Dg = 100
        # Compute minimal bbox from ROI mask
        bbox = proc._compute_min_bounding_box(self.roi_mask.astype(bool))
        if bbox is not None:
            xC, yC, wC, hC = proc._expand_and_clamp(bbox, Dg, W_ref, H_ref)
        else:
            xC, yC, wC, hC = 0, 0, W_ref, H_ref

        # Process each image pair
        total_pairs = len(image_files) - 1
        # Initialize progress UI for this run
        try:
            self.progress.configure(value=0)
            self.progress_text.config(text=f"0/{total_pairs}")
        except Exception:
            pass
        # Accumulate displacements in memory for consolidated save
        sequence_displacements = []

        for i in range(1, len(image_files)):
            # Allow user to abort processing between frames
            if self.stop_requested:
                break
            # Update progress bar
            try:
                percent = (i / max(1, total_pairs)) * 100.0
            except Exception:
                percent = 0.0
            self.progress.configure(value=percent)
            try:
                self.progress_text.config(text=f"{i}/{total_pairs}")
            except Exception:
                pass
            self.root.update_idletasks()

            # Load deformed image
            def_img_path = os.path.join(args.img_dir, image_files[i])
            def_image = proc.load_and_convert_image(def_img_path)
            def_roi = def_image[ymin:ymax, xmin:xmax]

            # Extract ROI mask corresponding area (for crop mode). For non-crop mode,
            # we will pass the full-image mask so the whole image is computed then masked.
            roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None

            
            # Always use memory-safe tiled DIC over ROI with context padding and feathered fusion
            disp_full, _ = proc.dic_over_roi_with_tiling(
                ref_image,
                def_image,
                self.roi_mask,
                args.model,
                args.device,
                D_global=args.D_global,
                g_tile=args.g_tile,
                overlap_ratio=args.overlap_ratio,
                p_max_pixels=args.p_max_pixels,
                prefer_square=args.prefer_square,
                use_smooth=args.use_smooth,
                sigma=args.sigma
            )
            displacement_field = disp_full[ymin:ymax, xmin:xmax, :]

            # Accumulate this frame displacement
            sequence_displacements.append(displacement_field)

            # If incremental mode, update reference image(s)
            if args.mode == "incremental":
                ref_roi = def_roi.copy()
                # Update full reference image for next iteration
                ref_image = def_image.copy()

        # Save consolidated files (one for Python, one for MATLAB)
        try:
            proc.save_displacement_sequence(
                sequence_displacements,
                args.project_root,
                roi_rect=self.roi_rect,
                roi_mask=self.roi_mask,
                save_numpy=True,
                save_matlab=True,
                numpy_filename="displacement_sequence.npz",
                matlab_filename="displacement_sequence.mat",
            )
        except Exception as e:
            print(f"Warning: Failed to save consolidated results: {e}")

        # Keep results in memory for preview/playback
        self.displacement_results = sequence_displacements
        if self.stop_requested:
            # Reset flag and notify
            self.stop_requested = False
            try:
                messagebox.showinfo("Stopped", "Processing stopped by user.")
            except Exception:
                pass
        
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
            
            # Load displacement field (supports in-memory arrays or file paths)
            item = self.displacement_results[current_frame]
            if isinstance(item, str):
                displacement = np.load(item)
            else:
                displacement = item
            
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
                except ValueError:
                    vmin_u = np.nanmin(u)
                    vmax_u = np.nanmax(u)
                try:
                    vmin_v = float(self.colorbar_v_min.get())
                    vmax_v = float(self.colorbar_v_max.get())
                except ValueError:
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
            
            # If we received a PIL image (fast path), use it directly
            if isinstance(fig, Image.Image):
                preview = fig
            else:
                # Save to memory buffer (avoid tight layout to prevent UI jumping)
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                plt.close(fig)
                buf.seek(0)
                preview = Image.open(buf)
            # Determine available drawing size from the visualization pane/frame
            try:
                avail_w = 0
                avail_h = 0
                try:
                    # Prefer current label size while resizing; fall back to frame
                    lw = int(self.displacement_label.winfo_width())
                    lh = int(self.displacement_label.winfo_height())
                    if lw > 1:
                        avail_w = lw
                    if lh > 1:
                        avail_h = lh
                except Exception:
                    pass
                try:
                    fw = int(self.disp_frame.winfo_width())
                    fh = int(self.disp_frame.winfo_height())
                    if fw > avail_w:
                        avail_w = fw
                    # Subtract controls' height if we can measure it
                    ch = int(self.disp_control_frame.winfo_height()) if hasattr(self, 'disp_control_frame') else 0
                    if fh - ch > avail_h:
                        avail_h = max(1, fh - ch)
                except Exception:
                    pass
                # Reasonable fallback if geometry not ready yet
                if avail_w <= 1:
                    avail_w = 800
                if avail_h <= 1:
                    avail_h = 400
                # Light padding to avoid clipping borders
                avail_w = max(100, int(avail_w) - 10)
                avail_h = max(100, int(avail_h) - 10)
            except Exception:
                avail_w, avail_h = 800, 400

            w, h = preview.size
            scale = min(avail_w / max(1, w), avail_h / max(1, h))
            display_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            if display_size[0] != w or display_size[1] != h:
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
                deformed_image = proc.load_and_convert_image(deformed_img_path)
                if deformed_image is None:
                    # Fallback to reference image if loading fails
                    deformed_image = self.current_image
            else:
                # Fallback to reference image
                deformed_image = self.current_image
                
            # Get ROI bounding box coordinates (for displacement field indexing)
            xmin, ymin, xmax, ymax = self.roi_rect

            # Get displacement components and cached ROI data
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            cache = self._ensure_roi_cache()
            roi_mask_crop = cache['roi_mask_crop'] if cache else (self.roi_mask[ymin:ymax, xmin:xmax] if self.roi_mask is not None else None)
            if roi_mask_crop is None:
                return self.create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)

            roi_h, roi_w = u.shape
            Xg = cache['Xg'] if cache else (np.mgrid[0:roi_h, 0:roi_w][1] + xmin)
            Yg = cache['Yg'] if cache else (np.mgrid[0:roi_h, 0:roi_w][0] + ymin)

            valid = roi_mask_crop & ~np.isnan(u) & ~np.isnan(v)
            # Subsample valid points for interpolation if requested
            try:
                sstep = max(1, int(self.interp_sample_step.get()))
            except Exception:
                sstep = 1
            if sstep > 1:
                stride_mask = np.zeros_like(valid, dtype=bool)
                stride_mask[::sstep, ::sstep] = True
                valid = valid & stride_mask
            if not np.any(valid):
                return self.create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)

            x_def = (Xg[valid] + u[valid]).astype(np.float64)
            y_def = (Yg[valid] + v[valid]).astype(np.float64)
            pts = np.column_stack((x_def, y_def))

            h, w = deformed_image.shape[:2]
            # Limit interpolation grid to bounding box of deformed points
            x0 = int(max(0, np.floor(x_def.min())))
            x1 = int(min(w-1, np.ceil(x_def.max())))
            y0 = int(max(0, np.floor(y_def.min())))
            y1 = int(min(h-1, np.ceil(y_def.max())))
            if x1 <= x0 or y1 <= y0:
                return self.create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v)

            Xq, Yq = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))

            method = self.deform_interp.get()
            from scipy.interpolate import griddata, Rbf
            # Interpolate U
            if method == 'rbf' and pts.shape[0] >= 10:
                try:
                    rbf_u = Rbf(x_def, y_def, u[valid], function='multiquadric')
                    u_grid = rbf_u(Xq, Yq)
                except Exception:
                    u_grid = griddata(pts, u[valid], (Xq, Yq), method='linear')
            else:
                u_grid = griddata(pts, u[valid], (Xq, Yq), method='linear')
                if np.isnan(u_grid).all():
                    u_grid = griddata(pts, u[valid], (Xq, Yq), method='nearest')

            # Interpolate V
            if method == 'rbf' and pts.shape[0] >= 10:
                try:
                    rbf_v = Rbf(x_def, y_def, v[valid], function='multiquadric')
                    v_grid = rbf_v(Xq, Yq)
                except Exception:
                    v_grid = griddata(pts, v[valid], (Xq, Yq), method='linear')
            else:
                v_grid = griddata(pts, v[valid], (Xq, Yq), method='linear')
                if np.isnan(v_grid).all():
                    v_grid = griddata(pts, v[valid], (Xq, Yq), method='nearest')

            # Build deformed mask by warping ROI mask forward (preserves holes)
            mask_grid = np.zeros_like(u_grid, dtype=np.uint8)
            xi = np.clip(np.round(x_def).astype(int) - x0, 0, (x1 - x0))
            yi = np.clip(np.round(y_def).astype(int) - y0, 0, (y1 - y0))
            mask_grid[yi, xi] = 1
            try:
                import cv2
                kernel = np.ones((3, 3), np.uint8)
                mask_grid = cv2.morphologyEx(mask_grid, cv2.MORPH_CLOSE, kernel, iterations=1)
            except Exception:
                pass

            # Mask interpolated fields outside deformed ROI (holes removed)
            u_grid = np.where(mask_grid.astype(bool), u_grid, np.nan)
            v_grid = np.where(mask_grid.astype(bool), v_grid, np.nan)

            # For faster display, use cropped region only
            img_crop = deformed_image[y0:y1+1, x0:x1+1]
            u_full = u_grid
            v_full = v_grid
            deformed_mask = ~np.isnan(u_full) & ~np.isnan(v_full)
            
            # Optional downsampling for display speed
            # Aggressive downsampling based on user preview scale
            try:
                pscale = float(self.preview_scale.get())
            except Exception:
                pscale = 0.5
            base_w, base_h = 800, 400
            max_w = int(base_w * max(0.25, min(1.0, pscale)))
            max_h = int(base_h * max(0.25, min(1.0, pscale)))
            target_w = max_w // 2
            target_h = max_h
            scale = min(target_w / (x1 - x0 + 1), target_h / (y1 - y0 + 1), 1.0)
            if scale < 1.0:
                try:
                    import cv2
                    new_size = (max(1, int((x1 - x0 + 1) * scale)), max(1, int((y1 - y0 + 1) * scale)))
                    img_crop = cv2.resize(img_crop, new_size, interpolation=cv2.INTER_AREA)
                    u_full = cv2.resize(u_full.astype(np.float32), new_size, interpolation=cv2.INTER_AREA)
                    v_full = cv2.resize(v_full.astype(np.float32), new_size, interpolation=cv2.INTER_AREA)
                    deformed_mask = cv2.resize(deformed_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)
                except Exception:
                    pass

            # If fast preview without colorbars, return a PIL image directly for speed
            if getattr(self, 'fast_preview', tk.BooleanVar(value=True)).get() and not getattr(self, 'show_colorbars', tk.BooleanVar(value=False)).get():
                return self._create_deformed_displacement_image_fast(
                    img_crop, u_full, v_full, deformed_mask,
                    alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v
                )

            # Create figure with two subplots
            fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
            
            # Display U component on deformed image
            ax_u.imshow(img_crop, cmap='gray')

            u_masked = np.ma.array(u_full, mask=~deformed_mask)
            im_u = ax_u.imshow(u_masked, cmap=current_colormap, alpha=alpha, vmin=vmin_u, vmax=vmax_u)
            ax_u.set_title("U Component on Deformed Image (Polygon ROI)")
            fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)

            # Display V component on deformed image
            ax_v.imshow(img_crop, cmap='gray')

            v_masked = np.ma.array(v_full, mask=~deformed_mask)
            im_v = ax_v.imshow(v_masked, cmap=current_colormap, alpha=alpha, vmin=vmin_v, vmax=vmax_v)
            ax_v.set_title("V Component on Deformed Image (Polygon ROI)")
            fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

            # Optional: show deformed ROI boundary
            if self.show_deformed_boundary.get():
                try:
                    # Extract boundary from roi_mask_crop
                    cnts, _ = cv2.findContours(roi_mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if len(cnts) > 0:
                        cnt = max(cnts, key=cv2.contourArea)  # main contour
                        pts_local = cnt[:, 0, :]  # (N,2) as (x,y) in ROI-local
                        xs = pts_local[:, 0]
                        ys = pts_local[:, 1]
                        xs_g = xs + xmin
                        ys_g = ys + ymin
                        # Sample u,v along boundary with nearest neighbor
                        xs_r = np.clip((xs).astype(int), 0, roi_w-1)
                        ys_r = np.clip((ys).astype(int), 0, roi_h-1)
                        u_b = u[ys_r, xs_r]
                        v_b = v[ys_r, xs_r]
                        x_def_b = xs_g + u_b
                        y_def_b = ys_g + v_b
                        # Shift into crop coordinates and apply display downscale
                        x_def_b = (x_def_b - x0) * scale
                        y_def_b = (y_def_b - y0) * scale
                        ax_u.plot(x_def_b, y_def_b, color='white', linewidth=1.0, alpha=0.8)
                        ax_v.plot(x_def_b, y_def_b, color='white', linewidth=1.0, alpha=0.8)
                except Exception:
                    pass

            # Quiver mode: optionally overlay sparse arrows
            if self.deform_display_mode.get() == 'quiver':
                try:
                    step = max(2, int(self.quiver_step.get()))
                except Exception:
                    step = 20
                # Sample on ROI local grid, then map to crop coordinates and downscale
                xsq = ((Xg[::step, ::step] - x0) * scale)
                ysq = ((Yg[::step, ::step] - y0) * scale)
                uq = u[::step, ::step]
                vq = v[::step, ::step]
                # Filter by ROI mask and finite values
                maskq = roi_mask_crop[::step, ::step] & np.isfinite(uq) & np.isfinite(vq)
                xsq = xsq[maskq]
                ysq = ysq[maskq]
                uq = uq[maskq]
                vq = vq[maskq]
                # Plot quivers on both axes
                ax_u.quiver(xsq, ysq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
                ax_v.quiver(xsq, ysq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
            
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

    def _create_deformed_displacement_image_fast(self, img_crop, u_full, v_full, deformed_mask,
                                                 alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        bg = img_crop
        if bg.ndim == 2:
            bg_rgb = np.stack([bg, bg, bg], axis=-1)
        else:
            bg_rgb = bg[:, :, :3]

        def blend_component(comp, vmin, vmax):
            norm = (comp - vmin) / max(1e-8, (vmax - vmin))
            norm = np.clip(norm, 0.0, 1.0)
            cmap = cm.get_cmap(current_colormap)
            rgba = cmap(norm)  # float 0-1
            rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
            # Alpha composite only on valid mask
            out = bg_rgb.copy()
            m = deformed_mask
            out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + rgb[m].astype(np.float32) * alpha).astype(np.uint8)
            return out

        img_u = blend_component(u_full, vmin_u, vmax_u)
        img_v = blend_component(v_full, vmin_v, vmax_v)
        combined = np.concatenate([img_u, img_v], axis=1)
        from PIL import Image
        return Image.fromarray(combined)

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
        
        # Create figure with two subplots and fixed layout for stability
        fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
        
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
        fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
        
        # Display V component
        ax_v.imshow(background_image, cmap='gray')
        mask_v = ~np.isnan(v_full)
        v_masked = np.ma.array(v_full, mask=~mask_v)
        
        im_v = ax_v.imshow(v_masked, cmap=current_colormap, 
                         alpha=alpha * mask_v, vmin=vmin_v, vmax=vmax_v)
        ax_v.set_title("V Component on Reference Image")
        fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
        
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
    
    def request_stop(self):
        """Signal the processing loop to stop after current frame."""
        self.stop_requested = True
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
        args.crop_size = (int(self.crop_size_h.get() or "0"), int(self.crop_size_w.get() or "0"))
        args.shift = int(self.shift_size.get() or "0")
        
        # Add smoothing parameters
        args.use_smooth = self.use_smooth.get()
        args.sigma = float(self.sigma.get())
        # New tiling/context parameters
        # Ensure D_global/g_tile reflect the selected displacement preset before reading
        # so that args picks up the correct values (Small/Large/Custom)
        self.on_disp_preset_change()
        try:
            args.D_global = int(float(self.D_global.get()))
        except Exception:
            # Fallback to Small preset default
            args.D_global = 100
            self.D_global.set(str(args.D_global))
        try:
            args.g_tile = int(float(self.g_tile.get()))
        except Exception:
            # Fallback to Small preset default
            args.g_tile = 100
            self.g_tile.set(str(args.g_tile))
        try:
            r = float(self.overlap_ratio.get())
            args.overlap_ratio = max(0.0, min(0.9, r))
            self.overlap_ratio.set(f"{args.overlap_ratio:.2f}")
        except Exception:
            args.overlap_ratio = 0.10
            self.overlap_ratio.set("0.10")
        
        # Parse pixel budget input allowing forms like "1100*1100" or "1100x1100"
        try:
            s = (self.p_max_pixels.get() or "").lower().replace(' ', '')
            if '*' in s:
                a,b = s.split('*',1)
                args.p_max_pixels = int(float(a)) * int(float(b))
            elif 'x' in s:
                a,b = s.split('x',1)
                args.p_max_pixels = int(float(a)) * int(float(b))
            else:
                args.p_max_pixels = int(float(s))
        except Exception:
            args.p_max_pixels = 1100*1100
            self.p_max_pixels.set("1100*1100")
        args.prefer_square = bool(self.prefer_square.get())
        # Load model
        model_args = mdl.Args()
        args.model = mdl.load_model("models/raft-dic_v1.pth", args=model_args)
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
            self.progress.configure(value=0)
            
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
            self.progress.configure(value=0)

    def start_cutting_roi(self):
        """Start ROI cutting mode"""
        # Kept for backward-compatibility; now redirects to polygon cut tool
        self.start_polygon_tool(cut=True)

    def start_roi_drawing(self):
        """Start ROI drawing"""
        # Kept for backward-compatibility; now redirects to polygon draw tool
        self.start_polygon_tool(cut=False)

    def start_polygon_tool(self, cut: bool):
        if self.current_image is None:
            messagebox.showerror("Error", "Please select input images first")
            return
        if cut and self.roi_mask is None:
            messagebox.showerror("Error", "Please draw initial ROI first")
            return

        self.is_cutting_mode = bool(cut)
        self.drawing_roi = True
        self.roi_points = []
        # ROI modified -> re-enable confirm button
        self._enable_confirm()

        # Prepare background preview
        preview = self.current_image.copy()
        if self.roi_mask is not None:
            overlay = np.zeros_like(preview)
            overlay[self.roi_mask] = [0, 255, 0]
            preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        self.update_roi_label(preview)

        # Bind mouse for polygon
        self.roi_canvas.bind('<Button-1>', self.add_roi_point)
        self.roi_canvas.bind('<Motion>', self.update_roi_preview)
        self.roi_canvas.bind('<Double-Button-1>', self.finish_roi_drawing)

        h, w = self.current_image.shape[:2]
        self.roi_scale_factor = min(400/w, 400/h)
        self.display_size = (int(w * self.roi_scale_factor), int(h * self.roi_scale_factor))

    def start_shape_tool(self, shape: str, cut: bool):
        # ROI modified -> re-enable confirm button
        self._enable_confirm()
        """Start a shape-drawing tool (rect/square/circle/triangle), with click-drag to define."""
        if self.current_image is None:
            messagebox.showerror("Error", "Please select input images first")
            return
        if cut and self.roi_mask is None:
            messagebox.showerror("Error", "Please draw initial ROI first")
            return
        self.is_cutting_mode = bool(cut)
        self.current_shape = shape
        self.shape_start = None
        self.roi_canvas.config(cursor='crosshair')

        # Overlay existing ROI if any
        preview = self.current_image.copy()
        if self.roi_mask is not None:
            overlay = np.zeros_like(preview)
            overlay[self.roi_mask] = [0, 255, 0]
            preview = cv2.addWeighted(preview, 1, overlay, 0.3, 0)
        self.update_roi_label(preview)

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
        self.update_roi_display()
        self.confirm_roi_btn.grid()
        self.confirm_roi_btn.configure(state='normal')

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
        # ROI changed -> enable confirm button
        self._enable_confirm()
        
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

    def import_roi_mask(self):
        """Import a mask image file and convert to boolean ROI mask with preview.
        Supports tif/tiff/png/jpg/jpeg/bmp and arbitrary bit depths and channels.
        Handles multi-frame TIFF by using the first frame. Provides resizing/padding
        to match the current image size and optional small-region cleanup.
        """
        if self.current_image is None:
            messagebox.showerror("No Image", "Please load an image before importing a mask.")
            return

        filetypes = [
            ("Mask Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
            ("TIFF", "*.tif *.tiff"),
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("Bitmap", "*.bmp"),
            ("All Files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select ROI Mask Image", filetypes=filetypes)
        if not path:
            return

        # Load image array with a robust fallback chain
        arr = None
        try:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            arr = None
        if arr is None:
            try:
                pil = Image.open(path)
                if getattr(pil, 'n_frames', 1) > 1:
                    pil.seek(0)
                arr = np.array(pil)
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load mask: {e}")
                return

        # Ensure 2D single-channel intensity array for conversion
        try:
            mask_bool = self._convert_to_bool_mask(arr)
        except Exception as e:
            messagebox.showerror("Conversion Error", f"Failed to convert mask to binary: {e}")
            return

        # Adapt to current image size
        H, W = self.current_image.shape[0], self.current_image.shape[1]
        mh, mw = mask_bool.shape
        if (mh, mw) != (H, W):
            # If transposed size matches, offer auto transpose
            if (mh, mw) == (W, H):
                if messagebox.askyesno("Mask Orientation", "Mask appears transposed relative to the image. Transpose it?"):
                    mask_bool = mask_bool.T
                    mh, mw = mask_bool.shape
            if (mh, mw) != (H, W):
                choice_resize = messagebox.askyesno(
                    "Size Mismatch",
                    f"Mask size {mw}x{mh} differs from image {W}x{H}.\n\n"
                    "Yes: Resize mask to image size (NN).\nNo: Pad/Crop at top-left to match.")
                try:
                    mask_bool = self._resize_or_pad_mask(mask_bool, (H, W), resize=choice_resize)
                except Exception as e:
                    messagebox.showerror("Sizing Error", f"Failed to adapt mask size: {e}")
                    return

        # Optionally remove tiny components by prompt
        default_thresh = max(50, int(0.0005 * H * W))
        if messagebox.askyesno("Small Region Cleanup", 
                               f"Remove small connected regions below {default_thresh} pixels?\n(You can change this later via 'Remove Small Regions'.)"):
            try:
                mask_bool = self._remove_small_components(mask_bool, default_thresh)
            except Exception as e:
                messagebox.showwarning("Cleanup Warning", f"Small-region cleanup failed: {e}")

        # Sanity checks on content
        pix = int(mask_bool.sum())
        if pix == 0:
            messagebox.showerror("Empty Mask", "The imported mask is empty after conversion. Please check the image or threshold.")
            return
        coverage = pix / float(H * W)
        if coverage < 1e-4:
            if not messagebox.askyesno("Very Small ROI", f"ROI covers only {coverage*100:.4f}% of the image. Continue?"):
                return

        # Apply and preview
        self.roi_mask = mask_bool.astype(bool)
        self.roi_points = []
        # ROI modified -> re-enable confirm button
        self._enable_confirm()
        self.extract_roi_rectangle()
        self.update_roi_display()
        self.confirm_roi_btn.grid()
        self.confirm_roi_btn.configure(state='normal')

    def invert_roi(self):
        """Invert the current ROI mask and refresh preview."""
        if self.roi_mask is None:
            messagebox.showerror("No ROI", "No ROI to invert. Please draw or import a mask first.")
            return
        self.roi_mask = ~self.roi_mask
        # Keep rectangle consistent
        try:
            self.extract_roi_rectangle()
        except Exception:
            pass
        self.update_roi_display()

    def clean_small_regions(self):
        """Remove small connected components from the current ROI mask."""
        if self.roi_mask is None:
            messagebox.showerror("No ROI", "No ROI to clean. Please draw or import a mask first.")
            return
        H, W = self.roi_mask.shape
        default_thresh = max(50, int(0.0005 * H * W))
        val = simpledialog.askinteger("Remove Small Regions",
                                      f"Remove connected regions smaller than how many pixels?",
                                      initialvalue=default_thresh,
                                      minvalue=1)
        if val is None:
            return
        try:
            cleaned = self._remove_small_components(self.roi_mask.astype(bool), val)
        except Exception as e:
            messagebox.showerror("Cleanup Error", f"Failed to remove small regions: {e}")
            return
        self.roi_mask = cleaned
        try:
            self.extract_roi_rectangle()
        except Exception:
            pass
        self.update_roi_display()

    def _convert_to_bool_mask(self, arr: np.ndarray) -> np.ndarray:
        """Convert an arbitrary image array into a boolean mask.
        Rules:
        - If 4-channel with alpha, prefer alpha as mask if informative.
        - If multi-channel, convert to grayscale via average.
        - If values appear binary (<= 4 unique values), treat > 0 as True.
        - Else, normalize to 8-bit and apply Otsu threshold.
        Returns a 2D boolean array.
        """
        if arr is None:
            raise ValueError("Empty array")
        a = arr
        if a.ndim == 3:
            # Handle alpha channel as mask if present and informative
            if a.shape[2] == 4:
                alpha = a[:, :, 3]
                unique_alpha = np.unique(alpha)
                if len(unique_alpha) > 1:
                    a = alpha
                else:
                    # Convert to grayscale via channel average (robust to channel ordering)
                    a = np.mean(a[:, :, :3], axis=2)
            else:
                # Convert BGR/RGB to grayscale by average to avoid cv2 dtype constraints
                a = np.mean(a, axis=2)
        elif a.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported array shape: {a.shape}")

        # Squeeze to native type for checks
        a = np.asarray(a)

        # If already looks binary / label-like
        uniq = None
        try:
            # For large images, sample uniques efficiently
            if a.size > 5_000_000:
                sample = a[::max(1, a.shape[0]//100), ::max(1, a.shape[1]//100)]
                uniq = np.unique(sample)
            else:
                uniq = np.unique(a)
        except Exception:
            uniq = None
        if uniq is not None and len(uniq) <= 4:
            return (a > 0).astype(bool)

        # Normalize to 8-bit for Otsu thresholding
        a_float = a.astype(np.float32)
        mn, mx = float(np.nanmin(a_float)), float(np.nanmax(a_float))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return (a_float > 0).astype(bool)
        scaled = ((a_float - mn) / (mx - mn) * 255.0).astype(np.uint8)
        # Apply Otsu
        _, th = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (th > 0)

    def _resize_or_pad_mask(self, mask: np.ndarray, target_hw, resize: bool = True) -> np.ndarray:
        """Resize (NN) or top-left pad/crop the mask to target (H, W)."""
        H, W = target_hw
        if resize:
            out = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            return out.astype(bool)
        # Pad or crop from top-left
        mh, mw = mask.shape
        out = np.zeros((H, W), dtype=bool)
        h = min(H, mh)
        w = min(W, mw)
        out[:h, :w] = mask[:h, :w]
        return out

    def _remove_small_components(self, mask: np.ndarray, min_pixels: int) -> np.ndarray:
        """Remove connected components smaller than min_pixels from a boolean mask."""
        bin_u8 = mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(bin_u8, connectivity=8)
        if num_labels <= 1:
            return mask.astype(bool)
        cleaned = np.zeros_like(bin_u8)
        for lbl in range(1, num_labels):
            area = int((labels == lbl).sum())
            if area >= min_pixels:
                cleaned[labels == lbl] = 1
        return cleaned.astype(bool)

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
        # ROI modified -> re-enable confirm button
        self._enable_confirm()
        self.roi_mask = None
        self.roi_rect = None
        self.is_cutting_mode = False
        
        # Reset preview
        if self.current_image is not None:
            self.update_roi_label(self.current_image)
        
        # Reset main preview area [CROP PREVIEW HIDDEN]
        # self.update_preview()
        
        # Hide confirm button
        self.confirm_roi_btn.grid_remove()
        try:
            self.confirm_roi_btn.configure(state="normal")
        except Exception:
            pass
        
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

        # Extract largest rectangle containing ROI and mark confirmed
        self.extract_roi_rectangle()
        self.roi_confirmed = True
        # Refresh info label with current ROI size (overwrite previous text)
        try:
            h, w = self.current_image.shape[:2]
            if self.roi_rect:
                xmin, ymin, xmax, ymax = self.roi_rect
                info_text = f"Image size: {w}x{h}\nNumber of images: {len(self.image_files)}\nROI size: {xmax-xmin}x{ymax-ymin}"
                self.image_info.config(text=info_text)
        except Exception:
            pass
        # Visual hint: overlay a subtle cyan mask
        try:
            preview = self.current_image.copy()
            overlay = np.zeros_like(preview)
            overlay[self.roi_mask] = [0, 180, 255]
            preview = cv2.addWeighted(preview, 1, overlay, 0.35, 0)
            self.update_roi_label(preview)
            self.confirm_roi_btn.configure(state="disabled")
        except Exception:
            pass
    def _draw_scale_ticks(self):
        try:
            self.roi_canvas.delete("scale_ticks")
            if self.current_image is None:
                return
            h, w = self.current_image.shape[:2]
            z = self.zoom_factor
            # choose tick spacing in image pixels
            if z >= 2.0: step = 50
            elif z >= 1.0: step = 100
            else: step = 200
            # top ticks (x)
            for x in range(0, w+1, step):
                cx = int(x * z)
                self.roi_canvas.create_line(cx, 0, cx, 6, fill="#888", tags="scale_ticks")
                if x % (step*2) == 0:
                    self.roi_canvas.create_text(cx+10, 10, anchor="nw", text=f"{x}", fill="#666", tags="scale_ticks")
            # left ticks (y)
            for y in range(0, h+1, step):
                cy = int(y * z)
                self.roi_canvas.create_line(0, cy, 6, cy, fill="#888", tags="scale_ticks")
                if y % (step*2) == 0:
                    self.roi_canvas.create_text(10, cy+10, anchor="nw", text=f"{y}", fill="#666", tags="scale_ticks")
        except Exception:
            pass
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

        # If ROI is confirmed, overlay a stable highlight color onto the ROI area
        try:
            if getattr(self, 'roi_confirmed', False) and self.roi_mask is not None:
                mask_resized = cv2.resize(self.roi_mask.astype(np.uint8), (display_w, display_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                overlay_rgb = np.array([255, 180, 0], dtype=np.uint8)  # orange highlight in RGB
                alpha = 0.35
                resized_masked = resized[mask_resized].astype(np.float32)
                resized[mask_resized] = np.clip(resized_masked * (1.0 - alpha) + overlay_rgb * alpha, 0, 255).astype(np.uint8)
        except Exception:
            pass
        
        # Create PhotoImage
        image_pil = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(image_pil)
        
        # Clear canvas and display new image
        self.roi_canvas.delete("all")
        # Draw base image onto canvas
        try:
            self.roi_canvas.create_image(0, 0, image=self.current_photo, anchor='nw', tags='base_image')
        except Exception:
            pass
        # Draw ROI mask outline if available
        try:
            self.roi_canvas.delete('roi_outline')
            if self.roi_mask is not None:
                cnts, _ = cv2.findContours(self.roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                z = self.zoom_factor
                for c in cnts:
                    if len(c) < 2:
                        continue
                    pts = []
                    for pt in c[:, 0, :]:
                        x, y = pt[0] * z, pt[1] * z
                        pts.extend([x, y])
                    self.roi_canvas.create_line(pts, fill='#00B4FF', width=2, tags='roi_outline')
        except Exception:
            pass
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
        
        # Draw tiles overlay if enabled
        self.draw_tiles_overlay()

        # Update ROI metrics text area (tiling/budget info)
        try:
            self.update_roi_metrics_text()
        except Exception:
            pass
        # Update scroll area
        self.roi_canvas.configure(scrollregion=(0, 0, display_w, display_h))
        # Draw scale ticks after image redraw
        self._draw_scale_ticks()

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
        
        # World coordinate under cursor before zoom
        if x is not None and y is not None:
            wx = x / old_zoom
            wy = y / old_zoom
        else:
            wx = wy = None

        # Update display (updates scrollregion based on new zoom)
        self.update_roi_label(self.current_image)

        # Adjust scroll to keep cursor-centered zoom if mouse position provided
        if wx is not None and wy is not None:
            canvas_width = self.roi_canvas.winfo_width()
            canvas_height = self.roi_canvas.winfo_height()
            display_w = int(self.current_image.shape[1] * self.zoom_factor)
            display_h = int(self.current_image.shape[0] * self.zoom_factor)

            left = wx * self.zoom_factor - canvas_width / 2
            top = wy * self.zoom_factor - canvas_height / 2
            # Clamp to valid range
            left = max(0, min(left, max(0, display_w - canvas_width)))
            top = max(0, min(top, max(0, display_h - canvas_height)))
    def draw_tiles_overlay(self):
        try:
            if not bool(self.show_tiles_overlay.get()):
                self.roi_canvas.delete('tiles_overlay')
                return
            if self.current_image is None or self.roi_mask is None:
                return
            H, W = self.current_image.shape[:2]
            bbox = proc._compute_min_bounding_box(self.roi_mask.astype(bool))
            if bbox is None:
                return
            try:
                Dg = int(float(self.D_global.get()))
            except Exception:
                Dg = 100
            try:
                gt = int(float(self.g_tile.get()))
            except Exception:
                gt = 100
            try:
                r = float(self.overlap_ratio.get())
            except Exception:
                r = 0.10
            pmax = self._parse_pmax()
            pref_sq = bool(self.prefer_square.get())
            xC, yC, wC, hC = proc._expand_and_clamp(bbox, Dg, W, H)
            Tx, Ty = proc._choose_tile_size(wC, hC, p_max_pixels=pmax, prefer_square=pref_sq)
            # Single-shot if (Tx,Ty) equals (wC,hC)
            single_shot = (int(Tx) == int(wC) and int(Ty) == int(hC))
            if single_shot:
                starts_x = [0]
                starts_y = [0]
            else:
                Ex = Tx - 2*gt
                Ey = Ty - 2*gt
                if Ex <= 0 or Ey <= 0:
                    self.roi_canvas.delete('tiles_overlay')
                    return
                Ex_olap = max(1, int(round(r * Ex)))
                Ey_olap = max(1, int(round(r * Ey)))
                stride_x = max(1, Ex - Ex_olap)
                stride_y = max(1, Ey - Ey_olap)
                starts_x = proc._make_starts_with_tail_snap(wC, Tx, stride_x)
                starts_y = proc._make_starts_with_tail_snap(hC, Ty, stride_y)
            z = self.zoom_factor
            # Clear old
            self.roi_canvas.delete('tiles_overlay')
            # Draw C rectangle
            self.roi_canvas.create_rectangle(xC*z, yC*z, (xC+wC)*z, (yC+hC)*z, outline='magenta', width=2, tags='tiles_overlay')
            for sy in starts_y:
                for sx in starts_x:
                    x0 = xC + sx
                    y0 = yC + sy
                    tile_w = min(Tx, wC - sx)
                    tile_h = min(Ty, hC - sy)
                    # Outer tile
                    self.roi_canvas.create_rectangle(x0*z, y0*z, (x0+tile_w)*z, (y0+tile_h)*z, outline='lightblue', width=1, tags='tiles_overlay')
                    # Inner valid area
                    if tile_w > 2*gt and tile_h > 2*gt:
                        ix0 = x0 + gt
                        iy0 = y0 + gt
                        ix1 = x0 + tile_w - gt
                        iy1 = y0 + tile_h - gt
                        self.roi_canvas.create_rectangle(ix0*z, iy0*z, ix1*z, iy1*z, outline='orange', width=1, tags='tiles_overlay')
        except Exception:
            # Be silent on overlay errors to not interrupt main UX
            return

    def _parse_pmax(self):
        """Parse pixel budget string to integer area; returns default on error."""
        try:
            s = (self.p_max_pixels.get() or "").lower().replace(' ', '')
            if '*' in s:
                a,b = s.split('*',1)
                return int(float(a)) * int(float(b))
            if 'x' in s:
                a,b = s.split('x',1)
                return int(float(a)) * int(float(b))
            return int(float(s))
        except Exception:
            return 1100*1100

    def update_roi_metrics_text(self):
        """Compute and display ROI, expanded ROI, tiles, and budget status."""
        if getattr(self, 'roi_metrics_text', None) is None:
            return
        lines = []
        try:
            if self.current_image is None or self.roi_mask is None or self.roi_rect is None:
                lines.append("ROI: not set")
            else:
                H, W = self.current_image.shape[:2]
                xmin, ymin, xmax, ymax = self.roi_rect
                wR = max(0, int(xmax - xmin))
                hR = max(0, int(ymax - ymin))
                areaR = wR * hR
                lines.append(f"ROI: {wR} x {hR} ({areaR} px)")

                # D and g from UI (after preset mapping)
                try:
                    Dg = int(float(self.D_global.get()))
                except Exception:
                    Dg = 100
                try:
                    gt = int(float(self.g_tile.get()))
                except Exception:
                    gt = 100

                # Expanded context C around mask bbox
                bbox = proc._compute_min_bounding_box(self.roi_mask.astype(bool))
                if bbox is None:
                    lines.append("Expanded: none (empty ROI mask)")
                else:
                    xC, yC, wC, hC = proc._expand_and_clamp(bbox, Dg, W, H)
                    areaC = int(wC * hC)
                    lines.append(f"Expanded (D={Dg}): {wC} x {hC} ({areaC} px)")

                    # Pixel budget and tiling estimate
                    pmax = self._parse_pmax()
                    try:
                        r = float(self.overlap_ratio.get())
                    except Exception:
                        r = 0.10
                    pref_sq = bool(self.prefer_square.get())
                    # Choose tile size under budget (same as processing)
                    Tx, Ty = proc._choose_tile_size(wC, hC, p_max_pixels=pmax, prefer_square=pref_sq)
                    single_shot = (int(Tx) == int(wC) and int(Ty) == int(hC))
                    if single_shot:
                        tiles_x = tiles_y = 1
                        tiles_n = 1
                        lines.append(f"Tile size: {Tx} x {Ty}; Tiles: 1 x 1 = 1")
                        lines.append("Budget check: within budget → single tile")
                    else:
                        Ex = max(0, int(Tx - 2*int(gt)))
                        Ey = max(0, int(Ty - 2*int(gt)))
                        if Ex <= 0 or Ey <= 0:
                            lines.append(f"Tiles: g={gt} too large for tile; reduce g or increase budget")
                        else:
                            Ex_olap = max(1, int(round(r * Ex)))
                            Ey_olap = max(1, int(round(r * Ey)))
                            stride_x = max(1, Ex - Ex_olap)
                            stride_y = max(1, Ey - Ey_olap)
                            xs = proc._make_starts_with_tail_snap(wC, Tx, stride_x)
                            ys = proc._make_starts_with_tail_snap(hC, Ty, stride_y)
                            tiles_x = len(xs)
                            tiles_y = len(ys)
                            tiles_n = tiles_x * tiles_y
                            lines.append(f"Tile size: {Tx} x {Ty}; Tiles: {tiles_x} x {tiles_y} = {tiles_n}")

                            # Decision
                            status = "exceeds" if areaC > pmax else "within"
                            if tiles_n <= 1:
                                lines.append("Budget check: within budget → single tile")
                            else:
                                lines.append(f"Budget check: {status} budget → tiled processing")
        except Exception as e:
            lines.append(f"Metrics error: {e}")

        try:
            self.roi_metrics_text.configure(state='normal')
            self.roi_metrics_text.delete('1.0', tk.END)
            self.roi_metrics_text.insert('1.0', "\n".join(lines))
            self.roi_metrics_text.configure(state='disabled')
        except Exception:
            pass
            # Convert to fraction of scrollregion size for moveto
            fx = left / max(1, display_w)
            fy = top / max(1, display_h)
            self.roi_canvas.xview_moveto(fx)
            self.roi_canvas.yview_moveto(fy)

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
        try:
            self.update_displacement_preview()
        except Exception:
            pass
        
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
        # [CROP PREVIEW HIDDEN]
        return

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
        """Create a compact info button that shows a help dialog on click."""
        def _show():
            try:
                msg = self.tooltips.get(tooltip_key, "")
                messagebox.showinfo("Info", msg)
            except Exception:
                pass
        btn = ttk.Button(parent, text="i", width=2, command=_show)
        return btn

    def update_crop_state(self, *args):
        """[CROP SETTINGS HIDDEN] No-op"""
        return

    def apply_theme(self):
        """Apply a modern CustomTkinter theme, fonts, and sizing."""
        try:
            # Appearance adapts to OS theme
            ctk.set_appearance_mode("system")
            # Neutral modern color theme
            ctk.set_default_color_theme("blue")

            # Set readable default fonts across widgets
            try:
                tkfont.nametofont('TkDefaultFont').configure(family='Segoe UI', size=12)
                tkfont.nametofont('TkTextFont').configure(family='Segoe UI', size=12)
                tkfont.nametofont('TkHeadingFont').configure(family='Segoe UI', size=13, weight='bold')
                tkfont.nametofont('TkMenuFont').configure(family='Segoe UI', size=12)
            except Exception:
                pass
        except Exception:
            pass

    def update_max_disp_hint(self):
        """[MAX DISPLACEMENT REMOVED] No-op"""
        return

    def update_smoothing_state(self):
        """Enable/disable sigma controls based on smoothing checkbox, keep slider and entry in sync."""
        # Ensure the sigma help text is readable (override any garbled content)
        try:
            self.sigma_hint.configure(text="Gaussian smoothing sigma in pixels (0.5–5). Higher reduces noise; default 2.0.")
        except Exception:
            pass
        state = 'normal' if self.use_smooth.get() else 'disabled'
        try:
            self.sigma_entry.configure(state=state)
            self.sigma_scale.state(("!disabled",)) if state == 'normal' else self.sigma_scale.state(("disabled",))
            # Sync slider with entry value
            try:
                val = float(self.sigma.get())
            except Exception:
                val = 2.0
            val = max(0.5, min(5.0, val))
            if abs(self.sigma_scale.get() - val) > 1e-6:
                self.sigma_scale.set(val)
        except Exception:
            pass

    def on_sigma_scale_change(self, value):
        """Round and propagate sigma from slider to entry variable."""
        try:
            val = round(float(value), 2)
            self.sigma.set(f"{val:.2f}")
        except Exception:
            pass

    def on_sigma_entry_change(self, event=None):
        """Propagate sigma from entry to slider (with bounds and rounding)."""
        try:
            val = float(self.sigma.get())
        except Exception:
            val = 2.0
        val = max(0.5, min(5.0, val))
        self.sigma.set(f"{val:.2f}")
        try:
            self.sigma_scale.set(val)
        except Exception:
            pass

    def on_preview_scale_change(self, value):
        try:
            val = float(value)
        except Exception:
            val = 0.5
        val = max(0.25, min(1.0, val))
        self.preview_scale.set(f"{val:.2f}")
        self.update_displacement_preview()

    # ----- Responsive split management (preserve ratios on resize, update on drag) -----
    def _on_sash_press(self, event=None):
        self._sash_dragging = True

    def _on_sash_release(self, event=None):
        try:
            total_w = max(1, self.main_paned.winfo_width())
            s0 = self.main_paned.sashpos(0)
            s1 = self.main_paned.sashpos(1)
            r1 = max(0.05, min(0.9, s0 / total_w))
            r2 = max(0.05, min(0.9, (s1 - s0) / total_w))
            r3 = max(0.05, 1.0 - r1 - r2)
            # Normalize small numeric drift
            s = r1 + r2 + r3
            if abs(s - 1.0) > 1e-6:
                r1, r2, r3 = r1 / s, r2 / s, r3 / s
            self._ratios = {'r1': r1, 'r2': r2, 'r3': r3}
        except Exception:
            pass
        self._sash_dragging = False

    def _on_root_resize(self, event=None):
        # Debounce and avoid feedback loops when applying sash positions
        if getattr(self, '_sash_dragging', False) or getattr(self, '_applying_layout', False):
            return
        try:
            if self._resize_after is not None:
                self.root.after_cancel(self._resize_after)
        except Exception:
            pass
        self._resize_after = self.root.after(30, self._apply_split_positions)

    def _apply_split_positions(self):
        self._resize_after = None
        if getattr(self, '_sash_dragging', False):
            return
        self._applying_layout = True
        try:
            total_w = max(1, self.main_paned.winfo_width())
            s0 = int(total_w * self._ratios['r1'])
            s1 = int(total_w * (self._ratios['r1'] + self._ratios['r2']))
            self.main_paned.sashpos(0, s0)
            self.main_paned.sashpos(1, s1)
        except Exception:
            pass
        finally:
            self._applying_layout = False

    def _ensure_roi_cache(self):
        """Cache ROI-invariant data: roi_mask_crop, Xg, Yg, and primary boundary contour."""
        if self.roi_rect is None or self.roi_mask is None:
            self._roi_cache = None
            return None
        xmin, ymin, xmax, ymax = self.roi_rect
        key = (xmin, ymin, xmax, ymax)
        if self._roi_cache and self._roi_cache.get('key') == key:
            return self._roi_cache
        roi_mask_crop = self.roi_mask[ymin:ymax, xmin:xmax]
        h = ymax - ymin
        w = xmax - xmin
        Y_local, X_local = np.mgrid[0:h, 0:w]
        Xg = X_local + xmin
        Yg = Y_local + ymin
        # Boundary contour
        contour = None
        try:
            import cv2
            cnts, _ = cv2.findContours(roi_mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(cnts) > 0:
                contour = max(cnts, key=cv2.contourArea)
        except Exception:
            contour = None
        self._roi_cache = {
            'key': key,
            'roi_mask_crop': roi_mask_crop,
            'Xg': Xg,
            'Yg': Yg,
            'contour': contour
        }
        return self._roi_cache

    def auto_set_colorbar_ranges(self):
        """Compute colorbar ranges from the first displacement result and update entries."""
        try:
            if not self.displacement_results:
                messagebox.showinfo("Info", "Run processing first to compute auto ranges.")
                return
            first = self.displacement_results[0]
            if isinstance(first, str):
                disp = np.load(first)
            else:
                disp = first
            u = disp[:, :, 0]
            v = disp[:, :, 1]
            umin, umax = np.nanmin(u), np.nanmax(u)
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            self.colorbar_u_min.set(f"{umin:.3f}")
            self.colorbar_u_max.set(f"{umax:.3f}")
            self.colorbar_v_min.set(f"{vmin:.3f}")
            self.colorbar_v_max.set(f"{vmax:.3f}")
            self.use_fixed_colorbar.set(True)
            self.update_displacement_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set auto ranges: {e}")

    def set_fixed_colorbar_from_frame(self):
        """Set U,V fixed ranges based on displacement stats of a chosen frame number."""
        try:
            if not self.displacement_results:
                messagebox.showinfo("Info", "Run processing first to compute ranges.")
                return
            try:
                idx = max(1, int(float(self.fixed_range_frame.get() or '1')))
            except Exception:
                idx = 1
            total = len(self.displacement_results)
            if idx > total:
                idx = total
            item = self.displacement_results[idx-1]
            disp = np.load(item) if isinstance(item, str) else item
            u = disp[:, :, 0]
            v = disp[:, :, 1]
            umin, umax = np.nanmin(u), np.nanmax(u)
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            self.colorbar_u_min.set(f"{umin:.3f}")
            self.colorbar_u_max.set(f"{umax:.3f}")
            self.colorbar_v_min.set(f"{vmin:.3f}")
            self.colorbar_v_max.set(f"{vmax:.3f}")
            self.use_fixed_colorbar.set(True)
            self.update_displacement_preview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set fixed range: {e}")

    def reset_visualization_settings(self):
        """Reset colormap and ranges to defaults and refresh preview."""
        self.colormap.set('viridis')
        self.use_fixed_colorbar.set(False)
        self.colorbar_u_min.set("-1")
        self.colorbar_u_max.set("1")
        self.colorbar_v_min.set("-1")
        self.colorbar_v_max.set("1")
        self.update_displacement_preview()

    def update_crop_size(self):
        """Update crop size entries with image dimensions"""
        if hasattr(self, 'current_image') and self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.crop_size_w.set(str(w))
            self.crop_size_h.set(str(h))


def main():
    """Main entry point for the RAFT-DIC GUI application."""
    root = ctk.CTk()
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







