import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

# ---------------------------------------------------------------------------
# CustomTkinter integration shims (visual-only refactor, logic unchanged)
# ---------------------------------------------------------------------------

# We introduce CTk-based replacements while preserving names and callbacks.
# A CTk-based collapsible container and a lightweight ttk adapter are provided
# to modernize the appearance without touching the functional logic.

class CTkCollapsibleFrame(ctk.CTkFrame):
    """
    A collapsible frame with header and content area (CTk-styled).
    Mimics the interface of the original CollapsibleFrame so no logic changes needed.
    """
    def __init__(self, parent, text="", **kwargs):
        super().__init__(parent, **kwargs)
        
        # Header
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew")
        self._header = self.header_frame # Alias for internal consistency if needed, or just use header_frame
        
        # Internal container for toggle and label (to allow external grid usage in header_frame)
        self._header_internal = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self._header_internal.grid(row=0, column=0, sticky="w")
        
        # Toggle button
        self._toggle_btn = ctk.CTkButton(self._header_internal, text="▼", width=20, height=20, 
                                         fg_color="transparent", text_color=("gray10", "gray90"),
                                         hover_color=("gray70", "gray30"),
                                         command=self.toggle)
        self._toggle_btn.pack(side="left", padx=5)
        
        # Title
        self._label = ctk.CTkLabel(self._header_internal, text=text, font=("Roboto Medium", -12))
        self._label.pack(side="left", padx=5)
        
        # Content placeholder
        self._content = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._content.grid(row=1, column=0, sticky="nsew", padx=6, pady=(2,6))
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self._collapsed = False
        
    def toggle(self):
        if self._collapsed:
            self._content.grid(row=1, column=0, sticky="nsew", padx=6, pady=(2,6))
            self._toggle_btn.configure(text="▼")
            self._collapsed = False
        else:
            self._content.grid_forget()
            self._toggle_btn.configure(text="▶")
            self._collapsed = True
            
    def get_content_frame(self):
        return self._content


class _ConfigAliasMixin:
    """Mixin to alias 'config' to 'configure' for compatibility with tk code."""
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
