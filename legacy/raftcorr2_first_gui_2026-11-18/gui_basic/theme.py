"""Light (default) and dark themes; the palettes match pyALDIC so the applications look alike."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    BG_WINDOW: str
    BG_SIDEBAR: str
    BG_PANEL: str
    BG_INPUT: str
    BG_HOVER: str
    BG_CANVAS: str
    ACCENT: str
    ACCENT_HOVER: str
    ACCENT_PRESSED: str
    TEXT: str
    TEXT_SECONDARY: str
    TEXT_MUTED: str
    BORDER: str
    DANGER: str
    SUCCESS: str
    WARNING: str
    SELECTION: str
    OVERLAY_BG: str
    is_dark: bool


LIGHT = Palette(
    name="light", BG_WINDOW="#ffffff", BG_SIDEBAR="#f6f7f9", BG_PANEL="#eef0f4", BG_INPUT="#ffffff",
    BG_HOVER="#e8eaf6", BG_CANVAS="#e5e7eb", ACCENT="#4f46e5", ACCENT_HOVER="#6366f1", ACCENT_PRESSED="#4338ca",
    TEXT="#111827", TEXT_SECONDARY="#4b5563", TEXT_MUTED="#6b7280", BORDER="#d1d5db", DANGER="#dc2626",
    SUCCESS="#16a34a", WARNING="#b45309", SELECTION="#e0e7ff", OVERLAY_BG="rgba(255, 255, 255, 230)", is_dark=False,
)
DARK = Palette(
    name="dark", BG_WINDOW="#0b0f1a", BG_SIDEBAR="#0f1322", BG_PANEL="#141929", BG_INPUT="#1a1f33",
    BG_HOVER="#1e2440", BG_CANVAS="#0c0d12", ACCENT="#6366f1", ACCENT_HOVER="#818cf8", ACCENT_PRESSED="#4f46e5",
    TEXT="#e2e8f0", TEXT_SECONDARY="#94a3b8", TEXT_MUTED="#64748b", BORDER="#1e293b", DANGER="#ef4444",
    SUCCESS="#22c55e", WARNING="#eab308", SELECTION="#312e81", OVERLAY_BG="rgba(16, 20, 24, 215)", is_dark=True,
)
THEMES: dict[str, Palette] = {"light": LIGHT, "dark": DARK}
DEFAULT_THEME = "light"

_current = {"palette": LIGHT}


def current() -> Palette:
    return _current["palette"]


def set_current(name: str) -> Palette:
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; available: {sorted(THEMES)}")
    _current["palette"] = THEMES[name]
    return THEMES[name]


def stylesheet(p: Palette | None = None) -> str:
    c = p or current()
    return f"""
QWidget {{ color: {c.TEXT}; }}
QMainWindow, QDialog {{ background: {c.BG_WINDOW}; }}
QWidget#sidebar, QScrollArea#sidebarScroll, QWidget#sidebarContents {{ background: {c.BG_SIDEBAR}; }}
QWidget#sidebar {{ border-right: 1px solid {c.BORDER}; }}
QWidget#rightSidebar {{ border-left: 1px solid {c.BORDER}; }}
QScrollArea {{ border: none; }}
QFrame#section {{
    background: {c.BG_PANEL}; border: 1px solid {c.BORDER}; border-radius: 6px;
}}
QToolButton#sectionHeader {{
    background: transparent; border: none; font-weight: 600; font-size: 12px;
    color: {c.TEXT_SECONDARY}; text-align: left; padding: 6px 4px;
}}
QToolButton#sectionHeader:hover {{ color: {c.ACCENT}; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {c.TEXT_MUTED}; font-size: 11px; }}
QLabel[role="secondary"] {{ color: {c.TEXT_SECONDARY}; }}
QLabel[role="heading"] {{ font-weight: 600; font-size: 14px; }}
QLabel[role="badge"] {{
    background: {c.SELECTION}; color: {c.ACCENT}; border-radius: 8px; padding: 1px 8px; font-size: 11px;
}}
QLabel[role="warning"] {{ color: {c.WARNING}; }}
QLabel[role="error"] {{ color: {c.DANGER}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {c.BG_INPUT}; border: 1px solid {c.BORDER}; border-radius: 4px; padding: 3px 6px;
    min-height: 20px; selection-background-color: {c.ACCENT}; selection-color: white;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {c.ACCENT}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {c.TEXT_MUTED}; }}
QComboBox QAbstractItemView {{
    background: {c.BG_INPUT}; border: 1px solid {c.BORDER}; selection-background-color: {c.SELECTION};
    selection-color: {c.TEXT};
}}
QPushButton {{
    background: {c.BG_INPUT}; border: 1px solid {c.BORDER}; border-radius: 5px; padding: 5px 12px;
}}
QPushButton:hover {{ background: {c.BG_HOVER}; border-color: {c.ACCENT}; }}
QPushButton:pressed {{ background: {c.SELECTION}; }}
QPushButton:disabled {{ color: {c.TEXT_MUTED}; border-color: {c.BORDER}; background: {c.BG_PANEL}; }}
QPushButton:checked {{ background: {c.SELECTION}; border-color: {c.ACCENT}; color: {c.ACCENT}; }}
QPushButton[role="primary"] {{
    background: {c.ACCENT}; color: white; border: 1px solid {c.ACCENT}; font-weight: 600; padding: 7px 14px;
}}
QPushButton[role="primary"]:hover {{ background: {c.ACCENT_HOVER}; }}
QPushButton[role="primary"]:pressed {{ background: {c.ACCENT_PRESSED}; }}
QPushButton[role="primary"]:disabled {{ background: {c.BG_PANEL}; color: {c.TEXT_MUTED}; border-color: {c.BORDER}; }}
QPushButton[role="danger"] {{ color: {c.DANGER}; }}
QPushButton[role="danger"]:hover {{ border-color: {c.DANGER}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 3px; }}
QToolButton:hover {{ background: {c.BG_HOVER}; border-color: {c.BORDER}; }}
QToolButton:checked {{ background: {c.SELECTION}; border-color: {c.ACCENT}; color: {c.ACCENT}; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QProgressBar {{
    background: {c.BG_INPUT}; border: 1px solid {c.BORDER}; border-radius: 4px; text-align: center; height: 16px;
}}
QProgressBar::chunk {{ background: {c.ACCENT}; border-radius: 3px; }}
QSlider::groove:horizontal {{ height: 4px; background: {c.BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c.ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {c.BG_INPUT}; border: 2px solid {c.ACCENT}; width: 12px; height: 12px; margin: -6px 0;
    border-radius: 8px;
}}
QListWidget, QTableWidget, QPlainTextEdit, QTextEdit {{
    background: {c.BG_INPUT}; border: 1px solid {c.BORDER}; border-radius: 4px;
    selection-background-color: {c.SELECTION}; selection-color: {c.TEXT};
}}
QHeaderView::section {{ background: {c.BG_PANEL}; border: none; border-bottom: 1px solid {c.BORDER}; padding: 3px; }}
QTabWidget::pane {{ border: 1px solid {c.BORDER}; border-radius: 4px; }}
QTabBar::tab {{ background: transparent; padding: 6px 12px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ border-bottom-color: {c.ACCENT}; color: {c.ACCENT}; }}
QMenuBar {{ background: {c.BG_SIDEBAR}; border-bottom: 1px solid {c.BORDER}; }}
QMenuBar::item:selected {{ background: {c.BG_HOVER}; }}
QMenu {{ background: {c.BG_INPUT}; border: 1px solid {c.BORDER}; }}
QMenu::item:selected {{ background: {c.SELECTION}; }}
QStatusBar {{ background: {c.BG_SIDEBAR}; border-top: 1px solid {c.BORDER}; color: {c.TEXT_SECONDARY}; }}
QDockWidget::title {{ background: {c.BG_SIDEBAR}; padding: 4px; border-bottom: 1px solid {c.BORDER}; }}
QSplitter::handle {{ background: {c.BORDER}; }}
QToolTip {{ background: {c.BG_INPUT}; color: {c.TEXT}; border: 1px solid {c.BORDER}; padding: 4px; }}
QGraphicsView {{ background: {c.BG_CANVAS}; border: none; }}
QScrollBar:vertical {{ width: 10px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {c.BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: {c.BORDER}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""
