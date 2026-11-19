"""A left-button drag pans the image in the main window and in the strain window.

Before, only the middle button panned: a left drag in the default mode did nothing. Drawing tools
keep the left button for their shapes, and the Analysis tab keeps it for placing probes.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QPixmap  # noqa: E402
from PySide6.QtWidgets import QGraphicsScene  # noqa: E402

LEFT = Qt.MouseButton.LeftButton
NONE = Qt.MouseButton.NoButton


def _event(kind, pos: QPointF, button, buttons) -> QMouseEvent:
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)


def _drag(view, start: QPointF, end: QPointF, button=LEFT) -> None:
    view.mousePressEvent(_event(QMouseEvent.Type.MouseButtonPress, start, button, button))
    view.mouseMoveEvent(_event(QMouseEvent.Type.MouseMove, end, NONE, button))
    view.mouseReleaseEvent(_event(QMouseEvent.Type.MouseButtonRelease, end, button, NONE))


def _zoomed(view, qtbot):
    """The view showing a large picture zoomed in, scrolled to the middle."""
    qtbot.addWidget(view)
    scene = view.scene() or QGraphicsScene(view)
    view.setScene(scene)
    scene.addPixmap(QPixmap(2000, 2000))
    view.resize(400, 300)
    view.show()
    view.scale(4.0, 4.0)
    for bar in (view.horizontalScrollBar(), view.verticalScrollBar()):
        bar.setValue((bar.minimum() + bar.maximum()) // 2)
    return view


def _scroll(view) -> tuple[int, int]:
    return view.horizontalScrollBar().value(), view.verticalScrollBar().value()


def test_left_drag_pans_the_main_canvas_in_the_default_mode(qtbot):
    from raftcorr.gui.panels.canvas_area import ImageCanvas

    view = _zoomed(ImageCanvas(), qtbot)
    assert view._current_tool == "select"
    x0, y0 = _scroll(view)
    _drag(view, QPointF(200, 150), QPointF(150, 110))     # drag left and up: the view moves right, down
    assert _scroll(view) == (x0 + 50, y0 + 40)
    assert not view._panning


def test_left_drag_still_draws_a_shape_with_a_drawing_tool(qtbot):
    from raftcorr.gui.panels.canvas_area import ImageCanvas

    view = _zoomed(ImageCanvas(), qtbot)
    view.set_tool("rect")
    before = _scroll(view)
    view.mousePressEvent(_event(QMouseEvent.Type.MouseButtonPress, QPointF(200, 150), LEFT, LEFT))
    view.mouseMoveEvent(_event(QMouseEvent.Type.MouseMove, QPointF(150, 110), NONE, LEFT))
    assert _scroll(view) == before and not view._panning
    assert view._draw_state is not None                    # the rectangle is being drawn


def test_left_drag_pans_the_strain_canvas(qtbot):
    from raftcorr.gui.panels.strain_canvas import StrainCanvas

    view = _zoomed(StrainCanvas(), qtbot)
    x0, y0 = _scroll(view)
    _drag(view, QPointF(200, 150), QPointF(150, 110))
    assert _scroll(view) == (x0 + 50, y0 + 40)
    assert not view._panning


def test_the_analysis_canvas_keeps_the_left_button_for_probes(qtbot):
    from raftcorr.gui.panels.probe_canvas import ProbeCanvas
    from raftcorr.gui.panels.strain_canvas import StrainCanvas

    assert ProbeCanvas.mousePressEvent is not StrainCanvas.mousePressEvent
