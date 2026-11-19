"""Translated text the pipeline controller shows about a run.

``PipelineController`` is not a ``QObject``, and lupdate extracts a
counted message as a plural only from ``self.tr(source, "", n)`` on a
``QObject``: ``QCoreApplication.translate(context, source, "", n)`` comes
out as a plain message with one form, and German, French and Spanish then
show the English source for n >= 2. So the run's sentences live here, on
:class:`PipelineMessages` (translation context ``PipelineMessages``): the
console lines, the dialogs of a run that cannot start or that failed, and
the confirmation asked at *Run* when some reference frames have no Region
of Interest of their own. The Starting Points errors are in
``seed_error_messages``.

A sentence that names a control gets the control's label through a ``%N``
placeholder, translated by the widget's own context (:func:`gui_label`),
so the message names it exactly as the user sees it.

Pure: no widget, no signal; each function returns text and changes
nothing. Text is looked up when it is shown, in the language then active.
Frame numbers are shown 1-based, like everywhere else in the GUI.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QCoreApplication, QObject

from raftcorr.i18n import tr_args
from raftcorr.gui.support.locale_format import format_number

# Frames listed by number in the confirmation; the rest are counted.
MAX_LISTED_FRAMES = 10
# Rich-text paragraph break between the confirmation's sentences.
_PARAGRAPH = "<br><br>"
# Separator of the listed frame numbers (data, not a sentence).
_LIST_SEPARATOR = ", "

# GUI labels the messages name: (translation context, source) of the widget
# that shows them. tests/test_gui/test_pipeline_messages_i18n.py checks that
# each is still extracted from its widget.
LABEL_IMAGES = ("LeftSidebar", "IMAGES")
LABEL_ROI_SECTION = ("LeftSidebar", "REGION OF INTEREST")
LABEL_INITIAL_GUESS = ("LeftSidebar", "INITIAL GUESS")
LABEL_PARAMETERS = ("LeftSidebar", "PARAMETERS")
LABEL_PLACE_POINTS = ("InitGuessWidget", "Place Starting Points")
LABEL_AUTO_PLACE = ("InitGuessWidget", "Auto-place")
LABEL_CLEAR_POINTS = ("InitGuessWidget", "Clear")
LABEL_FFT = ("InitGuessWidget", "FFT (cross-correlation)")
LABEL_SUBSET_SIZE = ("ParamPanel", "Subset Size")
LABEL_SUBSET_STEP = ("ParamPanel", "Subset Step")
LABEL_BATCH_IMPORT = ("ROIToolbar", "Batch Import")
LABEL_ADVANCED = ("LeftSidebar", "ADVANCED")
LABEL_GPU_MEMORY = ("AdvancedPanel", "GPU Memory")


def gui_label(label: tuple[str, str]) -> str:
    """``label`` as its widget shows it in the active language."""
    context, source = label
    return QCoreApplication.translate(context, source)


class PipelineMessages(QObject):
    """The sentences, in the ``PipelineMessages`` translation context.

    Pass every count positionally as a plain name, ``self.tr(source, "",
    n)``; lupdate does not extract other spellings as plurals
    (``tests/test_i18n_plural_calls.py``). Keep this class unsubclassed:
    ``self.tr`` looks up the runtime class name, and a subclass would look
    in a context that holds no translations.
    """

    def custom_roi_frames(self, n: int) -> str:
        """Console line: ``n`` frames besides frame 1 have their own mask."""
        return self.tr("  %n frame(s) with custom ROI masks", "", n)

    def results_received(self, n: int) -> str:
        """Console line: the run returned results for ``n`` frames."""
        return self.tr("Results received: %n frame(s)", "", n)

    def run_parameters(self, n: int, dump: str) -> str:
        """Console line: ``n`` images, then the ``key=value`` dump."""
        return tr_args(self.tr("  %n image(s), %1", "", n), dump)

    def roi_range(self, dump: str) -> str:
        return tr_args(self.tr("  ROI range: %1"), dump)

    def refinement(self, dump: str) -> str:
        return tr_args(self.tr("  Refinement: %1"), dump)

    def error_line(self, detail: str) -> str:
        """Console line of a failed run; ``detail`` is the exception."""
        return tr_args(self.tr("Error: %1"), detail)

    def run_canceled(self) -> str:
        return self.tr(
            "Run canceled. Give the listed reference frames a Region of "
            "Interest of their own, or run again and continue with frame "
            "1's region.")

    # -- Tracking quality (after a run) -----------------------------------------

    def quality_title(self) -> str:
        return self.tr("Tracking Quality Warning")

    def quality_summary(self) -> str:
        return self.tr("Part of the sequence may not be tracked correctly.")

    def quality_finding(self, first: int, now: float, before: float) -> str:
        """``first`` is the 1-based frame of the jump; residuals in grey levels."""
        return tr_args(self.tr(
            "The tracking quality drops from frame %1 on: the intensity residual rises to %2 grey "
            "levels, against %3 before."), first, f"{now:.1f}", f"{before:.1f}")

    def quality_advice(self, accumulative: bool) -> str:
        if accumulative:
            return self.tr(
                "The displacement from frame 1 may exceed the reach of the model. Try incremental "
                "tracking, or a model with a larger s.")
        return self.tr(
            "The step between reference frames may exceed the reach of the model. Use more "
            "reference frames, or a model with a larger s.")

    def quality_frames(self, listed: str) -> str:
        return tr_args(self.tr("Frames with a high residual: %1"), listed)

    def quality_residual_hint(self) -> str:
        return self.tr("Show the Residual field to see where the images do not match after the "
                       "displacement is applied.")

    # -- Dialogs: a run that cannot start ------------------------------------

    def cannot_start_title(self) -> str:
        return self.tr("Cannot start analysis")

    def too_few_images(self) -> str:
        return tr_args(self.tr(
            "Need at least 2 images. Load an image folder in the %1 section "
            "before running DIC."), gui_label(LABEL_IMAGES))

    def no_roi(self) -> str:
        return tr_args(self.tr(
            "Define a Region of Interest first: draw or import one for "
            "frame 1 in the %1 section."), gui_label(LABEL_ROI_SECTION))

    def empty_roi(self) -> str:
        return tr_args(self.tr(
            "The Region of Interest of frame 1 is empty (no pixels "
            "selected). Draw it again in the %1 section."),
            gui_label(LABEL_ROI_SECTION))

    def roi_too_small_title(self) -> str:
        return self.tr("Region of Interest too small")

    def roi_too_small(self, width: int, height: int, step: int) -> str:
        """``width`` x ``height`` px is less than two Subset Steps."""
        need = format_number(2 * step)
        return tr_args(self.tr(
            "The Region of Interest is %1×%2 px, but the current %3 (%4) "
            "needs at least %5×%5 px. Enlarge the Region of Interest or "
            "reduce %3."), format_number(width), format_number(height),
            gui_label(LABEL_SUBSET_STEP), format_number(step), need)

    def points_required_title(self) -> str:
        return self.tr("Starting Points required")

    def points_required(self) -> str:
        return tr_args(self.tr(
            "Place at least one Starting Point on the canvas before "
            "running: click “%1” or “%2” in the %3 section."),
            gui_label(LABEL_PLACE_POINTS), gui_label(LABEL_AUTO_PLACE),
            gui_label(LABEL_INITIAL_GUESS))

    def invalid_points_title(self) -> str:
        return self.tr("Invalid Starting Points")

    def invalid_points(self) -> str:
        return tr_args(self.tr(
            "One or more Starting Points do not lie in any part of the "
            "Region of Interest. Click “%1” in the %2 section, then place "
            "the Starting Points again."),
            gui_label(LABEL_CLEAR_POINTS), gui_label(LABEL_INITIAL_GUESS))

    # -- Dialogs: a run that failed ------------------------------------------

    def analysis_failed_title(self) -> str:
        return self.tr("DIC analysis failed")

    def failed_to_start_title(self) -> str:
        return self.tr("Failed to start analysis")

    def failed_to_process_title(self) -> str:
        return self.tr("Failed to process results")

    def runtime_error(self, detail: str) -> str:
        """``detail``: the exception text, shown as it is."""
        return tr_args(self.tr(
            "The analysis stopped with a runtime error:\n\n%1\n\n"
            "See the console log for the full traceback."), detail)

    def see_traceback(self, detail: str) -> str:
        """``detail``: the exception type and text, shown as they are."""
        return tr_args(self.tr(
            "%1\n\nSee the console log for the full traceback."), detail)

    def out_of_memory(self) -> str:
        """Advice above the text of a GPU out-of-memory error: smaller windows fit."""
        return tr_args(self.tr(
            "The GPU ran out of memory. Close other GPU programs, choose a model with a "
            "larger s, or lower %1 in %2."),
            gui_label(LABEL_GPU_MEMORY), gui_label(LABEL_ADVANCED))

    def memory_budget_too_small(self) -> str:
        """Advice above the engine's "the memory budget allows windows of about N x N px"
        error: the budget is below the model's smallest window, so lowering it cannot help."""
        return tr_args(self.tr(
            "There is not enough free GPU memory for the smallest window this model accepts. "
            "Close other GPU programs, or raise %1 in %2 if it is set low."),
            gui_label(LABEL_GPU_MEMORY), gui_label(LABEL_ADVANCED))

    # -- Run-time confirmation: reference frames without their own region --

    def missing_roi_title(self) -> str:
        return self.tr("Incomplete Region of Interest Coverage")

    def missing_roi_frames(self, n: int) -> str:
        """``n`` reference frames have no region; ``%1`` lists them."""
        return self.tr(
            "This incremental run has %n reference frame(s) with no Region "
            "of Interest of their own: frame(s) %1.", "", n)

    def frames_not_listed(self, n: int) -> str:
        """A frame list cut short: ``%1`` the frames listed, ``n`` the rest."""
        return self.tr(
            "%1 and %n more",
            "List of frame numbers cut short: %1 = the frame numbers shown, "
            "%n = how many more frames there are", n)

    def missing_roi_effect(self) -> str:
        return self.tr(
            "A reference frame without a Region of Interest of its own "
            "reuses frame 1's region unchanged: nothing moves the region "
            "with the specimen. Where the specimen has moved out of that "
            "region, the part outside it is not measured from such a frame, "
            "and the nodes there stop being tracked or have their "
            "displacement copied from the nearest measured node. The run "
            "reports these nodes when it finishes.")

    def missing_roi_advice(self) -> str:
        return self.tr(
            "To measure the whole specimen, give each reference frame its "
            "own Region of Interest that covers the specimen in that frame "
            "(Batch Import can assign Regions of Interest to many frames at "
            "once). A reference frame with a region of its own is measured "
            "on a grid that covers that region.")

    def missing_roi_question(self) -> str:
        return self.tr("Continue with frame 1's region?")


def memory_advice(exc: BaseException) -> str | None:
    """Translated advice for a run that failed for lack of GPU memory, or None for other errors.

    The engine's ``MemoryError`` from tiling (``raftcorr.engine.tiling.plan_tiles``) says "the
    memory budget allows windows of about N x N px"; a CUDA out-of-memory error says "out of
    memory". Both are matched on the engine's English text, which is never translated.
    """
    text = str(exc).lower()
    if isinstance(exc, MemoryError) and "memory budget" in text:
        return PipelineMessages().memory_budget_too_small()
    if "out of memory" in text:
        return PipelineMessages().out_of_memory()
    return None


def missing_roi_dialog(missing: Iterable[int]) -> tuple[str, str]:
    """``(title, text)`` of the confirmation asked at *Run* in incremental
    mode when the reference frames ``missing`` (0-based) have no Region of
    Interest of their own. ``text`` is rich text: one paragraph per
    sentence group, the closing question in bold."""
    m = PipelineMessages()
    frames = [f + 1 for f in sorted(missing)]
    n_missing = len(frames)
    text = _PARAGRAPH.join((
        tr_args(m.missing_roi_frames(n_missing), _frame_list(m, frames)),
        m.missing_roi_effect(),
        m.missing_roi_advice(),
        f"<b>{m.missing_roi_question()}</b>",
    ))
    return m.missing_roi_title(), text


def tracking_quality_warning(residual, done, accumulative: bool) -> tuple[str, str, str, str] | None:
    """``(title, text, informative, details)`` of the residual-jump warning, or None.

    The finding and the advice are shown; the flagged frames (1-based) and the hint about the
    Residual field sit under Show Details.
    """
    import numpy as np

    from raftcorr.post.quality import frame_residuals, suspect_frames

    if residual is None:
        return None
    medians = frame_residuals(residual, done)
    flagged = suspect_frames(medians)
    if not flagged:
        return None
    first = flagged[0]
    before = medians[[k for k in range(1, first) if np.isfinite(medians[k])]]
    m = PipelineMessages()
    informative = "\n\n".join((
        m.quality_finding(first + 1, float(medians[first]), float(before.max()) if before.size else float("nan")),
        m.quality_advice(accumulative),
    ))
    details = "\n\n".join((m.quality_frames(_frame_list(m, [k + 1 for k in flagged])),
                             m.quality_residual_hint()))
    return m.quality_title(), m.quality_summary(), informative, details


def _frame_list(m: PipelineMessages, frames: list[int]) -> str:
    """The first ``MAX_LISTED_FRAMES`` frame numbers, then how many more."""
    listed = _LIST_SEPARATOR.join(str(f) for f in frames[:MAX_LISTED_FRAMES])
    n_more = len(frames) - MAX_LISTED_FRAMES
    if n_more <= 0:
        return listed
    return tr_args(m.frames_not_listed(n_more), listed)
