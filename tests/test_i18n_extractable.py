"""Every interface string must be visible to lupdate, or it stays English in every language.

lupdate reads literal arguments only: ``self.tr("...")`` and ``QCoreApplication.translate("Ctx",
"...")``. A helper such as ``def _tr(text): return QCoreApplication.translate("Ctx", text)`` hides
every string passed to it (45 strings were never translated that way). The only calls allowed to pass
a variable are the lookups below, whose strings are declared elsewhere (``QT_TRANSLATE_NOOP`` or the
label's own widget).
"""

from __future__ import annotations

import ast
from pathlib import Path

GUI = Path(__file__).resolve().parents[1] / "src" / "raftcorr" / "gui"

ALLOWED = {
    ("controllers/pipeline_messages.py", "gui_label"),     # a widget label, translated by that widget
    ("dialogs/export_widgets.py", "field_label"),          # QT_TRANSLATE_NOOP("ExportDialog", ...)
    ("widgets/strain_field_selector.py", "field_label"),   # QT_TRANSLATE_NOOP("StrainFieldSelector", ...)
}


def _hidden_translate_calls() -> list[str]:
    found = []
    for path in sorted(GUI.rglob("*.py")):
        rel = path.relative_to(GUI).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or (rel, fn.name) in ALLOWED:
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "translate" and len(node.args) >= 2
                        and not all(isinstance(a, ast.Constant) for a in node.args[:2])):
                    found.append(f"{rel}:{node.lineno} in {fn.name}()")
    return found


def test_no_helper_hides_strings_from_lupdate():
    assert _hidden_translate_calls() == []


def test_self_tr_takes_literal_text():
    bad = []
    for path in sorted(GUI.rglob("*.py")):
        rel = path.relative_to(GUI).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "tr"
                    and node.args and not isinstance(node.args[0], ast.Constant)):
                bad.append(f"{rel}:{node.lineno}")
    assert bad == []
