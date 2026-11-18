"""The translation catalogs contain translations, not fragments of them.

Ported from pyALDIC (tests/test_i18n_translation_sanity.py). Coverage and
extract-drift checks count a message as done once it has any finished
translation. pyALDIC once shipped seventy messages with a single character --
the last character of the intended text, because they were filed in the plural
table and ``forms[-1]`` of a string is its last letter.

Guards: the fill tables must be well formed (the cause), no catalog may hold a
one-character translation of a word (the symptom), every translation keeps its
source's placeholders and HTML tags, and no ``tr()`` call is hidden from
lupdate (inside an f-string, or behind an alias).
"""

from __future__ import annotations

import ast
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TS_DIR = ROOT / "src" / "raftcorr" / "i18n" / "source"
GUI_DIR = ROOT / "src" / "raftcorr" / "gui"

#: Latin-script locales: a word never translates to one character.
_LATIN = ("de", "fr", "es")
#: ...except English function words whose translation genuinely is one
#: letter: "to" is "à" in French and "a" in Spanish, as in "de 3 à 7".
#: Allowed by SOURCE, never by translation: the corruption this guards against
#: turned "Line" into "a" (the last letter of "Línea").
_FUNCTION_WORDS = frozenset({"to", "and", "or"})
#: CJK locales legitimately use one character for a short word ("点" for
#: "Point"); only a long source rendered as one character is implausible.
_CJK = ("zh_CN", "zh_TW", "ja", "ko")
_CJK_MIN_SOURCE = 8

_PLACEHOLDER = re.compile(r"%(?:\d+|n)")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\s*/?>")


def test_translation_tables_are_well_formed():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import fill_translations
    finally:
        sys.path.pop(0)
    fill_translations.check_tables()


def _finished(lang: str) -> Iterator[tuple[str, str]]:
    tree = ET.parse(TS_DIR / f"raftcorr_{lang}.ts")
    for message in tree.getroot().iter("message"):
        source = message.findtext("source") or ""
        tr = message.find("translation")
        if tr is None or tr.get("type") in ("unfinished", "obsolete", "vanished"):
            continue
        forms = [f.text or "" for f in tr.findall("numerusform")] or [tr.text or ""]
        for text in forms:
            yield source, text


@pytest.mark.parametrize("lang", _LATIN + _CJK)
def test_no_word_is_translated_to_a_single_character(lang):
    threshold = 2 if lang in _LATIN else _CJK_MIN_SOURCE
    bad = [
        (src, text) for src, text in _finished(lang)
        if len(text.strip()) == 1 and len(src.strip()) >= threshold
        and src.strip().lower() not in _FUNCTION_WORDS
    ]
    assert not bad, (
        f"{len(bad)} {lang} translation(s) are a single character, e.g. "
        + ", ".join(f"{s!r} -> {t!r}" for s, t in bad[:5])
    )


def _markers(text: str) -> tuple[list[str], list[str]]:
    return sorted(_PLACEHOLDER.findall(text)), sorted(_TAG.findall(text))


@pytest.mark.parametrize("lang", _CJK + _LATIN)
def test_translations_keep_placeholders_and_tags(lang):
    """A lost %1 drops a value from the message; a lost <b> breaks rich text."""
    bad = [(src, text) for src, text in _finished(lang) if _markers(text) != _markers(src)]
    assert not bad, (
        f"{len(bad)} {lang} translation(s) change placeholders or tags, e.g. "
        + ", ".join(f"{s!r} -> {t!r}" for s, t in bad[:3])
    )


def test_no_translation_call_hides_inside_an_f_string():
    """lupdate does not look inside f-string expressions.

    A ``self.tr("...")`` written inside ``f"{...}"`` still runs, so the
    English shows, but the string is never extracted -- and an entry that
    used to be extracted is dropped from every catalog as obsolete, taking
    its translations with it. Bind the call to a name first.
    """
    hits = []
    for path in sorted(GUI_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FormattedValue):
                continue
            for sub in ast.walk(node.value):
                if not isinstance(sub, ast.Call):
                    continue
                func = sub.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                literal = sub.args[:2] if name == "translate" else sub.args[:1]
                if name in ("tr", "translate") and any(
                        isinstance(a, ast.Constant) and isinstance(a.value, str)
                        for a in literal):
                    hits.append(f"{path.relative_to(ROOT)}:{sub.lineno}")
    assert not hits, "tr() inside an f-string, invisible to lupdate: " + ", ".join(hits)


def test_no_translation_function_is_aliased():
    """lupdate matches ``tr(`` and ``QCoreApplication.translate(`` by name.

    ``tr = QCoreApplication.translate`` followed by ``tr("Ctx", "...")`` runs
    and translates, yet nothing is extracted, and the catalog drops the
    entries as obsolete on the next extract.
    """
    hits = []
    for path in sorted(GUI_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if isinstance(value, ast.Attribute) and value.attr in ("translate", "tr"):
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not hits, "aliased translate/tr, invisible to lupdate: " + ", ".join(hits)
