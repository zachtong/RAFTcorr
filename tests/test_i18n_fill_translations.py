"""scripts/fill_translations.py writes what lupdate would, and only where asked.

Runs the filler on a small catalog in lupdate's format: context-specific
entries win over the source-only table, plural messages get the language's
forms, apostrophes and quotes are escaped like lupdate escapes them (otherwise
the next extract rewrites every French translation), finished entries are
never touched, and a second run changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fill_translations as ft  # noqa: E402

sys.path.pop(0)

_CATALOG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="fr" sourcelanguage="en_US">
<context>
    <name>AdvancedPanel</name>
    <message>
        <location filename="../../gui/widgets/param_panel.py" line="128"/>
        <source>Margin</source>
        <translation type="unfinished"></translation>
    </message>
</context>
<context>
    <name>ExportDialog</name>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="10"/>
        <source>Margin</source>
        <translation type="unfinished">Marge</translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="11"/>
        <source>The model&apos;s &quot;reach&quot; &lt;b&gt;%1&lt;/b&gt;</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="12"/>
        <source>Light</source>
        <comment>Color theme</comment>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../../gui/dialogs/export_dialog.py" line="13"/>
        <source>Done</source>
        <translation>Fait</translation>
    </message>
    <message numerus="yes">
        <location filename="../../gui/dialogs/export_dialog.py" line="14"/>
        <source>%n file(s)</source>
        <translation type="unfinished">
            <numerusform></numerusform>
            <numerusform></numerusform>
        </translation>
    </message>
</context>
</TS>
"""


@pytest.fixture
def tables(monkeypatch):
    """Small tables in place of the real ones (every language filled)."""
    def every(text: str) -> dict[str, str]:
        return {lang: text for lang in ft.LANGUAGES}

    monkeypatch.setattr(ft, "TRANSLATIONS", {
        "Margin": every("Bordure"),
        "Done": every("Terminé"),
        'The model\'s "reach" <b>%1</b>': every("La « portée » de l'image <b>%1</b>"),
        "Light": every("Léger"),
    })
    monkeypatch.setattr(ft, "CONTEXTUAL", {("AdvancedPanel", "Margin"): every("Marge")})
    monkeypatch.setattr(ft, "DISAMBIGUATED", {("Light", "Color theme"): every("Clair")})
    monkeypatch.setattr(ft, "NUMERUS_TRANSLATIONS", {"%n file(s)": {
        lang: (("%n fichier", "%n fichiers") if ft.NUMERUS_FORMS[lang] == 2 else ("%n 个文件",))
        for lang in ft.LANGUAGES}})


def test_fill_writes_lupdate_format(tables):
    text, filled, missing = ft.fill_text(_CATALOG, "fr")
    assert (filled, missing) == (5, [])
    # Context entry wins over the source-only table; the other context does not.
    advanced, export = text.split("<name>ExportDialog</name>")
    assert "<translation>Marge</translation>" in advanced
    assert "<translation>Bordure</translation>" in export
    # lupdate's escaping, including &apos; (never a bare apostrophe).
    assert "<translation>La « portée » de l&apos;image &lt;b&gt;%1&lt;/b&gt;</translation>" in text
    # Disambiguation comment selects its own entry.
    assert "<translation>Clair</translation>" in text
    # Finished entries are left alone.
    assert "<translation>Fait</translation>" in text and "Terminé" not in text
    # Plural forms with lupdate's indentation.
    assert ("<translation>\n            <numerusform>%n fichier</numerusform>\n"
            "            <numerusform>%n fichiers</numerusform>\n        </translation>") in text
    assert 'type="unfinished"' not in text


def test_second_run_changes_nothing(tables):
    once, _, _ = ft.fill_text(_CATALOG, "fr")
    twice, filled, missing = ft.fill_text(once, "fr")
    assert (twice, filled, missing) == (once, 0, [])


def test_unknown_message_is_reported_not_guessed(tables, monkeypatch):
    monkeypatch.setattr(ft, "TRANSLATIONS", {})
    text, _, missing = ft.fill_text(_CATALOG, "fr")
    assert "[ExportDialog] Margin" in missing
    assert '<translation type="unfinished">Marge</translation>' in text


def test_wrong_number_of_plural_forms_is_refused(tables, monkeypatch):
    # The table passes its own check (fr declared with one form), but the
    # catalog has two slots: refuse rather than write a half-filled plural.
    forms = {**ft.NUMERUS_FORMS, "fr": 1}
    monkeypatch.setattr(ft, "NUMERUS_FORMS", forms)
    monkeypatch.setattr(ft, "NUMERUS_TRANSLATIONS", {"%n file(s)": {
        lang: ("%n x",) * forms[lang] for lang in ft.LANGUAGES}})
    ft.check_tables()
    with pytest.raises(ValueError, match="catalog expects 2"):
        ft.fill_text(_CATALOG, "fr")


@pytest.mark.parametrize("bad", ["Bordure sans valeur", "<b>%1 %2</b>", ""])
def test_tables_must_keep_placeholders_and_tags(tables, monkeypatch, bad):
    source = 'The model\'s "reach" <b>%1</b>'
    monkeypatch.setattr(ft, "TRANSLATIONS", {source: {lang: bad for lang in ft.LANGUAGES}})
    with pytest.raises(ValueError):
        ft.check_tables()


def test_escape_round_trips():
    raw = 'l\'image "x" <b>a & b</b>'
    assert ft.escape(raw) == "l&apos;image &quot;x&quot; &lt;b&gt;a &amp; b&lt;/b&gt;"
    assert ft.unescape(ft.escape(raw)) == raw


def test_real_catalogs_have_nothing_left_to_fill():
    """Every catalog is complete, or the tables cover what is missing."""
    for lang in ft.LANGUAGES:
        text = (ft.TS_DIR / f"raftcorr_{lang}.ts").read_text(encoding="utf-8")
        _, _, missing = ft.fill_text(text, lang)
        assert not missing, f"{lang}: no translation for {missing[:5]}"
