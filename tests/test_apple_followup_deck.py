"""Deck de apoyo a la respuesta a Riesgos (``src/studies/apple/followup_deck.py``).

Se construye sobre las salidas reales del estudio, que no están en git: sin ellas, se salta.
"""

from pathlib import Path

import pytest

pytest.importorskip("pptx")

from src.studies.apple.followup_deck import build, es, quarter_label, span_es  # noqa: E402

DATA = Path("output/apple_study")


def test_formatters():
    assert es(3.456, 2, "%") == "3,46%"
    assert es(float("nan")) == "—"
    assert quarter_label("2026Q2") == "T2 26"
    assert span_es("2026-08-01", "2026-11-01") == "ago-oct 2026"
    assert span_es("2025-03-01", "2026-03-01") == "mar 2025 - feb 2026"


@pytest.mark.skipif(not (DATA / "b1_tienda_distribucion_score.csv").exists(), reason="sin salidas reales del estudio")
def test_build_writes_nine_slides_with_five_charts(tmp_path):
    from pptx import Presentation

    out = build(DATA, tmp_path / "dudas.pptx", today="18 nov 2026")
    slides = list(Presentation(str(out)).slides)
    assert len(slides) == 9
    assert sum(sh.shape_type == 13 for s in slides for sh in s.shapes) == 5
    text = " ".join(sh.text_frame.text for s in slides for sh in s.shapes if sh.has_text_frame)
    assert "nan" not in text.lower() and "—" not in text
