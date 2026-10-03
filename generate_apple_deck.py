"""Deck del estudio Apple — extensión del score Equifax de e-commerce a tienda.

Variante del deck de resultados para este estudio concreto. No reutiliza
``generate_results_presentation.py`` porque aquel asume un relato que aquí no aplica:
segmentos de un solo canal, backtest out-of-time poblado y riesgo medido en todas las
poblaciones. Aquí hay dos canales, el riesgo de tienda es **imputado** y el backtest
no tiene cohorte madura.

Lee las salidas de ``run_apple_study.py`` (``output/apple_study/*.csv``) y escribe un
``.pptx`` de 9 slides. Los gráficos vienen de :mod:`src.apple_deck_charts`.

Uso:
    uv run python run_apple_study.py && uv run python generate_apple_deck.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from run_apple_study import BIN_EDGES  # una sola definición de los tramos, la del estudio
from src import apple_deck_charts as ac

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
INK = RGBColor(0x2C, 0x3E, 0x50)
MUTED = RGBColor(0x5D, 0x6D, 0x7E)
ACCENT = RGBColor(0x34, 0x98, 0xDB)
RISK = RGBColor(0xE7, 0x4C, 0x3C)
RULE = RGBColor(0xE5, 0xE7, 0xEB)
TILE_BG = RGBColor(0xF6, 0xF7, 0xF9)
FONT = "Arial"

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


# --------------------------------------------------------------------------- #
# primitivas de layout
# --------------------------------------------------------------------------- #
def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, x, y, w, h, text, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, spacing=1.0):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(text).split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color
    return box


_MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def periodo(desde: str, hasta: str) -> str:
    """'2025-03-01','2026-03-01' -> 'mar-2025 a feb-2026'. El fin es exclusivo."""
    a, b = date.fromisoformat(desde), date.fromisoformat(hasta)
    fin = date(b.year - 1, 12, 1) if b.month == 1 else date(b.year, b.month - 1, 1)
    inicio_txt = f"{_MESES[a.month - 1]}-{a.year}"
    fin_txt = f"{_MESES[fin.month - 1]}-{fin.year}"
    return inicio_txt if inicio_txt == fin_txt else f"{inicio_txt} a {fin_txt}"


def _header(slide, title: str, kicker: str = "", period: str = "") -> Emu:
    """Cabecera: antetítulo, periodo a la derecha, título y filete.

    El periodo va en **todas** las slides de dato: el estudio mezcla tres ventanas
    (Online maduro, tienda con score, años completos para la estacionalidad) y sin
    rotularlas dos cifras correctas de periodos distintos se leen como contradictorias.
    """
    y = Inches(0.42)
    if kicker or period:
        if kicker:
            _text(slide, MARGIN, y, Inches(7.0), Inches(0.3), kicker.upper(), size=11, bold=True, color=ACCENT)
        if period:
            _text(
                slide,
                W - MARGIN - Inches(5.2),
                y,
                Inches(5.2),
                Inches(0.3),
                period,
                size=11,
                color=MUTED,
                align=PP_ALIGN.RIGHT,
            )
        y += Inches(0.34)
    _text(slide, MARGIN, y, W - 2 * MARGIN, Inches(0.6), title, size=26, bold=True)
    line = slide.shapes.add_shape(1, MARGIN, y + Inches(0.72), W - 2 * MARGIN, Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False
    return y + Inches(0.95)


def _img_ratio(path: Path) -> float:
    from PIL import Image

    with Image.open(path) as img:
        return img.height / img.width


def _picture(slide, path: Path, y: Emu, max_h: Emu) -> None:
    """Inserta el PNG centrado, escalado para caber en el alto disponible."""
    ratio = _img_ratio(path)
    w = W - 2 * MARGIN
    h = Emu(int(w * ratio))
    if h > max_h:
        h, w = max_h, Emu(int(max_h / ratio))
    slide.shapes.add_picture(str(path), Emu(int((W - w) / 2)), y, width=w, height=h)


def _tile(slide, x, y, w, h, label: str, value: str, note: str = "", value_color=INK) -> None:
    """Stat tile: etiqueta arriba, cifra grande, nota debajo. La cifra es el gráfico."""
    card = slide.shapes.add_shape(1, x, y, w, h)
    card.fill.solid()
    card.fill.fore_color.rgb = TILE_BG
    card.line.color.rgb = RULE
    card.shadow.inherit = False
    _text(slide, x + Inches(0.18), y + Inches(0.14), w - Inches(0.36), Inches(0.4), label, size=11, color=MUTED)
    _text(
        slide,
        x + Inches(0.18),
        y + Inches(0.46),
        w - Inches(0.36),
        Inches(0.62),
        value,
        size=30,
        bold=True,
        color=value_color,
    )
    if note:
        _text(slide, x + Inches(0.18), y + h - Inches(0.52), w - Inches(0.36), Inches(0.42), note, size=10, color=MUTED)


def _table(slide, x, y, w, rows: list[list[str]], col_w: list[float] | None = None, row_h: float = 0.36) -> None:
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, Inches(row_h * n_rows))
    table = shape.table
    if col_w:
        total = sum(col_w)
        for i, frac in enumerate(col_w):
            table.columns[i].width = Emu(int(w * frac / total))
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            run = para.runs[0]
            run.font.size, run.font.name = Pt(11.5), FONT
            run.font.bold = r == 0
            run.font.color.rgb = INK if r == 0 else MUTED
            cell.fill.solid()
            cell.fill.fore_color.rgb = TILE_BG if r == 0 else RGBColor(0xFF, 0xFF, 0xFF)


def _bullets(slide, x, y, w, items: list[tuple[str, str]], size: int = 13) -> None:
    """Viñetas con prefijo en negrita: (prefijo, texto)."""
    # La caja se declara con el alto que queda hasta el pie: PowerPoint la crece sola si
    # el texto no cabe, pero declarar de menos deja la geometría mintiendo sobre lo que
    # ocupa de verdad, y cualquier render que no crezca la recortaría.
    box = slide.shapes.add_textbox(x, y, w, max(H - y - Inches(0.4), Inches(0.4) * len(items)))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (prefix, body) in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(9)
        if prefix:
            run = para.add_run()
            run.text = f"{prefix}  "
            run.font.size, run.font.bold, run.font.name = Pt(size), True, FONT
            run.font.color.rgb = INK
        run = para.add_run()
        run.text = body
        run.font.size, run.font.name = Pt(size), FONT
        run.font.color.rgb = MUTED


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def slide_title(prs, online, stores, today: str) -> None:
    slide = _blank(prs)
    band = slide.shapes.add_shape(1, Emu(0), Emu(0), W, Inches(0.16))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False
    _text(
        slide,
        MARGIN,
        Inches(1.9),
        W - 2 * MARGIN,
        Inches(0.4),
        "ESTUDIO DE RIESGO · APPLE",
        size=13,
        bold=True,
        color=ACCENT,
    )
    _text(
        slide,
        MARGIN,
        Inches(2.35),
        W - 2 * MARGIN,
        Inches(1.5),
        "Extensión del score Equifax\nde e-commerce a tienda",
        size=40,
        bold=True,
        spacing=1.05,
    )
    _text(
        slide,
        MARGIN,
        Inches(4.25),
        Inches(8.6),
        Inches(1.2),
        "Aplicar la parrilla de Apple Online a Apple Stores gana aceptación y baja riesgo a la vez. "
        "El riesgo de tienda sigue siendo estimado, pero su nivel ya está anclado a la morosidad "
        "real de tienda: solo se asume que el score ordena igual, no cuánto riesgo hay.",
        size=15,
        color=MUTED,
        spacing=1.25,
    )
    _text(
        slide,
        MARGIN,
        Inches(6.4),
        W - 2 * MARGIN,
        Inches(0.4),
        f"Riesgos  ·  {today}  ·  perímetro Apple Propio",
        size=11,
        color=MUTED,
    )


def slide_exec(prs, online, stores, picks, stores_img: Path, meta: dict, facts: dict, period: str = "") -> None:
    """Resumen ejecutivo en dos columnas: los números a la izquierda, el hallazgo a la derecha."""
    slide = _blank(prs)
    y = _header(slide, "Lo que dice el estudio", "Resumen ejecutivo", period)
    w, gap = Inches(3.02), Inches(0.18)
    tiles = [
        ("Riesgo real Apple Online", ac.es(online["riesgo_real_pct"], 2, "%"), "medido sobre cartera madura", RISK),
        ("Tasa de aceptación Online", ac.es(online["ta_pct"], 1, "%"), "sobre demanda en €", INK),
        (
            "Tienda con parrilla EFX",
            ac.es(stores["ta_efectiva_pct"], 1, "%"),
            f"TA, desde {ac.es(stores['ta_actual_pct'], 1, '%')} hoy",
            ACCENT,
        ),
        (
            "Riesgo imputado tienda",
            ac.es(stores["riesgo_imputado_pct"], 2, "%"),
            f"desde {ac.es(stores['riesgo_imputado_actual_pct'], 2, '%')} hoy",
            ACCENT,
        ),
    ]
    for i, (label, value, note, color) in enumerate(tiles):
        _tile(slide, MARGIN + i * (w + gap), y, w, Inches(1.62), label, value, note, color)

    y2 = y + Inches(1.95)
    left_w, right_x, right_w = Inches(6.15), MARGIN + Inches(6.45), Inches(5.65)

    _text(slide, MARGIN, y2, left_w, Inches(0.4), "Escenarios de riesgo global Apple", size=15, bold=True)
    # Solo el corte único en el resumen; la variante de parrilla desplazada va en la
    # slide del punto 3. Se publica la TA EFECTIVA, que es la comparable con el punto 1.
    rows = [["Escenario", "Corte", "TA efect.", "Producción"]]
    for row in picks.dropna(subset=["riesgo_total_pct"]).itertuples():
        if "parrilla desplazada" in row.escenario:
            continue
        rows.append(
            [
                row.escenario.replace(" · corte único", ""),
                str(row.corte),
                ac.es(row.ta_efectiva_pct, 1, "%"),
                f"{ac.es(row.produccion_mensual_eur / 1e6, 1)} M€/mes",
            ]
        )
    _table(slide, MARGIN, y2 + Inches(0.45), left_w, rows, col_w=[1.6, 1, 1, 1.25], row_h=0.30)

    _bullets(
        slide,
        MARGIN,
        y2 + Inches(2.55),
        left_w,
        [
            (
                "Riesgo de tienda:",
                "forma de la curva tomada de e-commerce; el nivel, anclado al "
                f"{ac.es(meta['riesgo_tienda_realizado_pct'], 2, '%')} realizado de tienda. No es una medición.",
            ),
            (
                "Ventana:",
                f"{facts['online_meses']} meses de originación madura ({ac.es(online['riesgo_real_pct'], 2, '%')} "
                f"frente a {ac.es(meta['riesgo_online_historico_pct'], 2, '%')} con "
                f"{meta['online_historico_meses']} meses).",
            ),
            (
                "Con reject inference:",
                f"los tramos poco aceptados llevan uplift (hasta {ac.es(facts['max_uplift'], 1)}x). Apenas mueve "
                f"los cortes: donde se decide, Online ya acepta al {ac.es(facts['zone_lo_pct'], 0)}-"
                f"{ac.es(facts['zone_hi_pct'], 0)}%.",
            ),
        ],
        size=11.5,
    )

    _text(
        slide,
        right_x,
        y2,
        right_w,
        Inches(0.4),
        "Tienda: la parrilla de e-commerce mejora las dos cosas",
        size=15,
        bold=True,
    )
    slide.shapes.add_picture(
        str(stores_img),
        right_x,
        y2 + Inches(0.45),
        width=right_w,
        height=Emu(int(right_w * _img_ratio(stores_img))),
    )


def _chain(slide, x, y, w, steps: list[tuple[str, str, str, str]]) -> None:
    """Cadena vertical de pasos: número, título, cuerpo y etiqueta de naturaleza.

    La etiqueta (MEDIDO / CORREGIDO / ASUMIDO) es el punto de la slide: en un estudio
    donde una parte sale del dato y otra de un supuesto, lo que un comité necesita ver
    de un vistazo es cuál es cuál, paso a paso.
    """
    tag_colors = {"MEDIDO": ACCENT, "CORREGIDO": MUTED, "ASUMIDO": RISK}
    row_h = Inches(1.02)
    for i, (num, title, body, tag) in enumerate(steps):
        top = y + row_h * i
        badge = slide.shapes.add_shape(1, x, top, Inches(0.46), Inches(0.46))
        badge.fill.solid()
        badge.fill.fore_color.rgb = INK
        badge.line.fill.background()
        badge.shadow.inherit = False
        _text(
            slide,
            x,
            top + Inches(0.06),
            Inches(0.46),
            Inches(0.34),
            num,
            size=14,
            bold=True,
            color=RGBColor(0xFF, 0xFF, 0xFF),
            align=PP_ALIGN.CENTER,
        )
        _text(slide, x + Inches(0.68), top, w - Inches(2.4), Inches(0.3), title, size=13, bold=True)
        _text(
            slide,
            x + Inches(0.68),
            top + Inches(0.3),
            w - Inches(2.4),
            Inches(0.66),
            body,
            size=11,
            color=MUTED,
            spacing=1.18,
        )
        _text(
            slide,
            x + w - Inches(1.6),
            top + Inches(0.02),
            Inches(1.6),
            Inches(0.3),
            tag,
            size=10,
            bold=True,
            color=tag_colors[tag],
            align=PP_ALIGN.RIGHT,
        )


def slide_method_risk(prs, meta: dict, facts: dict, period: str) -> None:
    """Cómo se estima el riesgo: la cadena completa, con lo medido y lo asumido separado."""
    slide = _blank(prs)
    y = _header(slide, "Cómo se estima el riesgo de tienda", "Metodología · Riesgo", period)
    factor = meta.get("factor_nivel_tienda", 1.0)
    realizado = meta.get("riesgo_tienda_realizado_pct", float("nan"))
    imputado = meta.get("riesgo_tienda_imputado_pct", float("nan"))
    _chain(
        slide,
        MARGIN,
        y,
        W - 2 * MARGIN,
        [
            (
                "1",
                "Riesgo realizado por tramo de score, en Apple Online",
                "Para cada uno de los 20 tramos de EFX, la morosidad observada de los contratos con 6 meses "
                "cumplidos. Es lo único del estudio que sale directamente del dato.",
                "MEDIDO",
            ),
            (
                "2",
                "Corrección por selección (reject inference)",
                "Los contratados de un tramo pasaron además el resto de reglas: son una muestra seleccionada y su "
                "morosidad subestima la de quien hoy se deniega. Se aplica el parceling del pipeline, con uplift "
                f"según la tasa de aceptación: {ac.es(facts['max_uplift'], 1)}x donde solo entra el "
                f"{ac.es(facts['min_acc_pct'], 0)}%, 1,0x donde entra el {ac.es(facts['max_acc_pct'], 0)}%.",
                "CORREGIDO",
            ),
            (
                "3",
                "Esa curva se traslada a la población de tienda",
                "A cada solicitud de tienda se le asigna el riesgo del tramo EFX en el que cae. Aquí está el "
                "supuesto de fondo: que el score ordena en tienda igual que en e-commerce.",
                "ASUMIDO",
            ),
            (
                "4",
                "El nivel se ancla a la morosidad real de tienda",
                f"La curva de Online predice un {ac.es(imputado, 2, '%')} sobre la cartera contratada de tienda; "
                f"lo realmente observado es {ac.es(realizado, 2, '%')}. Se reescala por {ac.es(factor, 3)} para que "
                "reproduzca el dato. El nivel deja de ser un supuesto: solo queda asumido el poder discriminante.",
                "MEDIDO",
            ),
        ],
    )
    _text(
        slide,
        MARGIN,
        H - Inches(1.15),
        W - 2 * MARGIN,
        Inches(0.9),
        "Qué significa en la práctica: si EFX discrimina en tienda peor de lo que discrimina en Online, "
        "los cortes de los escenarios se quedan cortos. Lo que ya no puede fallar es la altura de la curva, "
        "porque está calibrada contra la morosidad que tienda ha tenido de verdad.",
        size=11.5,
        color=MUTED,
        spacing=1.2,
    )


def slide_method_seasonal(prs, meta: dict, facts: dict, period: str) -> None:
    """Cómo se anualiza: por qué un mes no es un mes y qué se hace al respecto."""
    slide = _blank(prs)
    y = _header(slide, "Cómo se anualiza", "Metodología · Estacionalidad", period)
    _chain(
        slide,
        MARGIN,
        y,
        W - 2 * MARGIN,
        [
            (
                "1",
                "El problema: tienda solo tiene un mes con score, y es agosto",
                f"El score EFX se activó en tienda en {facts['stores_month']}. Comparar un mes suelto contra una media de doce "
                "exige saber cuánto vale ese mes.",
                "MEDIDO",
            ),
            (
                "2",
                "Agosto no es un mes medio, y cada canal tiene el suyo",
                f"Índice de agosto: {ac.es(facts['aug_online'], 2)} en Online y {ac.es(facts['aug_stores'], 2)} en "
                f"tienda. Online concentra la demanda en el lanzamiento de iPhone ({facts['peak_online'][0]} "
                f"{ac.es(facts['peak_online'][1], 2)}) y tienda en Navidad ({facts['peak_stores'][0]} "
                f"{ac.es(facts['peak_stores'][1], 2)}). La correlación entre ambos perfiles es "
                f"{ac.es(facts['index_corr'], 2)}: no son el mismo patrón.",
                "MEDIDO",
            ),
            (
                "3",
                "Meses efectivos en vez de meses de calendario",
                "El run-rate no se divide por el número de meses sino por la suma del índice estacional de los "
                "meses cubiertos. Una ventana natural completa suma 12,00 y no altera nada; el agosto de tienda "
                f"suma {ac.es(facts['aug_stores'], 2)} y corrige por ser un mes flojo.",
                "CORREGIDO",
            ),
            (
                "4",
                "Efecto sobre el peso de tienda en Apple Total",
                f"Sin corregir, tienda parecía pesar el {ac.es(meta['peso_tienda_calendario_pct'], 0)}% de la "
                f"demanda. Con su estacionalidad propia pesa el {ac.es(meta['peso_tienda_propio_pct'], 0)}%. Si se le "
                "hubiera aplicado la de Online —lo único posible antes de tener su histórico— habríamos dicho "
                f"{ac.es(meta['peso_tienda_indice_online_pct'], 0)}%, y todos los escenarios saldrían con más "
                "producción de la real.",
                "MEDIDO",
            ),
        ],
    )
    _text(
        slide,
        MARGIN,
        H - Inches(1.15),
        W - 2 * MARGIN,
        Inches(0.9),
        "El índice se estima promediando años naturales completos, de modo que el crecimiento de volumen "
        "entre años no se confunda con estacionalidad. Lo que queda abierto es si el mix de un único mes "
        "de tienda es representativo del año.",
        size=11.5,
        color=MUTED,
        spacing=1.2,
    )


def slide_chart(prs, title: str, kicker: str, img: Path, caption: str = "", period: str = "") -> None:
    slide = _blank(prs)
    y = _header(slide, title, kicker, period)
    body_h = H - y - (Inches(0.95) if caption else Inches(0.45))
    _picture(slide, img, y, body_h)
    if caption:
        _text(
            slide, MARGIN, H - Inches(0.86), W - 2 * MARGIN, Inches(0.6), caption, size=11.5, color=MUTED, spacing=1.2
        )


def slide_frontier(prs, img: Path, picks: pd.DataFrame, meta: dict, period: str = "") -> None:
    """La frontera lleva las cifras al lado, no dentro del gráfico: cuatro globos con
    tres cifras cada uno chocaban entre sí y tapaban la curva."""
    slide = _blank(prs)
    y = _header(slide, "Cuánto cuesta cada objetivo de riesgo", "Punto 3 · Escenarios", period)
    chart_w = Inches(7.35)
    chart_h = Emu(int(chart_w * _img_ratio(img)))
    slide.shapes.add_picture(str(img), MARGIN, y, width=chart_w, height=chart_h)

    x2 = MARGIN + chart_w + Inches(0.28)
    rows = [["Escenario", "Corte", "TA score", "TA efect."]]
    for row in picks.dropna(subset=["riesgo_total_pct"]).itertuples():
        rows.append(
            [
                row.escenario.replace(" · corte único", " · único").replace(" · parrilla desplazada", " · parrilla"),
                str(row.corte).replace(" puntos", "p"),
                "—" if pd.isna(row.ta_score_pct) else ac.es(row.ta_score_pct, 1, "%"),
                ac.es(row.ta_efectiva_pct, 1, "%"),
            ]
        )
    _table(slide, x2, y + Inches(0.1), Inches(4.75), rows, col_w=[1.7, 1.0, 0.9, 0.9], row_h=0.28)

    # El aviso va bajo el gráfico, que es donde queda sitio y donde lo verá quien mire la
    # curva. Es la pregunta que va a salir en comité: por qué el modelo produce más que lo
    # que de hecho se produjo.
    base_rows = picks[picks["escenario"].str.startswith("Parrilla Online")]
    base_prod = float(base_rows.iloc[0]["produccion_mensual_eur"]) if not base_rows.empty else float("nan")
    today_rows = picks[picks["escenario"].str.startswith("Hoy")]
    today_prod = float(today_rows.iloc[0]["produccion_mensual_eur"]) if not today_rows.empty else float("nan")
    # Debajo del gráfico, calculado desde su alto real: estimarlo a ojo lo hacía chocar
    # con la etiqueta del eje.
    note_y = y + chart_h + Inches(0.12)
    _text(
        slide,
        MARGIN,
        note_y,
        chart_w,
        Inches(0.32),
        "Por qué el modelo produce más que lo medido",
        size=12.5,
        bold=True,
        color=RISK,
    )
    _text(
        slide,
        MARGIN,
        note_y + Inches(0.34),
        chart_w,
        Inches(1.1),
        "La parrilla vigente calca las decisiones reales de Online (acepta el "
        f"{ac.es(meta['online_parrilla_pct'], 1, '%')} de la demanda; el motor aprobó el "
        f"{ac.es(meta['online_aprobado_real_pct'], 1, '%')}: swap-in {ac.es(meta['online_swap_in_pct'], 1, '%')}, "
        f"swap-out {ac.es(meta['online_swap_out_pct'], 1, '%')}) y cada solicitud aceptada convierte a la tasa de "
        f"su decisión real. La diferencia entre {ac.es(base_prod / 1e6, 1)} y {ac.es(today_prod / 1e6, 1)} M€/mes "
        f"es por tanto la ganancia de tienda, medida sobre {periodo(meta['tienda_desde'], meta['tienda_hasta'])} "
        "(único mes con EFX) frente a la ventana "
        "completa de 'hoy'.",
        size=11,
        color=MUTED,
        spacing=1.2,
    )
    _text(
        slide,
        x2,
        y + Inches(3.40),
        Inches(4.75),
        Inches(2.3),
        "TA score: demanda que supera el corte.\n"
        "TA efectiva: producción / demanda, la misma base que el punto 1. No son comparables entre sí.\n\n"
        "Desplazar la parrilla actual mantiene su forma —laxa donde el grupo es bueno— y da algo más "
        "de aceptación al mismo riesgo.",
        size=10.5,
        color=MUTED,
        spacing=1.25,
    )


def slide_grids(prs, grids: pd.DataFrame, period: str = "") -> None:
    """Las parrillas resultantes: el entregable que alguien tiene que implementar.

    Dos tablas en vez de una de diez columnas: un instrumento cada una, con la parrilla
    vigente repetida en ambas como referencia.
    """
    slide = _blank(prs)
    y = _header(slide, "Qué parrilla implica cada escenario", "Punto 4 · Parrillas", period)
    _text(
        slide,
        MARGIN,
        y,
        W - 2 * MARGIN,
        Inches(0.35),
        "Umbral de RECHAZO por grupo: se acepta por encima del valor.",
        size=11.5,
        color=MUTED,
    )
    y += Inches(0.45)

    for title, keyword in (("Corte único para todo Apple", "único"), ("Parrilla actual desplazada", "parrilla")):
        cols = ["grupo", "Parrilla actual"] + [c for c in grids.columns if c.endswith(keyword)]
        rows = [["Grupo", "Hoy"] + [c.split(" ·")[0] for c in cols[2:]]]
        rows += [[str(r[c]) for c in cols] for _, r in grids.iterrows()]
        _text(slide, MARGIN, y, W - 2 * MARGIN, Inches(0.32), title, size=14, bold=True)
        _table(
            slide, MARGIN, y + Inches(0.34), Inches(11.0), rows, col_w=[1.3, 1.0] + [1.0] * (len(cols) - 2), row_h=0.27
        )
        y += Inches(0.34) + Inches(0.27) * len(rows) + Inches(0.32)

    _text(
        slide,
        MARGIN,
        H - Inches(0.78),
        W - 2 * MARGIN,
        Inches(0.7),
        "El corte único es más simple de operar; desplazar la parrilla conserva su forma —laxa donde el grupo "
        "es bueno— y da algo más de aceptación al mismo riesgo. '>=G' se rechaza siempre en ambos.",
        size=11,
        color=MUTED,
        spacing=1.2,
    )


def slide_closing(prs, facts: dict) -> None:
    slide = _blank(prs)
    y = _header(slide, "Qué falta y qué haría falta decidir", "Limitaciones y siguientes pasos")
    _text(slide, MARGIN, y, Inches(5.9), Inches(0.4), "Limitaciones del estudio", size=15, bold=True, color=RISK)
    _bullets(
        slide,
        MARGIN,
        y + Inches(0.5),
        Inches(5.9),
        [
            (
                "Riesgo de tienda estimado.",
                "Las operaciones de tienda con EFX tienen 0-2 meses y no tienen "
                f"comportamiento observable. H3 hacia {facts['h3_month']}, H6 hacia {facts['h6_month']}.",
            ),
            (
                "Comparación condicional.",
                "El score interno y EFX se comparan con la vara de EFX. No demuestra que EFX sea mejor "
                "en tienda: lo asume. Lo contrastado es el nivel de riesgo, no el poder discriminante.",
            ),
            (
                "Tramos bajos débiles.",
                f"Los tramos 1-{facts['low_bins']} apenas tienen producción, y es donde más pesa el uplift por "
                f"selección ({ac.es(facts['max_uplift'], 1)}x). Son los que abren el escenario más laxo.",
            ),
            (
                "Un solo mes de tienda.",
                f"La foto sale de {facts['stores_month']}. Su estacionalidad propia ya está medida, pero el mix de un "
                "único mes puede no ser representativo.",
            ),
        ],
        size=12,
    )

    _text(
        slide, MARGIN + Inches(6.3), y, Inches(5.9), Inches(0.4), "Siguientes pasos", size=15, bold=True, color=ACCENT
    )
    _bullets(
        slide,
        MARGIN + Inches(6.3),
        y + Inches(0.5),
        Inches(5.9),
        [
            (
                "Extracción.",
                f"Refresco de Online a {facts['stores_month']}, score EFX en tienda y su histórico de comportamiento "
                "para anclar el nivel de riesgo.",
            ),
            ("Decisión de negocio.", "Objetivo de riesgo global y si el corte es único o por segmento."),
            (
                "Validación.",
                "Repetir con el pipeline completo: reject inference, intervalos de confianza y "
                "auditoría swap-in / swap-out.",
            ),
            ("Medición.", f"Revisar en {facts['h3_month']} con H3 y en {facts['h6_month']} con H6 reales de tienda."),
        ],
        size=12,
    )


def deck_facts(curve, p2, stores_actuals, grids, index, meta) -> dict:
    """Las cifras que los textos del deck citan, calculadas desde las salidas del estudio.

    Escritas en las slides se desincronizan en la primera corrida que cambie algo; aquí
    cada una tiene su fuente. Lo único que queda literal en el deck es estructural (20
    tramos, un año natural suma 12,00).
    """
    uplift = curve["b2_pct_rechazados"] / curve["b2_pct_contratados"]
    acceptance = 100 * curve["tasa_aceptacion"]
    # "donde se decide": los tramos donde caen los cortes únicos de los escenarios
    cuts = [int(v.strip("> ")) for col in grids.columns if "único" in col for v in grids[col] if str(v).startswith(">")]
    lower_edge = curve["bin"].map(lambda b: BIN_EDGES[int(b) - 1])
    zone = acceptance[(lower_edge >= min(cuts)) & (lower_edge <= max(cuts))]
    stores_total = p2[p2["segmento"] == "TOTAL"].iloc[0]
    new = p2[p2["segmento"].str.lower() == "new"]
    last_stores_month = pd.Timestamp(meta["tienda_hasta"]) - pd.Timedelta(days=1)

    def mes(ts: pd.Timestamp) -> str:
        return f"{MESES[ts.month - 1]}-{ts.year}"

    peak_online, peak_stores = int(index["indice"].idxmax()), int(index["indice_tienda"].idxmax())
    return {
        "max_uplift": float(uplift.max()),
        "min_acc_pct": float(acceptance.min()),
        "max_acc_pct": float(acceptance.max()),
        "zone_lo_pct": float(zone.min()),
        "zone_hi_pct": float(zone.max()),
        "low_bins": int((curve["tasa_aceptacion"] < 0.5).sum()),
        "new_share_pct": 100 * float(new["demanda_eur"].sum()) / float(stores_total["demanda_eur"]),
        "ta_stores_window_pct": float(stores_actuals.set_index("segmento").loc["TOTAL", "ta_pct"]),
        "ta_stores_aug_pct": float(stores_total["ta_actual_pct"]),
        "online_meses": (pd.Period(meta["online_hasta"], "M") - pd.Period(meta["online_desde"], "M")).n,
        "aug_online": float(index.loc[index["mes"] == 8, "indice"].iloc[0]),
        "aug_stores": float(index.loc[index["mes"] == 8, "indice_tienda"].iloc[0]),
        "peak_online": (MESES[int(index.loc[peak_online, "mes"]) - 1], float(index.loc[peak_online, "indice"])),
        "peak_stores": (MESES[int(index.loc[peak_stores, "mes"]) - 1], float(index.loc[peak_stores, "indice_tienda"])),
        "index_corr": float(index["indice"].corr(index["indice_tienda"])),
        "stores_month": mes(last_stores_month),
        "h3_month": mes(last_stores_month + pd.DateOffset(months=3)),
        "h6_month": mes(last_stores_month + pd.DateOffset(months=6)),
    }


# --------------------------------------------------------------------------- #
def build(data_dir: Path, out_path: Path) -> Path:
    read = lambda name: pd.read_csv(data_dir / f"{name}.csv")  # noqa: E731
    curve, p1, p2 = read("curva_riesgo_online"), read("p1_online_actual"), read("p2_tienda_parrilla_online")
    stores_actuals = read("p1b_tienda_actual")
    meta = json.loads((data_dir / "periodos.json").read_text(encoding="utf-8"))
    p_online = periodo(meta["online_desde"], meta["online_hasta"])
    p_tienda = periodo(meta["tienda_desde"], meta["tienda_hasta"])
    p_mixto = f"Online {p_online}  ·  tienda {p_tienda}"
    p_estacional = " y ".join(str(a) for a in meta["estacionalidad_anios"])
    ladder, picks, index = read("p3_escalera_corte_unico"), read("p3_escenarios_objetivo"), read("indice_estacional")
    grids = read("p4_parrillas_escenarios")
    online = p1[p1["segmento"] == "TOTAL"].iloc[0]
    stores = p2[p2["segmento"] == "TOTAL"].iloc[0]
    facts = deck_facts(curve, p2, stores_actuals, grids, index, meta)

    img_dir = out_path.parent / "images"
    charts = {
        "curva": ac.chart_risk_curve(curve, BIN_EDGES),
        "online": ac.chart_channel_actual(p1, "El riesgo de e-commerce se concentra en 'New'"),
        "tienda": ac.chart_stores_swap(p2),
        "tienda_compacto": ac.chart_stores_swap(p2, compact=True),
        "tienda_segmentos": ac.chart_stores_by_segment(p2),
        "tienda_actual": ac.chart_channel_actual(
            stores_actuals, "Tienda acepta casi el triple que Online y arriesga menos"
        ),
        "frontera": ac.chart_scenario_frontier(ladder, picks),
        "estacional": ac.chart_seasonality(index),
    }
    paths = {name: ac.save_chart(fig, img_dir / f"{name}.png") for name, fig in charts.items()}
    logger.info(f"{len(paths)} gráficos en {img_dir}")

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    today = date.today().strftime("%d-%m-%Y")

    # Orden del relato: resumen, los dos puntos medidos, los escenarios, la parrilla que
    # implican y cómo se ha calculado. Lo que sostiene las cifras va detrás, como anexo.
    slide_title(prs, online, stores, today)
    slide_exec(prs, online, stores, picks, paths["tienda_compacto"], meta, facts, p_mixto)
    slide_chart(
        prs,
        "Situación actual de Apple Online",
        "Punto 1 · Medido",
        paths["online"],
        "Riesgo realizado sobre originación con 6 meses cumplidos. Sin supuestos.",
        period=f"Originación {p_online}",
    )
    slide_chart(
        prs,
        "Situación actual de Apple Stores",
        "Punto 1 · Medido",
        paths["tienda_actual"],
        "Medido sobre el histórico propio de tienda, misma ventana que Online: para describir dónde está hoy "
        "no hace falta imputar nada. Acepta casi el triple que Online y arriesga menos. Aviso: su TA viene "
        f"cayendo ({ac.es(facts['ta_stores_window_pct'], 1, '%')} de {p_online}, "
        f"{ac.es(facts['ta_stores_aug_pct'], 1, '%')} en {p_tienda}), así que la base se está moviendo.",
        period=f"Originación {p_online}",
    )
    slide_chart(
        prs,
        "Tienda con la parrilla de e-commerce",
        "Punto 2 · Riesgo estimado",
        paths["tienda"],
        "Las dos columnas miden la misma población —el único mes con score EFX—, por eso aquí el 'hoy' es "
        f"{ac.es(facts['ta_stores_aug_pct'], 1, '%')} y en la slide anterior "
        f"{ac.es(facts['ta_stores_window_pct'], 1, '%')}. Sustitución de la regla de score; las demás reglas de tienda "
        "se mantienen. El riesgo va imputado desde la curva de Online, con el nivel anclado al realizado "
        "de tienda.",
        period=f"Tienda {p_tienda}",
    )
    slide_frontier(prs, paths["frontera"], picks, meta, p_mixto)
    slide_grids(prs, grids, "Política — sin periodo")
    slide_method_risk(prs, meta, facts, f"Online {p_online}  ·  tienda {p_tienda}")
    slide_method_seasonal(prs, meta, facts, f"Años completos {p_estacional}")
    slide_chart(
        prs,
        "El score EFX ordena el riesgo en e-commerce",
        "Anexo · El motor del estudio",
        paths["curva"],
        "La curva de riesgo por tramo es lo único medido del estudio. Todo lo demás —el riesgo de tienda, "
        "los escenarios— se deriva de ella.",
        period=f"Originación {p_online}",
    )
    slide_chart(
        prs,
        "De dónde sale la mejora en tienda",
        "Anexo · Desglose por segmento",
        paths["tienda_segmentos"],
        f"'New' concentra el {ac.es(facts['new_share_pct'], 0)}% de la demanda de tienda y es donde el riesgo baja. "
        "El take-up se mide por "
        "segmento y por decisión del motor (aprobada / en revisión). '>=G' no aparece: se rechaza siempre, así "
        "que no tiene ni aceptación ni riesgo.",
        period=f"Tienda {p_tienda}",
    )
    slide_chart(
        prs,
        "Los dos canales no comparten estacionalidad",
        "Anexo · Anualización",
        paths["estacional"],
        "El único mes con score en tienda es agosto, así que su peso en Apple depende de cómo se anualice. "
        "Tienda tiene su pico en diciembre y septiembre entre sus meses más flojos: usar la curva de Online "
        f"le habría inflado el peso al {ac.es(meta['peso_tienda_indice_online_pct'], 0)}% cuando con su propia "
        f"estacionalidad es el {ac.es(meta['peso_tienda_propio_pct'], 0)}%.",
        period=f"Años completos {p_estacional}",
    )
    slide_closing(prs, facts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="output/apple_study")
    p.add_argument("--output", default="output/apple_deck/Estudio_Apple_EFX.pptx")
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir)
    if not (data_dir / "p3_escenarios_objetivo.csv").exists():
        logger.error(f"No hay salidas en {data_dir}. Ejecuta antes: uv run python run_apple_study.py")
        return 1
    path = build(data_dir, Path(args.output))
    logger.info(f"Deck generado: {path}")
    print(f"\nDeck: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
