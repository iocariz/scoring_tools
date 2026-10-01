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
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

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

BIN_EDGES = [-np.inf, 2, 7, 11, 16, 22, 27, 33, 38, 43, 47, 51, 56, 62, 68, 73, 78, 83, 89, 94, np.inf]


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


def _header(slide, title: str, kicker: str = "") -> Emu:
    """Cabecera: antetítulo pequeño, título y filete. Devuelve la y donde empieza el cuerpo."""
    y = Inches(0.42)
    if kicker:
        _text(slide, MARGIN, y, W - 2 * MARGIN, Inches(0.3), kicker.upper(), size=11, bold=True, color=ACCENT)
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
    box = slide.shapes.add_textbox(x, y, w, Inches(0.4) * len(items))
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
        "El riesgo de tienda es estimado, no medido: no habrá medición hasta feb-2027.",
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


def slide_exec(prs, online, stores, picks, stores_img: Path) -> None:
    """Resumen ejecutivo en dos columnas: los números a la izquierda, el hallazgo a la derecha."""
    slide = _blank(prs)
    y = _header(slide, "Lo que dice el estudio", "Resumen ejecutivo")
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
    rows = [["Objetivo", "Corte EFX", "TA Apple", "Producción"]]
    for row in picks.dropna(subset=["corte_tramo"]).itertuples():
        rows.append(
            [
                ac.es(row.objetivo_pct, 1, "%"),
                f"> {ac.es(row.corte_score, 0)}",
                ac.es(row.ta_apple_pct, 1, "%"),
                f"{ac.es(row.produccion_mensual_eur / 1e6, 1)} M€/mes",
            ]
        )
    _table(slide, MARGIN, y2 + Inches(0.45), left_w, rows, col_w=[1, 1, 1, 1.25])

    _bullets(
        slide,
        MARGIN,
        y2 + Inches(2.5),
        left_w,
        [
            ("Riesgo de tienda:", "estimado desde la curva de e-commerce, no medido."),
            ("Ventana:", "12 meses. Con 21 cada objetivo saldría ~15 puntos de TA más barato."),
            ("Sin reject inference:", "los riesgos son optimistas; el pipeline los corrige al alza."),
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


def slide_method(prs) -> None:
    slide = _blank(prs)
    y = _header(slide, "Cómo se ha calculado", "Metodología")
    steps = [
        ("1 · Medir", "Riesgo realizado por tramo de score EFX en Apple Online, sobre cartera con 6 meses cumplidos."),
        ("2 · Imputar", "Esa curva se aplica a la población de tienda: mismo score, mismo riesgo por tramo."),
        ("3 · Combinar", "Los dos canales se agregan a run-rate mensual desestacionalizado para dar el total Apple."),
        ("4 · Optimizar", "Se recorre el corte único de EFX y se lee el que cumple cada objetivo de riesgo."),
    ]
    w, gap = Inches(3.02), Inches(0.18)
    for i, (title, body) in enumerate(steps):
        x = MARGIN + i * (w + gap)
        card = slide.shapes.add_shape(1, x, y, w, Inches(1.85))
        card.fill.solid()
        card.fill.fore_color.rgb = TILE_BG
        card.line.color.rgb = RULE
        card.shadow.inherit = False
        _text(
            slide,
            x + Inches(0.18),
            y + Inches(0.16),
            w - Inches(0.36),
            Inches(0.35),
            title,
            size=13,
            bold=True,
            color=ACCENT,
        )
        _text(
            slide,
            x + Inches(0.18),
            y + Inches(0.58),
            w - Inches(0.36),
            Inches(1.1),
            body,
            size=11.5,
            color=MUTED,
            spacing=1.2,
        )

    y2 = y + Inches(2.2)
    _text(slide, MARGIN, y2, W - 2 * MARGIN, Inches(0.4), "Decisiones tomadas, y por qué", size=15, bold=True)
    _bullets(
        slide,
        MARGIN,
        y2 + Inches(0.5),
        Inches(11.9),
        [
            (
                "Solo el corte de score.",
                "Las demás reglas de tienda (perfil, presupuesto, fraude) no cambian. "
                "El score explica 8 de los 55 puntos que Online rechaza: se mueve una palanca, no la política.",
            ),
            (
                "Ventana de 12 meses.",
                "La curva se ha desplazado al alza entre las dos mitades del periodo "
                "(3,86% a 21 meses frente a 4,21% a 12) y la decisión es prospectiva.",
            ),
            (
                "Julio-2026 excluido.",
                "Trae el 9% del volumen que le tocaría por estacionalidad: fue rollout parcial "
                "del score en tienda, no un mes de operación.",
            ),
            ("known_g siempre rechazado.", "Cuenta como demanda —baja la TA— pero nunca aporta producción ni riesgo."),
        ],
        size=12,
    )


def slide_chart(prs, title: str, kicker: str, img: Path, caption: str = "") -> None:
    slide = _blank(prs)
    y = _header(slide, title, kicker)
    body_h = H - y - (Inches(0.95) if caption else Inches(0.45))
    _picture(slide, img, y, body_h)
    if caption:
        _text(
            slide, MARGIN, H - Inches(0.86), W - 2 * MARGIN, Inches(0.6), caption, size=11.5, color=MUTED, spacing=1.2
        )


def slide_frontier(prs, img: Path, picks: pd.DataFrame) -> None:
    """La frontera lleva las cifras al lado, no dentro del gráfico: cuatro globos con
    tres cifras cada uno chocaban entre sí y tapaban la curva."""
    slide = _blank(prs)
    y = _header(slide, "Cuánto cuesta cada objetivo de riesgo", "Punto 3 · Escenarios")
    chart_w = Inches(8.35)
    slide.shapes.add_picture(str(img), MARGIN, y, width=chart_w, height=Emu(int(chart_w * _img_ratio(img))))

    x2 = MARGIN + chart_w + Inches(0.3)
    rows = [["Objetivo", "Corte", "TA", "Producción"]]
    for row in picks.dropna(subset=["corte_tramo"]).itertuples():
        rows.append(
            [
                ac.es(row.objetivo_pct, 1, "%"),
                f"> {ac.es(row.corte_score, 0)}",
                ac.es(row.ta_apple_pct, 1, "%"),
                f"{ac.es(row.produccion_mensual_eur / 1e6, 1)} M€",
            ]
        )
    _table(slide, x2, y + Inches(0.1), Inches(3.75), rows, col_w=[1, 0.85, 0.9, 1.15], row_h=0.38)
    _text(
        slide,
        x2,
        y + Inches(2.15),
        Inches(3.75),
        Inches(2.2),
        "La producción es run-rate mensual desestacionalizado.\n\n"
        "Un corte único para todo Apple cuesta producción frente a una parrilla por segmento: "
        "la parrilla puede ser laxa donde el segmento es bueno y dura donde es malo.",
        size=11.5,
        color=MUTED,
        spacing=1.25,
    )


def slide_closing(prs) -> None:
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
                "comportamiento observable. H3 hacia nov-2026, H6 hacia feb-2027.",
            ),
            (
                "Comparación condicional.",
                "El score interno y EFX se comparan con la vara de EFX. No demuestra "
                "que EFX sea mejor en tienda: lo asume.",
            ),
            (
                "Tramos bajos débiles.",
                "Los tramos 1-5 apenas tienen producción y son los que abren el escenario más laxo.",
            ),
            ("Estacionalidad prestada.", "El índice sale de Online; tienda podría ser aún más estacional."),
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
                "Refresco de Online a ago-2026, score EFX en tienda y su histórico de comportamiento "
                "para anclar el nivel de riesgo.",
            ),
            ("Decisión de negocio.", "Objetivo de riesgo global y si el corte es único o por segmento."),
            (
                "Validación.",
                "Repetir con el pipeline completo: reject inference, intervalos de confianza y "
                "auditoría swap-in / swap-out.",
            ),
            ("Medición.", "Revisar en nov-2026 con H3 y en feb-2027 con H6 reales de tienda."),
        ],
        size=12,
    )


# --------------------------------------------------------------------------- #
def build(data_dir: Path, out_path: Path) -> Path:
    read = lambda name: pd.read_csv(data_dir / f"{name}.csv")  # noqa: E731
    curve, p1, p2 = read("curva_riesgo_online"), read("p1_online_actual"), read("p2_tienda_parrilla_online")
    ladder, picks, index = read("p3_escalera_escenarios"), read("p3_escenarios_objetivo"), read("indice_estacional")
    online = p1[p1["segmento"] == "TOTAL"].iloc[0]
    stores = p2[p2["segmento"] == "TOTAL"].iloc[0]

    img_dir = out_path.parent / "images"
    charts = {
        "curva": ac.chart_risk_curve(curve, BIN_EDGES),
        "online": ac.chart_online_actual(p1),
        "tienda": ac.chart_stores_swap(p2),
        "tienda_compacto": ac.chart_stores_swap(p2, compact=True),
        "tienda_segmentos": ac.chart_stores_by_segment(p2),
        "frontera": ac.chart_scenario_frontier(ladder, picks),
        "estacional": ac.chart_seasonality(index),
    }
    paths = {name: ac.save_chart(fig, img_dir / f"{name}.png") for name, fig in charts.items()}
    logger.info(f"{len(paths)} gráficos en {img_dir}")

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    today = date.today().strftime("%d-%m-%Y")
    slide_title(prs, online, stores, today)
    slide_exec(prs, online, stores, picks, paths["tienda_compacto"])
    slide_method(prs)
    slide_chart(
        prs,
        "El score EFX ordena el riesgo en e-commerce",
        "El motor del estudio",
        paths["curva"],
        "La curva de riesgo por tramo es lo único medido del estudio. Todo lo demás —el riesgo de tienda, "
        "los escenarios— se deriva de ella.",
    )
    slide_chart(
        prs,
        "Situación actual de Apple Online",
        "Punto 1 · Medido",
        paths["online"],
        "Riesgo realizado sobre 12 meses de originación con 6 meses cumplidos. Sin supuestos.",
    )
    slide_chart(
        prs,
        "De dónde sale la mejora en tienda",
        "Punto 2 · Desglose por segmento",
        paths["tienda_segmentos"],
        "El cambio es una sustitución de la regla de score: las demás reglas de tienda se mantienen. "
        "'new' concentra el 82% de la demanda de tienda y es donde el riesgo baja. El take-up se mide por "
        "segmento. 'known_g' no aparece: se rechaza siempre, así que no tiene ni aceptación ni riesgo.",
    )
    slide_frontier(prs, paths["frontera"], picks)
    slide_chart(
        prs,
        "Por qué agosto no es un mes cualquiera",
        "Anualización",
        paths["estacional"],
        "El único mes con dato de tienda es el más bajo del año. Sin corregirlo, tienda parecía pesar el "
        "12% de Apple; desestacionalizada pesa el 31%, y eso cambia todos los escenarios.",
    )
    slide_closing(prs)

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
