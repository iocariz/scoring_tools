"""Apple study deck — extending the Equifax score from e-commerce to stores.

A variant of the results deck for this specific study. It does not reuse
``generate_results_presentation.py`` because that one assumes a story that does not
apply here: single-channel segments, a populated out-of-time backtest and measured
risk in every population. Here there are two channels, store risk is **imputed** and
the backtest has no mature cohort.

Reads the outputs of ``run_apple_study.py`` (``output/apple_study/*.csv``, written
with Spanish labels for the analyst) and writes a 14-slide ``.pptx`` in English.
Charts come from :mod:`src.apple_deck_charts`.

Usage:
    uv run python run_apple_study.py && uv run python generate_apple_deck.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from run_apple_study import BIN_EDGES, DEMAND_STATES, SAME_BASE_LABEL  # one definition, the study's
from src import apple_deck_charts as ac
from src.apple_deck_charts import CURRENT_GRID, SHIFTED_GRID, SINGLE_CUTOFF, TODAY_MEASURED, TODAY_SAME_BASE

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
INK = RGBColor(0x2C, 0x3E, 0x50)
MUTED = RGBColor(0x5D, 0x6D, 0x7E)
ACCENT = RGBColor(0x34, 0x98, 0xDB)
RISK = RGBColor(0xE7, 0x4C, 0x3C)
RULE = RGBColor(0xE5, 0xE7, 0xEB)
TILE_BG = RGBColor(0xF6, 0xF7, 0xF9)
FONT = "Arial"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LONG = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip


# --------------------------------------------------------------------------- #
# the study writes Spanish labels; the deck translates them once, on load
# --------------------------------------------------------------------------- #
_SCENARIOS = {
    "Hoy (medido)": TODAY_MEASURED,
    SAME_BASE_LABEL: TODAY_SAME_BASE,
    "Parrilla Online en ambos canales": CURRENT_GRID,
}
_CUTOFFS = {"política vigente": "current policy", "vigente": "current", "no alcanzable": "not reachable"}
_GRID_COLUMNS = {"grupo": "group", "Parrilla actual": "Current grid"}
_GRID_CELLS = {"rechazo": "reject", "sin corte": "no cutoff"}
_STATES = dict(
    zip(
        DEMAND_STATES,
        [
            "booked  (= acceptance rate)",
            "approved, then cancelled (unfinished demand)",
            "approved, rejected downstream",
            "rejected by the engine: score",
            "rejected by the engine: other rules",
        ],
        strict=True,
    )
)


def translate_outputs(picks: pd.DataFrame, grids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scenario and grid labels as the deck renders them.

    Everything downstream (charts, tables, ``deck_facts``) matches on the English
    labels, so the Spanish ones live in exactly one place.
    """
    picks = picks.copy()
    picks["escenario"] = picks["escenario"].map(
        lambda s: _SCENARIOS.get(
            s,
            s.replace(" · corte único", f" · {SINGLE_CUTOFF}").replace(" · parrilla desplazada", f" · {SHIFTED_GRID}"),
        )
    )
    picks["corte"] = picks["corte"].astype(str).map(lambda s: _CUTOFFS.get(s, s.replace(" puntos", " points")))
    grids = grids.rename(
        columns=lambda c: _GRID_COLUMNS.get(c, c.replace(" · único", " · single").replace(" · parrilla", " · shifted"))
    )
    return picks, grids.replace(_GRID_CELLS)


def money(eur: float, decimals: int = 1) -> str:
    """'€18.5M' from euros."""
    return f"€{ac.num(eur / 1e6, decimals)}M"


# --------------------------------------------------------------------------- #
# layout primitives
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


def span(start: str, end: str) -> str:
    """'2025-03-01','2026-03-01' -> 'Mar 2025 – Feb 2026'. The end is exclusive."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    last = date(b.year - 1, 12, 1) if b.month == 1 else date(b.year, b.month - 1, 1)
    if (a.year, a.month) == (last.year, last.month):
        return f"{MONTHS[a.month - 1]} {a.year}"
    if a.year == last.year:
        return f"{MONTHS[a.month - 1]}–{MONTHS[last.month - 1]} {a.year}"
    return f"{MONTHS[a.month - 1]} {a.year} – {MONTHS[last.month - 1]} {last.year}"


def _header(slide, title: str, kicker: str = "", period: str = "") -> Emu:
    """Header: kicker, period on the right, title and rule.

    The period goes on **every** data slide: the study mixes three windows (mature
    Online, scored stores, full years for seasonality) and without labelling them two
    correct figures from different periods read as contradictory.
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
    """Insert the PNG centred, scaled to fit the available height."""
    ratio = _img_ratio(path)
    w = W - 2 * MARGIN
    h = Emu(int(w * ratio))
    if h > max_h:
        h, w = max_h, Emu(int(max_h / ratio))
    slide.shapes.add_picture(str(path), Emu(int((W - w) / 2)), y, width=w, height=h)


def _tile(slide, x, y, w, h, label: str, value: str, note: str = "", value_color=INK) -> None:
    """Stat tile: label on top, big figure, note below. The figure is the chart."""
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
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
                for run in para.runs:
                    run.font.size, run.font.name = Pt(11.5), FONT
                    run.font.bold = r == 0
                    run.font.color.rgb = INK if r == 0 else MUTED
            cell.fill.solid()
            cell.fill.fore_color.rgb = TILE_BG if r == 0 else RGBColor(0xFF, 0xFF, 0xFF)


def _bullets(slide, x, y, w, items: list[tuple[str, str]], size: int = 13) -> None:
    """Bullets with a bold prefix: (prefix, body)."""
    # The box is declared with the height left down to the footer: PowerPoint grows it
    # on its own if the text does not fit, but declaring less leaves the geometry lying
    # about what it really occupies, and any renderer that does not grow would clip it.
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
        "RISK STUDY · APPLE",
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
        "Extending the Equifax score\nfrom e-commerce to stores",
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
        "Applying the Apple Online grid to Apple Stores gains acceptance and lowers risk at the same time. "
        "Store risk is still an estimate, but its level is now anchored to the delinquency stores have "
        "actually had: the only assumption left is that the score ranks the same way, not how much risk there is.",
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
        f"Risk  ·  {today}  ·  scope: Apple Propio (own channel, no APRS)",
        size=11,
        color=MUTED,
    )


def slide_exec(prs, online, stores, picks, stores_img: Path, meta: dict, facts: dict, period: str = "") -> None:
    """Executive summary in two columns: the numbers on the left, the finding on the right."""
    slide = _blank(prs)
    y = _header(slide, "What the study says", "Executive summary", period)
    w, gap = Inches(3.02), Inches(0.18)
    tiles = [
        ("Apple Online realized risk", ac.num(online["riesgo_real_pct"], 2, "%"), "measured on the mature book", RISK),
        ("Online acceptance rate", ac.num(online["ta_pct"], 1, "%"), "on € demand", INK),
        (
            "Stores with the EFX grid",
            ac.num(stores["ta_efectiva_pct"], 1, "%"),
            f"acceptance, from {ac.num(stores['ta_actual_pct'], 1, '%')} today",
            ACCENT,
        ),
        (
            "Stores imputed risk",
            ac.num(stores["riesgo_imputado_pct"], 2, "%"),
            f"from {ac.num(stores['riesgo_imputado_actual_pct'], 2, '%')} today",
            ACCENT,
        ),
    ]
    for i, (label, value, note, color) in enumerate(tiles):
        _tile(slide, MARGIN + i * (w + gap), y, w, Inches(1.62), label, value, note, color)

    y2 = y + Inches(1.95)
    left_w, right_x, right_w = Inches(6.15), MARGIN + Inches(6.45), Inches(5.65)

    _text(slide, MARGIN, y2, left_w, Inches(0.4), "Global Apple risk scenarios", size=15, bold=True)
    # Only the single cutoff in the summary; the shifted-grid variant goes on the point-3
    # slide. The EFFECTIVE acceptance is published, the one comparable with point 1.
    rows = [["Scenario", "Cutoff", "Eff. acc.", "Production"]]
    for row in picks.dropna(subset=["riesgo_total_pct"]).itertuples():
        if SHIFTED_GRID in row.escenario or row.escenario == TODAY_SAME_BASE:
            continue
        rows.append(
            [
                row.escenario.replace(f" · {SINGLE_CUTOFF}", ""),
                str(row.corte),
                ac.num(row.ta_efectiva_pct, 1, "%"),
                f"{money(row.produccion_mensual_eur)}/mo",
            ]
        )
    # Column 0 is sized for its longest label at 11.5 pt: a wrapped cell grows the row into the bullets below.
    _table(slide, MARGIN, y2 + Inches(0.45), left_w, rows, col_w=[2.1, 0.9, 0.9, 1.1], row_h=0.30)

    _bullets(
        slide,
        MARGIN,
        y2 + Inches(2.55),
        left_w,
        [
            (
                "Store risk:",
                "curve shape taken from e-commerce; level anchored to stores' realized "
                f"{ac.num(meta['riesgo_tienda_realizado_pct'], 2, '%')}. Not a measurement.",
            ),
            (
                "Window:",
                f"{facts['online_months']} months of mature originations ({ac.num(online['riesgo_real_pct'], 2, '%')} "
                f"vs {ac.num(meta['riesgo_online_historico_pct'], 2, '%')} over "
                f"{meta['online_historico_meses']} months).",
            ),
            (
                "With reject inference:",
                f"thinly accepted bins carry an uplift (up to {ac.num(facts['max_uplift'], 1)}x). It barely moves "
                f"the cutoffs: where the decision sits, Online already accepts {ac.num(facts['zone_lo_pct'], 0)}-"
                f"{ac.num(facts['zone_hi_pct'], 0)}%.",
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
        "Stores: the e-commerce grid improves both",
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
    """Vertical chain of steps: number, title, body and nature tag.

    The tag (MEASURED / CORRECTED / ASSUMED) is the point of the slide: in a study
    where one part comes from the data and another from an assumption, what a
    committee needs to see at a glance is which is which, step by step.
    """
    tag_colors = {"MEASURED": ACCENT, "CORRECTED": MUTED, "ASSUMED": RISK}
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
    """How risk is estimated: the full chain, with measured and assumed kept apart."""
    slide = _blank(prs)
    y = _header(slide, "How store risk is estimated", "Methodology · Risk", period)
    factor = meta.get("factor_nivel_tienda", 1.0)
    realized = meta.get("riesgo_tienda_realizado_pct", float("nan"))
    imputed = meta.get("riesgo_tienda_imputado_pct", float("nan"))
    _chain(
        slide,
        MARGIN,
        y,
        W - 2 * MARGIN,
        [
            (
                "1",
                "Realized risk by score bin, in Apple Online",
                "For each of the 20 EFX bins, the observed delinquency of contracts with 6 months on book. "
                "The only part of the study that comes straight from the data.",
                "MEASURED",
            ),
            (
                "2",
                "Selection correction (reject inference)",
                "Booked contracts in a bin also passed every other rule: they are a selected sample and their "
                "delinquency understates that of applicants rejected today. The pipeline's parceling is applied, "
                f"with an uplift driven by the acceptance rate: {ac.num(facts['max_uplift'], 1)}x where only "
                f"{ac.num(facts['min_acc_pct'], 0)}% get in, 1.0x where {ac.num(facts['max_acc_pct'], 0)}% do.",
                "CORRECTED",
            ),
            (
                "3",
                "That curve is carried over to the store population",
                "Each store application is assigned the risk of the EFX bin it falls in. This is the underlying "
                "assumption: that the score ranks store applicants the same way it ranks e-commerce ones.",
                "ASSUMED",
            ),
            (
                "4",
                "The level is anchored to stores' actual delinquency",
                f"The Online curve predicts {ac.num(imputed, 2, '%')} on the book stores actually wrote; the observed "
                f"figure is {ac.num(realized, 2, '%')}. The curve is rescaled by {ac.num(factor, 3)} to reproduce the "
                "data. The level is no longer an assumption: only the ranking power remains assumed.",
                "MEASURED",
            ),
        ],
    )
    _text(
        slide,
        MARGIN,
        H - Inches(1.15),
        W - 2 * MARGIN,
        Inches(0.9),
        "What this means in practice: if EFX ranks store applicants worse than it ranks Online ones, the "
        "scenario cutoffs fall short. What can no longer be wrong is the height of the curve, because it is "
        "calibrated against the delinquency stores have actually had.",
        size=11.5,
        color=MUTED,
        spacing=1.2,
    )


def slide_method_seasonal(prs, meta: dict, facts: dict, period: str) -> None:
    """How months are annualised: why a month is not a month and what is done about it."""
    slide = _blank(prs)
    y = _header(slide, "How months are annualised", "Methodology · Seasonality", period)
    n = facts["stores_n_months"]
    _chain(
        slide,
        MARGIN,
        y,
        W - 2 * MARGIN,
        [
            (
                "1",
                f"The problem: stores only have the score for {facts['stores_window']}",
                f"The EFX score went live in stores in {facts['stores_first_month']}. Comparing "
                f"{'a single month on its own' if n == 1 else f'{n} months on their own'} against a twelve-month average requires "
                "knowing what each month is worth.",
                "MEASURED",
            ),
            (
                "2",
                f"{facts['stores_months_txt']}: no average month, and each channel has its own",
                f"Index for {facts['stores_months_short']}: {facts['online_idx_txt']} in Online and "
                f"{facts['stores_idx_txt']} in stores. Online concentrates demand around the iPhone launch "
                f"({facts['peak_online'][0]} {ac.num(facts['peak_online'][1], 2)}) and stores around Christmas "
                f"({facts['peak_stores'][0]} {ac.num(facts['peak_stores'][1], 2)}). The correlation between the two "
                f"profiles is {ac.num(facts['index_corr'], 2)}: not the same pattern.",
                "MEASURED",
            ),
            (
                "3",
                "Effective months instead of calendar months",
                "The run-rate is divided not by the number of months but by the sum of the seasonal index over the "
                "months covered. A full calendar year sums to 12.00 and changes nothing; the stores window "
                f"({facts['stores_window']}) sums to {ac.num(facts['stores_eff_months'], 2)} and corrects for falling "
                "in weak store months.",
                "CORRECTED",
            ),
            (
                "4",
                "Effect on stores' weight in Apple Total",
                f"Uncorrected, stores appeared to be {ac.num(meta['peso_tienda_calendario_pct'], 0)}% of demand. With "
                f"their own seasonality they are {ac.num(meta['peso_tienda_propio_pct'], 0)}%. Had Online's profile "
                "been applied —the only option before stores' history was available— we would have said "
                f"{ac.num(meta['peso_tienda_indice_online_pct'], 0)}%, and every scenario would show "
                f"{'more' if meta['peso_tienda_indice_online_pct'] > meta['peso_tienda_propio_pct'] else 'less'} "
                "production than there really is.",
                "MEASURED",
            ),
        ],
    )
    _text(
        slide,
        MARGIN,
        H - Inches(1.15),
        W - 2 * MARGIN,
        Inches(0.9),
        "The index averages complete calendar years, so year-on-year volume growth is not mistaken for "
        f"seasonality. What remains open is whether the {facts['stores_window']} mix in stores is representative "
        "of the year.",
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
    """The frontier carries its figures beside the chart, not inside it: four callouts
    with three figures each collided and covered the curve."""
    slide = _blank(prs)
    y = _header(slide, "What each risk target costs", "Point 3 · Scenarios", period)
    chart_w = Inches(7.35)
    chart_h = Emu(int(chart_w * _img_ratio(img)))
    slide.shapes.add_picture(str(img), MARGIN, y, width=chart_w, height=chart_h)

    x2 = MARGIN + chart_w + Inches(0.28)
    rows = [["Scenario", "Cutoff", "Score acc.", "Eff. acc."]]
    for row in picks.dropna(subset=["riesgo_total_pct"]).itertuples():
        rows.append(
            [
                row.escenario.replace(f" · {SINGLE_CUTOFF}", " · single")
                .replace(f" · {SHIFTED_GRID}", " · shifted")
                .replace(TODAY_SAME_BASE, "Today, current demand")
                .replace(CURRENT_GRID, "Online grid (both)"),
                str(row.corte).replace(" points", "p"),
                "—" if pd.isna(row.ta_score_pct) else ac.num(row.ta_score_pct, 1, "%"),
                ac.num(row.ta_efectiva_pct, 1, "%"),
            ]
        )
    _table(slide, x2, y + Inches(0.1), Inches(4.75), rows, col_w=[2.0, 0.9, 0.85, 0.9], row_h=0.26)

    # The note goes under the chart, where there is room and where whoever reads the
    # curve will see it. It is the question that will come up in committee: why the
    # model produces more than what was actually produced.
    base_rows = picks[picks["escenario"].str.startswith(CURRENT_GRID)]
    base_prod = float(base_rows.iloc[0]["produccion_mensual_eur"]) if not base_rows.empty else float("nan")
    today_rows = picks[picks["escenario"] == TODAY_MEASURED]
    today_prod = float(today_rows.iloc[0]["produccion_mensual_eur"]) if not today_rows.empty else float("nan")
    same_rows = picks[picks["escenario"] == TODAY_SAME_BASE]
    same_prod = float(same_rows.iloc[0]["produccion_mensual_eur"]) if not same_rows.empty else float("nan")
    stores_window = span(meta["tienda_desde"], meta["tienda_hasta"])
    if pd.notna(same_prod):
        split = (
            f"That gap has two parts: the grid (+{money(base_prod - same_prod)}/mo on the same base, "
            f"{money(same_prod)} → {money(base_prod)}) and volume "
            f"({'+' if same_prod >= today_prod else '−'}{money(abs(same_prod - today_prod))}: store demand in "
            f"{stores_window} runs {ac.num(meta.get('demanda_tienda_vs_ventana_pct', float('nan')), 0)}% above "
            "the average of the 'today' window)."
        )
    else:
        split = f"It is therefore the stores gain, measured on {stores_window} against the full 'today' window."
    # Scenario rows carry the reject-inference uplift, which also falls on Online's own
    # book; the 'today' rows are realized. Said explicitly, or the table sells a basis
    # difference as an effect of the grid.
    same_risk = float(same_rows.iloc[0]["riesgo_total_pct"]) if not same_rows.empty else float("nan")
    grid_today_basis = meta.get("riesgo_parrilla_base_hoy_pct", float("nan"))
    ri_margin = meta.get("ri_margen_online_pp", float("nan"))
    if pd.notna(same_risk) and pd.notna(grid_today_basis) and pd.notna(ri_margin):
        ri_note = (
            "\n\nScenario risks carry the reject-inference uplift, a conservative margin that also falls on "
            f"Online's own book (+{ac.num(ri_margin, 2)} pp vs realized); the two 'today' rows are realized. Like "
            f"for like, the current grid moves risk by {grid_today_basis - same_risk:+.2f} pp."
        )
    else:
        ri_note = ""
    # Below the chart, computed from its real height: eyeballing it made it collide with
    # the axis label.
    note_y = y + chart_h + Inches(0.12)
    _text(
        slide,
        MARGIN,
        note_y,
        chart_w,
        Inches(0.32),
        "Why the model produces more than what was measured",
        size=12.5,
        bold=True,
        color=RISK,
    )
    _text(
        slide,
        MARGIN,
        note_y + Inches(0.34),
        chart_w,
        Inches(1.4),
        "The current grid replicates Online's actual decisions (it accepts "
        f"{ac.num(meta['online_parrilla_pct'], 1, '%')} of demand; the engine approved "
        f"{ac.num(meta['online_aprobado_real_pct'], 1, '%')}: swap-in {ac.num(meta['online_swap_in_pct'], 1, '%')}, "
        f"swap-out {ac.num(meta['online_swap_out_pct'], 1, '%')}) and every accepted application converts at the "
        f"take-up rate of its actual decision. Modelled and measured Online coincide: the gap between "
        f"{money(base_prod)} and {money(today_prod)} per month is stores. {split}",
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
        "Score acc.: demand above the cutoff.\n"
        "Eff. acc.: production / demand, the same base as point 1. The two are not comparable.\n\n"
        "Shifting the current grid keeps its shape —looser where the group is good— and gives slightly more "
        f"acceptance at the same risk.{ri_note}",
        size=10.5,
        color=MUTED,
        spacing=1.25,
    )


def slide_grids(prs, grids: pd.DataFrame, period: str = "") -> None:
    """The resulting grids: the deliverable someone has to implement.

    Two tables instead of one with ten columns: one instrument each, with the current
    grid repeated in both as the reference.
    """
    slide = _blank(prs)
    y = _header(slide, "What grid each scenario implies", "Point 4 · Grids", period)
    _text(
        slide,
        MARGIN,
        y,
        W - 2 * MARGIN,
        Inches(0.35),
        "REJECT threshold by group: applications above the value are accepted.",
        size=11.5,
        color=MUTED,
    )
    y += Inches(0.45)

    for title, keyword in (("Single cutoff for all of Apple", "single"), ("Current grid, shifted", "shifted")):
        cols = ["group", "Current grid"] + [c for c in grids.columns if c.endswith(keyword)]
        rows = [["Group", "Today"] + [c.split(" ·")[0] for c in cols[2:]]]
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
        "A single cutoff is simpler to operate; shifting the grid keeps its shape —looser where the group is "
        "good— and gives slightly more acceptance at the same risk. '>=G' is always rejected in both.",
        size=11,
        color=MUTED,
        spacing=1.2,
    )


def slide_closing(prs, meta: dict, facts: dict) -> None:
    slide = _blank(prs)
    y = _header(slide, "What is missing and what needs deciding", "Limitations and next steps")
    _text(slide, MARGIN, y, Inches(5.9), Inches(0.4), "Study limitations", size=15, bold=True, color=RISK)
    n = facts["stores_n_months"]
    _bullets(
        slide,
        MARGIN,
        y + Inches(0.5),
        Inches(5.9),
        [
            (
                "Store risk is estimated.",
                "Store contracts with EFX are 0–2 months old and have no observable performance yet. "
                f"H3 around {facts['h3_month']}, H6 around {facts['h6_month']}.",
            ),
            (
                "Conditional comparison.",
                "The internal score and EFX are compared with EFX's yardstick. It does not prove EFX is better "
                "in stores: it assumes it. What is checked is the risk level, not the ranking power.",
            ),
            (
                "Weak low bins.",
                f"Bins 1-{facts['low_bins']} carry under 0.5% of production each, and that is where the selection uplift "
                f"weighs most ({ac.num(facts['max_uplift'], 1)}x). They are the ones that open the loosest scenario.",
            ),
            (
                f"Only {facts['stores_window']} in stores.",
                f"Their own seasonality is already measured, but a {n}-month mix may not be representative, and "
                f"store demand runs {ac.num(meta.get('demanda_tienda_vs_ventana_pct', float('nan')), 0)}% above the "
                "average of the 'today' window: part of the improvement versus what was measured is volume, not grid.",
            ),
        ],
        size=12,
    )

    _text(slide, MARGIN + Inches(6.3), y, Inches(5.9), Inches(0.4), "Next steps", size=15, bold=True, color=ACCENT)
    _bullets(
        slide,
        MARGIN + Inches(6.3),
        y + Inches(0.5),
        Inches(5.9),
        [
            (
                "Extraction.",
                f"Refresh Online through {facts['stores_last_month']}, EFX score in stores and their performance "
                "history to anchor the risk level.",
            ),
            ("Business decision.", "Global risk target, and whether the cutoff is single or by segment."),
            (
                "Validation.",
                "Rerun with the full pipeline: reject inference, confidence intervals and swap-in / swap-out audit.",
            ),
            ("Measurement.", f"Review in {facts['h3_month']} with H3 and in {facts['h6_month']} with actual store H6."),
        ],
        size=12,
    )


def slide_definitions(prs, breakdown: pd.DataFrame, columns: dict[str, str], period: str) -> None:
    """What 'acceptance rate' means and where unfinished demand goes.

    The two questions the business asked after the first read. The table IS the
    definition: every € of demand lands in exactly one row, and the acceptance rate is
    the first row over the total. The extra rows show why two channels with similar
    engine approval end up with very different acceptance: take-up.
    """
    slide = _blank(prs)
    y = _header(slide, "How acceptance is measured", "Definitions", period)
    table = breakdown.pivot(index="estado", columns="canal", values="pct_demanda").reindex(list(DEMAND_STATES))
    approval = table.iloc[:3].sum()
    take_up = 100 * table.iloc[0] / approval
    rows = [["Where each € of demand ends up"] + list(columns.values())]
    for estado, label in _STATES.items():
        rows.append([label] + [ac.num(table.loc[estado, c], 1, "%") for c in columns])
    rows.append(["Total demand"] + ["100.0%"] * len(columns))
    rows.append(["Engine approval (first three rows)"] + [ac.num(approval[c], 1, "%") for c in columns])
    rows.append(["Take-up: booked / approved"] + [ac.num(take_up[c], 0, "%") for c in columns])
    # Column widths measured for the longest label at 11.5 pt; headers are two lines
    # (channel / period) so the period never has to fit on one.
    _table(slide, MARGIN, y, Inches(7.5), rows, col_w=[3.45, 1.35, 1.35, 1.35], row_h=0.34)

    _bullets(
        slide,
        MARGIN + Inches(7.8),
        y,
        Inches(4.25),
        [
            (
                "Acceptance rate (effective).",
                "€ booked / € applied for, over every application that comes through the door — rejected, "
                "cancelled and booked alike. The first row of the table. Point 1 and every scenario use it.",
            ),
            (
                "Score acceptance.",
                "€ above the score cutoff and not rejected by another rule, over € demand. Comparable with the "
                "engine's approval rate, not with the effective one; shown only in the scenario table.",
            ),
            (
                "Unfinished demand.",
                "Approved by the engine, then cancelled by the customer. It counts in the denominator and not in "
                "the numerator, like a rejection. In the scenarios every accepted application converts at the "
                "actual take-up of its decision, so cancellations are assumed to behave as today: a looser grid "
                "does not recover them.",
            ),
            (
                "Approved, rejected downstream.",
                "Approved by the engine, rejected at a later step (no reason in the extract). Treated like "
                "unfinished demand.",
            ),
            (
                "Reject inference.",
                "Neither state enters the per-bin acceptance rate — booked / (booked + score-rejected) — so they "
                "do not inflate the uplift.",
            ),
        ],
        size=10.5,
    )


def deck_facts(curve, p2, stores_actuals, grids, index, meta) -> dict:
    """The figures the deck's prose cites, computed from the study outputs.

    Written into the slides they drift on the first run that changes anything; here
    each one has its source. The only literals left in the deck are structural (20
    bins, a calendar year sums to 12.00).
    """
    uplift = curve["b2_pct_rechazados"] / curve["b2_pct_contratados"]
    acceptance = 100 * curve["tasa_aceptacion"]
    # "where the decision sits": the bins the scenarios' single cutoffs fall in
    cuts = [
        int(v.strip("> ")) for col in grids.columns if "single" in col for v in grids[col] if str(v).startswith(">")
    ]
    lower_edge = curve["bin"].map(lambda b: BIN_EDGES[int(b) - 1])
    zone = acceptance[(lower_edge >= min(cuts)) & (lower_edge <= max(cuts))]
    stores_total = p2[p2["segmento"] == "TOTAL"].iloc[0]
    new = p2[p2["segmento"].str.lower() == "new"]
    last_stores_month = pd.Timestamp(meta["tienda_hasta"]) - pd.Timedelta(days=1)
    stores_months = pd.period_range(meta["tienda_desde"], last_stores_month, freq="M")
    idx = index.set_index("mes")
    stores_idx = [float(idx.loc[p.month, "indice_tienda"]) for p in stores_months]
    online_idx = [float(idx.loc[p.month, "indice"]) for p in stores_months]

    def month(ts: pd.Timestamp) -> str:
        return f"{MONTHS[ts.month - 1]} {ts.year}"

    peak_online, peak_stores = int(index["indice"].idxmax()), int(index["indice_tienda"].idxmax())
    weak_stores = int(index["indice_tienda"].idxmin())
    return {
        "max_uplift": float(uplift.max()),
        "min_acc_pct": float(acceptance.min()),
        "max_acc_pct": float(acceptance.max()),
        "zone_lo_pct": float(zone.min()),
        "zone_hi_pct": float(zone.max()),
        # same criterion as the shading on the risk-curve chart (production under 0.5% of the total)
        "low_bins": int(curve.loc[curve["booked_eur"] < 0.005 * curve["booked_eur"].sum(), "bin"].max()),
        "new_share_pct": 100 * float(new["demanda_eur"].sum()) / float(stores_total["demanda_eur"]),
        "ta_stores_window_pct": float(stores_actuals.set_index("segmento").loc["TOTAL", "ta_pct"]),
        "ta_stores_recent_pct": float(stores_total["ta_actual_pct"]),
        "online_months": (pd.Period(meta["online_hasta"], "M") - pd.Period(meta["online_desde"], "M")).n,
        "peak_online": (MONTHS[int(index.loc[peak_online, "mes"]) - 1], float(index.loc[peak_online, "indice"])),
        "peak_stores": (MONTHS[int(index.loc[peak_stores, "mes"]) - 1], float(index.loc[peak_stores, "indice_tienda"])),
        "peak_stores_long": MONTHS_LONG[int(index.loc[peak_stores, "mes"]) - 1],
        "weak_stores_long": MONTHS_LONG[int(index.loc[weak_stores, "mes"]) - 1],
        "index_corr": float(index["indice"].corr(index["indice_tienda"])),
        "stores_first_month": month(pd.Timestamp(meta["tienda_desde"])),
        "stores_last_month": month(last_stores_month),
        "stores_window": span(meta["tienda_desde"], meta["tienda_hasta"]),
        "stores_n_months": len(stores_months),
        "stores_months_txt": " and ".join(MONTHS_LONG[p.month - 1] for p in stores_months),
        "stores_months_short": " and ".join(MONTHS[p.month - 1] for p in stores_months),
        "stores_idx_txt": " / ".join(ac.num(v, 2) for v in stores_idx),
        "online_idx_txt": " / ".join(ac.num(v, 2) for v in online_idx),
        "stores_eff_months": float(sum(stores_idx)),
        "h3_month": month(last_stores_month + pd.DateOffset(months=3)),
        "h6_month": month(last_stores_month + pd.DateOffset(months=6)),
    }


# --------------------------------------------------------------------------- #
def build(data_dir: Path, out_path: Path) -> Path:
    read = lambda name: pd.read_csv(data_dir / f"{name}.csv")  # noqa: E731
    curve, p1, p2 = read("curva_riesgo_online"), read("p1_online_actual"), read("p2_tienda_parrilla_online")
    stores_actuals = read("p1b_tienda_actual")
    meta = json.loads((data_dir / "periodos.json").read_text(encoding="utf-8"))
    p_online = span(meta["online_desde"], meta["online_hasta"])
    p_stores = span(meta["tienda_desde"], meta["tienda_hasta"])
    p_mixed = f"Online {p_online}  ·  stores {p_stores}"
    p_seasonal = " and ".join(str(a) for a in meta["estacionalidad_anios"])
    ladder, index = read("p3_escalera_corte_unico"), read("indice_estacional")
    breakdown = read("p0_demanda_por_estado")
    picks, grids = translate_outputs(read("p3_escenarios_objetivo"), read("p4_parrillas_escenarios"))
    online = p1[p1["segmento"] == "TOTAL"].iloc[0]
    stores = p2[p2["segmento"] == "TOTAL"].iloc[0]
    facts = deck_facts(curve, p2, stores_actuals, grids, index, meta)

    img_dir = out_path.parent / "images"
    charts = {
        "curve": ac.chart_risk_curve(curve, BIN_EDGES),
        "online": ac.chart_channel_actual(p1, "E-commerce risk is concentrated in 'New'"),
        "stores": ac.chart_stores_swap(p2),
        "stores_compact": ac.chart_stores_swap(p2, compact=True),
        "stores_segments": ac.chart_stores_by_segment(p2),
        "stores_today": ac.chart_channel_actual(
            stores_actuals, "Stores accept almost three times as much as Online and carry less risk"
        ),
        "frontier": ac.chart_scenario_frontier(ladder, picks),
        "seasonality": ac.chart_seasonality(index),
    }
    paths = {name: ac.save_chart(fig, img_dir / f"{name}.png") for name, fig in charts.items()}
    logger.info(f"{len(paths)} charts in {img_dir}")

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    today = date.today().strftime("%d %b %Y")

    # Story order: summary, the two measured points, the scenarios, the grid they imply
    # and how it was computed. What backs the figures goes last, as an appendix.
    slide_title(prs, online, stores, today)
    slide_exec(prs, online, stores, picks, paths["stores_compact"], meta, facts, p_mixed)
    short = lambda period: re.sub(r"\b20(\d\d)\b", r"\1", period)  # noqa: E731  — fits a 1.35" column in bold
    slide_definitions(
        prs,
        breakdown,
        {
            "online": f"Online\n{short(p_online)}",
            "tienda_ventana": f"Stores\n{short(p_online)}",
            "tienda_score": f"Stores\n{short(p_stores)}",
        },
        p_mixed,
    )
    slide_chart(
        prs,
        "Apple Online today",
        "Point 1 · Measured",
        paths["online"],
        "Realized risk on originations with 6 months on book. No assumptions.",
        period=f"Originations {p_online}",
    )
    slide_chart(
        prs,
        "Apple Stores today",
        "Point 1 · Measured",
        paths["stores_today"],
        "Measured on stores' own history, same window as Online: describing where they stand today needs no "
        "imputation. They accept almost three times as much as Online and carry less risk. Warning: their "
        f"acceptance rate is falling ({ac.num(facts['ta_stores_window_pct'], 1, '%')} over {p_online}, "
        f"{ac.num(facts['ta_stores_recent_pct'], 1, '%')} in {p_stores}), so the baseline is moving.",
        period=f"Originations {p_online}",
    )
    slide_chart(
        prs,
        "Stores with the e-commerce grid",
        "Point 2 · Estimated risk",
        paths["stores"],
        f"Both columns measure the same population —{facts['stores_window']}, the only period with an EFX score "
        f"in stores— which is why 'today' here is {ac.num(facts['ta_stores_recent_pct'], 1, '%')} and on the "
        f"previous slide {ac.num(facts['ta_stores_window_pct'], 1, '%')}. Only the score rule is swapped; stores' "
        "other rules stay. Risk is imputed from the Online curve, with the level anchored to stores' realized "
        "figure.",
        period=f"Stores {p_stores}",
    )
    slide_frontier(prs, paths["frontier"], picks, meta, p_mixed)
    slide_grids(prs, grids, "Policy — no period")
    slide_method_risk(prs, meta, facts, f"Online {p_online}  ·  stores {p_stores}")
    slide_method_seasonal(prs, meta, facts, f"Full years {p_seasonal}")
    slide_chart(
        prs,
        "The EFX score ranks risk in e-commerce",
        "Appendix · The engine of the study",
        paths["curve"],
        "The risk curve by bin is the only measured piece of the study. Everything else —store risk, the "
        "scenarios— derives from it.",
        period=f"Originations {p_online}",
    )
    slide_chart(
        prs,
        "Where the stores improvement comes from",
        "Appendix · Breakdown by segment",
        paths["stores_segments"],
        f"'New' is {ac.num(facts['new_share_pct'], 0)}% of store demand and where risk falls. Take-up is measured "
        "by segment and by engine decision (approved / referred). '>=G' is absent: it is always rejected, so it "
        "has neither acceptance nor risk.",
        period=f"Stores {p_stores}",
    )
    slide_chart(
        prs,
        "The two channels do not share a seasonality",
        "Appendix · Annualisation",
        paths["seasonality"],
        f"Stores only have the score for {facts['stores_window']}, so their weight in Apple depends on how it is "
        f"annualised. Stores peak in {facts['peak_stores_long']} and {facts['weak_stores_long']} is their weakest "
        "month, the opposite of Online: using Online's curve would have "
        f"{'inflated' if meta['peso_tienda_indice_online_pct'] > meta['peso_tienda_propio_pct'] else 'cut'} their "
        f"weight to {ac.num(meta['peso_tienda_indice_online_pct'], 0)}% when with their own seasonality it is "
        f"{ac.num(meta['peso_tienda_propio_pct'], 0)}%.",
        period=f"Full years {p_seasonal}",
    )
    slide_closing(prs, meta, facts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="output/apple_study")
    p.add_argument("--output", default="output/apple_deck/Apple_EFX_Study.pptx")
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir)
    if not (data_dir / "p3_escenarios_objetivo.csv").exists():
        logger.error(f"No study outputs in {data_dir}. Run first: uv run python run_apple_study.py")
        return 1
    path = build(data_dir, Path(args.output))
    logger.info(f"Deck written: {path}")
    print(f"\nDeck: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
