"""Deck de apoyo a la respuesta a la dirección de riesgos (dudas del 18-11-2026).

Lee las tablas de seguimiento del estudio Apple (``b1``–``b4_*.csv``, ver
:mod:`src.studies.apple.followup`) más ``p3_escenarios_objetivo.csv`` y ``periodos.json`` y
monta un ``.pptx`` **en castellano** —acompaña a ``reports/correo_bernard_dudas_apple.md``,
no al deck de negocio, que va en inglés—. Un gráfico donde la forma es el mensaje
(distribución, serie temporal, gradiente) y una tabla donde lo que se lee son cifras.

Mismas reglas de gráfico que :mod:`src.studies.apple.deck_charts` (paleta validada, un eje
por panel, etiquetas directas selectivas, rejilla recesiva); el rojo marca la población
problema y nunca comparte panel con un verde.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from src.studies.apple.deck_charts import CURRENT, GRID, INK, MUTED, PROPOSED, RISK, _style, save_chart
from src.studies.apple.followup import INCOME_LABELS, SCORE_RANGE_LABELS
from src.studies.apple.study import SAME_BASE_LABEL, SWAP_CATEGORIES

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
C_INK, C_MUTED = RGBColor(0x2C, 0x3E, 0x50), RGBColor(0x5D, 0x6D, 0x7E)
C_ACCENT, C_RISK = RGBColor(0x34, 0x98, 0xDB), RGBColor(0xE7, 0x4C, 0x3C)
C_RULE, C_TILE = RGBColor(0xE5, 0xE7, 0xEB), RGBColor(0xF6, 0xF7, 0xF9)
FONT = "Arial"

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# Ingresos: una sola magnitud ordenada -> rampa secuencial de un solo tono (claro = bajo).
INCOME_RAMP = ["#A9CCE3", "#7FB3D5", "#5499C7", "#2E86C1", "#1B4F72"]

INCOME_TXT = dict(
    zip(INCOME_LABELS, ["< 1.000 €", "1.000-1.499 €", "1.500-1.999 €", "2.000-2.999 €", "≥ 3.000 €"], strict=True)
)


def es(value: float, decimals: int = 1, suffix: str = "") -> str:
    """Formato castellano: coma decimal. NaN -> raya."""
    return "—" if pd.isna(value) else f"{value:.{decimals}f}".replace(".", ",") + suffix


def meur(value: float, decimals: int = 1) -> str:
    return "—" if pd.isna(value) else f"{es(value / 1e6, decimals)} M€"


def _comma_ticks(ax, decimals: int = 1) -> None:
    """Eje y con coma decimal, como el resto del deck."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: es(v, decimals)))


def quarter_label(q: str) -> str:
    """'2026Q2' -> 'T2 26'."""
    return f"T{q[-1]} {q[2:4]}"


def span_es(start: str, end: str) -> str:
    """'2026-08-01','2026-11-01' -> 'ago-oct 2026' (fin exclusivo)."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    last = date(b.year - 1, 12, 1) if b.month == 1 else date(b.year, b.month - 1, 1)
    if a.year == last.year:
        return f"{MESES[a.month - 1]}-{MESES[last.month - 1]} {a.year}"
    return f"{MESES[a.month - 1]} {a.year} - {MESES[last.month - 1]} {last.year}"


# --------------------------------------------------------------------------- #
# gráficos
# --------------------------------------------------------------------------- #
def chart_score_distribution(b1: pd.DataFrame, cutoff_bin: int, cutoff_score: float) -> plt.Figure:
    """Demanda por tramo EFX: tienda frente a Online. La forma (sin picos, cargada arriba) es el mensaje."""
    fig, ax = plt.subplots(figsize=(12.2, 4.3))
    x = np.arange(len(b1))
    ax.bar(x - 0.2, b1["online_demanda_pct"], 0.38, color=CURRENT, label="Online (ventana madura)")
    ax.bar(x + 0.2, b1["tienda_demanda_pct"], 0.38, color=PROPOSED, label="Tienda (meses con score)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(a)}-{int(b)}" for a, b in zip(b1["score_min"], b1["score_max"], strict=True)])
    edge = cutoff_bin - 0.5
    ax.axvline(edge, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(
        edge + 0.12,
        ax.get_ylim()[1] * 0.93,
        f"corte de clientes nuevos (> {cutoff_score:.0f})",
        fontsize=9,
        color=INK,
        va="top",
    )
    below_s = b1.loc[b1["tramo"] <= cutoff_bin, "tienda_demanda_pct"].sum()
    below_o = b1.loc[b1["tramo"] <= cutoff_bin, "online_demanda_pct"].sum()
    ax.text(
        edge - 0.12,
        ax.get_ylim()[1] * 0.93,
        f"por debajo: tienda {es(below_s)}%\nOnline {es(below_o)}%",
        fontsize=9,
        color=MUTED,
        va="top",
        ha="right",
    )
    _style(ax, ylabel="% de la demanda con score", xlabel="score Equifax (20 tramos de la curva)")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=2, bbox_to_anchor=(0.62, 1.02))
    ax.set_title(
        "Tienda puntúa más alto que Online, sin ningún punto de concentración",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    return fig


def chart_risk_drift(drift: pd.DataFrame) -> plt.Figure:
    """Índice realizado / esperado a igual composición, por trimestre; un panel por canal."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), sharey=True)
    titles = {"online": "Online (a igual tramo de score)", "tienda": "Tienda (a igual grupo de cliente)"}
    for ax, channel in zip(axes, ("online", "tienda"), strict=True):
        part = drift[drift["canal"] == channel]
        quarters = sorted(part["trimestre"].unique())
        pos = {q: i for i, q in enumerate(quarters)}
        ax.axhline(1.0, color=INK, linewidth=1)
        for horizon, color, name in (("h6", PROPOSED, "a 6 meses (H6)"), ("h3", CURRENT, "a 3 meses (H3)")):
            s = part[part["horizonte"] == horizon].sort_values("trimestre")
            xs = [pos[q] for q in s["trimestre"]]
            ax.plot(xs, s["indice"], color=color, linewidth=2, label=name, zorder=3)
            full = s["meses_maduros"] >= 3
            ax.scatter(np.array(xs)[full.to_numpy()], s.loc[full, "indice"], color=color, s=28, zorder=4)
            # trimestre con menos de tres meses maduros: marcador hueco, es un dato parcial
            ax.scatter(
                np.array(xs)[~full.to_numpy()],
                s.loc[~full, "indice"],
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                s=34,
                zorder=4,
            )
            if len(s):
                ax.annotate(
                    es(s["indice"].iloc[-1], 2),
                    (xs[-1], s["indice"].iloc[-1]),
                    textcoords="offset points",
                    xytext=(7, -3),
                    fontsize=9.5,
                    color=INK,
                )
        ax.set_xticks(range(len(quarters)))
        ax.set_xticklabels([quarter_label(q) for q in quarters])
        ax.set_xlim(-0.4, len(quarters) - 0.3)
        ax.set_title(titles[channel], fontsize=11, loc="left", color=MUTED)
        _style(ax, ylabel="riesgo realizado / esperado (1,00 = ventana)" if channel == "online" else "")
        _comma_ticks(ax)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle(
        "A igual score, el riesgo de Online sube en 2026; en tienda todavía no se ve",
        fontsize=13,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(0.06, -0.03, "Marcador hueco: trimestre con menos de tres meses maduros.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def chart_cesp_trend(trend: pd.DataFrame) -> plt.Figure:
    """Riesgo de tienda por trimestre de originación, PDV con CESP frente al resto."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    names = {"cesp": ("PDV con CESP", RISK), "no_cesp": ("PDV sin CESP", CURRENT)}
    for ax, (column, title) in zip(
        axes, (("riesgo_h6_pct", "Riesgo a 6 meses (H6)"), ("riesgo_h3_pct", "Riesgo a 3 meses (H3)")), strict=True
    ):
        quarters = sorted(trend.loc[trend[column].notna(), "trimestre"].unique())
        pos = {q: i for i, q in enumerate(quarters)}
        series = {
            key: trend[(trend["poblacion"] == key) & trend[column].notna()].sort_values("trimestre") for key in names
        }
        # las dos series acaban casi juntas: la etiqueta de la mayor sube y la de la menor baja
        last = {key: s[column].iloc[-1] for key, s in series.items()}
        upper = max(last, key=last.get)
        for key, (name, color) in names.items():
            s = series[key]
            xs = [pos[q] for q in s["trimestre"]]
            ax.plot(xs, s[column], color=color, linewidth=2, marker="o", markersize=4.5, label=name, zorder=3)
            ax.annotate(
                es(last[key], 2, "%"),
                (xs[-1], last[key]),
                textcoords="offset points",
                xytext=(7, 3 if key == upper else -11),
                fontsize=9.5,
                color=INK,
            )
        ax.set_xticks(range(len(quarters)))
        ax.set_xticklabels([quarter_label(q) for q in quarters], rotation=0, fontsize=8.5)
        ax.set_xlim(-0.4, len(quarters) - 0.2)
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=11, loc="left", color=MUTED)
        _style(ax, ylabel="b2 (%)")
        _comma_ticks(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, loc="upper right", ncol=2, bbox_to_anchor=(0.98, 0.9))
    fig.suptitle(
        "Las tiendas con CESP tenían la mitad de riesgo; desde mediados de 2025 igualan o superan al resto",
        fontsize=13,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def chart_cesp_income(risk: pd.DataFrame) -> plt.Figure:
    """Riesgo H6 de tienda por ingresos declarados: el gradiente se invierte en las tiendas con CESP."""
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    stores = risk[(risk["canal"] == "tienda") & risk["ingresos"].isin(INCOME_LABELS)]
    x = np.arange(len(INCOME_LABELS))
    for offset, key, name, color in ((-0.2, "no_cesp", "PDV sin CESP", CURRENT), (0.2, "cesp", "PDV con CESP", RISK)):
        values = stores[stores["poblacion"] == key].set_index("ingresos")["riesgo_h6_pct"].reindex(INCOME_LABELS)
        ax.bar(x + offset, values, 0.38, color=color, label=name)
        for xi, v in zip(x + offset, values, strict=True):
            ax.text(xi, v + 0.08, es(v), ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([INCOME_TXT[b] for b in INCOME_LABELS], fontsize=8.5)
    _style(ax, ylabel="b2 a 6 meses (%)", xlabel="ingresos mensuales declarados")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title(
        "Con CESP, el ingreso declarado no ordena el riesgo", fontsize=12.5, fontweight="bold", loc="left", pad=12
    )
    return fig


def chart_income_score_risk(risk: pd.DataFrame) -> plt.Figure:
    """Riesgo H6 de Online por rango de score, una línea por tramo de ingresos."""
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    online = risk[(risk["canal"] == "online") & risk["ingresos"].isin(INCOME_LABELS)]
    x = np.arange(len(SCORE_RANGE_LABELS))
    for band, color in zip(INCOME_LABELS, INCOME_RAMP, strict=True):
        s = online[online["ingresos"] == band].set_index("rango_score").reindex(SCORE_RANGE_LABELS)
        # con muy pocos contratos el punto no es un dato: se deja fuera de la línea
        values = s["riesgo_h6_pct"].where(s["contratos"] >= 50)
        ax.plot(x, values, color=color, linewidth=2.2, marker="o", markersize=5, label=INCOME_TXT[band])
        first = SCORE_RANGE_LABELS[0]
        if pd.isna(values[first]):
            continue
        ax.annotate(
            es(values[first], 1, "%"),
            (0, values[first]),
            textcoords="offset points",
            xytext=(-8, 0),
            ha="right",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_RANGE_LABELS)
    ax.set_xlim(-0.75, len(x) - 0.7)
    ax.set_ylim(0, None)
    _style(ax, ylabel="b2 a 6 meses (%)", xlabel="score Equifax")
    ax.legend(frameon=False, fontsize=9, title="ingresos declarados", title_fontsize=9, loc="upper right")
    ax.set_title(
        "Online: el ingreso atenúa, el score sigue ordenando", fontsize=12.5, fontweight="bold", loc="left", pad=12
    )
    return fig


# --------------------------------------------------------------------------- #
# piezas de slide
# --------------------------------------------------------------------------- #
def _text(slide, x, y, w, h, text, size=14, bold=False, color=C_INK, align=PP_ALIGN.LEFT, spacing=1.0):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(text).split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment, para.line_spacing = align, spacing
        run = para.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
        run.font.color.rgb = color
    return box


def _header(slide, title: str, kicker: str = "", period: str = "") -> Emu:
    y = Inches(0.42)
    if kicker:
        _text(slide, MARGIN, y, Inches(7.0), Inches(0.3), kicker.upper(), size=11, bold=True, color=C_ACCENT)
    if period:
        _text(
            slide,
            W - MARGIN - Inches(5.6),
            y,
            Inches(5.6),
            Inches(0.3),
            period,
            11,
            color=C_MUTED,
            align=PP_ALIGN.RIGHT,
        )
    y += Inches(0.34)
    _text(slide, MARGIN, y, W - 2 * MARGIN, Inches(0.6), title, size=24, bold=True)
    line = slide.shapes.add_shape(1, MARGIN, y + Inches(0.72), W - 2 * MARGIN, Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = C_RULE
    line.line.fill.background()
    line.shadow.inherit = False
    return y + Inches(0.95)


def _picture(slide, path: Path, x: Emu, y: Emu, max_w: Emu, max_h: Emu) -> Emu:
    """Inserta el PNG dentro de la caja (max_w × max_h) y devuelve la y de su borde inferior."""
    from PIL import Image

    with Image.open(path) as img:
        ratio = img.height / img.width
    w, h = max_w, Emu(int(max_w * ratio))
    if h > max_h:
        h, w = max_h, Emu(int(max_h / ratio))
    slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    return y + h


def _table(slide, x, y, w, rows: list[list[str]], col_w=None, row_h=0.34, font_pt=11.0, bold_rows=()) -> Emu:
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows)))
    table = shape.table
    if col_w:
        for i, frac in enumerate(col_w):
            table.columns[i].width = Emu(int(w * frac / sum(col_w)))
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
                for run in para.runs:
                    run.font.size, run.font.name = Pt(font_pt), FONT
                    run.font.bold = r == 0 or r in bold_rows
                    run.font.color.rgb = C_INK if (r == 0 or r in bold_rows) else C_MUTED
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_TILE if r == 0 else RGBColor(0xFF, 0xFF, 0xFF)
    return y + Inches(row_h * len(rows))


def _bullets(slide, x, y, w, items: list[tuple[str, str]], size: float = 12.5) -> None:
    box = slide.shapes.add_textbox(x, y, w, max(H - y - Inches(0.4), Inches(0.4)))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (prefix, body) in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(8)
        if prefix:
            run = para.add_run()
            run.text = f"{prefix}  "
            run.font.size, run.font.bold, run.font.name = Pt(size), True, FONT
            run.font.color.rgb = C_INK
        run = para.add_run()
        run.text = body
        run.font.size, run.font.name = Pt(size), FONT
        run.font.color.rgb = C_MUTED


# --------------------------------------------------------------------------- #
# cifras que cita la prosa: todas salen de los CSV
# --------------------------------------------------------------------------- #
def deck_facts(data: dict[str, pd.DataFrame], meta: dict) -> dict:
    b1, drift, sens = data["b1"], data["drift"], data["sens"]
    cesp, swap, inc, risk = data["cesp"], data["swap"], data["income_score"], data["income_risk"]
    cutoff_score = 27.0
    cutoff_bin = int(b1.loc[b1["score_max"] <= cutoff_score, "tramo"].max())

    def idx(channel: str, horizon: str, quarter: str) -> float:
        row = drift[(drift["canal"] == channel) & (drift["horizonte"] == horizon) & (drift["trimestre"] == quarter)]
        return float(row["indice"].iloc[0]) if len(row) else float("nan")

    scored = cesp[cesp["bloque"] == "meses_con_score"].set_index("poblacion")
    window = cesp[cesp["bloque"] == "ventana"].set_index("poblacion")
    swap_in, kept = SWAP_CATEGORIES[1], SWAP_CATEGORIES[0]
    by_cat = swap.set_index(["poblacion", "categoria"])
    accepted = swap[swap["categoria"].isin([kept, swap_in])].groupby("poblacion")["demanda_eur"].sum()
    total_inc = inc[inc["poblacion"] == "total"].set_index("ingresos")
    high = total_inc.loc[["2000-2999", ">=3000"]]
    months = (pd.Period(meta["tienda_hasta"], "M") - pd.Period(meta["tienda_desde"], "M")).n
    online_risk = risk[(risk["canal"] == "online")].set_index(["ingresos", "rango_score"])["riesgo_h6_pct"]
    target = sens["objetivo_pct"].max()
    top = sens[sens["objetivo_pct"] == target].set_index(["factor_online", "factor_tienda"])
    return {
        "cutoff_score": cutoff_score,
        "cutoff_bin": cutoff_bin,
        "below_stores": float(b1.loc[b1["tramo"] <= cutoff_bin, "tienda_demanda_pct"].sum()),
        "below_online": float(b1.loc[b1["tramo"] <= cutoff_bin, "online_demanda_pct"].sum()),
        "top_stores": float(b1.loc[b1["score_min"] >= 74, "tienda_demanda_pct"].sum()),
        "top_online": float(b1.loc[b1["score_min"] >= 74, "online_demanda_pct"].sum()),
        "idx": idx,
        "scored": scored,
        "window": window,
        "swap_in_cesp_pct": float(by_cat.loc[("cesp", swap_in), "pct_demanda"]),
        "swap_in_no_cesp_pct": float(by_cat.loc[("no_cesp", swap_in), "pct_demanda"]),
        "swap_in_share_cesp": float(by_cat.loc[("cesp", swap_in), "pct_categoria"]),
        "kept_share_cesp": float(by_cat.loc[("cesp", kept), "pct_categoria"]),
        "accepted_share_cesp": 100 * float(accepted["cesp"] / accepted["total"]),
        "high_swap_out_n": int(high["swap_out_n"].sum()),
        "high_swap_out_month": high["swap_out_n"].sum() / months,
        "stores_months": months,
        "risk_high_low": float(online_risk.loc[("2000-2999", "1-27")]),
        "risk_high_total": float(online_risk.loc[("2000-2999", "total")]),
        "target": float(target),
        "sens_top": top,
        "drift": float(meta.get("online_factor_riesgo", 1.0) or 1.0),
        "online_volume_pct": float(meta.get("online_volumen_vs_ventana_pct", 0.0) or 0.0),
        "stores_window": span_es(meta["tienda_desde"], meta["tienda_hasta"]),
        "online_window": span_es(meta["online_desde"], meta["online_hasta"]),
    }


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
def slide_title(prs, f: dict, today: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _text(
        slide,
        MARGIN,
        Inches(1.2),
        W - 2 * MARGIN,
        Inches(0.4),
        "ESTUDIO APPLE · SCORE EQUIFAX EN TIENDA",
        13,
        True,
        C_ACCENT,
    )
    _text(
        slide,
        MARGIN,
        Inches(1.65),
        W - 2 * MARGIN,
        Inches(1.0),
        "Cuatro dudas de Riesgos, con datos",
        size=38,
        bold=True,
    )
    sens = f["sens_top"]
    central = sens.loc[(f["drift"], 1.0)] if (f["drift"], 1.0) in sens.index else sens.iloc[0]
    _bullets(
        slide,
        MARGIN,
        Inches(2.95),
        W - 2 * MARGIN - Inches(1.2),
        [
            (
                "1 · Distribución por score.",
                f"Tienda puntúa más alto que Online y no tiene puntos de concentración: el {es(f['below_stores'], 0)}% "
                f"de su demanda queda bajo el corte de clientes nuevos, frente al {es(f['below_online'], 0)}% en Online.",
            ),
            (
                "2 · Evolución del riesgo.",
                f"La curva no lleva tendencia. En Online el riesgo a igual score va un "
                f"{es(100 * (f['drift'] - 1), 0)}% por encima de la ventana y ya está factorizado: el objetivo del "
                f"{es(f['target'], 0)}% pide subir la parrilla {es(central['desplazamiento'], 0)} puntos. "
                "En tienda el dato, hasta julio, todavía no muestra la subida.",
            ),
            (
                "3 · Tiendas con CESP.",
                "El score Equifax no las distingue de las demás, su riesgo ha convergido con el del resto y su ingreso "
                f"declarado no ordena el riesgo. La parrilla dejaría entrar más: son el {es(f['swap_in_share_cesp'], 0)}% "
                "del swap-in.",
            ),
            (
                "4 · Ingresos altos con score bajo.",
                f"Son pocos: unas {es(f['high_swap_out_month'], 0)} solicitudes al mes con 2.000 € o más que hoy se "
                f"aprueban y quedarían bajo el corte. En Online, con ese perfil el riesgo es del {es(f['risk_high_low'])}%.",
            ),
        ],
        size=14,
    )
    _text(
        slide,
        MARGIN,
        H - Inches(0.85),
        W - 2 * MARGIN,
        Inches(0.3),
        f"Riesgos  ·  {today}  ·  Apple Propio  ·  Online {f['online_window']}  ·  tienda {f['stores_window']}",
        size=11,
        color=C_MUTED,
    )


def slide_chart(prs, kicker: str, title: str, period: str, img: Path, notes: list[tuple[str, str]]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    y = _header(slide, title, kicker, period)
    bottom = _picture(slide, img, MARGIN, y, W - 2 * MARGIN, Inches(4.25))
    _bullets(slide, MARGIN, bottom + Inches(0.15), W - 2 * MARGIN, notes, size=12)


def slide_scenarios(prs, picks: pd.DataFrame, sens: pd.DataFrame, f: dict, period: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    y = _header(slide, "Qué le hace la deriva a los escenarios", "2 · Evolución del riesgo", period)
    p3 = picks.set_index("escenario")
    target = f["target"]
    shifted = f"{target:g}% · parrilla desplazada"
    rows = [["Apple total (caso central)", "aprobación", "producción", "riesgo"]]
    for label, key in (
        ("Hoy, con la demanda actual", SAME_BASE_LABEL),
        ("Parrilla de Online en ambos canales", "Parrilla Online en ambos canales"),
        (f"Objetivo {es(target, 0)}%: parrilla {p3.loc[shifted, 'corte']}", shifted),
    ):
        r = p3.loc[key]
        rows.append(
            [
                label,
                es(r["ta_score_pct"], 1, "%"),
                f"{meur(r['produccion_mensual_eur'])}/mes",
                es(r["riesgo_total_pct"], 2, "%"),
            ]
        )
    half = (W - 2 * MARGIN - Inches(0.4)) / 2
    _text(slide, MARGIN, y, half, Inches(0.3), "Caso central", size=13, bold=True)
    _table(slide, MARGIN, y + Inches(0.4), int(half), rows, col_w=[3.4, 1.2, 1.5, 1.0], font_pt=10.5, bold_rows=(3,))
    _bullets(
        slide,
        MARGIN,
        y + Inches(2.1),
        int(half),
        [
            (
                "Riesgo de Online.",
                f"Multiplicado por {es(f['drift'], 2)} en todas las filas simuladas. Tienda no lleva factor.",
            ),
            (
                "Volumen de Online.",
                f"Entra con su demanda actual, un {es(f['online_volume_pct'], 0)}% por encima de la de su ventana, "
                "igual que tienda entra con la de sus meses con score.",
            ),
            (
                "Lectura.",
                f"La parrilla en tienda da más producción a un riesgo parecido, pero el {es(target, 0)}% ya no se "
                "cumple sin mover los cortes.",
            ),
        ],
        size=12,
    )
    x2 = MARGIN + int(half) + Inches(0.4)
    _text(slide, x2, y, half, Inches(0.3), f"Sensibilidad al factor (objetivo {es(target, 0)}%)", size=13, bold=True)
    srows = [["riesgo real frente a la curva", "parrilla vigente", "para cumplirlo", "producción"]]
    central_row = 0
    top = sens[sens["objetivo_pct"] == target]
    for i, r in enumerate(top.itertuples(), start=1):
        ko, kt = r.factor_online, r.factor_tienda
        if ko == 1 and kt == 1:
            name = "como la ventana (sin factor)"
        elif kt == 1:
            name = f"Online +{es(100 * (ko - 1), 0)}%"
        else:
            name = f"Online y tienda +{es(100 * (ko - 1), 0)}%"
        if ko == f["drift"] and kt == 1:
            central_row, name = i, name + " (central)"
        shift = (
            "no alcanzable"
            if pd.isna(r.desplazamiento)
            else ("sin cambios" if r.desplazamiento == 0 else f"+{r.desplazamiento:.0f} puntos")
        )
        srows.append([name, es(r.riesgo_parrilla_vigente_pct, 2, "%"), shift, f"{meur(r.produccion_mensual_eur)}/mes"])
    _table(
        slide, x2, y + Inches(0.4), int(half), srows, col_w=[3.0, 1.4, 1.5, 1.5], font_pt=10.5, bold_rows=(central_row,)
    )
    _text(
        slide,
        x2,
        y + Inches(2.75),
        half,
        Inches(1.4),
        "Los factores son sobre la curva estimada en la ventana. El +30% es el nivel que marcan las cohortes más "
        "recientes a tres meses; queda como estrés. La fila con los dos canales da la referencia si se quiere "
        "aplicar a tienda la misma prudencia.",
        size=11.5,
        color=C_MUTED,
        spacing=1.15,
    )


def slide_cesp_profile(prs, f: dict, img: Path, period: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    y = _header(slide, "Equifax no distingue a las tiendas con CESP, y allí el ingreso no informa", "3 · CESP", period)
    half = (W - 2 * MARGIN - Inches(0.4)) / 2
    _picture(slide, img, MARGIN, y, int(half), Inches(4.6))
    s, w = f["scored"], f["window"]
    rows = [
        ["tienda", "PDV con CESP", "PDV sin CESP"],
        [
            f"% de la demanda, {f['online_window']}",
            es(w.loc["cesp", "pct_demanda"], 0, "%"),
            es(w.loc["no_cesp", "pct_demanda"], 0, "%"),
        ],
        [
            f"% de la demanda, {f['stores_window']}",
            es(s.loc["cesp", "pct_demanda"], 0, "%"),
            es(s.loc["no_cesp", "pct_demanda"], 0, "%"),
        ],
        [
            "aprobación del sistema",
            es(s.loc["cesp", "aprobacion_sistema_pct"], 0, "%"),
            es(s.loc["no_cesp", "aprobacion_sistema_pct"], 0, "%"),
        ],
        [
            "rechazo por score interno",
            es(s.loc["cesp", "rechazo_score_pct"], 0, "%"),
            es(s.loc["no_cesp", "rechazo_score_pct"], 0, "%"),
        ],
        ["score Equifax mediano", es(s.loc["cesp", "score_mediano"], 0), es(s.loc["no_cesp", "score_mediano"], 0)],
        [
            "% bajo el corte Equifax",
            es(s.loc["cesp", "bajo_corte_pct"], 1, "%"),
            es(s.loc["no_cesp", "bajo_corte_pct"], 1, "%"),
        ],
        [
            f"riesgo H6, {f['online_window']}",
            es(w.loc["cesp", "riesgo_h6_pct"], 2, "%"),
            es(w.loc["no_cesp", "riesgo_h6_pct"], 2, "%"),
        ],
    ]
    x2 = MARGIN + int(half) + Inches(0.4)
    bottom = _table(slide, x2, y + Inches(0.1), int(half), rows, col_w=[3.2, 1.4, 1.4], bold_rows=(5, 6))
    _text(
        slide,
        x2,
        bottom + Inches(0.2),
        half,
        Inches(1.6),
        f"Aprobación, rechazo y score: {f['stores_window']}. La marca CESP es del punto de venta: en esas tiendas no "
        "se verifica el ingreso y los requisitos son más livianos. El score Equifax da la misma distribución en "
        "los dos tipos de tienda, y el ingreso declarado solo ordena el riesgo donde se verifica.",
        size=11.5,
        color=C_MUTED,
        spacing=1.15,
    )


def slide_cesp_swap(prs, swap: pd.DataFrame, f: dict, period: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    y = _header(slide, "La parrilla dejaría entrar más solicitudes de tiendas con CESP", "3 · CESP", period)
    names = [
        "se mantiene aprobada",
        "swap-in: hoy rechazada por score interno",
        "swap-out: hoy aprobada, bajo el corte",
    ]
    by = swap.set_index(["poblacion", "categoria"])
    rows = [["qué hace la parrilla Equifax", "% demanda CESP", "% demanda sin CESP", "parte de la fila que es CESP"]]
    for name, cat in zip(names, SWAP_CATEGORIES[:3], strict=True):
        rows.append(
            [
                name,
                es(by.loc[("cesp", cat), "pct_demanda"], 1, "%"),
                es(by.loc[("no_cesp", cat), "pct_demanda"], 1, "%"),
                es(by.loc[("cesp", cat), "pct_categoria"], 0, "%"),
            ]
        )
    rows.append(
        [
            "aceptado por la parrilla (dos primeras)",
            es(sum(by.loc[("cesp", c), "pct_demanda"] for c in SWAP_CATEGORIES[:2]), 1, "%"),
            es(sum(by.loc[("no_cesp", c), "pct_demanda"] for c in SWAP_CATEGORIES[:2]), 1, "%"),
            es(f["accepted_share_cesp"], 0, "%"),
        ]
    )
    bottom = _table(
        slide, MARGIN, y + Inches(0.1), W - 2 * MARGIN, rows, col_w=[3.6, 1.5, 1.6, 2.0], row_h=0.38, bold_rows=(2, 4)
    )
    _bullets(
        slide,
        MARGIN,
        bottom + Inches(0.3),
        W - 2 * MARGIN,
        [
            (
                "Por qué.",
                "El score interno rechaza hoy más a las tiendas con CESP; como Equifax no las distingue, la parrilla las "
                f"recupera. Pasan del {es(f['kept_share_cesp'], 0)}% de lo que se mantiene al "
                f"{es(f['accepted_share_cesp'], 0)}% de lo aceptado.",
            ),
            (
                "Riesgo imputado.",
                "A ese swap-in se le imputa el riesgo de la curva de Online, donde no existe el fraude de situación "
                "laboral: para estas tiendas probablemente se queda corto.",
            ),
            (
                "Opción A.",
                "Aplicar la parrilla Equifax solo a las tiendas sin CESP y mantener en las demás la regla actual "
                "(score interno y verificación con la Seguridad Social).",
            ),
            (
                "Opción B.",
                "Aplicarla a todas con un nivel de riesgo propio para las tiendas con CESP, calibrado con su "
                "morosidad reciente.",
            ),
        ],
        size=12.5,
    )


def slide_income(prs, inc: pd.DataFrame, f: dict, img: Path, period: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    y = _header(
        slide, "Ingresos altos con score bajo: pocos casos, y con más riesgo que su tramo", "4 · Ingresos", period
    )
    half = (W - 2 * MARGIN - Inches(0.4)) / 2
    total = inc[inc["poblacion"] == "total"].set_index("ingresos")
    rows = [["tienda, por ingresos", "demanda", "score mediano", "bajo el corte", "de ellos, aprobados hoy"]]
    for band in INCOME_LABELS:
        r = total.loc[band]
        rows.append(
            [
                INCOME_TXT[band],
                meur(r["demanda_eur"]),
                es(r["score_mediano"], 0),
                es(r["bajo_corte_pct"], 1, "%"),
                f"{es(r['swap_out_pct'], 1, '%')} ({int(r['swap_out_n'])})",
            ]
        )
    _text(slide, MARGIN, y, half, Inches(0.3), f"Tienda, {f['stores_window']} (sin clientes ≥G)", size=13, bold=True)
    bottom = _table(
        slide, MARGIN, y + Inches(0.4), int(half), rows, col_w=[1.9, 1.1, 1.2, 1.2, 1.9], font_pt=10.5, bold_rows=(4, 5)
    )
    _bullets(
        slide,
        MARGIN,
        bottom + Inches(0.45),
        int(half),
        [
            (
                "El caso del vendedor.",
                f"Aprobada hoy y bajo el corte Equifax con 2.000 € o más: {f['high_swap_out_n']} solicitudes en "
                f"{f['stores_months']} meses, unas {es(f['high_swap_out_month'], 0)} al mes.",
            ),
            (
                "Su riesgo.",
                f"En Online, 2.000-2.999 € con score hasta 27: {es(f['risk_high_low'])}%, frente al "
                f"{es(f['risk_high_total'])}% de su tramo de ingresos.",
            ),
            (
                "Excepción posible.",
                "Un corte algo más bajo para ingresos altos, solo en tiendas sin CESP, que es donde el ingreso se verifica.",
            ),
        ],
        size=12,
    )
    _picture(slide, img, MARGIN + int(half) + Inches(0.4), y, int(half), Inches(4.6))


def slide_closing(prs, f: dict) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    y = _header(slide, "Qué proponemos y qué no sabemos todavía", "Propuesta")
    half = (W - 2 * MARGIN - Inches(0.5)) / 2
    _text(slide, MARGIN, y, half, Inches(0.3), "Propuesta", size=14, bold=True, color=C_ACCENT)
    _bullets(
        slide,
        MARGIN,
        y + Inches(0.45),
        int(half),
        [
            (
                "Factor de deriva.",
                f"Escenarios con Online a +{es(100 * (f['drift'] - 1), 0)}% como caso central y +30% como estrés. "
                "A confirmar si ese es el nivel adecuado.",
            ),
            (
                "Tiendas con CESP.",
                "Simular la parrilla solo en tiendas sin CESP y compararla con la versión para todas.",
            ),
            (
                "Ingresos altos.",
                "Simular un corte reducido en tiendas sin CESP: cuánta demanda recupera y a qué riesgo.",
            ),
        ],
        size=13,
    )
    x2 = MARGIN + int(half) + Inches(0.5)
    _text(slide, x2, y, half, Inches(0.3), "Límites", size=14, bold=True, color=C_RISK)
    _bullets(
        slide,
        x2,
        y + Inches(0.45),
        int(half),
        [
            (
                "Riesgo de tienda estimado.",
                "Las operaciones de tienda con score Equifax aún no tienen comportamiento. Que el score ordene igual "
                "que en Online es un supuesto hasta que maduren.",
            ),
            (
                "Subida del riesgo en tienda.",
                "No aparece en el dato disponible (comportamiento a tres meses hasta julio de 2026). Si se ve en "
                "indicadores más recientes, este extracto aún no los trae.",
            ),
            (
                "CESP es una marca de tienda.",
                "La comparación es entre tiendas, no entre clientes: parte de las diferencias puede venir del tipo de tienda.",
            ),
            (
                "Poca muestra en el extremo.",
                "Ingresos altos con score bajo son pocos contratos; por encima de 3.000 € no hay base para medir su riesgo.",
            ),
        ],
        size=13,
    )


# --------------------------------------------------------------------------- #
def build(data_dir: Path, out_path: Path, today: str | None = None) -> Path:
    read = lambda name: pd.read_csv(data_dir / f"{name}.csv")  # noqa: E731
    data = {
        "b1": read("b1_tienda_distribucion_score"),
        "drift": read("b2_deriva_riesgo"),
        "sens": read("b2_sensibilidad_deriva"),
        "cesp": read("b3_tienda_cesp"),
        "trend": read("b3_tienda_cesp_trimestre"),
        "swap": read("b3_tienda_cesp_swap"),
        "income_score": read("b4_ingresos_score"),
        "income_risk": read("b4_ingresos_riesgo"),
    }
    picks = read("p3_escenarios_objetivo")
    meta = json.loads((data_dir / "periodos.json").read_text(encoding="utf-8"))
    f = deck_facts(data, meta)
    img = out_path.parent / "images_dudas"
    charts = {
        "score": save_chart(
            chart_score_distribution(data["b1"], f["cutoff_bin"], f["cutoff_score"]), img / "b1_score.png"
        ),
        "drift": save_chart(chart_risk_drift(data["drift"]), img / "b2_deriva.png"),
        "cesp_trend": save_chart(chart_cesp_trend(data["trend"]), img / "b3_cesp_serie.png"),
        "cesp_income": save_chart(chart_cesp_income(data["income_risk"]), img / "b3_cesp_ingresos.png"),
        "income": save_chart(chart_income_score_risk(data["income_risk"]), img / "b4_ingresos.png"),
    }
    stores_p, both_p = f"tienda {f['stores_window']}", f"Online {f['online_window']}  ·  tienda {f['stores_window']}"
    idx = f["idx"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    d = date.today()
    slide_title(prs, f, today or f"{d.day} {MESES[d.month - 1]} {d.year}")
    slide_chart(
        prs,
        "1 · Distribución por score",
        "Cómo se reparte la demanda de tienda por score Equifax",
        both_p,
        charts["score"],
        [
            (
                "Cargada arriba.",
                f"El {es(f['top_stores'], 0)}% de la demanda de tienda tiene score de 74 o más ({es(f['top_online'], 0)}% en "
                "Online). La producción se concentra algo más arriba todavía, porque el score interno ya aprueba más "
                "cuanto mejor es el score Equifax.",
            ),
            (
                "Sin picos.",
                "La distribución crece de forma suave y no se acumula junto a los cortes; los tres meses con score "
                "son prácticamente iguales.",
            ),
        ],
    )
    slide_chart(
        prs,
        "2 · Evolución del riesgo",
        "La curva no lleva tendencia: cuánto se ha movido el riesgo desde la ventana",
        f"curva estimada sobre {f['online_window']}",
        charts["drift"],
        [
            (
                "Online.",
                f"A seis meses, las cohortes de 2026 van en {es(idx('online', 'h6', '2026Q1'), 2)}; a tres meses el segundo "
                f"trimestre va en {es(idx('online', 'h3', '2026Q2'), 2)}, cuando un año antes estaba en "
                f"{es(idx('online', 'h3', '2025Q2'), 2)}: no es estacionalidad. Su demanda no ha empeorado en score, "
                "así que el score no lo recoge.",
            ),
            (
                "Tienda.",
                f"El primer semestre de 2026 va por debajo de la ventana ({es(idx('tienda', 'h3', '2026Q2'), 2)} a tres "
                "meses). Julio repunta, pero queda por debajo de julio de 2025.",
            ),
        ],
    )
    slide_scenarios(prs, picks, data["sens"], f, both_p)
    slide_chart(
        prs,
        "3 · CESP",
        "Tiendas con CESP frente al resto: cómo ha evolucionado su riesgo",
        "tienda, por trimestre de originación",
        charts["cesp_trend"],
        [
            (
                "Convergencia.",
                "Hasta mediados de 2025 las tiendas con CESP tenían en torno a la mitad del riesgo del resto; desde "
                "entonces igualan o superan. En la ventana completa siguen por debajo "
                f"({es(f['window'].loc['cesp', 'riesgo_h6_pct'], 2)}% frente a "
                f"{es(f['window'].loc['no_cesp', 'riesgo_h6_pct'], 2)}%), así que el nivel medio todavía no lo refleja.",
            ),
        ],
    )
    slide_cesp_profile(prs, f, charts["cesp_income"], stores_p)
    slide_cesp_swap(prs, data["swap"], f, stores_p)
    slide_income(prs, data["income_score"], f, charts["income"], both_p)
    slide_closing(prs, f)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path


__all__ = ["GRID", "build", "deck_facts", "es", "quarter_label", "span_es"]
