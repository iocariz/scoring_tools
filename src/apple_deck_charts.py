"""Gráficos del deck Apple — extensión del score Equifax de e-commerce a tienda.

Lee las salidas de ``run_apple_study.py`` y devuelve figuras matplotlib listas para PNG.

Decisiones de diseño (skill dataviz), por si alguien las revisa:

* **Un solo eje por panel, nunca eje doble.** Riesgo y producción/TA son magnitudes de
  escala distinta, así que van en paneles apilados que comparten el eje x en vez de
  compartir un gráfico con dos escalas —la alineación entre dos escalas es arbitraria
  e inventa correlaciones que no están en el dato.
* **Paleta validada**, no elegida a ojo: se portaron los seis checks del skill
  (banda de luminosidad OKLCH, suelo de croma, separación CVD Machado-2009,
  suelo de visión normal, contraste WCAG). El gris de la paleta del proyecto
  (``#BDC3C7``) **falla** como marca de datos sobre blanco —L=0,814 fuera de banda
  y 1,78:1 de contraste—, así que la serie "hoy" usa ``#7F8C8D`` (L=0,630, 3,48:1,
  dE CVD frente al azul 11,7). Rojo y verde de la paleta pasan el gate CVD, pero
  no se usan juntos porque riesgo y producción nunca comparten panel.
* **Etiquetas directas selectivas**, nunca un número en cada punto; leyenda solo
  cuando hay dos o más series; rejilla fina y recesiva; el texto lleva tinta de
  texto, nunca el color de la serie.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, determinista
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- paleta (validada; ver docstring) --------------------------------------- #
INK = "#2C3E50"  # texto primario
MUTED = "#5D6D7E"  # texto secundario (5,31:1 sobre blanco)
GRID = "#E5E7EB"  # rejilla, un paso sobre la superficie
CURRENT = "#7F8C8D"  # serie "hoy" / actual
PROPOSED = "#3498DB"  # serie "propuesto" / accent
RISK = "#E74C3C"  # magnitud de riesgo
SURFACE = "#FFFFFF"

BAR_MAX_PX = 24


def es(value: float, decimals: int = 2, suffix: str = "") -> str:
    """Formato español: coma decimal. El deck se presenta en español."""
    return f"{value:,.{decimals}f}".replace(",", "\u00a0").replace(".", ",") + suffix


plt.rcParams.update(
    {
        # Arial para igualar la tipografía de las slides; DejaVu es el respaldo que
        # matplotlib siempre trae, para que el deck se regenere en Docker/CI sin Arial.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "axes.grid": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def save_chart(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _style(ax, *, ylabel: str = "", xlabel: str = "", hgrid: bool = True) -> None:
    """Ejes recesivos: sin marco, rejilla horizontal fina y por debajo de los datos."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=9)
    if hgrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=6)


def _bar_width(n_slots: int, fig_width_in: float, dpi: int = 100) -> float:
    """Ancho de barra en unidades de dato, capado a 24 px para que la banda respire."""
    px_per_slot = fig_width_in * dpi / max(n_slots, 1)
    return min(0.82, BAR_MAX_PX / px_per_slot)


# --------------------------------------------------------------------------- #
# 1. La curva de riesgo: el motor de todo el estudio
# --------------------------------------------------------------------------- #
def chart_risk_curve(curve: pd.DataFrame, edges: list[float]) -> plt.Figure:
    """Riesgo realizado por tramo EFX y producción que sostiene cada estimación.

    Dos paneles con eje x compartido en vez de un eje doble: el de arriba responde
    "cuánto discrimina el score", el de abajo "cuánto me puedo fiar de cada punto".
    La zona de tramos con producción residual se sombrea en ambos paneles —es donde
    la estimación es más débil y justo la que abren los escenarios más laxos.
    """
    df = curve.dropna(subset=["b2_pct"]).sort_values("bin")
    bins = df["bin"].to_numpy()
    thin = df[df["booked_eur"] < 0.005 * df["booked_eur"].sum()]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.1}
    )

    if not thin.empty:
        lo, hi = thin["bin"].min() - 0.5, thin["bin"].max() + 0.5
        for ax in (ax1, ax2):
            ax.axvspan(lo, hi, color=GRID, alpha=0.55, zorder=0)

    ax1.plot(bins, df["b2_pct"], color=RISK, linewidth=2, solid_capstyle="round", zorder=3)
    ax1.scatter(bins, df["b2_pct"], s=34, color=RISK, edgecolor=SURFACE, linewidth=2, zorder=4)
    _style(ax1, ylabel="Riesgo b2 realizado (%)")
    ax1.set_ylim(0, df["b2_pct"].max() * 1.18)
    # etiquetas donde se decide el corte, no en todos los puntos
    for b in (bins[0], 8.0, 12.0, 16.0, bins[-1]):
        row = df[df["bin"] == b]
        if row.empty:
            continue
        ax1.annotate(
            es(float(row["b2_pct"].iloc[0]), 1, "%"),
            (b, float(row["b2_pct"].iloc[0])),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
    if not thin.empty:
        ax1.annotate(
            f"tramos {int(thin['bin'].min())}-{int(thin['bin'].max())}\nproducción residual:\nriesgo poco fiable",
            (thin["bin"].max() + 0.25, df["b2_pct"].max() * 0.60),
            ha="left",
            fontsize=8.5,
            color=MUTED,
        )
    ax1.set_title(
        "El score EFX ordena el riesgo de principio a fin en e-commerce",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )

    ax2.bar(bins, df["booked_eur"] / 1e6, 0.6, color=CURRENT, zorder=3)
    _style(ax2, ylabel="Producción (M€)", xlabel="Tramo EFX  ·  umbral de score")
    ax2.set_xticks(bins[::2])
    labels = []
    for b in bins[::2]:
        edge = edges[int(b) - 1]
        labels.append(f"{int(b)}\n{'≤' + es(edges[1], 0) if not np.isfinite(edge) else '>' + es(edge, 0)}")
    ax2.set_xticklabels(labels, fontsize=8.5)
    return fig


# --------------------------------------------------------------------------- #
# 2. Punto 1 — Apple Online medido
# --------------------------------------------------------------------------- #
def chart_channel_actual(actuals: pd.DataFrame, title: str) -> plt.Figure:
    """Riesgo real por grupo, con la TA como etiqueta directa (una sola magnitud por eje).

    Sirve para los dos canales: la situación actual de ambos está medida, no imputada.
    """
    df = actuals[actuals["segmento"] != "TOTAL"].dropna(subset=["riesgo_real_pct"]).copy()
    df = df.sort_values("riesgo_real_pct")
    total = actuals[actuals["segmento"] == "TOTAL"].iloc[0]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.barh(y, df["riesgo_real_pct"], height=0.42, color=RISK, zorder=3)
    xmax = df["riesgo_real_pct"].max() * 1.42
    for i, row in enumerate(df.itertuples()):
        # ambos valores fuera de la barra: una barra corta no puede alojar texto sin recortarlo
        ax.annotate(
            f"  {es(row.riesgo_real_pct, 2, '%')}",
            (row.riesgo_real_pct, i),
            va="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
        ax.annotate(f"TA {es(row.ta_pct, 1, '%')}", (xmax, i), ha="right", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(df["segmento"], fontsize=10.5, color=INK)
    _style(ax, xlabel="Riesgo b2 realizado (%)", hgrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)
    ax.axvline(total["riesgo_real_pct"], color=INK, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"media del canal {es(total['riesgo_real_pct'], 2, '%')}",
        (total["riesgo_real_pct"], len(df) - 0.5),
        xytext=(7, 4),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
        annotation_clip=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=18)
    return fig


# --------------------------------------------------------------------------- #
# 3. Punto 2 — tienda: hoy vs parrilla de Online
# --------------------------------------------------------------------------- #
def chart_stores_swap(grid: pd.DataFrame, compact: bool = False) -> plt.Figure:
    """Dos paneles, una magnitud cada uno: la TA sube y el riesgo baja a la vez.

    El color marca **identidad** —qué política— y no estado: las dos barras "hoy" van
    en gris y las dos "parrilla EFX" en azul. Pintar de rojo la barra de riesgo nueva
    la haría leer como "malo" justo cuando el mensaje es que el riesgo baja.

    ``compact`` devuelve una versión para media slide: lienzo más pequeño con los
    mismos cuerpos de letra, para que al reducirla el texto siga siendo legible en
    proyección. Escalar la versión grande achicaría las fuentes a la mitad.
    """
    total = grid[grid["segmento"] == "TOTAL"].iloc[0]
    ta0, ta1 = total["ta_actual_pct"], total["ta_efectiva_pct"]
    r0, r1 = total["riesgo_imputado_actual_pct"], total["riesgo_imputado_pct"]
    # Los veredictos se calculan: un texto fijo mentiría en cuanto el dato cambiara de signo.
    panels = [
        ("Tasa de aceptación (% demanda €)", ta0, ta1, "más aceptación" if ta1 >= ta0 else "menos aceptación"),
        ("Riesgo imputado (%)", r0, r1, "menos riesgo" if r1 <= r0 else "más riesgo"),
    ]
    figsize = (6.8, 3.2) if compact else (10.5, 4.4)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (title, before, after, verdict) in zip(axes, panels, strict=True):
        ax.bar([0, 1], [before, after], 0.34, color=[CURRENT, PROPOSED], zorder=3)
        for x, v in ((0, before), (1, after)):
            ax.annotate(
                es(v, 2, "%"),
                (x, v),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=12 if compact else 13,
                fontweight="bold",
                color=INK,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Score interno\n(hoy)", "Parrilla EFX\nde e-commerce"], fontsize=9 if compact else 9.5, color=INK
        )
        _style(ax, ylabel=title)
        ax.set_ylim(0, max(before, after) * 1.32)
        ax.annotate(
            f"{'+' if after >= before else ''}{es(after - before, 2)} pp  ·  {verdict}",
            (0.5, max(before, after) * 1.20),
            ha="center",
            fontsize=9.5 if compact else 10.5,
            fontweight="bold",
            color=MUTED,
        )
    if not compact:
        fig.suptitle(
            "En tienda, cambiar al score EFX gana aceptación y baja riesgo a la vez",
            fontsize=13,
            fontweight="bold",
            x=0.085,
            ha="left",
            y=1.02,
            color=INK,
        )
    fig.tight_layout()
    return fig


def chart_stores_by_segment(grid: pd.DataFrame) -> plt.Figure:
    """Desglose por segmento: de dónde sale la mejora de tienda.

    Dumbbell y no barras agrupadas: lo que se quiere leer es el **cambio** de cada
    segmento, y un par de puntos unidos lo muestra directamente —la longitud del
    conector es la magnitud y su dirección el signo—. Dos paneles, una magnitud cada
    uno, compartiendo el eje de segmentos.
    """
    df = grid[(grid["segmento"] != "TOTAL") & grid["riesgo_imputado_pct"].notna()].copy()
    df = df.sort_values("demanda_eur")
    y = np.arange(len(df))
    panels = [
        ("Tasa de aceptación (% demanda €)", "ta_actual_pct", "ta_efectiva_pct"),
        ("Riesgo imputado (%)", "riesgo_imputado_actual_pct", "riesgo_imputado_pct"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, col_before, col_after) in zip(axes, panels, strict=True):
        before, after = df[col_before].to_numpy(), df[col_after].to_numpy()
        ax.hlines(y, before, after, color=GRID, linewidth=3.5, zorder=2)
        ax.scatter(before, y, s=95, color=CURRENT, edgecolor=SURFACE, linewidth=2, zorder=3, label="Hoy")
        ax.scatter(after, y, s=95, color=PROPOSED, edgecolor=SURFACE, linewidth=2, zorder=4, label="Parrilla EFX")
        span = max(after.max(), before.max())
        for i, value in enumerate(after):
            # solo el valor de llegada: el de partida lo da el punto gris y la leyenda
            ax.annotate(
                es(value, 1, "%"),
                (value, i),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color=INK,
            )
        _style(ax, xlabel=title, hgrid=False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, span * 1.18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["segmento"], fontsize=10.5, color=INK)
    axes[0].set_ylim(-0.6, len(df) - 0.4)
    legend = axes[0].legend(loc="lower right", frameon=False, fontsize=10, handletextpad=0.4)
    for text in legend.get_texts():
        text.set_color(MUTED)
    # El titular se calcula: un grupo puede perder décimas de TA (A-C lo hace) y entonces
    # "todos ganan" sería falso. Y "donde más pesa" se refiere al grupo con más demanda.
    gain = int((df["ta_efectiva_pct"] > df["ta_actual_pct"]).sum())
    biggest = df.iloc[-1]  # df está ordenado por demanda ascendente
    head = "Todos los grupos ganan aceptación" if gain == len(df) else f"{gain} de {len(df)} grupos ganan aceptación"
    tail = (
        "el riesgo baja donde más pesa"
        if biggest["riesgo_imputado_pct"] < biggest["riesgo_imputado_actual_pct"]
        else "el riesgo sube donde más pesa"
    )
    fig.suptitle(
        f"{head}; {tail}",
        fontsize=13,
        fontweight="bold",
        x=0.075,
        ha="left",
        y=1.0,
        color=INK,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Punto 3 — la frontera riesgo / aceptación
# --------------------------------------------------------------------------- #
def chart_scenario_frontier(ladder: pd.DataFrame, picks: pd.DataFrame) -> plt.Figure:
    """Frontera riesgo-aceptación, con la política actual marcada.

    La TA del eje es la **efectiva** (producción sobre demanda), que es la misma base
    que la del punto 1. La otra tasa que maneja el estudio —demanda que supera el
    corte— es mucho más alta y no es comparable con "hoy"; va en la tabla, no aquí.

    El punto de la parrilla actual es lo que convierte la curva en una decisión: sin él
    no se ve que los cuatro objetivos son endurecimientos.
    """
    df = ladder.sort_values("riesgo_total_pct")
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(
        df["riesgo_total_pct"], df["ta_efectiva_pct"], color=PROPOSED, linewidth=2, solid_capstyle="round", zorder=3
    )
    ax.scatter(
        df["riesgo_total_pct"], df["ta_efectiva_pct"], s=24, color=PROPOSED, edgecolor=SURFACE, linewidth=1.6, zorder=4
    )

    marks = picks[picks["escenario"].str.contains("corte único", na=False)].dropna(subset=["riesgo_total_pct"])
    ax.scatter(
        marks["riesgo_total_pct"],
        marks["ta_efectiva_pct"],
        s=170,
        facecolor=SURFACE,
        edgecolor=INK,
        linewidth=2.2,
        zorder=5,
    )
    for row in marks.itertuples():
        ax.annotate(
            f"objetivo {row.escenario.split(' ·')[0]}",
            (row.riesgo_total_pct, row.ta_efectiva_pct),
            textcoords="offset points",
            xytext=(-14, 13),
            ha="right",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
    medido = picks[picks["escenario"].str.startswith("Hoy")]
    if not medido.empty:
        m = medido.iloc[0]
        ax.scatter(
            [m["riesgo_total_pct"]],
            [m["ta_efectiva_pct"]],
            s=200,
            color=CURRENT,
            edgecolor=SURFACE,
            linewidth=2.4,
            zorder=6,
        )
        ax.annotate(
            f"hoy, medido: {es(m['ta_efectiva_pct'], 1, '%')} de TA",
            (m["riesgo_total_pct"], m["ta_efectiva_pct"]),
            textcoords="offset points",
            xytext=(14, -16),
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color=MUTED,
        )
    base = picks[picks["escenario"].str.startswith("Parrilla Online")]
    if not base.empty:
        b = base.iloc[0]
        ax.scatter(
            [b["riesgo_total_pct"]],
            [b["ta_efectiva_pct"]],
            s=200,
            color=RISK,
            edgecolor=SURFACE,
            linewidth=2.4,
            zorder=6,
        )
        ax.annotate(
            f"parrilla Online en ambos: {es(b['ta_efectiva_pct'], 1, '%')} de TA",
            (b["riesgo_total_pct"], b["ta_efectiva_pct"]),
            textcoords="offset points",
            xytext=(16, -5),
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=RISK,
        )
    _style(ax, ylabel="Tasa de aceptación efectiva (producción / demanda, %)", xlabel="Riesgo global Apple (%)")
    # El eje se recorta a la zona de decisión: aflojar más allá de la política actual
    # no está sobre la mesa, y con la cola entera los cuatro objetivos se apelotonan.
    if not base.empty:
        ax.set_xlim(0, float(base.iloc[0]["riesgo_total_pct"]) * 1.45)
        visible = df[df["riesgo_total_pct"] <= float(base.iloc[0]["riesgo_total_pct"]) * 1.45]
        ax.set_ylim(0, max(visible["ta_efectiva_pct"].max(), 1) * 1.3)
    else:
        ax.set_ylim(0, max(df["ta_efectiva_pct"].max(), 1) * 1.25)
    ax.set_title(
        "El objetivo del 4% es prácticamente el punto de partida",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    return fig


# --------------------------------------------------------------------------- #
# 5. Estacionalidad — por qué agosto no es un mes
# --------------------------------------------------------------------------- #
_MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def chart_seasonality(index: pd.DataFrame) -> plt.Figure:
    """Los dos perfiles estacionales, que no se parecen.

    Dos series en el mismo eje —es la misma magnitud, un índice mes-del-año— así que
    aquí sí comparten gráfico: lo que hay que leer es precisamente la divergencia.
    Leyenda presente por ser dos series, y etiqueta directa solo en septiembre, que es
    donde se separan.
    """
    df = index.sort_values("mes")
    months = df["mes"].to_numpy()
    has_stores = "indice_tienda" in df.columns
    fig, ax = plt.subplots(figsize=(11, 4.6))
    series = [("Online", df["indice"].to_numpy(), CURRENT)]
    if has_stores:
        series.append(("Tienda", df["indice_tienda"].to_numpy(), PROPOSED))
    for label, values, color in series:
        ax.plot(months, values, color=color, linewidth=2, solid_capstyle="round", zorder=3, label=label)
        ax.scatter(months, values, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.axhline(1.0, color=INK, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "mes medio = 1,00", (5.0, 1.0), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=MUTED
    )
    for _label, values, _color in series:
        ax.annotate(
            es(values[8], 2),
            (9, values[8]),
            xytext=(10, -4),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    top = max(v.max() for _, v, _ in series)
    ax.set_xticks(months)
    ax.set_xticklabels([_MONTHS[m - 1] for m in months], fontsize=9.5)
    _style(ax, ylabel="Índice estacional de demanda")
    ax.set_ylim(0, top * 1.22)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10.5, handletextpad=0.5)
    for text in legend.get_texts():
        text.set_color(MUTED)
    if has_stores:
        ax.annotate(
            "septiembre: el lanzamiento de iPhone es un fenómeno de Online;\nen tienda es de los meses más flojos",
            (1.5, top * 0.10),
            fontsize=9.5,
            color=MUTED,
        )
    ax.set_title("Los dos canales no comparten estacionalidad", fontsize=13, fontweight="bold", loc="left", pad=14)
    return fig
