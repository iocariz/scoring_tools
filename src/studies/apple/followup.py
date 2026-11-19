"""Estudio Apple — análisis de seguimiento (dudas de la dirección de riesgos, 18-11-2026).

Cuatro preguntas sobre el estudio principal (``study.py``), cada una con su tabla:

1. Cómo se reparten la demanda y la producción de tienda por score EFX
   (``score_distribution``).
2. Qué hace la simulación con la deriva reciente del riesgo y del perfil
   (``risk_drift``, ``demand_profile``, ``drift_sensitivity``). La curva se estima
   sobre una ventana madura y NO lleva tendencia: estas tablas miden cuánto se ha
   movido el riesgo *a igual score* desde entonces y qué le haría a los escenarios.
3. Clientes CESP de tienda por separado (``cesp_summary``, ``cesp_trend``,
   ``cesp_score_distribution``, ``cesp_swap``).
4. Score por tramo de ingresos, y riesgo de los ingresos altos por rango de score
   (``income_score_table``, ``income_risk``).

Todo es descriptivo: ninguna función cambia una cifra del estudio principal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.constants import Columns, RejectReason, SystemDecision
from src.studies.apple.study import (
    SCORE_COL,
    SEGMENT_COL,
    SWAP_CATEGORIES,
    _booked,
    _unscored,
    stores_swap_breakdown,
)

# 's' = la CESP (condición especial) se aplicó a la operación: sin verificación de ingresos. El comercio puede
# tenerla y el cliente optar por aportar nómina (entonces 'n'). Solo se analiza en tienda.
CESP_COL = "cespa"

INCOME_COL = "income_t1_m"  # ingresos mensuales declarados del primer titular

INCOME_EDGES: list[float] = [-np.inf, 1000, 1500, 2000, 3000, np.inf]

INCOME_LABELS: list[str] = ["<1000", "1000-1499", "1500-1999", "2000-2999", ">=3000"]

SCORE_RANGE_EDGES: list[float] = [0, 27, 50, 75, 99]

SCORE_RANGE_LABELS: list[str] = ["1-27", "28-50", "51-75", "76-99"]

# (horizonte, numerador, denominador, multiplicador): H3 madura tres meses antes que H6.
HORIZONS: tuple[tuple[str, str, str, float], ...] = (
    ("h6", "todu_30ever_h6", "todu_amt_pile_h6", 7.0),
    ("h3", "todu_30ever_h3", "todu_amt_pile_h3", 4.0),
)

# Un mes cuenta como maduro en un horizonte si casi todos sus contratos tienen exposición.
MATURE_COVERAGE = 0.9

# (factor Online, factor tienda) sobre el riesgo de los escenarios; ver drift_sensitivity.
DRIFT_FACTORS: tuple[tuple[float, float], ...] = ((1.0, 1.0), (1.15, 1.0), (1.30, 1.0), (1.15, 1.15), (1.30, 1.30))

TOTAL = "total"


def _eur(frame: pd.DataFrame) -> float:
    return float(frame[Columns.OA_AMT].sum())


def _b2(frame: pd.DataFrame, num: str = "todu_30ever_h6", den: str = "todu_amt_pile_h6", mult: float = 7.0) -> float:
    exposure = frame[den].sum()
    return float(100 * mult * frame[num].sum() / exposure) if exposure > 0 else float("nan")


def _approved(frame: pd.DataFrame) -> pd.Series:
    return frame[Columns.SE_DECISION_ID].isin([SystemDecision.OK.value, SystemDecision.RV.value])


def _score_ko(frame: pd.DataFrame) -> pd.Series:
    return (frame[Columns.REJECT_REASON] == RejectReason.SCORE) & ~_unscored(frame)


def _below_cutoff(frame: pd.DataFrame, cutoffs: dict[str, float | None]) -> pd.Series:
    """Con score y por debajo (o en) el umbral de rechazo de su grupo; ``>=G`` siempre."""
    threshold = frame[SEGMENT_COL].map(cutoffs).astype(float).fillna(-np.inf)
    return frame[SCORE_COL].notna() & (frame[SCORE_COL] <= threshold)


def _cesp_splits(stores: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Total, CESP y no CESP; solo el total si el extracto no trae la marca."""
    if CESP_COL not in stores.columns:
        return [(TOTAL, stores)]
    flag = stores[CESP_COL].astype(str).str.lower()
    return [(TOTAL, stores), ("cesp", stores[flag == "s"]), ("no_cesp", stores[flag != "s"])]


def income_band(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame[INCOME_COL], bins=INCOME_EDGES, labels=INCOME_LABELS, right=False).astype(object)


def score_range(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame[SCORE_COL], bins=SCORE_RANGE_EDGES, labels=SCORE_RANGE_LABELS).astype(object)


# ---------------------------------------------------------------- 1) distribución por score


def score_distribution(stores: pd.DataFrame, ecom: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Demanda y producción por tramo EFX: tienda (meses con score) frente a Online (ventana).

    Los porcentajes son sobre la demanda/producción CON score de cada canal. La
    contratación de tienda por tramo es la de HOY (decide el score interno), no la
    que daría la parrilla.
    """
    rows = {}
    for label, channel in (("tienda", stores), ("online", ecom)):
        scored = channel[channel["bin"].notna()]
        demand = scored.groupby("bin")[Columns.OA_AMT].sum()
        booked = _booked(scored).groupby("bin")["oa_amt_h0"].sum()
        rows[f"{label}_demanda_eur"] = demand
        rows[f"{label}_demanda_pct"] = 100 * demand / demand.sum()
        rows[f"{label}_produccion_eur"] = booked
        rows[f"{label}_produccion_pct"] = 100 * booked / booked.sum()
        rows[f"{label}_contratacion_pct"] = 100 * booked / demand
    frame = pd.DataFrame(rows).reindex(curve.index).fillna(0.0)
    scored_stores = stores[stores["bin"].notna()]
    frame.insert(0, "score_max", scored_stores.groupby("bin")[SCORE_COL].max())
    frame.insert(0, "score_min", scored_stores.groupby("bin")[SCORE_COL].min())
    frame["riesgo_curva_pct"] = curve["b2_pct"]
    return frame.rename_axis("tramo").reset_index()


# ---------------------------------------------------------------- 2) deriva del riesgo y del perfil


def risk_drift(
    channel: pd.DataFrame, by: list[str], window_from: str, window_to: str, label: str, since: str = "2025-01-01"
) -> pd.DataFrame:
    """Riesgo realizado por trimestre frente al esperado *a igual composición*.

    El esperado aplica a los contratos de cada trimestre la tasa que su celda ``by``
    (tramo EFX en Online; grupo de letra en tienda, que no tiene score histórico) tuvo
    en la ventana del estudio. ``indice`` = realizado / esperado: 1,00 es "se comporta
    como la ventana"; por encima, el riesgo ha subido sin que la composición lo explique,
    que es justo lo que una curva estimada en la ventana no ve. Lleva estacionalidad
    (la ventana es un año entero): comparar cada trimestre con el mismo del año anterior.
    """
    booked = _booked(channel).dropna(subset=by)
    month = booked[Columns.MIS_DATE].dt.to_period("M")
    window = booked[(booked[Columns.MIS_DATE] >= window_from) & (booked[Columns.MIS_DATE] < window_to)]
    rows = []
    for horizon, num, den, mult in HORIZONS:
        if num not in booked.columns or den not in booked.columns:
            continue
        mature = (booked[den] > 0).groupby(month).mean() >= MATURE_COVERAGE
        recent = booked[month.map(mature).astype(bool) & (booked[Columns.MIS_DATE] >= since)]
        cells = window.groupby(by, observed=True)[[num, den]].sum()
        rate = (cells[num] / cells[den]).rename("_rate")
        scored = recent.join(rate, on=by)
        scored["_expected"] = scored["_rate"] * scored[den]
        quarter = scored[Columns.MIS_DATE].dt.to_period("Q")
        for q, g in scored.groupby(quarter):
            realized = 100 * mult * g[num].sum() / g[den].sum()
            expected = 100 * mult * g["_expected"].sum() / g[den].sum()
            rows.append(
                {
                    "canal": label,
                    "horizonte": horizon,
                    "trimestre": str(q),
                    "meses_maduros": int(g[Columns.MIS_DATE].dt.to_period("M").nunique()),
                    "contratos": int(len(g)),
                    "riesgo_real_pct": realized,
                    "riesgo_esperado_pct": expected,
                    "indice": realized / expected if expected > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def demand_profile(
    channel: pd.DataFrame, cutoffs: dict[str, float | None], label: str, since: str = "2025-01-01"
) -> pd.DataFrame:
    """Perfil de la demanda por trimestre: lo que el score SÍ ve cuando la población cambia."""
    recent = channel[channel[Columns.MIS_DATE] >= since]
    rows = []
    for q, g in recent.groupby(recent[Columns.MIS_DATE].dt.to_period("Q")):
        scored = g[g[SCORE_COL].notna()]
        gradable = scored[scored[SEGMENT_COL] != ">=G"]
        rows.append(
            {
                "canal": label,
                "trimestre": str(q),
                "demanda_eur": _eur(g),
                "score_mediano": float(scored[SCORE_COL].median()) if len(scored) else float("nan"),
                "bajo_corte_pct": (
                    100 * _eur(gradable[_below_cutoff(gradable, cutoffs)]) / _eur(gradable)
                    if len(gradable)
                    else float("nan")
                ),
                "aprobacion_sistema_pct": 100 * _eur(g[_approved(g)]) / _eur(g),
                "clientes_nuevos_pct": 100 * _eur(g[g[SEGMENT_COL] == "New"]) / _eur(g),
                "ingreso_mediano": float(g[INCOME_COL].median()) if INCOME_COL in g.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def drift_sensitivity(
    ladder_grid: pd.DataFrame,
    targets: list[float],
    factors: tuple[tuple[float, float], ...] = DRIFT_FACTORS,
    online_applied: float = 1.0,
) -> pd.DataFrame:
    """Qué le pasa a cada objetivo si el riesgo real va un factor por encima de la curva.

    El factor multiplica el riesgo de un canal entero (su exposición no cambia), así que
    el total se recompone con el peso de exposición de cada canal, despejado de la propia
    escalera (total = w·online + (1−w)·tienda). Para cada objetivo se devuelve el menor
    desplazamiento de la parrilla que lo cumple ya con el riesgo estresado.

    Los factores son ABSOLUTOS frente a la curva: si la escalera ya lleva una deriva de
    Online (``online_applied``, el caso central del estudio), se descuenta para no
    aplicarla dos veces.
    """
    spread = ladder_grid["online_riesgo_pct"] - ladder_grid["tienda_riesgo_pct"]
    weight = ((ladder_grid["riesgo_total_pct"] - ladder_grid["tienda_riesgo_pct"]) / spread).where(spread.abs() > 1e-9)
    weight = weight.fillna(0.5).clip(0, 1)
    current = ladder_grid.loc[ladder_grid["desplazamiento"] == 0].index
    rows = []
    for k_online, k_stores in factors:
        stressed = weight * (k_online / online_applied) * ladder_grid["online_riesgo_pct"] + (
            1 - weight
        ) * k_stores * ladder_grid["tienda_riesgo_pct"].fillna(0)
        for target in targets:
            ok = ladder_grid[stressed <= target]
            pick = ok.loc[ok["produccion_mensual_eur"].idxmax()] if not ok.empty else None
            rows.append(
                {
                    "factor_online": k_online,
                    "factor_tienda": k_stores,
                    "riesgo_parrilla_vigente_pct": float(stressed.loc[current].iloc[0]) if len(current) else np.nan,
                    "objetivo_pct": target,
                    "desplazamiento": float(pick["desplazamiento"]) if pick is not None else np.nan,
                    "riesgo_pct": float(stressed.loc[pick.name]) if pick is not None else np.nan,
                    "ta_score_pct": float(pick["ta_score_pct"]) if pick is not None else np.nan,
                    "produccion_mensual_eur": float(pick["produccion_mensual_eur"]) if pick is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3) CESP


def cesp_summary(blocks: dict[str, pd.DataFrame], cutoffs: dict[str, float | None]) -> pd.DataFrame:
    """CESP frente a no CESP en tienda, por bloque (ventana madura / meses con score)."""
    rows = []
    for block, frame in blocks.items():
        for name, g in _cesp_splits(frame):
            if g.empty:
                continue
            booked = _booked(g)
            scored = g[g[SCORE_COL].notna() & (g[SEGMENT_COL] != ">=G")]
            rows.append(
                {
                    "bloque": block,
                    "poblacion": name,
                    "solicitudes": int(len(g)),
                    "demanda_eur": _eur(g),
                    "pct_demanda": 100 * _eur(g) / _eur(frame),
                    "aprobacion_sistema_pct": 100 * _eur(g[_approved(g)]) / _eur(g),
                    "rechazo_score_pct": 100 * _eur(g[_score_ko(g)]) / _eur(g),
                    "tasa_contratacion_pct": 100 * booked["oa_amt_h0"].sum() / _eur(g),
                    "produccion_eur": float(booked["oa_amt_h0"].sum()),
                    "riesgo_h6_pct": _b2(booked),
                    "riesgo_h3_pct": _b2(booked, "todu_30ever_h3", "todu_amt_pile_h3", 4.0),
                    "score_mediano": float(g[SCORE_COL].median()) if len(scored) else float("nan"),
                    "bajo_corte_pct": (
                        100 * _eur(scored[_below_cutoff(scored, cutoffs)]) / _eur(scored)
                        if len(scored)
                        else float("nan")
                    ),
                    "ingreso_mediano": float(g[INCOME_COL].median()) if INCOME_COL in g.columns else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def cesp_trend(all_stores: pd.DataFrame, since: str = "2024-01-01") -> pd.DataFrame:
    """Peso y riesgo (H6 y H3) de CESP y no CESP por trimestre; NaN donde aún no ha madurado."""
    recent = all_stores[all_stores[Columns.MIS_DATE] >= since]
    rows = []
    for q, g in recent.groupby(recent[Columns.MIS_DATE].dt.to_period("Q")):
        for name, part in _cesp_splits(g):
            booked = _booked(part)
            rows.append(
                {
                    "trimestre": str(q),
                    "poblacion": name,
                    "pct_demanda": 100 * _eur(part) / _eur(g),
                    "aprobacion_sistema_pct": 100 * _eur(part[_approved(part)]) / _eur(part) if len(part) else np.nan,
                    "produccion_eur": float(booked["oa_amt_h0"].sum()),
                    "riesgo_h6_pct": _b2(booked),
                    "riesgo_h3_pct": _b2(booked, "todu_30ever_h3", "todu_amt_pile_h3", 4.0),
                }
            )
    return pd.DataFrame(rows)


def cesp_score_distribution(stores: pd.DataFrame) -> pd.DataFrame:
    """% de la demanda con score de cada población en cada tramo EFX."""
    scored = stores[stores["bin"].notna()]
    columns = {}
    for name, g in _cesp_splits(scored):
        demand = g.groupby("bin")[Columns.OA_AMT].sum()
        columns[f"{name}_demanda_pct"] = 100 * demand / demand.sum()
    return pd.DataFrame(columns).fillna(0.0).rename_axis("tramo").reset_index()


def cesp_swap(
    stores: pd.DataFrame, curve: pd.DataFrame, cutoffs: dict[str, float | None], level_factor: float
) -> pd.DataFrame:
    """El desglose del swap de la parrilla (``stores_swap_breakdown``) para CESP y no CESP.

    ``pct_demanda`` es sobre la demanda de cada población; ``pct_categoria`` dice qué
    parte de cada categoría (p.ej. del swap-in) es de esa población.
    """
    total = stores_swap_breakdown(stores, curve, cutoffs, level_factor).set_index("categoria")["demanda_eur"]
    frames = []
    for name, g in _cesp_splits(stores):
        if g.empty:
            continue
        part = stores_swap_breakdown(g, curve, cutoffs, level_factor)
        part.insert(0, "poblacion", name)
        part["pct_categoria"] = 100 * part["demanda_eur"] / part["categoria"].map(total)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 4) ingresos


def income_score_table(stores: pd.DataFrame, cutoffs: dict[str, float | None]) -> pd.DataFrame:
    """Score EFX por tramo de ingresos en tienda (meses con score), total y CESP / no CESP.

    ``bajo_corte_pct`` = demanda que la parrilla rechazaría por score (sin ``>=G``, que
    se rechaza por letra). ``swap_out_*`` es el caso que llega al vendedor: aprobada hoy
    y por debajo del corte EFX. ``swap_in_pct`` es el contrario: rechazada hoy por el
    score interno y por encima del corte.
    """
    scored = stores[stores[SCORE_COL].notna() & (stores[SEGMENT_COL] != ">=G")].copy()
    scored["_ingresos"] = income_band(scored)
    scored["_rango"] = score_range(scored)
    below = _below_cutoff(scored, cutoffs)
    approved = _approved(scored)
    score_ko = _score_ko(scored)
    rows = []
    for name, pop in _cesp_splits(scored):
        groups = [(band, pop[pop["_ingresos"] == band]) for band in INCOME_LABELS] + [(TOTAL, pop)]
        for band, g in groups:
            if g.empty:
                continue
            demand = _eur(g)
            row = {
                "poblacion": name,
                "ingresos": band,
                "solicitudes": int(len(g)),
                "demanda_eur": demand,
                "score_mediano": float(g[SCORE_COL].median()),
                "bajo_corte_pct": 100 * _eur(g[below.loc[g.index]]) / demand,
                "swap_out_pct": 100 * _eur(g[(below & approved).loc[g.index]]) / demand,
                "swap_out_n": int((below & approved).loc[g.index].sum()),
                "rechazo_score_hoy_pct": 100 * _eur(g[score_ko.loc[g.index]]) / demand,
                "swap_in_pct": 100 * _eur(g[(~below & score_ko).loc[g.index]]) / demand,
            }
            for label in SCORE_RANGE_LABELS:
                row[f"score_{label}_pct"] = 100 * _eur(g[g["_rango"] == label]) / demand
            rows.append(row)
    return pd.DataFrame(rows)


def income_risk(ecom: pd.DataFrame, stores_window: pd.DataFrame) -> pd.DataFrame:
    """Riesgo realizado (H6) por tramo de ingresos.

    Online: ingresos × rango de score (tiene las dos cosas). Tienda: solo por ingresos,
    total y CESP / no CESP — su histórico maduro no lleva EFX.
    """
    rows = []
    online = _booked(ecom[ecom[SCORE_COL].notna()]).copy()
    online["_ingresos"] = income_band(online)
    online["_rango"] = score_range(online)
    for band in INCOME_LABELS:
        in_band = online[online["_ingresos"] == band]
        for label in [*SCORE_RANGE_LABELS, TOTAL]:
            g = in_band if label == TOTAL else in_band[in_band["_rango"] == label]
            rows.append(_income_risk_row("online", TOTAL, band, label, g))
    for label in SCORE_RANGE_LABELS:
        rows.append(_income_risk_row("online", TOTAL, TOTAL, label, online[online["_rango"] == label]))
    booked = _booked(stores_window).copy()
    booked["_ingresos"] = income_band(booked)
    for name, pop in _cesp_splits(booked):
        for band in [*INCOME_LABELS, TOTAL]:
            g = pop if band == TOTAL else pop[pop["_ingresos"] == band]
            rows.append(_income_risk_row("tienda", name, band, "sin score", g))
    return pd.DataFrame(rows)


def _income_risk_row(channel: str, population: str, band: str, score: str, g: pd.DataFrame) -> dict:
    return {
        "canal": channel,
        "poblacion": population,
        "ingresos": band,
        "rango_score": score,
        "contratos": int(len(g)),
        "produccion_eur": float(g["oa_amt_h0"].sum()),
        "riesgo_h6_pct": _b2(g),
    }


__all__ = [
    "CESP_COL",
    "DRIFT_FACTORS",
    "INCOME_COL",
    "INCOME_LABELS",
    "SCORE_RANGE_LABELS",
    "SWAP_CATEGORIES",
    "cesp_score_distribution",
    "cesp_summary",
    "cesp_swap",
    "cesp_trend",
    "demand_profile",
    "drift_sensitivity",
    "income_risk",
    "income_score_table",
    "income_band",
    "risk_drift",
    "score_distribution",
    "score_range",
]
