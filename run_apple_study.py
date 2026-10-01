"""Estudio Apple — extensión del score Equifax de e-commerce (Online) a tienda (Stores).

Responde a los tres puntos de la petición de negocio:

1. TA y riesgo REAL de Apple Online (medido sobre cartera madura).
2. TA y riesgo de Apple Stores aplicando la parrilla de Online, como **swap de la
   regla de score**: se acepta si supera el corte EFX y no venía rechazada por un
   motivo distinto de ``09-score``. El resto de reglas de tienda no cambian.
3. Escenarios de riesgo global Apple (Online + tienda) a objetivos configurables.

Supuesto central, declarado: **tienda no tiene resultados observados** (las cohortes
de jul/ago-2026 tienen 0-2 meses y su exposición H6 es cero), así que su riesgo se
IMPUTA desde la curva riesgo-por-tramo de Online. No es una medición. La comparación
resultante usa la vara de medir de EFX y no demuestra que EFX supere al score interno.

Limitaciones frente al pipeline (``run_batch.py``), que es la fuente de los números
definitivos: aquí NO hay reject inference (se asume que quien entra en un tramo hoy
denegado se comporta como los contratados de ese tramo, lo cual es optimista), ni
intervalos de confianza, ni auditoría swap-in/swap-out.

Uso:
    uv run python run_apple_study.py
    uv run python run_apple_study.py --main-from 2024-06-01     # ventana de 21 meses
    uv run python run_apple_study.py --cutoff new=47 --cutoff known_cd=11
    uv run python run_apple_study.py --targets 2.5 3 3.5 4 --stores-level-factor 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import DEFAULT_RISK_MULTIPLIER, Columns, RejectReason, SystemDecision
from src.data_manager import load_data, standardize_columns_and_values

CHANNEL_COL = "segment_cutoff_1"
CHANNEL_ECOM = "e-commerce"
CHANNEL_STORES = "off-line"
SEGMENT_COL = "segment_cut_off"
SCORE_COL = "risk_score_rf"

# Tramos EFX del config.toml vigente. Fijos por diseño: el corte de negocio es un
# valor de score, y los bordes deben ser estables entre corridas para que "tramo 11"
# signifique siempre el mismo rango de score.
# Contratos mínimos para fiarse del take-up propio de un segmento en vez del global.
MIN_BOOKED_FOR_TAKE_UP = 100

BIN_EDGES: list[float] = [
    -np.inf,
    2,
    7,
    11,
    16,
    22,
    27,
    33,
    38,
    43,
    47,
    51,
    56,
    62,
    68,
    73,
    78,
    83,
    89,
    94,
    np.inf,
]

# Valor de corte que significa "este segmento se rechaza siempre": ningún score lo
# supera. El segmento sigue contando como demanda (denominador de la TA) pero nunca
# aporta producción ni exposición, que es como se comporta known_g en el dato real
# (0 contratos sobre 23,9 M€ de demanda Online).
ALWAYS_REJECT = float("inf")

# PLACEHOLDER — derivados de los cortes que el optimizador eligió en la corrida del
# 30-09-2026, NO de la parrilla viva. Sustituir por los cortes reales (anexo A.1 de
# reports/peticion_datos_apple_stores_efx.md). None = el segmento no tiene corte de score.
PLACEHOLDER_CUTOFFS: dict[str, float | None] = {
    "inactive": None,
    "known_ab": None,
    "known_cd": 11.0,
    "known_ef": 43.0,
    "new": 47.0,
    "known_g": ALWAYS_REJECT,  # decisión de negocio (30-09-2026): siempre rechazado
}


def load_study_data(data_path: str) -> pd.DataFrame:
    """Carga el extracto, estandariza y añade canal y tramo EFX."""
    data = standardize_columns_and_values(load_data(data_path, encoding="latin-1"))
    missing = {CHANNEL_COL, SEGMENT_COL, SCORE_COL} - set(data.columns)
    if missing:
        raise KeyError(f"El extracto no trae las columnas {sorted(missing)}")

    before = len(data)
    data = data[(data["fuera_norma"] == "n") & (data["fraud_flag"] == "n") & (data["nature_holder"] != "legal")].copy()
    logger.info(f"Filtros estándar: {before:,} -> {len(data):,} filas")

    data["bin"] = pd.cut(data[SCORE_COL], bins=BIN_EDGES, labels=range(1, len(BIN_EDGES))).astype(float)
    if data["bin"].isna().any():
        logger.warning(f"{int(data['bin'].isna().sum()):,} filas sin tramo (score nulo) — excluidas")
        data = data[data["bin"].notna()]
    return data


def _booked(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["acct_booked_h0"] > 0]


def estimate_risk_curve(ecom: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Curva riesgo-por-tramo estimada sobre la cartera contratada de Online.

    Devuelve ``b2_pct`` (riesgo realizado del tramo) y ``exposure_ratio``
    (exposición H6 por euro financiado), que convierte producción proyectada en
    exposición proyectada — el peso con el que se agrega el riesgo.
    """
    agg = (
        _booked(ecom)
        .groupby("bin")
        .agg(num=("todu_30ever_h6", "sum"), den=("todu_amt_pile_h6", "sum"), booked_eur=("oa_amt_h0", "sum"))
    )
    curve = pd.DataFrame(index=agg.index)
    curve["b2_pct"] = multiplier * 100 * agg["num"] / agg["den"]
    curve["exposure_ratio"] = agg["den"] / agg["booked_eur"]
    curve["booked_eur"] = agg["booked_eur"]
    thin = curve[curve["booked_eur"] < 0.005 * curve["booked_eur"].sum()]
    if not thin.empty:
        logger.warning(
            f"Tramos con muy poca producción, riesgo poco fiable: {sorted(thin.index.astype(int))} "
            "— son los que abren los escenarios más laxos."
        )
    return curve


def imputed_risk(applications: pd.DataFrame, curve: pd.DataFrame, level_factor: float = 1.0) -> float:
    """Riesgo imputado de un conjunto de solicitudes, ponderando por exposición esperada.

    La tasa de financiación se cancela en el cociente si es plana entre tramos, así que
    el peso es demanda x exposure_ratio. ``level_factor`` reescala el nivel de la curva
    (para anclar tienda a su morosidad real; 1.0 = curva de Online tal cual).
    """
    demand = applications.groupby("bin")[Columns.OA_AMT].sum()
    weight = (demand * curve["exposure_ratio"]).dropna()
    if weight.sum() <= 0:
        return float("nan")
    return float(level_factor * (weight * curve["b2_pct"]).sum() / weight.sum())


def grid_swap_mask(stores: pd.DataFrame, cutoffs: dict[str, float | None]) -> pd.Series:
    """Máscara del swap de la regla de score (sólo cambia el corte EFX).

    Aceptada si supera el corte de su segmento Y no venía rechazada por un motivo
    distinto de ``09-score``. Mismo criterio que el ``System Rejection Rate`` del
    pipeline, para que los números sean comparables con el motor de decisión.
    """
    threshold = stores[SEGMENT_COL].map(cutoffs).astype(float).fillna(-np.inf)
    passes_score = stores[SCORE_COL] > threshold
    rejected_elsewhere = (stores[Columns.SE_DECISION_ID] == SystemDecision.KO) & (
        stores[Columns.REJECT_REASON] != RejectReason.SCORE
    )
    return passes_score & ~rejected_elsewhere


def online_actuals(ecom: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Punto 1: TA y riesgo realizado de Online, por segmento y total. Todo medido."""
    rows = []
    for segment, group in list(ecom.groupby(SEGMENT_COL, observed=True)) + [("TOTAL", ecom)]:
        booked = _booked(group)
        rows.append(
            {
                "segmento": segment,
                "demanda_n": len(group),
                "demanda_eur": group[Columns.OA_AMT].sum(),
                "booked_n": len(booked),
                "produccion_eur": booked[Columns.OA_AMT_H0].sum(),
                "ta_pct": 100 * booked[Columns.OA_AMT_H0].sum() / group[Columns.OA_AMT].sum(),
                "riesgo_real_pct": multiplier * 100 * booked["todu_30ever_h6"].sum() / booked["todu_amt_pile_h6"].sum(),
            }
        )
    return pd.DataFrame(rows)


def stores_under_grid(
    stores: pd.DataFrame, curve: pd.DataFrame, cutoffs: dict[str, float | None], level_factor: float
) -> pd.DataFrame:
    """Punto 2: TA y riesgo IMPUTADO de tienda bajo la parrilla de Online.

    El take-up (qué parte de lo aprobado acaba contratado) se mide **por segmento**:
    varía mucho entre ellos, y aplicar una tasa global hacía que algún segmento
    apareciera perdiendo TA al cambiar de parrilla cuando en realidad solo tenía un
    take-up por encima de la media. Los segmentos con poca contratación caen a la
    tasa global, que es más estable que su propia proporción.
    """
    accepted_all = grid_swap_mask(stores, cutoffs)
    approved = stores[stores[Columns.SE_DECISION_ID] == SystemDecision.OK]
    global_take_up = _booked(stores)[Columns.OA_AMT_H0].sum() / approved[Columns.OA_AMT].sum()
    logger.info(f"Take-up global en tienda sobre aprobado: {100 * global_take_up:.1f}%")

    def _take_up(group: pd.DataFrame, name: str) -> float:
        booked = _booked(group)
        ok_eur = group[group[Columns.SE_DECISION_ID] == SystemDecision.OK][Columns.OA_AMT].sum()
        if len(booked) < MIN_BOOKED_FOR_TAKE_UP or ok_eur <= 0:
            logger.warning(
                f"[{name}] solo {len(booked)} contratos: take-up propio poco fiable, se usa el global "
                f"({100 * global_take_up:.1f}%)"
            )
            return global_take_up
        return booked[Columns.OA_AMT_H0].sum() / ok_eur

    rows = []
    groups = list(stores.groupby(SEGMENT_COL, observed=True)) + [("TOTAL", stores)]
    for segment, group in groups:
        mask = accepted_all.loc[group.index]
        accepted, booked = group[mask], _booked(group)
        demand = group[Columns.OA_AMT].sum()
        take_up = global_take_up if segment == "TOTAL" else _take_up(group, segment)
        rows.append(
            {
                "segmento": segment,
                "corte_efx": cutoffs.get(segment) if segment != "TOTAL" else None,
                "demanda_eur": demand,
                "pct_demanda_aceptada": 100 * accepted[Columns.OA_AMT].sum() / demand,
                "produccion_est_eur": accepted[Columns.OA_AMT].sum() * take_up,
                "ta_efectiva_pct": 100 * accepted[Columns.OA_AMT].sum() * take_up / demand,
                "riesgo_imputado_pct": imputed_risk(accepted, curve, level_factor),
                "ta_actual_pct": 100 * booked[Columns.OA_AMT_H0].sum() / demand,
                "riesgo_imputado_actual_pct": imputed_risk(booked, curve, level_factor),
                "take_up_pct": 100 * take_up,
            }
        )
    frame = pd.DataFrame(rows)
    # el TOTAL se re-agrega desde los segmentos: con take-up por segmento, la suma de
    # las partes ya no coincide con aplicar la tasa global al agregado.
    seg_rows = frame[frame["segmento"] != "TOTAL"]
    total_idx = frame.index[frame["segmento"] == "TOTAL"][0]
    total_demand = seg_rows["demanda_eur"].sum()
    frame.loc[total_idx, "produccion_est_eur"] = seg_rows["produccion_est_eur"].sum()
    frame.loc[total_idx, "ta_efectiva_pct"] = 100 * seg_rows["produccion_est_eur"].sum() / total_demand
    return frame


def scenario_ladder(
    ecom: pd.DataFrame,
    stores: pd.DataFrame,
    curve: pd.DataFrame,
    cutoffs: dict[str, float | None],
    months_ecom: float,
    months_stores: float,
    level_factor: float,
) -> pd.DataFrame:
    """Punto 3: un corte EFX único para todo Apple, tramo a tramo.

    Ambos canales se normalizan a run-rate mensual: Online cubre muchos más meses que
    tienda, y sumar totales crudos sobre-pondería Online en el mix.

    Los segmentos marcados ``ALWAYS_REJECT`` cuentan en la demanda (denominador de la
    TA) pero nunca en la parte aceptada, así que bajan la TA sin tocar el riesgo.
    """
    blocked = {seg for seg, cut in cutoffs.items() if cut == ALWAYS_REJECT}
    eligible_ecom = ecom[~ecom[SEGMENT_COL].isin(blocked)]
    eligible_stores = stores[~stores[SEGMENT_COL].isin(blocked)]
    if blocked:
        logger.info(f"Segmentos siempre rechazados (solo demanda, nunca producción): {sorted(blocked)}")

    # La tasa de transformación se mide sobre la demanda elegible: un segmento que nunca
    # contrata la deprimiría artificialmente.
    tf_ecom = _booked(eligible_ecom)[Columns.OA_AMT_H0].sum() / eligible_ecom[Columns.OA_AMT].sum()
    tf_stores = _booked(eligible_stores)[Columns.OA_AMT_H0].sum() / eligible_stores[Columns.OA_AMT].sum()
    dem_ecom = eligible_ecom.groupby("bin")[Columns.OA_AMT].sum() / months_ecom
    dem_stores = eligible_stores.groupby("bin")[Columns.OA_AMT].sum() / months_stores
    total_demand = ecom[Columns.OA_AMT].sum() / months_ecom + stores[Columns.OA_AMT].sum() / months_stores
    w_ecom = (dem_ecom * curve["exposure_ratio"] * tf_ecom).fillna(0)
    w_stores = (dem_stores * curve["exposure_ratio"] * tf_stores).fillna(0) * level_factor

    rows = []
    for cut in curve.index:
        sel = curve.index >= cut
        weight = w_ecom[sel].sum() + w_stores[sel].sum()
        risk = ((w_ecom[sel] + w_stores[sel]) * curve.loc[sel, "b2_pct"]).sum() / weight
        acc_e, acc_s = dem_ecom[dem_ecom.index >= cut].sum(), dem_stores[dem_stores.index >= cut].sum()
        rows.append(
            {
                "corte_tramo": int(cut),
                "corte_score": BIN_EDGES[int(cut) - 1],
                "riesgo_total_pct": risk,
                "ta_apple_pct": 100 * (acc_e + acc_s) / total_demand,
                "produccion_mensual_eur": acc_e * tf_ecom + acc_s * tf_stores,
                "ta_online_pct": 100 * acc_e / (ecom[Columns.OA_AMT].sum() / months_ecom),
                "ta_tienda_pct": 100 * acc_s / (stores[Columns.OA_AMT].sum() / months_stores),
            }
        )
    return pd.DataFrame(rows)


def pick_targets(ladder: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    """Corte más laxo que cumple cada objetivo de riesgo global."""
    rows = []
    for target in targets:
        feasible = ladder[ladder["riesgo_total_pct"] <= target]
        if feasible.empty:
            rows.append({"objetivo_pct": target, "corte_tramo": None, "nota": "no alcanzable"})
            continue
        best = feasible.iloc[0]
        rows.append(
            {
                "objetivo_pct": target,
                "corte_tramo": int(best["corte_tramo"]),
                "corte_score": best["corte_score"],
                "riesgo_logrado_pct": best["riesgo_total_pct"],
                "ta_apple_pct": best["ta_apple_pct"],
                "produccion_mensual_eur": best["produccion_mensual_eur"],
                "nota": "",
            }
        )
    return pd.DataFrame(rows)


def seasonal_index(ecom: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Índice estacional multiplicativo mes-del-año, estimado sobre la demanda de Online.

    Apple es extremadamente estacional: el lanzamiento de iPhone concentra la demanda en
    septiembre-octubre, y agosto es el mes más bajo del año (~0,53 frente a ~1,94 de
    septiembre, un factor 3,6x). Sin corregirlo, un canal observado solo en agosto parece
    tres veces más pequeño de lo que es en run-rate anual.

    La ventana debe cubrir un múltiplo entero de 12 meses naturales para que la media del
    índice sea 1 y el ajuste no introduzca sesgo de nivel.
    """
    window = ecom[(ecom[Columns.MIS_DATE] >= start) & (ecom[Columns.MIS_DATE] < end)]
    n_months = len(pd.period_range(pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(days=1), freq="M"))
    if n_months % 12:
        logger.warning(
            f"La ventana del índice estacional cubre {n_months} meses, no un múltiplo de 12: "
            "la media no será 1 y el ajuste sesgará el nivel."
        )
    monthly = window.groupby(window[Columns.MIS_DATE].dt.month)[Columns.OA_AMT].sum()
    if len(monthly) < 12:
        raise ValueError(f"El índice estacional necesita los 12 meses naturales, hay {len(monthly)}")
    index = monthly / monthly.mean()
    logger.info(
        f"Índice estacional ({start} a {end}): mín {index.min():.2f} (mes {index.idxmin()}), "
        f"máx {index.max():.2f} (mes {index.idxmax()})"
    )
    return index


def effective_months(start: str, end: str, index: pd.Series | None) -> float:
    """Meses efectivos de una ventana: suma del índice estacional de los meses cubiertos.

    Sustituye al conteo crudo de meses como divisor del run-rate. Una ventana de 12 meses
    naturales suma ~12 (el índice tiene media 1), así que para Online no cambia nada; una
    ventana de un solo agosto suma ~0,53 y el run-rate se corrige por ser el mes más flojo.
    Los meses parcialmente cubiertos se prorratean por días.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    total = 0.0
    for period in pd.period_range(start_ts, end_ts - pd.Timedelta(days=1), freq="M"):
        month_start = period.start_time
        month_end = period.end_time.normalize() + pd.Timedelta(days=1)
        covered = (min(end_ts, month_end) - max(start_ts, month_start)).days / (month_end - month_start).days
        total += covered * (1.0 if index is None else float(index.loc[period.month]))
    return max(total, 1e-6)


def warn_partial_rollout(stores: pd.DataFrame, index: pd.Series, threshold: float = 0.5) -> list[str]:
    """Detecta meses de tienda con volumen muy por debajo de su perfil estacional.

    El score EFX se activó en tienda en agosto, pero el extracto trae también julio con
    una fracción del volumen: un mes de rollout parcial, no un mes de operación. Incluirlo
    distorsiona el mix y el run-rate, así que conviene detectarlo en vez de confiar en que
    alguien recuerde excluirlo.
    """
    monthly = stores.groupby(stores[Columns.MIS_DATE].dt.to_period("M"))[Columns.OA_AMT].sum()
    if len(monthly) < 2:
        return []
    per_effective = monthly / monthly.index.map(lambda p: float(index.loc[p.month]))
    suspicious = per_effective[per_effective < threshold * per_effective.max()]
    for period, value in suspicious.items():
        logger.warning(
            f"Tienda {period}: {value / per_effective.max():.0%} del volumen desestacionalizado del mejor mes "
            "— parece rollout parcial, no un mes de operación. Excluir con --stores-from."
        )
    return [str(p) for p in suspicious.index]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default="data/demanda_apple.sas7bdat")
    p.add_argument("--main-from", default="2025-03-01", help="Inicio ventana Online (default: 12 meses)")
    p.add_argument("--main-to", default="2026-03-01")
    p.add_argument(
        "--stores-from",
        default="2026-08-01",
        help="Inicio ventana tienda. Default agosto: julio fue rollout parcial (~9% del volumen esperado)",
    )
    p.add_argument("--stores-to", default="2026-09-01")
    p.add_argument(
        "--cutoff",
        action="append",
        default=[],
        metavar="SEG=SCORE",
        help="Corte EFX real por segmento, p.ej. --cutoff new=47. Sobrescribe el placeholder.",
    )
    p.add_argument("--targets", nargs="+", type=float, default=[2.5, 3.0, 3.5, 4.0])
    p.add_argument(
        "--stores-level-factor",
        type=float,
        default=1.0,
        help="Reescala el nivel de riesgo de tienda (1.0 = curva de Online sin anclar)",
    )
    p.add_argument("--multiplier", type=float, default=float(DEFAULT_RISK_MULTIPLIER))
    p.add_argument("--seasonal-from", default="2024-09-01", help="Ventana del índice estacional (12 meses naturales)")
    p.add_argument("--seasonal-to", default="2025-09-01")
    p.add_argument(
        "--no-seasonal-adjust",
        action="store_true",
        help="Desactiva el ajuste estacional y vuelve al conteo crudo de meses (no recomendado)",
    )
    p.add_argument("--output", default="output/apple_study")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cutoffs = dict(PLACEHOLDER_CUTOFFS)
    for override in args.cutoff:
        segment, _, value = override.partition("=")
        cutoffs[segment.strip()] = None if value.strip().lower() in {"", "none"} else float(value)
    if not args.cutoff:
        logger.warning(
            "Usando los cortes PLACEHOLDER (los que eligió el optimizador, no la parrilla viva). "
            "Pasar --cutoff SEG=SCORE con los cortes reales antes de presentar resultados."
        )
    if args.stores_level_factor != 1.0:
        logger.info(f"Nivel de riesgo de tienda reescalado x{args.stores_level_factor}")

    data = load_study_data(args.data_path)
    data = data[data[SEGMENT_COL].isin(cutoffs)]
    ecom = data[
        (data[CHANNEL_COL] == CHANNEL_ECOM)
        & (data[Columns.MIS_DATE] >= args.main_from)
        & (data[Columns.MIS_DATE] < args.main_to)
    ]
    stores = data[
        (data[CHANNEL_COL] == CHANNEL_STORES)
        & (data[Columns.MIS_DATE] >= args.stores_from)
        & (data[Columns.MIS_DATE] < args.stores_to)
    ]
    if ecom.empty or stores.empty:
        logger.error(f"Ventana vacía: Online {len(ecom):,} filas, tienda {len(stores):,}. Revisar fechas y canal.")
        return 1
    logger.info(f"Online {len(ecom):,} solicitudes ({args.main_from} a {args.main_to}) | tienda {len(stores):,}")

    all_ecom = data[data[CHANNEL_COL] == CHANNEL_ECOM]
    index = None if args.no_seasonal_adjust else seasonal_index(all_ecom, args.seasonal_from, args.seasonal_to)
    if index is None:
        logger.warning(
            "Ajuste estacional DESACTIVADO: el run-rate de un canal observado en pocos meses no es comparable"
        )
    else:
        warn_partial_rollout(
            data[(data[CHANNEL_COL] == CHANNEL_STORES) & (data[Columns.MIS_DATE] >= args.stores_from)], index
        )
    eff_ecom = effective_months(args.main_from, args.main_to, index)
    eff_stores = effective_months(args.stores_from, args.stores_to, index)
    logger.info(f"Meses efectivos: Online {eff_ecom:.2f} | tienda {eff_stores:.2f}")

    curve = estimate_risk_curve(ecom, args.multiplier)
    actuals = online_actuals(ecom, args.multiplier)
    grid = stores_under_grid(stores, curve, cutoffs, args.stores_level_factor)
    ladder = scenario_ladder(
        ecom,
        stores,
        curve,
        cutoffs,
        eff_ecom,
        eff_stores,
        args.stores_level_factor,
    )
    picks = pick_targets(ladder, args.targets)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("curva_riesgo_online", curve.reset_index()),
        (
            "indice_estacional",
            (pd.DataFrame() if index is None else index.rename("indice").rename_axis("mes").reset_index()),
        ),
        ("p1_online_actual", actuals),
        ("p2_tienda_parrilla_online", grid),
        ("p3_escalera_escenarios", ladder),
        ("p3_escenarios_objetivo", picks),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
    logger.info(f"Resultados en {out}/ (5 ficheros)")

    print("\n=== 1) APPLE ONLINE — actual, medido ===")
    print(actuals.round(2).to_string(index=False))
    print("\n=== 2) APPLE STORES — parrilla de Online (riesgo IMPUTADO) ===")
    print(grid.round(2).to_string(index=False))
    print("\n=== 3) ESCENARIOS — corte EFX único para todo Apple ===")
    print(picks.round(2).to_string(index=False))
    print("\nRiesgo de tienda imputado desde Online: NO es una medición (ver docstring).")
    print("Sin reject inference: los riesgos son optimistas. Números definitivos vía run_batch.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
