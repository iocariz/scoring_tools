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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import DEFAULT_RISK_MULTIPLIER, Columns, RejectReason, SystemDecision
from src.data_manager import load_data, standardize_columns_and_values

_BOOKED_COL = "b2_pct_contratados"  # curva sin corregir por selección; ver apply_reject_inference

CHANNEL_COL = "segment_cutoff_1"
CHANNEL_ECOM = "e-commerce"
CHANNEL_STORES = "off-line"
SEGMENT_COL = "grupo_parrilla"
LETTER_COL = "scrv_customer_init"  # estandarizado a minúsculas por data_manager
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

# Grupos de la parrilla, derivados de SCRV_customer_init (la letra). OJO: no coinciden
# con `segment_cut_off`, que agrupa A-B / C-D / E-F; la parrilla usa A-C / D-F, así que
# `known_cd` queda partido entre las dos reglas. La política manda, de modo que el
# estudio segmenta por esta columna.
PARRILLA_GROUPS: dict[str, str] = (
    dict.fromkeys("ABC", "A-C")
    | dict.fromkeys("DEF", "D-F")
    | dict.fromkeys("GHIJKQRWX", ">=G")
    | dict.fromkeys(["Y", "Z", "5", "7"], "Inactive")
)
PARRILLA_DEFAULT_GROUP = "New"

# Parrilla vigente de Apple Online (ecommerce_cutoff.xlsx, confirmada 01-10-2026).
# El valor es el umbral de RECHAZO: la hoja dice "<= X", así que se acepta por encima.
# Verificado contra el dato: la lectura inversa rechazaría el 80-95% de la demanda.
PARRILLA_ECOMMERCE: dict[str, float | None] = {
    "New": 27.0,
    "Inactive": 22.0,
    "A-C": 16.0,
    "D-F": 22.0,
    ">=G": ALWAYS_REJECT,  # "all rj" en la hoja; confirmado: 0 contratos en ambos canales
}


def load_study_data(data_path: str) -> pd.DataFrame:
    """Carga el extracto, estandariza y añade canal y tramo EFX."""
    data = standardize_columns_and_values(load_data(data_path, encoding="latin-1"))
    missing = {CHANNEL_COL, LETTER_COL, SCORE_COL} - set(data.columns)
    if missing:
        raise KeyError(f"El extracto no trae las columnas {sorted(missing)}")

    before = len(data)
    data = data[(data["fuera_norma"] == "n") & (data["fraud_flag"] == "n") & (data["nature_holder"] != "legal")].copy()
    logger.info(f"Filtros estándar: {before:,} -> {len(data):,} filas")

    data[SEGMENT_COL] = data[LETTER_COL].astype(str).str.upper().map(PARRILLA_GROUPS).fillna(PARRILLA_DEFAULT_GROUP)
    logger.info(f"Grupos de parrilla: {dict(data[SEGMENT_COL].value_counts())}")

    data["bin"] = pd.cut(data[SCORE_COL], bins=BIN_EDGES, labels=range(1, len(BIN_EDGES))).astype(float)
    # Las filas sin score NO se descartan aquí: el histórico de tienda no lleva EFX
    # —esas decisiones las tomó el octroi interno— y es justo el que ancla el nivel de
    # riesgo (ver stores_level_anchor). Se filtran donde hace falta el tramo.
    if data["bin"].isna().any():
        logger.info(f"{int(data['bin'].isna().sum()):,} filas sin score: fuera del análisis EFX, sí para el ancla")
    return data


def stores_level_anchor(all_stores: pd.DataFrame, scored_booked: pd.DataFrame, curve: pd.DataFrame) -> dict[str, float]:
    """Factor que reescala la curva de Online para reproducir la morosidad real de tienda.

    El supuesto del estudio tiene dos componentes: la **forma** de la curva —cuánto
    discrimina el score— y su **nivel**. La forma hay que asumirla; el nivel no, porque
    tienda tiene morosidad observada propia. Se compara lo que la curva de Online predice
    sobre la cartera contratada de tienda con lo que tienda realmente ha tenido, y se
    reescala. Así el único supuesto que queda es el poder discriminante.

    El realizado usa TODO el histórico de tienda, que no lleva EFX (no hace falta: es un
    nivel agregado); el imputado usa la cartera que sí tiene score.
    """
    mature = all_stores[(all_stores["acct_booked_h0"] > 0) & (all_stores["todu_amt_pile_h6"] > 0)]
    if mature.empty:
        logger.warning("Sin cartera madura en tienda: no se puede anclar el nivel, se usa la curva de Online tal cual")
        return {"factor": 1.0, "realizado_pct": float("nan"), "imputado_pct": float("nan")}
    realized = DEFAULT_RISK_MULTIPLIER * 100 * mature["todu_30ever_h6"].sum() / mature["todu_amt_pile_h6"].sum()
    imputed = imputed_risk(scored_booked, curve)
    if not np.isfinite(imputed) or imputed <= 0:
        logger.warning("Riesgo imputado no calculable: no se ancla el nivel")
        return {"factor": 1.0, "realizado_pct": realized, "imputado_pct": float("nan")}
    factor = realized / imputed
    logger.info(
        f"Ancla de nivel de tienda: realizado {realized:.2f}% ({len(mature):,} contratos maduros) frente a "
        f"imputado {imputed:.2f}% -> factor {factor:.4f} "
        f"(la curva de Online {'sobreestima' if factor < 1 else 'subestima'} el riesgo de tienda "
        f"un {abs(100 * (1 / factor - 1)):.1f}%)"
    )
    return {"factor": factor, "realizado_pct": realized, "imputado_pct": imputed}


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


def apply_reject_inference(curve: pd.DataFrame, demand: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Añade a la curva el riesgo corregido por selección, reutilizando el *parceling* del pipeline.

    El riesgo por tramo se mide sobre los contratados, que son una muestra **seleccionada**:
    dentro del mismo tramo pasaron además el resto de reglas y aceptaron la oferta. Usar su
    morosidad para los que hoy se deniegan la subestima, y es justo a ellos a quienes abre
    un corte más laxo. El pipeline corrige eso con parceling: cuanto menor es la tasa de
    aceptación de un tramo, más seleccionado está y mayor el multiplicador.

    Se reutilizan ``compute_acceptance_rates`` y ``apply_parceling_adjustment`` de
    :mod:`src.reject_inference` —no una versión propia— para que el estudio y la corrida
    definitiva usen exactamente el mismo ajuste.

    El resultado NO es un recargo plano sobre el tramo: dentro de cada uno se mezcla la
    parte ya contratada (a su riesgo realizado) con la que entraría nueva (al riesgo
    corregido), en proporción a la tasa de aceptación observada.
    """
    from src.reject_inference import apply_parceling_adjustment, compute_acceptance_rates

    rates = compute_acceptance_rates(
        demand,
        ["bin"],
        bayesian_smoothing=kwargs.get("bayesian_smoothing", True),
        bayesian_prior_strength=kwargs.get("bayesian_prior_strength", 20.0),
    )
    adjusted = apply_parceling_adjustment(
        curve.reset_index()[["bin", "b2_pct"]].rename(columns={"b2_pct": "todu_30ever_h6"}),
        rates,
        ["bin"],
        reject_uplift_factor=kwargs.get("uplift", 1.5),
        max_risk_multiplier=kwargs.get("max_multiplier", 3.0),
        method=kwargs.get("method", "linear"),
        enforce_monotonicity=kwargs.get("enforce_monotonicity", True),
        # OBLIGATORIO: el tramo va de peor a mejor score, así que el multiplicador debe
        # DECRECER. Sin declararlo, la isotónica exige lo contrario, "corrige" 181 pares
        # y aplana el multiplicador a una constante — un recargo plano disfrazado de
        # reject inference, que además no da error.
        inv_vars=["bin"],
        quiet=True,
    ).set_index("bin")

    out = curve.copy()
    # b2 es proporcional al numerador, así que el multiplicador del parceling lo escala igual
    out["b2_pct_rechazados"] = adjusted["todu_30ever_h6"].reindex(out.index)
    out["tasa_aceptacion"] = rates.set_index("bin")["acceptance_rate"].reindex(out.index).clip(0, 1).fillna(0)
    out["b2_pct_ri"] = out["tasa_aceptacion"] * out["b2_pct"] + (1 - out["tasa_aceptacion"]) * out["b2_pct_rechazados"]
    # Ponderado por producción: el agregado crudo lo dominarían los tramos bajos, que
    # multiplican por 2,4 pero no tienen volumen, y daría una cifra irreconocible.
    w = out["booked_eur"]
    lift = (w * out["b2_pct_ri"]).sum() / (w * out["b2_pct"]).sum()
    mult = out["b2_pct_rechazados"].div(out["b2_pct"])
    logger.info(
        f"Reject inference: multiplicador sobre los denegados entre {mult.min():.2f}x y {mult.max():.2f}x "
        f"(más uplift donde menos se acepta); ponderada por producción, la curva sube un {100 * (lift - 1):.1f}%"
    )
    return out


def imputed_risk(
    applications: pd.DataFrame, curve: pd.DataFrame, level_factor: float = 1.0, column: str = "b2_pct"
) -> float:
    """Riesgo imputado de un conjunto de solicitudes, ponderando por exposición esperada.

    La tasa de financiación se cancela en el cociente si es plana entre tramos, así que
    el peso es demanda x exposure_ratio. ``level_factor`` reescala el nivel de la curva
    (para anclar tienda a su morosidad real; 1.0 = curva de Online tal cual).
    """
    demand = applications.groupby("bin")[Columns.OA_AMT].sum()
    weight = (demand * curve["exposure_ratio"]).dropna()
    if weight.sum() <= 0:
        return float("nan")
    return float(level_factor * (weight * curve[column]).sum() / weight.sum())


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


def channel_actuals(channel: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """TA y riesgo REALIZADO de un canal, por grupo y total. Todo medido, sin supuestos.

    Vale para los dos canales: tienda tiene histórico propio con resultados realizados,
    así que su situación actual se mide igual que la de Online —no hace falta imputarla—.
    Lo único que se imputa en el estudio es tienda **bajo la parrilla de Online**, porque
    para eso sí hace falta el score EFX, que solo existe desde agosto.
    """
    rows = []
    for segment, group in list(channel.groupby(SEGMENT_COL, observed=True)) + [("TOTAL", channel)]:
        booked = _booked(group)
        rows.append(
            {
                "segmento": segment,
                "demanda_n": len(group),
                "demanda_eur": group[Columns.OA_AMT].sum(),
                "booked_n": len(booked),
                "produccion_eur": booked[Columns.OA_AMT_H0].sum(),
                "ta_pct": 100 * booked[Columns.OA_AMT_H0].sum() / group[Columns.OA_AMT].sum(),
                "riesgo_real_pct": (
                    multiplier * 100 * mature["todu_30ever_h6"].sum() / mature["todu_amt_pile_h6"].sum()
                    if not (mature := booked[booked["todu_amt_pile_h6"] > 0]).empty
                    else float("nan")
                ),
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
                # La cartera ya contratada ES la población seleccionada: valorarla con la
                # curva corregida por selección sería corregir dos veces. Va con la de
                # contratados; el uplift solo pesa sobre lo que entraría nuevo.
                "riesgo_imputado_actual_pct": imputed_risk(booked, curve, level_factor, column=_BOOKED_COL),
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


def policy_kpis(
    ecom: pd.DataFrame,
    stores: pd.DataFrame,
    curve: pd.DataFrame,
    cutoffs: dict[str, float | None],
    months_ecom: float,
    months_stores: float,
    level_factor: float,
) -> dict[str, float]:
    """KPIs de Apple bajo una política cualquiera, expresada como corte por grupo.

    Devuelve **dos tasas de aceptación distintas**, porque miden cosas distintas y
    confundirlas da lecturas muy equivocadas:

    * ``ta_score_pct`` — demanda que supera el corte. Aísla la palanca que se mueve.
    * ``ta_efectiva_pct`` — producción sobre demanda, la misma base que el punto 1.
      Incorpora las demás reglas y el take-up, así que es la comparable con "hoy".

    Ambos canales se normalizan a run-rate mensual con su propia estacionalidad.
    """
    total_demand = ecom[Columns.OA_AMT].sum() / months_ecom + stores[Columns.OA_AMT].sum() / months_stores
    num = den = accepted = production = 0.0
    for channel, months, factor in ((ecom, months_ecom, 1.0), (stores, months_stores, level_factor)):
        # Mismo modelo de swap que grid_swap_mask: solo cambia la regla de score, las demás
        # reglas del canal se mantienen. Antes esto solo aplicaba el umbral, de modo que la
        # escalera suponía que TODO el que supera el corte entra — en contra de la premisa
        # del estudio— y además inflaba la población sobre la que se mide el riesgo.
        eligible = channel[grid_swap_mask(channel, cutoffs)]
        # El take-up es CONDICIONAL a estar aprobado. Medido sobre la demanda total llevaría
        # dentro el rechazo por score, y al aplicarlo al aceptado lo contaría dos veces:
        # la producción de la política vigente salía un 16% por debajo de la real.
        approved = channel[channel[Columns.SE_DECISION_ID] == SystemDecision.OK]
        take_up = _booked(channel)[Columns.OA_AMT_H0].sum() / max(approved[Columns.OA_AMT].sum(), 1e-9)
        accepted_rows = eligible
        demand_by_bin = accepted_rows.groupby("bin")[Columns.OA_AMT].sum() / months
        weight = (demand_by_bin * curve["exposure_ratio"] * take_up).fillna(0) * factor
        num += (weight * curve["b2_pct"]).sum()
        den += weight.sum()
        accepted += accepted_rows[Columns.OA_AMT].sum() / months
        production += accepted_rows[Columns.OA_AMT].sum() / months * take_up
    return {
        "riesgo_total_pct": num / den if den else float("nan"),
        "ta_score_pct": 100 * accepted / total_demand,
        "ta_efectiva_pct": 100 * production / total_demand,
        "produccion_mensual_eur": production,
    }


def _uniform(cutoffs: dict[str, float | None], score: float) -> dict[str, float | None]:
    """Mismo corte para todos los grupos, respetando los que se rechazan siempre."""
    return {g: (ALWAYS_REJECT if c == ALWAYS_REJECT else score) for g, c in cutoffs.items()}


def _shifted(cutoffs: dict[str, float | None], delta: float) -> dict[str, float | None]:
    """La parrilla actual desplazada ``delta`` puntos de score."""
    return {
        g: (ALWAYS_REJECT if c == ALWAYS_REJECT else (delta if c is None else c + delta)) for g, c in cutoffs.items()
    }


def scenario_ladders(
    ecom: pd.DataFrame,
    stores: pd.DataFrame,
    curve: pd.DataFrame,
    cutoffs: dict[str, float | None],
    months_ecom: float,
    months_stores: float,
    level_factor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dos instrumentos alternativos para alcanzar un objetivo de riesgo.

    * **corte único** para todo Apple: simple de explicar y de operar.
    * **parrilla actual desplazada**: mantiene la forma de la política viva —laxa donde
      el grupo es bueno, dura donde es malo— y por eso suele dar algo más de aceptación
      al mismo riesgo.
    """
    pos = (ecom, stores, curve)
    rest = (months_ecom, months_stores, level_factor)
    uniform = pd.DataFrame(
        [
            {"corte_tramo": int(cut), "corte_score": BIN_EDGES[int(cut) - 1]}
            | policy_kpis(*pos, _uniform(cutoffs, BIN_EDGES[int(cut) - 1]), *rest)
            for cut in curve.index
        ]
    )
    shifted = pd.DataFrame(
        [{"desplazamiento": d} | policy_kpis(*pos, _shifted(cutoffs, d), *rest) for d in range(0, 61)]
    )
    return uniform, shifted


def measured_status_quo(
    ecom: pd.DataFrame, stores_window: pd.DataFrame, months_ecom: float, multiplier: float
) -> dict[str, float]:
    """El statu quo MEDIDO: lo que los dos canales hacen hoy, cada uno con su política.

    No es comparable columna a columna con las filas modeladas: aquí tienda va con su
    score interno sobre la ventana completa, mientras los escenarios la evalúan con la
    parrilla EFX sobre agosto, el único mes con score. Se publica precisamente para que
    esa diferencia se vea en lugar de sorprender: el modelo dice que la parrilla de
    Online produciría más de lo que hoy se produce, y conviene saber cuánto de eso es
    el cambio propuesto y cuánto es que la parrilla vigente es más laxa que la política
    que de hecho estuvo en vigor durante la ventana.
    """
    frames = [(ecom, months_ecom), (stores_window, months_ecom)]
    demand = sum(f[Columns.OA_AMT].sum() / m for f, m in frames)
    production = sum(_booked(f)[Columns.OA_AMT_H0].sum() / m for f, m in frames)
    mature = pd.concat([_booked(f)[_booked(f)["todu_amt_pile_h6"] > 0] for f, _ in frames])
    return {
        "riesgo_total_pct": multiplier * 100 * mature["todu_30ever_h6"].sum() / mature["todu_amt_pile_h6"].sum(),
        "ta_score_pct": float("nan"),  # tienda no decide por score EFX hoy: no hay tasa de score que dar
        "ta_efectiva_pct": 100 * production / demand,
        "produccion_mensual_eur": production,
    }


def pick_targets(
    uniform: pd.DataFrame,
    shifted: pd.DataFrame,
    baseline: dict[str, float],
    measured: dict[str, float],
    targets: list[float],
) -> pd.DataFrame:
    """Para cada objetivo, la política más laxa que lo cumple con cada instrumento.

    La primera fila es la **parrilla actual**: sin ella los escenarios se leen en el
    vacío y no se ve que los cuatro objetivos son endurecimientos.
    """
    # OJO con el nombre: esta fila NO es el statu quo. Es la parrilla de Online aplicada a
    # los DOS canales, que para tienda es justamente el cambio propuesto (hoy usa su score
    # interno). Y para Online la parrilla vigente es ~1,12x más laxa que la política media
    # de la ventana de observación, porque el canal ha ido aflojando. Llamarla "actual"
    # invita a leerla como "lo que hacemos hoy", y no lo es.
    rows = [
        {"escenario": "Hoy (medido)", "corte": "política vigente"} | measured,
        {"escenario": "Parrilla Online en ambos canales", "corte": "vigente"} | baseline,
    ]
    for target in targets:
        for frame, label, fmt in (
            (uniform, "corte único", lambda r: f"> {r['corte_score']:.0f}"),
            (shifted, "parrilla desplazada", lambda r: f"+{r['desplazamiento']:.0f} puntos"),
        ):
            feasible = frame[frame["riesgo_total_pct"] <= target]
            if feasible.empty:
                rows.append({"escenario": f"{target:g}% · {label}", "corte": "no alcanzable"})
                continue
            best = feasible.iloc[0]
            rows.append(
                {"escenario": f"{target:g}% · {label}", "corte": fmt(best)}
                | {
                    k: best[k]
                    for k in ("riesgo_total_pct", "ta_score_pct", "ta_efectiva_pct", "produccion_mensual_eur")
                }
            )
    return pd.DataFrame(rows)


def scenario_grids(
    uniform: pd.DataFrame, shifted: pd.DataFrame, cutoffs: dict[str, float | None], targets: list[float]
) -> pd.DataFrame:
    """La parrilla resultante de cada escenario, grupo a grupo.

    Es el entregable operable: los escenarios se resumen como "corte > 68" o "+39
    puntos", pero quien tenga que implementarlo necesita el umbral de cada grupo. Una
    fila por grupo, una columna por escenario, más la parrilla vigente como referencia.
    """
    groups = list(cutoffs)

    def _fmt(value: float | None) -> str:
        if value == ALWAYS_REJECT:
            return "rechazo"
        return "sin corte" if value is None else f"> {value:.0f}"

    table = {"grupo": groups, "Parrilla actual": [_fmt(cutoffs[g]) for g in groups]}
    for target in targets:
        for frame, label, builder in (
            (uniform, "único", lambda row: _uniform(cutoffs, row["corte_score"])),
            (shifted, "parrilla", lambda row: _shifted(cutoffs, row["desplazamiento"])),
        ):
            feasible = frame[frame["riesgo_total_pct"] <= target]
            applied = builder(feasible.iloc[0]) if not feasible.empty else dict.fromkeys(groups)
            table[f"{target:g}% · {label}"] = [_fmt(applied[g]) for g in groups]
    return pd.DataFrame(table)


def seasonal_index(channel_data: pd.DataFrame, label: str, fallback: pd.Series | None = None) -> pd.Series | None:
    """Índice estacional multiplicativo mes-del-año, estimado **por canal**.

    Apple es extremadamente estacional, pero **no igual en los dos canales**: Online
    concentra la demanda en el lanzamiento de iPhone (septiembre ~2,0) mientras tienda
    tiene ahí uno de sus meses más flojos (~0,7) y su pico en diciembre; la correlación
    entre ambos índices es 0,39. Aplicar el de Online a tienda —que era lo único posible
    sin histórico— sobrecorregía agosto y le inflaba el peso en el total Apple.

    Se promedia el índice de cada **año natural completo**, de modo que el crecimiento de
    volumen entre años no se confunda con estacionalidad.
    """
    data = channel_data.dropna(subset=[Columns.MIS_DATE])
    per_year = []
    for _year, group in data.groupby(data[Columns.MIS_DATE].dt.year):
        monthly = group.groupby(group[Columns.MIS_DATE].dt.month)[Columns.OA_AMT].sum()
        if len(monthly) == 12:
            per_year.append(monthly / monthly.mean())
    if not per_year:
        logger.warning(
            f"[{label}] sin ningún año natural completo: no se puede estimar su estacionalidad"
            + (", se usa la del otro canal" if fallback is not None else ", no se ajusta")
        )
        return fallback
    index = pd.concat(per_year, axis=1).mean(axis=1)
    logger.info(
        f"[{label}] índice estacional sobre {len(per_year)} año(s) completo(s): "
        f"mín {index.min():.2f} (mes {index.idxmin()}), máx {index.max():.2f} (mes {index.idxmax()})"
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
        default=None,
        help="Reescala el nivel de riesgo de tienda. Por defecto se calcula del dato "
        "(morosidad real de tienda / imputada). Pasar 1.0 desactiva el anclaje.",
    )
    p.add_argument("--multiplier", type=float, default=float(DEFAULT_RISK_MULTIPLIER))
    p.add_argument(
        "--no-seasonal-adjust",
        action="store_true",
        help="Desactiva el ajuste estacional y vuelve al conteo crudo de meses (no recomendado)",
    )
    p.add_argument(
        "--no-reject-inference",
        action="store_true",
        help="Desactiva la corrección por selección (los riesgos quedan optimistas)",
    )
    p.add_argument("--reject-uplift", type=float, default=1.5, help="Coeficiente de uplift del parceling")
    p.add_argument("--reject-max-multiplier", type=float, default=3.0, help="Tope del multiplicador por tramo")
    p.add_argument("--reject-method", default="linear", choices=["linear", "power", "sigmoid"])
    p.add_argument("--output", default="output/apple_study")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cutoffs = dict(PARRILLA_ECOMMERCE)
    for override in args.cutoff:
        segment, _, value = override.partition("=")
        cutoffs[segment.strip()] = None if value.strip().lower() in {"", "none"} else float(value)
    logger.info(f"Parrilla aplicada (umbral de rechazo, se acepta por encima): {cutoffs}")

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

    all_stores = data[data[CHANNEL_COL] == CHANNEL_STORES]
    if args.no_seasonal_adjust:
        logger.warning("Ajuste estacional DESACTIVADO: el run-rate de un canal con pocos meses no es comparable")
        idx_ecom = idx_stores = None
    else:
        # El índice se estima sobre TODA la demanda del canal, lleve score o no: es un
        # perfil de volumen y tienda no tiene EFX antes de agosto.
        idx_ecom = seasonal_index(data[data[CHANNEL_COL] == CHANNEL_ECOM], "Online")
        idx_stores = seasonal_index(all_stores, "Tienda", fallback=idx_ecom)
        if idx_stores is not None:
            warn_partial_rollout(all_stores[all_stores[Columns.MIS_DATE] >= args.stores_from], idx_stores)

    data = data[data["bin"].notna()]
    eff_ecom = effective_months(args.main_from, args.main_to, idx_ecom)
    eff_stores = effective_months(args.stores_from, args.stores_to, idx_stores)
    logger.info(f"Meses efectivos: Online {eff_ecom:.2f} | tienda {eff_stores:.2f}")

    # Tienda se mide sobre su propio histórico, sin score: para un realizado no hace falta.
    # Misma ventana que Online, para que el nivel realizado de tienda sea UNA sola cifra
    # en todo el estudio (el ancla y la slide de situación actual deben coincidir).
    stores_window = all_stores[
        (all_stores[Columns.MIS_DATE] >= args.main_from) & (all_stores[Columns.MIS_DATE] < args.main_to)
    ]

    curve = estimate_risk_curve(ecom, args.multiplier)
    # El ancla va ANTES del reject inference: compara contra la morosidad realizada de
    # tienda, que es base contratados. Con la curva ya corregida por selección se estaría
    # corrigiendo dos veces y el factor saldría artificialmente bajo.
    anchor = stores_level_anchor(stores_window, _booked(stores), curve)  # siempre, para dejar constancia
    level_factor = args.stores_level_factor if args.stores_level_factor is not None else anchor["factor"]
    if not args.no_reject_inference:
        curve = apply_reject_inference(
            curve,
            ecom,
            uplift=args.reject_uplift,
            max_multiplier=args.reject_max_multiplier,
            method=args.reject_method,
        )
        curve["b2_pct_contratados"] = curve["b2_pct"]
        curve["b2_pct"] = curve["b2_pct_ri"]
    else:
        logger.warning("Reject inference DESACTIVADO: el riesgo de los tramos poco aceptados queda subestimado")
        curve[_BOOKED_COL] = curve["b2_pct"]
    actuals = channel_actuals(ecom, args.multiplier)
    stores_actuals = channel_actuals(stores_window, args.multiplier)
    grid = stores_under_grid(stores, curve, cutoffs, level_factor)
    ladder, ladder_grid = scenario_ladders(ecom, stores, curve, cutoffs, eff_ecom, eff_stores, level_factor)
    baseline = policy_kpis(ecom, stores, curve, cutoffs, eff_ecom, eff_stores, level_factor)
    measured = measured_status_quo(ecom, stores_window, eff_ecom, args.multiplier)
    picks = pick_targets(ladder, ladder_grid, baseline, measured, args.targets)
    grids = scenario_grids(ladder, ladder_grid, cutoffs, args.targets)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Los periodos se persisten para que el deck los rotule desde el dato: escritos a mano
    # en las slides se desincronizan del run en cuanto alguien cambia una ventana.
    seasonal_years = sorted(
        {
            int(y)
            for y, g in data.groupby(data[Columns.MIS_DATE].dt.year)
            if g.groupby(g[Columns.MIS_DATE].dt.month).ngroups == 12
        }
    )
    json.dump(
        {
            "online_desde": args.main_from,
            "online_hasta": args.main_to,
            "tienda_desde": args.stores_from,
            "tienda_hasta": args.stores_to,
            "estacionalidad_anios": seasonal_years,
            "reject_inference": not args.no_reject_inference,
            "factor_nivel_tienda": round(level_factor, 4),
            "riesgo_tienda_realizado_pct": round(anchor["realizado_pct"], 4),
            "riesgo_tienda_imputado_pct": round(anchor["imputado_pct"], 4),
        },
        (out / "periodos.json").open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    for name, frame in (
        ("curva_riesgo_online", curve.reset_index()),
        (
            "indice_estacional",
            (
                pd.DataFrame()
                if idx_ecom is None
                else pd.DataFrame(
                    {"mes": idx_ecom.index, "indice": idx_ecom.values, "indice_tienda": idx_stores.values}
                )
            ),
        ),
        ("p1_online_actual", actuals),
        ("p1b_tienda_actual", stores_actuals),
        ("p2_tienda_parrilla_online", grid),
        ("p3_escalera_corte_unico", ladder),
        ("p3_escalera_parrilla_desplazada", ladder_grid),
        ("p3_escenarios_objetivo", picks),
        ("p4_parrillas_escenarios", grids),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
    logger.info(f"Resultados en {out}/ (5 ficheros)")

    print("\n=== 1) APPLE ONLINE — actual, medido ===")
    print(actuals.round(2).to_string(index=False))
    print("\n=== 1b) APPLE STORES — actual, medido sobre su propio histórico ===")
    print(stores_actuals.round(2).to_string(index=False))
    print("\n=== 2) APPLE STORES — parrilla de Online (riesgo IMPUTADO) ===")
    print(grid.round(2).to_string(index=False))
    print("\n=== 3) ESCENARIOS — riesgo global Apple ===")
    show = picks.rename(
        columns={
            "riesgo_total_pct": "riesgo %",
            "ta_score_pct": "TA score %",
            "ta_efectiva_pct": "TA efectiva %",
            "produccion_mensual_eur": "produccion M€/mes",
        }
    )
    show["produccion M€/mes"] = (show["produccion M€/mes"] / 1e6).round(1)
    print(show.round(2).to_string(index=False))
    print("\n=== 4) PARRILLAS DE CADA ESCENARIO (umbral de rechazo por grupo) ===")
    print(grids.to_string(index=False))
    print(
        "\nTA score = demanda que supera el corte.  TA efectiva = producción / demanda, "
        "la misma base que el punto 1 (no son comparables entre sí)."
    )
    print("\nRiesgo de tienda imputado desde Online: NO es una medición (ver docstring).")
    print(
        "Sin reject inference: riesgos optimistas."
        if args.no_reject_inference
        else "Con reject inference (parceling del pipeline). Números definitivos vía run_batch.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
