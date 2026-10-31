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

import pandas as pd
from loguru import logger

from src.core.constants import DEFAULT_RISK_MULTIPLIER, Columns
from src.studies.apple.study import (
    _BOOKED_COL,
    CHANNEL_COL,
    CHANNEL_ECOM,
    CHANNEL_STORES,
    DEMAND_STATES,
    LIGHT_COL,
    PARRILLA_ECOMMERCE,
    SEGMENT_COL,
    _booked,
    apply_reject_inference,
    booked_without_bin,
    channel_actuals,
    channel_blend,
    demand_breakdown,
    effective_months,
    estimate_risk_curve,
    grid_fidelity,
    load_study_data,
    measured_status_quo,
    pick_targets,
    policy_kpis,
    same_base_status_quo,
    scenario_grids,
    scenario_ladders,
    seasonal_index,
    stores_level_anchor,
    stores_swap_breakdown,
    stores_under_grid,
    take_up_by_decision,
    warn_partial_rollout,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default="data/demanda_apple.sas7bdat")
    p.add_argument("--main-from", default="2025-03-01", help="Inicio ventana Online (default: 12 meses)")
    p.add_argument("--main-to", default="2026-03-01")
    p.add_argument(
        "--stores-from",
        default="2026-08-01",
        help="Inicio ventana tienda. Default agosto-2026: primer mes con score EFX en tienda",
    )
    p.add_argument("--stores-to", default="2026-10-01", help="Fin exclusivo. Default: hasta septiembre-2026 incluido")
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
    all_ecom = data[data[CHANNEL_COL] == CHANNEL_ECOM]
    if args.no_seasonal_adjust:
        logger.warning("Ajuste estacional DESACTIVADO: el run-rate de un canal con pocos meses no es comparable")
        idx_ecom = idx_stores = None
    else:
        # El índice se estima sobre TODA la demanda del canal, lleve score o no: es un
        # perfil de volumen y tienda no tiene EFX antes de agosto.
        # Sobre la población extraída de forma homogénea en toda la historia: las rechazadas
        # en estudio ligero solo vienen desde jul-2025 (tienda) / may-2026 (Online) y, dentro,
        # inflarían la segunda mitad de 2025 como si fuera estacionalidad.
        consistent = data[~data[LIGHT_COL]]
        idx_ecom = seasonal_index(consistent[consistent[CHANNEL_COL] == CHANNEL_ECOM], "Online")
        idx_stores = seasonal_index(consistent[consistent[CHANNEL_COL] == CHANNEL_STORES], "Tienda", fallback=idx_ecom)
        if idx_stores is not None:
            warn_partial_rollout(all_stores[all_stores[Columns.MIS_DATE] >= args.stores_from], idx_stores)
    light_from = {
        name: (m.min().strftime("%Y-%m-01") if not (m := frame.loc[frame[LIGHT_COL], Columns.MIS_DATE]).empty else None)
        for name, frame in (("tienda", all_stores), ("online", all_ecom))
    }
    logger.info(f"Estudio ligero en el extracto desde: {light_from}")

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
    breakdown = pd.concat(
        [
            demand_breakdown(ecom, "online"),
            demand_breakdown(stores_window, "tienda_ventana"),
            demand_breakdown(stores, "tienda_score"),
        ],
        ignore_index=True,
    )
    grid = stores_under_grid(stores, curve, cutoffs, level_factor)
    swap = stores_swap_breakdown(stores, curve, cutoffs, level_factor)
    fidelity = grid_fidelity(ecom, cutoffs)
    # Cifras que el deck rotula y que antes iban escritas a mano: el riesgo de Online sobre
    # todo su histórico maduro (robustez de la ventana) y cuánto pesa tienda en Apple según
    # cómo se anualice su agosto (calendario / su índice / el índice de Online).
    ecom_history = all_ecom[all_ecom[Columns.MIS_DATE] < args.main_to]
    history_from = ecom_history[Columns.MIS_DATE].min().strftime("%Y-%m-01")
    history_total = channel_actuals(ecom_history, args.multiplier).set_index("segmento").loc["TOTAL"]
    online_rate = ecom[Columns.OA_AMT].sum() / eff_ecom
    stores_demand = stores[Columns.OA_AMT].sum()

    def stores_weight(months: float) -> float:
        return 100 * (stores_demand / months) / (stores_demand / months + online_rate)

    weights = {
        "peso_tienda_calendario_pct": stores_weight(effective_months(args.stores_from, args.stores_to, None)),
        "peso_tienda_propio_pct": stores_weight(eff_stores),
        "peso_tienda_indice_online_pct": stores_weight(effective_months(args.stores_from, args.stores_to, idx_ecom)),
    }
    logger.info(
        f"Riesgo Online sobre todo el histórico maduro desde {history_from}: {history_total['riesgo_real_pct']:.2f}% | "
        + " | ".join(f"{k.removeprefix('peso_tienda_').removesuffix('_pct')} {v:.1f}%" for k, v in weights.items())
    )
    logger.info(
        "Fidelidad de la parrilla vigente en Online (% demanda): aprobado real "
        f"{fidelity['aprobado_real_pct']:.1f} | parrilla {fidelity['parrilla_pct']:.1f} | "
        f"swap-in {fidelity['swap_in_pct']:.1f} | swap-out {fidelity['swap_out_pct']:.1f}"
    )
    ladder, ladder_grid = scenario_ladders(ecom, stores, curve, cutoffs, eff_ecom, eff_stores, level_factor)
    baseline = policy_kpis(ecom, stores, curve, cutoffs, eff_ecom, eff_stores, level_factor)
    blend = channel_blend(ecom, stores, curve, cutoffs, eff_ecom, eff_stores, level_factor)
    take_up = {"online": take_up_by_decision(ecom), "tienda": take_up_by_decision(stores)}
    measured = measured_status_quo(
        ecom, stores_window, eff_ecom, effective_months(args.main_from, args.main_to, idx_stores), args.multiplier
    )
    same_base = same_base_status_quo(ecom, stores, curve, eff_ecom, eff_stores, level_factor)
    # Cuánto más demanda trae tienda en los meses con score que en la ventana de "hoy":
    # es la parte de la diferencia entre ambas filas que es volumen y no política.
    stores_window_months = effective_months(args.main_from, args.main_to, idx_stores)
    # Sobre la población homogénea: el estudio ligero solo está en parte de la ventana y
    # sumaría volumen que no es crecimiento.
    stores_volume_vs_window_pct = 100 * (
        (stores.loc[~stores[LIGHT_COL], Columns.OA_AMT].sum() / eff_stores)
        / (stores_window.loc[~stores_window[LIGHT_COL], Columns.OA_AMT].sum() / stores_window_months)
        - 1
    )
    logger.info(
        f"Política de hoy sobre la base de los escenarios: TA {same_base['ta_efectiva_pct']:.2f}% | "
        f"{same_base['produccion_mensual_eur'] / 1e6:.2f} M€/mes | riesgo {same_base['riesgo_total_pct']:.2f}% "
        f"(demanda de tienda {stores_volume_vs_window_pct:+.1f}% frente a la ventana)"
    )
    # Base de reject inference: los escenarios llevan la curva corregida por selección, un
    # margen conservador que recae también sobre la cartera ya contratada de Online; las
    # filas de "hoy" van a realizado. Se persiste el efecto de la parrilla sobre el riesgo
    # en la MISMA base que "hoy" para que el deck no venda como efecto de la parrilla lo
    # que es diferencia de base.
    booked_curve = curve.assign(b2_pct=curve[_BOOKED_COL])
    grid_on_today_basis = policy_kpis(ecom, stores, booked_curve, cutoffs, eff_ecom, eff_stores, level_factor)
    online_under_grid = policy_kpis(ecom, ecom.iloc[0:0], curve, cutoffs, eff_ecom, 1.0, 1.0)
    ri_margin_online_pp = online_under_grid["riesgo_total_pct"] - float(
        actuals.set_index("segmento").loc["TOTAL", "riesgo_real_pct"]
    )
    logger.info(
        f"Base RI: Online bajo la parrilla vigente {online_under_grid['riesgo_total_pct']:.2f}% con curva RI "
        f"(+{ri_margin_online_pp:.2f} pp sobre el realizado) | parrilla vigente en base 'hoy' "
        f"{grid_on_today_basis['riesgo_total_pct']:.2f}% frente a {baseline['riesgo_total_pct']:.2f}% con RI"
    )
    unscored_booked = booked_without_bin(ecom, args.multiplier)
    if unscored_booked["n"]:
        logger.warning(
            f"Online: {unscored_booked['n']} contratos sin tramo EFX (aprobados sin llamada a Equifax), "
            f"{unscored_booked['produccion_pct']:.2f}% de la producción con {unscored_booked['riesgo_pct']:.1f}% de "
            f"riesgo: cuentan en el realizado ({measured['online_riesgo_pct']:.2f}%) pero no en la base de los "
            f"escenarios ({same_base['online_riesgo_pct']:.2f}%)"
        )
    picks = pick_targets(ladder, ladder_grid, baseline, measured, same_base, args.targets)
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
            **{f"online_{k}": round(v, 2) for k, v in fidelity.items()},
            "online_historico_desde": history_from,
            "online_historico_meses": round(effective_months(history_from, args.main_to, None)),
            "riesgo_online_historico_pct": round(float(history_total["riesgo_real_pct"]), 4),
            **{k: round(v, 2) for k, v in weights.items()},
            "demanda_tienda_vs_ventana_pct": round(stores_volume_vs_window_pct, 2),
            "estudio_ligero_tienda_desde": light_from["tienda"],
            "estudio_ligero_online_desde": light_from["online"],
            "estudio_ligero_meses_ventana_tienda": int(
                stores_window.loc[stores_window[LIGHT_COL], Columns.MIS_DATE].dt.to_period("M").nunique()
            ),
            **{
                f"take_up_{ch}_{dec}_pct": round(100 * r, 2)
                for ch, rr in take_up.items()
                for dec, r in rr.items()
                if dec != "ko"
            },
            "riesgo_parrilla_base_hoy_pct": round(grid_on_today_basis["riesgo_total_pct"], 4),
            "ri_margen_online_pp": round(ri_margin_online_pp, 4),
            "online_contratados_sin_efx_n": unscored_booked["n"],
            "online_contratados_sin_efx_produccion_pct": round(unscored_booked["produccion_pct"], 3),
            "online_contratados_sin_efx_riesgo_pct": round(unscored_booked["riesgo_pct"], 2),
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
        ("p0_demanda_por_estado", breakdown),
        ("p2b_tienda_swap", swap),
        ("p3b_mezcla_canales", blend),
    ):
        frame.to_csv(out / f"{name}.csv", index=False)
    logger.info(f"Resultados en {out}/")

    print("\n=== 2b) TIENDA: QUÉ HACE LA PARRILLA DE ONLINE CON CADA EURO ===")
    print(swap.round(2).to_string(index=False))
    print("\n=== 3b) MEZCLA DE CANALES (run-rate mensual, ponderada por exposición) ===")
    print(blend.round(2).to_string(index=False))
    print("\n=== 0) DÓNDE ACABA CADA EURO SOLICITADO (% de la demanda) ===")
    print(breakdown.pivot(index="estado", columns="canal", values="pct_demanda").reindex(list(DEMAND_STATES)).round(1))

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
            "ta_score_pct": "aprobación sistema %",
            "ta_efectiva_pct": "tasa contratación %",
            "produccion_mensual_eur": "produccion M€/mes",
        }
    )
    show["produccion M€/mes"] = (show["produccion M€/mes"] / 1e6).round(1)
    print(show.round(2).to_string(index=False))
    print("\n=== 4) PARRILLAS DE CADA ESCENARIO (umbral de rechazo por grupo) ===")
    print(grids.to_string(index=False))
    print(
        "\nAprobación del sistema = (OK + RV) / demanda; bajo una parrilla, demanda que supera el corte sin otro "
        "rechazo.  Tasa de contratación = contratado / demanda = aprobación × take-up, "
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
