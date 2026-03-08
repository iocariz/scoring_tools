"""
Marginal Risk (MR) pipeline for credit risk analysis and optimization.

This module orchestrates the marginal risk analysis workflow:
- Calculating risk metrics from optimal cut points
- Processing MR periods with demand and production data
- Generating risk production summary tables
- Running optimization pipelines for portfolio management

Key functions:
- calculate_metrics_from_cuts: Apply optimal cuts to aggregated data
- process_mr_period: Execute full MR analysis for a time period
"""

import traceback
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from src import styles
from src.config import OutputPaths
from src.constants import DEFAULT_RISK_MULTIPLIER, PSI_UNSTABLE_THRESHOLD, StatusName
from src.inference_optimized import run_optimization_pipeline
from src.models import calculate_B2
from src.preprocess_improved import filter_by_date
from src.stability import compare_main_vs_mr
from src.utils import calculate_b2_ever_h6, calculate_todu_30ever_from_b2, extrapolate_h3_to_h6

if TYPE_CHECKING:
    from src.config import PreprocessingSettings


def calculate_metrics_from_cuts(
    data_summary_desagregado: pd.DataFrame,
    optimal_solution_df: pd.DataFrame | None,
    variables: list[str],
    inv_vars: list[str] | None = None,
    mask: np.ndarray | None = None,
    grid: object | None = None,
    multiplier_h3: float | None = None,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    total_demand: float = 0.0,
) -> pd.DataFrame | None:
    """
    Generates the Risk Production Summary Table by applying optimal cuts to aggregated data.

    For N>2 variables (when mask/grid are provided), uses classify_by_mask instead of
    the 2-variable cut_map approach.
    """
    try:
        var0_col = variables[0]
        var1_col = variables[1]

        # Verify we have the optimal solution
        if optimal_solution_df is None or optimal_solution_df.empty:
            logger.warning("optimal_solution_df is missing or empty. Cannot calculate summary table.")
            return None

        df = data_summary_desagregado.copy()

        # Determine passes_cut via mask (N-d) or cut_map (2-var)
        if mask is not None and grid is not None:
            from src.optimization_utils import classify_by_mask

            df["passes_cut"] = classify_by_mask(df, mask, grid)
        else:
            opt_sol_row = optimal_solution_df.iloc[0]

            # Get unique bins from data
            bins = sorted(data_summary_desagregado[var0_col].unique())
            cut_map = {}

            for bin_val in bins:
                if bin_val in optimal_solution_df.columns:
                    cut_map[bin_val] = opt_sol_row[bin_val]
                elif str(bin_val) in optimal_solution_df.columns:
                    cut_map[bin_val] = opt_sol_row[str(bin_val)]
                elif str(float(bin_val)) in optimal_solution_df.columns:
                    cut_map[bin_val] = opt_sol_row[str(float(bin_val))]
                else:
                    logger.warning(
                        f"Warning: Bin {bin_val} not found in optimal solution columns. Defaulting to strict rejection."
                    )
                    cut_map[bin_val] = np.inf if (inv_vars and var1_col in inv_vars) else -np.inf

            logger.info(f"Optimal Cuts: {cut_map}")

            df["cut_limit"] = df[var0_col].map(cut_map)

            if inv_vars and var1_col in inv_vars:
                df["passes_cut"] = df[var1_col] >= df["cut_limit"]
            else:
                df["passes_cut"] = df[var1_col] <= df["cut_limit"]

        summary_data = []

        # Check if h3 columns are available
        has_h3 = multiplier_h3 is not None and "todu_30ever_h3_boo" in df.columns

        # Helper to calc metrics from a filtered subset
        def calc_metrics(subset, suffix):
            prod = subset[f"oa_amt_h0{suffix}"].sum()
            risk_num = subset[f"todu_30ever_h6{suffix}"].sum()
            risk_den = subset[f"todu_amt_pile_h6{suffix}"].sum()
            b2_ever_raw = calculate_b2_ever_h6(risk_num, risk_den, multiplier=multiplier, as_percentage=True)
            b2_ever = float(b2_ever_raw) if pd.notna(b2_ever_raw) else None
            # H3 metrics
            h3_rn, h3_rd, h3_risk = 0.0, 0.0, None
            if has_h3:
                h3_col_num = f"todu_30ever_h3{suffix}"
                h3_col_den = f"todu_amt_pile_h3{suffix}"
                if h3_col_num in subset.columns and h3_col_den in subset.columns:
                    h3_rn = subset[h3_col_num].sum()
                    h3_rd = subset[h3_col_den].sum()
                    h3_raw = calculate_b2_ever_h6(h3_rn, h3_rd, multiplier=multiplier_h3, as_percentage=True)
                    h3_risk = float(h3_raw) if pd.notna(h3_raw) else None
            return prod, b2_ever, risk_num, risk_den, h3_risk, h3_rn, h3_rd

        # Actual (All Booked)
        actual_prod, actual_risk, actual_rn, actual_rd, actual_h3, actual_h3_rn, actual_h3_rd = calc_metrics(df, "_boo")

        # Total demand (through the door) = booked + all rejected + canceled
        if total_demand > 0:
            _total_demand = total_demand
        else:
            # Backward compat fallback: booked + repesca
            total_rep_prod = df["oa_amt_h0_rep"].sum() if "oa_amt_h0_rep" in df.columns else 0.0
            _total_demand = actual_prod + total_rep_prod

        row_actual = {
            "Metric": "Actual",
            "Risk (%)": actual_risk,
            "Production (€)": actual_prod,
            "Production (%)": 1.0,
            "todu_30ever_h6": actual_rn,
            "todu_amt_pile_h6": actual_rd,
            "Rejection Rate (%)": (1 - actual_prod / _total_demand) * 100 if _total_demand > 0 else 0.0,
            "Total Demand (€)": _total_demand,
        }
        if has_h3:
            row_actual["Risk H3 (%)"] = actual_h3
            row_actual["todu_30ever_h3"] = actual_h3_rn
            row_actual["todu_amt_pile_h3"] = actual_h3_rd
        summary_data.append(row_actual)

        # Swap-in (Repesca that passes)
        swap_in_df = df[df["passes_cut"]]
        si_prod, si_risk, si_rn, si_rd, si_h3, si_h3_rn, si_h3_rd = calc_metrics(swap_in_df, "_rep")
        row_si = {
            "Metric": "Swap-in",
            "Risk (%)": si_risk,
            "Production (€)": si_prod,
            "Production (%)": si_prod / actual_prod if actual_prod else 0,
            "todu_30ever_h6": si_rn,
            "todu_amt_pile_h6": si_rd,
            "Rejection Rate (%)": None,
        }
        if has_h3:
            row_si["Risk H3 (%)"] = si_h3
            row_si["todu_30ever_h3"] = si_h3_rn
            row_si["todu_amt_pile_h3"] = si_h3_rd
        summary_data.append(row_si)

        # Swap-out (Booked that fails)
        swap_out_df = df[~df["passes_cut"]]
        so_prod, so_risk, so_rn, so_rd, so_h3, so_h3_rn, so_h3_rd = calc_metrics(swap_out_df, "_boo")
        row_so = {
            "Metric": "Swap-out",
            "Risk (%)": so_risk,
            "Production (€)": so_prod,
            "Production (%)": so_prod / actual_prod if actual_prod else 0,
            "todu_30ever_h6": so_rn,
            "todu_amt_pile_h6": so_rd,
            "Rejection Rate (%)": None,
        }
        if has_h3:
            row_so["Risk H3 (%)"] = so_h3
            row_so["todu_30ever_h3"] = so_h3_rn
            row_so["todu_amt_pile_h3"] = so_h3_rd
        summary_data.append(row_so)

        # Optimum
        opt_prod = (actual_prod - so_prod) + si_prod
        opt_rn = (actual_rn - so_rn) + si_rn
        opt_rd = (actual_rd - so_rd) + si_rd
        opt_risk_raw = calculate_b2_ever_h6(opt_rn, opt_rd, multiplier=multiplier, as_percentage=True)
        opt_risk = float(opt_risk_raw) if pd.notna(opt_risk_raw) else None

        row_opt = {
            "Metric": "Optimum selected",
            "Risk (%)": opt_risk,
            "Production (€)": opt_prod,
            "Production (%)": opt_prod / actual_prod if actual_prod else 0,
            "todu_30ever_h6": opt_rn,
            "todu_amt_pile_h6": opt_rd,
            "Rejection Rate (%)": (1 - opt_prod / _total_demand) * 100 if _total_demand > 0 else 0.0,
        }
        if has_h3:
            opt_h3_rn = (actual_h3_rn - so_h3_rn) + si_h3_rn
            opt_h3_rd = (actual_h3_rd - so_h3_rd) + si_h3_rd
            opt_h3_risk_raw = calculate_b2_ever_h6(opt_h3_rn, opt_h3_rd, multiplier=multiplier_h3, as_percentage=True)
            opt_h3_risk = float(opt_h3_risk_raw) if pd.notna(opt_h3_risk_raw) else None
            row_opt["Risk H3 (%)"] = opt_h3_risk
            row_opt["todu_30ever_h3"] = opt_h3_rn
            row_opt["todu_amt_pile_h3"] = opt_h3_rd
        summary_data.append(row_opt)

        return pd.DataFrame(summary_data)

    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Error calculating metrics from cuts: {e}")
        logger.error(traceback.format_exc())
        return None


def _mr_outcomes_available(data_demand_mr: pd.DataFrame) -> bool:
    """Check if MR-period outcome columns exist and have non-null data."""
    required = ["todu_30ever_h6", "todu_amt_pile_h6"]
    return all(col in data_demand_mr.columns and data_demand_mr[col].notna().any() for col in required)


def _compute_hybrid_mr_risk(
    data_booked: pd.DataFrame,
    data_demand_mr: pd.DataFrame,
    merge_keys: list[str],
    min_obs: int,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    multiplier_h3: float | None = None,
    mr_extrapolation_method: str = "linear",
    mr_extrapolation_curvature: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-bin ``b2_ever_h6_tmp`` using MR outcomes when sufficient, else main-period.

    Returns
    -------
    merge_df : DataFrame
        ``merge_keys`` + ``b2_ever_h6_tmp`` ready for merging into ``data_demand_mr``.
    comparison_df : DataFrame
        Per-bin diagnostic table with columns:
        ``merge_keys``, ``b2_main``, ``b2_mr``, ``n_obs_main``, ``n_obs_mr``,
        ``b2_ever_h6_tmp``, ``risk_source``, ``b2_delta``, ``b2_delta_pct``, ``mr_production``.
    """
    # --- Main-period aggregation (existing logic) ---
    main_agg = data_booked.groupby(merge_keys)[["todu_30ever_h6", "todu_amt_pile_h6"]].sum().reset_index()
    main_agg["b2_main"] = calculate_b2_ever_h6(
        main_agg["todu_30ever_h6"], main_agg["todu_amt_pile_h6"], multiplier=multiplier
    ).fillna(0.0)
    n_obs_main = data_booked.groupby(merge_keys).size().reset_index(name="n_obs_main")
    main_agg = main_agg.merge(n_obs_main, on=merge_keys, how="left")
    main_agg = main_agg[merge_keys + ["b2_main", "n_obs_main"]]

    # --- MR-period aggregation ---
    mr_booked = data_demand_mr[data_demand_mr["status_name"] == StatusName.BOOKED.value]
    mr_h6_valid = mr_booked[mr_booked["todu_30ever_h6"].notna() & mr_booked["todu_amt_pile_h6"].notna()]
    mr_agg = mr_h6_valid.groupby(merge_keys)[["todu_30ever_h6", "todu_amt_pile_h6"]].sum().reset_index()
    mr_agg["b2_mr"] = calculate_b2_ever_h6(
        mr_agg["todu_30ever_h6"], mr_agg["todu_amt_pile_h6"], multiplier=multiplier
    ).fillna(0.0)
    n_obs_mr = mr_h6_valid.groupby(merge_keys).size().reset_index(name="n_obs_mr")
    mr_agg = mr_agg.merge(n_obs_mr, on=merge_keys, how="left")

    # MR production per bin
    if "oa_amt_h0" in mr_booked.columns:
        mr_prod = mr_booked.groupby(merge_keys)["oa_amt_h0"].sum().reset_index()
        mr_prod = mr_prod.rename(columns={"oa_amt_h0": "mr_production"})
        mr_agg = mr_agg.merge(mr_prod, on=merge_keys, how="outer")
    else:
        mr_agg["mr_production"] = 0.0

    mr_agg["mr_production"] = mr_agg["mr_production"].fillna(0.0)
    mr_agg = mr_agg[merge_keys + ["b2_mr", "n_obs_mr", "mr_production"]]

    # --- Outer join ---
    combined = main_agg.merge(mr_agg, on=merge_keys, how="outer")

    # --- H3 aggregation (needed before risk source selection for extrapolation) ---
    has_h3 = multiplier_h3 is not None and "todu_30ever_h3" in data_booked.columns
    if has_h3:
        mr_booked_h3 = data_demand_mr[data_demand_mr["status_name"] == StatusName.BOOKED.value]
        h3_cols = ["todu_30ever_h3", "todu_amt_pile_h3"]

        # Main-period H3
        if all(c in data_booked.columns for c in h3_cols):
            main_h3_agg = data_booked.groupby(merge_keys)[h3_cols].sum().reset_index()
            main_h3_agg["b2_main_h3"] = calculate_b2_ever_h6(
                main_h3_agg["todu_30ever_h3"], main_h3_agg["todu_amt_pile_h3"], multiplier=multiplier_h3
            ).fillna(0.0)
            combined = combined.merge(main_h3_agg[merge_keys + ["b2_main_h3"]], on=merge_keys, how="left")
        else:
            combined["b2_main_h3"] = np.nan

        # MR-period H3 — only accounts with mature H3 (first 3 mis_dates in 6-month MR window)
        if all(c in mr_booked_h3.columns for c in h3_cols):
            mr_h3_valid = mr_booked_h3[
                mr_booked_h3["todu_30ever_h3"].notna() & mr_booked_h3["todu_amt_pile_h3"].notna()
            ]
            mr_h3_agg = mr_h3_valid.groupby(merge_keys)[h3_cols].sum().reset_index()
            n_obs_mr_h3 = mr_h3_valid.groupby(merge_keys).size().reset_index(name="n_obs_mr_h3")
            mr_h3_agg = mr_h3_agg.merge(n_obs_mr_h3, on=merge_keys, how="left")
            mr_h3_agg["b2_mr_h3"] = calculate_b2_ever_h6(
                mr_h3_agg["todu_30ever_h3"], mr_h3_agg["todu_amt_pile_h3"], multiplier=multiplier_h3
            ).fillna(0.0)
            combined = combined.merge(mr_h3_agg[merge_keys + ["b2_mr_h3", "n_obs_mr_h3"]], on=merge_keys, how="left")
        else:
            combined["b2_mr_h3"] = np.nan
            combined["n_obs_mr_h3"] = 0

    # --- Auto-calibrate extrapolation curvature ---
    if mr_extrapolation_method == "auto" and has_h3:
        from src.utils import fit_h3_extrapolation_curve

        valid_fit = combined["b2_main_h3"].notna() & combined["b2_main"].notna()
        if valid_fit.sum() >= 4:
            method_auto, curvature_auto, fit_diag = fit_h3_extrapolation_curve(
                combined.loc[valid_fit, "b2_main_h3"].values,
                combined.loc[valid_fit, "b2_main"].values,
                weights=combined.loc[valid_fit, "n_obs_main"].values,
            )
            mr_extrapolation_method = method_auto
            mr_extrapolation_curvature = curvature_auto
            logger.info(
                f"Auto-calibrated H3→H6 extrapolation: method={method_auto}, "
                f"curvature={curvature_auto:.3f} (alpha={fit_diag['alpha']:.3f}, "
                f"SE={fit_diag['se']:.3f}, R²={fit_diag['r_squared']:.3f}, "
                f"n_bins={fit_diag['n_bins']})"
            )
        else:
            mr_extrapolation_method = "linear"
            mr_extrapolation_curvature = 1.0
            logger.warning("Auto-calibration: insufficient bins for fitting, falling back to linear.")
    elif mr_extrapolation_method == "auto":
        mr_extrapolation_method = "linear"
        mr_extrapolation_curvature = 1.0
        logger.warning("Auto-calibration requires H3 data (multiplier_h3). Falling back to linear.")

    # --- Choose source per bin ---
    use_mr = combined["n_obs_mr"].fillna(0) >= min_obs

    # Bins only in MR (no main data) and below threshold: leave NaN for model fallback
    only_mr_sparse = combined["b2_main"].isna() & ~use_mr

    if has_h3:
        # H3 ratio: main-period H6/H3 scaling factor
        # Require meaningful denominator (>1% risk) and sufficient observations
        ratio_valid = (
            (combined["b2_main_h3"].fillna(0).abs() > 0.01)
            & (combined["n_obs_main"].fillna(0) >= min_obs)
        )
        h6_h3_ratio_raw = np.where(ratio_valid, combined["b2_main"] / combined["b2_main_h3"], np.nan)
        # Clip extreme ratios to guard against noisy small-sample bins
        h6_h3_ratio = np.clip(h6_h3_ratio_raw, 0.5, 5.0)

        # H3 extrapolation: observed MR H3 × main-period ratio
        has_h3_obs = combined["b2_mr_h3"].notna() & (combined["n_obs_mr_h3"].fillna(0) >= min_obs)
        can_extrapolate = ~use_mr & ratio_valid & has_h3_obs
        h6_from_h3 = extrapolate_h3_to_h6(
            combined["b2_mr_h3"], h6_h3_ratio,
            method=mr_extrapolation_method, curvature=mr_extrapolation_curvature,
        )

        # Priority: h3_extrapolated > mr_observed > main_imputed > model_fallback
        conditions = [only_mr_sparse, use_mr, can_extrapolate]
        risk_choices = [np.nan, combined["b2_mr"], h6_from_h3]
        source_choices = ["model_fallback", "mr_observed", "h3_extrapolated"]

        combined["b2_ever_h6_tmp"] = np.select(conditions, risk_choices, default=combined["b2_main"])
        combined["risk_source"] = np.select(conditions, source_choices, default="main_imputed")

        combined["h6_h3_ratio"] = h6_h3_ratio
    else:
        # Original logic (no H3 columns)
        conditions = [only_mr_sparse, use_mr]
        risk_choices = [np.nan, combined["b2_mr"]]
        source_choices = ["model_fallback", "mr_observed"]

        combined["b2_ever_h6_tmp"] = np.select(conditions, risk_choices, default=combined["b2_main"])
        combined["risk_source"] = np.select(conditions, source_choices, default="main_imputed")

    # --- Comparison diagnostics ---
    combined["b2_delta"] = combined["b2_mr"] - combined["b2_main"]
    combined["b2_delta_pct"] = np.where(
        combined["b2_main"].abs() > 1e-9,
        combined["b2_delta"] / combined["b2_main"] * 100,
        np.nan,
    )

    comparison_cols = merge_keys + [
        "b2_main",
        "b2_mr",
        "n_obs_main",
        "n_obs_mr",
        "b2_ever_h6_tmp",
        "risk_source",
        "b2_delta",
        "b2_delta_pct",
        "mr_production",
    ]

    # --- H3 comparison diagnostics ---
    if has_h3:
        combined["b2_delta_h3"] = combined["b2_mr_h3"] - combined["b2_main_h3"]
        combined["b2_delta_pct_h3"] = np.where(
            combined["b2_main_h3"].abs() > 1e-9,
            combined["b2_delta_h3"] / combined["b2_main_h3"] * 100,
            np.nan,
        )
        comparison_cols += ["b2_main_h3", "b2_mr_h3", "n_obs_mr_h3", "b2_delta_h3", "b2_delta_pct_h3", "h6_h3_ratio"]

    comparison_df = combined[comparison_cols].copy()

    # Record resolved extrapolation settings (useful when auto was used)
    comparison_df["fitted_method"] = mr_extrapolation_method
    comparison_df["fitted_curvature"] = mr_extrapolation_curvature

    merge_df = combined[merge_keys + ["b2_ever_h6_tmp"]].copy()
    return merge_df, comparison_df


def process_mr_period(
    data_clean: pd.DataFrame,
    data_booked: pd.DataFrame,
    settings: "PreprocessingSettings",
    risk_inference: dict[str, Any],
    reg_todu_amt_pile: Any,
    stress_factor: float,
    tasa_fin: float,
    annual_coef: float,
    optimal_solution_df: pd.DataFrame | None = None,
    file_suffix: str = "",
    output: OutputPaths | None = None,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> None:
    """
    Process the MR period data: filtering, inference, aggregation, visualization, and summary table.
    """
    if output is None:
        output = OutputPaths()

    logger.info(f"Processing MR period data (suffix: '{file_suffix}')...")

    if not (settings.date_ini_book_obs_mr and settings.date_fin_book_obs_mr):
        logger.warning("MR dates not configured. Skipping MR period processing.")
        return

    try:
        indicators_mr = ["acct_booked_h0", "oa_amt", "oa_amt_h0"]
        if settings.use_mr_outcomes:
            indicators_mr += ["todu_30ever_h6", "todu_amt_pile_h6"]
        # Include h3 columns for complementary monitoring when configured
        if "todu_30ever_h3" in settings.indicators:
            indicators_mr += ["todu_30ever_h3", "todu_amt_pile_h3"]
        # Ensure merge keys (variables) are included
        merge_keys = settings.variables
        mr_cols = settings.keep_vars + indicators_mr + merge_keys

        # Create data_demand_mr (filter by date and select columns)
        data_mr_period = filter_by_date(
            data_clean, "mis_date", settings.date_ini_book_obs_mr, settings.date_fin_book_obs_mr
        )

        available_mr_cols = [c for c in mr_cols if c in data_mr_period.columns]
        available_mr_cols = list(dict.fromkeys(available_mr_cols))

        data_demand_mr = data_mr_period[available_mr_cols].copy()

        # --- Calculate b2_ever_h6_tmp ---
        required_agg_cols = merge_keys + ["todu_30ever_h6", "todu_amt_pile_h6"]

        if settings.use_mr_outcomes and _mr_outcomes_available(data_demand_mr):
            # Hybrid mode: use MR observed risk where sufficient, else main-period
            logger.info(
                f"Hybrid MR risk: using MR outcomes where n_obs >= {settings.mr_min_obs_per_bin}, "
                f"falling back to main-period for sparse bins."
            )
            merge_df, comparison_df = _compute_hybrid_mr_risk(
                data_booked,
                data_demand_mr,
                merge_keys,
                settings.mr_min_obs_per_bin,
                multiplier=settings.multiplier,
                multiplier_h3=settings.multiplier_h3,
                mr_extrapolation_method=settings.mr_extrapolation_method,
                mr_extrapolation_curvature=settings.mr_extrapolation_curvature,
            )

            # Save comparison CSV
            comp_path = output.mr_risk_comparison_csv(file_suffix)
            comparison_df.to_csv(comp_path, index=False)
            logger.info(f"MR risk comparison saved to {comp_path}")

            # Log bins with large deviations
            has_both = comparison_df["b2_delta_pct"].notna()
            large_dev = comparison_df.loc[has_both & (comparison_df["b2_delta_pct"].abs() > 20)]
            if not large_dev.empty:
                logger.warning(f"RISK DRIFT: {len(large_dev)} bins show >20% deviation between main and MR risk:")
                for _, row in large_dev.iterrows():
                    keys_str = ", ".join(f"{k}={row[k]}" for k in merge_keys)
                    logger.warning(
                        f"  {keys_str}: main={row['b2_main']:.4f}, mr={row['b2_mr']:.4f}, "
                        f"delta={row['b2_delta_pct']:+.1f}%, source={row['risk_source']}"
                    )

            mr_source_counts = comparison_df["risk_source"].value_counts().to_dict()
            logger.info(f"Hybrid risk sources: {mr_source_counts}")

            logger.info("Merging b2_ever_h6_tmp into data_demand_mr...")
            # Drop MR outcome columns to avoid conflicts with downstream todu prediction
            data_demand_mr = data_demand_mr.drop(columns=["todu_30ever_h6", "todu_amt_pile_h6"], errors="ignore")
            data_demand_mr = pd.merge(data_demand_mr, merge_df, on=merge_keys, how="left")

            # Keep variable only for booked accounts
            non_booked_mask = data_demand_mr["status_name"] != StatusName.BOOKED.value
            data_demand_mr.loc[non_booked_mask, "b2_ever_h6_tmp"] = np.nan

        elif all(col in data_booked.columns for col in required_agg_cols):
            # Default mode: use main-period risk for all bins
            logger.info(f"Calculating b2_ever_h6_tmp aggregated by {merge_keys} from initial period...")
            agg_data = data_booked.groupby(merge_keys)[["todu_30ever_h6", "todu_amt_pile_h6"]].sum().reset_index()

            # Calculate b2_ever_h6_tmp
            agg_data["b2_ever_h6_tmp"] = calculate_b2_ever_h6(
                agg_data["todu_30ever_h6"], agg_data["todu_amt_pile_h6"], multiplier=settings.multiplier
            ).fillna(0.0)

            merge_df = agg_data[merge_keys + ["b2_ever_h6_tmp"]]

            logger.info("Merging b2_ever_h6_tmp into data_demand_mr...")
            data_demand_mr = pd.merge(data_demand_mr, merge_df, on=merge_keys, how="left")

            # Keep variable only for booked accounts
            non_booked_mask = data_demand_mr["status_name"] != StatusName.BOOKED.value
            data_demand_mr.loc[non_booked_mask, "b2_ever_h6_tmp"] = np.nan

            # Check for booked accounts with null b2_ever_h6_tmp
            booked_mask = data_demand_mr["status_name"] == StatusName.BOOKED.value
            null_b2_mask = booked_mask & data_demand_mr["b2_ever_h6_tmp"].isna()
            null_count = null_b2_mask.sum()

            if null_count > 0:
                # Get the missing bin combinations for logging
                missing_bins = data_demand_mr.loc[null_b2_mask, merge_keys].drop_duplicates()
                logger.warning(
                    f"Found {null_count:,} booked accounts with null b2_ever_h6_tmp. "
                    f"These bin combinations exist in MR period but not in initial period. "
                    f"Inferring b2_ever_h6 using the risk model..."
                )
                for bin_combo in missing_bins.itertuples(index=False):
                    logger.warning(f"  Missing bin: {bin_combo._asdict()}")

                # Use inference model to predict b2_ever_h6 for missing bins
                try:
                    best_model_info = risk_inference.get("best_model_info")
                    if best_model_info is None:
                        raise KeyError("'best_model_info' not found in risk_inference")
                    final_model = best_model_info.get("model")
                    if final_model is None:
                        raise KeyError("'model' not found in risk_inference['best_model_info']")
                    final_features = risk_inference.get("features")
                    if final_features is None:
                        raise KeyError("'features' not found in risk_inference")

                    # Use the model's actual training variables (inference_variables),
                    # not merge_keys (settings.variables) which may include extra
                    # dimensions the model was not trained on.
                    model_vars = risk_inference.get("model_variables", merge_keys)

                    # Create a DataFrame with missing bin combinations for prediction
                    missing_bins_df = missing_bins.copy()

                    # Apply calculate_B2 to predict b2_ever_h6 for missing bins
                    missing_bins_df = calculate_B2(
                        missing_bins_df, final_model, model_vars, stress_factor, final_features
                    )

                    # Clip inferred risk to the observed training range to prevent extrapolation
                    observed_risk = agg_data["b2_ever_h6_tmp"].dropna()
                    if len(observed_risk) > 0:
                        risk_floor = float(observed_risk.min())
                        risk_ceil = float(observed_risk.max())
                        missing_bins_df["b2_ever_h6"] = missing_bins_df["b2_ever_h6"].clip(
                            lower=risk_floor, upper=risk_ceil
                        )
                        logger.info(
                            f"  Clipped model-imputed risk to observed range [{risk_floor:.4f}, {risk_ceil:.4f}]"
                        )

                    # Merge clipped inferred values into data_demand_mr
                    inferred_b2 = missing_bins_df[merge_keys + ["b2_ever_h6"]].rename(
                        columns={"b2_ever_h6": "b2_ever_h6_inferred"}
                    )
                    data_demand_mr = pd.merge(data_demand_mr, inferred_b2, on=merge_keys, how="left")

                    # Fill missing b2_ever_h6_tmp with inferred values
                    fill_mask = data_demand_mr["b2_ever_h6_tmp"].isna() & data_demand_mr["b2_ever_h6_inferred"].notna()
                    data_demand_mr.loc[fill_mask, "b2_ever_h6_tmp"] = data_demand_mr.loc[
                        fill_mask, "b2_ever_h6_inferred"
                    ]
                    # Track which rows were imputed (for diagnostics below)
                    imputed_mask = fill_mask.copy()

                    # Drop the helper column
                    data_demand_mr = data_demand_mr.drop(columns=["b2_ever_h6_inferred"], errors="ignore")

                    # Recompute booked_mask after merge to avoid stale index alignment
                    booked_mask = data_demand_mr["status_name"] == StatusName.BOOKED.value

                    # Verify all booked accounts now have values
                    remaining_nulls = (booked_mask & data_demand_mr["b2_ever_h6_tmp"].isna()).sum()
                    if remaining_nulls > 0:
                        logger.error(
                            f"Still have {remaining_nulls:,} booked accounts with null b2_ever_h6_tmp after inference"
                        )
                        raise ValueError("Inference failed to fill all missing b2_ever_h6_tmp values")

                    logger.info(
                        f"Successfully inferred b2_ever_h6 for {null_count:,} booked accounts "
                        f"across {len(missing_bins)} bin combinations using risk model"
                    )

                    # Diagnostic: fraction of MR production from imputed cells
                    total_booked = booked_mask.sum()
                    imputed_pct = null_count / max(total_booked, 1) * 100
                    imputed_prod = (
                        data_demand_mr.loc[imputed_mask, "oa_amt_h0"].sum()
                        if "oa_amt_h0" in data_demand_mr.columns
                        else 0
                    )
                    total_prod = (
                        data_demand_mr.loc[booked_mask, "oa_amt_h0"].sum()
                        if "oa_amt_h0" in data_demand_mr.columns
                        else 0
                    )
                    prod_pct = imputed_prod / max(total_prod, 1) * 100
                    logger.info(
                        f"  Imputed accounts: {null_count:,}/{total_booked:,} ({imputed_pct:.1f}%) "
                        f"| Imputed production: {prod_pct:.1f}% of MR total"
                    )
                    if prod_pct > 20:
                        logger.warning(
                            f"HIGH IMPUTATION RATIO: {prod_pct:.1f}% of MR production comes from "
                            f"imputed cells ({len(missing_bins)} missing bin combinations). "
                            f"MR results may be unreliable — consider coarser binning or validating "
                            f"imputed risk values against out-of-sample benchmarks."
                        )

                except (ValueError, KeyError, RuntimeError) as e:
                    logger.error(f"Error inferring b2_ever_h6 for missing bins: {e}")
                    raise ValueError(
                        f"Data integrity error: {null_count:,} booked accounts in MR period "
                        f"have no matching b2_ever_h6 from initial period, and inference failed: {e}"
                    ) from e
            else:
                logger.info(f"Validation passed: all {booked_mask.sum():,} booked accounts have b2_ever_h6_tmp values")

        else:
            logger.warning(
                f"Missing columns for aggregation. Required: {required_agg_cols}. Skipping b2_ever_h6_tmp calculation."
            )

        # --- Calculate todu_amt_pile_h6 using inference model ---
        # The inference model (reg_todu_amt_pile) was trained on SUMMED bins.
        # To avoid multiplying the intercept by the number of loans,
        # we predict on the sum of oa_amt per bin and pro-rate the result.
        logger.info("Calculating todu_amt_pile_h6 for booked accounts in MR period (bin-aggregated)...")

        data_demand_mr["todu_amt_pile_h6"] = np.nan
        booked_mask = (data_demand_mr["status_name"] == StatusName.BOOKED.value) & (data_demand_mr["oa_amt"].notna())

        if booked_mask.any():
            try:
                # 1. Sum oa_amt per bin
                bin_sums = data_demand_mr.loc[booked_mask].groupby(merge_keys)["oa_amt"].sum().reset_index()

                # 2. Predict on bin sums
                bin_sums["todu_amt_pile_h6_bin"] = reg_todu_amt_pile.predict(bin_sums[["oa_amt"]])

                # 3. Map back to data_demand_mr, pro-rated by oa_amt
                bin_sums_idx = bin_sums.set_index(merge_keys)
                merged = data_demand_mr.loc[booked_mask, merge_keys + ["oa_amt"]].join(
                    bin_sums_idx, on=merge_keys, rsuffix="_sum"
                )

                # Safe division — warn about zero-production bins that cannot be pro-rated
                zero_prod_bins = bin_sums.loc[bin_sums["oa_amt"] == 0, merge_keys]
                if len(zero_prod_bins) > 0:
                    logger.warning(
                        f"Found {len(zero_prod_bins)} bin(s) with zero total oa_amt — "
                        f"todu_amt_pile_h6 cannot be pro-rated for these bins and will be set to 0."
                    )
                divisor = merged["oa_amt_sum"].replace(0, np.nan)
                preds = merged["todu_amt_pile_h6_bin"] * (merged["oa_amt"] / divisor)

                data_demand_mr.loc[booked_mask, "todu_amt_pile_h6"] = preds.fillna(0.0)
            except (ValueError, KeyError) as e:
                logger.error(f"Error predicting todu_amt_pile_h6: {e}")
        else:
            logger.warning("No booked accounts with valid oa_amt found for prediction.")

        # --- Calculate todu_30ever_h6 ---
        logger.info("Calculating todu_30ever_h6 for booked accounts in MR period...")
        data_demand_mr["todu_30ever_h6"] = np.nan

        calc_mask = data_demand_mr["todu_amt_pile_h6"].notna() & data_demand_mr["b2_ever_h6_tmp"].notna()

        if calc_mask.any():
            data_demand_mr.loc[calc_mask, "todu_30ever_h6"] = calculate_todu_30ever_from_b2(
                data_demand_mr.loc[calc_mask, "b2_ever_h6_tmp"],
                data_demand_mr.loc[calc_mask, "todu_amt_pile_h6"],
                multiplier=settings.multiplier,
            )

        # Create data_booked_mr
        data_booked_mr = data_demand_mr[data_demand_mr["status_name"] == StatusName.BOOKED.value].copy()

        # --- Apply Full Optimization Pipeline to MR Dataset ---
        logger.info("Applying full optimization pipeline to MR dataset...")

        data_summary_desagregado_mr = run_optimization_pipeline(
            data_booked=data_booked_mr,
            data_demand=data_demand_mr,
            risk_inference=risk_inference,
            reg_todu_amt_pile=reg_todu_amt_pile,
            stressor=stress_factor,
            tasa_fin=tasa_fin,
            indicators=settings.indicators,
            variables=settings.variables,
            annual_coef=annual_coef,
            reject_inference_method=settings.reject_inference_method,
            reject_uplift_factor=settings.reject_uplift_factor,
            reject_max_risk_multiplier=settings.reject_max_risk_multiplier,
            reject_parceling_method=settings.reject_parceling_method,
            reject_bayesian_smoothing=settings.reject_bayesian_smoothing,
            reject_bayesian_prior_strength=settings.reject_bayesian_prior_strength,
            reject_enforce_monotonicity=settings.reject_enforce_monotonicity,
            reject_include_all_rejections=settings.reject_include_all_rejections,
        )

        # Save MR summary
        summary_path = output.mr_summary_csv(file_suffix)
        data_summary_desagregado_mr.to_csv(summary_path, index=False)
        logger.info(f"MR summary data saved to {summary_path}")

        # --- Visualize b2_ever_h6 for MR ---
        VARIABLES = settings.variables

        data_surf_mr = data_summary_desagregado_mr.copy()
        data_surf_mr["b2_ever_h6"] = calculate_b2_ever_h6(
            data_surf_mr["todu_30ever_h6"],
            data_surf_mr["todu_amt_pile_h6"],
            multiplier=settings.multiplier,
            as_percentage=True,
        )
        # Compute complementary H3 metric on MR surface when columns are available
        if "todu_30ever_h3" in data_surf_mr.columns and "todu_amt_pile_h3" in data_surf_mr.columns:
            data_surf_mr["b2_ever_h3"] = calculate_b2_ever_h6(
                data_surf_mr["todu_30ever_h3"],
                data_surf_mr["todu_amt_pile_h3"],
                multiplier=settings.multiplier_h3,
                as_percentage=True,
            )

        if len(VARIABLES) == 2:
            logger.info("Generating b2_ever_h6 visualization for MR dataset...")

            fig_mr = go.Figure()

            data_surf_pivot_mr = data_surf_mr.pivot(index=VARIABLES[1], columns=VARIABLES[0], values="b2_ever_h6")

            fig_mr.add_trace(
                go.Surface(
                    x=data_surf_pivot_mr.columns,
                    y=data_surf_pivot_mr.index,
                    z=data_surf_pivot_mr.values,
                    colorscale="turbo",
                )
            )

            styles.apply_plotly_style(
                fig_mr,
                title=f"B2 Ever H6 vs. Octroi and Risk Score (MR Period - Aggregated){file_suffix}",
                width=1500,
                height=700,
            )

            fig_mr.update_layout(
                scene=dict(
                    xaxis=dict(title=VARIABLES[0]),
                    yaxis=dict(title=VARIABLES[1]),
                    zaxis=dict(title="b2_ever_h6"),
                    aspectratio=dict(x=1, y=1, z=1),
                )
            )

            output_plot_path_mr = output.mr_b2_visualization_html(file_suffix)
            fig_mr.write_html(output_plot_path_mr)
            logger.info(f"MR Visualization saved to {output_plot_path_mr}")
        elif len(VARIABLES) >= 3:
            logger.info(f"Generating per-slice MR heatmaps for {len(VARIABLES)}-variable grid...")
            from plotly.subplots import make_subplots

            var0, var1, var2 = VARIABLES[0], VARIABLES[1], VARIABLES[2]
            v2_vals = sorted(data_surf_mr[var2].unique())

            fig_mr = make_subplots(
                rows=1,
                cols=len(v2_vals),
                subplot_titles=[f"{var2} = {v}" for v in v2_vals],
            )

            for col_idx, v2 in enumerate(v2_vals, start=1):
                slice_df = data_surf_mr[data_surf_mr[var2] == v2]
                pivot = slice_df.pivot(index=var1, columns=var0, values="b2_ever_h6").sort_index(ascending=True)
                text_pivot = slice_df.copy()
                text_pivot["_text"] = text_pivot.apply(
                    lambda r: f"{r.get('oa_amt_h0', 0) / 1e6:,.1f}M\n{r['b2_ever_h6']:.1f}%", axis=1
                )
                text_piv = text_pivot.pivot(index=var1, columns=var0, values="_text").sort_index(ascending=True)

                fig_mr.add_trace(
                    go.Heatmap(
                        x=[str(c) for c in pivot.columns],
                        y=[str(r) for r in pivot.index],
                        z=pivot.values,
                        text=text_piv.values if not text_piv.empty else None,
                        texttemplate="%{text}",
                        colorscale=[(0, "rgba(255,255,255,1)"), (1, "rgba(157,13,20,1)")],
                        zmin=0,
                        showscale=col_idx == len(v2_vals),
                    ),
                    row=1,
                    col=col_idx,
                )
                fig_mr.update_xaxes(title_text=var0, row=1, col=col_idx)
                if col_idx == 1:
                    fig_mr.update_yaxes(title_text=var1, row=1, col=col_idx)

            styles.apply_plotly_style(
                fig_mr,
                title=f"B2 Ever H6 — MR Period (Aggregated){file_suffix}",
                width=max(500 * len(v2_vals), 800),
                height=500,
            )

            output_plot_path_mr = output.mr_b2_visualization_html(file_suffix)
            fig_mr.write_html(output_plot_path_mr)
            logger.info(f"MR per-slice visualization saved to {output_plot_path_mr}")

        # --- Cleanup ---
        if "b2_ever_h6_tmp" in data_demand_mr.columns:
            logger.info("Dropping b2_ever_h6_tmp from data_demand_mr and data_booked_mr...")
            data_demand_mr = data_demand_mr.drop(columns=["b2_ever_h6_tmp"], errors="ignore")
            data_booked_mr = data_booked_mr.drop(columns=["b2_ever_h6_tmp"], errors="ignore")

        # --- Generate Risk Production Summary Table for MR ---
        logger.info("Generating Risk Production Summary Table for MR period...")

        if "status_name" in data_demand_mr.columns and "oa_amt_h0" in data_demand_mr.columns:
            mr_total_demand = data_demand_mr.loc[data_demand_mr["status_name"] != "canceled", "oa_amt_h0"].sum()
        else:
            mr_total_demand = data_demand_mr["oa_amt_h0"].sum() if "oa_amt_h0" in data_demand_mr.columns else 0.0

        mr_summary_table = calculate_metrics_from_cuts(
            data_summary_desagregado_mr,
            optimal_solution_df,
            VARIABLES,
            settings.inv_vars,
            mask=mask,
            grid=grid,
            multiplier_h3=settings.multiplier_h3,
            multiplier=settings.multiplier,
            total_demand=mr_total_demand,
        )

        if mr_summary_table is not None:
            mr_summary_path = output.mr_risk_production_summary_csv(file_suffix)
            mr_summary_table.to_csv(mr_summary_path, index=False)
            logger.info(f"MR Risk Production Summary Table saved to {mr_summary_path}")
            logger.info(f"MR Table:\n{mr_summary_table.to_string()}")

        # --- Calculate PSI/CSI Stability Metrics ---
        logger.info("Calculating PSI/CSI stability metrics (Main vs MR)...")
        try:
            # Prefer known score columns, fall back to any shared numeric columns
            requested_vars = ["score_rf", "risk_score_rf", "oa_amt"]
            stability_vars = [v for v in requested_vars if v in data_booked.columns and v in data_booked_mr.columns]
            if not stability_vars:
                shared_cols = set(data_booked.columns) & set(data_booked_mr.columns)
                stability_vars = [
                    c for c in shared_cols if data_booked[c].dtype.kind in ("f", "i") and c not in VARIABLES
                ][:5]
                if stability_vars:
                    logger.info(f"Stability: using fallback numeric columns: {stability_vars}")

            if stability_vars:
                # Determine main score variable for overall PSI
                score_var = "risk_score_rf" if "risk_score_rf" in stability_vars else stability_vars[0]

                stability_report = compare_main_vs_mr(
                    main_df=data_booked,
                    mr_df=data_booked_mr,
                    variables=stability_vars,
                    score_variable=score_var,
                    output_path=output.stability_report_html(file_suffix),
                    verbose=True,
                )

                # Save stability results to CSV
                stability_df = stability_report.to_dataframe()
                stability_csv_path = output.stability_psi_csv(file_suffix)
                stability_df.to_csv(stability_csv_path, index=False)
                logger.info(f"Stability metrics saved to {stability_csv_path}")

                # Generate structured drift alerts
                try:
                    from src.alerts import generate_drift_alerts

                    alert_report = generate_drift_alerts(
                        stability_report,
                        segment=settings.segment_filter,
                        period="MR",
                    )
                    alert_json_path = output.drift_alerts_json(file_suffix)
                    alert_report.to_json(alert_json_path)
                    logger.info(f"Drift alerts saved to {alert_json_path}")
                except (ImportError, ValueError) as e:
                    logger.warning(f"Failed to generate drift alerts: {e}")

                # Log summary
                if stability_report.unstable_vars:
                    logger.warning(
                        f"STABILITY WARNING: {len(stability_report.unstable_vars)} variables "
                        f"show significant drift (PSI >= {PSI_UNSTABLE_THRESHOLD}): "
                        f"{[r.variable for r in stability_report.unstable_vars]}"
                    )
            else:
                logger.warning("No numeric variables found for stability analysis")

        except (ValueError, KeyError) as e:
            logger.warning(f"Error calculating stability metrics: {e}")
            logger.warning(traceback.format_exc())

    except (ValueError, KeyError, RuntimeError) as e:
        logger.error(f"Error processing MR period: {e}")
        logger.error(traceback.format_exc())
