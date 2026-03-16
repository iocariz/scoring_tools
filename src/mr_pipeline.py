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
        var1_col = variables[1] if len(variables) > 1 else None

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
            if var1_col is None:
                raise ValueError("2-var cut_map path requires at least 2 variables; use mask/grid for 1-var configs")
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

        logger.info(
            f"MR summary diagnostics: n_bins={len(df)}, passes_cut={df['passes_cut'].sum()}/{len(df)}, "
            f"actual: todu_30ever_h6_boo={actual_rn:.4f}, todu_amt_pile_h6_boo={actual_rd:.4f}, "
            f"risk={actual_risk}%"
        )
        if actual_risk == 0.0 or actual_risk is None:
            # Log per-bin _boo values to diagnose zero-risk
            for _, row in df.iterrows():
                keys_str = ", ".join(f"{k}={row[k]}" for k in variables)
                logger.warning(
                    f"  {keys_str}: todu_30ever_h6_boo={row.get('todu_30ever_h6_boo', 'MISSING')}, "
                    f"todu_amt_pile_h6_boo={row.get('todu_amt_pile_h6_boo', 'MISSING')}, "
                    f"passes_cut={row.get('passes_cut', 'MISSING')}"
                )

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
    mr_maturity_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-bin ``b2_ever_h6_tmp`` using MR outcomes when sufficient, else main-period.

    Parameters
    ----------
    mr_maturity_months : int
        Minimum number of months since booking for an account to be
        considered mature for H6 outcomes.  Accounts booked too recently
        are excluded from the ``n_obs_mr`` count and the ``b2_mr``
        calculation to avoid diluting risk with immature zeros.  Set to 0
        to disable maturity filtering (original behavior).

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
        main_agg["todu_30ever_h6"], main_agg["todu_amt_pile_h6"], multiplier=multiplier, decimals=6
    )
    # NaN means "no usable H6 data" (zero exposure) — preserve to distinguish from zero risk
    n_obs_main = data_booked.groupby(merge_keys).size().reset_index(name="n_obs_main")
    main_agg = main_agg.merge(n_obs_main, on=merge_keys, how="left")
    main_agg = main_agg[merge_keys + ["b2_main", "n_obs_main"]]

    # --- MR-period aggregation ---
    mr_booked = data_demand_mr[data_demand_mr["status_name"] == StatusName.BOOKED.value]
    mr_h6_valid = mr_booked[mr_booked["todu_30ever_h6"].notna() & mr_booked["todu_amt_pile_h6"].notna()]

    # Filter for mature H6 observations: maturity = max(mis_date) - mis_date.
    # Only accounts with at least mr_maturity_months of seasoning are included
    # in b2_mr to avoid diluting risk with immature zeros.
    if mr_maturity_months > 0 and "mis_date" in mr_h6_valid.columns:
        max_date = mr_h6_valid["mis_date"].max()
        maturity = (max_date.year - mr_h6_valid["mis_date"].dt.year) * 12 + (
            max_date.month - mr_h6_valid["mis_date"].dt.month
        )
        mature_mask = maturity >= mr_maturity_months
        n_total = len(mr_h6_valid)
        n_mature = mature_mask.sum()
        if n_mature > 0:
            cutoff_date = max_date - pd.DateOffset(months=mr_maturity_months)
            logger.info(
                f"H6 maturity filter: {n_mature:,}/{n_total:,} accounts with >={mr_maturity_months}mo "
                f"maturity (booked on or before {cutoff_date.date()}, max_date={max_date.date()})"
            )
            mr_h6_valid = mr_h6_valid[mature_mask]
        else:
            logger.warning(
                f"No accounts with >={mr_maturity_months}mo H6 maturity (max_date={max_date.date()}). "
                f"Falling back to main-period or H3 extrapolation for all bins."
            )
            mr_h6_valid = mr_h6_valid.iloc[:0]  # empty DataFrame, forces fallback

    mr_agg = mr_h6_valid.groupby(merge_keys)[["todu_30ever_h6", "todu_amt_pile_h6"]].sum().reset_index()
    mr_agg["b2_mr"] = calculate_b2_ever_h6(
        mr_agg["todu_30ever_h6"], mr_agg["todu_amt_pile_h6"], multiplier=multiplier, decimals=6
    )
    # NaN means "no usable H6 data" — preserve to distinguish from zero risk
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
                main_h3_agg["todu_30ever_h3"], main_h3_agg["todu_amt_pile_h3"], multiplier=multiplier_h3, decimals=6
            )
            # NaN means "no usable H3 data" — preserve for ratio computation
            combined = combined.merge(main_h3_agg[merge_keys + ["b2_main_h3"]], on=merge_keys, how="left")
        else:
            combined["b2_main_h3"] = np.nan

        # MR-period H3 — only accounts with mature H3 (maturity >= 3 months)
        if all(c in mr_booked_h3.columns for c in h3_cols):
            mr_h3_valid = mr_booked_h3[
                mr_booked_h3["todu_30ever_h3"].notna() & mr_booked_h3["todu_amt_pile_h3"].notna()
            ]
            # Filter for H3 maturity: maturity = max(mis_date) - mis_date >= 3 months
            h3_maturity_months = 3
            if "mis_date" in mr_h3_valid.columns and len(mr_h3_valid) > 0:
                max_date_h3 = mr_h3_valid["mis_date"].max()
                h3_maturity = (max_date_h3.year - mr_h3_valid["mis_date"].dt.year) * 12 + (
                    max_date_h3.month - mr_h3_valid["mis_date"].dt.month
                )
                h3_mature_mask = h3_maturity >= h3_maturity_months
                n_h3_total = len(mr_h3_valid)
                n_h3_mature = h3_mature_mask.sum()
                if n_h3_mature > 0:
                    logger.info(
                        f"H3 maturity filter: {n_h3_mature:,}/{n_h3_total:,} accounts with "
                        f">={h3_maturity_months}mo maturity (max_date={max_date_h3.date()})"
                    )
                    mr_h3_valid = mr_h3_valid[h3_mature_mask]
                else:
                    logger.warning(
                        f"No accounts with >={h3_maturity_months}mo H3 maturity. "
                        f"H3 extrapolation will not be available."
                    )
                    mr_h3_valid = mr_h3_valid.iloc[:0]
            mr_h3_agg = mr_h3_valid.groupby(merge_keys)[h3_cols].sum().reset_index()
            n_obs_mr_h3 = mr_h3_valid.groupby(merge_keys).size().reset_index(name="n_obs_mr_h3")
            mr_h3_agg = mr_h3_agg.merge(n_obs_mr_h3, on=merge_keys, how="left")
            mr_h3_agg["b2_mr_h3"] = calculate_b2_ever_h6(
                mr_h3_agg["todu_30ever_h3"], mr_h3_agg["todu_amt_pile_h3"], multiplier=multiplier_h3, decimals=6
            )
            # Do NOT fillna(0.0) — NaN means "no usable H3 data" and should
            # trigger fallback to main-period.  A genuine 0.0 (zero todu_30ever_h3)
            # is preserved and handled below via the can_extrapolate guard.
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

    # --- Diagnostic: monthly H6/H3 ratio trend in main period ---
    if has_h3 and "mis_date" in data_booked.columns:
        h3_cols_trend = ["todu_30ever_h3", "todu_amt_pile_h3"]
        h6_cols_trend = ["todu_30ever_h6", "todu_amt_pile_h6"]
        if all(c in data_booked.columns for c in h3_cols_trend + h6_cols_trend):
            booked_main = data_booked.copy()
            booked_main["_month"] = booked_main["mis_date"].dt.to_period("M")
            monthly = booked_main.groupby("_month")[h6_cols_trend + h3_cols_trend].sum()
            # Compute unrounded ratios to avoid losing precision on small monthly values
            safe_pile_h6 = monthly["todu_amt_pile_h6"].replace(0, np.nan)
            safe_pile_h3 = monthly["todu_amt_pile_h3"].replace(0, np.nan)
            monthly["b2_h6"] = multiplier * monthly["todu_30ever_h6"] / safe_pile_h6
            monthly["b2_h3"] = multiplier_h3 * monthly["todu_30ever_h3"] / safe_pile_h3
            monthly["h6_h3_ratio"] = np.where(
                monthly["b2_h3"].notna() & (monthly["b2_h3"].abs() > 1e-6),
                monthly["b2_h6"] / monthly["b2_h3"],
                np.nan,
            )
            valid_months = monthly["h6_h3_ratio"].dropna()

            logger.info(
                f"H6/H3 ratio trend: {len(monthly)} months in main period, "
                f"{len(valid_months)} with valid ratio"
            )
            if len(valid_months) >= 3:
                logger.info("H6/H3 RATIO TREND (main period, monthly):")
                for period, row in monthly.iterrows():
                    ratio_str = f"{row['h6_h3_ratio']:.2f}" if pd.notna(row["h6_h3_ratio"]) else "—"
                    logger.info(
                        f"  {period}:  H6={row['b2_h6'] * 100:.2f}%  "
                        f"H3={row['b2_h3'] * 100:.2f}%  ratio={ratio_str}"
                    )

                # Overall ratio from raw numerators/denominators (not sum of rates)
                valid_months_mask = monthly["h6_h3_ratio"].notna()
                raw_h6 = multiplier * monthly.loc[valid_months_mask, "todu_30ever_h6"].sum()
                raw_h3_den = monthly.loc[valid_months_mask, "todu_amt_pile_h6"].sum()
                raw_h3_num = multiplier_h3 * monthly.loc[valid_months_mask, "todu_30ever_h3"].sum()
                raw_h3_den_h3 = monthly.loc[valid_months_mask, "todu_amt_pile_h3"].sum()
                overall_b2_h6 = raw_h6 / raw_h3_den if raw_h3_den > 0 else np.nan
                overall_b2_h3 = raw_h3_num / raw_h3_den_h3 if raw_h3_den_h3 > 0 else np.nan
                ratios_arr = valid_months.values
                if pd.notna(overall_b2_h6) and pd.notna(overall_b2_h3) and overall_b2_h3 > 0:
                    overall_ratio = overall_b2_h6 / overall_b2_h3
                    logger.info(
                        f"  Overall ratio (Σh6/Σh3): {overall_ratio:.2f}  |  "
                        f"Mean of monthly ratios: {np.mean(ratios_arr):.2f}  |  "
                        f"Median of monthly ratios: {np.median(ratios_arr):.2f}"
                    )

                # Simple linear trend
                x = np.arange(len(valid_months), dtype=float)
                y = valid_months.values
                if len(x) >= 3:
                    slope, intercept = np.polyfit(x, y, 1)
                    pct_change = slope * len(x) / np.mean(y) * 100 if np.mean(y) > 0 else 0.0
                    logger.info(
                        f"  Trend: slope={slope:.4f}/month, total change={pct_change:+.1f}% "
                        f"over {len(x)} months (mean monthly ratio={np.mean(y):.2f})"
                    )
                    if abs(pct_change) > 20:
                        logger.warning(
                            f"  H6/H3 ratio shows significant trend ({pct_change:+.1f}%). "
                            f"Static ratio extrapolation may be inaccurate for the MR period."
                        )
            else:
                logger.warning(
                    f"H6/H3 ratio trend: only {len(valid_months)} valid months "
                    f"(need ≥3). Skipping trend analysis."
                )
        else:
            missing = [c for c in h3_cols_trend + h6_cols_trend if c not in data_booked.columns]
            logger.info(f"H6/H3 ratio trend: skipped — missing columns in data_booked: {missing}")
    elif has_h3:
        logger.info("H6/H3 ratio trend: skipped — mis_date not in data_booked")
    else:
        logger.debug("H6/H3 ratio trend: skipped — no H3 data configured")

    # --- Choose risk source per bin ---
    # Mature H6 observed data (may be empty if MR window < mr_maturity_months)
    use_mr = combined["n_obs_mr"].fillna(0) >= min_obs

    if has_h3:
        # H3→H6 ratio from main-period (fully mature) data
        # Guard: both b2_main and b2_main_h3 must be valid and > 0 for a meaningful ratio
        ratio_valid = (
            combined["b2_main"].notna()
            & (combined["b2_main"] > 0)
            & combined["b2_main_h3"].notna()
            & (combined["b2_main_h3"].abs() > 0.01)
            & (combined["n_obs_main"].fillna(0) >= min_obs)
        )
        h6_h3_ratio_raw = np.where(ratio_valid, combined["b2_main"] / combined["b2_main_h3"], np.nan)

        # Compute a global (median) ratio for bins with unreliable per-bin estimates
        valid_ratios = h6_h3_ratio_raw[np.isfinite(h6_h3_ratio_raw)]
        global_h6_h3_ratio = float(np.median(valid_ratios)) if len(valid_ratios) > 0 else np.nan

        # --- Reconciliation: compare per-bin vs global ratio computations ---
        if len(valid_ratios) > 0:
            # Per-bin ratios (what we use for extrapolation)
            per_bin_mean = float(np.mean(valid_ratios))
            per_bin_median = float(np.median(valid_ratios))

            # Global ratio from raw numerators/denominators (not sum of rates)
            # Recompute from data_booked directly (main_agg may have dropped raw columns)
            valid_mask = ratio_valid
            valid_keys = combined.loc[valid_mask, merge_keys]
            if len(valid_keys) > 0:
                booked_valid = data_booked.merge(valid_keys, on=merge_keys, how="inner")
                h6_cols_raw = ["todu_30ever_h6", "todu_amt_pile_h6"]
                h3_cols_raw = ["todu_30ever_h3", "todu_amt_pile_h3"]
                if all(c in booked_valid.columns for c in h6_cols_raw):
                    global_h6_rate = calculate_b2_ever_h6(
                        booked_valid["todu_30ever_h6"].sum(),
                        booked_valid["todu_amt_pile_h6"].sum(),
                        multiplier=multiplier, decimals=6,
                    )
                else:
                    global_h6_rate = np.nan
                if all(c in booked_valid.columns for c in h3_cols_raw):
                    global_h3_rate = calculate_b2_ever_h6(
                        booked_valid["todu_30ever_h3"].sum(),
                        booked_valid["todu_amt_pile_h3"].sum(),
                        multiplier=multiplier_h3, decimals=6,
                    )
                else:
                    global_h3_rate = np.nan
                ratio_of_sums = float(global_h6_rate / global_h3_rate) if pd.notna(global_h6_rate) and pd.notna(global_h3_rate) and global_h3_rate > 0 else np.nan
            else:
                ratio_of_sums = np.nan

            # Weighted ratio (by n_obs)
            weights = combined.loc[ratio_valid, "n_obs_main"].values
            wt_ratio = float(np.average(valid_ratios, weights=weights)) if weights.sum() > 0 else np.nan

            logger.info("H6/H3 RATIO RECONCILIATION (main period):")
            logger.info("  Per-bin (used for extrapolation):")
            logger.info(f"    median={per_bin_median:.3f}  mean={per_bin_mean:.3f}  n_bins={len(valid_ratios)}")
            logger.info(f"    obs-weighted mean={wt_ratio:.3f}")
            logger.info("  Per-bin individual ratios:")
            valid_positions = np.where(ratio_valid)[0]
            for i, (_idx, row) in enumerate(combined[ratio_valid].iterrows()):
                keys_str = ", ".join(f"{k}={row[k]}" for k in merge_keys)
                pos = valid_positions[i]  # position in the full DataFrame
                logger.info(
                    f"    {keys_str}: b2_main={row['b2_main'] * 100:.2f}%  "
                    f"b2_main_h3={row['b2_main_h3'] * 100:.2f}%  "
                    f"ratio={h6_h3_ratio_raw[pos]:.3f}  n_obs={int(row['n_obs_main'])}"
                )
            logger.info(f"  Global (Σb2_main / Σb2_main_h3): {ratio_of_sums:.3f}")
            logger.info(
                f"  NOTE: extrapolation uses per-bin median ({per_bin_median:.3f}). "
                f"Monthly trend mean ({np.mean(valid_ratios):.3f}) differs due to "
                f"aggregation axis (bins vs months) and statistic (median vs mean)."
            )

        # Use per-bin ratio where reliable, fall back to global median
        h6_h3_ratio = np.where(np.isfinite(h6_h3_ratio_raw), h6_h3_ratio_raw, global_h6_h3_ratio)
        h6_h3_ratio = np.clip(h6_h3_ratio, 0.5, 5.0)

        # H3 extrapolation: MR H3 risk × calibrated H6/H3 scaling
        # Use a lower threshold for H3 obs — we only have ~half the MR window
        # with mature H3, so requiring the full min_obs is too strict
        h3_min_obs = max(min_obs // 2, 10)
        has_h3_obs = (
            combined["b2_mr_h3"].notna()
            # Zero H3 risk with enough observations IS valid — it means genuinely low risk.
            # Only NaN (no data) should trigger fallback. We keep b2_mr_h3 >= 0 to exclude
            # negative artifacts while accepting genuine zeros.
            & (combined["b2_mr_h3"] >= 0)
            & (combined["n_obs_mr_h3"].fillna(0) >= h3_min_obs)
        )

        can_extrapolate = has_h3_obs & np.isfinite(h6_h3_ratio)
        h6_from_h3 = extrapolate_h3_to_h6(
            combined["b2_mr_h3"], h6_h3_ratio,
            method=mr_extrapolation_method, curvature=mr_extrapolation_curvature,
            b2_h3_main=combined.get("b2_main_h3"),
        )

        # Bins only in MR (no main data) with insufficient H6 AND no H3 extrapolation:
        # leave NaN for model fallback.  Must be computed AFTER can_extrapolate to avoid
        # blocking valid H3 extrapolation for MR-only bins.
        only_mr_sparse = combined["b2_main"].isna() & ~use_mr & ~can_extrapolate

        # Priority (highest to lowest):
        #   1. MR H6 observed — direct MR risk when maturity and sample size are sufficient
        #   2. H3 extrapolation — MR H3 scaled by main-period H6/H3 ratio
        #   3. Main-period imputation — fallback when no MR signal available
        #   4. Model fallback — bins that only exist in MR with no main data and no H3
        conditions = [only_mr_sparse, use_mr, can_extrapolate]
        risk_choices = [np.nan, combined["b2_mr"], h6_from_h3]
        source_choices = ["model_fallback", "mr_observed", "h3_extrapolated"]

        combined["b2_ever_h6_tmp"] = np.select(conditions, risk_choices, default=combined["b2_main"])
        combined["risk_source"] = np.select(conditions, source_choices, default="main_imputed")

        combined["h6_h3_ratio"] = h6_h3_ratio

        n_extrapolated = (combined["risk_source"] == "h3_extrapolated").sum()
        n_main = (combined["risk_source"] == "main_imputed").sum()
        n_mr_obs = (combined["risk_source"] == "mr_observed").sum()
        logger.info(
            f"Risk sources: h3_extrapolated={n_extrapolated}, mr_observed={n_mr_obs}, "
            f"main_imputed={n_main}, model_fallback={(combined['risk_source'] == 'model_fallback').sum()} "
            f"(global H6/H3 ratio={global_h6_h3_ratio:.3f}, h3_min_obs={h3_min_obs})"
        )
    else:
        # No H3 columns — use H6 observed or main-period only
        only_mr_sparse = combined["b2_main"].isna() & ~use_mr
        conditions = [only_mr_sparse, use_mr]
        risk_choices = [np.nan, combined["b2_mr"]]
        source_choices = ["model_fallback", "mr_observed"]

        combined["b2_ever_h6_tmp"] = np.select(conditions, risk_choices, default=combined["b2_main"])
        combined["risk_source"] = np.select(conditions, source_choices, default="main_imputed")

    # --- Per-bin diagnostic: risk source, contribution, and weighted average ---
    prod_col = "mr_production"
    total_prod = combined[prod_col].sum() if prod_col in combined.columns else 0.0

    # All b2_* values are decimals (e.g. 0.0092 = 0.92%). Convert to % for display.
    def _pct(v):
        return f"{v * 100:.2f}%" if pd.notna(v) else "—"

    logger.info("=" * 110)
    logger.info("PER-BIN MR RISK DIAGNOSTIC  (all risk values shown as percentages)")
    logger.info("-" * 110)
    header_keys = "  ".join(f"{k:>12}" for k in merge_keys)
    has_ratio_col = "h6_h3_ratio" in combined.columns
    logger.info(
        f"{header_keys}  {'source':>16}  {'b2_h6_tmp':>10}  {'b2_mr':>10}  "
        f"{'b2_mr_h3':>10}  {'h6/h3':>7}  {'b2_main':>10}  {'prod':>12}  {'wt_contrib':>10}"
    )
    logger.info("-" * 110)

    for _, row in combined.sort_values(merge_keys).iterrows():
        keys_str = "  ".join(f"{row[k]:>12.1f}" if isinstance(row[k], float) else f"{row[k]:>12}" for k in merge_keys)
        prod = row.get(prod_col, 0.0) or 0.0
        risk_tmp = row["b2_ever_h6_tmp"]
        wt_contrib = risk_tmp * 100 * prod / total_prod if total_prod > 0 and pd.notna(risk_tmp) else 0.0
        ratio_val = f"{row['h6_h3_ratio']:.2f}" if has_ratio_col and pd.notna(row.get("h6_h3_ratio")) else "—"
        logger.info(
            f"{keys_str}  {row['risk_source']:>16}  {_pct(risk_tmp):>10}  {_pct(row.get('b2_mr')):>10}  "
            f"{_pct(row.get('b2_mr_h3')):>10}  {ratio_val:>7}  {_pct(row.get('b2_main')):>10}  "
            f"{prod:>12,.0f}  {wt_contrib:>9.4f}%"
        )
    logger.info("-" * 110)

    # Weighted average risk per source and overall
    if total_prod > 0:
        for src in ["mr_observed", "h3_extrapolated", "main_imputed", "model_fallback"]:
            src_mask = combined["risk_source"] == src
            if src_mask.any():
                src_prod = combined.loc[src_mask, prod_col].sum()
                src_risk_avg = (
                    (combined.loc[src_mask, "b2_ever_h6_tmp"] * combined.loc[src_mask, prod_col]).sum()
                    / src_prod
                    * 100
                    if src_prod > 0
                    else 0.0
                )
                logger.info(
                    f"  {src:>16}: weighted_risk={src_risk_avg:.2f}%  "
                    f"production={src_prod:,.0f} ({src_prod / total_prod * 100:.1f}%)  "
                    f"bins={src_mask.sum()}"
                )

        # Exclude model_fallback bins (NaN risk) from weighted average
        has_risk = combined["b2_ever_h6_tmp"].notna()
        prod_with_risk = combined.loc[has_risk, prod_col].sum()
        overall_risk_prod = (
            (combined.loc[has_risk, "b2_ever_h6_tmp"] * combined.loc[has_risk, prod_col]).sum()
            / prod_with_risk * 100
            if prod_with_risk > 0 else 0.0
        )
        logger.info(
            f"  {'OVERALL':>16}: weighted_risk={overall_risk_prod:.2f}% (production-weighted)  "
            f"production={total_prod:,.0f}"
        )
        logger.info("")
        logger.info(
            "  NOTE: The summary table uses multiplier × Σ(todu_30ever_h6) / Σ(todu_amt_pile_h6), "
            "which is exposure-weighted — it will differ from the production-weighted average above."
        )
    logger.info("=" * 110)

    # --- Diagnostic: warn if MR-observed risk is suspiciously low ---
    mr_source_summary = combined["risk_source"].value_counts()
    n_mr_observed = int(mr_source_summary.get("mr_observed", 0))
    n_total_bins = len(combined)
    if n_mr_observed > 0:
        avg_b2_mr = combined.loc[combined["risk_source"] == "mr_observed", "b2_mr"].mean()
        avg_b2_main = combined.loc[combined["risk_source"] == "mr_observed", "b2_main"].mean()
        if avg_b2_main > 0 and avg_b2_mr / avg_b2_main < 0.1:
            logger.warning(
                f"LOW MR RISK: {n_mr_observed}/{n_total_bins} bins use MR-observed H6 risk, "
                f"but avg MR risk ({avg_b2_mr:.4f}) is <10% of main-period ({avg_b2_main:.4f}). "
                f"This may indicate immature loans diluting H6 outcomes. "
                f"Consider increasing mr_maturity_months or mr_min_obs_per_bin."
            )

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
    per_bin_stress: pd.DataFrame | None = None,
    per_bin_tasa_fin: pd.DataFrame | None = None,
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
                mr_maturity_months=settings.mr_maturity_months,
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

            # --- Infer risk for model_fallback bins using trained model ---
            booked_mask = data_demand_mr["status_name"] == StatusName.BOOKED.value
            null_b2_mask = booked_mask & data_demand_mr["b2_ever_h6_tmp"].isna()
            null_count = null_b2_mask.sum()

            if null_count > 0:
                missing_bins = data_demand_mr.loc[null_b2_mask, merge_keys].drop_duplicates()
                logger.warning(
                    f"Hybrid MR: {null_count:,} booked accounts in {len(missing_bins)} model_fallback bins "
                    f"have no risk estimate. Inferring b2_ever_h6 using the risk model..."
                )
                for bin_combo in missing_bins.itertuples(index=False):
                    logger.warning(f"  model_fallback bin: {bin_combo._asdict()}")

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

                    model_vars = risk_inference.get("model_variables", merge_keys)
                    missing_bins_df = missing_bins.copy()
                    missing_bins_df = calculate_B2(
                        missing_bins_df, final_model, model_vars, stress_factor, final_features
                    )

                    # Clip inferred risk using observed risk range from comparison_df
                    observed_risk = comparison_df.loc[
                        comparison_df["risk_source"] != "model_fallback", "b2_ever_h6_tmp"
                    ].dropna()
                    if len(observed_risk) > 0:
                        risk_floor = float(observed_risk.min())
                        risk_ceil_base = float(observed_risk.max())
                        risk_ceil_scaled = risk_ceil_base * settings.mr_extrapolation_risk_multiplier
                        risk_ceil = min(risk_ceil_scaled, settings.mr_extrapolation_hard_cap)
                        missing_bins_df["b2_ever_h6"] = missing_bins_df["b2_ever_h6"].clip(
                            lower=risk_floor, upper=risk_ceil
                        )
                        logger.info(
                            f"  Clipped model-imputed risk to [{risk_floor:.4f}, {risk_ceil:.4f}] "
                            f"(base max={risk_ceil_base:.4f}, mult={settings.mr_extrapolation_risk_multiplier}, "
                            f"cap={settings.mr_extrapolation_hard_cap})"
                        )

                    inferred_b2 = missing_bins_df[merge_keys + ["b2_ever_h6"]].rename(
                        columns={"b2_ever_h6": "b2_ever_h6_inferred"}
                    )
                    data_demand_mr = pd.merge(data_demand_mr, inferred_b2, on=merge_keys, how="left")

                    fill_mask = data_demand_mr["b2_ever_h6_tmp"].isna() & data_demand_mr["b2_ever_h6_inferred"].notna()
                    data_demand_mr.loc[fill_mask, "b2_ever_h6_tmp"] = data_demand_mr.loc[
                        fill_mask, "b2_ever_h6_inferred"
                    ]
                    data_demand_mr = data_demand_mr.drop(columns=["b2_ever_h6_inferred"], errors="ignore")

                    booked_mask = data_demand_mr["status_name"] == StatusName.BOOKED.value
                    remaining_nulls = (booked_mask & data_demand_mr["b2_ever_h6_tmp"].isna()).sum()
                    if remaining_nulls > 0:
                        logger.error(
                            f"Still have {remaining_nulls:,} booked accounts with null b2_ever_h6_tmp "
                            f"after model_fallback inference"
                        )

                    logger.info(
                        f"Successfully inferred b2_ever_h6 for {null_count:,} booked accounts "
                        f"across {len(missing_bins)} model_fallback bins using risk model"
                    )

                except (ValueError, KeyError, RuntimeError) as e:
                    logger.error(f"Error inferring b2_ever_h6 for model_fallback bins: {e}")
                    logger.warning(
                        "model_fallback bins will have NaN risk — downstream aggregation "
                        "will treat them as zero risk."
                    )

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

                    # Clip inferred risk to prevent unbounded extrapolation
                    observed_risk = agg_data["b2_ever_h6_tmp"].dropna()
                    if len(observed_risk) > 0:
                        risk_floor = float(observed_risk.min())
                        risk_ceil_base = float(observed_risk.max())

                        risk_ceil_scaled = risk_ceil_base * settings.mr_extrapolation_risk_multiplier
                        risk_ceil = min(risk_ceil_scaled, settings.mr_extrapolation_hard_cap)

                        missing_bins_df["b2_ever_h6"] = missing_bins_df["b2_ever_h6"].clip(
                            lower=risk_floor, upper=risk_ceil
                        )
                        logger.info(
                            f"  Clipped model-imputed risk to [{risk_floor:.4f}, {risk_ceil:.4f}] "
                            f"(base max={risk_ceil_base:.4f}, mult={settings.mr_extrapolation_risk_multiplier}, "
                            f"cap={settings.mr_extrapolation_hard_cap})"
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

                # 2. Predict on bin sums (clip to >= 0: exposure cannot be negative)
                raw_preds = reg_todu_amt_pile.predict(bin_sums[["oa_amt"]])
                n_negative = (raw_preds < 0).sum()
                if n_negative > 0:
                    logger.warning(
                        f"reg_todu_amt_pile predicted {n_negative}/{len(raw_preds)} negative "
                        f"todu_amt_pile_h6 values (min={raw_preds.min():.2f}). Clipping to 0."
                    )
                bin_sums["todu_amt_pile_h6_bin"] = np.clip(raw_preds, 0, None)

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
            per_bin_stress=per_bin_stress,
            per_bin_tasa_fin=per_bin_tasa_fin,
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

        if len(VARIABLES) == 1:
            logger.info("Generating b2_ever_h6 bar chart for MR dataset (1-variable)...")
            var0 = VARIABLES[0]
            data_bar = data_surf_mr.sort_values(var0)

            fig_mr = go.Figure()
            fig_mr.add_trace(
                go.Bar(
                    x=data_bar[var0].astype(str),
                    y=data_bar["b2_ever_h6"],
                    marker_color="indianred",
                    text=data_bar["b2_ever_h6"].round(2),
                    textposition="outside",
                )
            )
            styles.apply_plotly_style(
                fig_mr,
                title=f"B2 Ever H6 by {var0} (MR Period){file_suffix}",
                width=900,
                height=500,
            )
            fig_mr.update_layout(xaxis_title=var0, yaxis_title="b2_ever_h6 (%)")

            output_plot_path_mr = output.mr_b2_visualization_html(file_suffix)
            fig_mr.write_html(output_plot_path_mr)
            logger.info(f"MR 1D visualization saved to {output_plot_path_mr}")

        elif len(VARIABLES) == 2:
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
