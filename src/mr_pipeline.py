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

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from src import styles
from src.constants import StatusName
from src.inference_optimized import run_optimization_pipeline
from src.models import calculate_B2
from src.preprocess_improved import filter_by_date
from src.stability import compare_main_vs_mr
from src.utils import calculate_b2_ever_h6


def calculate_metrics_from_cuts(
    data_summary_desagregado: pd.DataFrame, optimal_solution_df: pd.DataFrame | None, variables: list[str]
) -> pd.DataFrame | None:
    """
    Generates the Risk Production Summary Table by applying optimal cuts to aggregated data.
    """
    try:
        var0_col = variables[0]
        var1_col = variables[1]

        # Verify we have the optimal solution
        if optimal_solution_df is None or optimal_solution_df.empty:
            logger.warning("optimal_solution_df is missing or empty. Cannot calculate summary table.")
            return None

        opt_sol_row = optimal_solution_df.iloc[0]  # Take the first (selected) solution

        # Get unique bins from data
        bins = sorted(data_summary_desagregado[var0_col].unique())
        cut_map = {}

        for bin_val in bins:
            # Try to find the column matching the bin value
            if bin_val in optimal_solution_df.columns:
                cut_map[bin_val] = opt_sol_row[bin_val]
            elif str(bin_val) in optimal_solution_df.columns:
                cut_map[bin_val] = opt_sol_row[str(bin_val)]
            else:
                logger.warning(f"Warning: Bin {bin_val} not found in optimal solution columns. Using default/max.")
                cut_map[bin_val] = np.inf

        logger.info(f"Optimal Cuts: {cut_map}")

        # 2. Apply cuts
        df = data_summary_desagregado.copy()

        # Vectorized mapping of cuts
        df["cut_limit"] = df[var0_col].map(cut_map)
        df["passes_cut"] = df[var1_col] <= df["cut_limit"]

        summary_data = []

        # Helper to calc metrics from a filtered subset
        def calc_metrics(subset, suffix):
            prod = subset[f"oa_amt_h0{suffix}"].sum()
            risk_num = subset[f"todu_30ever_h6{suffix}"].sum()
            risk_den = subset[f"todu_amt_pile_h6{suffix}"].sum()
            b2_ever = float(np.nan_to_num(calculate_b2_ever_h6(risk_num, risk_den)))
            return prod, b2_ever, risk_num, risk_den

        # Actual (All Booked)
        actual_prod, actual_risk, actual_rn, actual_rd = calc_metrics(df, "_boo")
        summary_data.append(
            {
                "Metric": "Actual",
                "Risk (%)": actual_risk,
                "Production (€)": actual_prod,
                "Production (%)": 1.0,
                "todu_30ever_h6": actual_rn,
                "todu_amt_pile_h6": actual_rd,
            }
        )

        # Swap-in (Repesca that passes)
        swap_in_df = df[df["passes_cut"]]
        si_prod, si_risk, si_rn, si_rd = calc_metrics(swap_in_df, "_rep")
        summary_data.append(
            {
                "Metric": "Swap-in",
                "Risk (%)": si_risk,
                "Production (€)": si_prod,
                "Production (%)": si_prod / actual_prod if actual_prod else 0,
                "todu_30ever_h6": si_rn,
                "todu_amt_pile_h6": si_rd,
            }
        )

        # Swap-out (Booked that fails)
        swap_out_df = df[~df["passes_cut"]]
        so_prod, so_risk, so_rn, so_rd = calc_metrics(swap_out_df, "_boo")
        summary_data.append(
            {
                "Metric": "Swap-out",
                "Risk (%)": so_risk,
                "Production (€)": so_prod,
                "Production (%)": so_prod / actual_prod if actual_prod else 0,
                "todu_30ever_h6": so_rn,
                "todu_amt_pile_h6": so_rd,
            }
        )

        # Optimum
        opt_prod = (actual_prod - so_prod) + si_prod
        opt_rn = (actual_rn - so_rn) + si_rn
        opt_rd = (actual_rd - so_rd) + si_rd
        opt_risk = float(np.nan_to_num(calculate_b2_ever_h6(opt_rn, opt_rd)))

        summary_data.append(
            {
                "Metric": "Optimum selected",
                "Risk (%)": opt_risk,
                "Production (€)": opt_prod,
                "Production (%)": opt_prod / actual_prod if actual_prod else 0,
                "todu_30ever_h6": opt_rn,
                "todu_amt_pile_h6": opt_rd,
            }
        )

        return pd.DataFrame(summary_data)

    except Exception as e:
        logger.error(f"Error calculating metrics from cuts: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def process_mr_period(
    data_clean: pd.DataFrame,
    data_booked: pd.DataFrame,
    config_data: dict[str, Any],
    risk_inference: dict[str, Any],
    reg_todu_amt_pile: Any,
    stress_factor: float,
    tasa_fin: float,
    annual_coef: float,
    optimal_solution_df: pd.DataFrame | None = None,
    file_suffix: str = "",
) -> None:
    """
    Process the MR period data: filtering, inference, aggregation, visualization, and summary table.
    """
    logger.info(f"Processing MR period data (suffix: '{file_suffix}')...")

    if not (config_data.get("date_ini_book_obs_mr") and config_data.get("date_fin_book_obs_mr")):
        logger.warning("MR dates not configured. Skipping MR period processing.")
        return

    try:
        indicators_mr = ["acct_booked_h0", "oa_amt", "oa_amt_h0"]
        # Ensure merge keys (variables) are included
        merge_keys = config_data.get("variables", ["sc_octroi_new_clus", "new_efx_clus"])
        mr_cols = config_data["keep_vars"] + indicators_mr + merge_keys

        # Create data_demand_mr (filter by date and select columns)
        data_mr_period = filter_by_date(
            data_clean, "mis_date", config_data["date_ini_book_obs_mr"], config_data["date_fin_book_obs_mr"]
        )

        available_mr_cols = [c for c in mr_cols if c in data_mr_period.columns]
        available_mr_cols = list(dict.fromkeys(available_mr_cols))

        data_demand_mr = data_mr_period[available_mr_cols].copy()

        # --- Calculate b2_ever_h6_tmp from initial period (data_booked) ---
        logger.info(f"Calculating b2_ever_h6_tmp aggregated by {merge_keys} from initial period...")

        required_agg_cols = merge_keys + ["todu_30ever_h6", "todu_amt_pile_h6"]
        if all(col in data_booked.columns for col in required_agg_cols):
            agg_data = data_booked.groupby(merge_keys)[["todu_30ever_h6", "todu_amt_pile_h6"]].sum().reset_index()

            # Calculate b2_ever_h6_tmp
            agg_data["b2_ever_h6_tmp"] = calculate_b2_ever_h6(
                agg_data["todu_30ever_h6"], agg_data["todu_amt_pile_h6"]
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
                for _, row in missing_bins.iterrows():
                    logger.warning(f"  Missing bin: {dict(row)}")

                # Use inference model to predict b2_ever_h6 for missing bins
                try:
                    final_model = risk_inference["best_model_info"]["model"]
                    final_features = risk_inference["features"]

                    # Create a DataFrame with missing bin combinations for prediction
                    missing_bins_df = missing_bins.copy()

                    # Apply calculate_B2 to predict b2_ever_h6 for missing bins
                    missing_bins_df = calculate_B2(
                        missing_bins_df, final_model, merge_keys, stress_factor, final_features
                    )

                    # Merge inferred values back into agg_data
                    inferred_b2 = missing_bins_df[merge_keys + ["b2_ever_h6"]].rename(
                        columns={"b2_ever_h6": "b2_ever_h6_inferred"}
                    )

                    # Merge inferred values into data_demand_mr
                    data_demand_mr = pd.merge(data_demand_mr, inferred_b2, on=merge_keys, how="left")

                    # Fill missing b2_ever_h6_tmp with inferred values
                    fill_mask = data_demand_mr["b2_ever_h6_tmp"].isna() & data_demand_mr["b2_ever_h6_inferred"].notna()
                    data_demand_mr.loc[fill_mask, "b2_ever_h6_tmp"] = data_demand_mr.loc[
                        fill_mask, "b2_ever_h6_inferred"
                    ]

                    # Drop the helper column
                    data_demand_mr = data_demand_mr.drop(columns=["b2_ever_h6_inferred"], errors="ignore")

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

                except Exception as e:
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
        logger.info("Calculating todu_amt_pile_h6 for booked accounts in MR period...")

        data_demand_mr["todu_amt_pile_h6"] = np.nan
        booked_mask = (data_demand_mr["status_name"] == StatusName.BOOKED.value) & (data_demand_mr["oa_amt"].notna())

        if booked_mask.any():
            X_pred = data_demand_mr.loc[booked_mask, ["oa_amt"]]
            try:
                preds = reg_todu_amt_pile.predict(X_pred)
                data_demand_mr.loc[booked_mask, "todu_amt_pile_h6"] = preds
            except Exception as e:
                logger.error(f"Error predicting todu_amt_pile_h6: {e}")
        else:
            logger.warning("No booked accounts with valid oa_amt found for prediction.")

        # --- Calculate todu_30ever_h6 ---
        logger.info("Calculating todu_30ever_h6 for booked accounts in MR period...")
        data_demand_mr["todu_30ever_h6"] = np.nan

        calc_mask = data_demand_mr["todu_amt_pile_h6"].notna() & data_demand_mr["b2_ever_h6_tmp"].notna()

        if calc_mask.any():
            data_demand_mr.loc[calc_mask, "todu_30ever_h6"] = (
                data_demand_mr.loc[calc_mask, "todu_amt_pile_h6"] * data_demand_mr.loc[calc_mask, "b2_ever_h6_tmp"]
            ) / 7

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
            config_data=config_data,
            annual_coef=annual_coef,
        )

        # Save MR summary
        summary_path = f"data/data_summary_desagregado_mr{file_suffix}.csv"
        data_summary_desagregado_mr.to_csv(summary_path, index=False)
        logger.info(f"MR summary data saved to {summary_path}")

        # --- Visualize b2_ever_h6 for MR ---
        logger.info("Generating b2_ever_h6 visualization for MR dataset...")
        VARIABLES = config_data.get("variables", ["sc_octroi_new_clus", "new_efx_clus"])

        fig_mr = go.Figure()
        data_surf_mr = data_summary_desagregado_mr.copy()

        data_surf_mr["b2_ever_h6"] = calculate_b2_ever_h6(
            data_surf_mr["todu_30ever_h6"], data_surf_mr["todu_amt_pile_h6"]
        )

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

        output_plot_path_mr = f"images/b2_ever_h6_vs_octroi_and_risk_score_mr{file_suffix}.html"
        fig_mr.write_html(output_plot_path_mr)
        logger.info(f"MR Visualization saved to {output_plot_path_mr}")

        # --- Cleanup ---
        if "b2_ever_h6_tmp" in data_demand_mr.columns:
            logger.info("Dropping b2_ever_h6_tmp from data_demand_mr and data_booked_mr...")
            data_demand_mr = data_demand_mr.drop(columns=["b2_ever_h6_tmp"], errors="ignore")
            data_booked_mr = data_booked_mr.drop(columns=["b2_ever_h6_tmp"], errors="ignore")

        # --- Generate Risk Production Summary Table for MR ---
        logger.info("Generating Risk Production Summary Table for MR period...")

        mr_summary_table = calculate_metrics_from_cuts(data_summary_desagregado_mr, optimal_solution_df, VARIABLES)

        if mr_summary_table is not None:
            mr_summary_path = f"data/risk_production_summary_table_mr{file_suffix}.csv"
            mr_summary_table.to_csv(mr_summary_path, index=False)
            logger.info(f"MR Risk Production Summary Table saved to {mr_summary_path}")
            logger.info(f"MR Table:\n{mr_summary_table.to_string()}")

        # --- Calculate PSI/CSI Stability Metrics ---
        logger.info("Calculating PSI/CSI stability metrics (Main vs MR)...")
        try:
            # Get numeric columns for stability analysis
            stability_vars = []
            for col in data_booked.columns:
                if col in data_booked_mr.columns:
                    if pd.api.types.is_numeric_dtype(data_booked[col]):
                        # Exclude ID columns and date columns
                        if not any(x in col.lower() for x in ["id", "date", "mis_"]):
                            stability_vars.append(col)

            # Include key score/cluster variables
            key_vars = ["sc_octroi", "new_efx", "oa_amt", "risk_score_rf"]
            stability_vars = list(set(stability_vars + [v for v in key_vars if v in data_booked.columns]))

            if stability_vars:
                # Determine main score variable for overall PSI
                score_var = "sc_octroi" if "sc_octroi" in stability_vars else stability_vars[0]

                stability_report = compare_main_vs_mr(
                    main_df=data_booked,
                    mr_df=data_booked_mr,
                    variables=stability_vars,
                    score_variable=score_var,
                    output_path=f"images/stability_report{file_suffix}.html",
                    verbose=True,
                )

                # Save stability results to CSV
                stability_df = stability_report.to_dataframe()
                stability_csv_path = f"data/stability_psi{file_suffix}.csv"
                stability_df.to_csv(stability_csv_path, index=False)
                logger.info(f"Stability metrics saved to {stability_csv_path}")

                # Log summary
                if stability_report.unstable_vars:
                    logger.warning(
                        f"STABILITY WARNING: {len(stability_report.unstable_vars)} variables "
                        f"show significant drift (PSI >= 0.25): "
                        f"{[r.variable for r in stability_report.unstable_vars]}"
                    )
            else:
                logger.warning("No numeric variables found for stability analysis")

        except Exception as e:
            logger.error(f"Error calculating stability metrics: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error processing MR period: {e}")
        import traceback

        logger.error(traceback.format_exc())
