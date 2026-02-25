"""
Audit table generation for cutoff analysis.

This module provides functions to generate audit tables that track individual
records and their classification based on cutoff decisions.

Classification logic matches the main pipeline:
- Only rejected records with reject_reason="09-score" are eligible for swap-in
- Swap-in amounts are multiplied by the financing rate (tasa_fin)
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import RejectReason, StatusName


def generate_audit_table(
    data: pd.DataFrame,
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    financing_rate: float = 1.0,
    inv_var1: bool = False,
    audit_columns: list[str] | None = None,
    n_months: int | None = None,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> pd.DataFrame:
    """
    Generate an audit table with individual record classifications.

    Args:
        data: DataFrame with individual records (must include status_name, reject_reason, var0, var1).
        optimal_solution_df: DataFrame with optimal solution (first row used).
        variables: List of variable names [var0, var1, ...].
        financing_rate: Rate to multiply swap-in amounts (tasa_fin). Default 1.0.
        inv_var1: If True, use >= comparison for var1 cutoff (2-var path only).
        audit_columns: Columns to include in audit table. If None, uses defaults.
        n_months: Number of months in the period. Used to annualize amounts (12/n_months).
        mask: Optional binary acceptance mask for N-d classify_by_mask path.
        grid: Optional CellGrid for N-d classify_by_mask path.

    Returns:
        DataFrame with audit information for each record.
    """
    var0_col = variables[0]
    var1_col = variables[1]

    # Default audit columns - now includes reject_reason
    if audit_columns is None:
        audit_columns = [
            "authorization_id",
            "status_name",
            "reject_reason",
            "risk_score_rf",
            "score_rf",
            var1_col,
            var0_col,
            "oa_amt",
        ]
        # Include extra variables for N>2
        for v in variables[2:]:
            if v not in audit_columns:
                audit_columns.append(v)

    # Filter to columns that exist in data
    available_columns = [col for col in audit_columns if col in data.columns]
    missing_columns = [col for col in audit_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"Audit columns not found in data: {missing_columns}")

    # Determine passes_cut via mask (N-d) or cut_map (2-var)
    if mask is not None and grid is not None:
        from src.optimization_utils import classify_by_mask

        passes_cut = classify_by_mask(data, mask, grid)
        # Create audit DataFrame
        audit_df = data[available_columns].copy()
        audit_df["cut_limit"] = np.nan  # not applicable for N-d
    else:
        # Extract cutoffs from optimal solution (first row)
        opt_sol_row = optimal_solution_df.iloc[0]

        # Build cut_map: var0_bin -> var1_cutoff
        cut_map = {}
        for col in optimal_solution_df.columns:
            if col == "sol_fac":
                continue
            try:
                bin_val = float(col)
                cut_map[bin_val] = opt_sol_row[col]
            except (ValueError, TypeError):
                continue

        if not cut_map:
            raise ValueError("No valid cutoff bins found in optimal_solution_df")

        logger.info(f"Cutoff map: {cut_map}")

        # Create audit DataFrame
        audit_df = data[available_columns].copy()

        # Add cutoff limit for each record
        audit_df["cut_limit"] = data[var0_col].astype(float).map(cut_map)

        # Vectorized classification
        if inv_var1:
            passes_cut = data[var1_col] >= audit_df["cut_limit"]
        else:
            passes_cut = data[var1_col] <= audit_df["cut_limit"]

    is_booked = data["status_name"] == StatusName.BOOKED.value
    if "reject_reason" in data.columns:
        reject_reason_col = data["reject_reason"].astype(object).fillna("").astype(str)
    else:
        reject_reason_col = pd.Series("", index=data.index)
    is_score_rejected = (data["status_name"] == StatusName.REJECTED.value) & (
        reject_reason_col == RejectReason.SCORE.value
    )

    audit_df["classification"] = np.select(
        [
            is_booked & passes_cut,
            is_booked & ~passes_cut,
            is_score_rejected & passes_cut,
            is_score_rejected & ~passes_cut,
        ],
        ["keep", "swap_out", "swap_in", "rejected"],
        default="rejected_other",
    )

    # Handle rows with no cutoff found (2-var path only)
    if mask is None:
        no_cutoff = audit_df["cut_limit"].isna()
        if no_cutoff.any():
            logger.warning(f"{no_cutoff.sum()} records had no matching cutoff bin — classified as 'unknown'.")
            audit_df.loc[no_cutoff, "classification"] = "unknown"

    audit_df["passes_cut"] = passes_cut

    # Calculate annualization coefficient
    annual_coef = 12 / n_months if n_months else 1.0

    # Vectorized adjusted amount (replaces row-by-row apply)
    if "oa_amt" in audit_df.columns:
        is_swap_in = audit_df["classification"] == "swap_in"
        audit_df["oa_amt_adjusted"] = audit_df["oa_amt"] * annual_coef
        audit_df.loc[is_swap_in, "oa_amt_adjusted"] *= financing_rate

    logger.info(f"Annualization: {n_months} months -> coefficient {annual_coef:.4f}")
    logger.info(f"Financing rate applied to swap-in: {financing_rate:.2%}")

    return audit_df


def generate_audit_summary(audit_df: pd.DataFrame, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Generate summary statistics from audit table.

    Args:
        audit_df: Audit DataFrame with classification column.
        use_adjusted: If True, use oa_amt_adjusted for totals. Default True.

    Returns:
        Summary DataFrame with counts and amounts by classification.
    """
    amount_col = "oa_amt_adjusted" if use_adjusted and "oa_amt_adjusted" in audit_df.columns else "oa_amt"

    if amount_col in audit_df.columns:
        summary = (
            audit_df.groupby("classification")
            .agg(
                count=("classification", "size"),
                total_oa_amt=(amount_col, "sum"),
            )
            .reset_index()
        )
    else:
        summary = (
            audit_df.groupby("classification")
            .agg(
                count=("classification", "size"),
            )
            .reset_index()
        )
        summary["total_oa_amt"] = 0

    return summary


def save_audit_tables(
    data_main: pd.DataFrame,
    data_mr: pd.DataFrame,
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    scenario_name: str,
    output_dir: str = "data",
    inv_var1: bool = False,
    financing_rate: float = 1.0,
    n_months_main: int | None = None,
    n_months_mr: int | None = None,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Generate and save audit tables for main and MR periods.

    Args:
        data_main: Main period data with individual records.
        data_mr: MR period data with individual records.
        optimal_solution_df: Optimal solution DataFrame.
        variables: List of variable names [var0, var1, ...].
        scenario_name: Scenario name (e.g., 'base', 'optimistic', 'pessimistic').
        output_dir: Directory to save audit tables.
        inv_var1: If True, use >= comparison for var1 cutoff (2-var path only).
        financing_rate: Rate to multiply swap-in amounts (tasa_fin). Default 1.0.
        n_months_main: Number of months in main period for annualization.
        n_months_mr: Number of months in MR period for annualization.
        mask: Optional binary acceptance mask for N-d classify_by_mask path.
        grid: Optional CellGrid for N-d classify_by_mask path.

    Returns:
        Dictionary with audit DataFrames for main and MR periods.
    """
    results = {}

    # Generate audit for main period
    logger.info(f"Generating audit table for main period - {scenario_name}")
    audit_main = generate_audit_table(
        data=data_main,
        optimal_solution_df=optimal_solution_df,
        variables=variables,
        financing_rate=financing_rate,
        inv_var1=inv_var1,
        n_months=n_months_main,
        mask=mask,
        grid=grid,
    )

    # Save main period audit
    main_path = f"{output_dir}/audit_{scenario_name}.csv"
    audit_main.to_csv(main_path, index=False)
    logger.info(f"Main period audit saved to {main_path}")
    results["main"] = audit_main

    # Log summary for main period
    summary_main = generate_audit_summary(audit_main)
    logger.info(f"Main period audit summary:\n{summary_main.to_string()}")

    # Generate audit for MR period
    logger.info(f"Generating audit table for MR period - {scenario_name}")
    audit_mr = generate_audit_table(
        data=data_mr,
        optimal_solution_df=optimal_solution_df,
        variables=variables,
        financing_rate=financing_rate,
        inv_var1=inv_var1,
        n_months=n_months_mr,
        mask=mask,
        grid=grid,
    )

    # Save MR period audit
    mr_path = f"{output_dir}/audit_{scenario_name}_mr.csv"
    audit_mr.to_csv(mr_path, index=False)
    logger.info(f"MR period audit saved to {mr_path}")
    results["mr"] = audit_mr

    # Log summary for MR period
    summary_mr = generate_audit_summary(audit_mr)
    logger.info(f"MR period audit summary:\n{summary_mr.to_string()}")

    return results


def validate_audit_against_summary(
    audit_df: pd.DataFrame,
    summary_table: pd.DataFrame,
    tolerance: float = 0.01,
) -> bool | None:
    """
    Validate that audit table totals match the summary table.

    Uses oa_amt_adjusted for swap-in comparison to account for financing rate.

    Args:
        audit_df: Audit DataFrame with classifications.
        summary_table: Risk production summary table.
        tolerance: Allowed relative difference (default 1%).

    Returns:
        True if validation passes, False if it fails, None if validation
        could not be performed (e.g., missing columns).
    """
    # Calculate totals from audit using adjusted amounts
    amount_col = "oa_amt_adjusted" if "oa_amt_adjusted" in audit_df.columns else "oa_amt"
    audit_totals = audit_df.groupby("classification")[amount_col].sum()

    # Map to summary table metrics
    # swap_in = Swap-in (only 09-score rejected that pass), swap_out = Swap-out
    swap_in_audit = audit_totals.get("swap_in", 0)
    swap_out_audit = audit_totals.get("swap_out", 0)
    keep_audit = audit_totals.get("keep", 0)

    logger.info(
        f"Audit totals - swap_in: {swap_in_audit:,.0f}, swap_out: {swap_out_audit:,.0f}, keep: {keep_audit:,.0f}"
    )

    # Try to extract from summary table
    try:
        swap_in_row = summary_table[summary_table["Metric"] == "Swap-in"]
        swap_out_row = summary_table[summary_table["Metric"] == "Swap-out"]

        if not swap_in_row.empty and not swap_out_row.empty:
            swap_in_summary = swap_in_row["Production (€)"].values[0]
            swap_out_summary = swap_out_row["Production (€)"].values[0]

            # Compare with tolerance
            si_diff = abs(swap_in_audit - swap_in_summary) / max(swap_in_summary, 1)
            so_diff = abs(swap_out_audit - swap_out_summary) / max(swap_out_summary, 1)

            if si_diff > tolerance or so_diff > tolerance:
                logger.warning(f"Audit validation warning: swap_in diff={si_diff:.2%}, swap_out diff={so_diff:.2%}")
                return False

            logger.info("Audit validation passed: totals match summary table")
            return True

    except (KeyError, ValueError) as e:
        logger.warning(f"Could not validate audit against summary: {e}")
        return None
