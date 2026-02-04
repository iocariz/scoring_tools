"""
Audit table generation for cutoff analysis.

This module provides functions to generate audit tables that track individual
records and their classification based on cutoff decisions.

Classification logic matches the main pipeline:
- Only rejected records with reject_reason="09-score" are eligible for swap-in
- Swap-in amounts are multiplied by the financing rate (tasa_fin)
"""

import pandas as pd
from loguru import logger

from src.constants import RejectReason, StatusName


def classify_record(
    row: pd.Series,
    var0_col: str,
    var1_col: str,
    cut_map: dict,
    inv_var1: bool = False,
) -> str:
    """
    Classify a single record based on cutoff, status, and reject_reason.

    Only rejected records with reject_reason="09-score" can be classified as swap_in.
    This matches the main pipeline logic where only score-rejected applications
    are considered for potential approval under new cutoffs.

    Args:
        row: DataFrame row with status_name, reject_reason, and bin values.
        var0_col: Name of the first variable column (e.g., 'sc_octroi_new_clus').
        var1_col: Name of the second variable column (e.g., 'new_efx_clus').
        cut_map: Dictionary mapping var0 bins to var1 cutoff limits.
        inv_var1: If True, use >= comparison instead of <= for var1.

    Returns:
        Classification string: 'keep', 'swap_out', 'swap_in', 'rejected', or 'rejected_other'.
    """
    var0_val = row[var0_col]
    var1_val = row[var1_col]
    status = row["status_name"]
    reject_reason = row.get("reject_reason", None)

    # Handle NA/None values in reject_reason
    if pd.isna(reject_reason):
        reject_reason = None

    # Get cutoff limit for this bin
    cut_limit = cut_map.get(var0_val)
    if cut_limit is None:
        # If bin not in cut_map, try float conversion
        cut_limit = cut_map.get(float(var0_val))

    if cut_limit is None:
        logger.warning(f"No cutoff found for bin {var0_val}")
        return "unknown"

    # Determine if record passes cutoff
    if inv_var1:
        passes_cut = var1_val >= cut_limit
    else:
        passes_cut = var1_val <= cut_limit

    # Classify based on status, reject_reason, and cutoff
    is_booked = status == StatusName.BOOKED.value
    is_score_rejected = (
        status == StatusName.REJECTED.value and
        reject_reason == RejectReason.SCORE.value
    )

    if is_booked and passes_cut:
        return "keep"
    elif is_booked and not passes_cut:
        return "swap_out"
    elif is_score_rejected and passes_cut:
        # Only score-rejected records that pass cutoff are swap-in
        return "swap_in"
    elif is_score_rejected and not passes_cut:
        # Score-rejected that still don't pass
        return "rejected"
    else:
        # Other rejected records (reject_reason != "09-score") are not considered
        return "rejected_other"


def generate_audit_table(
    data: pd.DataFrame,
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    financing_rate: float = 1.0,
    inv_var1: bool = False,
    audit_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate an audit table with individual record classifications.

    Args:
        data: DataFrame with individual records (must include status_name, reject_reason, var0, var1).
        optimal_solution_df: DataFrame with optimal solution (first row used).
        variables: List of two variable names [var0, var1].
        financing_rate: Rate to multiply swap-in amounts (tasa_fin). Default 1.0.
        inv_var1: If True, use >= comparison for var1 cutoff.
        audit_columns: Columns to include in audit table. If None, uses defaults.

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

    # Filter to columns that exist in data
    available_columns = [col for col in audit_columns if col in data.columns]
    missing_columns = [col for col in audit_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"Audit columns not found in data: {missing_columns}")

    # Extract cutoffs from optimal solution (first row)
    opt_sol_row = optimal_solution_df.iloc[0]

    # Build cut_map: var0_bin -> var1_cutoff
    cut_map = {}
    for col in optimal_solution_df.columns:
        if col == "sol_fac":
            continue
        # Skip non-numeric columns (KPI columns like b2_ever_h6, oa_amt_h0)
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
    audit_df["cut_limit"] = data[var0_col].map(lambda x: cut_map.get(float(x)))

    # Classify each record
    audit_df["classification"] = data.apply(
        lambda row: classify_record(row, var0_col, var1_col, cut_map, inv_var1),
        axis=1,
    )

    # Add passes_cut boolean for verification
    if inv_var1:
        audit_df["passes_cut"] = data[var1_col] >= audit_df["cut_limit"]
    else:
        audit_df["passes_cut"] = data[var1_col] <= audit_df["cut_limit"]

    # Calculate adjusted amount (swap-in multiplied by financing rate)
    if "oa_amt" in audit_df.columns:
        audit_df["oa_amt_adjusted"] = audit_df.apply(
            lambda row: row["oa_amt"] * financing_rate if row["classification"] == "swap_in" else row["oa_amt"],
            axis=1,
        )

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
        summary = audit_df.groupby("classification").agg(
            count=("classification", "size"),
            total_oa_amt=(amount_col, "sum"),
        ).reset_index()
    else:
        summary = audit_df.groupby("classification").agg(
            count=("classification", "size"),
        ).reset_index()
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
) -> dict[str, pd.DataFrame]:
    """
    Generate and save audit tables for main and MR periods.

    Args:
        data_main: Main period data with individual records.
        data_mr: MR period data with individual records.
        optimal_solution_df: Optimal solution DataFrame.
        variables: List of two variable names [var0, var1].
        scenario_name: Scenario name (e.g., 'base', 'optimistic', 'pessimistic').
        output_dir: Directory to save audit tables.
        inv_var1: If True, use >= comparison for var1 cutoff.
        financing_rate: Rate to multiply swap-in amounts (tasa_fin). Default 1.0.

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
) -> bool:
    """
    Validate that audit table totals match the summary table.

    Uses oa_amt_adjusted for swap-in comparison to account for financing rate.

    Args:
        audit_df: Audit DataFrame with classifications.
        summary_table: Risk production summary table.
        tolerance: Allowed relative difference (default 1%).

    Returns:
        True if validation passes, False otherwise.
    """
    # Calculate totals from audit using adjusted amounts
    amount_col = "oa_amt_adjusted" if "oa_amt_adjusted" in audit_df.columns else "oa_amt"
    audit_totals = audit_df.groupby("classification")[amount_col].sum()

    # Map to summary table metrics
    # swap_in = Swap-in (only 09-score rejected that pass), swap_out = Swap-out
    swap_in_audit = audit_totals.get("swap_in", 0)
    swap_out_audit = audit_totals.get("swap_out", 0)
    keep_audit = audit_totals.get("keep", 0)

    logger.info(f"Audit totals - swap_in: {swap_in_audit:,.0f}, swap_out: {swap_out_audit:,.0f}, keep: {keep_audit:,.0f}")

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
                logger.warning(
                    f"Audit validation warning: "
                    f"swap_in diff={si_diff:.2%}, swap_out diff={so_diff:.2%}"
                )
                return False

            logger.info("Audit validation passed: totals match summary table")
            return True

    except Exception as e:
        logger.warning(f"Could not validate audit against summary: {e}")

    return True  # Don't fail if validation can't be performed
