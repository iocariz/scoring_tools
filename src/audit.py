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


def primary_amount_column(df: pd.DataFrame) -> str:
    """Column used for production amounts in audit: prefer ``oa_amt_h0`` (grid-aligned), else ``oa_amt``."""
    if "oa_amt_h0" in df.columns:
        return "oa_amt_h0"
    return "oa_amt"


def audit_production_kpis(audit_df: pd.DataFrame | None) -> dict[str, float]:
    """
    Production (€) totals from an audit extract — same basis as ``reconcile_risk_production_summary_with_audit``.

    Uses ``oa_amt_adjusted`` when present, else ``primary_amount_column`` (``oa_amt_h0`` or ``oa_amt``).
    """
    empty = {"actual": 0.0, "swap_in": 0.0, "swap_out": 0.0, "optimum": 0.0, "total_demand": 0.0}
    if audit_df is None or audit_df.empty or "classification" not in audit_df.columns:
        return empty.copy()
    if "oa_amt_adjusted" in audit_df.columns:
        amt = "oa_amt_adjusted"
    else:
        amt = primary_amount_column(audit_df)
    if amt not in audit_df.columns:
        return empty.copy()
    cls = audit_df["classification"].astype(str).str.strip()
    swap_in = float(audit_df.loc[cls == "swap_in", amt].sum())
    swap_out = float(audit_df.loc[cls == "swap_out", amt].sum())
    if "status_name" in audit_df.columns:
        st = audit_df["status_name"].astype(str).str.strip()
        booked = st == StatusName.BOOKED.value
        actual = float(audit_df.loc[booked, amt].sum())
    else:
        actual = 0.0
    optimum = actual - swap_out + swap_in
    total_demand = _total_demand_from_audit_subset(audit_df, amt)
    return {
        "actual": actual,
        "swap_in": swap_in,
        "swap_out": swap_out,
        "optimum": optimum,
        "total_demand": total_demand,
    }


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
            Production amounts use ``oa_amt_h0`` when present, otherwise ``oa_amt``.
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
    if optimal_solution_df is None or optimal_solution_df.empty:
        logger.warning("optimal_solution_df is missing or empty. Cannot generate audit table.")
        return pd.DataFrame()

    var0_col = variables[0]
    var1_col = variables[1] if len(variables) > 1 else None

    # Default audit columns - now includes reject_reason
    if audit_columns is None:
        amt_col = primary_amount_column(data)
        audit_columns = [
            "authorization_id",
            "status_name",
            "reject_reason",
            "risk_score_rf",
            "score_rf",
            var0_col,
            amt_col,
            "income_t1t2_m",
            "todu_30ever_h6",
            "todu_amt_pile_h6",
        ]
        if var1_col is not None:
            audit_columns.insert(-1, var1_col)
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
        audit_df["decision_source"] = "mask"
    else:
        if var1_col is None:
            raise ValueError("2-var cut_map audit path requires at least 2 variables; use mask/grid for 1-var configs")
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
        audit_df["decision_source"] = "cutoff"

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

    # Human-readable key for the grid cell used in the decision
    if variables:
        key_parts = []
        for v in variables:
            if v in data.columns:
                key_parts.append(data[v].astype(str))
        if key_parts:
            cell_key = key_parts[0]
            for part in key_parts[1:]:
                cell_key = cell_key + "|" + part
            audit_df["cell_key"] = cell_key
        else:
            audit_df["cell_key"] = ""
    else:
        audit_df["cell_key"] = ""

    audit_df["passes_cut"] = passes_cut

    # Calculate annualization coefficient
    annual_coef = 12 / n_months if n_months and n_months > 0 else 1.0

    # Vectorized adjusted amount (annualization + financing on swap-in); base = oa_amt_h0 when present
    base_amt = primary_amount_column(audit_df)
    if base_amt in audit_df.columns:
        is_swap_in = audit_df["classification"] == "swap_in"
        audit_df["oa_amt_adjusted"] = audit_df[base_amt] * annual_coef
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
    if use_adjusted and "oa_amt_adjusted" in audit_df.columns:
        amount_col = "oa_amt_adjusted"
    else:
        amount_col = primary_amount_column(audit_df)

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


def _total_demand_from_audit_subset(audit_df: pd.DataFrame, amount_col: str) -> float:
    """Demand (€) for rejection-rate math: all non-canceled applications in *audit_df*."""
    if "status_name" not in audit_df.columns:
        return 0.0
    sn = audit_df["status_name"].astype(str).str.strip().str.lower()
    non_cancel = sn != StatusName.CANCELED.value.lower()
    return float(audit_df.loc[non_cancel, amount_col].sum())


def reconcile_risk_production_summary_with_audit(
    summary_table: pd.DataFrame,
    audit_df: pd.DataFrame | None,
    *,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Align risk production summary ``Production (€)`` and ``Production (%)`` with loan-level audit totals.

    The summary table is built from grid / frontier KPIs; the audit table sums ``oa_amt_h0`` / ``oa_amt``
    (or ``oa_amt_adjusted`` after annualization) per classification. Those can differ slightly when
    grid and loan-level paths diverge. This overwrites production rows so Excel/CSV summary
    matches ``audit_*.csv`` drill-down totals.
    """
    if audit_df is None or audit_df.empty or "classification" not in audit_df.columns:
        return summary_table

    if "oa_amt_adjusted" in audit_df.columns:
        amount_col = "oa_amt_adjusted"
    else:
        amount_col = primary_amount_column(audit_df)
    if amount_col not in audit_df.columns:
        return summary_table

    st = summary_table.copy()
    metric_col = "Metric"
    if metric_col not in st.columns:
        return summary_table

    m = st[metric_col].astype(str).str.strip()
    cls = audit_df["classification"].astype(str).str.strip()

    swap_in = float(audit_df.loc[cls == "swap_in", amount_col].sum())
    swap_out = float(audit_df.loc[cls == "swap_out", amount_col].sum())

    if "status_name" in audit_df.columns:
        status_s = audit_df["status_name"].astype(str).str.strip()
        booked_mask = status_s == StatusName.BOOKED.value
    else:
        booked_mask = pd.Series(False, index=audit_df.index)
    actual_prod = float(audit_df.loc[booked_mask, amount_col].sum())
    optimum_prod = actual_prod - swap_out + swap_in

    def _set_rows(metric_name: str, prod: float, pct: float | None) -> None:
        mask = m.str.lower() == metric_name.lower()
        if not mask.any():
            return
        if "Production (€)" in st.columns:
            st.loc[mask, "Production (€)"] = prod
        if pct is not None and "Production (%)" in st.columns:
            st.loc[mask, "Production (%)"] = pct

    if actual_prod > 0:
        _set_rows("Actual", actual_prod, 1.0)
        _set_rows("Swap-in", swap_in, swap_in / actual_prod)
        _set_rows("Swap-out", swap_out, swap_out / actual_prod)
        _set_rows("Optimum selected", optimum_prod, optimum_prod / actual_prod)
        delta = optimum_prod - actual_prod
        _set_rows("Summary", delta, delta / actual_prod)
    else:
        _set_rows("Actual", actual_prod, None)
        _set_rows("Swap-in", swap_in, None)
        _set_rows("Swap-out", swap_out, None)
        _set_rows("Optimum selected", optimum_prod, None)
        _set_rows("Summary", optimum_prod - actual_prod, None)

    # Total demand and rejection rates from the same audit subset (required for per-income-bin tables)
    td_col = next((c for c in st.columns if "total demand" in c.lower()), None)
    total_demand = _total_demand_from_audit_subset(audit_df, amount_col)
    if td_col and td_col in st.columns and total_demand > 0:
        act_row = m.str.lower() == "actual"
        if act_row.any():
            st.loc[act_row, td_col] = total_demand

    if total_demand > 0 and "Rejection Rate (%)" in st.columns:
        act_mask = m.str.lower() == "actual"
        opt_mask = m.str.lower() == "optimum selected"
        if act_mask.any():
            st.loc[act_mask, "Rejection Rate (%)"] = (1.0 - actual_prod / total_demand) * 100.0
        if opt_mask.any():
            st.loc[opt_mask, "Rejection Rate (%)"] = (1.0 - optimum_prod / total_demand) * 100.0
    elif td_col and td_col in st.columns:
        td_vals = st.loc[m.str.lower() == "actual", td_col]
        total_demand_fb = float(td_vals.values[0]) if len(td_vals) and pd.notna(td_vals.values[0]) else 0.0
        if total_demand_fb > 0 and "Rejection Rate (%)" in st.columns:
            act_mask = m.str.lower() == "actual"
            opt_mask = m.str.lower() == "optimum selected"
            if act_mask.any():
                st.loc[act_mask, "Rejection Rate (%)"] = (1.0 - actual_prod / total_demand_fb) * 100.0
            if opt_mask.any():
                st.loc[opt_mask, "Rejection Rate (%)"] = (1.0 - optimum_prod / total_demand_fb) * 100.0

    log_fn = logger.debug if silent else logger.info
    log_fn(
        f"Reconciled risk production summary with audit ({amount_col}): "
        f"actual={actual_prod:,.0f}, swap_out={swap_out:,.0f}, swap_in={swap_in:,.0f}, optimum={optimum_prod:,.0f}"
    )
    return st


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
    audit_main: pd.DataFrame | None = None,
    audit_mr: pd.DataFrame | None = None,
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
    if audit_main is None:
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
    else:
        logger.debug(f"Using precomputed main audit table for {scenario_name}")

    # Save main period audit
    main_path = f"{output_dir}/audit_{scenario_name}.csv"
    audit_main.to_csv(main_path, index=False)
    logger.info(f"Main period audit saved to {main_path}")
    results["main"] = audit_main

    # Log summary for main period
    summary_main = generate_audit_summary(audit_main)
    logger.info(f"Main period audit summary:\n{summary_main.to_string()}")

    # Generate audit for MR period
    if audit_mr is None:
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
    else:
        logger.debug(f"Using precomputed MR audit table for {scenario_name}")

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
        ``True`` if audit totals match the summary within tolerance.
        ``False`` if they diverge OR a precondition fails (missing
        Swap-in/Swap-out rows, NaN values, structural mismatch). The
        difference between "failed validation" and "could not validate"
        is logged — callers that need the distinction should inspect
        the logs rather than relying on a tri-state return.
    """
    # Calculate totals from audit using adjusted amounts
    if "oa_amt_adjusted" in audit_df.columns:
        amount_col = "oa_amt_adjusted"
    else:
        amount_col = primary_amount_column(audit_df)
    audit_totals = audit_df.groupby("classification")[amount_col].sum()

    # Map to summary table metrics
    # swap_in = Swap-in (only 09-score rejected that pass), swap_out = Swap-out
    swap_in_audit = audit_totals.get("swap_in", 0)
    swap_out_audit = audit_totals.get("swap_out", 0)
    keep_audit = audit_totals.get("keep", 0)

    logger.info(
        f"Audit totals - swap_in: {swap_in_audit:,.0f}, swap_out: {swap_out_audit:,.0f}, keep: {keep_audit:,.0f}"
    )

    try:
        swap_in_row = summary_table[summary_table["Metric"] == "Swap-in"]
        swap_out_row = summary_table[summary_table["Metric"] == "Swap-out"]
    except (KeyError, ValueError) as e:
        logger.warning(f"Audit validation failed: summary_table does not expose a Metric column ({e})")
        return False

    if swap_in_row.empty or swap_out_row.empty:
        logger.warning(
            "Audit validation failed: summary_table is missing Swap-in or Swap-out rows "
            "(empty={'swap_in': %s, 'swap_out': %s}). Treating as validation failure, not a silent skip."
            % (swap_in_row.empty, swap_out_row.empty)
        )
        return False

    swap_in_summary = swap_in_row["Production (€)"].values[0]
    swap_out_summary = swap_out_row["Production (€)"].values[0]

    if pd.isna(swap_in_summary) or pd.isna(swap_out_summary):
        logger.warning(
            f"Audit validation failed: summary_table contains NaN reference value(s) "
            f"(swap_in={swap_in_summary}, swap_out={swap_out_summary}). Cannot compare."
        )
        return False

    # Compare with tolerance
    si_diff = abs(swap_in_audit - swap_in_summary) / max(swap_in_summary, 1)
    so_diff = abs(swap_out_audit - swap_out_summary) / max(swap_out_summary, 1)

    if si_diff > tolerance or so_diff > tolerance:
        logger.warning(f"Audit validation warning: swap_in diff={si_diff:.2%}, swap_out diff={so_diff:.2%}")
        return False

    logger.info("Audit validation passed: totals match summary table")
    return True
