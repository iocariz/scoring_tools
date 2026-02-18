"""
Optimization utilities for credit risk scoring.

This module contains functions for generating feasible solutions,
calculating KPIs, and finding Pareto-optimal cutoffs.
"""

import gc
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from .constants import DEFAULT_RISK_MULTIPLIER
from .utils import MAX_PARALLEL_JOBS, calculate_b2_ever_h6, optimize_dtypes


def _build_cutoff_dataframe(
    var0_bins: list,
    var1_cutoffs: list,
    values_var0: list | np.ndarray,
) -> pd.DataFrame:
    """Construct single-row DataFrame with sol_fac + bin columns."""
    solution_data = {"sol_fac": [0]}
    bin_to_cutoff = {float(b): c for b, c in zip(var0_bins, var1_cutoffs)}

    for bin_val in values_var0:
        cutoff_val = bin_to_cutoff.get(float(bin_val))
        if cutoff_val is None:
            raise ValueError(
                f"No cutoff defined for bin {bin_val}. Fixed cutoffs must cover all data bins: {list(values_var0)}"
            )
        solution_data[bin_val] = [cutoff_val]

    return pd.DataFrame(solution_data)


def _validate_cutoff_dict_structure(
    fixed_cutoffs: dict[str, list[int | float]],
    variables: list[str],
) -> tuple[list, list]:
    """Validate fixed_cutoffs has correct keys and matching lengths."""
    var0_name = variables[0]
    var1_name = variables[1]

    if var0_name not in fixed_cutoffs or var1_name not in fixed_cutoffs:
        raise ValueError(f"fixed_cutoffs must contain both variables: {variables}. Got: {list(fixed_cutoffs.keys())}")

    var0_bins = fixed_cutoffs[var0_name]
    var1_cutoffs = fixed_cutoffs[var1_name]

    if len(var0_bins) != len(var1_cutoffs):
        raise ValueError(
            f"Length mismatch: {var0_name} has {len(var0_bins)} bins, "
            f"but {var1_name} has {len(var1_cutoffs)} cutoffs. They must match."
        )

    return var0_bins, var1_cutoffs


def _validate_bins_match_data(
    var0_bins: list,
    values_var0: list | np.ndarray,
    strict_validation: bool,
) -> None:
    """Verify config bins exist in data bins."""
    values_var0_set = {float(v) for v in values_var0}
    var0_bins_set = {float(v) for v in var0_bins}

    if var0_bins_set != values_var0_set:
        msg = f"Fixed cutoff bins {sorted(var0_bins_set)} don't exactly match data bins {sorted(values_var0_set)}. "
        if strict_validation:
            raise ValueError(msg + "Enable strict_validation=False to proceed anyway.")
        logger.warning(msg + "Proceeding with provided cutoffs.")


def _validate_cutoff_monotonicity(
    var0_bins: list,
    var1_cutoffs: list,
    inv_var1: bool,
    strict_validation: bool,
) -> None:
    """Check non-decreasing/non-increasing pattern of cutoffs."""
    sorted_pairs = sorted(zip(var0_bins, var1_cutoffs), key=lambda x: float(x[0]))
    monotonicity_issues = []
    for i in range(1, len(sorted_pairs)):
        prev_bin, prev_cutoff = sorted_pairs[i - 1]
        curr_bin, curr_cutoff = sorted_pairs[i]
        if inv_var1:
            if curr_cutoff > prev_cutoff:
                monotonicity_issues.append(
                    f"bin {prev_bin} (cutoff={prev_cutoff}) -> bin {curr_bin} (cutoff={curr_cutoff})"
                )
        else:
            if curr_cutoff < prev_cutoff:
                monotonicity_issues.append(
                    f"bin {prev_bin} (cutoff={prev_cutoff}) -> bin {curr_bin} (cutoff={curr_cutoff})"
                )

    if monotonicity_issues:
        direction = "non-increasing" if inv_var1 else "non-decreasing"
        msg = (
            f"Non-monotonic cutoffs detected. Expected {direction} cutoffs for ascending bins. "
            f"Issues: {monotonicity_issues}. This may indicate a configuration error."
        )
        if strict_validation:
            raise ValueError(msg)
        logger.warning(msg)


def _validate_cutoff_range(
    var1_cutoffs: list,
    values_var1: list | np.ndarray | None,
    strict_validation: bool,
) -> None:
    """Verify cutoff values are within data min/max."""
    if values_var1 is None or len(values_var1) == 0:
        return

    min_var1, max_var1 = min(values_var1), max(values_var1)
    out_of_range = [cutoff for cutoff in var1_cutoffs if cutoff < min_var1 or cutoff > max_var1]
    if out_of_range:
        msg = (
            f"Cutoff values {out_of_range} are outside data range [{min_var1}, {max_var1}]. "
            f"This may result in unexpected acceptance rates."
        )
        if strict_validation:
            raise ValueError(msg)
        logger.warning(msg)


def create_fixed_cutoff_solution(
    fixed_cutoffs: dict[str, list[int | float]],
    variables: list[str],
    values_var0: list | np.ndarray,
    values_var1: list | np.ndarray | None = None,
    strict_validation: bool = False,
    inv_var1: bool = False,
) -> pd.DataFrame:
    """
    Create a single-row solution DataFrame from predefined cutoffs.

    This function allows bypassing the optimization process by specifying
    exact cutoff values for each bin. Useful for applying known/validated
    cutoffs or for scenario analysis with specific cutoff configurations.

    Args:
        fixed_cutoffs: Dictionary mapping variable names to cutoff lists.
            For a 2-variable system with var0 bins [1,2,3,4]:
            {
                "sc_octroi_new_clus": [1, 2, 3, 4],  # var0 bin values
                "new_efx_clus": [2, 2, 3, 4]         # var1 cutoff for each var0 bin
            }
            The var1 list specifies the maximum var1 value to accept for each var0 bin.
        variables: List of two variable names [var0, var1].
        values_var0: Array/list of bin values for the first variable.
        values_var1: Optional array/list of bin values for the second variable.
            If provided, validates that cutoffs are within data range.
        strict_validation: If True, raise errors instead of warnings for
            bin mismatches and validation issues. Default False.
        inv_var1: If True, the var1 cutoffs represent minimum values to accept
            (inverted logic). Affects monotonicity validation direction.

    Returns:
        DataFrame with single row containing:
        - sol_fac: Solution identifier (0 for fixed cutoffs)
        - Columns for each var0 bin value containing the var1 cutoff limit

    Raises:
        ValueError: If cutoffs don't match expected structure or lengths,
            or if strict_validation is True and validation fails.
    """
    var0_bins, var1_cutoffs = _validate_cutoff_dict_structure(fixed_cutoffs, variables)
    _validate_bins_match_data(var0_bins, values_var0, strict_validation)
    _validate_cutoff_monotonicity(var0_bins, var1_cutoffs, inv_var1, strict_validation)
    _validate_cutoff_range(var1_cutoffs, values_var1, strict_validation)

    df = _build_cutoff_dataframe(var0_bins, var1_cutoffs, values_var0)
    logger.info(f"Created fixed cutoff solution: {dict(zip(var0_bins, var1_cutoffs))}")

    return df


def get_fact_sol(
    values_var0: list[float],
    values_var1: list[float],
    inv_var1: bool = False,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """
    Generate all feasible solutions (cut combinations) with monotonicity constraint.

    Optimized using numpy-based combination generation for memory efficiency.

    Args:
        values_var0: Bin indices (columns in output)
        values_var1: Possible cut values for each bin
        inv_var1: If True, cuts must be monotonically decreasing; otherwise increasing
        chunk_size: Unused, kept for backward compatibility

    Returns:
        DataFrame with columns ['sol_fac'] + values_var0, where each row is a valid combination
    """
    try:
        n_bins = len(values_var0)
        # Include 0 as a possible cut, and deduplicate + sort
        cut_values = np.array(sorted(set([0] + list(values_var1))))

        logger.info("--Getting feasible solutions")
        logger.info(f"Bins: {n_bins}, Cut values: {len(cut_values)}")

        # Generate monotonically non-decreasing combinations using numpy
        # combinations_with_replacement returns indices we can use directly
        n_values = len(cut_values)

        # Use numpy to generate combinations more efficiently
        # Create array directly from combinations_with_replacement
        comb_indices = np.array(
            list(combinations_with_replacement(range(n_values), n_bins)),
            dtype=np.int16,
        )
        combinations_array = cut_values[comb_indices]

        logger.info(f"Number of feasible solutions: {len(combinations_array):,}")

        # If inv_var1, we need monotonically decreasing, so reverse each row
        if inv_var1:
            combinations_array = combinations_array[:, ::-1]

        # Create DataFrame with bin indices as column names
        df_v = pd.DataFrame(combinations_array, columns=values_var0)
        df_v = optimize_dtypes(df_v)

        # Add solution index
        df_v.insert(0, "sol_fac", np.arange(len(df_v), dtype=np.int32))

        return df_v

    except (ValueError, MemoryError) as e:
        logger.error(f"Error in get_fact_sol: {str(e)}")
        raise


def process_kpi_chunk(
    chunk_data: pd.DataFrame,
    values_var0: list[float],
    data_sumary_desagregado: pd.DataFrame,
    variables: list[str],
    inv_var1: bool = False,
) -> pd.DataFrame:
    """Process a chunk of data for KPI calculation (picklable helper)"""
    # Melt the chunk
    chunk_melt = chunk_data.melt(
        id_vars=["sol_fac"],
        value_vars=values_var0,
        var_name=variables[0],
        value_name=f"{variables[1]}_lim",
    )

    # Ensure numeric types
    chunk_melt[variables[0]] = chunk_melt[variables[0]].astype(float)

    # Get distinct combinations
    chunk_distinct = chunk_melt.drop_duplicates(subset=[variables[0], f"{variables[1]}_lim"])[
        [variables[0], f"{variables[1]}_lim"]
    ]

    # Merge with summary data
    data_sumary = chunk_distinct.merge(data_sumary_desagregado, how="left", on=variables[0])

    # Apply filters
    if inv_var1:
        data_sumary = data_sumary[data_sumary[variables[1]] > data_sumary[f"{variables[1]}_lim"]]
    else:
        data_sumary = data_sumary[data_sumary[variables[1]] <= data_sumary[f"{variables[1]}_lim"]]

    # Identify numeric columns for aggregation
    numeric_cols = data_sumary.select_dtypes(include=[np.number]).columns
    agg_dict = {col: "sum" for col in numeric_cols if col not in [variables[0], f"{variables[1]}_lim"]}

    # Group and aggregate
    data_sumary = data_sumary.groupby([variables[0], f"{variables[1]}_lim"]).agg(agg_dict).reset_index()

    # Merge back with chunk data and aggregate by solution
    chunk_result = (
        chunk_melt.merge(data_sumary, how="left", on=[variables[0], f"{variables[1]}_lim"])
        .fillna(0)
        .groupby("sol_fac", observed=True)
        .agg(agg_dict)
        .reset_index()
    )

    return chunk_result


def kpi_of_fact_sol(
    df_v: pd.DataFrame,
    values_var0: np.ndarray,
    data_sumary_desagregado: pd.DataFrame,
    variables: list[str],
    indicadores: list[str],
    inv_var1: bool = False,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """
    Calculate KPIs for all feasible solutions by applying cuts to aggregated data.

    For each feasible solution (defined by cut points), calculates aggregate KPIs
    by matching records to cut points and summing indicators. Uses parallel
    processing to efficiently handle large solution spaces.

    Args:
        df_v: DataFrame of feasible solutions from get_fact_sol(), with columns
            for solution ID and cut points for each bin.
        values_var0: Array of bin values for the first variable.
        data_sumary_desagregado: Aggregated data with indicators by variable combinations.
            Expected columns: variables, indicators with '_boo' and '_rep' suffixes.
        variables: List of two variable names [var0, var1].
        indicadores: List of indicator column names to aggregate.
        inv_var1: If True, apply inverse ordering for var1 (decreasing cuts).
        chunk_size: Number of solutions to process per parallel chunk.

    Returns:
        DataFrame with one row per solution containing:
        - sol_fac: Solution identifier
        - Aggregated indicators with suffixes: _boo (booked), _rep (repesca), _cut (rejected)
        - Calculated b2_ever_h6 metrics for each suffix
        Sorted by b2_ever_h6 (risk) and oa_amt_h0 (production).

    Raises:
        Exception: If parallel processing or aggregation fails.
    """
    try:
        logger.info("--Calculating KPIs for feasible solutions")

        # Prepare chunks
        chunks = [df_v.iloc[i : i + chunk_size] for i in range(0, len(df_v), chunk_size)]

        # Parallel Processing
        # n_jobs=-1 uses all available cores
        chunks_results = Parallel(n_jobs=MAX_PARALLEL_JOBS)(
            delayed(process_kpi_chunk)(chunk, values_var0, data_sumary_desagregado, variables, inv_var1)
            for chunk in tqdm(chunks, desc="Processing chunks (Parallel)")
        )

        # Combine results from all chunks
        if not chunks_results:
            return pd.DataFrame()

        # Combine chunks efficiently
        final_result = pd.concat(chunks_results, ignore_index=True)
        del chunks_results
        gc.collect()

        # Group combined results
        final_result = final_result.groupby("sol_fac", observed=True).sum().reset_index()

        # Calculate cut metrics
        for kpi in indicadores:
            raw_cut = data_sumary_desagregado[f"{kpi}_boo"].sum() - final_result[f"{kpi}_boo"]
            neg_count = (raw_cut < 0).sum()
            if neg_count > 0:
                logger.warning(
                    f"{kpi}_cut has {neg_count} negative values (min={raw_cut.min():.2f}). "
                    f"This may indicate double-counting in KPI aggregation. Clipping to 0."
                )
            final_result[f"{kpi}_cut"] = raw_cut.clip(lower=0)

        # Calculate B2 metrics
        metrics = ["", "_cut", "_rep", "_boo"]
        for metric in metrics:
            todu_30 = f"todu_30ever_h6{metric}"
            todu_amt = f"todu_amt_pile_h6{metric}"
            if todu_30 in final_result.columns and todu_amt in final_result.columns:
                final_result[f"b2_ever_h6{metric}"] = calculate_b2_ever_h6(
                    final_result[todu_30].astype(float),
                    final_result[todu_amt].replace(0, np.nan).astype(float),
                    multiplier=DEFAULT_RISK_MULTIPLIER,
                    as_percentage=True,
                ).fillna(0)

        return final_result.sort_values(["b2_ever_h6", "oa_amt_h0"])

    except (ValueError, KeyError) as e:
        logger.error(f"Error in kpi_of_fact_sol: {str(e)}")
        raise


def process_optimal_chunk(df_chunk: pd.DataFrame, data_sumary: pd.DataFrame) -> pd.DataFrame | None:
    """Process a chunk for optimal solution finding"""
    chunk_result = df_chunk.merge(data_sumary, how="inner", on="sol_fac")

    if chunk_result.empty:
        return None
    return chunk_result


def get_optimal_solutions(df_v: pd.DataFrame, data_sumary: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    """Memory-optimized version of get_optimal_solutions with parallel processing"""
    try:
        logger.info("--Getting optimal solutions")

        # Sort and deduplicate efficiently
        data_sumary = data_sumary.sort_values(by=["b2_ever_h6", "oa_amt_h0"])
        data_sumary = data_sumary.drop_duplicates(subset=["b2_ever_h6"], keep="last")

        # Find Pareto optimal solutions using vectorized cummax
        cummax = data_sumary["oa_amt_h0"].cummax()
        pareto_mask = data_sumary["oa_amt_h0"] >= cummax
        data_sumary = data_sumary[pareto_mask]

        # Merge in chunks
        chunks = [df_v.iloc[i : i + chunk_size] for i in range(0, len(df_v), chunk_size)]

        # Parallel Processing
        chunks_results = Parallel(n_jobs=MAX_PARALLEL_JOBS)(
            delayed(process_optimal_chunk)(chunk, data_sumary.reset_index())
            for chunk in tqdm(chunks, desc="Processing chunks (Parallel)")
        )

        # Filter None results
        chunks_results = [res for res in chunks_results if res is not None]

        # Combine results
        final_result = pd.concat(chunks_results, ignore_index=True)
        del chunks_results
        gc.collect()

        # Optimize final datatypes
        final_result = optimize_dtypes(final_result)

        logger.info(f"Number of optimal solutions: {len(final_result):,}")
        return final_result.sort_values(by=["b2_ever_h6", "oa_amt_h0"])

    except (ValueError, KeyError) as e:
        logger.error(f"Error in get_optimal_solutions: {str(e)}")
        raise
