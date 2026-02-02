"""
Utility functions for credit risk scoring and portfolio optimization.

This module provides core utility functions used throughout the scoring tools:
- Risk metric calculations (b2_ever_h6)
- Data optimization and memory management
- Feasible solution generation for portfolio optimization
- KPI calculations and stress testing
- Visualization helpers for transformation rates

Key functions:
- calculate_b2_ever_h6: Calculate the primary risk metric
- get_fact_sol: Generate feasible solutions with monotonicity constraints
- kpi_of_fact_sol: Calculate KPIs for feasible solutions
- get_optimal_solutions: Find Pareto-optimal solutions
- calculate_stress_factor: Compute stress factors from historical data
"""

import gc
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from loguru import logger
from plotly.subplots import make_subplots
from tqdm import tqdm

from . import styles
from .constants import DEFAULT_RISK_MULTIPLIER, Columns, StatusName


def calculate_b2_ever_h6(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    as_percentage: bool = False,
    decimals: int = 2,
) -> pd.Series | np.ndarray | float:
    """
    Calculate the b2_ever_h6 risk metric.

    Formula: multiplier * numerator / denominator

    Args:
        numerator: todu_30ever_h6 values
        denominator: todu_amt_pile_h6 values
        multiplier: Risk multiplier (default: 7)
        as_percentage: If True, multiply result by 100
        decimals: Number of decimal places to round to

    Returns:
        Calculated b2_ever_h6 values, with division-by-zero handled as NaN
    """
    # Handle division by zero
    if isinstance(denominator, (pd.Series, np.ndarray)):
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
    else:
        safe_denominator = np.nan if denominator == 0 else denominator

    result = multiplier * numerator / safe_denominator

    if as_percentage:
        result = result * 100

    return np.round(result, decimals)


def get_data_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display DataFrame information and return a DataFrame with variable details.
    """
    # Display DataFrame information
    logger.info(f"Number of rows/records: {df.shape[0]}")
    logger.info(f"Number of columns/variables: {df.shape[1]}")
    logger.info("-" * 50)

    # Create a DataFrame with variable information
    variables_df = pd.DataFrame(
        {
            "Variable": df.columns,
            "Number of unique values": df.nunique(),
            "Variable Type": df.dtypes,
            "Number of missing values": df.isnull().sum(),
            "Percentage missing values": df.isnull().mean() * 100,
        }
    )

    # Sort variables by percentage of missing values
    variables_df = variables_df.sort_values(by="Percentage missing values", ascending=False)

    # Return the DataFrame with variable information
    return variables_df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by choosing appropriate dtypes"""
    for col in df.columns:
        # Convert integer columns
        if df[col].dtype == "int64":
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() < 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype("int8")
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")

        # Convert float columns
        elif df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    return df


def _get_fact_sol(values_var0: list[float], values_var1: list[float], inv_var1: bool = False) -> pd.DataFrame:
    """
    Generate all feasible solutions (cut combinations) with monotonicity constraint.

    Uses itertools.combinations_with_replacement for O(C(n+k-1, k)) complexity
    instead of O(n^k) from repeated merges.

    Args:
        values_var0: Bin indices (columns in output)
        values_var1: Possible cut values for each bin
        inv_var1: If True, cuts must be monotonically decreasing; otherwise increasing

    Returns:
        DataFrame with columns ['sol_fac'] + values_var0, where each row is a valid combination
    """
    from itertools import combinations_with_replacement

    n_bins = len(values_var0)
    # Include 0 as a possible cut, and deduplicate + sort
    cut_values = sorted(set([0] + list(values_var1)))

    logger.info("--Obteniendo soluciones factibles")
    logger.info(f"Bins: {n_bins}, Cut values: {len(cut_values)}")

    # Generate monotonically non-decreasing combinations directly
    # combinations_with_replacement gives tuples where values are in sorted order
    combinations = list(combinations_with_replacement(cut_values, n_bins))

    logger.info(f"Número de soluciones factibles: {len(combinations):,}")

    # If inv_var1, we need monotonically decreasing, so reverse each combination
    if inv_var1:
        combinations = [tuple(reversed(c)) for c in combinations]

    # Create DataFrame with bin indices as column names
    df_v = pd.DataFrame(combinations, columns=values_var0)
    df_v = optimize_dtypes(df_v)

    # Add solution index
    df_v = df_v.reset_index().rename(columns={"index": "sol_fac"})

    return df_v


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
        id_vars=["sol_fac"], value_vars=values_var0, var_name=variables[0], value_name=f"{variables[1]}_lim"
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


def process_optimal_chunk(df_chunk: pd.DataFrame, data_sumary: pd.DataFrame) -> pd.DataFrame | None:
    """Process a chunk for optimal solution finding"""
    chunk_result = df_chunk.merge(data_sumary, how="inner", on="sol_fac")

    if chunk_result.empty:
        return None
    return chunk_result


def get_fact_sol(
    values_var0: list[float], values_var1: list[float], inv_var1: bool = False, chunk_size: int = 10000
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
    from itertools import combinations_with_replacement

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
        comb_indices = np.array(list(combinations_with_replacement(range(n_values), n_bins)), dtype=np.int16)
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

    except Exception as e:
        logger.error(f"Error in get_fact_sol: {str(e)}")
        raise


def create_fixed_cutoff_solution(
    fixed_cutoffs: dict[str, list[int | float]],
    variables: list[str],
    values_var0: list | np.ndarray,
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

    Returns:
        DataFrame with single row containing:
        - sol_fac: Solution identifier (0 for fixed cutoffs)
        - Columns for each var0 bin value containing the var1 cutoff limit

    Raises:
        ValueError: If cutoffs don't match expected structure or lengths.

    Example:
        >>> fixed_cutoffs = {
        ...     "sc_octroi_new_clus": [1, 2, 3, 4],
        ...     "new_efx_clus": [2, 2, 3, 4]
        ... }
        >>> df = create_fixed_cutoff_solution(fixed_cutoffs, ["sc_octroi_new_clus", "new_efx_clus"], [1, 2, 3, 4])
        >>> # Result: DataFrame with columns [sol_fac, 1.0, 2.0, 3.0, 4.0]
        >>> #         Values:                [0,       2,   2,   3,   4]
    """
    var0_name = variables[0]
    var1_name = variables[1]

    # Validate fixed_cutoffs structure
    if var0_name not in fixed_cutoffs or var1_name not in fixed_cutoffs:
        raise ValueError(f"fixed_cutoffs must contain both variables: {variables}. Got: {list(fixed_cutoffs.keys())}")

    var0_bins = fixed_cutoffs[var0_name]
    var1_cutoffs = fixed_cutoffs[var1_name]

    if len(var0_bins) != len(var1_cutoffs):
        raise ValueError(
            f"Length mismatch: {var0_name} has {len(var0_bins)} bins, "
            f"but {var1_name} has {len(var1_cutoffs)} cutoffs. They must match."
        )

    # Validate that var0_bins match the actual data bins
    values_var0_set = {float(v) for v in values_var0}
    var0_bins_set = {float(v) for v in var0_bins}

    if var0_bins_set != values_var0_set:
        logger.warning(
            f"Fixed cutoff bins {sorted(var0_bins_set)} don't exactly match data bins {sorted(values_var0_set)}. "
            f"Proceeding with provided cutoffs."
        )

    # Create solution DataFrame
    # Columns are the var0 bin values, values are the var1 cutoff limits
    solution_data = {"sol_fac": [0]}
    for bin_val, cutoff_val in zip(var0_bins, var1_cutoffs):
        solution_data[float(bin_val)] = [cutoff_val]

    df = pd.DataFrame(solution_data)
    logger.info(f"Created fixed cutoff solution: {dict(zip(var0_bins, var1_cutoffs))}")

    return df


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
        chunks_results = Parallel(n_jobs=-1)(
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
            final_result[f"{kpi}_cut"] = (
                data_sumary_desagregado[f"{kpi}_boo"].sum() - final_result[f"{kpi}_boo"]
            ).clip(lower=0)

        # Calculate B2 metrics
        metrics = ["", "_cut", "_rep", "_boo"]
        for metric in metrics:
            todu_30 = f"todu_30ever_h6{metric}"
            todu_amt = f"todu_amt_pile_h6{metric}"
            if todu_30 in final_result.columns and todu_amt in final_result.columns:
                final_result[f"b2_ever_h6{metric}"] = np.round(
                    100
                    * 7
                    * final_result[todu_30].astype(float)
                    / final_result[todu_amt].replace(0, np.nan).astype(float),
                    2,
                ).fillna(0)

        return final_result.sort_values(["b2_ever_h6", "oa_amt_h0"])

    except Exception as e:
        logger.error(f"Error in kpi_of_fact_sol: {str(e)}")
        raise


def get_optimal_solutions(df_v: pd.DataFrame, data_sumary: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    """Memory-optimized version of get_optimal_solutions with parallel processing"""
    try:
        logger.info("--Getting optimal solutions")

        # Sort and deduplicate efficiently
        data_sumary = data_sumary.sort_values(by=["b2_ever_h6", "oa_amt_h0"])
        data_sumary = data_sumary.drop_duplicates(subset=["b2_ever_h6"], keep="last")

        # Find Pareto optimal solutions efficiently
        data_sumary["optimal"] = False
        current_max = float("-inf")

        for idx in data_sumary.index:
            value = data_sumary.loc[idx, "oa_amt_h0"]
            if value > current_max:
                current_max = value
                data_sumary.loc[idx, "optimal"] = True

        data_sumary = data_sumary[data_sumary["optimal"]].drop(columns=["optimal"])

        # Merge in chunks
        chunks = [df_v.iloc[i : i + chunk_size] for i in range(0, len(df_v), chunk_size)]

        # Parallel Processing
        chunks_results = Parallel(n_jobs=-1)(
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

    except Exception as e:
        logger.error(f"Error in get_optimal_solutions: {str(e)}")
        raise


def calculate_stress_factor(
    df: pd.DataFrame,
    status_col: str = "status_name",
    score_col: str = "risk_score_rf",
    num_col: str = "todu_30ever_h6",
    den_col: str = "todu_amt_pile_h6",
    frac: float = 0.05,
    target_status: str = "booked",
    bad_rate: float = 0.05,
) -> float:
    # Filter for target status
    df_target = df[df[status_col] == target_status].copy()

    if df_target.empty:
        logger.warning(f"No records found with {status_col} = {target_status}")
        return 0.0

    # Calculate overall bad rate
    total_num = df_target[num_col].sum()
    total_den = df_target[den_col].sum()

    overall_bad_rate = (total_num / total_den * 7) if total_den > 0 else bad_rate

    # Calculate cutoff using quantile from the known population
    cutoff_score = df_target[score_col].quantile(frac)

    # Select worst population based on score cutoff
    # Assuming lower score is worse (ascending=True)
    df_worst = df_target[df_target[score_col] <= cutoff_score]

    logger.debug(f"Score cutoff (frac={frac}): {cutoff_score}")
    logger.debug(
        f"Selected {len(df_worst)}/{len(df_target)} records ({len(df_worst) / len(df_target):.2%}) as worst population"
    )

    # Calculate bad rate for worst fraction
    worst_num = df_worst[num_col].sum()
    worst_den = df_worst[den_col].sum()

    worst_bad_rate = (worst_num / worst_den * 7) if worst_den > 0 else 0.0

    # Calculate stress factor
    if overall_bad_rate > 0:
        stress_factor = worst_bad_rate / overall_bad_rate
    else:
        stress_factor = 0.0

    return float(stress_factor)


def calculate_and_plot_transformation_rate(
    data: pd.DataFrame,
    date_col: str,
    amount_col: str = "oa_amt",
    n_months: int | None = None,
    plot_width: int = 1200,
    plot_height: int = 500,
) -> dict[str, Any]:
    """
    Calculates transformation rate and generates a dual-axis plot.

    Transformation Rate = Booked Amount / Eligible Amount (Decision in OK/RV)

    Returns:
    --------
    dict containing:
      - 'stats': Dictionary of overall stats
      - 'monthly_data': DataFrame of monthly breakdown
      - 'figure': Plotly go.Figure object
    """

    # 1. Data Preparation
    # ---------------------------------------------------------
    df = data.copy()
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Filter for last n months
    if n_months is not None:
        max_date = df[date_col].max()
        cutoff_date = max_date - pd.DateOffset(months=n_months)
        df = df[df[date_col] >= cutoff_date]

    # Filter eligible (Denominator: Decision OK or RV)
    eligible_mask = df["se_decision_id"].isin(["ok", "rv"])
    df_eligible = df[eligible_mask].copy()

    # Identify booked (Numerator: Status Booked AND Eligible)
    # Note: We rely on the row being in the eligible set first
    df_eligible["is_booked"] = df_eligible[Columns.STATUS_NAME] == StatusName.BOOKED.value

    # 2. Calculation
    # ---------------------------------------------------------
    # Overall Stats
    total_eligible = df_eligible[amount_col].sum()
    total_booked = df_eligible.loc[df_eligible["is_booked"], amount_col].sum()
    overall_rate = (total_booked / total_eligible) if total_eligible > 0 else 0

    # Monthly Aggregation
    # We use to_period('M') for grouping, but convert back to timestamp for plotting
    df_eligible["period"] = df_eligible[date_col].dt.to_period("M")

    monthly = (
        df_eligible.groupby("period")
        .agg(
            eligible_amt=(amount_col, "sum"),
            booked_amt=(amount_col, lambda x: x[df_eligible.loc[x.index, "is_booked"]].sum()),
        )
        .reset_index()
    )

    monthly["rate"] = monthly["booked_amt"] / monthly["eligible_amt"]
    monthly["rate"] = monthly["rate"].fillna(0)

    # Convert period back to timestamp for Plotly (uses start of month)
    monthly["plot_date"] = monthly["period"].dt.to_timestamp()

    # 3. Plotting
    # ---------------------------------------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Trace 1: Volume (Bars) - Plotted on Secondary Y (Right)
    # We plot this first so it stays in the background
    fig.add_trace(
        go.Bar(
            x=monthly["plot_date"],
            y=monthly["eligible_amt"],
            name="Eligible Volume (€)",
            opacity=0.3,
            marker_color=styles.COLOR_SECONDARY,
            hovertemplate="Volume: %{y:,.0f} €<extra></extra>",
        ),
        secondary_y=True,
    )

    # Trace 2: Transformation Rate (Line) - Plotted on Primary Y (Left)
    fig.add_trace(
        go.Scatter(
            x=monthly["plot_date"],
            y=monthly["rate"],
            name="Transformation Rate",
            mode="lines+markers",
            line=dict(color=styles.COLOR_ACCENT, width=3),
            marker=dict(size=8),
            hovertemplate="Rate: %{y:.1%}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Add Overall Average Line
    fig.add_hline(
        y=overall_rate,
        line_dash="dot",
        line_color=styles.COLOR_RISK,
        annotation_text=f"Avg: {overall_rate:.1%}",
        annotation_position="top left",
        secondary_y=False,
    )

    # 4. Layout Improvements
    # ---------------------------------------------------------
    styles.apply_plotly_style(fig, width=plot_width, height=plot_height)
    fig.update_layout(
        title="<b>Monthly Transformation Rate</b><br><sup>(Booked / [OK + RV]) by Amount</sup>",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Left Axis (Rate)
        yaxis=dict(
            title="Transformation Rate",
            tickformat=".0%",  # Displays 0.5 as 50%
            range=[0, max(monthly["rate"].max() * 1.1, overall_rate * 1.1)],
            showgrid=True,
            gridcolor="lightgrey",
        ),
        # Right Axis (Volume)
        yaxis2=dict(title="Eligible Volume (€)", showgrid=False, zeroline=False),
        xaxis=dict(title="Month", showgrid=False),
    )

    return {
        "overall_rate": overall_rate,
        "overall_booked_amt": total_booked,
        "overall_eligible_amt": total_eligible,
        "monthly_amounts": monthly.drop(columns=["plot_date"]),  # Return clean DF
        "figure": fig,
    }


def calculate_annual_coef(date_ini_book_obs: pd.Timestamp, date_fin_book_obs: pd.Timestamp) -> float:
    """
    Calculate annual coefficient based on the time range.
    """
    n_month = (
        (date_fin_book_obs.year - date_ini_book_obs.year) * 12 + (date_fin_book_obs.month - date_ini_book_obs.month) + 1
    )
    annual_coef = 12 / n_month
    return annual_coef


def generate_cutoff_summary(
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    segment_name: str,
    scenario_name: str = "base",
    risk_value: float | None = None,
    production_value: float | None = None,
) -> pd.DataFrame:
    """
    Generate a readable summary of cutoff points by segment.

    This function transforms the optimal solution into a human-readable format
    showing the cutoff threshold for each bin of the first variable.

    Args:
        optimal_solution_df: DataFrame containing the optimal solution with bin columns
        variables: List of two variable names [var0, var1] used in the optimization
        segment_name: Name of the segment being analyzed
        scenario_name: Name of the scenario (e.g., 'pessimistic', 'base', 'optimistic')
        risk_value: Optional risk (b2_ever_h6) value for this solution
        production_value: Optional production (oa_amt_h0) value for this solution

    Returns:
        DataFrame with columns:
        - segment: Segment name
        - scenario: Scenario name
        - var0_bin: Bin number of the first variable
        - var0_name: Name of the first variable
        - cutoff_value: Maximum value of var1 to accept for this bin
        - var1_name: Name of the second variable
        - risk_pct: Risk percentage (if provided)
        - production: Production amount (if provided)

    Example output:
        segment     scenario  var0_bin  var0_name          cutoff_value  var1_name
        loan_known  base      1         sc_octroi_new_clus  2            new_efx_clus
        loan_known  base      2         sc_octroi_new_clus  4            new_efx_clus
        ...
    """
    if optimal_solution_df is None or optimal_solution_df.empty:
        logger.warning("No optimal solution provided for cutoff summary")
        return pd.DataFrame()

    var0_name = variables[0] if len(variables) > 0 else "var0"
    var1_name = variables[1] if len(variables) > 1 else "var1"

    # Get the first (selected) solution row
    opt_row = optimal_solution_df.iloc[0]

    # Find bin columns (numeric column names representing bins)
    bin_columns = []
    for col in optimal_solution_df.columns:
        try:
            bin_val = int(col) if isinstance(col, str) else col
            if isinstance(bin_val, (int, float)) and not pd.isna(bin_val):
                bin_columns.append((bin_val, col))
        except (ValueError, TypeError):
            continue

    # Sort bins numerically
    bin_columns.sort(key=lambda x: x[0])

    if not bin_columns:
        logger.warning("No bin columns found in optimal solution")
        return pd.DataFrame()

    # Build summary rows
    summary_rows = []
    for bin_val, col_name in bin_columns:
        cutoff = opt_row[col_name]
        summary_rows.append(
            {
                "segment": segment_name,
                "scenario": scenario_name,
                f"{var0_name}_bin": int(bin_val),
                "var0_name": var0_name,
                "cutoff_value": int(cutoff) if pd.notna(cutoff) else None,
                "var1_name": var1_name,
                "risk_pct": risk_value,
                "production": production_value,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    logger.info(f"Generated cutoff summary for segment '{segment_name}', scenario '{scenario_name}'")
    logger.info(
        f"  Bins: {len(bin_columns)}, Cutoff range: "
        f"[{summary_df['cutoff_value'].min()}, {summary_df['cutoff_value'].max()}]"
    )

    return summary_df


def format_cutoff_summary_table(
    cutoff_summary: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Format cutoff summary into a wide pivot table for easier reading.

    Args:
        cutoff_summary: DataFrame from generate_cutoff_summary
        variables: List of variable names [var0, var1]

    Returns:
        Pivoted DataFrame with segments/scenarios as rows and bins as columns
    """
    if cutoff_summary.empty:
        return pd.DataFrame()

    var0_name = variables[0] if len(variables) > 0 else "var0"
    bin_col = f"{var0_name}_bin"

    # Pivot to wide format
    pivot_df = cutoff_summary.pivot_table(
        index=["segment", "scenario", "risk_pct", "production"], columns=bin_col, values="cutoff_value", aggfunc="first"
    ).reset_index()

    # Rename bin columns to be more readable
    pivot_df.columns = [f"bin_{col}" if isinstance(col, (int, float)) else col for col in pivot_df.columns]

    return pivot_df


def consolidate_cutoff_summaries(
    summaries: list[pd.DataFrame],
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Consolidate multiple cutoff summaries into a single DataFrame.

    Args:
        summaries: List of DataFrames from generate_cutoff_summary
        output_path: Optional path to save the consolidated summary as CSV

    Returns:
        Consolidated DataFrame with all summaries
    """
    if not summaries:
        logger.warning("No summaries provided for consolidation")
        return pd.DataFrame()

    # Filter out empty DataFrames
    valid_summaries = [s for s in summaries if not s.empty]

    if not valid_summaries:
        logger.warning("All provided summaries are empty")
        return pd.DataFrame()

    consolidated = pd.concat(valid_summaries, ignore_index=True)

    if output_path:
        consolidated.to_csv(output_path, index=False)
        logger.info(f"Consolidated cutoff summary saved to {output_path}")

    return consolidated
