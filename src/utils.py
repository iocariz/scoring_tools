"""
Utility functions for credit risk scoring and portfolio optimization.

This module provides core utility functions used throughout the scoring tools:
- Risk metric calculations (b2_ever_h6)
- Data optimization and memory management

"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from .constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_RISK_MULTIPLIER

# Cap parallel workers to avoid OOM on many-core servers.
# Override with SCORING_TOOLS_MAX_JOBS environment variable.
MAX_PARALLEL_JOBS = int(os.environ.get("SCORING_TOOLS_MAX_JOBS", min(4, os.cpu_count() or 4)))


# ---------------------------------------------------------------------------
# Supersegment resolution helpers
# ---------------------------------------------------------------------------

def resolve_modelling_supersegment(segment_config: dict) -> str | None:
    """Return the modelling supersegment for a segment.

    Resolution order: ``modelling_supersegment`` > ``supersegment`` > ``None``.
    """
    return segment_config.get("modelling_supersegment") or segment_config.get("supersegment")


def resolve_reporting_supersegment(segment_config: dict) -> str | None:
    """Return the reporting supersegment for a segment.

    Resolution order: ``reporting_supersegment`` > ``supersegment`` > ``None``.
    """
    return segment_config.get("reporting_supersegment") or segment_config.get("supersegment")


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
        Calculated b2_ever_h6 values, with division-by-zero handled as NaN.
        Callers that need 0 instead of NaN should apply np.nan_to_num() at the
        display/output boundary.
    """
    # Handle division by zero
    if isinstance(denominator, (pd.Series, np.ndarray)):
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
    else:
        safe_denominator = np.nan if denominator == 0 else denominator

    result = multiplier * numerator / safe_denominator

    if as_percentage:
        result = result * 100

    # Risk cannot be negative — clip to 0 (preserves NaN for missing cells)
    result = np.clip(result, 0, None)

    return np.round(result, decimals)


def calculate_todu_30ever_from_b2(
    b2_ever_h6: pd.Series | np.ndarray | float,
    todu_amt_pile_h6: pd.Series | np.ndarray | float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> pd.Series | np.ndarray | float:
    """
    Calculate todu_30ever_h6 from b2_ever_h6 and todu_amt_pile_h6.

    This is the inverse of calculate_b2_ever_h6:
        todu_30ever_h6 = b2_ever_h6 * todu_amt_pile_h6 / multiplier

    Args:
        b2_ever_h6: Risk metric values
        todu_amt_pile_h6: Exposure values
        multiplier: Risk multiplier (default: 7)

    Returns:
        Calculated todu_30ever_h6 values
    """
    return b2_ever_h6 * todu_amt_pile_h6 / multiplier


def extrapolate_h3_to_h6(
    b2_h3: pd.Series | np.ndarray | float,
    h6_h3_ratio: pd.Series | np.ndarray | float,
    method: str = "linear",
    curvature: float = 1.0,
    b2_h3_main: pd.Series | np.ndarray | float | None = None,
) -> pd.Series | np.ndarray | float:
    """Extrapolate H6 risk from observed H3 risk using a configurable curve.

    Args:
        b2_h3: Observed H3 risk metric values (MR period).
        h6_h3_ratio: Main-period H6/H3 ratio used as base scaling factor.
        method: Extrapolation curve — ``"linear"``, ``"power"``, or ``"logistic"``.
        curvature: Tuning parameter. For **power**: exponent alpha (1.0 = linear).
            For **logistic**: steepness *k*. Ignored for linear.
        b2_h3_main: Main-period H3 risk per bin (required for ``"power"``).
            Used as the reference point so that the power law is applied to
            the deviation ``b2_h3 / b2_h3_main`` rather than to the ratio.

    Returns:
        Extrapolated H6 risk values, same type/shape as *b2_h3*.
    """
    if method == "linear":
        return b2_h3 * h6_h3_ratio
    elif method == "power":
        # Power law: b2_h6 = exp(c) * b2_h3^alpha  fitted via log-log regression.
        # Using h6_h3_ratio = b2_h6_main / b2_h3_main and the fitted alpha:
        #   b2_h6_mr = b2_h3_mr * ratio * (b2_h3_mr / b2_h3_main)^(alpha - 1)
        # This matches the fitted model and degrades to linear when alpha=1.
        if b2_h3_main is not None:
            main_arr = np.asarray(b2_h3_main, dtype=float)
            safe_main = np.where(main_arr > 0, main_arr, np.nan)
            deviation = np.asarray(b2_h3, dtype=float) / safe_main
            power_result = b2_h3 * h6_h3_ratio * np.power(np.clip(deviation, 0.01, 100.0), curvature - 1.0)
            # Fall back to linear for elements where b2_h3_main is NaN/0 (e.g. MR-only bins)
            linear_result = b2_h3 * h6_h3_ratio
            return np.where(np.isfinite(power_result), power_result, linear_result)
        # Fallback when b2_h3_main not available: apply curvature to ratio (legacy)
        return b2_h3 * np.power(h6_h3_ratio, curvature)
    elif method == "logistic":
        # Smoothly caps the scaling for extreme ratios while approaching
        # linear for moderate ratios (ratio ≈ 1).
        # Uses tanh to compress deviations: cap at 1 + 2/curvature.
        # Slope at ratio=1 equals 1 (matches linear for small deviations).
        scale = 1 + 2 * np.tanh(curvature * (h6_h3_ratio - 1) / 2) / curvature
        return b2_h3 * scale
    else:
        raise ValueError(f"Unknown extrapolation method: {method!r}. Use 'linear', 'power', or 'logistic'.")


def fit_h3_extrapolation_curve(
    b2_h3: np.ndarray,
    b2_h6: np.ndarray,
    weights: np.ndarray | None = None,
    min_bins: int = 4,
) -> tuple[str, float, dict]:
    """Fit the H3→H6 extrapolation curvature from observed main-period data.

    Performs weighted log-log regression ``log(b2_h6) = c + alpha * log(b2_h3)``
    to determine whether the relationship is convex (alpha > 1), linear (~1),
    or concave (alpha < 1).

    Args:
        b2_h3: Per-bin H3 risk values (main period).
        b2_h6: Per-bin H6 risk values (main period).
        weights: Optional observation counts per bin for weighted regression.
        min_bins: Minimum valid bins required for fitting.

    Returns:
        ``(method, curvature, diagnostics)`` where *method* is ``"linear"`` or
        ``"power"``, *curvature* is the fitted alpha (clipped to [0.3, 3.0]),
        and *diagnostics* is a dict with ``alpha``, ``se``, ``r_squared``, ``n_bins``.
    """
    valid = (np.asarray(b2_h3) > 0) & (np.asarray(b2_h6) > 0)
    n_valid = int(valid.sum())

    fallback_diag = {"alpha": 1.0, "se": float("nan"), "r_squared": float("nan"), "n_bins": n_valid}
    if n_valid < min_bins:
        return ("linear", 1.0, {**fallback_diag, "note": "insufficient bins"})

    log_h3 = np.log(np.asarray(b2_h3)[valid])
    log_h6 = np.log(np.asarray(b2_h6)[valid])
    w = np.asarray(weights)[valid] if weights is not None else None

    # np.polyfit(w=w) weights the residual (not squared residual) by w,
    # so the effective least-squares weight is w^2.  Use w^2 consistently
    # for all diagnostic statistics (SE, R²) to match the fitted model.
    if w is not None:
        eff_w = w ** 2  # effective weights matching np.polyfit's convention
    else:
        eff_w = np.ones_like(log_h3)
    x_mean = np.average(log_h3, weights=eff_w)
    ss_x = float(np.sum(eff_w * (log_h3 - x_mean) ** 2))

    if not np.isfinite(ss_x) or ss_x <= 1e-10:
        return ("linear", 1.0, {**fallback_diag, "note": "ill_conditioned_design"})

    coeffs = np.polyfit(log_h3, log_h6, 1, w=w)
    alpha = float(coeffs[0])
    intercept = float(coeffs[1])

    predicted = alpha * log_h3 + intercept
    residuals = log_h6 - predicted
    ss_res = float(np.sum(eff_w * residuals**2))
    ss_tot = float(np.sum(eff_w * (log_h6 - np.average(log_h6, weights=eff_w)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mse = ss_res / max(n_valid - 2, 1)
    se = float(np.sqrt(mse / ss_x)) if ss_x > 0 else float("inf")

    diagnostics = {"alpha": alpha, "se": se, "r_squared": r_squared, "n_bins": n_valid}

    # Decision: if 95% CI includes 1.0, use linear
    if abs(alpha - 1.0) < 2.0 * se:
        return ("linear", 1.0, diagnostics)

    curvature = float(np.clip(alpha, 0.3, 3.0))
    return ("power", curvature, diagnostics)


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
    """Optimize DataFrame memory usage by choosing appropriate integer dtypes.

    Works on a copy to avoid mutating the input. Float64 columns are preserved
    to avoid precision loss on financial data.
    """
    df = df.copy()
    for col in df.columns:
        # Convert integer columns
        if df[col].dtype == "int64":
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype("int8")
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")

        # Float64 columns are intentionally preserved to avoid precision loss

    return df


def calculate_stress_factor(
    df: pd.DataFrame,
    status_col: str = "status_name",
    score_col: str = "risk_score_rf",
    num_col: str = "todu_30ever_h6",
    den_col: str = "todu_amt_pile_h6",
    frac: float = 0.05,
    target_status: str = "booked",
    bad_rate: float = 0.05,
    higher_is_worse: bool = False,
) -> float:
    # Filter for target status
    df_target = df[df[status_col] == target_status].copy()

    if df_target.empty:
        logger.warning(f"No records found with {status_col} = {target_status}; returning neutral stress factor 1.0")
        return 1.0

    # Calculate overall bad rate
    total_num = df_target[num_col].sum()
    total_den = df_target[den_col].sum()

    overall_bad_rate = (total_num / total_den * DEFAULT_RISK_MULTIPLIER) if total_den > 0 else bad_rate

    # Select worst population based on score cutoff
    if higher_is_worse:
        cutoff_score = df_target[score_col].quantile(1.0 - frac)
        df_worst = df_target[df_target[score_col] >= cutoff_score]
    else:
        cutoff_score = df_target[score_col].quantile(frac)
        df_worst = df_target[df_target[score_col] <= cutoff_score]

    logger.debug(f"Score cutoff (frac={frac}): {cutoff_score}")
    logger.debug(
        f"Selected {len(df_worst)}/{len(df_target)} records ({len(df_worst) / len(df_target):.2%}) as worst population"
    )

    # Calculate bad rate for worst fraction
    worst_num = df_worst[num_col].sum()
    worst_den = df_worst[den_col].sum()

    worst_bad_rate = (worst_num / worst_den * DEFAULT_RISK_MULTIPLIER) if worst_den > 0 else 0.0

    # Calculate stress factor
    if overall_bad_rate > 0:
        stress_factor = worst_bad_rate / overall_bad_rate
    else:
        logger.warning("Overall bad rate is 0; returning neutral stress factor 1.0")
        stress_factor = 1.0

    return float(stress_factor)


def calculate_per_bin_stress_factors(
    df: pd.DataFrame,
    variables: list[str],
    status_col: str = "status_name",
    score_col: str = "risk_score_rf",
    num_col: str = "todu_30ever_h6",
    den_col: str = "todu_amt_pile_h6",
    frac: float = 0.05,
    target_status: str = "booked",
    higher_is_worse: bool = False,
    min_obs_per_bin: int = 20,
) -> pd.DataFrame:
    """Compute per-bin stress factors.

    For each bin combination, the stress factor is the ratio of the risk rate
    in the worst ``frac`` fraction (by score) to the overall risk rate within
    that bin.  Bins with fewer than ``min_obs_per_bin`` records use the global
    stress factor as a fallback.

    Returns
    -------
    DataFrame with ``[*variables, "stress_factor"]``.
    """
    df_target = df[df[status_col] == target_status].copy()
    if df_target.empty:
        logger.warning("No target records for per-bin stress; returning empty DataFrame")
        return pd.DataFrame(columns=variables + ["stress_factor"])

    global_stress = calculate_stress_factor(
        df, status_col=status_col, score_col=score_col,
        num_col=num_col, den_col=den_col, frac=frac,
        target_status=target_status, higher_is_worse=higher_is_worse,
    )

    rows: list[dict] = []
    for bin_vals, group in df_target.groupby(variables, observed=True):
        if not isinstance(bin_vals, tuple):
            bin_vals = (bin_vals,)
        row = dict(zip(variables, bin_vals))

        if len(group) < min_obs_per_bin:
            row["stress_factor"] = global_stress
            rows.append(row)
            continue

        total_den = group[den_col].sum()
        if total_den == 0:
            row["stress_factor"] = global_stress
            rows.append(row)
            continue

        overall_rate = DEFAULT_RISK_MULTIPLIER * group[num_col].sum() / total_den

        if higher_is_worse:
            cutoff = group[score_col].quantile(1.0 - frac)
            worst = group[group[score_col] >= cutoff]
        else:
            cutoff = group[score_col].quantile(frac)
            worst = group[group[score_col] <= cutoff]

        worst_den = worst[den_col].sum()
        worst_rate = (DEFAULT_RISK_MULTIPLIER * worst[num_col].sum() / worst_den) if worst_den > 0 else 0.0

        row["stress_factor"] = worst_rate / overall_rate if overall_rate > 0 else global_stress
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.debug(
        f"Per-bin stress factors: {len(result)} bins | "
        f"mean={result['stress_factor'].mean():.3f} | "
        f"min={result['stress_factor'].min():.3f} | "
        f"max={result['stress_factor'].max():.3f}"
    )
    return result


def calculate_annual_coef(date_ini_book_obs: pd.Timestamp, date_fin_book_obs: pd.Timestamp) -> float:
    """
    Calculate annual coefficient based on the time range.
    """
    if date_fin_book_obs < date_ini_book_obs:
        raise ValueError(
            f"date_fin_book_obs ({date_fin_book_obs.date()}) is before "
            f"date_ini_book_obs ({date_ini_book_obs.date()})"
        )
    n_month = (
        (date_fin_book_obs.year - date_ini_book_obs.year) * 12 + (date_fin_book_obs.month - date_ini_book_obs.month) + 1
    )
    annual_coef = 12 / n_month
    return annual_coef


def _bootstrap_worker(
    df: pd.DataFrame,
    cut_map: dict[float, float],
    variables: list[str],
    multiplier: float,
    random_state: int | None = None,
    inv_var1: bool = False,
    annual_coef: float = 1.0,
    repesca_production: float = 0.0,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> tuple[float, float]:
    """Worker function for bootstrap resampling."""
    # Resample with replacement
    sample = df.sample(frac=1.0, replace=True, random_state=random_state)

    # Apply cuts — N-dimensional mask-based path or legacy 2-var cut_map path
    if mask is not None and grid is not None:
        from src.optimization_utils import classify_by_mask

        passes = classify_by_mask(sample, mask, grid)
    else:
        var0 = variables[0]
        if len(variables) < 2:
            raise ValueError("2-var cut_map bootstrap path requires at least 2 variables")
        var1 = variables[1]

        # Map cuts to each row based on var0 bin
        # For missing bins, default to strict rejection:
        #   non-inverted (var1 <= cutoff): fillna(-inf) → always rejects
        #   inverted (var1 >= cutoff): fillna(+inf) → always rejects
        fallback = np.inf if inv_var1 else -np.inf
        full_cut_series = sample[var0].map(cut_map).fillna(fallback)

        # Filter passed — inverted variables use >= (higher bin = safer)
        if inv_var1:
            passes = sample[var1] >= full_cut_series
        else:
            passes = sample[var1] <= full_cut_series

    passed_df = sample[passes]

    # Use oa_amt_h0 to match the optimization pipeline metric
    prod_col = "oa_amt_h0" if "oa_amt_h0" in passed_df.columns else "oa_amt"
    production_booked = passed_df[prod_col].sum() if not passed_df.empty else 0.0
    production = (production_booked * annual_coef) + repesca_production

    risk_num = passed_df["todu_30ever_h6"].sum() if not passed_df.empty else 0.0
    risk_den = passed_df["todu_amt_pile_h6"].sum() if not passed_df.empty else 0.0

    risk = calculate_b2_ever_h6(risk_num, risk_den, multiplier=multiplier, as_percentage=False, decimals=6)

    return production, float(risk)


def calculate_bootstrap_intervals(
    data_booked: pd.DataFrame,
    cut_map: dict[float, float],
    variables: list[str],
    multiplier: float,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    inv_var1: bool = False,
    model_cv_se_risk: float | None = None,
    annual_coef: float = 1.0,
    repesca_production: float = 0.0,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> dict[str, float]:
    """
    Calculate confidence intervals for Risk and Production using bootstrap resampling.

    Args:
        data_booked: DataFrame of booked accounts (patient-level data)
        cut_map: Dictionary mapping var0 bin values to var1 cutoff thresholds
        variables: List of [var0, var1] names
        multiplier: Risk multiplier
        n_bootstraps: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95)
        random_state: Seed for reproducibility (default: 42)
        inv_var1: If True, use >= comparison for var1 (higher bin = safer)
        model_cv_se_risk: Optional standard error of the risk model's CV predictions.
            When provided, the risk CI is widened to account for model prediction
            uncertainty (which the bootstrap alone does not capture, since it only
            resamples booked records and ignores model inference error on the
            rejected/swap-in population).  The total SE is computed as
            ``sqrt(bootstrap_se² + model_cv_se²)``.
        mask: Optional binary acceptance mask for N-d classify_by_mask path.
        grid: Optional CellGrid for N-d classify_by_mask path.

    Returns:
        Dictionary with lower/upper bounds for production and risk
    """
    logger.info(f"Calculating {confidence_level:.0%} CI with {n_bootstraps} bootstraps...")

    # Generate per-iteration seeds for reproducibility
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        seeds = rng.randint(0, 2**31 - 1, size=n_bootstraps)
    else:
        seeds = [None] * n_bootstraps

    # Parallel execution (capped to avoid OOM).
    # prefer="threads" avoids heavy fork overhead; numpy releases the GIL so
    # threads still achieve true parallelism for the numerical work.
    results = Parallel(n_jobs=MAX_PARALLEL_JOBS, prefer="threads")(
        delayed(_bootstrap_worker)(
            data_booked,
            cut_map,
            variables,
            multiplier,
            random_state=int(seed) if seed is not None else None,
            inv_var1=inv_var1,
            annual_coef=annual_coef,
            repesca_production=repesca_production,
            mask=mask,
            grid=grid,
        )
        for seed in seeds
    )

    # Unzip results
    productions, risks = zip(*results)

    # Calculate percentiles
    alpha = (1 - confidence_level) / 2
    lower_p = alpha * 100
    upper_p = (1 - alpha) * 100

    prod_lower = np.nanpercentile(productions, lower_p)
    prod_upper = np.nanpercentile(productions, upper_p)
    risk_lower = np.nanpercentile(risks, lower_p)
    risk_upper = np.nanpercentile(risks, upper_p)

    # Note: model_cv_se_risk is no longer used for inflation because it is on
    # the RMSE scale (prediction error), not the risk-ratio scale.  Combining
    # the two via variance addition is statistically invalid.  The percentile
    # bootstrap CI already captures sampling uncertainty in the risk estimate.
    if model_cv_se_risk is not None and model_cv_se_risk > 0:
        logger.info(f"  Model CV SE (informational only, not used for inflation): {model_cv_se_risk:.6f}")

    return {
        "production_ci_lower": prod_lower,
        "production_ci_upper": prod_upper,
        # Return risk CIs as percentage to match the Risk (%) column convention.
        # Single authoritative conversion point — callers should NOT multiply by 100.
        "risk_ci_lower": risk_lower * 100,
        "risk_ci_upper": risk_upper * 100,
    }


def generate_cutoff_summary(
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    segment_name: str,
    scenario_name: str = "base",
    risk_value: float | None = None,
    production_value: float | None = None,
    ci_data: dict[str, float] | None = None,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> pd.DataFrame:
    """
    Generate a readable summary of cutoff points by segment.

    For 2-variable grids: produces the classic var0_bin → cutoff_value table.
    For N>2 variable grids (when ``mask``/``grid`` are provided): produces a
    cell-level table with one row per grid cell showing accepted/rejected.

    Args:
        optimal_solution_df: DataFrame containing the optimal solution with bin columns
        variables: List of variable names used in the optimization
        segment_name: Name of the segment being analyzed
        scenario_name: Name of the scenario (e.g., 'pessimistic', 'base', 'optimistic')
        risk_value: Optional risk (b2_ever_h6) value for this solution
        production_value: Optional production (oa_amt_h0) value for this solution
        ci_data: Optional confidence interval data dict
        mask: Optional binary acceptance mask (for N>2 cell-level summary)
        grid: Optional CellGrid (for N>2 cell-level summary)

    Returns:
        DataFrame with cutoff information.
    """
    if optimal_solution_df is None or optimal_solution_df.empty:
        logger.warning("No optimal solution provided for cutoff summary")
        return pd.DataFrame()

    # N>2: produce cell-level summary from mask
    if len(variables) != 2 and mask is not None and grid is not None:
        from src.optimization_utils import CellGrid

        if isinstance(grid, CellGrid):
            cell_df = grid.cell_data[grid.variables].copy()
            cell_df["accepted"] = mask
            cell_df["segment"] = segment_name
            cell_df["scenario"] = scenario_name
            cell_df["risk_pct"] = risk_value
            cell_df["production"] = production_value
            if ci_data:
                cell_df["production_ci_lower"] = ci_data.get("production_ci_lower")
                cell_df["production_ci_upper"] = ci_data.get("production_ci_upper")
                cell_df["risk_ci_lower"] = ci_data.get("risk_ci_lower", 0)
                cell_df["risk_ci_upper"] = ci_data.get("risk_ci_upper", 0)
            logger.info(
                f"Generated N-d cutoff summary for segment '{segment_name}', scenario '{scenario_name}' "
                f"({len(cell_df)} cells, {int(mask.sum())} accepted)"
            )
            return cell_df

    # 2-variable path: classic var0_bin → cutoff_value table
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

        if pd.notna(cutoff) and np.isfinite(cutoff):
            safe_cutoff = int(cutoff)
        elif pd.notna(cutoff):
            safe_cutoff = float(cutoff)  # Keep as inf or -inf
        else:
            safe_cutoff = None

        row_data = {
            "segment": segment_name,
            "scenario": scenario_name,
            f"{var0_name}_bin": int(bin_val),
            "var0_name": var0_name,
            "cutoff_value": safe_cutoff,
            "var1_name": var1_name,
            "risk_pct": risk_value,
            "production": production_value,
        }

        if ci_data:
            row_data["production_ci_lower"] = ci_data.get("production_ci_lower")
            row_data["production_ci_upper"] = ci_data.get("production_ci_upper")
            row_data["risk_ci_lower"] = ci_data.get("risk_ci_lower", 0)
            row_data["risk_ci_upper"] = ci_data.get("risk_ci_upper", 0)

        summary_rows.append(row_data)

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

    # Build pivot index — include CI columns if present so they survive pivot
    index_cols = ["segment", "scenario", "risk_pct", "production"]
    ci_cols = ["production_ci_lower", "production_ci_upper", "risk_ci_lower", "risk_ci_upper"]
    available_ci = [c for c in ci_cols if c in cutoff_summary.columns]
    index_cols.extend(available_ci)

    # Pivot to wide format
    pivot_df = cutoff_summary.pivot_table(
        index=index_cols, columns=bin_col, values="cutoff_value", aggfunc="first"
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
