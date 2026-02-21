"""
Temporal trend analysis for monthly metric evolution.

Provides functions for:
- Computing monthly aggregated metrics (approval rate, production, risk)
- Plotting metric trends over time
- Detecting trend changes using statistical process control (SPC)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from src.constants import Columns, StatusName
from src.utils import calculate_b2_ever_h6


def compute_monthly_metrics(
    data: pd.DataFrame,
    date_column: str = "mis_date",
    segment_filter: str | None = None,
) -> pd.DataFrame:
    """
    Compute monthly aggregated metrics from raw data.

    Args:
        data: DataFrame with at least date_column, status_name, and production columns.
        date_column: Name of the date column.
        segment_filter: Optional segment value to filter on (uses segment_cut_off column).

    Returns:
        DataFrame indexed by month with aggregated metrics.
    """
    df = data.copy()

    # Parse dates
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=False)

    # Filter by segment if provided
    if segment_filter is not None and "segment_cut_off" in df.columns:
        df = df[df["segment_cut_off"] == segment_filter]

    if df.empty:
        logger.warning("No data after filtering for monthly metrics")
        return pd.DataFrame()

    # Create year-month period column
    df["year_month"] = df[date_column].dt.to_period("M")
    df["_is_booked"] = df[Columns.STATUS_NAME] == StatusName.BOOKED.value

    # Production metrics (booked-only will be handled post-agg)
    has_oa_amt = Columns.OA_AMT in df.columns
    has_risk = Columns.TODU_30EVER_H6 in df.columns and Columns.TODU_AMT_PILE_H6 in df.columns

    # Use vectorized groupby for booked-only metrics
    booked_df = df[df["_is_booked"]]
    grouped_all = df.groupby("year_month")
    grouped_booked = booked_df.groupby("year_month") if not booked_df.empty else None

    # Total records and booked counts
    counts = grouped_all["_is_booked"].agg(["sum", "count"])
    counts.columns = ["booked_count", "total_records"]
    counts["approval_rate"] = counts["booked_count"] / counts["total_records"]

    result = counts.copy()

    # Production metrics from booked records
    if has_oa_amt and grouped_booked is not None:
        prod = grouped_booked[Columns.OA_AMT].agg(["mean", "sum"])
        prod.columns = ["mean_production", "total_production"]
        result = result.join(prod)

    # Risk metrics from booked records
    if has_risk and grouped_booked is not None:
        risk_agg = grouped_booked[[Columns.TODU_30EVER_H6, Columns.TODU_AMT_PILE_H6]].sum()
        result["risk_rate"] = calculate_b2_ever_h6(
            risk_agg[Columns.TODU_30EVER_H6], risk_agg[Columns.TODU_AMT_PILE_H6], as_percentage=True
        )

    # Score metrics from booked records
    for score_col in ["sc_octroi", "new_efx", "risk_score_rf"]:
        if score_col in df.columns and grouped_booked is not None:
            result[f"mean_{score_col}"] = grouped_booked[score_col].mean()

    # Convert period index to timestamp
    result.index = result.index.to_timestamp()
    result = result.reset_index().rename(columns={"year_month": "year_month"})
    result = result.sort_values("year_month").reset_index(drop=True)

    logger.info(f"Monthly metrics computed: {len(result)} months, {result['total_records'].sum():,} total records")
    return result


def plot_metric_trends(
    monthly_df: pd.DataFrame,
    metrics: list[str],
    output_path: str | None = None,
) -> go.Figure:
    """
    Plot monthly metric trends with linear trend lines.

    Args:
        monthly_df: DataFrame from compute_monthly_metrics.
        metrics: List of column names to plot.
        output_path: Optional path to save HTML file.

    Returns:
        Plotly Figure with subplots for each metric.
    """
    # Filter to metrics that exist in the data
    available_metrics = [m for m in metrics if m in monthly_df.columns]
    if not available_metrics:
        logger.warning(f"None of the requested metrics found: {metrics}")
        fig = go.Figure()
        fig.add_annotation(text="No metrics available", x=0.5, y=0.5, showarrow=False)
        return fig

    n_metrics = len(available_metrics)
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        subplot_titles=[m.replace("_", " ").title() for m in available_metrics],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    x = monthly_df["year_month"]
    x_numeric = np.arange(len(x))

    for i, metric in enumerate(available_metrics, 1):
        y = monthly_df[metric]
        valid_mask = y.notna()

        # Data line
        fig.add_trace(
            go.Scatter(
                x=x[valid_mask],
                y=y[valid_mask],
                mode="lines+markers",
                name=metric.replace("_", " ").title(),
                marker=dict(size=5),
                line=dict(width=2),
            ),
            row=i,
            col=1,
        )

        # Linear trend line
        if valid_mask.sum() >= 2:
            coeffs = np.polyfit(x_numeric[valid_mask], y[valid_mask], 1)
            trend = np.polyval(coeffs, x_numeric[valid_mask])
            fig.add_trace(
                go.Scatter(
                    x=x[valid_mask],
                    y=trend,
                    mode="lines",
                    name=f"{metric} trend",
                    line=dict(dash="dash", width=1, color="gray"),
                    showlegend=False,
                ),
                row=i,
                col=1,
            )

    fig.update_layout(
        height=250 * n_metrics,
        template="plotly_white",
        title="Monthly Metric Trends",
        showlegend=True,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Metric trends saved to {output_path}")

    return fig


def detect_trend_changes(
    monthly_df: pd.DataFrame,
    metric: str,
    window: int = 6,
    n_sigma: float = 2.0,
) -> pd.DataFrame:
    """
    Detect trend changes using statistical process control (SPC).

    Flags months where the metric exceeds rolling mean +/- n_sigma * rolling_std.

    Args:
        monthly_df: DataFrame from compute_monthly_metrics.
        metric: Column name to analyze.
        window: Rolling window size (in months).
        n_sigma: Number of standard deviations for bounds.

    Returns:
        DataFrame with columns: year_month, value, rolling_mean, upper_bound,
        lower_bound, is_anomaly, direction.
    """
    if metric not in monthly_df.columns:
        raise ValueError(f"Metric '{metric}' not found in data. Available: {list(monthly_df.columns)}")

    df = monthly_df[["year_month", metric]].copy()
    df = df.rename(columns={metric: "value"})
    df = df.dropna(subset=["value"])

    if len(df) < window:
        logger.warning(f"Not enough data points ({len(df)}) for window size {window}")
        df["rolling_mean"] = np.nan
        df["upper_bound"] = np.nan
        df["lower_bound"] = np.nan
        df["is_anomaly"] = False
        df["direction"] = None
        return df

    # Compute robust rolling stats: rolling median and rolling MAD
    rolling_median = df["value"].rolling(window=window, min_periods=window).median().shift(1)

    # We need a custom rolling MAD function
    def mad(x):
        median = np.median(x)
        return np.median(np.abs(x - median))

    rolling_mad = df["value"].rolling(window=window, min_periods=window).apply(mad, raw=True).shift(1)

    # For small windows, use t-distribution critical value instead of normal z
    # to maintain the intended false-positive rate (n_sigma maps to tail probability)
    if window < 20:
        from scipy.stats import norm
        from scipy.stats import t as t_dist

        target_alpha = 2 * (1 - norm.cdf(n_sigma))  # two-tailed alpha implied by n_sigma
        effective_sigma = t_dist.ppf(1 - target_alpha / 2, df=window - 1)
        logger.debug(
            f"SPC: small window ({window}), adjusting sigma from {n_sigma:.2f} to {effective_sigma:.2f} "
            f"(t-distribution, df={window - 1})"
        )
    else:
        effective_sigma = n_sigma

    # Convert MAD to standard deviation equivalent: std = MAD / 0.6745
    rolling_std_est = rolling_mad / 0.6745

    # Handle cases where MAD is 0 (e.g. constant values in window)
    # Fallback to standard deviation
    rolling_std_traditional = df["value"].rolling(window=window, min_periods=window).std().shift(1)
    rolling_std_est = rolling_std_est.fillna(rolling_std_traditional)
    rolling_std_est = np.where(rolling_std_est == 0, rolling_std_traditional, rolling_std_est)

    df["rolling_mean"] = rolling_median  # using median as the center line for robust SPC
    df["upper_bound"] = rolling_median + effective_sigma * rolling_std_est
    df["lower_bound"] = rolling_median - effective_sigma * rolling_std_est

    # Flag anomalies
    df["is_anomaly"] = (df["value"] > df["upper_bound"]) | (df["value"] < df["lower_bound"])
    df["direction"] = None
    df.loc[df["value"] > df["upper_bound"], "direction"] = "above"
    df.loc[df["value"] < df["lower_bound"], "direction"] = "below"

    n_anomalies = df["is_anomaly"].sum()
    if n_anomalies > 0:
        logger.info(f"Detected {n_anomalies} anomalous months for '{metric}'")

    return df.reset_index(drop=True)
