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

    # Compute metrics per month
    monthly = []
    for period, group in df.groupby("year_month"):
        row = {"year_month": period.to_timestamp()}

        # Volume
        row["total_records"] = len(group)
        booked_mask = group[Columns.STATUS_NAME] == StatusName.BOOKED.value
        row["booked_count"] = booked_mask.sum()
        row["approval_rate"] = row["booked_count"] / row["total_records"] if row["total_records"] > 0 else 0.0

        # Production metrics
        if Columns.OA_AMT in group.columns:
            booked_data = group.loc[booked_mask]
            row["mean_production"] = booked_data[Columns.OA_AMT].mean() if len(booked_data) > 0 else 0.0
            row["total_production"] = booked_data[Columns.OA_AMT].sum() if len(booked_data) > 0 else 0.0

        # Risk metrics
        if Columns.TODU_30EVER_H6 in group.columns and Columns.TODU_AMT_PILE_H6 in group.columns:
            booked_data = group.loc[booked_mask]
            if len(booked_data) > 0:
                b2 = calculate_b2_ever_h6(
                    booked_data[Columns.TODU_30EVER_H6],
                    booked_data[Columns.TODU_AMT_PILE_H6],
                )
                row["risk_rate"] = b2.mean() if len(b2) > 0 else np.nan
            else:
                row["risk_rate"] = np.nan

        # Score metrics (check common score columns)
        for score_col in ["sc_octroi", "new_efx", "risk_score_rf"]:
            if score_col in group.columns:
                booked_data = group.loc[booked_mask]
                row[f"mean_{score_col}"] = booked_data[score_col].mean() if len(booked_data) > 0 else np.nan

        monthly.append(row)

    if not monthly:
        return pd.DataFrame()

    result = pd.DataFrame(monthly)
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
    window: int = 3,
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

    # Compute rolling stats from previous values (shifted by 1 so current value
    # is compared against the window of preceding observations)
    rolling_mean = df["value"].rolling(window=window, min_periods=window).mean().shift(1)
    rolling_std = df["value"].rolling(window=window, min_periods=window).std().shift(1)

    df["rolling_mean"] = rolling_mean
    df["upper_bound"] = rolling_mean + n_sigma * rolling_std
    df["lower_bound"] = rolling_mean - n_sigma * rolling_std

    # Flag anomalies
    df["is_anomaly"] = (df["value"] > df["upper_bound"]) | (df["value"] < df["lower_bound"])
    df["direction"] = None
    df.loc[df["value"] > df["upper_bound"], "direction"] = "above"
    df.loc[df["value"] < df["lower_bound"], "direction"] = "below"

    n_anomalies = df["is_anomaly"].sum()
    if n_anomalies > 0:
        logger.info(f"Detected {n_anomalies} anomalous months for '{metric}'")

    return df.reset_index(drop=True)
