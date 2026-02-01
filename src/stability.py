"""
Population Stability Index (PSI) and Characteristic Stability Index (CSI) Module

Measures distribution drift between two populations (e.g., Main period vs MR period).
Used to detect model degradation and data drift over time.

PSI Interpretation:
- PSI < 0.1: No significant change (green)
- 0.1 ≤ PSI < 0.25: Moderate change, investigation recommended (yellow)
- PSI ≥ 0.25: Significant change, action required (red)
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots


class StabilityStatus(Enum):
    """Stability status based on PSI/CSI value."""

    STABLE = "stable"  # PSI < 0.1
    MODERATE = "moderate"  # 0.1 <= PSI < 0.25
    UNSTABLE = "unstable"  # PSI >= 0.25


@dataclass
class PSIResult:
    """Result of a PSI calculation for a single variable."""

    variable: str
    psi_value: float
    status: StabilityStatus
    n_bins: int
    bin_details: pd.DataFrame  # Breakdown by bin

    @property
    def status_icon(self) -> str:
        icons = {StabilityStatus.STABLE: "✓", StabilityStatus.MODERATE: "⚠", StabilityStatus.UNSTABLE: "✗"}
        return icons[self.status]

    def __str__(self) -> str:
        return f"{self.status_icon} {self.variable}: PSI={self.psi_value:.4f} ({self.status.value})"


@dataclass
class StabilityReport:
    """Complete stability report comparing two populations."""

    baseline_name: str
    comparison_name: str
    baseline_count: int
    comparison_count: int
    psi_results: list[PSIResult] = field(default_factory=list)
    overall_psi: float | None = None

    @property
    def stable_vars(self) -> list[PSIResult]:
        return [r for r in self.psi_results if r.status == StabilityStatus.STABLE]

    @property
    def moderate_vars(self) -> list[PSIResult]:
        return [r for r in self.psi_results if r.status == StabilityStatus.MODERATE]

    @property
    def unstable_vars(self) -> list[PSIResult]:
        return [r for r in self.psi_results if r.status == StabilityStatus.UNSTABLE]

    @property
    def is_stable(self) -> bool:
        """Returns True if no unstable variables."""
        return len(self.unstable_vars) == 0

    def add(self, result: PSIResult) -> None:
        self.psi_results.append(result)

    def summary(self) -> str:
        total = len(self.psi_results)
        return (
            f"Stability: {len(self.stable_vars)}/{total} stable, "
            f"{len(self.moderate_vars)} moderate, {len(self.unstable_vars)} unstable"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for export."""
        data = []
        for r in self.psi_results:
            data.append({"variable": r.variable, "psi": r.psi_value, "status": r.status.value, "n_bins": r.n_bins})
        return pd.DataFrame(data).sort_values("psi", ascending=False)

    def print_report(self) -> None:
        """Print formatted stability report to console."""
        print("\n" + "=" * 70)
        print("STABILITY REPORT (PSI/CSI)")
        print("=" * 70)
        print(f"\nBaseline:   {self.baseline_name} (n={self.baseline_count:,})")
        print(f"Comparison: {self.comparison_name} (n={self.comparison_count:,})")

        if self.overall_psi is not None:
            status = get_psi_status(self.overall_psi)
            icon = {"stable": "✓", "moderate": "⚠", "unstable": "✗"}[status.value]
            print(f"\nOverall PSI: {self.overall_psi:.4f} {icon} ({status.value})")

        # Group by status
        for status, label, _color in [
            (StabilityStatus.UNSTABLE, "UNSTABLE (PSI ≥ 0.25)", "red"),
            (StabilityStatus.MODERATE, "MODERATE (0.1 ≤ PSI < 0.25)", "yellow"),
            (StabilityStatus.STABLE, "STABLE (PSI < 0.1)", "green"),
        ]:
            results = [r for r in self.psi_results if r.status == status]
            if results:
                print(f"\n{label}:")
                for r in sorted(results, key=lambda x: -x.psi_value):
                    print(f"  {r}")

        print("\n" + "-" * 70)
        print(self.summary())
        print("=" * 70 + "\n")


def get_psi_status(psi_value: float) -> StabilityStatus:
    """Determine stability status from PSI value."""
    if psi_value < 0.1:
        return StabilityStatus.STABLE
    elif psi_value < 0.25:
        return StabilityStatus.MODERATE
    else:
        return StabilityStatus.UNSTABLE


def calculate_psi(
    baseline: pd.Series, comparison: pd.Series, bins: int | list[float] = 10, min_pct: float = 0.0001
) -> tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index between two distributions.

    Args:
        baseline: Series with baseline/expected distribution (e.g., Main period)
        comparison: Series with comparison/actual distribution (e.g., MR period)
        bins: Number of bins or list of bin edges
        min_pct: Minimum percentage to avoid division by zero (default: 0.01%)

    Returns:
        Tuple of (PSI value, DataFrame with bin-level breakdown)

    Formula:
        PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    """
    # Handle missing values
    baseline_clean = baseline.dropna()
    comparison_clean = comparison.dropna()

    if len(baseline_clean) == 0 or len(comparison_clean) == 0:
        logger.warning("Empty series provided for PSI calculation")
        return 0.0, pd.DataFrame()

    # Create bins from baseline distribution
    if isinstance(bins, int):
        # Use quantiles from baseline to create bins
        try:
            _, bin_edges = pd.qcut(baseline_clean, q=bins, retbins=True, duplicates="drop")
        except ValueError:
            # Fallback to equal-width bins if quantiles fail
            _, bin_edges = pd.cut(baseline_clean, bins=bins, retbins=True)
    else:
        bin_edges = bins

    # Ensure edges cover full range
    bin_edges = list(bin_edges)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Bin both distributions
    baseline_binned = pd.cut(baseline_clean, bins=bin_edges, include_lowest=True)
    comparison_binned = pd.cut(comparison_clean, bins=bin_edges, include_lowest=True)

    # Calculate percentages
    baseline_pct = baseline_binned.value_counts(normalize=True).sort_index()
    comparison_pct = comparison_binned.value_counts(normalize=True).sort_index()

    # Align indices
    all_bins = baseline_pct.index.union(comparison_pct.index)
    baseline_pct = baseline_pct.reindex(all_bins, fill_value=0)
    comparison_pct = comparison_pct.reindex(all_bins, fill_value=0)

    # Apply minimum percentage to avoid log(0)
    baseline_pct = baseline_pct.clip(lower=min_pct)
    comparison_pct = comparison_pct.clip(lower=min_pct)

    # Calculate PSI components
    psi_components = (comparison_pct - baseline_pct) * np.log(comparison_pct / baseline_pct)
    psi_value = psi_components.sum()

    # Create breakdown DataFrame
    breakdown = pd.DataFrame(
        {
            "bin": [str(b) for b in all_bins],
            "baseline_pct": baseline_pct.values,
            "comparison_pct": comparison_pct.values,
            "psi_component": psi_components.values,
        }
    )

    return psi_value, breakdown


def calculate_psi_for_variable(
    baseline_df: pd.DataFrame, comparison_df: pd.DataFrame, variable: str, bins: int | list[float] = 10
) -> PSIResult:
    """
    Calculate PSI for a single variable.

    Args:
        baseline_df: Baseline DataFrame
        comparison_df: Comparison DataFrame
        variable: Column name to analyze
        bins: Number of bins or bin edges

    Returns:
        PSIResult with PSI value and breakdown
    """
    if variable not in baseline_df.columns:
        raise ValueError(f"Variable '{variable}' not found in baseline DataFrame")
    if variable not in comparison_df.columns:
        raise ValueError(f"Variable '{variable}' not found in comparison DataFrame")

    psi_value, breakdown = calculate_psi(baseline_df[variable], comparison_df[variable], bins=bins)

    return PSIResult(
        variable=variable,
        psi_value=psi_value,
        status=get_psi_status(psi_value),
        n_bins=len(breakdown),
        bin_details=breakdown,
    )


def calculate_stability_report(
    baseline_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    variables: list[str],
    baseline_name: str = "Baseline",
    comparison_name: str = "Comparison",
    bins: int | dict[str, int | list[float]] = 10,
    score_variable: str | None = None,
) -> StabilityReport:
    """
    Calculate PSI/CSI for multiple variables and generate a stability report.

    Args:
        baseline_df: Baseline DataFrame (e.g., Main period)
        comparison_df: Comparison DataFrame (e.g., MR period)
        variables: List of variables to analyze
        baseline_name: Name for baseline period
        comparison_name: Name for comparison period
        bins: Number of bins (int) or dict mapping variable names to bins
        score_variable: Optional main score variable for overall PSI

    Returns:
        StabilityReport with all PSI results
    """
    logger.info(f"Calculating stability metrics for {len(variables)} variables...")

    report = StabilityReport(
        baseline_name=baseline_name,
        comparison_name=comparison_name,
        baseline_count=len(baseline_df),
        comparison_count=len(comparison_df),
    )

    for var in variables:
        if var not in baseline_df.columns or var not in comparison_df.columns:
            logger.warning(f"Skipping variable '{var}' - not found in both DataFrames")
            continue

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(baseline_df[var]):
            logger.debug(f"Skipping non-numeric variable '{var}'")
            continue

        # Get bins for this variable
        var_bins = bins if isinstance(bins, int) else bins.get(var, 10)

        try:
            result = calculate_psi_for_variable(baseline_df, comparison_df, var, bins=var_bins)
            report.add(result)
            logger.debug(f"  {result}")
        except Exception as e:
            logger.warning(f"Failed to calculate PSI for '{var}': {e}")

    # Calculate overall PSI if score variable provided
    if score_variable and score_variable in baseline_df.columns:
        try:
            overall_psi, _ = calculate_psi(
                baseline_df[score_variable],
                comparison_df[score_variable],
                bins=20,  # More bins for overall score
            )
            report.overall_psi = overall_psi
        except Exception as e:
            logger.warning(f"Failed to calculate overall PSI: {e}")

    logger.info(report.summary())
    return report


def plot_psi_comparison(result: PSIResult, title: str | None = None) -> go.Figure:
    """
    Create a bar chart comparing baseline vs comparison distributions.

    Args:
        result: PSIResult with bin details
        title: Optional custom title

    Returns:
        Plotly Figure
    """
    df = result.bin_details

    if title is None:
        title = f"{result.variable} - PSI: {result.psi_value:.4f} ({result.status.value})"

    fig = go.Figure()

    # Baseline bars
    fig.add_trace(go.Bar(name="Baseline", x=df["bin"], y=df["baseline_pct"], marker_color="steelblue", opacity=0.7))

    # Comparison bars
    fig.add_trace(go.Bar(name="Comparison", x=df["bin"], y=df["comparison_pct"], marker_color="coral", opacity=0.7))

    fig.update_layout(
        title=title,
        xaxis_title="Bin",
        yaxis_title="Percentage",
        barmode="group",
        yaxis_tickformat=".1%",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template="plotly_white",
    )

    return fig


def plot_stability_dashboard(report: StabilityReport, top_n: int = 10) -> go.Figure:
    """
    Create a dashboard showing PSI values for all variables.

    Args:
        report: StabilityReport
        top_n: Number of top variables to show in detail

    Returns:
        Plotly Figure
    """
    df = report.to_dataframe()

    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No variables to display", x=0.5, y=0.5, showarrow=False)
        return fig

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=(f"PSI by Variable ({report.baseline_name} vs {report.comparison_name})", "PSI Distribution"),
        vertical_spacing=0.15,
    )

    # Color based on status
    colors = []
    for _, row in df.iterrows():
        if row["psi"] >= 0.25:
            colors.append("red")
        elif row["psi"] >= 0.1:
            colors.append("orange")
        else:
            colors.append("green")

    # Bar chart of PSI values
    fig.add_trace(
        go.Bar(
            x=df["variable"],
            y=df["psi"],
            marker_color=colors,
            name="PSI",
            text=[f"{v:.3f}" for v in df["psi"]],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # Add threshold lines
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Moderate (0.1)", row=1, col=1)
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Unstable (0.25)", row=1, col=1)

    # Histogram of PSI values
    fig.add_trace(go.Histogram(x=df["psi"], nbinsx=20, marker_color="steelblue", name="Distribution"), row=2, col=1)

    # Add threshold lines to histogram
    fig.add_vline(x=0.1, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_vline(x=0.25, line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_white",
        title=dict(
            text=f"Stability Report: {len(report.stable_vars)} stable, "
            f"{len(report.moderate_vars)} moderate, {len(report.unstable_vars)} unstable",
            x=0.5,
        ),
    )

    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_yaxes(title_text="PSI Value", row=1, col=1)
    fig.update_xaxes(title_text="PSI Value", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return fig


def compare_main_vs_mr(
    main_df: pd.DataFrame,
    mr_df: pd.DataFrame,
    variables: list[str],
    score_variable: str = "sc_octroi",
    output_path: str | None = None,
    verbose: bool = True,
) -> StabilityReport:
    """
    Convenience function to compare Main period vs MR period stability.

    Args:
        main_df: Main period DataFrame
        mr_df: MR period DataFrame
        variables: List of variables to analyze
        score_variable: Main score variable for overall PSI
        output_path: Optional path to save HTML dashboard
        verbose: Whether to print report

    Returns:
        StabilityReport
    """
    report = calculate_stability_report(
        baseline_df=main_df,
        comparison_df=mr_df,
        variables=variables,
        baseline_name="Main Period",
        comparison_name="MR Period",
        score_variable=score_variable,
    )

    if verbose:
        report.print_report()

    if output_path:
        fig = plot_stability_dashboard(report)
        fig.write_html(output_path)
        logger.info(f"Stability dashboard saved to {output_path}")

    return report


def calculate_csi_for_categorical(
    baseline: pd.Series, comparison: pd.Series, min_pct: float = 0.0001
) -> tuple[float, pd.DataFrame]:
    """
    Calculate Characteristic Stability Index for categorical variables.

    Args:
        baseline: Baseline categorical series
        comparison: Comparison categorical series
        min_pct: Minimum percentage to avoid division by zero

    Returns:
        Tuple of (CSI value, DataFrame with category-level breakdown)
    """
    # Get value counts as percentages
    baseline_pct = baseline.value_counts(normalize=True)
    comparison_pct = comparison.value_counts(normalize=True)

    # Align categories
    all_cats = baseline_pct.index.union(comparison_pct.index)
    baseline_pct = baseline_pct.reindex(all_cats, fill_value=0)
    comparison_pct = comparison_pct.reindex(all_cats, fill_value=0)

    # Apply minimum percentage
    baseline_pct = baseline_pct.clip(lower=min_pct)
    comparison_pct = comparison_pct.clip(lower=min_pct)

    # Calculate CSI (same formula as PSI)
    csi_components = (comparison_pct - baseline_pct) * np.log(comparison_pct / baseline_pct)
    csi_value = csi_components.sum()

    breakdown = pd.DataFrame(
        {
            "category": all_cats,
            "baseline_pct": baseline_pct.values,
            "comparison_pct": comparison_pct.values,
            "csi_component": csi_components.values,
        }
    )

    return csi_value, breakdown
