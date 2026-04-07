"""
Visualization functions for credit risk model analysis and reporting.

This module provides plotting functions for model evaluation and risk analysis:
- ROC curves and Gini visualization
- Precision-Recall curves
- KS statistic plots
- CAP curves and rejection analysis
- Score distribution comparison plots
- 3D surface plots for risk landscapes
- Interactive risk vs production visualizations
- Group statistics and confidence interval plots

Key components:
- plot_roc_curve: Plot ROC curve with Gini and KS annotations
- plot_precision_recall_curve: Plot precision-recall curve with AP annotation
- plot_score_distribution: Compare score distributions for goods vs bads
- visualize_metrics: Create comprehensive model evaluation dashboard
- plot_group_statistics: Visualize performance by risk groups
- RiskProductionVisualizer: Interactive risk/production trade-off visualization
- plot_risk_vs_production: Static risk vs production scatter plot
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

from . import styles
from .constants import DEFAULT_RISK_MULTIPLIER, Columns, StatusName
from .metrics import compute_precision_recall, ks_statistic
from .utils import calculate_b2_ever_h6


def plot_roc_curve(ax, y_true, scores, name, color):
    """Plot ROC curve on given axes."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    gini = 2 * auc(fpr, tpr) - 1
    ks = ks_statistic(y_true, scores)

    ax.plot(fpr, tpr, lw=2.5, label=f"GINI {name} ({gini:.3f})", color=color)
    ax.fill_between(fpr, tpr, color=color, alpha=0.05)
    ks_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[ks_idx], tpr[ks_idx], color="red", s=60, zorder=5)
    ax.annotate(
        f"KS={ks:.3f}",
        (fpr[ks_idx], tpr[ks_idx]),
        fontsize=12,
        xytext=(-80, -10),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->"),
    )


def visualize_metrics(y_true: np.ndarray, scores_dict: dict[str, np.ndarray], ax: list) -> None:
    """
    Create a comprehensive model evaluation dashboard with ROC and CAP curves.

    Generates four subplots comparing multiple scoring models:
    - Individual models ROC curves (ax[0])
    - Combined models ROC curves (ax[1])
    - Individual models CAP curves (ax[2])
    - Combined models CAP curves (ax[3])

    Models are classified as "combined" if their name contains "combined" (case-insensitive).

    Args:
        y_true: Binary array of true outcomes (1=bad, 0=good).
        scores_dict: Dictionary mapping model names to score arrays.
        ax: List of 4 matplotlib axes for the subplots.
    """
    styles.apply_matplotlib_style()
    palette = sns.color_palette("viridis", len(scores_dict))

    for (name, scores), color in zip(scores_dict.items(), palette):
        current_ax_roc = ax[0] if "combined" not in name.lower() else ax[1]
        current_ax_cap = ax[2] if "combined" not in name.lower() else ax[3]

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        gini = 2 * auc(fpr, tpr) - 1
        current_ax_roc.plot(fpr, tpr, lw=2.5, label=f"GINI {name} ({gini:.3f})", color=color)
        current_ax_roc.fill_between(fpr, tpr, color=color, alpha=0.05)

        # Plot CAP Curve
        y_true_array = np.array(y_true)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_true_values = y_true_array[sorted_indices]
        cumulative_true_positives = np.cumsum(sorted_true_values)
        total_true_positives = np.sum(y_true_array)
        current_ax_cap.plot(
            np.linspace(0, 1, len(y_true_array)),
            cumulative_true_positives / total_true_positives,
            lw=2.5,
            label=f"CAP {name}",
            color=color,
        )

    for axis in ax[:2]:
        axis.plot([0, 1], [0, 1], "k--", lw=1.5)  # Ideal line for ROC
        axis.set_xlabel("False Positive Rate", fontsize=14)
        axis.set_ylabel("True Positive Rate", fontsize=14)
        axis.legend(loc="lower right", fontsize=12)
        axis.set_ylim(0, 1.05)
        axis.grid(True, which="both", linestyle="--", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in ax[2:]:
        axis.plot([0, 1], [0, 1], "k--", lw=1.5)  # Ideal line for CAP
        axis.set_xlabel("Fraction of Population", fontsize=14)
        axis.set_ylabel("Fraction of Total Positives", fontsize=14)
        axis.legend(loc="lower right", fontsize=12)
        axis.set_ylim(0, 1.05)
        axis.grid(True, which="both", linestyle="--", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    ax[0].set_title("Individual Models ROC", fontsize=16)
    ax[1].set_title("Combined Models ROC", fontsize=16)
    ax[2].set_title("Individual Models CAP", fontsize=16)
    ax[3].set_title("Combined Models CAP", fontsize=16)


def plot_gini_confidence_intervals(ax, df):
    """
    Plots the Gini confidence intervals for each model.
    """
    # Sort dataframe by Gini Score for a better visualization
    df = df.sort_values(by="Gini Score")

    # Extract lower and upper bounds of confidence intervals
    lower_bounds = [ci[0] for ci in df["Gini CI"]]
    upper_bounds = [ci[1] for ci in df["Gini CI"]]

    # Create a range for the y axis
    y_range = list(range(len(df)))

    ax.errorbar(
        df["Gini Score"],
        y_range,
        xerr=[df["Gini Score"] - lower_bounds, upper_bounds - df["Gini Score"]],
        fmt="o",
        color=styles.COLOR_ACCENT,
        ecolor=styles.COLOR_SECONDARY,
        elinewidth=3,
        capsize=5,
        markersize=8,
    )

    # Annotate with model names
    for y, model in enumerate(df["Model"]):
        ax.annotate(model, (0.05, y), color="black", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("Gini Score")
    ax.set_title("Gini Confidence Intervals", fontsize=16)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()  # Higher scores at the top


def plot_precision_recall_curve(
    ax,
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
) -> None:
    """
    Plot precision-recall curves for one or more models.

    Precision-recall curves are more informative than ROC curves when
    evaluating models on imbalanced datasets (e.g., low default rates).

    Args:
        ax: Matplotlib axes to plot on.
        y_true: Binary array of true outcomes (1=bad, 0=good).
        scores_dict: Dictionary mapping model names to score arrays.
    """
    palette = sns.color_palette("viridis", len(scores_dict))

    for (name, scores), color in zip(scores_dict.items(), palette):
        precision, recall, _, ap = compute_precision_recall(y_true, scores)
        ax.plot(recall, precision, lw=2.5, label=f"{name} (AP={ap:.3f})", color=color)
        ax.fill_between(recall, precision, color=color, alpha=0.05)

    # Baseline: fraction of positives (random classifier)
    baseline = np.asarray(y_true).mean()
    ax.axhline(y=baseline, linestyle="--", color="grey", lw=1.5, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Curve", fontsize=16)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_score_distribution(
    ax,
    y_true: np.ndarray,
    scores: np.ndarray,
    name: str = "Score",
    bins: int = 50,
) -> None:
    """
    Plot overlapping score distributions for goods (0) vs bads (1).

    Useful for visual assessment of model discrimination: well-separated
    distributions indicate a strong model. Also shows the KDE overlay.

    Args:
        ax: Matplotlib axes to plot on.
        y_true: Binary array of true outcomes (1=bad, 0=good).
        scores: Model scores.
        name: Label for the score.
        bins: Number of histogram bins.
    """
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)

    goods = scores_arr[y_true_arr == 0]
    bads = scores_arr[y_true_arr == 1]

    ax.hist(goods, bins=bins, density=True, alpha=0.4, color=styles.COLOR_GOOD, label="Good (0)")
    ax.hist(bads, bins=bins, density=True, alpha=0.4, color=styles.COLOR_BAD, label="Bad (1)")

    # KDE overlays
    if len(goods) > 1:
        sns.kdeplot(goods, ax=ax, color=styles.COLOR_GOOD, lw=2)
    if len(bads) > 1:
        sns.kdeplot(bads, ax=ax, color=styles.COLOR_BAD, lw=2)

    ax.set_xlabel(name, fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"Score Distribution: {name}", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_group_statistics(data_frame, group_col, binary_outcome):
    """
    Plot the risk rate and frequency for each group.
    """
    grouped = data_frame.groupby(group_col).agg({binary_outcome: ["mean", "count"]})
    grouped.columns = ["Target_mean", "Target_count"]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = styles.COLOR_ACCENT
    ax1.set_xlabel("Groups")
    ax1.set_ylabel("Frequency", color=color)
    ax1.bar(grouped.index.astype(str), grouped["Target_count"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_title("Risk rate per groups")
    ax1.set_xticklabels(grouped.index, rotation=45)

    ax2 = ax1.twinx()
    color = styles.COLOR_RISK
    ax2.set_ylabel("Risk Rate", color=color)
    ax2.plot(grouped.index.astype(str).to_numpy(), grouped["Target_mean"].to_numpy(), color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    return fig


def plot_3d_graph(
    data_train: pd.DataFrame, data_surf: pd.DataFrame, variables: list[str], var_target: str
) -> go.Figure:
    """
    Create a 3D visualization of regression predictions with actual data points.

    Displays:
    - Blue scatter points: Training data (actual values)
    - Red scatter points: Outliers/test data
    - Viridis surface: Regression model predictions

    Args:
        data_train: Training data with actual target values (filtered/clean).
        data_surf: Full data including outliers and predicted values.
            Must contain '{var_target}_pred' column with predictions.
        variables: List of two variable names [var0, var1] for X and Y axes.
        var_target: Name of the target variable for Z axis.

    Returns:
        Plotly Figure object with 3D surface and scatter plots.
    """
    if len(variables) > 2:
        logger.info(f"plot_3d_graph: N>2 variables ({len(variables)}), marginalizing to first 2 for surface")
        agg_cols = [f"{var_target}_pred", var_target]
        agg_cols = [c for c in agg_cols if c in data_surf.columns]
        data_surf = data_surf.groupby(variables[:2])[agg_cols].mean().reset_index()
        data_train = data_train.groupby(variables[:2])[[var_target]].mean().reset_index()
        variables = variables[:2]

    if len(variables) == 1:
        # 1D: line chart instead of 3D surface
        var0 = variables[0]
        pred_col = f"{var_target}_pred"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_train[var0], y=data_train[var_target],
            mode="markers", name=f"{var_target} (actual)",
            marker=dict(size=8, color="blue", opacity=0.7),
        ))
        if pred_col in data_surf.columns:
            surf_sorted = data_surf.sort_values(var0)
            fig.add_trace(go.Scatter(
                x=surf_sorted[var0], y=surf_sorted[pred_col],
                mode="lines+markers", name="Regression",
                line=dict(color="red", width=2),
            ))
        styles.apply_plotly_style(fig, title="Regression Plot (1D)", width=1000, height=600)
        fig.update_layout(xaxis_title=var0, yaxis_title=var_target)
        return fig

    data_surf_pivot = data_surf.pivot(index=variables[1], columns=variables[0], values=f"{var_target}_pred")

    # gráfico
    fig = go.Figure()

    # Scatter plot for outliers
    fig = fig.add_trace(
        go.Scatter3d(
            x=data_surf[variables[0]],
            y=data_surf[variables[1]],
            z=data_surf[var_target],
            name=f"{var_target} (outliers)",
            showlegend=True,
            mode="markers",
            marker=dict(size=5, color="red", opacity=0.5),
        )
    )
    # Scatter plot for actual data
    fig = fig.add_trace(
        go.Scatter3d(
            x=data_train[variables[0]],
            y=data_train[variables[1]],
            z=data_train[var_target],
            name=var_target,
            showlegend=True,
            mode="markers",
            marker=dict(size=5, color="blue", opacity=0.5),
        )
    )

    fig = fig.add_trace(
        go.Surface(
            x=data_surf_pivot.columns,
            y=data_surf_pivot.index,
            z=data_surf_pivot.values,
            colorscale="Viridis",
            name="Regression",
            showlegend=True,
            opacity=0.7,
        )
    )
    styles.apply_plotly_style(fig, title="Regression Plot", width=1200, height=900)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=variables[0], gridcolor="white"),
            yaxis=dict(title=variables[1], gridcolor="white"),
            zaxis=dict(title=var_target, gridcolor="white"),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        legend=dict(y=1, x=0.8),
    )

    return fig


class RiskProductionVisualizer:
    """
    Interactive visualization for risk vs production trade-off analysis.

    Creates a two-panel Plotly figure:
    - Left panel: Scatter plot showing optimal frontier of risk vs production
    - Right panel: Heatmap showing risk levels with optimal cut-off overlay

    The visualization helps identify the optimal balance between portfolio
    production (revenue) and risk levels based on feasible cut-off solutions.

    Attributes:
        data_summary: Summary DataFrame with optimal solutions.
        data_summary_disaggregated: Disaggregated data by variable combinations.
        optimum_risk: Target risk threshold for selecting optimal solution.
        B2_0: Actual (current) B2 risk metric.
        OA_0: Actual (current) production amount.
    """

    def __init__(
        self,
        data_summary: pd.DataFrame,
        data_summary_disaggregated: pd.DataFrame,
        data_summary_sample_no_opt: pd.DataFrame,
        variables: list[str],
        optimum_risk: float,
        tasa_fin: float,
        *,
        values_per_var: dict[str, list | np.ndarray] | None = None,
        values_var0: np.ndarray | list | None = None,
        values_var1: np.ndarray | list | None = None,
        target_sol_fac: int | None = None,
        directions: dict[str, int] | None = None,
        pareto_masks: list[np.ndarray] | None = None,
        grid: Any | None = None,
        multiplier: float = DEFAULT_RISK_MULTIPLIER,
        total_demand: float | None = None,
    ):
        """
        Initialize the RiskProductionVisualizer.

        Args:
            data_summary: DataFrame with optimal solutions containing columns:
                b2_ever_h6, oa_amt_h0, and cut values for each bin.
            data_summary_disaggregated: Disaggregated booked data with '_boo' suffixes.
            data_summary_sample_no_opt: Sample of non-optimal solutions for context.
            variables: List of variable names (at least 2).
            optimum_risk: Target risk level (%) for automatic solution selection.
            tasa_fin: Financing rate (used for repesca calculations).
            values_per_var: Dict mapping variable names to sorted bin values.
                Preferred over values_var0/values_var1.
            values_var0: (Legacy) Array of bin values for first variable.
            values_var1: (Legacy) Array of bin values for second variable.
            target_sol_fac: Optional specific solution ID to select instead of
                using optimum_risk threshold.
            multiplier: Risk multiplier for b2_ever_h6 calculation.
        """
        self.data_summary = data_summary
        self.data_summary_disaggregated = data_summary_disaggregated
        self.data_summary_sample_no_opt = data_summary_sample_no_opt
        self.variables = variables
        self._is_1d = len(self.variables) == 1
        self.optimum_risk = optimum_risk
        self.tasa_fin = tasa_fin

        # Derive values_var0/values_var1 from values_per_var or legacy params
        if values_per_var is not None:
            self.values_per_var = values_per_var
            self.values_var0 = values_per_var[variables[0]]
            self.values_var1 = values_per_var[variables[1]] if len(variables) > 1 else []
        else:
            self.values_var0 = values_var0 if values_var0 is not None else []
            self.values_var1 = values_var1 if values_var1 is not None else []
            self.values_per_var = {variables[0]: self.values_var0}
            if len(variables) > 1:
                self.values_per_var[variables[1]] = self.values_var1

        self.target_sol_fac = target_sol_fac
        self.directions = directions or {}
        self._pareto_masks = pareto_masks or []
        self._nd_grid = grid
        self.multiplier = multiplier
        self._total_demand = total_demand

        # Calculate initial metrics
        self.calculate_initial_metrics()

        # Create the visualization
        self.create_figure()
        self.create_slider()

    def calculate_initial_metrics(self):
        """Calculate initial B2 and OA metrics"""
        # Calculate B2_0
        tudu_30_ever = self.data_summary_disaggregated["todu_30ever_h6_boo"].sum()
        tudu_amt_pile = self.data_summary_disaggregated["todu_amt_pile_h6_boo"].sum()

        # Store raw actuals
        self.actual_todu_30 = tudu_30_ever
        self.actual_todu_amt = tudu_amt_pile

        self.B2_0 = calculate_b2_ever_h6(tudu_30_ever, tudu_amt_pile, multiplier=self.multiplier, as_percentage=True, decimals=2)

        # Calculate OA_0
        self.OA_0 = self.data_summary_disaggregated["oa_amt_h0_boo"].sum()

        # Calculate limits
        self.lim_sup_B2 = self.data_summary["b2_ever_h6"].max()
        self.lim_inf_B2 = 0
        self.lim_sup_OA = self.data_summary["oa_amt_h0"].max()
        self.lim_inf_OA = 0

        # For N>2 variables: marginalize disaggregated data to first 2 variables for heatmap
        self._is_2d = len(self.variables) == 2
        if self._is_1d:
            # 1D mode: no heatmap, aggregate to var0 only
            var0 = self.variables[0]
            numeric_cols = [
                c
                for c in self.data_summary_disaggregated.columns
                if c != var0 and pd.api.types.is_numeric_dtype(self.data_summary_disaggregated[c])
            ]
            agg_dict = dict.fromkeys(numeric_cols, "sum")
            self._bar_data = self.data_summary_disaggregated.groupby([var0]).agg(agg_dict).reset_index()
            self._bar_data["b2_ever_h6"] = calculate_b2_ever_h6(
                self._bar_data["todu_30ever_h6"],
                self._bar_data["todu_amt_pile_h6"],
                multiplier=self.multiplier,
                as_percentage=True,
            )
            self._heatmap_data = None
            self.xx, self.yy = None, None
            logger.info(f"1D mode: bar chart for {var0}")
        elif not self._is_2d:
            var0, var1 = self.variables[0], self.variables[1]
            numeric_cols = [
                c
                for c in self.data_summary_disaggregated.columns
                if c not in self.variables and pd.api.types.is_numeric_dtype(self.data_summary_disaggregated[c])
            ]
            agg_dict = dict.fromkeys(numeric_cols, "sum")
            self._heatmap_data = self.data_summary_disaggregated.groupby([var0, var1]).agg(agg_dict).reset_index()
            # Recompute b2 and text for marginalized data
            self._heatmap_data["b2_ever_h6"] = calculate_b2_ever_h6(
                self._heatmap_data["todu_30ever_h6"],
                self._heatmap_data["todu_amt_pile_h6"],
                multiplier=self.multiplier,
                as_percentage=True,
            )
            self._heatmap_data["text"] = self._heatmap_data.apply(
                lambda x: f"{x['oa_amt_h0'] / 1e6:,.2f}M {x['b2_ever_h6'] / 100:.2%}", axis=1
            )
            logger.info(
                f"N>2 variables ({len(self.variables)}): heatmap marginalized to "
                f"{var0} x {var1} (extra dimensions summed)"
            )
            # Create meshgrid for heatmap
            self.xx, self.yy = np.meshgrid(self.values_var0, self.values_var1)
        else:
            self._heatmap_data = self.data_summary_disaggregated
            # Create meshgrid for heatmap
            self.xx, self.yy = np.meshgrid(self.values_var0, self.values_var1)

    def create_figure(self):
        """Create the main figure with subplots"""
        if self._is_1d:
            self.fig = go.Figure(
                make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        "Optimal relationship between production and risk",
                        f"Risk & Production by {self.variables[0]}",
                    ),
                )
            )
        else:
            self.fig = go.Figure(
                make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=(
                        "Optimal relationship between production and risk",
                        "Production & B2 ever H6 Heatmap / Optimal Cut-Off",
                    ),
                )
            )

        self._add_scatter_traces()
        if self._is_1d:
            self._add_bar_traces()
        else:
            self._add_heatmap_traces()
        self._update_layout()

    def _add_scatter_traces(self):
        """Add scatter plot traces to the figure"""
        # Non-optimal solutions
        self.fig.add_trace(
            go.Scatter(
                x=self.data_summary_sample_no_opt["oa_amt_h0"],
                y=self.data_summary_sample_no_opt["b2_ever_h6"],
                name="Non-optimal solutions",
                mode="markers",
                marker=dict(size=1.5, color="rgba(209,142,145,0.5)"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Optimal line
        self.fig.add_trace(
            go.Scatter(
                x=self.data_summary["oa_amt_h0"],
                y=self.data_summary["b2_ever_h6"],
                name="Optimal line",
                marker=dict(size=0),
                line=dict(color="rgba(157,13,20,.8)", width=2),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Actual point
        self.fig.add_trace(
            go.Scatter(
                x=[self.OA_0],
                y=[self.B2_0],
                name="Actual",
                mode="markers",
                marker=dict(size=8, color="rgba(0,0,0,1)"),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Optimum selected point
        self.optimum_trace = go.Scatter(
            x=[0],
            y=[0],
            name="Optimum selected",
            mode="markers",
            marker=dict(size=8, color="rgba(33,150,243,1)", line=dict(width=0.2, color="black")),
            showlegend=True,
        )
        self.fig.add_trace(self.optimum_trace, row=1, col=1)

    def _add_bar_traces(self):
        """Add bar chart traces for 1D mode (replaces heatmap)"""
        bar_data = self._bar_data
        var0 = self.variables[0]
        bins = bar_data[var0].values

        # Risk bars
        self.fig.add_trace(
            go.Bar(
                x=bins,
                y=bar_data["b2_ever_h6"],
                name="B2 ever H6 (%)",
                marker_color="rgba(157,13,20,0.7)",
                text=[f"{v:.1f}%" for v in bar_data["b2_ever_h6"]],
                textposition="outside",
            ),
            row=1,
            col=2,
        )

        # Mask overlay bar — will be updated by _apply_static_update
        self._bar_mask_colors = ["rgba(0,0,0,0)"] * len(bins)
        self.mask_bar_trace = go.Bar(
            x=bins,
            y=bar_data["b2_ever_h6"],
            marker_color=self._bar_mask_colors,
            marker_pattern_shape="/",
            showlegend=False,
            hoverinfo="skip",
        )
        self.fig.add_trace(self.mask_bar_trace, row=1, col=2)

    def _add_heatmap_traces(self):
        """Add heatmap traces to the figure"""
        heatmap_data = self._heatmap_data
        # Main heatmap
        self.fig.add_trace(
            go.Heatmap(
                x=heatmap_data[self.variables[0]],
                y=heatmap_data[self.variables[1]],
                z=heatmap_data["b2_ever_h6"],
                colorscale=[(0.00, "rgba(255,255,255,1)"), (1.00, "rgba(157,13,20,1)")],
                zmin=0,
                zmax=20,
                text=heatmap_data["text"],
                texttemplate="%{text}",
            ),
            row=1,
            col=2,
        )

        # Mask heatmap — rejected cells get a dark red overlay, accepted are transparent
        self.mask_trace = go.Heatmap(
            x=self.values_var0,
            y=self.values_var1,
            z=np.zeros_like(self.yy),
            colorscale=[
                (0.00, "rgba(120,0,0,0.55)"),
                (1.00, "rgba(0,0,0,0)"),
            ],
            showscale=False,
        )
        self.fig.add_trace(self.mask_trace, row=1, col=2)

    def _update_layout(self):
        """Update the figure layout"""
        styles.apply_plotly_style(self.fig, width=1700, height=600)
        self.fig.update_layout(legend=dict(y=0.1, x=0.33))

        # Update axes
        var0_dir = self.directions.get(self.variables[0], 1)
        var0_label = f"{self.variables[0]} (Higher = {'Riskier' if var0_dir == 1 else 'Safer'})"

        self.fig.update_xaxes(title_text="OA AMT (€)", range=[self.lim_inf_OA, self.lim_sup_OA], row=1, col=1)
        self.fig.update_yaxes(title_text="B2 ever H6 (%€)", range=[self.lim_inf_B2, self.lim_sup_B2], row=1, col=1)

        if self._is_1d:
            self.fig.update_xaxes(title_text=var0_label, dtick=1, row=1, col=2)
            self.fig.update_yaxes(title_text="B2 ever H6 (%)", row=1, col=2)
        else:
            var1_dir = self.directions.get(self.variables[1], 1)
            var1_label = f"{self.variables[1]} (Higher = {'Riskier' if var1_dir == 1 else 'Safer'})"
            self.fig.update_xaxes(title_text=var0_label, dtick=1, row=1, col=2)
            self.fig.update_yaxes(title_text=var1_label, dtick=1, row=1, col=2)

        # Apply the update logic statically here
        self._apply_static_update()

    def _get_selected_solution_row(self):
        """Helper to find the selected optimal solution based on cz2024 or target_sol_fac"""
        if self.target_sol_fac is not None:
            # Find solution with sol_fac matching target_sol_fac
            if "sol_fac" in self.data_summary.columns:
                data_filtered = self.data_summary[self.data_summary["sol_fac"] == self.target_sol_fac]
                if not data_filtered.empty:
                    return data_filtered
            else:
                # Fallback if sol_fac is not index/column (it really should be there)
                pass

        # Exclude NaN-risk solutions (zero-exposure cells, not "zero risk")
        valid = self.data_summary["b2_ever_h6"].notna()
        candidates = self.data_summary[valid]
        if candidates.empty:
            candidates = self.data_summary

        b2_col = candidates["b2_ever_h6"]
        data_filtered = candidates[b2_col <= self.optimum_risk]
        if data_filtered.empty:
            min_b2 = b2_col.min()
            max_b2 = b2_col.max()
            logger.warning(
                f"No Pareto solution with b2_ever_h6 <= {self.optimum_risk:.2f}%. "
                f"Pareto b2 range: [{min_b2:.2f}%, {max_b2:.2f}%]. "
                f"Falling back to minimum-risk solution (b2={min_b2:.2f}%). "
                f"Consider increasing optimum_risk in config.toml."
            )
            data_filtered = candidates.sort_values("b2_ever_h6").head(1)
        else:
            # Explicitly select max production under risk cap (#30)
            data_filtered = data_filtered.sort_values(
                ["b2_ever_h6", "oa_amt_h0"]
            ).tail(1)
        return data_filtered

    def _apply_static_update(self):
        """Update the visualization based on selected solution"""
        # Filter data based on risk level
        data_filtered = self._get_selected_solution_row()

        metrics = data_filtered[
            ["b2_ever_h6", "oa_amt_h0", "b2_ever_h6_cut", "oa_amt_h0_cut", "b2_ever_h6_rep", "oa_amt_h0_rep"]
        ].values[0, :]
        B2, OA, B2_CUT, OA_CUT, B2_REP, OA_REP = metrics

        if self._is_1d:
            # 1D mode: resolve mask per bin from pareto_masks or cutoff columns
            var0 = self.variables[0]
            mask_1d = np.zeros(len(self.values_var0))
            if self._pareto_masks and self._nd_grid is not None:
                sol_fac = int(data_filtered.iloc[0].get("sol_fac", 0))
                if 0 <= sol_fac < len(self._pareto_masks):
                    selected_mask = self._pareto_masks[sol_fac]
                    cell_df = self._nd_grid.cell_data.copy()
                    cell_df["_m"] = selected_mask
                    for i, bv in enumerate(self.values_var0):
                        row = cell_df[cell_df[var0] == bv]
                        if not row.empty:
                            mask_1d[i] = row["_m"].values[0]

            # Update bar mask colors: rejected = semi-transparent dark red hatching
            colors = []
            for m in mask_1d:
                if m == 1:
                    colors.append("rgba(0,0,0,0)")  # accepted — transparent
                else:
                    colors.append("rgba(120,0,0,0.55)")  # rejected — dark overlay
            # Trace indices: scatter adds 4 (0,1,2,3), bar adds 2 (4=risk bars, 5=mask bars)
            self.fig.data[5].marker.color = colors
            self.fig.data[3].update(x=[OA], y=[B2])
            return

        # Update cut-off mask (2D+ modes)
        if self._is_2d:
            CUT_OFF = data_filtered[self.values_var0].values[0]
            var1_inverted = self.directions.get(self.variables[1], 1) == -1
            if var1_inverted:
                z_mask = (self.yy >= CUT_OFF) * 1
            else:
                z_mask = (self.yy <= CUT_OFF) * 1
        elif self._pareto_masks and self._nd_grid is not None:
            # N>2: resolve selected mask from sol_fac, then project onto var0 × var1
            sol_fac = int(data_filtered.iloc[0].get("sol_fac", 0))
            if 0 <= sol_fac < len(self._pareto_masks):
                selected_mask = self._pareto_masks[sol_fac]
                var0, var1 = self.variables[0], self.variables[1]
                cell_df = self._nd_grid.cell_data.copy()
                cell_df["_m"] = selected_mask
                # For each (var0, var1) pair, accepted if ANY higher-dim cell is accepted
                proj = cell_df.groupby([var0, var1])["_m"].max().reset_index()
                proj_pivot = (
                    proj.pivot(index=var1, columns=var0, values="_m")
                    .reindex(index=self.values_var1, columns=self.values_var0)
                    .fillna(0)
                )
                z_mask = proj_pivot.values
            else:
                z_mask = np.zeros_like(self.yy)
        else:
            z_mask = np.zeros_like(self.yy)

        # Trace indices: _add_scatter_traces adds 4 traces (0, 1, 2, 3)
        # _add_heatmap_traces adds 2 traces (4, 5)
        self.fig.data[5].z = z_mask
        self.fig.data[3].update(x=[OA], y=[B2])

        # Draw cutoff boundary line on the heatmap (col=2 → xaxis2/yaxis2)
        # Remove previous cutoff shapes (tagged with name)
        self.fig.layout.shapes = [s for s in (self.fig.layout.shapes or []) if getattr(s, "name", None) != "_cutoff"]

        boundary_x, boundary_y = self._compute_cutoff_boundary(z_mask)
        if boundary_x:
            self.fig.add_trace(
                go.Scatter(
                    x=boundary_x,
                    y=boundary_y,
                    mode="lines",
                    line=dict(color="rgba(0,200,80,0.9)", width=3, dash="solid"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )

    def _compute_cutoff_boundary(self, z_mask):
        """Compute a step-function boundary path between accepted (1) and rejected (0) cells.

        Returns (x_coords, y_coords) for a line tracing the cutoff edge on the heatmap.
        """
        v0 = list(self.values_var0)
        v1 = list(self.values_var1)
        if len(v0) < 2 or len(v1) < 2:
            return [], []

        # Half-step sizes for cell boundaries
        dx = (v0[1] - v0[0]) / 2 if len(v0) > 1 else 0.5
        dy = (v1[1] - v1[0]) / 2 if len(v1) > 1 else 0.5

        var1_inverted = self.directions.get(self.variables[1], 1) == -1

        # z_mask shape: (len(v1), len(v0)) — rows=var1, cols=var0
        path_x = []
        path_y = []

        for ci in range(len(v0)):
            if var1_inverted:
                # Inverted: accepted cells are at the top; find the lowest accepted row
                min_accepted = len(v1)
                for ri in range(len(v1)):
                    if z_mask[ri][ci] == 1:
                        min_accepted = ri
                        break
                if min_accepted < len(v1):
                    boundary_y = v1[min_accepted] - dy
                else:
                    boundary_y = v1[-1] + dy
            else:
                # Non-inverted: accepted cells are at the bottom; find the highest accepted row
                max_accepted = -1
                for ri in range(len(v1)):
                    if z_mask[ri][ci] == 1:
                        max_accepted = ri
                if max_accepted >= 0:
                    boundary_y = v1[max_accepted] + dy
                else:
                    boundary_y = v1[0] - dy

            left_x = v0[ci] - dx
            right_x = v0[ci] + dx

            # Add horizontal segment across this column
            if path_x:
                # Vertical connector from previous boundary_y to this one
                path_x.append(left_x)
                path_y.append(path_y[-1])
                path_x.append(left_x)
                path_y.append(boundary_y)
            else:
                path_x.append(left_x)
                path_y.append(boundary_y)

            path_x.append(right_x)
            path_y.append(boundary_y)

        return path_x, path_y

    def create_slider(self):
        """Slider creation is handled inline by create_figure; kept for API compat."""
        pass

    def display(self):
        """Display the visualization"""
        self.fig.show()

    def save_html(self, path: str):
        """Save the figure to an HTML file"""
        self.fig.write_html(path)

    def get_selected_solution(self):
        """Return the selected optimal solution dataframe row"""
        return self._get_selected_solution_row()

    def get_summary_table(self):
        """Return a DataFrame with the performance metrics"""
        # Get selected solution
        data_filtered = self._get_selected_solution_row()

        metrics = data_filtered[
            ["b2_ever_h6", "oa_amt_h0", "b2_ever_h6_cut", "oa_amt_h0_cut", "b2_ever_h6_rep", "oa_amt_h0_rep"]
        ].values[0, :]
        B2, OA, B2_CUT, OA_CUT, B2_REP, OA_REP = metrics

        # Extract raw risk components
        raw_metrics = data_filtered[
            [
                "todu_30ever_h6",
                "todu_amt_pile_h6",
                "todu_30ever_h6_cut",
                "todu_amt_pile_h6_cut",
                "todu_30ever_h6_rep",
                "todu_amt_pile_h6_rep",
            ]
        ].values[0, :]

        TODU30, TODU_AMT, TODU30_CUT, TODU_AMT_CUT, TODU30_REP, TODU_AMT_REP = raw_metrics

        # Total demand (through the door) = booked + all rejected + canceled
        if self._total_demand is not None and self._total_demand > 0:
            total_demand = self._total_demand
        else:
            # Backward compat fallback: booked + repesca
            total_rep = (
                self.data_summary_disaggregated["oa_amt_h0_rep"].sum()
                if "oa_amt_h0_rep" in self.data_summary_disaggregated.columns
                else 0.0
            )
            total_demand = self.OA_0 + total_rep

        # Rejection rate = rejected / total_demand (total_demand excludes canceled)
        actual_rej = (1 - self.OA_0 / total_demand) * 100 if total_demand > 0 else 0.0
        optimum_rej = (1 - OA / total_demand) * 100 if total_demand > 0 else 0.0

        # Construct DataFrame
        summary_data = {
            "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected", "Summary"],
            "Risk (%)": [
                max(self.B2_0, 0.000001),
                max(B2_REP, 0.000001),
                max(B2_CUT, 0.000001),
                max(B2, 0.000001),
                B2 - self.B2_0,
            ],
            "Production (€)": [self.OA_0, OA_REP, OA_CUT, OA, OA - self.OA_0],
            "Production (%)": [
                self.OA_0 / self.OA_0,
                OA_REP / self.OA_0,
                OA_CUT / self.OA_0,
                OA / self.OA_0,
                (OA - self.OA_0) / self.OA_0,
            ],
            "todu_30ever_h6": [self.actual_todu_30, TODU30_REP, TODU30_CUT, TODU30, TODU30 - self.actual_todu_30],
            "todu_amt_pile_h6": [
                self.actual_todu_amt,
                TODU_AMT_REP,
                TODU_AMT_CUT,
                TODU_AMT,
                TODU_AMT - self.actual_todu_amt,
            ],
            "Rejection Rate (%)": [actual_rej, None, None, optimum_rej, None],
            "Total Demand (€)": [total_demand, None, None, None, None],
        }

        df_summary = pd.DataFrame(summary_data)

        # Add a formatted column for display if needed or just return raw values
        # Let's keep it raw for programmatic use, but maybe add string formatting for print equivalent?
        # The user asked for "output table with relevant information". A DataFrame is best.

        return df_summary


def plot_risk_vs_production(
    data: pd.DataFrame,
    indicadores: list[str],
    comfort_zones: dict[int, float],  # CHANGED: Pass a dict {2022: 4.5, 2023: 4.0}
    data_booked: pd.DataFrame,
    rolling_window: int = 6,
    plot_width: int = 1500,
    plot_height: int = 500,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> go.Figure:
    """
    Creates an interactive plot comparing risk metrics against production data over time
    with dynamic Comfort Zone (CZ) thresholds.

    Args:
        comfort_zones: Dict mapping year (int) to CZ limit (float).
                       Example: {2022: 4.50, 2023: 4.25, 2024: 4.00}
    """

    # 1. Data Preparation
    # ---------------------------------------------------------
    # Ensure dates are datetime objects
    if not pd.api.types.is_datetime64_any_dtype(data[Columns.MIS_DATE]):
        data[Columns.MIS_DATE] = pd.to_datetime(data[Columns.MIS_DATE])

    # Filter indicators to only those present in the data
    available_indicators = [c for c in indicadores if c in data.columns]
    df_plot = (
        data[data[Columns.STATUS_NAME] == StatusName.BOOKED.value]
        .groupby([Columns.MIS_DATE], as_index=False)
        .agg(dict.fromkeys(available_indicators, "sum"))
    )

    # Calculate rolling sums (handling potential missing cols gracefully)
    risk_cols = [Columns.TODU_30EVER_H6, Columns.TODU_AMT_PILE_H6]
    if not all(col in df_plot.columns for col in risk_cols):
        raise ValueError(f"Data is missing required columns: {risk_cols}")

    # Create rolling columns
    for col in risk_cols:
        df_plot[f"{col}_MA{rolling_window}"] = df_plot[col].rolling(rolling_window).sum()

    # Calculate risk percentages
    df_plot[Columns.B2_EVER_H6] = calculate_b2_ever_h6(
        df_plot[Columns.TODU_30EVER_H6], df_plot[Columns.TODU_AMT_PILE_H6], multiplier=multiplier, as_percentage=True
    )

    df_plot["b2_ever_h6_MA"] = calculate_b2_ever_h6(
        df_plot[f"{Columns.TODU_30EVER_H6}_MA{rolling_window}"],
        df_plot[f"{Columns.TODU_AMT_PILE_H6}_MA{rolling_window}"],
        multiplier=multiplier,
        as_percentage=True,
    )

    # 2. Dynamic Comfort Zone Logic
    # ---------------------------------------------------------
    # Map the year of the date to the value in the dictionary
    df_plot["year"] = df_plot[Columns.MIS_DATE].dt.year
    df_plot["cz_target"] = df_plot["year"].map(comfort_zones)

    # 3. Plotting
    # ---------------------------------------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # -- Trace 1: Production (Area) --
    fig.add_trace(
        go.Scatter(
            x=df_plot["mis_date"],
            y=df_plot["oa_amt_h0"],
            mode="lines",
            line=dict(width=1, color=styles.COLOR_PRODUCTION),
            name="Production Amt",
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.2)",  # Production green with opacity
            hovertemplate="%{y:,.0f} €<extra></extra>",  # Cleaner tooltip
        ),
        secondary_y=False,
    )

    # -- Trace 2: Risk Percentage (Line) --
    fig.add_trace(
        go.Scatter(
            x=df_plot["mis_date"],
            y=df_plot["b2_ever_h6"],
            mode="lines",
            line=dict(width=2, color=styles.COLOR_RISK),
            name="Risk %",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # -- Trace 3: Moving Average (Dashed) --
    # Filter NaNs for cleaner line connection
    mask_ma = ~df_plot["b2_ever_h6_MA"].isna()
    fig.add_trace(
        go.Scatter(
            x=df_plot.loc[mask_ma, "mis_date"],
            y=df_plot.loc[mask_ma, "b2_ever_h6_MA"],
            mode="lines",
            line=dict(width=1.5, color="#555555", dash="dot"),
            name=f"Risk % (MA-{rolling_window})",
            hovertemplate="%{y:.2f}% (Avg)<extra></extra>",
        ),
        secondary_y=True,
    )

    # -- Trace 4: Comfort Zone (Dynamic) --
    # We plot this only where we have defined values
    mask_cz = ~df_plot["cz_target"].isna()
    if mask_cz.any():
        fig.add_trace(
            go.Scatter(
                x=df_plot.loc[mask_cz, "mis_date"],
                y=df_plot.loc[mask_cz, "cz_target"],
                mode="lines",
                # 'hv' shape creates a step function (changes exactly at year start)
                # Remove 'shape' if you prefer a slope transition between years
                line=dict(width=2, color=styles.COLOR_ACCENT, shape="hv"),
                name="Comfort Zone (Limit)",
                hovertemplate="Limit: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

    # 4. Highlight Selected Period
    # ---------------------------------------------------------
    if not data_booked.empty:
        # Check if dates are datetime, convert if not
        dates_bk = data_booked["mis_date"]
        if not pd.api.types.is_datetime64_any_dtype(dates_bk):
            dates_bk = pd.to_datetime(dates_bk)

        fig.add_vrect(
            x0=dates_bk.min() - pd.Timedelta(days=15),
            x1=dates_bk.max() + pd.Timedelta(days=15),
            fillcolor="rgba(0, 0, 255, 0.05)",  # Very light blue
            line_width=0,
            annotation_text="Booked Period",
            annotation_position="top left",
        )

    # 5. Layout Update
    # ---------------------------------------------------------
    # Calculate max ranges for dynamic axis scaling
    max_risk = max(df_plot["b2_ever_h6"].max(), df_plot["cz_target"].max() if "cz_target" in df_plot else 0)

    styles.apply_plotly_style(fig, width=plot_width, height=plot_height)

    fig.update_layout(
        title=dict(text="<b>Risk vs Production Analysis</b>", x=0.01),
        plot_bgcolor="rgba(255,255,255,1)",
        hovermode="x unified",  # Shows all values for a specific date at once
        xaxis=dict(title="Date", gridcolor="#f0f0f0", showgrid=True, zeroline=False),
        # Left Y-Axis (Usually Primary in specs, but you requested specific sides)
        # Note: In make_subplots(secondary_y=True),
        # yaxis = Left, yaxis2 = Right by default.
        # Below we map them to visually match your request.
        yaxis=dict(
            title="Production (€)",
            side="right",  # Production on Right
            showgrid=False,
            zeroline=False,
        ),
        yaxis2=dict(
            title="Risk Metric (%€)",
            side="left",  # Risk on Left
            gridcolor="#f0f0f0",
            range=[0, max_risk * 1.15],  # Add 15% headroom
            showgrid=True,
            zeroline=True,
            zerolinecolor="#dcdcdc",
        ),
        legend=dict(
            orientation="h",  # Horizontal legend saves vertical space
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: str | None = None,
) -> go.Figure:
    """
    Create a horizontal bar chart of mean |SHAP| values per feature.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: List of feature names.
        output_path: Optional path to save HTML file.

    Returns:
        Plotly Figure.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs)
    sorted_feature = [feature_names[i] for i in sorted_idx]
    sorted_shap = mean_abs[sorted_idx]

    fig = go.Figure(
        go.Bar(
            x=sorted_shap,
            y=sorted_feature,
            orientation="h",
            marker=dict(color=styles.COLOR_ACCENT, line=dict(color=styles.COLOR_PRIMARY, width=1)),
        )
    )

    styles.apply_plotly_style(
        fig, title="Feature Importance (SHAP)", width=1000, height=max(500, len(feature_names) * 25)
    )
    fig.update_layout(xaxis_title="mean(|SHAP value|)", yaxis_title="Feature")

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_data: pd.DataFrame,
    target_feature: str,
    output_path: str | None = None,
) -> go.Figure:
    """
    Create a SHAP dependence plot for a specific feature.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: List of feature names matching the columns in feature_data.
        feature_data: DataFrame containing the actual feature values.
        target_feature: The name of the feature to plot.
        output_path: Optional path to save HTML file.

    Returns:
        Plotly Figure.
    """
    if target_feature not in feature_names:
        raise ValueError(f"Feature '{target_feature}' not found in feature names.")

    feature_idx = feature_names.index(target_feature)
    target_shap_values = shap_values[:, feature_idx]
    target_feature_values = feature_data[target_feature].values

    fig = go.Figure(
        go.Scatter(
            x=target_feature_values,
            y=target_shap_values,
            mode="markers",
            marker=dict(
                size=6,
                color=styles.COLOR_PRIMARY,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=(f"<b>{target_feature}</b>: %{{x}}<br><b>SHAP Value</b>: %{{y:.4f}}<extra></extra>"),
        )
    )

    styles.apply_plotly_style(
        fig,
        title=f"SHAP Dependence: {target_feature}",
        width=900,
        height=500,
    )

    fig.update_layout(
        xaxis_title=target_feature,
        yaxis_title="SHAP Value",
        plot_bgcolor="white",
    )

    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    if output_path:
        fig.write_html(output_path)

    return fig


def _prepare_transformation_data(
    data: pd.DataFrame,
    date_col: str,
    n_months: int | None,
) -> pd.DataFrame:
    """Copy data, convert dates, filter last N months, filter eligible, mark booked."""
    df = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    if n_months is not None:
        max_date = df[date_col].max()
        cutoff_date = max_date - pd.DateOffset(months=n_months)
        df = df[df[date_col] >= cutoff_date]

    eligible_mask = df["se_decision_id"].isin(["ok", "rv"])
    df_eligible = df[eligible_mask].copy()
    df_eligible["is_booked"] = df_eligible[Columns.STATUS_NAME] == StatusName.BOOKED.value

    return df_eligible


def _calculate_monthly_rates(
    df_eligible: pd.DataFrame,
    amount_col: str,
    date_col: str,
) -> tuple[float, float, float, pd.DataFrame]:
    """Compute overall rate and monthly aggregation."""
    total_eligible = df_eligible[amount_col].sum()
    total_booked = df_eligible.loc[df_eligible["is_booked"], amount_col].sum()
    overall_rate = (total_booked / total_eligible) if total_eligible > 0 else 0

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
    monthly["plot_date"] = monthly["period"].dt.to_timestamp()

    return overall_rate, total_booked, total_eligible, monthly


def create_transformation_plot(
    monthly: pd.DataFrame,
    overall_rate: float,
    plot_width: int,
    plot_height: int,
) -> go.Figure:
    """Create dual-axis Plotly figure with bars + line + avg line + layout."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

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

    fig.add_hline(
        y=overall_rate,
        line_dash="dot",
        line_color=styles.COLOR_RISK,
        annotation_text=f"Avg: {overall_rate:.1%}",
        annotation_position="top left",
        secondary_y=False,
    )

    styles.apply_plotly_style(fig, width=plot_width, height=plot_height)
    fig.update_layout(
        title="<b>Monthly Transformation Rate</b><br><sup>(Booked / [OK + RV]) by Amount</sup>",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title="Transformation Rate",
            tickformat=".0%",
            range=[0, max(monthly["rate"].max() * 1.1, overall_rate * 1.1)],
            showgrid=True,
            gridcolor="lightgrey",
        ),
        yaxis2=dict(title="Eligible Volume (€)", showgrid=False, zeroline=False),
        xaxis=dict(title="Month", showgrid=False),
    )

    return fig


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

    df_eligible = _prepare_transformation_data(data, date_col, n_months)
    overall_rate, total_booked, total_eligible, monthly = _calculate_monthly_rates(df_eligible, amount_col, date_col)
    fig = create_transformation_plot(monthly, overall_rate, plot_width, plot_height)

    return {
        "overall_rate": overall_rate,
        "overall_booked_amt": total_booked,
        "overall_eligible_amt": total_eligible,
        "monthly_amounts": monthly.drop(columns=["plot_date"]),
        "figure": fig,
    }


def calculate_per_bin_transformation_rate(
    data: pd.DataFrame,
    variables: list[str],
    date_col: str = "mis_date",
    amount_col: str = "oa_amt",
    n_months: int | None = None,
    global_tasa_fin: float = 1.0,
    min_eligible_per_bin: int = 10,
) -> pd.DataFrame:
    """Compute per-bin transformation rates.

    For each bin combination, ``tasa_fin = booked_amount / eligible_amount``.
    Bins with fewer than *min_eligible_per_bin* eligible records fall back to
    *global_tasa_fin*.

    Returns
    -------
    DataFrame with ``[*variables, "tasa_fin"]``.
    """
    df_eligible = _prepare_transformation_data(data, date_col, n_months)

    rows: list[dict] = []
    for bin_vals, group in df_eligible.groupby(variables, observed=True):
        if not isinstance(bin_vals, tuple):
            bin_vals = (bin_vals,)
        row = dict(zip(variables, bin_vals))

        if len(group) < min_eligible_per_bin:
            row["tasa_fin"] = global_tasa_fin
        else:
            eligible_amt = group[amount_col].sum()
            booked_amt = group.loc[group["is_booked"], amount_col].sum()
            rate = booked_amt / eligible_amt if eligible_amt > 0 else global_tasa_fin
            row["tasa_fin"] = rate if 0 < rate <= 5.0 else global_tasa_fin

        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        logger.debug(
            f"Per-bin tasa_fin: {len(result)} bins | "
            f"mean={result['tasa_fin'].mean():.3f} | "
            f"min={result['tasa_fin'].min():.3f} | "
            f"max={result['tasa_fin'].max():.3f}"
        )
    return result


# =============================================================================
# Income bin diagnostics
# =============================================================================


def plot_income_bin_trends(
    data: pd.DataFrame,
    income_bin_col: str,
    date_col: str = "mis_date",
    target_col: str = "early_bad",
    amount_col: str = "oa_amt",
    output_path: str | None = None,
) -> go.Figure:
    """Plot volume and risk trends per income bin over time.

    Produces a two-subplot Plotly figure:
    - **Top**: Record count per income bin by month (grouped bars).
    - **Bottom**: Bad rate (``target_col``) per income bin by month (line chart).

    Parameters
    ----------
    data : pd.DataFrame
        Data with ``income_bin_col``, ``date_col``, ``target_col``, and ``amount_col``.
    income_bin_col : str
        Binned income column (e.g. ``"income_bin"``).
    date_col : str
        Date column (default ``"mis_date"``).
    target_col : str
        Binary target for bad rate (default ``"early_bad"``).
    amount_col : str
        Amount column for volume (default ``"oa_amt"``).
    output_path : str | None
        If provided, save figure as HTML.

    Returns
    -------
    go.Figure
    """
    df = data.copy()
    df["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)

    agg = (
        df.groupby(["_month", income_bin_col])
        .agg(count=(amount_col, "size"), total_amt=(amount_col, "sum"), bad_rate=(target_col, "mean"))
        .reset_index()
    )

    bins_sorted = sorted(agg[income_bin_col].unique())

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Volume per Income Bin", "Bad Rate per Income Bin"))

    for b in bins_sorted:
        sub = agg[agg[income_bin_col] == b].sort_values("_month")
        fig.add_trace(
            go.Bar(x=sub["_month"], y=sub["count"], name=f"Bin {b}", legendgroup=str(b)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["_month"],
                y=sub["bad_rate"],
                mode="lines+markers",
                name=f"Bin {b}",
                legendgroup=str(b),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(barmode="group", height=700, width=1000, title_text="Income Bin Diagnostics")
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Bad Rate", tickformat=".2%", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Income bin diagnostics saved to {output_path}")

    return fig


def _format_bin_edges(bin_edges: list[float]) -> list[str]:
    """Format bin edges into human-readable interval labels (1-indexed).

    Given edges ``[-inf, 350, 450, inf]`` returns:
    ``["1: (-inf, 350]", "2: (350, 450]", "3: (450, inf)"]``
    """
    labels = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        lo_s = f"{lo:,.1f}" if np.isfinite(lo) else "-inf"
        hi_s = f"{hi:,.1f}" if np.isfinite(hi) else "inf"
        left = "(" if np.isfinite(lo) else "("
        right = "]" if np.isfinite(hi) else ")"
        labels.append(f"{i + 1}: {left}{lo_s}, {hi_s}{right}")
    return labels


def plot_bin_threshold_diagnostic(
    data: pd.DataFrame,
    bin_col: str,
    date_col: str = "mis_date",
    todu_30_col: str = Columns.TODU_30EVER_H6,
    todu_amt_col: str = Columns.TODU_AMT_PILE_H6,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    bin_edges: list[float] | None = None,
    source_col: str | None = None,
    output_path: str | None = None,
) -> go.Figure:
    """Bin threshold diagnostic: monthly + full-period volume, B2EverH6, and risk separation.

    Validates that bin thresholds are correctly placed by checking:

    1. **Population share** — bins should stay roughly balanced over time.
    2. **Risk ordering** — bins should maintain monotone risk ordering across
       all months.  Lines that cross indicate a poorly placed threshold.
    3. **Risk separation** — the B2EverH6 ratio between adjacent bins should
       stay consistently above 1 (or below 1, depending on direction).

    A **"Full Period"** column is appended to the right of each panel,
    showing the overall aggregation across all months.

    Parameters
    ----------
    data : pd.DataFrame
        Record-level data with ``bin_col``, ``date_col``, ``todu_30_col``,
        and ``todu_amt_col``.
    bin_col : str
        Binned column name (e.g. ``"sc_octroi_new_clus"``).
    date_col : str
        Date column (default ``"mis_date"``).
    todu_30_col : str
        Numerator column for B2EverH6 (default ``"todu_30ever_h6"``).
    todu_amt_col : str
        Denominator column for B2EverH6 (default ``"todu_amt_pile_h6"``).
    multiplier : float
        Risk multiplier (default 7).
    bin_edges : list[float] | None
        Actual threshold values (e.g. ``[-inf, 350, 450, inf]``).
        Shown in legend labels and title when provided.
    source_col : str | None
        Raw column name that was binned (e.g. ``"score_rf"``).
        Used in title for context.
    output_path : str | None
        If provided, save figure as HTML.

    Returns
    -------
    go.Figure
    """
    df = data.copy()
    df["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)

    # Aggregate per month × bin
    agg = (
        df.groupby(["_month", bin_col])
        .agg(
            count=(date_col, "size"),
            todu_30=(todu_30_col, "sum"),
            todu_amt=(todu_amt_col, "sum"),
        )
        .reset_index()
    )

    # Full-period aggregate (all months combined)
    full = (
        df.groupby(bin_col)
        .agg(
            count=(date_col, "size"),
            todu_30=(todu_30_col, "sum"),
            todu_amt=(todu_amt_col, "sum"),
        )
        .reset_index()
    )
    full["_month"] = "Full Period"
    full["b2_ever_h6"] = calculate_b2_ever_h6(
        full["todu_30"], full["todu_amt"], multiplier=multiplier, as_percentage=True
    )
    full_total = full["count"].sum()
    full["share_pct"] = full["count"] / full_total * 100 if full_total > 0 else 0.0

    # Compute B2EverH6 per cell
    agg["b2_ever_h6"] = calculate_b2_ever_h6(agg["todu_30"], agg["todu_amt"], multiplier=multiplier, as_percentage=True)

    # Monthly totals for population share
    monthly_totals = agg.groupby("_month")["count"].transform("sum")
    agg["share_pct"] = agg["count"] / monthly_totals * 100

    # Combine monthly + full period
    agg = pd.concat([agg, full], ignore_index=True)

    bins_sorted = sorted(agg[bin_col].unique())
    n_bins = len(bins_sorted)
    colors = [f"hsl({int(i * 360 / max(n_bins, 1))}, 70%, 50%)" for i in range(n_bins)]

    # Build bin labels: include threshold ranges when bin_edges are provided
    edge_labels = _format_bin_edges(bin_edges) if bin_edges and len(bin_edges) > 1 else None
    bin_labels = {}
    for i, b in enumerate(bins_sorted):
        if edge_labels and i < len(edge_labels):
            bin_labels[b] = f"Bin {edge_labels[i]}"
        else:
            bin_labels[b] = f"Bin {b}"

    # Build title
    title_parts = [f"Bin Threshold Diagnostic — {bin_col}"]
    if source_col:
        title_parts.append(f"(source: {source_col})")
    if bin_edges:
        finite_edges = [e for e in bin_edges if np.isfinite(e)]
        if finite_edges:
            edge_str = ", ".join(f"{e:,.1f}" for e in finite_edges)
            title_parts.append(f"<br><sub>Thresholds: [{edge_str}]</sub>")
    title_text = " ".join(title_parts)

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            f"Population Share by {bin_col} (%)",
            f"B2EverH6 (%) by {bin_col}",
            "Risk Separation (adjacent bin ratio)",
        ),
        vertical_spacing=0.08,
    )

    months_sorted = sorted(m for m in agg["_month"].unique() if m != "Full Period")
    x_order = months_sorted + ["Full Period"]

    # --- Panel 1: Stacked area of population share ---
    for i, b in enumerate(bins_sorted):
        sub = agg[agg[bin_col] == b].set_index("_month").reindex(x_order).reset_index()
        fig.add_trace(
            go.Bar(
                x=sub["_month"],
                y=sub["share_pct"],
                name=bin_labels[b],
                legendgroup=str(b),
                marker_color=colors[i],
                hovertemplate=(
                    f"{bin_labels[b]}<br>%{{x}}<br>Share: %{{y:.1f}}%<br>Count: %{{customdata}}<extra></extra>"
                ),
                customdata=sub["count"].values,
            ),
            row=1,
            col=1,
        )

    # --- Panel 2: B2EverH6 per bin ---
    for i, b in enumerate(bins_sorted):
        sub = agg[agg[bin_col] == b].set_index("_month").reindex(x_order).reset_index()
        fig.add_trace(
            go.Scatter(
                x=sub["_month"],
                y=sub["b2_ever_h6"],
                mode="lines+markers",
                name=bin_labels[b],
                legendgroup=str(b),
                showlegend=False,
                line=dict(color=colors[i]),
                hovertemplate=f"{bin_labels[b]}<br>%{{x}}<br>B2EverH6: %{{y:.2f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # --- Panel 3: Risk separation ratio between adjacent bins ---
    if n_bins >= 2:
        pivot = agg.pivot_table(index="_month", columns=bin_col, values="b2_ever_h6")
        pivot = pivot.reindex(x_order)
        for j in range(len(bins_sorted) - 1):
            b_lo, b_hi = bins_sorted[j], bins_sorted[j + 1]
            if b_lo in pivot.columns and b_hi in pivot.columns:
                ratio = pivot[b_hi] / pivot[b_lo].replace(0, np.nan)
                fig.add_trace(
                    go.Scatter(
                        x=ratio.index.astype(str),
                        y=ratio.values,
                        mode="lines+markers",
                        name=f"{bin_labels[b_hi]} / {bin_labels[b_lo]}",
                        legendgroup=f"ratio_{j}",
                        hovertemplate=(
                            f"{bin_labels[b_hi]} / {bin_labels[b_lo]}<br>%{{x}}<br>Ratio: %{{y:.2f}}<extra></extra>"
                        ),
                    ),
                    row=3,
                    col=1,
                )

        # Reference line at ratio=1
        fig.add_trace(
            go.Scatter(
                x=x_order,
                y=[1.0] * len(x_order),
                mode="lines",
                line=dict(color="grey", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )

    # Vertical separator before "Full Period"
    for row in range(1, 4):
        fig.add_vline(
            x=len(months_sorted) - 0.5,
            line=dict(color="grey", dash="dot", width=1),
            row=row,
            col=1,
        )

    fig.update_layout(
        height=1000,
        width=1200,
        title_text=title_text,
        barmode="stack",
    )
    fig.update_yaxes(title_text="Share (%)", row=1, col=1)
    fig.update_yaxes(title_text="B2EverH6 (%)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=3, col=1)
    fig.update_xaxes(title_text="Period", row=3, col=1)
    # Force categorical x-axes so Plotly doesn't auto-parse period strings as dates
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=x_order)

    styles.apply_plotly_style(fig)

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Bin threshold diagnostic saved to {output_path}")

    return fig


def _add_nd_cutoff_boundary(
    fig: go.Figure,
    pivot_mask: "pd.DataFrame",
    x_labels: list[str],
    y_labels: list[str],
    row: int,
    col: int,
    var1_inverted: bool = False,
) -> None:
    """Draw a step-function boundary line between accepted and rejected cells on a categorical heatmap.

    For each column (var0 value), finds the boundary between accepted and rejected cells.
    For non-inverted var1: boundary sits above the top-most accepted cell.
    For inverted var1: boundary sits below the bottom-most accepted cell.
    The line uses categorical axis coordinates (integer positions where 0 = first category).
    """
    if len(x_labels) < 1 or len(y_labels) < 1:
        return

    # Build step path in numeric coordinates: cell centers at 0, 1, 2, ...
    # Cell boundaries at -0.5, 0.5, 1.5, ...
    path_x: list[float] = []
    path_y: list[float] = []
    z = pivot_mask.values  # shape (n_y, n_x)

    for ci in range(len(x_labels)):
        if var1_inverted:
            # Inverted: accepted cells at top, find lowest accepted row
            min_accepted = len(y_labels)
            for ri in range(len(y_labels)):
                if z[ri, ci] == 1:
                    min_accepted = ri
                    break
            boundary_y = min_accepted - 0.5 if min_accepted < len(y_labels) else len(y_labels) - 0.5
        else:
            max_accepted = -1
            for ri in range(len(y_labels)):
                if z[ri, ci] == 1:
                    max_accepted = ri
            boundary_y = max_accepted + 0.5 if max_accepted >= 0 else -0.5
        left_x = ci - 0.5
        right_x = ci + 0.5

        if path_x:
            path_x.append(left_x)
            path_y.append(path_y[-1])
            path_x.append(left_x)
            path_y.append(boundary_y)
        else:
            path_x.append(left_x)
            path_y.append(boundary_y)

        path_x.append(right_x)
        path_y.append(boundary_y)

    if not path_x:
        return

    # Resolve subplot axis references via Plotly's get_subplot
    subplot = fig.get_subplot(row, col)
    xaxis_name = subplot.xaxis.plotly_name  # "xaxis", "xaxis2", etc.
    yaxis_name = subplot.yaxis.plotly_name
    xref = xaxis_name.replace("axis", "")  # "x", "x2", etc.
    yref = yaxis_name.replace("axis", "")

    for i in range(len(path_x) - 1):
        fig.add_shape(
            type="line",
            x0=path_x[i],
            y0=path_y[i],
            x1=path_x[i + 1],
            y1=path_y[i + 1],
            xref=xref,
            yref=yref,
            line=dict(color="white", width=3),
        )


def plot_acceptance_grid_nd(
    mask: np.ndarray,
    grid: Any,
    output_path: str | None = None,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    inv_vars: list[str] | None = None,
) -> go.Figure:
    """Plot acceptance heatmaps for a 3+-variable grid with risk annotations.

    For each unique value of the 3rd variable, produces a var0 x var1 heatmap
    coloured by risk (b2_ever_h6) with an acceptance overlay showing accepted
    (blue border) vs rejected (red) cells.

    Parameters
    ----------
    mask : np.ndarray
        Binary acceptance mask aligned with ``grid.cell_index``.
    grid : CellGrid
        The N-dimensional cell grid.
    output_path : str | None
        If provided, save figure as HTML.

    Returns
    -------
    go.Figure
    """
    n_vars = len(grid.variables)
    if n_vars < 3:
        logger.warning("plot_acceptance_grid_nd: requires >= 3 variables, skipping")
        return go.Figure()

    var0, var1, var2 = grid.variables[0], grid.variables[1], grid.variables[2]
    v2_vals = grid.values_per_var[var2]

    fig = make_subplots(
        rows=2,
        cols=len(v2_vals),
        subplot_titles=[f"{var2} = {v}" for v in v2_vals] + [f"Risk — {var2} = {v}" for v in v2_vals],
        vertical_spacing=0.12,
    )

    cell_df = grid.cell_data.copy()
    cell_df["_mask"] = mask

    # Compute per-cell b2_ever_h6 for risk annotations
    if "todu_30ever_h6" in cell_df.columns and "todu_amt_pile_h6" in cell_df.columns:
        from .utils import calculate_b2_ever_h6

        raw_b2 = calculate_b2_ever_h6(
            cell_df["todu_30ever_h6"], cell_df["todu_amt_pile_h6"], multiplier=multiplier, as_percentage=True
        )
        cell_df["_b2_missing"] = raw_b2.isna()
        cell_df["_b2"] = raw_b2.fillna(0)
    else:
        cell_df["_b2"] = 0.0

    # Adaptive formatting based on grid size
    n_rows = len(grid.values_per_var[var1])
    n_cols_grid = len(grid.values_per_var[var0])
    n_cells = n_rows * n_cols_grid

    # Scale font size to grid dimensions so text fits inside cells
    if n_cells > 150:
        font_size = 7
    elif n_cells > 80:
        font_size = 8
    else:
        font_size = 10

    # Format production values with appropriate unit (k/M) based on magnitude
    def _fmt_production(val: float) -> str:
        if abs(val) >= 1e6:
            return f"{val / 1e6:,.1f}M"
        if abs(val) >= 1e3:
            return f"{val / 1e3:,.0f}k"
        return f"{val:,.0f}"

    # Compact cell text: production + risk only; ACC/REJ is conveyed by color
    if "oa_amt_h0" in cell_df.columns:
        cell_df["_text"] = cell_df.apply(
            lambda r: "N/A" if r.get("_b2_missing", False)
            else f"{_fmt_production(r['oa_amt_h0'])}<br>{r['_b2']:.1f}%",
            axis=1,
        )
    else:
        cell_df["_text"] = cell_df.apply(
            lambda r: "N/A" if r.get("_b2_missing", False) else f"{r['_b2']:.1f}%",
            axis=1,
        )

    # High-contrast discrete colorscale: rejected = muted red, accepted = green
    mask_colorscale = [
        (0.0, "rgb(210,70,70)"),
        (0.5, "rgb(210,70,70)"),
        (0.5, "rgb(50,160,80)"),
        (1.0, "rgb(50,160,80)"),
    ]

    for col_idx, v2 in enumerate(v2_vals, start=1):
        slice_df = cell_df[cell_df[var2] == v2]

        # Row 1: Acceptance mask
        pivot_mask = slice_df.pivot(index=var1, columns=var0, values="_mask").sort_index(ascending=True)
        pivot_text = slice_df.pivot(index=var1, columns=var0, values="_text").sort_index(ascending=True)

        x_labels = [str(c) for c in pivot_mask.columns]
        y_labels = [str(r) for r in pivot_mask.index]

        fig.add_trace(
            go.Heatmap(
                x=x_labels,
                y=y_labels,
                z=pivot_mask.values,
                text=pivot_text.values,
                texttemplate="%{text}",
                textfont=dict(size=font_size, color="white"),
                colorscale=mask_colorscale,
                zmin=0,
                zmax=1,
                xgap=1,
                ygap=1,
                showscale=col_idx == 1,
                colorbar=dict(title="Accepted", tickvals=[0, 1], ticktext=["REJ", "ACC"]) if col_idx == 1 else None,
            ),
            row=1,
            col=col_idx,
        )

        # Add cutoff boundary line on the acceptance heatmap
        var1_inv = var1 in (inv_vars or [])
        _add_nd_cutoff_boundary(fig, pivot_mask, x_labels, y_labels, row=1, col=col_idx, var1_inverted=var1_inv)

        # Row 2: Risk heatmap with value annotations
        pivot_risk = slice_df.pivot(index=var1, columns=var0, values="_b2").sort_index(ascending=True)
        fig.add_trace(
            go.Heatmap(
                x=[str(c) for c in pivot_risk.columns],
                y=[str(r) for r in pivot_risk.index],
                z=pivot_risk.values,
                text=pivot_risk.values,
                texttemplate="%{text:.1f}",
                textfont=dict(size=font_size),
                colorscale=[(0, "rgba(255,255,255,1)"), (1, "rgba(157,13,20,1)")],
                zmin=0,
                xgap=1,
                ygap=1,
                showscale=col_idx == len(v2_vals),
                colorbar=dict(title="b2 (%)") if col_idx == len(v2_vals) else None,
            ),
            row=2,
            col=col_idx,
        )

        fig.update_xaxes(title_text=var0, row=1, col=col_idx, tickfont=dict(size=9))
        fig.update_xaxes(title_text=var0, row=2, col=col_idx, tickfont=dict(size=9))
        if col_idx == 1:
            fig.update_yaxes(title_text=var1, row=1, col=col_idx, tickfont=dict(size=9))
            fig.update_yaxes(title_text=var1, row=2, col=col_idx, tickfont=dict(size=9))

    # Scale height to grid rows so cells aren't squashed
    row_height = max(22, min(35, 700 // n_rows))
    total_height = max(700, row_height * n_rows * 2 + 120)

    fig.update_layout(
        title_text="Acceptance Grid & Risk by Income Bin",
        height=total_height,
        width=max(500 * len(v2_vals), 800),
        margin=dict(l=60, r=40, t=60, b=40),
    )

    if output_path:
        fig.write_html(output_path)
        logger.info(f"Acceptance grid saved to {output_path}")

    return fig
