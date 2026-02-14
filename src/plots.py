"""
Visualization functions for credit risk model analysis and reporting.

This module provides plotting functions for model evaluation and risk analysis:
- ROC curves and Gini visualization
- KS statistic plots
- CAP curves and rejection analysis
- 3D surface plots for risk landscapes
- Interactive risk vs production visualizations
- Group statistics and confidence interval plots

Key components:
- plot_roc_curve: Plot ROC curve with Gini and KS annotations
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
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

from . import styles
from .constants import DEFAULT_RISK_MULTIPLIER, Columns, StatusName
from .metrics import ks_statistic
from .utils import calculate_b2_ever_h6

# Keep track of previous KS annotation y-coordinates
previous_ks_y_positions = []


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
        values_var0: np.ndarray,
        values_var1: np.ndarray,
        optimum_risk: float,
        tasa_fin: float,
        target_sol_fac: int | None = None,
    ):
        """
        Initialize the RiskProductionVisualizer.

        Args:
            data_summary: DataFrame with optimal solutions containing columns:
                b2_ever_h6, oa_amt_h0, and cut values for each bin.
            data_summary_disaggregated: Disaggregated booked data with '_boo' suffixes.
            data_summary_sample_no_opt: Sample of non-optimal solutions for context.
            variables: List of two variable names [var0, var1].
            values_var0: Array of bin values for first variable.
            values_var1: Array of bin values for second variable.
            optimum_risk: Target risk level (%) for automatic solution selection.
            tasa_fin: Financing rate (used for repesca calculations).
            target_sol_fac: Optional specific solution ID to select instead of
                using optimum_risk threshold.
        """
        self.data_summary = data_summary
        self.data_summary_disaggregated = data_summary_disaggregated
        self.data_summary_sample_no_opt = data_summary_sample_no_opt
        self.variables = variables
        self.values_var0 = values_var0
        self.values_var1 = values_var1
        self.optimum_risk = optimum_risk
        self.tasa_fin = tasa_fin

        self.target_sol_fac = target_sol_fac

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

        self.B2_0 = np.round(100 * DEFAULT_RISK_MULTIPLIER * tudu_30_ever / tudu_amt_pile, 2)

        # Calculate OA_0
        self.OA_0 = self.data_summary_disaggregated["oa_amt_h0_boo"].sum()

        # Calculate limits
        self.lim_sup_B2 = self.data_summary["b2_ever_h6"].max()
        self.lim_inf_B2 = 0
        self.lim_sup_OA = self.data_summary["oa_amt_h0"].max()
        self.lim_inf_OA = 0

        # Create meshgrid for heatmap
        self.xx, self.yy = np.meshgrid(self.values_var0, self.values_var1)

    def create_figure(self):
        """Create the main figure with subplots"""
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

    def _add_heatmap_traces(self):
        """Add heatmap traces to the figure"""
        # Main heatmap
        self.fig.add_trace(
            go.Heatmap(
                x=self.data_summary_disaggregated[self.variables[0]],
                y=self.data_summary_disaggregated[self.variables[1]],
                z=self.data_summary_disaggregated["b2_ever_h6"],
                colorscale=[(0.00, "rgba(255,255,255,1)"), (1.00, "rgba(157,13,20,1)")],
                zmin=0,
                zmax=20,
                text=self.data_summary_disaggregated["text"],
                texttemplate="%{text}",
            ),
            row=1,
            col=2,
        )

        # Mask heatmap
        self.mask_trace = go.Heatmap(
            x=self.values_var0,
            y=self.values_var1,
            z=np.zeros_like(self.yy),
            colorscale=[(0.00, "rgba(0,0,0,0.5)"), (1.00, "rgba(0,0,0,0)")],
            showscale=False,
        )
        self.fig.add_trace(self.mask_trace, row=1, col=2)

    def _update_layout(self):
        """Update the figure layout"""
        styles.apply_plotly_style(self.fig, width=1700, height=600)
        self.fig.update_layout(legend=dict(y=0.1, x=0.33))

        # Update axes
        self.fig.update_xaxes(title_text="OA AMT (€)", range=[self.lim_inf_OA, self.lim_sup_OA], row=1, col=1)
        self.fig.update_xaxes(title_text=self.variables[0], dtick=1, row=1, col=2)
        self.fig.update_yaxes(title_text="B2 ever H6 (%€)", range=[self.lim_inf_B2, self.lim_sup_B2], row=1, col=1)
        self.fig.update_yaxes(title_text=self.variables[1], dtick=1, row=1, col=2)

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

        data_filtered = self.data_summary[self.data_summary["b2_ever_h6"] <= self.optimum_risk]
        if data_filtered.empty:
            data_filtered = self.data_summary.sort_values("b2_ever_h6").head(1)
        else:
            data_filtered = data_filtered.tail(1)
        return data_filtered

    def _apply_static_update(self):
        """Update the visualization based on cz2024 value"""
        # Filter data based on risk level
        data_filtered = self._get_selected_solution_row()

        metrics = data_filtered[
            ["b2_ever_h6", "oa_amt_h0", "b2_ever_h6_cut", "oa_amt_h0_cut", "b2_ever_h6_rep", "oa_amt_h0_rep"]
        ].values[0, :]
        B2, OA, B2_CUT, OA_CUT, B2_REP, OA_REP = metrics

        # Update cut-off mask
        CUT_OFF = data_filtered[self.values_var0].values[0]
        z_mask = (self.yy <= CUT_OFF) * 1

        # Update visualization
        # In original code: self.fig.data[5].z = z_mask (Trace 5 is the mask heatmap)
        # self.fig.data[3].update(x=[OA], y=[B2]) (Trace 3 is Optimum selected point)

        # We need to rely on the order of traces added in create_figure
        # _add_scatter_traces adds 4 traces (0, 1, 2, 3)
        # _add_heatmap_traces adds 2 traces (4, 5)
        # So trace 5 is indeed the mask trace.

        self.fig.data[5].z = z_mask
        self.fig.data[3].update(x=[OA], y=[B2])

    def create_slider(self):
        """No longer used"""
        pass

    def update(self, x):
        """No longer used"""
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

        # Construct DataFrame
        summary_data = {
            "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected", "Summary"],
            "Risk (%)": [
                max(self.B2_0 / 100, 0.000001),
                max(B2_REP / 100, 0.000001),
                max(B2_CUT / 100, 0.000001),
                max(B2 / 100, 0.000001),
                (B2 - self.B2_0) / 100,
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

    df_plot = (
        data[data[Columns.STATUS_NAME] == StatusName.BOOKED.value]
        .groupby([Columns.MIS_DATE], as_index=False)
        .agg(dict.fromkeys(indicadores, "sum"))
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
        df_plot[Columns.TODU_30EVER_H6], df_plot[Columns.TODU_AMT_PILE_H6], as_percentage=True
    )

    df_plot["b2_ever_h6_MA"] = calculate_b2_ever_h6(
        df_plot[f"{Columns.TODU_30EVER_H6}_MA{rolling_window}"],
        df_plot[f"{Columns.TODU_AMT_PILE_H6}_MA{rolling_window}"],
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
