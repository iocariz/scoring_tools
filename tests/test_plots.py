"""Tests for the plots visualization module."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.constants import Columns, StatusName
from src.plots import (
    plot_bin_threshold_diagnostic,
    plot_precision_recall_curve,
    plot_risk_vs_production,
    plot_roc_curve,
    plot_score_distribution,
    plot_shap_summary,
)

# =============================================================================
# Matplotlib cleanup fixture
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


# =============================================================================
# TestPlotRocCurve
# =============================================================================


class TestPlotRocCurve:
    def test_runs_without_error(self):
        """Verify plot_roc_curve executes without raising on simple binary data."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plot_roc_curve(ax, y_true, scores, name="test_model", color="blue")

    def test_ax_has_lines_after_plotting(self):
        """Verify the axes object contains line objects after plotting."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plot_roc_curve(ax, y_true, scores, name="test_model", color="red")
        assert len(ax.lines) > 0, "Expected at least one line on the axes"


# =============================================================================
# TestPlotRiskVsProduction
# =============================================================================


class TestPlotRiskVsProduction:
    def _make_data(self):
        """Create a simple DataFrame with the columns required by plot_risk_vs_production."""
        n = 30
        dates = pd.date_range("2023-01-01", periods=n, freq="MS")
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                Columns.MIS_DATE: dates,
                Columns.STATUS_NAME: [StatusName.BOOKED.value] * n,
                Columns.TODU_30EVER_H6: rng.uniform(100, 500, n),
                Columns.TODU_AMT_PILE_H6: rng.uniform(5000, 20000, n),
                Columns.OA_AMT_H0: rng.uniform(10000, 50000, n),
            }
        )
        return df

    def test_returns_plotly_figure(self):
        """Verify plot_risk_vs_production returns a plotly go.Figure."""
        data = self._make_data()
        indicadores = [
            Columns.TODU_30EVER_H6,
            Columns.TODU_AMT_PILE_H6,
            Columns.OA_AMT_H0,
        ]
        comfort_zones = {2023: 4.50, 2024: 4.00, 2025: 3.75}
        # Use a small subset as data_booked
        data_booked = data.head(6)

        fig = plot_risk_vs_production(
            data=data,
            indicadores=indicadores,
            comfort_zones=comfort_zones,
            data_booked=data_booked,
        )
        assert isinstance(fig, go.Figure)


# =============================================================================
# TestPlotShapSummary
# =============================================================================


class TestPlotShapSummary:
    def test_returns_plotly_figure(self):
        """Verify plot_shap_summary returns a plotly go.Figure."""
        shap_values = np.random.RandomState(42).randn(10, 3)
        feature_names = ["feature_a", "feature_b", "feature_c"]
        fig = plot_shap_summary(shap_values, feature_names)
        assert isinstance(fig, go.Figure)

    def test_runs_without_error_various_shapes(self):
        """Verify it handles different input shapes without error."""
        rng = np.random.RandomState(0)
        shap_values = rng.randn(20, 5)
        feature_names = [f"feat_{i}" for i in range(5)]
        fig = plot_shap_summary(shap_values, feature_names)
        assert isinstance(fig, go.Figure)


# =============================================================================
# TestPlotPrecisionRecallCurve
# =============================================================================


class TestPlotPrecisionRecallCurve:
    def test_runs_without_error(self):
        """Verify plot_precision_recall_curve executes without raising."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores_dict = {"Model A": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
        plot_precision_recall_curve(ax, y_true, scores_dict)

    def test_multiple_models(self):
        """Verify it works with multiple models."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores_dict = {
            "Good": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "Random": np.random.RandomState(42).rand(10),
        }
        plot_precision_recall_curve(ax, y_true, scores_dict)
        # Should have lines for each model + baseline
        assert len(ax.lines) >= 2

    def test_axes_labels(self):
        """Verify correct axis labels are set."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 1, 1])
        scores_dict = {"Test": np.array([0.1, 0.4, 0.6, 0.9])}
        plot_precision_recall_curve(ax, y_true, scores_dict)
        assert ax.get_xlabel() == "Recall"
        assert ax.get_ylabel() == "Precision"


# =============================================================================
# TestPlotScoreDistribution
# =============================================================================


class TestPlotScoreDistribution:
    def test_runs_without_error(self):
        """Verify plot_score_distribution executes without raising."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plot_score_distribution(ax, y_true, scores)

    def test_custom_name(self):
        """Verify custom name appears in title."""
        fig, ax = plt.subplots()
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        plot_score_distribution(ax, y_true, scores, name="My Score")
        assert "My Score" in ax.get_title()

    def test_has_histogram_patches(self):
        """Verify histogram patches are drawn."""
        fig, ax = plt.subplots()
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        scores = rng.rand(200)
        plot_score_distribution(ax, y_true, scores, bins=20)
        assert len(ax.patches) > 0


# =============================================================================
# TestPlotBinThresholdDiagnostic
# =============================================================================


class TestPlotBinThresholdDiagnostic:
    """Tests for plot_bin_threshold_diagnostic."""

    @staticmethod
    def _make_data(n_months=6, n_per_month=100, n_bins=3, seed=42):
        rng = np.random.RandomState(seed)
        months = pd.date_range("2023-01-01", periods=n_months, freq="MS")
        rows = []
        for m in months:
            for _ in range(n_per_month):
                b = rng.randint(1, n_bins + 1)
                rows.append(
                    {
                        "mis_date": m,
                        "bin_col": b,
                        "todu_30ever_h6": rng.exponential(100 * b),
                        "todu_amt_pile_h6": rng.exponential(5000),
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_figure(self):
        """Should return a plotly Figure."""
        df = self._make_data()
        fig = plot_bin_threshold_diagnostic(df, bin_col="bin_col")
        assert isinstance(fig, go.Figure)

    def test_three_subplots(self):
        """Should have 3 subplot rows (share, risk, separation)."""
        df = self._make_data()
        fig = plot_bin_threshold_diagnostic(df, bin_col="bin_col")
        # 3 subplot titles
        annotations = [a for a in fig.layout.annotations if a.text]
        assert len(annotations) == 3

    def test_single_bin_no_crash(self):
        """Should handle a single bin gracefully (no separation panel data)."""
        df = self._make_data(n_bins=1)
        fig = plot_bin_threshold_diagnostic(df, bin_col="bin_col")
        assert isinstance(fig, go.Figure)

    def test_saves_html(self, tmp_path):
        """Should write HTML file when output_path is provided."""
        df = self._make_data()
        path = str(tmp_path / "diag.html")
        plot_bin_threshold_diagnostic(df, bin_col="bin_col", output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_bin_edges_in_title(self):
        """Threshold values should appear in the figure title."""
        df = self._make_data()
        edges = [float("-inf"), 350.0, 450.0, float("inf")]
        fig = plot_bin_threshold_diagnostic(df, bin_col="bin_col", bin_edges=edges, source_col="score_rf")
        title = fig.layout.title.text
        assert "350" in title
        assert "450" in title
        assert "score_rf" in title

    def test_full_period_in_x_axis(self):
        """'Full Period' summary column should be present in the data."""
        df = self._make_data()
        fig = plot_bin_threshold_diagnostic(df, bin_col="bin_col")
        # Check that at least one trace has 'Full Period' in its x data
        found = False
        for trace in fig.data:
            xs = trace.x if trace.x is not None else []
            if "Full Period" in list(xs):
                found = True
                break
        assert found

    def test_bin_edge_labels_in_legend(self):
        """Bin legend labels should contain interval ranges when edges provided."""
        df = self._make_data()
        edges = [float("-inf"), 350.0, 450.0, float("inf")]
        fig = plot_bin_threshold_diagnostic(df, bin_col="bin_col", bin_edges=edges)
        names = [t.name for t in fig.data if t.name]
        # At least one trace should have a threshold range in its name
        assert any("350" in n for n in names)
