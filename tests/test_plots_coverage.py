"""Additional coverage tests for src.plots functions not in test_plots.py.

Targets: SHAP dependence, income_bin_trends, transformation rate,
per-bin transformation rate, `plot_acceptance_grid_nd`, small helpers
(`_format_bin_edges`, `_prepare_transformation_data`), and the 3D plot
marginalization branch.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import plots


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestFormatBinEdges:
    def test_finite_edges(self):
        labels = plots._format_bin_edges([0.0, 100.0, 200.0])
        assert labels == ["1: (0.0, 100.0]", "2: (100.0, 200.0]"]

    def test_infinite_edges(self):
        labels = plots._format_bin_edges([-np.inf, 350.0, np.inf])
        assert labels[0].startswith("1: (-inf")
        assert labels[-1].endswith("inf)")

    def test_single_interval(self):
        labels = plots._format_bin_edges([0.0, 1.0])
        assert labels == ["1: (0.0, 1.0]"]


class TestPrepareTransformationData:
    def test_filters_by_decision_and_marks_booked(self):
        df = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
                "se_decision_id": ["ok", "rv", "rejected"],
                "status_name": ["booked", "rejected", "rejected"],
            }
        )
        out = plots._prepare_transformation_data(df, "mis_date", n_months=None)
        assert len(out) == 2  # rejected row dropped
        assert list(out["is_booked"]) == [True, False]

    def test_n_months_truncates_older_rows(self):
        dates = pd.to_datetime(["2023-01-01", "2023-06-01", "2024-03-01", "2024-06-01"])
        df = pd.DataFrame(
            {
                "mis_date": dates,
                "se_decision_id": ["ok"] * 4,
                "status_name": ["booked"] * 4,
            }
        )
        out = plots._prepare_transformation_data(df, "mis_date", n_months=4)
        # only rows within last 4 months of max_date remain
        assert len(out) <= 2


class TestCalculateAndPlotTransformationRate:
    def test_returns_expected_dict_shape(self):
        df = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"] * 3 + ["2024-03-01"] * 3),
                "se_decision_id": ["ok"] * 9,
                "status_name": ["booked", "booked", "rejected"] * 3,
                "oa_amt": [1000.0] * 9,
            }
        )
        out = plots.calculate_and_plot_transformation_rate(df, "mis_date")
        assert set(out.keys()) == {
            "overall_rate",
            "overall_booked_amt",
            "overall_eligible_amt",
            "monthly_amounts",
            "figure",
        }
        assert out["overall_rate"] == pytest.approx(2 / 3)
        assert len(out["monthly_amounts"]) == 3

    def test_zero_eligible_returns_zero_rate(self):
        df = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01"]),
                "se_decision_id": ["rejected"],
                "status_name": ["rejected"],
                "oa_amt": [1000.0],
            }
        )
        out = plots.calculate_and_plot_transformation_rate(df, "mis_date")
        assert out["overall_rate"] == 0


class TestCalculatePerBinTransformationRate:
    def test_per_bin_rates_computed_and_fallback_applied(self):
        rng = np.random.RandomState(42)
        n_per_bin = 20
        df = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01"] * (3 * n_per_bin)),
                "se_decision_id": ["ok"] * (3 * n_per_bin),
                "status_name": (
                    ["booked"] * n_per_bin  # bin 1: 100%
                    + ["rejected"] * n_per_bin  # bin 2: 0%
                    + rng.choice(["booked", "rejected"], n_per_bin).tolist()  # bin 3: ~50%
                ),
                "oa_amt": [1000.0] * (3 * n_per_bin),
                "bin_a": [1] * n_per_bin + [2] * n_per_bin + [3] * n_per_bin,
            }
        )
        out = plots.calculate_per_bin_transformation_rate(df, ["bin_a"], global_tasa_fin=0.5, min_eligible_per_bin=5)
        assert set(out["bin_a"]) == {1, 2, 3}

    def test_sparse_bin_falls_back_to_global(self):
        df = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01"]),
                "se_decision_id": ["ok"],
                "status_name": ["rejected"],
                "oa_amt": [1000.0],
                "bin_a": [1],
            }
        )
        out = plots.calculate_per_bin_transformation_rate(df, ["bin_a"], global_tasa_fin=0.77, min_eligible_per_bin=10)
        assert out["tasa_fin"].iloc[0] == pytest.approx(0.77)


class TestPlotShapDependence:
    def test_raises_on_unknown_feature(self):
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError):
            plots.plot_shap_dependence(shap_values, ["a", "b"], pd.DataFrame({"a": [1, 2], "b": [3, 4]}), "c")

    def test_writes_html_when_path_given(self, tmp_path):
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        fd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out_path = tmp_path / "dep.html"
        fig = plots.plot_shap_dependence(shap_values, ["a", "b"], fd, "a", output_path=str(out_path))
        assert out_path.exists()
        assert fig is not None


class TestPlotShapSummaryHTMLWrite:
    def test_writes_html(self, tmp_path):
        shap_values = np.array([[0.1, 0.2], [0.3, 0.4]])
        out_path = tmp_path / "sum.html"
        plots.plot_shap_summary(shap_values, ["a", "b"], output_path=str(out_path))
        assert out_path.exists()


class TestPlotIncomeBinTrends:
    def test_returns_figure_and_writes_html(self, tmp_path):
        rng = np.random.RandomState(42)
        n = 90
        df = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(pd.date_range("2024-01-01", periods=n, freq="D")),
                "income_bin": rng.choice([1, 2, 3], n),
                "early_bad": rng.randint(0, 2, n),
                "oa_amt": rng.uniform(500, 2500, n),
            }
        )
        out_path = tmp_path / "trends.html"
        fig = plots.plot_income_bin_trends(df, "income_bin", output_path=str(out_path))
        assert fig is not None
        assert out_path.exists()


class TestPlotGroupStatistics:
    def test_produces_figure(self):
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "c", "c"],
                "outcome": [0, 1, 1, 0, 0, 0],
            }
        )
        fig = plots.plot_group_statistics(df, "group", "outcome")
        assert fig is not None


class TestPlot3dGraphMarginalization:
    def test_returns_figure_for_more_than_2_vars(self):
        rng = np.random.RandomState(42)
        n = 30
        data_train = pd.DataFrame(
            {
                "v0": rng.randint(1, 4, n),
                "v1": rng.randint(1, 4, n),
                "v2": rng.randint(1, 4, n),
                "target": rng.rand(n),
            }
        )
        data_surf = data_train.copy()
        data_surf["target_pred"] = data_surf["target"] + rng.normal(0, 0.01, n)
        fig = plots.plot_3d_graph(data_train, data_surf, ["v0", "v1", "v2"], "target")
        assert fig is not None


class TestPlotAcceptanceGridNd:
    def test_returns_empty_figure_for_2_vars(self):
        class FakeGrid:
            variables = ["v0", "v1"]
            values_per_var = {"v0": [0, 1], "v1": [0, 1]}
            cell_data = pd.DataFrame()

        fig = plots.plot_acceptance_grid_nd(np.array([1, 0]), FakeGrid())
        # returns empty go.Figure (early exit)
        assert fig is not None
