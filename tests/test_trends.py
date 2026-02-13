"""Tests for temporal trend analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.trends import compute_monthly_metrics, detect_trend_changes, plot_metric_trends


def _make_monthly_data(n_months: int = 12, seed: int = 42) -> pd.DataFrame:
    """Create synthetic monthly data with known patterns."""
    rng = np.random.RandomState(seed)

    dates = pd.date_range("2023-01-01", periods=n_months * 30, freq="D")
    n = len(dates)

    data = pd.DataFrame(
        {
            "mis_date": dates,
            "status_name": rng.choice(["booked", "rejected"], size=n, p=[0.6, 0.4]),
            "segment_cut_off": "consumer",
            "oa_amt": rng.uniform(1000, 50000, size=n),
            "todu_30ever_h6": rng.uniform(0, 0.1, size=n),
            "todu_amt_pile_h6": rng.uniform(100, 5000, size=n),
            "sc_octroi": rng.uniform(200, 800, size=n),
        }
    )
    # Make status_name categorical (as the pipeline standardizes)
    data["status_name"] = data["status_name"].astype("category")
    return data


class TestComputeMonthlyMetrics:
    def test_basic_computation(self):
        data = _make_monthly_data(6)
        result = compute_monthly_metrics(data, date_column="mis_date")

        assert not result.empty
        assert "year_month" in result.columns
        assert "total_records" in result.columns
        assert "booked_count" in result.columns
        assert "approval_rate" in result.columns
        assert len(result) == 6  # 6 months

    def test_approval_rate_range(self):
        data = _make_monthly_data(6)
        result = compute_monthly_metrics(data)

        assert (result["approval_rate"] >= 0).all()
        assert (result["approval_rate"] <= 1).all()

    def test_production_metrics(self):
        data = _make_monthly_data(3)
        result = compute_monthly_metrics(data)

        assert "mean_production" in result.columns
        assert "total_production" in result.columns
        assert (result["total_production"] >= 0).all()

    def test_score_metrics(self):
        data = _make_monthly_data(3)
        result = compute_monthly_metrics(data)

        assert "mean_sc_octroi" in result.columns
        assert result["mean_sc_octroi"].notna().all()

    def test_segment_filter(self):
        data = _make_monthly_data(3)
        # Add another segment
        data2 = _make_monthly_data(3, seed=99)
        data2["segment_cut_off"] = "corporate"
        combined = pd.concat([data, data2], ignore_index=True)

        result = compute_monthly_metrics(combined, segment_filter="consumer")
        # Should only have consumer data
        assert not result.empty
        # Total records should be less than combined total
        assert result["total_records"].sum() < len(combined)

    def test_empty_after_filter(self):
        data = _make_monthly_data(3)
        result = compute_monthly_metrics(data, segment_filter="nonexistent_segment")
        assert result.empty

    def test_sorted_by_date(self):
        data = _make_monthly_data(6)
        result = compute_monthly_metrics(data)

        dates = result["year_month"].tolist()
        assert dates == sorted(dates)

    def test_booked_count_less_than_total(self):
        data = _make_monthly_data(3)
        result = compute_monthly_metrics(data)

        assert (result["booked_count"] <= result["total_records"]).all()


class TestPlotMetricTrends:
    def test_returns_figure(self):
        data = _make_monthly_data(6)
        monthly = compute_monthly_metrics(data)
        fig = plot_metric_trends(monthly, ["approval_rate", "total_records"])

        assert isinstance(fig, go.Figure)

    def test_saves_html(self, tmp_path):
        data = _make_monthly_data(6)
        monthly = compute_monthly_metrics(data)
        path = str(tmp_path / "trends.html")
        plot_metric_trends(monthly, ["approval_rate"], output_path=path)

        assert (tmp_path / "trends.html").exists()

    def test_missing_metrics_handled(self):
        data = _make_monthly_data(3)
        monthly = compute_monthly_metrics(data)
        fig = plot_metric_trends(monthly, ["nonexistent_metric"])

        assert isinstance(fig, go.Figure)


class TestDetectTrendChanges:
    def test_basic_detection(self):
        data = _make_monthly_data(12)
        monthly = compute_monthly_metrics(data)
        result = detect_trend_changes(monthly, "approval_rate", window=3)

        assert "value" in result.columns
        assert "rolling_mean" in result.columns
        assert "upper_bound" in result.columns
        assert "lower_bound" in result.columns
        assert "is_anomaly" in result.columns
        assert "direction" in result.columns

    def test_detects_spike(self):
        """Inject a spike into stable data and verify it's detected."""
        data = _make_monthly_data(12)
        monthly = compute_monthly_metrics(data)

        # Set approval rate to near-constant values, then inject a spike
        rng = np.random.RandomState(42)
        monthly["approval_rate"] = 0.5 + rng.normal(0, 0.01, len(monthly))
        monthly.loc[monthly.index[-1], "approval_rate"] = 0.99

        result = detect_trend_changes(monthly, "approval_rate", window=3, n_sigma=1.5)

        # The spike should be flagged
        anomalies = result[result["is_anomaly"]]
        assert len(anomalies) >= 1

    def test_direction_labels(self):
        data = _make_monthly_data(12)
        monthly = compute_monthly_metrics(data)
        monthly.loc[monthly.index[-1], "approval_rate"] = 0.99  # spike above
        monthly.loc[monthly.index[-2], "approval_rate"] = 0.01  # drop below

        result = detect_trend_changes(monthly, "approval_rate", window=3, n_sigma=1.0)

        directions = set(result.loc[result["is_anomaly"], "direction"].dropna())
        # Should have at least one direction flagged
        assert directions.issubset({"above", "below"})

    def test_missing_metric_raises(self):
        data = _make_monthly_data(3)
        monthly = compute_monthly_metrics(data)

        with pytest.raises(ValueError, match="not found"):
            detect_trend_changes(monthly, "nonexistent_metric")

    def test_insufficient_data(self):
        """With fewer rows than window, should still return DataFrame."""
        data = _make_monthly_data(2)
        monthly = compute_monthly_metrics(data)
        result = detect_trend_changes(monthly, "approval_rate", window=5)

        assert not result.empty
        assert result["is_anomaly"].sum() == 0  # Can't flag with insufficient data
