"""
End-to-end integration tests for the P1 enhancements.

Tests that the new modules work correctly together:
- Schema validation -> data pipeline
- Stability -> drift alerts -> JSON serialization
- Monthly metrics -> trend detection -> visualization
- Model training -> SHAP computation -> model persistence
- Reproducibility: identical results with same seed
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.linear_model import Ridge

from src.alerts import AlertReport, AlertSeverity, generate_drift_alerts
from src.data_manager import DataValidationError
from src.inference_optimized import _compute_shap_values
from src.metrics import bootstrap_confidence_interval
from src.persistence import save_model_with_metadata
from src.plots import plot_shap_summary
from src.stability import StabilityReport, StabilityStatus, calculate_stability_report
from src.trends import compute_monthly_metrics, detect_trend_changes, plot_metric_trends

try:
    from src.schema import validate_raw_data

    _has_pandera = True
except ImportError:
    validate_raw_data = None
    _has_pandera = False

try:
    import shap  # noqa: F401

    _has_shap = True
except ImportError:
    _has_shap = False

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_df():
    """Minimal valid DataFrame that passes schema validation."""
    n = 200
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "status_name": pd.Categorical(rng.choice(["booked", "rejected"], size=n, p=[0.6, 0.4])),
            "segment_cut_off": "consumer",
            "mis_date": dates,
            "oa_amt": rng.uniform(1000, 50000, size=n),
            "oa_amt_h0": rng.uniform(1000, 50000, size=n),
            "todu_30ever_h6": rng.uniform(0, 5, size=n),
            "todu_amt_pile_h6": rng.uniform(100, 5000, size=n),
            "sc_octroi": rng.uniform(200, 800, size=n),
            "new_efx": rng.uniform(100, 500, size=n),
        }
    )


@pytest.fixture
def booked_main_df(valid_df):
    """Booked records for main period."""
    main = valid_df[valid_df["status_name"] == "booked"].copy()
    main["sc_octroi"] = np.random.RandomState(1).uniform(300, 700, len(main))
    return main


@pytest.fixture
def booked_mr_df(valid_df):
    """Booked records for MR period with slight drift."""
    mr = valid_df[valid_df["status_name"] == "booked"].copy()
    # Introduce drift: shift scores by adding bias
    rng = np.random.RandomState(99)
    mr["sc_octroi"] = rng.uniform(350, 750, len(mr))  # shifted distribution
    return mr


# ---------------------------------------------------------------------------
# Integration: Schema -> Data Flow
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_pandera, reason="pandera not installed")
class TestSchemaIntegration:
    def test_valid_data_passes_schema(self, valid_df):
        """Schema validation should pass for well-formed data."""
        validated = validate_raw_data(valid_df, raise_on_error=True)
        assert validated is not None
        assert len(validated) == len(valid_df)

    def test_schema_catches_invalid_status(self, valid_df):
        """Schema rejects invalid status_name values."""
        bad_df = valid_df.copy()
        bad_df["status_name"] = pd.Categorical(["invalid_status"] * len(bad_df))
        with pytest.raises(DataValidationError):
            validate_raw_data(bad_df, raise_on_error=True)

    def test_schema_non_raise_mode(self, valid_df):
        """Non-raise mode returns (True, []) for valid data."""
        is_valid, errors = validate_raw_data(valid_df, raise_on_error=False)
        assert is_valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# Integration: Stability -> Alerts -> JSON
# ---------------------------------------------------------------------------


class TestStabilityAlertIntegration:
    def test_stability_report_to_alerts(self, booked_main_df, booked_mr_df):
        """Full flow: compute PSI -> generate alerts -> serialize to JSON."""
        variables = ["sc_octroi", "oa_amt"]

        # Step 1: Calculate stability
        report = calculate_stability_report(
            baseline_df=booked_main_df,
            comparison_df=booked_mr_df,
            variables=variables,
            score_variable="sc_octroi",
        )

        assert isinstance(report, StabilityReport)
        assert len(report.psi_results) > 0

        # Step 2: Generate alerts
        alert_report = generate_drift_alerts(report, "consumer", "MR")

        assert isinstance(alert_report, AlertReport)
        assert len(alert_report.alerts) > 0
        assert alert_report.segment == "consumer"

        # Step 3: Serialize to JSON
        json_data = {
            "segment": alert_report.segment,
            "generated_at": alert_report.generated_at,
            "summary": alert_report.summary,
            "alerts": [a.to_dict() for a in alert_report.alerts],
        }
        json_str = json.dumps(json_data, indent=2)
        parsed = json.loads(json_str)

        assert parsed["segment"] == "consumer"
        assert "critical" in parsed["summary"]
        assert len(parsed["alerts"]) == len(alert_report.alerts)

    def test_alerts_json_roundtrip(self, tmp_path, booked_main_df, booked_mr_df):
        """Alerts can be saved and loaded from JSON."""
        report = calculate_stability_report(
            booked_main_df,
            booked_mr_df,
            ["sc_octroi"],
            score_variable="sc_octroi",
        )
        alert_report = generate_drift_alerts(report, "consumer", "MR")

        path = str(tmp_path / "alerts.json")
        alert_report.to_json(path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["segment"] == "consumer"
        assert sum(loaded["summary"].values()) == len(loaded["alerts"])


# ---------------------------------------------------------------------------
# Integration: Monthly Metrics -> Trend Detection -> Plotting
# ---------------------------------------------------------------------------


class TestTrendsIntegration:
    def test_full_trends_pipeline(self, valid_df, tmp_path):
        """Full flow: monthly metrics -> trend detection -> visualization."""
        # Step 1: Compute monthly metrics
        monthly = compute_monthly_metrics(valid_df, date_column="mis_date")
        assert not monthly.empty
        assert "approval_rate" in monthly.columns

        # Step 2: Detect trend changes
        changes = detect_trend_changes(monthly, "approval_rate", window=2)
        assert "is_anomaly" in changes.columns
        assert "direction" in changes.columns

        # Step 3: Plot trends
        path = str(tmp_path / "trends.html")
        fig = plot_metric_trends(
            monthly,
            ["approval_rate", "total_records"],
            output_path=path,
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "trends.html").exists()


# ---------------------------------------------------------------------------
# Integration: Model Training -> SHAP -> Persistence
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_shap, reason="shap not installed")
class TestModelShapIntegration:
    def test_train_shap_save(self, tmp_path):
        """Full flow: train model -> compute SHAP -> save with metadata."""
        # Step 1: Train a model
        rng = np.random.RandomState(42)
        n, p = 80, 4
        X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
        y = 2 * X["f0"] - X["f1"] + rng.randn(n) * 0.1
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        # Step 2: Compute SHAP
        shap_result = _compute_shap_values(model, X, list(X.columns))
        assert shap_result is not None
        assert shap_result["shap_values"].shape == (n, p)

        # Step 3: Save model with SHAP metadata
        metadata = {
            "cv_mean_r2": 0.95,
            "cv_std_r2": 0.02,
            "full_r2": 0.97,
            "shap_values": shap_result["shap_values"],
        }
        model_path = save_model_with_metadata(
            model,
            list(X.columns),
            metadata,
            base_path=str(tmp_path / "models"),
        )

        # Verify SHAP file saved
        from pathlib import Path

        saved_files = list(Path(model_path).glob("*"))
        file_names = [f.name for f in saved_files]
        assert "model.pkl" in file_names
        assert "metadata.json" in file_names
        assert "shap_values.npy" in file_names

        # Verify SHAP values can be loaded
        loaded_shap = np.load(Path(model_path) / "shap_values.npy")
        np.testing.assert_array_equal(loaded_shap, shap_result["shap_values"])

    def test_shap_plot_from_model(self, tmp_path):
        """SHAP values can be used for visualization."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])
        y = X["a"] + rng.randn(50) * 0.1
        model = Ridge()
        model.fit(X, y)

        shap_result = _compute_shap_values(model, X, list(X.columns))
        assert shap_result is not None

        fig = plot_shap_summary(
            shap_result["shap_values"],
            shap_result["feature_names"],
            output_path=str(tmp_path / "shap_summary.html"),
        )
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "shap_summary.html").exists()


# ---------------------------------------------------------------------------
# Integration: Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_bootstrap_same_seed_same_result(self):
        """Bootstrap CI with same seed produces identical results."""
        rng = np.random.RandomState(42)
        y_true = pd.Series(rng.choice([0, 1], size=200, p=[0.7, 0.3]))
        y_scores = pd.Series(rng.uniform(0, 1, size=200))

        result1 = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50, random_state=123)
        result2 = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50, random_state=123)

        assert result1 == result2, f"Results differ: {result1} != {result2}"

    def test_bootstrap_different_seed_different_result(self):
        """Bootstrap CI with different seeds produces different results."""
        rng = np.random.RandomState(42)
        y_true = pd.Series(rng.choice([0, 1], size=200, p=[0.7, 0.3]))
        y_scores = pd.Series(rng.uniform(0, 1, size=200))

        result1 = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50, random_state=1)
        result2 = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50, random_state=2)

        # With different seeds, at least some CI bounds should differ
        assert result1 != result2


# ---------------------------------------------------------------------------
# Integration: Cross-Module Consistency
# ---------------------------------------------------------------------------


class TestCrossModuleConsistency:
    def test_stability_status_maps_to_alert_severity(self):
        """All StabilityStatus values have corresponding AlertSeverity mappings."""
        from src.alerts import _SEVERITY_MAP

        for status in StabilityStatus:
            assert status in _SEVERITY_MAP, f"Missing severity mapping for {status}"
            assert isinstance(_SEVERITY_MAP[status], AlertSeverity)

    def test_psi_result_to_alert_preserves_values(self, booked_main_df, booked_mr_df):
        """PSI values from stability report match values in generated alerts."""
        variables = ["sc_octroi"]
        report = calculate_stability_report(
            booked_main_df,
            booked_mr_df,
            variables,
            score_variable="sc_octroi",
        )
        alert_report = generate_drift_alerts(report, "test", "MR")

        # Variable alerts should have same PSI values as stability results
        psi_by_var = {r.variable: r.psi_value for r in report.psi_results}
        for alert in alert_report.alerts:
            if alert.variable != "__overall__":
                assert alert.variable in psi_by_var
                assert abs(alert.psi_value - psi_by_var[alert.variable]) < 1e-10
