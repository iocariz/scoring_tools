"""Tests for the drift alert system."""

import json

import pandas as pd

from src.alerts import (
    AlertReport,
    AlertSeverity,
    DriftAlert,
    generate_drift_alerts,
)
from src.stability import PSIResult, StabilityReport, StabilityStatus


def _make_psi_result(variable: str, psi_value: float, status: StabilityStatus) -> PSIResult:
    """Create a PSIResult with minimal bin details."""
    return PSIResult(
        variable=variable,
        psi_value=psi_value,
        status=status,
        n_bins=10,
        bin_details=pd.DataFrame(
            {"bin": ["a"], "baseline_pct": [0.5], "comparison_pct": [0.5], "psi_component": [0.0]}
        ),
    )


def _make_stability_report(
    psi_results: list[PSIResult] | None = None,
    overall_psi: float | None = None,
) -> StabilityReport:
    """Create a StabilityReport for testing."""
    report = StabilityReport(
        baseline_name="Main",
        comparison_name="MR",
        baseline_count=1000,
        comparison_count=800,
        overall_psi=overall_psi,
    )
    if psi_results:
        report.psi_results = psi_results
    return report


class TestAlertSeverity:
    def test_enum_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestDriftAlert:
    def test_to_dict(self):
        alert = DriftAlert(
            timestamp="2024-01-01T00:00:00",
            severity=AlertSeverity.CRITICAL,
            variable="sc_octroi",
            psi_value=0.31234567,
            status="unstable",
            message="Test message",
            segment="consumer",
            period="MR",
        )
        d = alert.to_dict()
        assert d["severity"] == "critical"
        assert d["psi_value"] == 0.3123  # rounded to 4 decimals
        assert d["variable"] == "sc_octroi"
        assert d["segment"] == "consumer"
        assert d["period"] == "MR"

    def test_to_dict_preserves_all_fields(self):
        alert = DriftAlert(
            timestamp="ts",
            severity=AlertSeverity.INFO,
            variable="var",
            psi_value=0.01,
            status="stable",
            message="msg",
            segment="seg",
            period="per",
        )
        d = alert.to_dict()
        assert set(d.keys()) == {
            "timestamp",
            "severity",
            "variable",
            "psi_value",
            "status",
            "message",
            "segment",
            "period",
        }


class TestAlertReport:
    def test_empty_report(self):
        report = AlertReport()
        assert report.summary == {"info": 0, "warning": 0, "critical": 0}
        assert report.critical_alerts == []
        assert report.warning_alerts == []

    def test_summary_counts(self):
        report = AlertReport(
            alerts=[
                DriftAlert("ts", AlertSeverity.CRITICAL, "a", 0.3, "unstable", "m", "s", "p"),
                DriftAlert("ts", AlertSeverity.WARNING, "b", 0.15, "moderate", "m", "s", "p"),
                DriftAlert("ts", AlertSeverity.INFO, "c", 0.05, "stable", "m", "s", "p"),
                DriftAlert("ts", AlertSeverity.CRITICAL, "d", 0.4, "unstable", "m", "s", "p"),
            ],
            segment="test",
            generated_at="2024-01-01",
        )
        assert report.summary == {"info": 1, "warning": 1, "critical": 2}
        assert len(report.critical_alerts) == 2
        assert len(report.warning_alerts) == 1

    def test_to_json(self, tmp_path):
        report = AlertReport(
            alerts=[
                DriftAlert("ts", AlertSeverity.CRITICAL, "a", 0.3, "unstable", "msg", "seg", "MR"),
            ],
            segment="consumer",
            generated_at="2024-01-01T00:00:00",
        )
        path = str(tmp_path / "alerts.json")
        report.to_json(path)

        with open(path) as f:
            data = json.load(f)

        assert data["segment"] == "consumer"
        assert data["generated_at"] == "2024-01-01T00:00:00"
        assert len(data["alerts"]) == 1
        assert data["alerts"][0]["severity"] == "critical"
        assert data["summary"]["critical"] == 1

    def test_to_dataframe_empty(self):
        report = AlertReport()
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_to_dataframe_with_alerts(self):
        report = AlertReport(
            alerts=[
                DriftAlert("ts", AlertSeverity.INFO, "a", 0.05, "stable", "m", "s", "p"),
                DriftAlert("ts", AlertSeverity.WARNING, "b", 0.15, "moderate", "m", "s", "p"),
            ]
        )
        df = report.to_dataframe()
        assert len(df) == 2
        assert "severity" in df.columns
        assert "variable" in df.columns


class TestGenerateDriftAlerts:
    def test_all_stable(self):
        results = [
            _make_psi_result("var1", 0.02, StabilityStatus.STABLE),
            _make_psi_result("var2", 0.05, StabilityStatus.STABLE),
        ]
        stability_report = _make_stability_report(results, overall_psi=0.03)
        alert_report = generate_drift_alerts(stability_report, "consumer", "MR")

        # 2 variable alerts + 1 overall alert
        assert len(alert_report.alerts) == 3
        assert alert_report.summary["critical"] == 0
        assert alert_report.summary["warning"] == 0
        assert alert_report.summary["info"] == 3

    def test_mixed_statuses(self):
        results = [
            _make_psi_result("stable_var", 0.02, StabilityStatus.STABLE),
            _make_psi_result("moderate_var", 0.15, StabilityStatus.MODERATE),
            _make_psi_result("unstable_var", 0.30, StabilityStatus.UNSTABLE),
        ]
        stability_report = _make_stability_report(results, overall_psi=0.15)
        alert_report = generate_drift_alerts(stability_report, "consumer", "MR")

        # 3 variable alerts + 1 overall alert
        assert len(alert_report.alerts) == 4
        assert alert_report.summary["critical"] == 1
        assert alert_report.summary["warning"] == 2  # moderate_var + overall
        assert alert_report.summary["info"] == 1

    def test_unstable_overall(self):
        results = [
            _make_psi_result("var1", 0.30, StabilityStatus.UNSTABLE),
        ]
        stability_report = _make_stability_report(results, overall_psi=0.30)
        alert_report = generate_drift_alerts(stability_report, "consumer", "MR")

        # Check overall alert message mentions recalibration
        overall_alert = [a for a in alert_report.alerts if a.variable == "__overall__"]
        assert len(overall_alert) == 1
        assert "recalibration" in overall_alert[0].message.lower()

    def test_no_overall_psi(self):
        results = [
            _make_psi_result("var1", 0.05, StabilityStatus.STABLE),
        ]
        stability_report = _make_stability_report(results, overall_psi=None)
        alert_report = generate_drift_alerts(stability_report, "consumer", "MR")

        # Only 1 variable alert, no overall
        assert len(alert_report.alerts) == 1
        assert all(a.variable != "__overall__" for a in alert_report.alerts)

    def test_segment_and_period_propagated(self):
        results = [
            _make_psi_result("var1", 0.05, StabilityStatus.STABLE),
        ]
        stability_report = _make_stability_report(results)
        alert_report = generate_drift_alerts(stability_report, "corporate", "Q2")

        assert alert_report.segment == "corporate"
        for alert in alert_report.alerts:
            assert alert.segment == "corporate"
            assert alert.period == "Q2"

    def test_severity_mapping(self):
        """Verify severity mapping: STABLE->INFO, MODERATE->WARNING, UNSTABLE->CRITICAL."""
        results = [
            _make_psi_result("s", 0.02, StabilityStatus.STABLE),
            _make_psi_result("m", 0.15, StabilityStatus.MODERATE),
            _make_psi_result("u", 0.30, StabilityStatus.UNSTABLE),
        ]
        stability_report = _make_stability_report(results)
        alert_report = generate_drift_alerts(stability_report, "seg", "per")

        alerts_by_var = {a.variable: a for a in alert_report.alerts}
        assert alerts_by_var["s"].severity == AlertSeverity.INFO
        assert alerts_by_var["m"].severity == AlertSeverity.WARNING
        assert alerts_by_var["u"].severity == AlertSeverity.CRITICAL

    def test_message_contains_variable_and_psi(self):
        results = [
            _make_psi_result("sc_octroi", 0.15, StabilityStatus.MODERATE),
        ]
        stability_report = _make_stability_report(results)
        alert_report = generate_drift_alerts(stability_report, "seg", "per")

        alert = alert_report.alerts[0]
        assert "sc_octroi" in alert.message
        assert "0.1500" in alert.message

    def test_empty_stability_report(self):
        stability_report = _make_stability_report([], overall_psi=None)
        alert_report = generate_drift_alerts(stability_report, "seg", "per")

        assert len(alert_report.alerts) == 0
        assert alert_report.summary == {"info": 0, "warning": 0, "critical": 0}
