"""
Automated drift alert system.

Generates structured alerts from PSI/CSI stability reports, enabling
downstream consumption (JSON files, logging, notification systems).
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import pandas as pd
from loguru import logger

from src.stability import StabilityReport, StabilityStatus


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """A single drift alert for a variable or overall score."""

    timestamp: str
    severity: AlertSeverity
    variable: str
    psi_value: float
    status: str
    message: str
    segment: str
    period: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "variable": self.variable,
            "psi_value": round(self.psi_value, 4),
            "status": self.status,
            "message": self.message,
            "segment": self.segment,
            "period": self.period,
        }


@dataclass
class AlertReport:
    """Collection of drift alerts from a stability analysis run."""

    alerts: list[DriftAlert] = field(default_factory=list)
    segment: str = ""
    generated_at: str = ""

    @property
    def summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in AlertSeverity}
        for alert in self.alerts:
            counts[alert.severity.value] += 1
        return counts

    @property
    def critical_alerts(self) -> list[DriftAlert]:
        return [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]

    @property
    def warning_alerts(self) -> list[DriftAlert]:
        return [a for a in self.alerts if a.severity == AlertSeverity.WARNING]

    def to_json(self, path: str) -> None:
        """Save alerts to a JSON file."""
        data = {
            "segment": self.segment,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "alerts": [a.to_dict() for a in self.alerts],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert alerts to a DataFrame."""
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame([a.to_dict() for a in self.alerts])


_SEVERITY_MAP = {
    StabilityStatus.STABLE: AlertSeverity.INFO,
    StabilityStatus.MODERATE: AlertSeverity.WARNING,
    StabilityStatus.UNSTABLE: AlertSeverity.CRITICAL,
}

_MESSAGE_MAP = {
    StabilityStatus.STABLE: "Variable '{var}' is stable (PSI={psi:.4f}).",
    StabilityStatus.MODERATE: "Variable '{var}' shows moderate drift (PSI={psi:.4f}). Investigation recommended.",
    StabilityStatus.UNSTABLE: "Variable '{var}' shows significant drift (PSI={psi:.4f}). Action required.",
}


def generate_drift_alerts(
    stability_report: StabilityReport,
    segment: str,
    period: str,
) -> AlertReport:
    """
    Generate structured alerts from a stability report.

    Args:
        stability_report: PSI/CSI stability report.
        segment: Segment name for context.
        period: Period label (e.g., "MR").

    Returns:
        AlertReport with one alert per variable plus an optional overall alert.
    """
    now = datetime.now(UTC).isoformat()

    report = AlertReport(segment=segment, generated_at=now)

    for result in stability_report.psi_results:
        severity = _SEVERITY_MAP[result.status]
        message = _MESSAGE_MAP[result.status].format(var=result.variable, psi=result.psi_value)

        report.alerts.append(
            DriftAlert(
                timestamp=now,
                severity=severity,
                variable=result.variable,
                psi_value=result.psi_value,
                status=result.status.value,
                message=message,
                segment=segment,
                period=period,
            )
        )

    # Overall PSI alert
    if stability_report.overall_psi is not None:
        from src.stability import get_psi_status

        overall_status = get_psi_status(stability_report.overall_psi)
        severity = _SEVERITY_MAP[overall_status]
        message = f"Overall score PSI={stability_report.overall_psi:.4f} ({overall_status.value})."
        if overall_status == StabilityStatus.UNSTABLE:
            message += " Model recalibration may be required."

        report.alerts.append(
            DriftAlert(
                timestamp=now,
                severity=severity,
                variable="__overall__",
                psi_value=stability_report.overall_psi,
                status=overall_status.value,
                message=message,
                segment=segment,
                period=period,
            )
        )

    # Log critical alerts
    for alert in report.critical_alerts:
        logger.warning(f"[DRIFT ALERT] {alert.message}")

    summary = report.summary
    logger.info(
        f"Drift alerts generated for {segment}/{period}: "
        f"{summary['critical']} critical, {summary['warning']} warning, {summary['info']} info"
    )

    return report
