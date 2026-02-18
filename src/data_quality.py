"""
Data Quality Checks Module

Pre-flight validation to catch data issues before running the pipeline.
Helps avoid wasted compute time on bad data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from src.config import PreprocessingSettings


class CheckStatus(Enum):
    """Status of a data quality check."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single data quality check."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        icon = {"passed": "✓", "warning": "⚠", "failed": "✗", "skipped": "○"}[self.status.value]
        return f"{icon} {self.name}: {self.message}"


@dataclass
class DataQualityReport:
    """Complete data quality report."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.PASSED]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.WARNING]

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == CheckStatus.FAILED]

    @property
    def is_valid(self) -> bool:
        """Returns True if no failures (warnings are acceptable)."""
        return len(self.failures) == 0

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def summary(self) -> str:
        """Generate a summary string."""
        total = len(self.checks)
        return (
            f"Data Quality: {len(self.passed)}/{total} passed, "
            f"{len(self.warnings)} warnings, {len(self.failures)} failures"
        )

    def print_report(self) -> None:
        """Print formatted report to console."""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)

        # Group by status
        for status, label in [
            (CheckStatus.FAILED, "FAILURES"),
            (CheckStatus.WARNING, "WARNINGS"),
            (CheckStatus.PASSED, "PASSED"),
        ]:
            checks = [c for c in self.checks if c.status == status]
            if checks:
                print(f"\n{label}:")
                for check in checks:
                    print(f"  {check}")
                    if check.details:
                        for key, value in check.details.items():
                            print(f"    - {key}: {value}")

        print("\n" + "-" * 60)
        print(self.summary())
        print("=" * 60 + "\n")


def check_required_columns(df: pd.DataFrame, required_columns: list[str], context: str = "data") -> CheckResult:
    """Check that required columns exist in the DataFrame."""
    df_columns = set(df.columns.str.lower())
    required_lower = {col.lower(): col for col in required_columns}

    missing = [orig for lower, orig in required_lower.items() if lower not in df_columns]

    if not missing:
        return CheckResult(
            name="Required Columns",
            status=CheckStatus.PASSED,
            message=f"All {len(required_columns)} required columns present",
        )
    else:
        return CheckResult(
            name="Required Columns",
            status=CheckStatus.FAILED,
            message=f"Missing {len(missing)} required columns",
            details={"missing_columns": missing},
        )


def check_missing_values(
    df: pd.DataFrame, columns: list[str], threshold_warn: float = 0.05, threshold_fail: float = 0.20
) -> CheckResult:
    """Check for missing values in specified columns."""
    # Filter to columns that exist
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        return CheckResult(name="Missing Values", status=CheckStatus.SKIPPED, message="No columns to check")

    if len(df) == 0:
        return CheckResult(name="Missing Values", status=CheckStatus.SKIPPED, message="DataFrame is empty")

    missing_stats = {}
    max_missing_pct = 0

    for col in existing_cols:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df)
        if missing_pct > 0:
            missing_stats[col] = f"{missing_pct:.1%} ({missing_count:,} rows)"
        max_missing_pct = max(max_missing_pct, missing_pct)

    if max_missing_pct >= threshold_fail:
        return CheckResult(
            name="Missing Values",
            status=CheckStatus.FAILED,
            message=f"High missing rate detected (>{threshold_fail:.0%})",
            details=missing_stats,
        )
    elif max_missing_pct >= threshold_warn:
        return CheckResult(
            name="Missing Values",
            status=CheckStatus.WARNING,
            message=f"Some missing values detected (>{threshold_warn:.0%})",
            details=missing_stats,
        )
    else:
        return CheckResult(name="Missing Values", status=CheckStatus.PASSED, message="No significant missing values")


def check_segment_exists(df: pd.DataFrame, segment_filter: str, segment_column: str = "segment_cut_off") -> CheckResult:
    """Check that the segment filter matches data in the DataFrame."""
    if segment_column not in df.columns:
        return CheckResult(
            name="Segment Filter", status=CheckStatus.SKIPPED, message=f"Column '{segment_column}' not found"
        )

    # Handle regex patterns (supersegments)
    if "|" in segment_filter:
        # Regex pattern
        mask = df[segment_column].astype(str).str.match(segment_filter, na=False)
        match_count = mask.sum()
    else:
        # Exact match
        match_count = (df[segment_column] == segment_filter).sum()

    if match_count == 0:
        # Show available segments for debugging
        available = df[segment_column].value_counts().head(10).to_dict()
        return CheckResult(
            name="Segment Filter",
            status=CheckStatus.FAILED,
            message=f"No data matches segment filter '{segment_filter}'",
            details={"available_segments (top 10)": available},
        )
    else:
        pct = match_count / len(df)
        return CheckResult(
            name="Segment Filter",
            status=CheckStatus.PASSED,
            message=f"Found {match_count:,} rows ({pct:.1%}) matching segment",
            details={"segment_filter": segment_filter},
        )


def check_segment_size(
    df: pd.DataFrame,
    segment_filter: str,
    segment_column: str = "segment_cut_off",
    min_rows_warn: int = 1000,
    min_rows_fail: int = 100,
) -> CheckResult:
    """Check that segment has sufficient data for modeling."""
    if segment_column not in df.columns:
        return CheckResult(
            name="Segment Size", status=CheckStatus.SKIPPED, message=f"Column '{segment_column}' not found"
        )

    # Count matching rows
    if "|" in segment_filter:
        mask = df[segment_column].astype(str).str.match(segment_filter, na=False)
        count = mask.sum()
    else:
        count = (df[segment_column] == segment_filter).sum()

    if count < min_rows_fail:
        return CheckResult(
            name="Segment Size",
            status=CheckStatus.FAILED,
            message=f"Segment too small: {count:,} rows (minimum: {min_rows_fail:,})",
            details={"row_count": count, "minimum_required": min_rows_fail},
        )
    elif count < min_rows_warn:
        return CheckResult(
            name="Segment Size",
            status=CheckStatus.WARNING,
            message=f"Segment may be too small: {count:,} rows (recommended: {min_rows_warn:,}+)",
            details={"row_count": count, "recommended_minimum": min_rows_warn},
        )
    else:
        return CheckResult(name="Segment Size", status=CheckStatus.PASSED, message=f"Segment has {count:,} rows")


def check_date_range(df: pd.DataFrame, date_column: str, expected_start: str, expected_end: str) -> CheckResult:
    """Check that data covers the expected date range."""
    if date_column not in df.columns:
        return CheckResult(name="Date Range", status=CheckStatus.SKIPPED, message=f"Column '{date_column}' not found")

    try:
        dates = pd.to_datetime(df[date_column])
        actual_start = dates.min()
        actual_end = dates.max()
        expected_start_dt = pd.to_datetime(expected_start)
        expected_end_dt = pd.to_datetime(expected_end)

        issues = []

        if actual_start > expected_start_dt:
            gap_days = (actual_start - expected_start_dt).days
            issues.append(f"Data starts {gap_days} days after expected start")

        if actual_end < expected_end_dt:
            gap_days = (expected_end_dt - actual_end).days
            issues.append(f"Data ends {gap_days} days before expected end")

        details = {
            "expected_range": f"{expected_start} to {expected_end}",
            "actual_range": f"{actual_start.date()} to {actual_end.date()}",
        }

        if issues:
            return CheckResult(
                name="Date Range", status=CheckStatus.WARNING, message="; ".join(issues), details=details
            )
        else:
            return CheckResult(
                name="Date Range", status=CheckStatus.PASSED, message="Data covers expected date range", details=details
            )
    except (ValueError, TypeError) as e:
        return CheckResult(name="Date Range", status=CheckStatus.WARNING, message=f"Could not parse dates: {e}")


def check_numeric_outliers(
    df: pd.DataFrame, columns: list[str], z_threshold: float = 5.0, max_outlier_pct: float = 0.01
) -> CheckResult:
    """Check for extreme outliers in numeric columns."""
    existing_cols = [c for c in columns if c in df.columns]
    numeric_cols = [c for c in existing_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols:
        return CheckResult(name="Numeric Outliers", status=CheckStatus.SKIPPED, message="No numeric columns to check")

    outlier_stats = {}
    has_severe_outliers = False

    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue

        mean = values.mean()
        std = values.std()

        if std == 0:
            continue

        z_scores = np.abs((values - mean) / std)
        outlier_count = (z_scores > z_threshold).sum()
        outlier_pct = outlier_count / len(values)

        if outlier_pct > max_outlier_pct:
            outlier_stats[col] = f"{outlier_pct:.2%} outliers (z>{z_threshold})"
            has_severe_outliers = True

    if has_severe_outliers:
        return CheckResult(
            name="Numeric Outliers",
            status=CheckStatus.WARNING,
            message=f"Extreme outliers detected in {len(outlier_stats)} columns",
            details=outlier_stats,
        )
    else:
        return CheckResult(
            name="Numeric Outliers",
            status=CheckStatus.PASSED,
            message=f"No severe outliers in {len(numeric_cols)} numeric columns",
        )


def check_indicator_values(df: pd.DataFrame, indicators: list[str]) -> CheckResult:
    """Check that indicator columns have valid values (non-negative for counts/amounts)."""
    existing = [c for c in indicators if c in df.columns]

    if not existing:
        return CheckResult(name="Indicator Values", status=CheckStatus.SKIPPED, message="No indicator columns found")

    issues = {}

    for col in existing:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            issues[col] = f"{neg_count:,} negative values"

    if issues:
        return CheckResult(
            name="Indicator Values",
            status=CheckStatus.WARNING,
            message=f"Negative values in {len(issues)} indicator columns",
            details=issues,
        )
    else:
        return CheckResult(
            name="Indicator Values",
            status=CheckStatus.PASSED,
            message=f"All {len(existing)} indicators have valid values",
        )


def check_booked_ratio(
    df: pd.DataFrame,
    status_column: str = "status_name",
    booked_value: str = "booked",
    min_ratio_warn: float = 0.05,
    min_ratio_fail: float = 0.01,
) -> CheckResult:
    """Check that there's a reasonable ratio of booked vs total applications."""
    if status_column not in df.columns:
        return CheckResult(
            name="Booked Ratio", status=CheckStatus.SKIPPED, message=f"Column '{status_column}' not found"
        )

    status_lower = df[status_column].astype(str).str.lower()
    booked_count = (status_lower == booked_value.lower()).sum()
    total_count = len(df)
    ratio = booked_count / total_count if total_count > 0 else 0

    details = {"booked_count": f"{booked_count:,}", "total_count": f"{total_count:,}", "ratio": f"{ratio:.1%}"}

    if ratio < min_ratio_fail:
        return CheckResult(
            name="Booked Ratio",
            status=CheckStatus.FAILED,
            message=f"Very low booked ratio: {ratio:.1%} (minimum: {min_ratio_fail:.0%})",
            details=details,
        )
    elif ratio < min_ratio_warn:
        return CheckResult(
            name="Booked Ratio", status=CheckStatus.WARNING, message=f"Low booked ratio: {ratio:.1%}", details=details
        )
    else:
        return CheckResult(
            name="Booked Ratio",
            status=CheckStatus.PASSED,
            message=f"Booked ratio: {ratio:.1%} ({booked_count:,} of {total_count:,})",
        )


def check_duplicate_rows(
    df: pd.DataFrame, key_columns: list[str] | None = None, max_dup_pct: float = 0.01
) -> CheckResult:
    """Check for duplicate rows."""
    if key_columns:
        existing_keys = [c for c in key_columns if c in df.columns]
        if not existing_keys:
            return CheckResult(name="Duplicate Rows", status=CheckStatus.SKIPPED, message="No key columns found")
        dup_count = df.duplicated(subset=existing_keys).sum()
        context = f"on key columns {existing_keys}"
    else:
        dup_count = df.duplicated().sum()
        context = "across all columns"

    dup_pct = dup_count / len(df) if len(df) > 0 else 0

    if dup_pct > max_dup_pct:
        return CheckResult(
            name="Duplicate Rows",
            status=CheckStatus.WARNING,
            message=f"{dup_count:,} duplicate rows ({dup_pct:.1%}) {context}",
            details={"duplicate_count": dup_count, "percentage": f"{dup_pct:.2%}"},
        )
    else:
        return CheckResult(
            name="Duplicate Rows", status=CheckStatus.PASSED, message=f"No significant duplicates {context}"
        )


def run_data_quality_checks(
    df: pd.DataFrame, settings: "PreprocessingSettings", verbose: bool = True
) -> DataQualityReport:
    """
    Run all data quality checks on the input data.

    Args:
        df: Input DataFrame
        settings: Configuration settings object
        verbose: Whether to print the report

    Returns:
        DataQualityReport with all check results
    """
    report = DataQualityReport()

    # Extract config values
    keep_vars = settings.keep_vars
    indicators = settings.indicators
    segment_filter = settings.segment_filter
    date_ini = settings.date_ini_book_obs
    date_fin = settings.date_fin_book_obs

    required_columns = keep_vars + indicators + ["segment_cut_off", "status_name", "mis_date"]

    # Columns that naturally have missing values (only populated for booked records)
    missing_exempt = {"reject_reason", "early_bad", "acct_booked_h0", "todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"}
    missing_check_columns = [c for c in required_columns if c not in missing_exempt]

    logger.info("Running data quality checks...")

    # Run checks
    report.add(check_required_columns(df, required_columns))
    report.add(check_missing_values(df, missing_check_columns))
    report.add(check_segment_exists(df, segment_filter))
    report.add(check_segment_size(df, segment_filter))
    report.add(check_date_range(df, "mis_date", date_ini, date_fin))
    report.add(check_numeric_outliers(df, indicators))
    report.add(check_indicator_values(df, indicators))
    report.add(check_booked_ratio(df))
    report.add(check_duplicate_rows(df))

    if verbose:
        report.print_report()

    return report


def validate_data_or_fail(
    df: pd.DataFrame, settings: "PreprocessingSettings", allow_warnings: bool = True
) -> DataQualityReport:
    """
    Run data quality checks and raise an exception if validation fails.

    Args:
        df: Input DataFrame
        settings: Configuration settings object
        allow_warnings: If True, only fail on errors; if False, fail on warnings too

    Returns:
        DataQualityReport if validation passes

    Raises:
        ValueError: If validation fails
    """
    report = run_data_quality_checks(df, settings, verbose=True)

    if not report.is_valid:
        raise ValueError(
            f"Data quality validation failed with {len(report.failures)} errors. "
            "Fix the issues above before running the pipeline."
        )

    if not allow_warnings and report.warnings:
        raise ValueError(
            f"Data quality validation found {len(report.warnings)} warnings. Set allow_warnings=True to proceed anyway."
        )

    logger.info(f"Data quality validation passed: {report.summary()}")
    return report
