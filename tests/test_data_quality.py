import numpy as np
import pandas as pd
import pytest

from src.data_quality import (
    CheckResult,
    CheckStatus,
    DataQualityReport,
    check_booked_ratio,
    check_date_range,
    check_duplicate_rows,
    check_indicator_values,
    check_missing_values,
    check_numeric_outliers,
    check_required_columns,
    check_segment_exists,
    check_segment_size,
    run_data_quality_checks,
    validate_data_or_fail,
)


# =============================================================================
# CheckResult Tests
# =============================================================================


class TestCheckResult:
    def test_str_passed(self):
        r = CheckResult(name="Test", status=CheckStatus.PASSED, message="ok")
        assert "✓" in str(r)
        assert "Test" in str(r)

    def test_str_failed(self):
        r = CheckResult(name="Test", status=CheckStatus.FAILED, message="bad")
        assert "✗" in str(r)

    def test_str_warning(self):
        r = CheckResult(name="Test", status=CheckStatus.WARNING, message="warn")
        assert "⚠" in str(r)

    def test_str_skipped(self):
        r = CheckResult(name="Test", status=CheckStatus.SKIPPED, message="skip")
        assert "○" in str(r)

    def test_details_optional(self):
        r = CheckResult(name="Test", status=CheckStatus.PASSED, message="ok")
        assert r.details is None


# =============================================================================
# DataQualityReport Tests
# =============================================================================


class TestDataQualityReport:
    def test_empty_report_is_valid(self):
        report = DataQualityReport()
        assert report.is_valid
        assert len(report.checks) == 0

    def test_add_check(self):
        report = DataQualityReport()
        report.add(CheckResult("Test", CheckStatus.PASSED, "ok"))
        assert len(report.checks) == 1

    def test_passed_filter(self):
        report = DataQualityReport()
        report.add(CheckResult("Pass1", CheckStatus.PASSED, "ok"))
        report.add(CheckResult("Fail1", CheckStatus.FAILED, "bad"))
        assert len(report.passed) == 1
        assert report.passed[0].name == "Pass1"

    def test_warnings_filter(self):
        report = DataQualityReport()
        report.add(CheckResult("Warn1", CheckStatus.WARNING, "warn"))
        report.add(CheckResult("Pass1", CheckStatus.PASSED, "ok"))
        assert len(report.warnings) == 1

    def test_failures_filter(self):
        report = DataQualityReport()
        report.add(CheckResult("Fail1", CheckStatus.FAILED, "bad"))
        report.add(CheckResult("Fail2", CheckStatus.FAILED, "bad2"))
        assert len(report.failures) == 2

    def test_is_valid_with_failures(self):
        report = DataQualityReport()
        report.add(CheckResult("Fail1", CheckStatus.FAILED, "bad"))
        assert not report.is_valid

    def test_is_valid_with_warnings_only(self):
        report = DataQualityReport()
        report.add(CheckResult("Warn1", CheckStatus.WARNING, "warn"))
        assert report.is_valid

    def test_summary(self):
        report = DataQualityReport()
        report.add(CheckResult("Pass1", CheckStatus.PASSED, "ok"))
        report.add(CheckResult("Warn1", CheckStatus.WARNING, "warn"))
        report.add(CheckResult("Fail1", CheckStatus.FAILED, "bad"))
        summary = report.summary()
        assert "1/3 passed" in summary
        assert "1 warnings" in summary
        assert "1 failures" in summary

    def test_print_report(self, capsys):
        report = DataQualityReport()
        report.add(CheckResult("Pass1", CheckStatus.PASSED, "ok"))
        report.add(CheckResult("Fail1", CheckStatus.FAILED, "bad", details={"key": "val"}))
        report.print_report()
        captured = capsys.readouterr()
        assert "DATA QUALITY REPORT" in captured.out
        assert "FAILURES" in captured.out
        assert "key" in captured.out


# =============================================================================
# check_required_columns Tests
# =============================================================================


class TestCheckRequiredColumns:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = check_required_columns(df, ["a", "b", "c"])
        assert result.status == CheckStatus.PASSED

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        result = check_required_columns(df, ["a", "b", "c"])
        assert result.status == CheckStatus.FAILED
        assert "c" in result.details["missing_columns"]

    def test_case_insensitive(self):
        df = pd.DataFrame({"Status_Name": [1]})
        result = check_required_columns(df, ["status_name"])
        assert result.status == CheckStatus.PASSED


# =============================================================================
# check_missing_values Tests
# =============================================================================


class TestCheckMissingValues:
    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = check_missing_values(df, ["a", "b"])
        assert result.status == CheckStatus.PASSED

    def test_high_missing_fails(self):
        df = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, np.nan]})
        result = check_missing_values(df, ["a"], threshold_fail=0.5)
        assert result.status == CheckStatus.FAILED

    def test_moderate_missing_warns(self):
        df = pd.DataFrame({"a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]})
        result = check_missing_values(df, ["a"], threshold_warn=0.05, threshold_fail=0.20)
        assert result.status == CheckStatus.WARNING

    def test_nonexistent_columns_skipped(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = check_missing_values(df, ["nonexistent"])
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# check_segment_exists Tests
# =============================================================================


class TestCheckSegmentExists:
    def test_segment_found(self):
        df = pd.DataFrame({"segment_cut_off": ["A", "A", "B"]})
        result = check_segment_exists(df, "A")
        assert result.status == CheckStatus.PASSED

    def test_segment_not_found(self):
        df = pd.DataFrame({"segment_cut_off": ["A", "A"]})
        result = check_segment_exists(df, "B")
        assert result.status == CheckStatus.FAILED

    def test_regex_pattern(self):
        df = pd.DataFrame({"segment_cut_off": ["seg_a", "seg_b", "seg_c"]})
        result = check_segment_exists(df, "seg_a|seg_b")
        assert result.status == CheckStatus.PASSED

    def test_missing_column(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        result = check_segment_exists(df, "A")
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# check_segment_size Tests
# =============================================================================


class TestCheckSegmentSize:
    def test_sufficient_size(self):
        df = pd.DataFrame({"segment_cut_off": ["A"] * 2000})
        result = check_segment_size(df, "A", min_rows_warn=1000, min_rows_fail=100)
        assert result.status == CheckStatus.PASSED

    def test_too_small_fails(self):
        df = pd.DataFrame({"segment_cut_off": ["A"] * 50})
        result = check_segment_size(df, "A", min_rows_fail=100)
        assert result.status == CheckStatus.FAILED

    def test_small_warns(self):
        df = pd.DataFrame({"segment_cut_off": ["A"] * 500})
        result = check_segment_size(df, "A", min_rows_warn=1000, min_rows_fail=100)
        assert result.status == CheckStatus.WARNING

    def test_regex_segment(self):
        df = pd.DataFrame({"segment_cut_off": ["seg_a"] * 500 + ["seg_b"] * 600})
        result = check_segment_size(df, "seg_a|seg_b", min_rows_warn=500, min_rows_fail=100)
        assert result.status == CheckStatus.PASSED

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        result = check_segment_size(df, "A")
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# check_date_range Tests
# =============================================================================


class TestCheckDateRange:
    def test_covers_expected_range(self):
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", "2023-12-31", freq="MS")})
        result = check_date_range(df, "date", "2023-01-01", "2023-12-01")
        assert result.status == CheckStatus.PASSED

    def test_data_starts_late(self):
        df = pd.DataFrame({"date": pd.date_range("2023-06-01", "2023-12-31", freq="MS")})
        result = check_date_range(df, "date", "2023-01-01", "2023-12-01")
        assert result.status == CheckStatus.WARNING
        assert "starts" in result.message

    def test_data_ends_early(self):
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", "2023-06-30", freq="MS")})
        result = check_date_range(df, "date", "2023-01-01", "2023-12-01")
        assert result.status == CheckStatus.WARNING
        assert "ends" in result.message

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        result = check_date_range(df, "date", "2023-01-01", "2023-12-01")
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# check_numeric_outliers Tests
# =============================================================================


class TestCheckNumericOutliers:
    def test_no_outliers(self):
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
        result = check_numeric_outliers(df, ["a"], z_threshold=5.0)
        assert result.status == CheckStatus.PASSED

    def test_with_outliers(self):
        data = np.random.normal(0, 1, 1000)
        data[:50] = 100  # Extreme outliers
        df = pd.DataFrame({"a": data})
        result = check_numeric_outliers(df, ["a"], z_threshold=3.0, max_outlier_pct=0.01)
        assert result.status == CheckStatus.WARNING

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = check_numeric_outliers(df, ["a"])
        assert result.status == CheckStatus.SKIPPED

    def test_nonexistent_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = check_numeric_outliers(df, ["nonexistent"])
        assert result.status == CheckStatus.SKIPPED

    def test_zero_std_column(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
        result = check_numeric_outliers(df, ["a"])
        assert result.status == CheckStatus.PASSED


# =============================================================================
# check_indicator_values Tests
# =============================================================================


class TestCheckIndicatorValues:
    def test_all_positive(self):
        df = pd.DataFrame({"ind1": [1, 2, 3], "ind2": [4, 5, 6]})
        result = check_indicator_values(df, ["ind1", "ind2"])
        assert result.status == CheckStatus.PASSED

    def test_negative_values_warn(self):
        df = pd.DataFrame({"ind1": [1, -2, 3]})
        result = check_indicator_values(df, ["ind1"])
        assert result.status == CheckStatus.WARNING

    def test_no_indicators(self):
        df = pd.DataFrame({"a": [1]})
        result = check_indicator_values(df, ["nonexistent"])
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# check_booked_ratio Tests
# =============================================================================


class TestCheckBookedRatio:
    def test_good_ratio(self):
        df = pd.DataFrame({"status_name": ["booked"] * 30 + ["rejected"] * 70})
        result = check_booked_ratio(df)
        assert result.status == CheckStatus.PASSED

    def test_very_low_ratio_fails(self):
        df = pd.DataFrame({"status_name": ["rejected"] * 1000})
        result = check_booked_ratio(df, min_ratio_fail=0.01)
        assert result.status == CheckStatus.FAILED

    def test_low_ratio_warns(self):
        df = pd.DataFrame({"status_name": ["booked"] * 3 + ["rejected"] * 97})
        result = check_booked_ratio(df, min_ratio_warn=0.05, min_ratio_fail=0.01)
        assert result.status == CheckStatus.WARNING

    def test_missing_column(self):
        df = pd.DataFrame({"other": [1]})
        result = check_booked_ratio(df)
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# check_duplicate_rows Tests
# =============================================================================


class TestCheckDuplicateRows:
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = check_duplicate_rows(df)
        assert result.status == CheckStatus.PASSED

    def test_with_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": [4, 4, 4, 5, 6]})
        result = check_duplicate_rows(df, max_dup_pct=0.01)
        assert result.status == CheckStatus.WARNING

    def test_key_columns(self):
        df = pd.DataFrame({"id": [1, 1, 2, 3], "val": [10, 20, 30, 40]})
        result = check_duplicate_rows(df, key_columns=["id"], max_dup_pct=0.01)
        assert result.status == CheckStatus.WARNING

    def test_nonexistent_key_columns(self):
        df = pd.DataFrame({"a": [1]})
        result = check_duplicate_rows(df, key_columns=["nonexistent"])
        assert result.status == CheckStatus.SKIPPED


# =============================================================================
# run_data_quality_checks Tests
# =============================================================================


class TestRunDataQualityChecks:
    def _make_df(self):
        return pd.DataFrame(
            {
                "segment_cut_off": ["A"] * 200,
                "status_name": ["booked"] * 100 + ["rejected"] * 100,
                "mis_date": pd.date_range("2023-01-01", periods=200, freq="D"),
                "oa_amt": np.random.rand(200) * 1000,
                "score": np.random.rand(200),
            }
        )

    def test_basic_run(self):
        df = self._make_df()
        config = {
            "keep_vars": ["score"],
            "indicators": ["oa_amt"],
            "segment_filter": "A",
            "date_ini_book_obs": "2023-01-01",
            "date_fin_book_obs": "2023-07-01",
        }
        report = run_data_quality_checks(df, config, verbose=False)
        assert isinstance(report, DataQualityReport)
        assert len(report.checks) > 0


# =============================================================================
# validate_data_or_fail Tests
# =============================================================================


class TestValidateDataOrFail:
    def test_fails_on_failure(self):
        df = pd.DataFrame({"a": [1]})  # Missing required columns
        config = {
            "keep_vars": ["missing_col"],
            "indicators": ["missing_ind"],
            "segment_filter": "A",
            "date_ini_book_obs": "2023-01-01",
            "date_fin_book_obs": "2023-12-01",
        }
        with pytest.raises(ValueError, match="Data quality validation failed"):
            validate_data_or_fail(df, config)

    def test_passes_valid_data(self):
        df = pd.DataFrame(
            {
                "segment_cut_off": ["A"] * 200,
                "status_name": ["booked"] * 100 + ["rejected"] * 100,
                "mis_date": pd.date_range("2023-01-01", periods=200, freq="D"),
                "oa_amt": np.random.rand(200) * 1000,
            }
        )
        config = {
            "keep_vars": ["oa_amt"],
            "indicators": [],
            "segment_filter": "A",
            "date_ini_book_obs": "2023-01-01",
            "date_fin_book_obs": "2023-07-01",
        }
        report = validate_data_or_fail(df, config)
        assert report.is_valid
