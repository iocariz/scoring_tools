import numpy as np
import pandas as pd
import pytest

from src.stability import (
    PSIResult,
    StabilityReport,
    StabilityStatus,
    calculate_csi_for_categorical,
    calculate_psi,
    calculate_psi_for_variable,
    calculate_stability_report,
    get_psi_status,
)

# =============================================================================
# get_psi_status Tests
# =============================================================================


class TestGetPsiStatus:
    def test_stable(self):
        assert get_psi_status(0.05) == StabilityStatus.STABLE

    def test_moderate(self):
        assert get_psi_status(0.15) == StabilityStatus.MODERATE

    def test_unstable(self):
        assert get_psi_status(0.30) == StabilityStatus.UNSTABLE

    def test_boundary_stable(self):
        assert get_psi_status(0.0) == StabilityStatus.STABLE

    def test_boundary_moderate(self):
        assert get_psi_status(0.1) == StabilityStatus.MODERATE

    def test_boundary_unstable(self):
        assert get_psi_status(0.25) == StabilityStatus.UNSTABLE


# =============================================================================
# PSIResult Tests
# =============================================================================


class TestPSIResult:
    def test_str_representation(self):
        r = PSIResult(
            variable="score",
            psi_value=0.05,
            status=StabilityStatus.STABLE,
            n_bins=10,
            bin_details=pd.DataFrame(),
        )
        s = str(r)
        assert "score" in s
        assert "0.05" in s
        assert "stable" in s

    def test_status_icon_stable(self):
        r = PSIResult("v", 0.01, StabilityStatus.STABLE, 5, pd.DataFrame())
        assert r.status_icon == "✓"

    def test_status_icon_moderate(self):
        r = PSIResult("v", 0.15, StabilityStatus.MODERATE, 5, pd.DataFrame())
        assert r.status_icon == "⚠"

    def test_status_icon_unstable(self):
        r = PSIResult("v", 0.30, StabilityStatus.UNSTABLE, 5, pd.DataFrame())
        assert r.status_icon == "✗"


# =============================================================================
# StabilityReport Tests
# =============================================================================


class TestStabilityReport:
    def _make_report(self):
        report = StabilityReport(baseline_name="Main", comparison_name="MR", baseline_count=1000, comparison_count=500)
        report.add(PSIResult("v1", 0.01, StabilityStatus.STABLE, 10, pd.DataFrame()))
        report.add(PSIResult("v2", 0.15, StabilityStatus.MODERATE, 10, pd.DataFrame()))
        report.add(PSIResult("v3", 0.30, StabilityStatus.UNSTABLE, 10, pd.DataFrame()))
        return report

    def test_stable_vars(self):
        report = self._make_report()
        assert len(report.stable_vars) == 1
        assert report.stable_vars[0].variable == "v1"

    def test_moderate_vars(self):
        report = self._make_report()
        assert len(report.moderate_vars) == 1

    def test_unstable_vars(self):
        report = self._make_report()
        assert len(report.unstable_vars) == 1

    def test_is_stable(self):
        report = StabilityReport("A", "B", 100, 100)
        report.add(PSIResult("v", 0.05, StabilityStatus.STABLE, 5, pd.DataFrame()))
        assert report.is_stable

    def test_not_stable(self):
        report = self._make_report()
        assert not report.is_stable

    def test_summary(self):
        report = self._make_report()
        s = report.summary()
        assert "1/3 stable" in s
        assert "1 moderate" in s
        assert "1 unstable" in s

    def test_to_dataframe(self):
        report = self._make_report()
        df = report.to_dataframe()
        assert len(df) == 3
        assert "variable" in df.columns
        assert "psi" in df.columns
        assert "status" in df.columns

    def test_print_report(self):
        from io import StringIO

        from loguru import logger

        report = self._make_report()
        report.overall_psi = 0.12

        # Capture loguru output via a temporary sink
        buffer = StringIO()
        sink_id = logger.add(buffer, format="{message}")
        try:
            report.print_report()
        finally:
            logger.remove(sink_id)

        output = buffer.getvalue()
        assert "STABILITY REPORT" in output
        assert "Main" in output
        assert "MR" in output


# =============================================================================
# calculate_psi Tests (stability.py version)
# =============================================================================


class TestCalculatePsi:
    def test_identical_distributions(self):
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 500))
        psi, breakdown = calculate_psi(data, data, bins=10)
        assert psi < 0.01
        assert isinstance(breakdown, pd.DataFrame)

    def test_different_distributions(self):
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 500))
        comparison = pd.Series(np.random.normal(2, 1, 500))
        psi, _ = calculate_psi(baseline, comparison, bins=10)
        assert psi > 0.1  # Should show drift

    def test_empty_baseline(self):
        baseline = pd.Series([], dtype=float)
        comparison = pd.Series([1, 2, 3])
        psi, breakdown = calculate_psi(baseline, comparison)
        assert psi == 0.0
        assert breakdown.empty

    def test_empty_comparison(self):
        baseline = pd.Series([1, 2, 3])
        comparison = pd.Series([], dtype=float)
        psi, breakdown = calculate_psi(baseline, comparison)
        assert psi == 0.0

    def test_with_nan_values(self):
        np.random.seed(42)
        baseline = pd.Series([1, 2, 3, np.nan, 5, 6])
        comparison = pd.Series([1, np.nan, 3, 4, 5, 6])
        psi, breakdown = calculate_psi(baseline, comparison, bins=3)
        assert isinstance(psi, float)

    def test_custom_bin_edges(self):
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 200))
        comparison = pd.Series(np.random.normal(0, 1, 200))
        psi, breakdown = calculate_psi(baseline, comparison, bins=[-3, -1, 0, 1, 3])
        assert isinstance(psi, float)
        assert len(breakdown) > 0

    def test_psi_non_negative(self):
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 300))
        comparison = pd.Series(np.random.normal(0.5, 1.5, 300))
        psi, _ = calculate_psi(baseline, comparison, bins=10)
        assert psi >= 0

    def test_psi_uses_original_difference_with_epsilon_log(self):
        baseline = pd.Series([0] * 100)
        comparison = pd.Series([1] * 100)

        psi, _ = calculate_psi(baseline, comparison, bins=[-0.5, 0.5, 1.5], min_pct=0.2)

        # Standard PSI: difference uses original proportions (0, 1),
        # epsilon only protects the log term: (0-1)*log(0.2/1) + (1-0)*log(1/0.2) = 2*log(5)
        assert psi == pytest.approx(2.0 * np.log(5.0))

    def test_psi_symmetric_under_population_swap(self):
        """Audit #11: the eps-only-in-log variant treats appearing and disappearing
        bins identically. PSI is invariant under swapping the two populations
        (PSI(a,e) = sum((a-e)*log(a/e)) = PSI(e,a)), and swapping turns every
        appearing bin into a disappearing one — so swap-invariance IS the proof that
        there is no asymmetric weighting of appearing vs disappearing bins."""
        bins = [-np.inf, 0.5, 1.5, np.inf]
        # baseline lives in bins {0,1}, comparison in {1,2}: bin 2 appears, bin 0 disappears.
        a = pd.Series([0] * 60 + [1] * 40)
        b = pd.Series([1] * 40 + [2] * 60)

        psi_ab, _ = calculate_psi(a, b, bins=bins)
        psi_ba, _ = calculate_psi(b, a, bins=bins)

        assert psi_ab == pytest.approx(psi_ba)
        assert psi_ab > 0  # a genuine shift, not a degenerate zero

    def test_psi_matches_textbook_within_epsilon(self):
        """Audit #11: the eps-only-in-log variant equals the textbook eps-in-BOTH-terms
        PSI to O(eps) even with empty bins, so the 0.1/0.25 thresholds stay valid."""
        bins = [-np.inf, 0.5, 1.5, 2.5, np.inf]
        a = pd.Series([0] * 50 + [1] * 50)  # nothing in bins 2,3 -> empty bins
        b = pd.Series([1] * 50 + [3] * 50)

        psi_variant, breakdown = calculate_psi(a, b, bins=bins)

        # Textbook PSI: floor BOTH terms with the same epsilon, then compare.
        eps = 0.0001
        exp = breakdown["baseline_pct"].to_numpy()
        act = breakdown["comparison_pct"].to_numpy()
        exp_s = np.where(exp > 0, exp, eps)
        act_s = np.where(act > 0, act, eps)
        psi_textbook = float(((act_s - exp_s) * np.log(act_s / exp_s)).sum())

        assert psi_variant == pytest.approx(psi_textbook, abs=1e-2)

    def test_breakdown_columns(self):
        np.random.seed(42)
        baseline = pd.Series(np.random.normal(0, 1, 100))
        comparison = pd.Series(np.random.normal(0, 1, 100))
        _, breakdown = calculate_psi(baseline, comparison, bins=5)
        assert "bin" in breakdown.columns
        assert "baseline_pct" in breakdown.columns
        assert "comparison_pct" in breakdown.columns
        assert "psi_component" in breakdown.columns


# =============================================================================
# calculate_psi_for_variable Tests
# =============================================================================


class TestCalculatePsiForVariable:
    def test_basic(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.normal(0, 1, 200)})
        comparison_df = pd.DataFrame({"score": np.random.normal(0, 1, 200)})
        result = calculate_psi_for_variable(baseline_df, comparison_df, "score", bins=5)
        assert isinstance(result, PSIResult)
        assert result.variable == "score"
        assert result.psi_value >= 0

    def test_missing_variable_baseline(self):
        baseline_df = pd.DataFrame({"a": [1]})
        comparison_df = pd.DataFrame({"score": [1]})
        with pytest.raises(ValueError, match="not found in baseline"):
            calculate_psi_for_variable(baseline_df, comparison_df, "score")

    def test_missing_variable_comparison(self):
        baseline_df = pd.DataFrame({"score": [1]})
        comparison_df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found in comparison"):
            calculate_psi_for_variable(baseline_df, comparison_df, "score")


# =============================================================================
# calculate_stability_report Tests
# =============================================================================


class TestCalculateStabilityReport:
    def test_basic_report(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.normal(0, 1, 200), "amount": np.random.exponential(1000, 200)})
        comparison_df = pd.DataFrame({"score": np.random.normal(0, 1, 200), "amount": np.random.exponential(1000, 200)})
        report = calculate_stability_report(baseline_df, comparison_df, ["score", "amount"])
        assert isinstance(report, StabilityReport)
        assert len(report.psi_results) == 2
        assert report.baseline_count == 200
        assert report.comparison_count == 200

    def test_with_score_variable(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.normal(0, 1, 200)})
        comparison_df = pd.DataFrame({"score": np.random.normal(0, 1, 200)})
        report = calculate_stability_report(baseline_df, comparison_df, ["score"], score_variable="score")
        assert report.overall_psi is not None

    def test_skips_missing_variable(self):
        baseline_df = pd.DataFrame({"score": [1, 2, 3]})
        comparison_df = pd.DataFrame({"score": [1, 2, 3]})
        report = calculate_stability_report(baseline_df, comparison_df, ["score", "missing_var"])
        assert len(report.psi_results) == 1

    def test_skips_non_numeric(self):
        baseline_df = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
        comparison_df = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3]})
        report = calculate_stability_report(baseline_df, comparison_df, ["cat", "num"])
        # Only numeric column should be processed
        assert len(report.psi_results) == 1

    def test_custom_bins_per_variable(self):
        np.random.seed(42)
        baseline_df = pd.DataFrame({"score": np.random.normal(0, 1, 100), "amt": np.random.exponential(100, 100)})
        comparison_df = pd.DataFrame({"score": np.random.normal(0, 1, 100), "amt": np.random.exponential(100, 100)})
        report = calculate_stability_report(baseline_df, comparison_df, ["score", "amt"], bins={"score": 5, "amt": 8})
        assert len(report.psi_results) == 2


# =============================================================================
# calculate_csi_for_categorical Tests
# =============================================================================


class TestCalculateCsiForCategorical:
    def test_identical_distributions(self):
        baseline = pd.Series(["A", "A", "B", "B", "C", "C"])
        comparison = pd.Series(["A", "A", "B", "B", "C", "C"])
        csi, breakdown = calculate_csi_for_categorical(baseline, comparison)
        assert csi < 0.01

    def test_different_distributions(self):
        baseline = pd.Series(["A"] * 50 + ["B"] * 50)
        comparison = pd.Series(["A"] * 90 + ["B"] * 10)
        csi, breakdown = calculate_csi_for_categorical(baseline, comparison)
        assert csi > 0.1

    def test_new_category_in_comparison(self):
        baseline = pd.Series(["A", "A", "B", "B"])
        comparison = pd.Series(["A", "B", "C", "C"])
        csi, breakdown = calculate_csi_for_categorical(baseline, comparison)
        assert isinstance(csi, float)
        assert len(breakdown) == 3  # A, B, C

    def test_csi_uses_original_difference_with_epsilon_log(self):
        baseline = pd.Series(["A"] * 100)
        comparison = pd.Series(["B"] * 100)

        csi, _ = calculate_csi_for_categorical(baseline, comparison, min_pct=0.2)

        # Standard CSI: difference uses original proportions,
        # epsilon only protects the log term
        assert csi == pytest.approx(2.0 * np.log(5.0))

    def test_csi_symmetric_under_population_swap(self):
        """Audit #11: CSI shares the PSI formula, so it is likewise swap-invariant —
        appearing and disappearing categories are weighted identically."""
        # category A disappears, C appears between the two populations.
        a = pd.Series(["A"] * 60 + ["B"] * 40)
        b = pd.Series(["B"] * 40 + ["C"] * 60)

        csi_ab, _ = calculate_csi_for_categorical(a, b)
        csi_ba, _ = calculate_csi_for_categorical(b, a)

        assert csi_ab == pytest.approx(csi_ba)
        assert csi_ab > 0

    def test_breakdown_columns(self):
        baseline = pd.Series(["A", "B", "C"])
        comparison = pd.Series(["A", "B", "C"])
        _, breakdown = calculate_csi_for_categorical(baseline, comparison)
        assert "category" in breakdown.columns
        assert "baseline_pct" in breakdown.columns
        assert "comparison_pct" in breakdown.columns
        assert "csi_component" in breakdown.columns
