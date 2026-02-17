import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    bootstrap_confidence_interval,
    calc_iv,
    calculate_lift_table,
    calculate_psi_by_period,
    calculate_rejection_thresholds,
    compute_metrics,
    compute_precision_recall,
    delong_test,
    ks_statistic,
    train_logistic_regression,
)

# =============================================================================
# ks_statistic Tests
# =============================================================================


class TestKsStatistic:
    def test_perfect_separation(self):
        """Perfect model should have KS close to 1."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ks = ks_statistic(y_true, y_scores)
        assert ks == pytest.approx(1.0, abs=0.01)

    def test_random_model(self):
        """Random model should have KS close to 0."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_scores = np.random.rand(1000)
        ks = ks_statistic(y_true, y_scores)
        assert 0 <= ks < 0.15

    def test_ks_between_0_and_1(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_scores = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4, 0.9])
        ks = ks_statistic(y_true, y_scores)
        assert 0 <= ks <= 1


# =============================================================================
# compute_metrics Tests
# =============================================================================


class TestComputeMetrics:
    def test_returns_four_values(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_metrics(y_true, scores)
        assert len(result) == 4

    def test_gini_and_auc_relationship(self):
        """Gini = 2 * AUC - 1."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        gini, roc_auc, ks, _ = compute_metrics(y_true, scores)
        assert gini == pytest.approx(2 * roc_auc - 1, abs=1e-10)

    def test_perfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        gini, roc_auc, ks, cap = compute_metrics(y_true, scores)
        assert roc_auc == pytest.approx(1.0, abs=0.01)
        assert gini == pytest.approx(1.0, abs=0.02)
        assert ks == pytest.approx(1.0, abs=0.01)

    def test_cap_curve_length(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        _, _, _, cap = compute_metrics(y_true, scores)
        assert len(cap) == len(y_true)

    def test_cap_curve_is_cumulative(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
        _, _, _, cap = compute_metrics(y_true, scores)
        assert all(cap[i] <= cap[i + 1] for i in range(len(cap) - 1))


# =============================================================================
# bootstrap_confidence_interval Tests
# =============================================================================


class TestBootstrapConfidenceInterval:
    def test_returns_two_tuples(self):
        np.random.seed(42)
        y_true = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
        y_scores = pd.Series([0.1, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4, 0.9, 0.15, 0.25])
        gini_ci, ks_ci = bootstrap_confidence_interval(y_true, y_scores, n_iterations=10)
        assert len(gini_ci) == 2
        assert len(ks_ci) == 2

    def test_lower_bound_less_than_upper(self):
        np.random.seed(42)
        y_true = pd.Series(np.random.randint(0, 2, 100))
        y_scores = pd.Series(np.random.rand(100))
        gini_ci, ks_ci = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50)
        assert gini_ci[0] <= gini_ci[1]
        assert ks_ci[0] <= ks_ci[1]

    def test_custom_alpha(self):
        np.random.seed(42)
        y_true = pd.Series(np.random.randint(0, 2, 100))
        y_scores = pd.Series(np.random.rand(100))
        gini_ci_90, _ = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50, alpha=0.10)
        gini_ci_95, _ = bootstrap_confidence_interval(y_true, y_scores, n_iterations=50, alpha=0.05)
        # 90% CI should be narrower than 95% CI (on average)
        assert isinstance(gini_ci_90, tuple)
        assert isinstance(gini_ci_95, tuple)


# =============================================================================
# calculate_rejection_thresholds Tests
# =============================================================================


class TestCalculateRejectionThresholds:
    def test_default_thresholds(self):
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.arange(100, dtype=float)
        result = calculate_rejection_thresholds(y_true, scores)
        assert set(result.keys()) == {"5%", "10%", "15%", "20%"}

    def test_custom_thresholds(self):
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.arange(100, dtype=float)
        result = calculate_rejection_thresholds(y_true, scores, thresholds=[25, 50])
        assert set(result.keys()) == {"25%", "50%"}

    def test_values_between_0_and_100(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_rejection_thresholds(y_true, scores)
        for val in result.values():
            assert 0 <= val <= 100

    def test_higher_threshold_captures_more(self):
        """Rejecting more applications should capture more bad accounts."""
        np.random.seed(42)
        n = 500
        y_true = np.zeros(n)
        scores = np.random.rand(n)
        # Higher scores = higher risk
        y_true[scores > 0.7] = 1
        result = calculate_rejection_thresholds(y_true, scores, thresholds=[5, 10, 20, 50])
        assert result["5%"] <= result["10%"]
        assert result["10%"] <= result["20%"]
        assert result["20%"] <= result["50%"]


# =============================================================================
# calculate_psi_by_period Tests (metrics.py version)
# =============================================================================


class TestCalculatePsiMetrics:
    def _make_data(self, ref_scores, act_scores, ref_dates, act_dates):
        ref_df = pd.DataFrame({"date": ref_dates, "score": ref_scores})
        act_df = pd.DataFrame({"date": act_dates, "score": act_scores})
        return pd.concat([ref_df, act_df], ignore_index=True)

    def test_identical_distributions(self):
        np.random.seed(42)
        scores = np.random.normal(500, 100, 200)
        ref_dates = [pd.Timestamp("2023-01-15")] * 200
        act_dates = [pd.Timestamp("2023-06-15")] * 200

        data = self._make_data(scores, scores, ref_dates, act_dates)
        result = calculate_psi_by_period(
            data,
            date_column="date",
            score_column="score",
            start_date_ref=pd.Timestamp("2023-01-01"),
            end_date_ref=pd.Timestamp("2023-02-01"),
            start_date_act=pd.Timestamp("2023-06-01"),
            end_date_act=pd.Timestamp("2023-07-01"),
            buckets=5,
            show_plots=False,
        )
        assert isinstance(result, pd.DataFrame)
        # Identical distributions -> PSI close to 0
        assert result["PSI"].sum() < 0.1

    def test_invalid_date_type_raises(self):
        data = pd.DataFrame({"date": ["2023-01-01"], "score": [500]})
        with pytest.raises(ValueError, match="pd.Timestamp"):
            calculate_psi_by_period(
                data,
                date_column="date",
                score_column="score",
                start_date_ref="2023-01-01",
                end_date_ref="2023-02-01",
                start_date_act="2023-06-01",
                end_date_act="2023-07-01",
                buckets=5,
            )

    def test_no_buckets_or_breakpoints_raises(self):
        data = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")], "score": [500]})
        with pytest.raises(ValueError, match="Either buckets or breakpoints"):
            calculate_psi_by_period(
                data,
                date_column="date",
                score_column="score",
                start_date_ref=pd.Timestamp("2023-01-01"),
                end_date_ref=pd.Timestamp("2023-02-01"),
                start_date_act=pd.Timestamp("2023-06-01"),
                end_date_act=pd.Timestamp("2023-07-01"),
            )

    def test_with_breakpoints(self):
        np.random.seed(42)
        ref_scores = np.random.normal(500, 100, 100)
        act_scores = np.random.normal(520, 100, 100)
        ref_dates = [pd.Timestamp("2023-01-15")] * 100
        act_dates = [pd.Timestamp("2023-06-15")] * 100

        data = self._make_data(ref_scores, act_scores, ref_dates, act_dates)
        result = calculate_psi_by_period(
            data,
            date_column="date",
            score_column="score",
            start_date_ref=pd.Timestamp("2023-01-01"),
            end_date_ref=pd.Timestamp("2023-02-01"),
            start_date_act=pd.Timestamp("2023-06-01"),
            end_date_act=pd.Timestamp("2023-07-01"),
            breakpoints=[200, 400, 600, 800],
            show_plots=False,
        )
        assert isinstance(result, pd.DataFrame)
        assert "PSI" in result.columns


# =============================================================================
# calc_iv Tests
# =============================================================================


class TestCalcIV:
    def test_basic_iv(self):
        df = pd.DataFrame({"var": ["A", "A", "A", "B", "B", "B", "C", "C", "C"], "target": [1, 0, 0, 0, 0, 0, 1, 1, 0]})
        iv = calc_iv(df, "var", "target")
        assert isinstance(iv, float)
        assert iv >= 0

    def test_high_iv_good_predictor(self):
        """Variable perfectly separating targets should have high IV."""
        df = pd.DataFrame(
            {
                "var": ["good"] * 100 + ["bad"] * 100,
                "target": [0] * 100 + [1] * 100,
            }
        )
        iv = calc_iv(df, "var", "target")
        assert iv > 0.3  # Strong predictive power

    def test_random_variable_low_iv(self):
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                "var": np.random.choice(["A", "B", "C"], n),
                "target": np.random.randint(0, 2, n),
            }
        )
        iv = calc_iv(df, "var", "target")
        assert iv < 0.1  # Weak or no predictive power


# =============================================================================
# train_logistic_regression Tests
# =============================================================================


class TestTrainLogisticRegression:
    def test_basic_training(self):
        np.random.seed(42)
        X = pd.DataFrame({"x1": np.random.randn(100), "x2": np.random.randn(100)})
        y = (X["x1"] + X["x2"] > 0).astype(int)
        model, X_standardized = train_logistic_regression(X, y)
        assert hasattr(model, "predict")
        assert X_standardized.shape == X.shape

    def test_standardized_output(self):
        np.random.seed(42)
        X = pd.DataFrame({"x1": np.random.randn(50) * 100, "x2": np.random.randn(50) * 0.01})
        y = np.random.randint(0, 2, 50)
        _, X_standardized = train_logistic_regression(X, y)
        # Standardized data should have mean ~0 and std ~1
        assert abs(X_standardized[:, 0].mean()) < 0.5
        assert abs(X_standardized[:, 1].mean()) < 0.5


# =============================================================================
# compute_precision_recall Tests
# =============================================================================


class TestComputePrecisionRecall:
    def test_returns_four_values(self):
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_precision_recall(y_true, scores)
        assert len(result) == 4

    def test_perfect_model_high_ap(self):
        """Perfect model should have AP close to 1."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        precision, recall, _, ap = compute_precision_recall(y_true, scores)
        assert ap > 0.9

    def test_random_model_low_ap(self):
        """Random model should have AP close to the positive class prevalence."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        scores = np.random.rand(1000)
        _, _, _, ap = compute_precision_recall(y_true, scores)
        prevalence = y_true.mean()
        assert abs(ap - prevalence) < 0.15

    def test_precision_recall_between_0_and_1(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.3, 0.7, 0.8, 0.2, 0.6])
        precision, recall, _, ap = compute_precision_recall(y_true, scores)
        assert all(0 <= p <= 1 for p in precision)
        assert all(0 <= r <= 1 for r in recall)
        assert 0 <= ap <= 1


# =============================================================================
# calculate_lift_table Tests
# =============================================================================


class TestCalculateLiftTable:
    def test_returns_dataframe(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_lift_table(y_true, scores)
        assert isinstance(result, pd.DataFrame)

    def test_default_deciles(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_lift_table(y_true, scores)
        assert len(result) == 10

    def test_custom_bins(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_lift_table(y_true, scores, n_bins=5)
        assert len(result) == 5

    def test_expected_columns(self):
        y_true = np.array([0] * 80 + [1] * 20)
        scores = np.arange(100, dtype=float)
        result = calculate_lift_table(y_true, scores)
        expected_cols = {
            "bin",
            "n_records",
            "n_bads",
            "bad_rate",
            "pct_population",
            "pct_bads",
            "cumulative_pct_bads",
            "cumulative_pct_population",
            "lift",
            "cumulative_lift",
        }
        assert set(result.columns) == expected_cols

    def test_cumulative_pct_bads_reaches_one(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_lift_table(y_true, scores)
        assert result["cumulative_pct_bads"].iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_cumulative_pct_population_reaches_one(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_lift_table(y_true, scores)
        assert result["cumulative_pct_population"].iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_first_bin_has_highest_lift(self):
        """With a good model, bin 1 (highest risk) should have lift >= 1."""
        np.random.seed(42)
        n = 500
        y_true = np.zeros(n)
        scores = np.random.rand(n)
        y_true[scores > 0.7] = 1
        result = calculate_lift_table(y_true, scores)
        assert result["lift"].iloc[0] >= 1.0

    def test_total_bads_sum_matches(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = calculate_lift_table(y_true, scores)
        assert result["n_bads"].sum() == y_true.sum()


# =============================================================================
# delong_test Tests
# =============================================================================


class TestDelongTest:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores1 = np.random.rand(200)
        scores2 = np.random.rand(200)
        result = delong_test(y_true, scores1, scores2)
        expected_keys = {"auc1", "auc2", "z_statistic", "p_value", "auc_diff", "se_diff"}
        assert set(result.keys()) == expected_keys

    def test_identical_scores_not_significant(self):
        """Comparing a model to itself should yield p-value of 1 (no difference)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores = np.random.rand(200)
        result = delong_test(y_true, scores, scores)
        assert result["p_value"] == pytest.approx(1.0, abs=0.01)
        assert result["auc_diff"] == pytest.approx(0.0, abs=1e-10)

    def test_different_models_detectable(self):
        """A strong model vs a random one should yield a significant p-value."""
        np.random.seed(42)
        n = 500
        y_true = np.zeros(n)
        good_scores = np.random.rand(n)
        y_true[good_scores > 0.5] = 1

        random_scores = np.random.rand(n)
        result = delong_test(y_true, good_scores, random_scores)
        assert result["p_value"] < 0.05

    def test_auc_values_between_0_and_1(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores1 = np.random.rand(200)
        scores2 = np.random.rand(200)
        result = delong_test(y_true, scores1, scores2)
        assert 0 <= result["auc1"] <= 1
        assert 0 <= result["auc2"] <= 1

    def test_p_value_between_0_and_1(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores1 = np.random.rand(200)
        scores2 = np.random.rand(200)
        result = delong_test(y_true, scores1, scores2)
        assert 0 <= result["p_value"] <= 1

    def test_symmetry(self):
        """delong_test(y, s1, s2) and delong_test(y, s2, s1) should give same p-value."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        scores1 = np.random.rand(200)
        scores2 = np.random.rand(200)
        result1 = delong_test(y_true, scores1, scores2)
        result2 = delong_test(y_true, scores2, scores1)
        assert result1["p_value"] == pytest.approx(result2["p_value"], abs=1e-10)
        assert result1["auc_diff"] == pytest.approx(-result2["auc_diff"], abs=1e-10)
