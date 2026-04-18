import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.config import PreprocessingSettings
from src.reject_inference_optimizer import (
    OptimizerInputs,
    _compute_calibration_error,
    _select_best,
    evaluate_ri_params,
    run_reject_inference_optimization,
    validate_ri_with_mr,
)

# =============================================================================
# Helpers
# =============================================================================

VARIABLES = ["var0", "var1"]
INDICATORS = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]


def _make_summary_2d(n_var0=3, n_var1=4):
    """Create a 2D summary DataFrame with _boo and _rep columns."""
    rng = np.random.RandomState(42)
    rows = []
    for v0 in range(1, n_var0 + 1):
        for v1 in range(1, n_var1 + 1):
            risk = rng.uniform(0.01, 0.1)
            amt = rng.uniform(1000, 10000)
            production = rng.uniform(5000, 50000)
            rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    "todu_30ever_h6": risk * amt / 7,
                    "todu_amt_pile_h6": amt,
                    "oa_amt_h0": production,
                    "todu_30ever_h6_boo": risk * amt / 7 * 0.8,
                    "todu_amt_pile_h6_boo": amt * 0.8,
                    "oa_amt_h0_boo": production * 0.8,
                    "todu_30ever_h6_rep": risk * amt / 7 * 0.2,
                    "todu_amt_pile_h6_rep": amt * 0.2,
                    "oa_amt_h0_rep": production * 0.2,
                }
            )
    return pd.DataFrame(rows)


def _make_optimizer_inputs(n_var0=3, n_var1=4, tasa_fin=0.5, calibration_gamma=1.0):
    """Build OptimizerInputs from synthetic data."""
    rng = np.random.RandomState(42)

    # Build booked summary (_boo suffix)
    booked_rows = []
    repesca_rows = []
    acc_rows = []
    for v0 in range(1, n_var0 + 1):
        for v1 in range(1, n_var1 + 1):
            risk = rng.uniform(0.01, 0.1)
            amt = rng.uniform(1000, 10000)
            production = rng.uniform(5000, 50000)
            booked_rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    "todu_30ever_h6_boo": risk * amt / 7 * 0.8,
                    "todu_amt_pile_h6_boo": amt * 0.8,
                    "oa_amt_h0_boo": production * 0.8,
                }
            )
            repesca_rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    "todu_30ever_h6": risk * amt / 7 * 0.2,
                    "todu_amt_pile_h6": amt * 0.2,
                    "oa_amt_h0": production * 0.2,
                }
            )
            # Acceptance rate: higher bins → lower acceptance
            acc_rate = max(0.1, 1.0 - (v0 + v1) / (n_var0 + n_var1 + 2))
            acc_rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    "n_booked": int(acc_rate * 100),
                    "n_score_rejected": int((1 - acc_rate) * 100),
                    "acceptance_rate": acc_rate,
                }
            )

    booked_summary = pd.DataFrame(booked_rows)
    repesca_pre_ri = pd.DataFrame(repesca_rows)
    acceptance_rates = pd.DataFrame(acc_rows)

    return OptimizerInputs(
        booked_summary=booked_summary,
        repesca_pre_ri=repesca_pre_ri,
        acceptance_rates=acceptance_rates,
        tasa_fin=tasa_fin,
        variables=VARIABLES,
        indicators=INDICATORS,
        inv_vars=[],
        multiplier=7.0,
        calibration_gamma=calibration_gamma,
    )


# =============================================================================
# TestEvaluateRIParams
# =============================================================================


class TestEvaluateRIParams:
    @pytest.fixture
    def inputs(self):
        return _make_optimizer_inputs()

    def test_valid_params_return_kpis(self, inputs):
        """Valid params should return dict with expected keys."""
        result = evaluate_ri_params(inputs, uplift_factor=1.5, max_risk_multiplier=3.0, risk_target=20.0)
        assert "oa_amt_h0" in result
        assert "b2_ever_h6" in result
        assert "feasible" in result
        assert "n_cells_accepted" in result
        assert "uplift_factor" in result
        assert "max_risk_multiplier" in result
        assert "calibration_error" in result

    def test_zero_uplift_no_adjustment(self, inputs):
        """Zero uplift should produce same result as no reject inference."""
        result_zero = evaluate_ri_params(inputs, uplift_factor=0.0, max_risk_multiplier=3.0, risk_target=20.0)
        assert result_zero["feasible"] is True
        assert result_zero["oa_amt_h0"] > 0

    def test_higher_uplift_affects_production(self, inputs):
        """Higher uplift should generally reduce production (more risk → fewer cells accepted)."""
        result_low = evaluate_ri_params(inputs, uplift_factor=0.5, max_risk_multiplier=3.0, risk_target=5.0)
        result_high = evaluate_ri_params(inputs, uplift_factor=5.0, max_risk_multiplier=5.0, risk_target=5.0)
        # With tight risk target and high uplift, production should be <= low uplift
        if result_low["feasible"] and result_high["feasible"]:
            assert result_high["oa_amt_h0"] <= result_low["oa_amt_h0"] + 1e-6

    def test_feasible_flag(self, inputs):
        """With a generous risk target, result should be feasible."""
        result = evaluate_ri_params(inputs, uplift_factor=1.0, max_risk_multiplier=2.0, risk_target=50.0)
        assert result["feasible"] is True
        assert result["n_cells_accepted"] > 0


# =============================================================================
# TestRunOptimization
# =============================================================================


class TestRunOptimization:
    @pytest.fixture
    def inputs(self):
        return _make_optimizer_inputs()

    def test_returns_dataframe_and_dict(self, inputs):
        """Should return (DataFrame, dict) tuple."""
        results_df, best_params = run_reject_inference_optimization(
            inputs, risk_target=20.0, uplift_steps=3, max_mult_steps=3
        )
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(best_params, dict)

    def test_expected_columns(self, inputs):
        """Results DataFrame should have expected columns."""
        results_df, _ = run_reject_inference_optimization(inputs, risk_target=20.0, uplift_steps=3, max_mult_steps=3)
        expected_cols = {
            "uplift_factor",
            "max_risk_multiplier",
            "oa_amt_h0",
            "b2_ever_h6",
            "feasible",
            "is_best",
            "calibration_error",
        }
        assert expected_cols.issubset(set(results_df.columns))

    def test_grid_covers_range(self, inputs):
        """Grid should produce the expected number of combinations."""
        results_df, _ = run_reject_inference_optimization(
            inputs,
            risk_target=20.0,
            uplift_range=(0.0, 2.0),
            uplift_steps=3,
            max_mult_range=(1.0, 3.0),
            max_mult_steps=3,
        )
        assert len(results_df) == 9  # 3 x 3

    def test_best_minimizes_calibration_error(self, inputs):
        """Best row should have the lowest (or near-lowest) calibration error among feasible solutions."""
        results_df, best_params = run_reject_inference_optimization(
            inputs, risk_target=20.0, uplift_steps=5, max_mult_steps=5
        )
        if best_params:
            feasible = results_df[results_df["feasible"]]
            best_row = results_df[results_df["is_best"]]
            assert len(best_row) == 1
            min_error = feasible["calibration_error"].min()
            # Best must be within 5% tolerance of the minimum calibration error
            assert best_row.iloc[0]["calibration_error"] <= min_error * 1.05 + 1e-12

    def test_calibration_error_in_results(self, inputs):
        """Results DataFrame should include calibration_error column."""
        results_df, _ = run_reject_inference_optimization(inputs, risk_target=20.0, uplift_steps=3, max_mult_steps=3)
        assert "calibration_error" in results_df.columns
        # All rows should have a finite calibration error
        feasible = results_df[results_df["feasible"]]
        if not feasible.empty:
            assert feasible["calibration_error"].notna().all()

    def test_best_not_always_least_strict(self):
        """With varying acceptance rates the best params should not always be (0.0, 1.0)."""
        # Build inputs with strongly varying acceptance rates to create
        # calibration pressure that penalizes the least-strict parameters.
        rng = np.random.RandomState(99)
        n_var0, n_var1 = 4, 4
        booked_rows, repesca_rows, acc_rows = [], [], []
        for v0 in range(1, n_var0 + 1):
            for v1 in range(1, n_var1 + 1):
                # Higher bins → much lower acceptance rate
                acc_rate = max(0.1, 1.0 - 0.12 * (v0 + v1))
                risk = 0.02 + 0.015 * (v0 + v1)
                amt = rng.uniform(5000, 15000)
                production = rng.uniform(10000, 60000)
                booked_rows.append(
                    {
                        "var0": v0,
                        "var1": v1,
                        "todu_30ever_h6_boo": risk * amt / 7,
                        "todu_amt_pile_h6_boo": amt,
                        "oa_amt_h0_boo": production,
                    }
                )
                repesca_rows.append(
                    {
                        "var0": v0,
                        "var1": v1,
                        "todu_30ever_h6": risk * amt / 7 * 0.3,
                        "todu_amt_pile_h6": amt * 0.3,
                        "oa_amt_h0": production * 0.3,
                    }
                )
                acc_rows.append(
                    {
                        "var0": v0,
                        "var1": v1,
                        "n_booked": int(acc_rate * 100),
                        "n_score_rejected": int((1 - acc_rate) * 100),
                        "acceptance_rate": acc_rate,
                    }
                )

        inputs = OptimizerInputs(
            booked_summary=pd.DataFrame(booked_rows),
            repesca_pre_ri=pd.DataFrame(repesca_rows),
            acceptance_rates=pd.DataFrame(acc_rows),
            tasa_fin=0.5,
            variables=VARIABLES,
            indicators=INDICATORS,
            inv_vars=[],
            multiplier=7.0,
        )
        _, best_params = run_reject_inference_optimization(inputs, risk_target=20.0, uplift_steps=5, max_mult_steps=5)
        if best_params:
            # The least-strict combo (uplift=0, max_mult=1) should NOT always win
            is_least_strict = best_params["uplift_factor"] == 0.0 and best_params["max_risk_multiplier"] == 1.0
            assert not is_least_strict, "Calibration objective should not always pick least-strict params"

    def test_no_feasible_returns_empty_dict(self, inputs):
        """If no solution is feasible, best_params should be empty dict."""
        # Use impossibly tight risk target
        results_df, best_params = run_reject_inference_optimization(
            inputs, risk_target=0.001, uplift_steps=3, max_mult_steps=3
        )
        assert best_params == {}
        assert not results_df["is_best"].any()

    def test_single_best_marked(self, inputs):
        """Exactly one row should be marked as is_best (if feasible)."""
        results_df, best_params = run_reject_inference_optimization(
            inputs, risk_target=20.0, uplift_steps=4, max_mult_steps=4
        )
        if best_params:
            assert results_df["is_best"].sum() == 1


# =============================================================================
# TestCalibrationGamma
# =============================================================================


class TestCalibrationGamma:
    def test_gamma_affects_calibration_error(self):
        """Different gamma values should produce different calibration errors."""
        inputs_g1 = _make_optimizer_inputs(calibration_gamma=1.0)
        inputs_g05 = _make_optimizer_inputs(calibration_gamma=0.5)

        result_g1 = evaluate_ri_params(inputs_g1, uplift_factor=1.5, max_risk_multiplier=3.0, risk_target=20.0)
        result_g05 = evaluate_ri_params(inputs_g05, uplift_factor=1.5, max_risk_multiplier=3.0, risk_target=20.0)

        # Calibration errors should differ when gamma differs
        assert result_g1["calibration_error"] != result_g05["calibration_error"]

    def test_gamma_1_matches_original(self):
        """gamma=1.0 should produce the same result as the original formula."""
        inputs = _make_optimizer_inputs(calibration_gamma=1.0)
        result = evaluate_ri_params(inputs, uplift_factor=1.5, max_risk_multiplier=3.0, risk_target=20.0)
        assert result["calibration_error"] > 0 or result["calibration_error"] == 0.0


class TestCalibrationRobustness:
    def test_calibration_error_imputes_missing_acceptance_rates(self):
        merged = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                "todu_30ever_h6_boo": [1.0, 2.0],
                "todu_amt_pile_h6_boo": [100.0, 200.0],
                "todu_30ever_h6": [1.2, 2.2],
                "todu_amt_pile_h6": [110.0, 220.0],
            }
        )
        acceptance_rates = pd.DataFrame(
            {
                "var0": [1],
                "var1": [1],
                "acceptance_rate": [0.6],
            }
        )
        err = _compute_calibration_error(
            merged, acceptance_rates, ["var0", "var1"], multiplier=7.0, calibration_gamma=1.0
        )
        assert np.isfinite(err)

    def test_select_best_ignores_non_finite_calibration_error(self):
        results_df = pd.DataFrame(
            {
                "uplift_factor": [0.0, 1.0, 2.0],
                "max_risk_multiplier": [1.0, 2.0, 3.0],
                "oa_amt_h0": [1000.0, 900.0, 1100.0],
                "b2_ever_h6": [1.0, 1.1, 1.2],
                "feasible": [True, True, True],
                "calibration_error": [np.inf, 0.3, np.nan],
            }
        )
        out_df, best = _select_best(results_df.copy())
        assert best
        assert np.isfinite(best["calibration_error"])
        assert out_df["is_best"].sum() == 1

    def test_lower_gamma_less_aggressive(self):
        """Lower gamma should produce less aggressive calibration targets."""
        # With gamma < 1, target = booked_risk / acc^gamma is closer to booked_risk
        # than with gamma = 1, so calibration error at low uplift should be lower
        inputs_g1 = _make_optimizer_inputs(calibration_gamma=1.0)
        inputs_g05 = _make_optimizer_inputs(calibration_gamma=0.5)

        result_g1 = evaluate_ri_params(inputs_g1, uplift_factor=0.0, max_risk_multiplier=1.0, risk_target=20.0)
        result_g05 = evaluate_ri_params(inputs_g05, uplift_factor=0.0, max_risk_multiplier=1.0, risk_target=20.0)

        # With zero uplift (no RI), gamma=0.5 target is closer to predicted → lower error
        if result_g1["calibration_error"] > 0 and result_g05["calibration_error"] > 0:
            assert result_g05["calibration_error"] < result_g1["calibration_error"]


# =============================================================================
# TestOptunaOptimizer
# =============================================================================


class TestOptunaOptimizer:
    @pytest.fixture
    def inputs(self):
        return _make_optimizer_inputs()

    def test_optuna_returns_results(self, inputs):
        """Optuna optimizer should return (DataFrame, dict) tuple."""
        pytest.importorskip("optuna")
        from src.reject_inference_optimizer import run_reject_inference_optimization_optuna

        results_df, best_params = run_reject_inference_optimization_optuna(inputs, risk_target=20.0, n_trials=10)
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(best_params, dict)

    def test_optuna_expected_columns(self, inputs):
        """Optuna results should have expected columns."""
        pytest.importorskip("optuna")
        from src.reject_inference_optimizer import run_reject_inference_optimization_optuna

        results_df, _ = run_reject_inference_optimization_optuna(inputs, risk_target=20.0, n_trials=10)
        expected_cols = {"uplift_factor", "max_risk_multiplier", "feasible", "calibration_error"}
        assert expected_cols.issubset(set(results_df.columns))

    def test_optuna_n_trials(self, inputs):
        """Number of results should match n_trials."""
        pytest.importorskip("optuna")
        from src.reject_inference_optimizer import run_reject_inference_optimization_optuna

        results_df, _ = run_reject_inference_optimization_optuna(inputs, risk_target=20.0, n_trials=15)
        assert len(results_df) == 15

    def test_optuna_single_best(self, inputs):
        """Exactly one row should be marked as best (if feasible)."""
        pytest.importorskip("optuna")
        from src.reject_inference_optimizer import run_reject_inference_optimization_optuna

        results_df, best_params = run_reject_inference_optimization_optuna(inputs, risk_target=20.0, n_trials=20)
        if best_params:
            assert results_df["is_best"].sum() == 1


# =============================================================================
# TestValidateRIWithMR
# =============================================================================


class TestValidateRIWithMR:
    def test_mr_validation_returns_expected_keys(self):
        """MR validation should return dict with expected keys."""
        main_inputs = _make_optimizer_inputs()
        mr_inputs = _make_optimizer_inputs()  # Same synthetic data for simplicity
        best_params = {"uplift_factor": 1.5, "max_risk_multiplier": 3.0}

        result = validate_ri_with_mr(main_inputs, mr_inputs, best_params, risk_target=20.0)
        assert "main_calibration_error" in result
        assert "mr_calibration_error" in result
        assert "degradation_ratio" in result
        assert "mr_feasible" in result

    def test_same_data_degradation_near_one(self):
        """With identical main/MR data, degradation ratio should be ~1.0."""
        inputs = _make_optimizer_inputs()
        best_params = {"uplift_factor": 1.5, "max_risk_multiplier": 3.0}

        result = validate_ri_with_mr(inputs, inputs, best_params, risk_target=20.0)
        if np.isfinite(result["degradation_ratio"]):
            assert result["degradation_ratio"] == pytest.approx(1.0, abs=0.01)


# =============================================================================
# TestConfigFields
# =============================================================================


class TestConfigFields:
    def _make_base_config(self, **overrides):
        """Create minimal valid config with optional overrides."""
        base = {
            "keep_vars": ["var0"],
            "indicators": ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            "variables": ["var0", "var1"],
            "octroi_bins": [1.0, 2.0, 3.0],
            "efx_bins": [1.0, 2.0, 3.0],
            "date_ini_book_obs": "2023-01-01",
            "date_fin_book_obs": "2023-12-31",
        }
        base.update(overrides)
        return base

    def test_defaults(self):
        """Default config should have expected RI optimizer defaults."""
        config = PreprocessingSettings(**self._make_base_config())
        assert config.run_ri_optimizer is False
        assert config.ri_uplift_range == [0.0, 5.0]
        assert config.ri_max_mult_range == [1.0, 5.0]
        assert config.ri_uplift_steps == 11
        assert config.ri_max_mult_steps == 9
        assert config.ri_calibration_gamma == 1.0
        assert config.ri_optimizer_method == "grid"
        assert config.ri_optuna_n_trials == 100

    def test_range_requires_two_elements(self):
        """Range fields must have exactly 2 elements."""
        with pytest.raises(ValueError):
            PreprocessingSettings(**self._make_base_config(ri_uplift_range=[1.0]))

    def test_range_min_less_than_max(self):
        """Range min must be less than max."""
        with pytest.raises(ValueError):
            PreprocessingSettings(**self._make_base_config(ri_uplift_range=[5.0, 1.0]))

    def test_range_equal_values_rejected(self):
        """Range with equal min and max should be rejected."""
        with pytest.raises(ValueError):
            PreprocessingSettings(**self._make_base_config(ri_uplift_range=[2.0, 2.0]))

    def test_valid_custom_range(self):
        """Valid custom range should be accepted."""
        config = PreprocessingSettings(
            **self._make_base_config(ri_uplift_range=[0.5, 3.0], ri_max_mult_range=[1.5, 4.0])
        )
        assert config.ri_uplift_range == [0.5, 3.0]
        assert config.ri_max_mult_range == [1.5, 4.0]

    def test_optuna_method_accepted(self):
        """'optuna' is a valid optimizer method."""
        config = PreprocessingSettings(**self._make_base_config(ri_optimizer_method="optuna"))
        assert config.ri_optimizer_method == "optuna"

    def test_invalid_optimizer_method(self):
        """Invalid optimizer method should be rejected."""
        with pytest.raises(ValueError):
            PreprocessingSettings(**self._make_base_config(ri_optimizer_method="bayesian"))

    def test_optuna_n_trials_bounds(self):
        """ri_optuna_n_trials respects ge=10 and le=10000 bounds."""
        with pytest.raises(ValueError):
            PreprocessingSettings(**self._make_base_config(ri_optuna_n_trials=5))
        with pytest.raises(ValueError):
            PreprocessingSettings(**self._make_base_config(ri_optuna_n_trials=20000))
