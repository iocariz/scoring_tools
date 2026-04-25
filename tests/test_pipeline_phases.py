"""Coverage tests for the small phase wrappers extracted in R2b-iv (#63).

These modules are thin orchestration glue — we verify the gating semantics
(run-or-skip), the happy path invokes the right downstream helpers, and
failures are swallowed as "non-blocking". Heavy dependencies are mocked.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import src.pipeline.config_loader as config_loader_module
import src.pipeline.ri_optimizer as ri_optimizer_module
import src.pipeline.scenarios as scenarios_module
import src.pipeline.sensitivity as sensitivity_module
from src.config import OutputPaths, PreprocessingSettings


def make_settings(**overrides):
    values = {
        "keep_vars": ["mis_date"],
        "indicators": ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        "segment_filter": "seg_a",
        "date_ini_book_obs": "2024-01-01",
        "date_fin_book_obs": "2024-12-31",
        "variables": ["sc_octroi_new_clus", "new_efx_clus"],
        "octroi_bins": [0.0, 1.0, 2.0],
        "efx_bins": [0.0, 1.0, 2.0],
    }
    values.update(overrides)
    return PreprocessingSettings(**values)


# ---------------------------------------------------------------------------
# load_and_validate_config
# ---------------------------------------------------------------------------


class TestLoadAndValidateConfig:
    def test_loads_and_returns_tuple(self, tmp_path):
        config_toml = tmp_path / "config.toml"
        config_toml.write_text(
            "[preprocessing]\n"
            'data_path = "ignored.sas7bdat"\n'
            'keep_vars = ["mis_date"]\n'
            'indicators = ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]\n'
            'segment_filter = "unit"\n'
            'date_ini_book_obs = "2024-01-01"\n'
            'date_fin_book_obs = "2024-06-30"\n'
            'variables = ["sc_octroi_new_clus", "new_efx_clus"]\n'
            "octroi_bins = [0.0, 1.0]\n"
            "efx_bins = [0.0, 1.0]\n"
        )

        settings, date_ini, date_fin, annual_coef = config_loader_module.load_and_validate_config(str(config_toml))

        assert isinstance(settings, PreprocessingSettings)
        assert settings.segment_filter == "unit"
        assert date_ini == pd.Timestamp("2024-01-01")
        assert date_fin == pd.Timestamp("2024-06-30")
        # 6 months Jan–Jun → 12/6 = 2.0
        assert annual_coef == pytest.approx(2.0)

    def test_propagates_validation_error(self, tmp_path):
        config_toml = tmp_path / "bad.toml"
        config_toml.write_text('[preprocessing]\nvariables = ["only_one"]\n')

        with pytest.raises(Exception):  # noqa: B017,PT011 — Pydantic ValidationError class is internal
            config_loader_module.load_and_validate_config(str(config_toml))


# ---------------------------------------------------------------------------
# run_sensitivity_phase
# ---------------------------------------------------------------------------


class TestRunSensitivityPhase:
    def test_returns_early_when_disabled(self, tmp_path):
        settings = make_settings(run_sensitivity=False)
        output = OutputPaths(base_dir=tmp_path)

        result = sensitivity_module.run_sensitivity_phase(pd.DataFrame(), pd.DataFrame(), settings, output=output)

        assert result is None
        # No files written
        assert list(tmp_path.rglob("sensitivity*.csv")) == []

    def test_happy_path_writes_outputs(self, monkeypatch, tmp_path):
        settings = make_settings(run_sensitivity=True, sensitivity_levels=[0.1])
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        desagregado = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2],
                "new_efx_clus": [1, 2],
                "oa_amt": [100.0, 200.0],
                "oa_amt_h0": [100.0, 200.0],
                "todu_30ever_h6": [1.0, 2.0],
                "todu_amt_pile_h6": [10.0, 20.0],
            }
        )

        fake_grid = MagicMock()
        fake_grid.cell_data = np.zeros((2, 4))

        monkeypatch.setattr(
            "src.optimization_utils.CellGrid.from_summary",
            classmethod(lambda cls, df, variables: fake_grid),
        )
        monkeypatch.setattr(
            "src.optimization_utils.milp_solve_cutoffs",
            lambda *a, **kw: np.array([1, 0]),
        )
        monkeypatch.setattr(
            "src.sensitivity.run_sensitivity_analysis",
            lambda *a, **kw: pd.DataFrame({"level": [0.1], "swap_in": [42.0]}),
        )
        monkeypatch.setattr(
            "src.sensitivity.sensitivity_cell_detail",
            lambda *a, **kw: pd.DataFrame({"cell": [0], "impact": [1.0]}),
        )
        monkeypatch.setattr(
            "src.sensitivity.compute_cell_marginal_impact",
            lambda *a, **kw: pd.DataFrame({"cell": [0], "marginal": [0.5]}),
        )

        sensitivity_module.run_sensitivity_phase(desagregado, pd.DataFrame(), settings, output=output)

        assert pd.read_csv(output.sensitivity_analysis_csv("_base"))["swap_in"].iloc[0] == 42.0
        assert pd.read_csv(output.sensitivity_analysis_csv("_cell_detail"))["impact"].iloc[0] == 1.0
        assert pd.read_csv(output.cell_marginal_impact_csv("_base"))["marginal"].iloc[0] == 0.5

    def test_infeasible_baseline_logs_and_skips(self, monkeypatch, tmp_path):
        settings = make_settings(run_sensitivity=True)
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        fake_grid = MagicMock()
        fake_grid.cell_data = np.zeros((2, 4))
        monkeypatch.setattr(
            "src.optimization_utils.CellGrid.from_summary",
            classmethod(lambda cls, df, variables: fake_grid),
        )
        monkeypatch.setattr(
            "src.optimization_utils.milp_solve_cutoffs",
            lambda *a, **kw: None,
        )

        sensitivity_module.run_sensitivity_phase(pd.DataFrame(), pd.DataFrame(), settings, output=output)

        assert list(tmp_path.rglob("sensitivity*.csv")) == []

    def test_exception_is_non_blocking(self, monkeypatch, tmp_path):
        settings = make_settings(run_sensitivity=True)
        output = OutputPaths(base_dir=tmp_path)

        monkeypatch.setattr(
            "src.optimization_utils.CellGrid.from_summary",
            classmethod(lambda cls, df, variables: (_ for _ in ()).throw(RuntimeError("boom"))),
        )
        # Should not raise
        sensitivity_module.run_sensitivity_phase(pd.DataFrame(), pd.DataFrame(), settings, output=output)


# ---------------------------------------------------------------------------
# run_ri_optimizer_phase
# ---------------------------------------------------------------------------


class TestRunRiOptimizerPhase:
    def test_returns_none_when_disabled(self, tmp_path):
        settings = make_settings(run_ri_optimizer=False)
        result = ri_optimizer_module.run_ri_optimizer_phase(
            data_booked=pd.DataFrame(),
            data_demand=pd.DataFrame(),
            risk_inference={},
            reg_todu_amt_pile=None,
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=OutputPaths(base_dir=tmp_path),
        )
        assert result is None

    def test_returns_none_when_inference_method_is_none(self, tmp_path):
        settings = make_settings(run_ri_optimizer=True, reject_inference_method="none")
        result = ri_optimizer_module.run_ri_optimizer_phase(
            data_booked=pd.DataFrame(),
            data_demand=pd.DataFrame(),
            risk_inference={},
            reg_todu_amt_pile=None,
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=OutputPaths(base_dir=tmp_path),
        )
        assert result is None

    def test_happy_path_writes_csv_and_returns_params(self, monkeypatch, tmp_path):
        settings = make_settings(
            run_ri_optimizer=True,
            reject_inference_method="parceling",
            ri_validation_split=1.0,
            ri_optimizer_method="grid",
        )
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        data_booked = pd.DataFrame({"mis_date": pd.to_datetime(["2024-01-01"])})
        data_demand = pd.DataFrame({"mis_date": pd.to_datetime(["2024-01-01"])})

        monkeypatch.setattr(
            "src.inference_optimized.compute_pre_reject_inference_data",
            lambda **kw: (pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]})),
        )
        monkeypatch.setattr(
            "src.reject_inference.compute_acceptance_rates",
            lambda *a, **kw: pd.DataFrame({"rate": [0.5]}),
        )

        best = {"uplift_factor": 1.2, "max_risk_multiplier": 1.4, "error": 0.01}
        results_df = pd.DataFrame({"uplift_factor": [1.2], "max_risk_multiplier": [1.4], "is_best": [True]})
        monkeypatch.setattr(
            "src.reject_inference_optimizer.run_reject_inference_optimization",
            lambda *a, **kw: (results_df, best),
        )

        result = ri_optimizer_module.run_ri_optimizer_phase(
            data_booked=data_booked,
            data_demand=data_demand,
            risk_inference={},
            reg_todu_amt_pile=None,
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=output,
        )

        assert result == best
        csv_rows = pd.read_csv(output.ri_optimizer_csv())
        assert csv_rows["uplift_factor"].iloc[0] == pytest.approx(1.2)

    def test_exception_is_non_blocking(self, monkeypatch, tmp_path):
        settings = make_settings(run_ri_optimizer=True, reject_inference_method="parceling")
        monkeypatch.setattr(
            "src.inference_optimized.compute_pre_reject_inference_data",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = ri_optimizer_module.run_ri_optimizer_phase(
            data_booked=pd.DataFrame(),
            data_demand=pd.DataFrame(),
            risk_inference={},
            reg_todu_amt_pile=None,
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=OutputPaths(base_dir=tmp_path),
        )
        assert result is None

    def test_optuna_branch_invoked(self, monkeypatch, tmp_path):
        settings = make_settings(
            run_ri_optimizer=True,
            reject_inference_method="parceling",
            ri_validation_split=1.0,
            ri_optimizer_method="optuna",
            ri_optuna_n_trials=10,
        )
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        calls: list[str] = []

        monkeypatch.setattr(
            "src.inference_optimized.compute_pre_reject_inference_data",
            lambda **kw: (pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [2]})),
        )
        monkeypatch.setattr(
            "src.reject_inference.compute_acceptance_rates",
            lambda *a, **kw: pd.DataFrame({"rate": [0.5]}),
        )

        def fake_optuna(*a, **kw):
            calls.append("optuna")
            return pd.DataFrame({"is_best": [True]}), {"uplift_factor": 1.0, "max_risk_multiplier": 1.0}

        def fake_grid(*a, **kw):
            calls.append("grid")
            return pd.DataFrame(), {}

        monkeypatch.setattr(
            "src.reject_inference_optimizer.run_reject_inference_optimization_optuna",
            fake_optuna,
        )
        monkeypatch.setattr(
            "src.reject_inference_optimizer.run_reject_inference_optimization",
            fake_grid,
        )

        ri_optimizer_module.run_ri_optimizer_phase(
            data_booked=pd.DataFrame({"mis_date": pd.to_datetime(["2024-01-01"])}),
            data_demand=pd.DataFrame({"mis_date": pd.to_datetime(["2024-01-01"])}),
            risk_inference={},
            reg_todu_amt_pile=None,
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=output,
        )

        assert calls == ["optuna"]


# ---------------------------------------------------------------------------
# scenarios.py helpers
# ---------------------------------------------------------------------------


class TestBuildScenarioList:
    def test_baseline_mode_returns_only_base(self):
        settings = make_settings(baseline_mode=True, optimum_risk=2.0, risk_step=0.5)
        result = scenarios_module.build_scenario_list(settings, use_fixed_cutoffs=False)
        assert result == [(2.0, "base")]

    def test_base_scenario_only_returns_only_base(self):
        settings = make_settings(base_scenario_only=True, optimum_risk=2.0, risk_step=0.5)
        result = scenarios_module.build_scenario_list(settings, use_fixed_cutoffs=False)
        assert result == [(2.0, "base")]

    def test_fixed_cutoffs_without_run_all_returns_base_only(self):
        settings = make_settings(optimum_risk=2.0, fixed_cutoffs={"run_all_scenarios": False})
        result = scenarios_module.build_scenario_list(settings, use_fixed_cutoffs=True)
        assert result == [(2.0, "base")]

    def test_fixed_cutoffs_with_run_all_returns_three(self):
        settings = make_settings(optimum_risk=2.0, risk_step=0.5, fixed_cutoffs={"run_all_scenarios": True})
        result = scenarios_module.build_scenario_list(settings, use_fixed_cutoffs=True)
        assert result == [(1.5, "pessimistic"), (2.0, "base"), (2.5, "optimistic")]

    def test_default_returns_three_scenarios(self):
        settings = make_settings(optimum_risk=2.0, risk_step=0.5)
        result = scenarios_module.build_scenario_list(settings, use_fixed_cutoffs=False)
        assert result == [(1.5, "pessimistic"), (2.0, "base"), (2.5, "optimistic")]


class TestComputeMrAnnualCoef:
    def test_returns_one_when_mr_dates_unconfigured(self):
        settings = make_settings()
        # MR dates default to None
        assert scenarios_module.compute_mr_annual_coef(settings) == 1.0

    def test_computes_coef_for_6_month_mr_period(self):
        settings = make_settings(date_ini_book_obs_mr="2024-01-01", date_fin_book_obs_mr="2024-06-30")
        # Jan-Jun → 6 months → 12/6 = 2.0
        assert scenarios_module.compute_mr_annual_coef(settings) == pytest.approx(2.0)


class TestSaveCutoffSummaries:
    def test_nd_writes_cell_level_summary(self, monkeypatch, tmp_path):
        settings = make_settings(
            variables=["v1", "v2", "v3"],
            octroi_bins=[0.0, 1.0, 2.0],
            efx_bins=[0.0, 1.0, 2.0],
            bins={
                "v1": {"source_col": "a", "output_col": "v1", "bin_edges": [0.0, 1.0, 2.0]},
                "v2": {"source_col": "b", "output_col": "v2", "bin_edges": [0.0, 1.0, 2.0]},
                "v3": {"source_col": "c", "output_col": "v3", "bin_edges": [0.0, 1.0, 2.0]},
            },
        )
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        consolidated = pd.DataFrame({"cell": [0, 1, 2], "accepted": [1, 0, 1]})
        monkeypatch.setattr(
            scenarios_module, "consolidate_cutoff_summaries", lambda summaries, output_path: consolidated
        )

        scenarios_module.save_cutoff_summaries([pd.DataFrame()], settings, output=output)

        saved = pd.read_csv(output.cutoff_summary_wide_csv)
        assert saved["accepted"].sum() == 2

    def test_2var_writes_wide_pivot(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        consolidated = pd.DataFrame({"var0_bin": [0, 1], "var1_cutoff": [5.0, 6.0], "segment": ["a", "a"]})
        monkeypatch.setattr(
            scenarios_module, "consolidate_cutoff_summaries", lambda summaries, output_path: consolidated
        )
        monkeypatch.setattr(
            scenarios_module,
            "format_cutoff_summary_table",
            lambda cutoff_summary, variables: pd.DataFrame({"pivot": [1, 2]}),
        )

        scenarios_module.save_cutoff_summaries([pd.DataFrame()], settings, output=output)

        saved = pd.read_csv(output.cutoff_summary_wide_csv)
        assert list(saved["pivot"]) == [1, 2]

    def test_empty_consolidated_does_not_write(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(
            scenarios_module, "consolidate_cutoff_summaries", lambda summaries, output_path: pd.DataFrame()
        )
        # Should not raise, and should not write cutoff_summary_wide_csv
        scenarios_module.save_cutoff_summaries([pd.DataFrame()], settings, output=output)
        assert not (tmp_path / "data" / "cutoff_summary_wide.csv").exists()


class TestRunScenarioAnalysisEarlyExit:
    def test_returns_empty_when_no_feasible_solution(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images").mkdir(parents=True, exist_ok=True)

        # Stub visualizer so get_selected_solution returns empty
        class DummyVisualizer:
            def __init__(self, **kwargs):
                pass

            def get_selected_solution(self):
                return pd.DataFrame()

            def get_summary_table(self):
                return pd.DataFrame()

            def save_html(self, path):
                pass

        monkeypatch.setattr(scenarios_module, "RiskProductionVisualizer", DummyVisualizer)

        result = scenarios_module.run_scenario_analysis(
            scenario_risk=2.0,
            scenario_name="base",
            data_summary=pd.DataFrame(),
            data_summary_desagregado=pd.DataFrame(),
            data_summary_sample_no_opt=pd.DataFrame(),
            data_clean=pd.DataFrame({"mis_date": pd.to_datetime([])}),
            data_booked=pd.DataFrame(),
            settings=settings,
            risk_inference={},
            reg_todu_amt_pile=None,
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef_mr=1.0,
            values_per_var={"sc_octroi_new_clus": [0.0, 1.0]},
            output=output,
        )

        assert result.empty
