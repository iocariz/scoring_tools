from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.pipeline.cutoff_optimization as optimization_module
import src.pipeline.inference as inference_module

# After R2b-iv (#63), run_optimization_phase + its helper imports live in
# src.pipeline.cutoff_optimization. The legacy `src.pipeline.optimization`
# shim re-exports for backward compat but does NOT re-expose the module's
# internal imports (run_optimization_pipeline, trace_pareto_frontier, etc.),
# so test monkey-patches must target the new module.
import src.pipeline.preprocessing as preprocessing_module
import src.pipeline.reporting as reporting_module
from src.config import OutputPaths, PreprocessingSettings


class DummyFigure:
    def __init__(self):
        self.paths = []

    def write_html(self, path):
        self.paths.append(path)


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


class TestRunPreprocessingPhase:
    def test_returns_none_when_data_quality_fails(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)

        monkeypatch.setattr(
            preprocessing_module,
            "run_data_quality_checks",
            lambda data, settings, verbose=True: SimpleNamespace(is_valid=False, warnings=[]),
        )

        def fail_complete(*args, **kwargs):
            raise AssertionError("complete_preprocessing_pipeline should not run when DQ fails")

        monkeypatch.setattr(preprocessing_module, "complete_preprocessing_pipeline", fail_complete)

        result = preprocessing_module.run_preprocessing_phase(
            pd.DataFrame({"raw": [1]}),
            settings,
            skip_dq_checks=False,
            output=output,
        )

        assert result is None

    def test_skips_dq_and_normalizes_non_positive_tasa_fin(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        risk_fig = DummyFigure()
        rate_fig = DummyFigure()
        diagnostics = []

        def fail_dq(*args, **kwargs):
            raise AssertionError("run_data_quality_checks should be skipped")

        data_clean = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01"]),
                "oa_amt": [1000.0],
                "sc_octroi_new_clus": [1],
                "new_efx_clus": [1],
            }
        )
        data_booked = pd.DataFrame({"booked": [1]})
        data_demand = pd.DataFrame({"demand": [1]})

        monkeypatch.setattr(preprocessing_module, "run_data_quality_checks", fail_dq)
        monkeypatch.setattr(
            preprocessing_module,
            "complete_preprocessing_pipeline",
            lambda data, settings: (data_clean, data_booked, data_demand),
        )
        monkeypatch.setattr(preprocessing_module, "plot_risk_vs_production", lambda *args, **kwargs: risk_fig)
        monkeypatch.setattr(
            preprocessing_module,
            "plot_bin_threshold_diagnostic",
            lambda data, bin_col, bin_edges, source_col, output_path: diagnostics.append(bin_col),
        )
        monkeypatch.setattr(preprocessing_module, "calculate_stress_factor", lambda data: 1.25)
        monkeypatch.setattr(
            preprocessing_module,
            "calculate_and_plot_transformation_rate",
            lambda *args, **kwargs: {"figure": rate_fig, "overall_rate": 0.0},
        )

        result = preprocessing_module.run_preprocessing_phase(
            pd.DataFrame({"raw": [1]}),
            settings,
            skip_dq_checks=True,
            output=output,
        )

        assert result.data_clean is data_clean
        assert result.data_booked is data_booked
        assert result.data_demand is data_demand
        assert result.stress_factor == pytest.approx(1.25)
        assert result.tasa_fin == pytest.approx(1.0)
        assert risk_fig.paths == [output.risk_vs_production_html]
        assert rate_fig.paths == [output.transformation_rate_html]
        assert diagnostics == ["sc_octroi_new_clus", "new_efx_clus"]

    def test_transformation_rate_computed_in_observation_window(self, monkeypatch, tmp_path):
        """run_preprocessing_phase must not compute tasa_fin on months outside the configured window."""
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        risk_fig = DummyFigure()
        rate_fig = DummyFigure()

        data_clean = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2023-12-01", "2024-06-01"]),
                "oa_amt": [1000.0, 2000.0],
            }
        )

        def fake_complete_preprocessing_pipeline(data, settings):
            return data_clean, pd.DataFrame(), pd.DataFrame()

        monkeypatch.setattr(
            preprocessing_module, "complete_preprocessing_pipeline", fake_complete_preprocessing_pipeline
        )
        monkeypatch.setattr(preprocessing_module, "plot_risk_vs_production", lambda *args, **kwargs: risk_fig)
        monkeypatch.setattr(preprocessing_module, "calculate_stress_factor", lambda *args, **kwargs: 1.0)

        captured = {}

        def fake_calculate_and_plot_transformation_rate(data, date_col, amount_col="oa_amt", n_months=None, **kwargs):
            captured["min_date"] = data[date_col].min()
            captured["max_date"] = data[date_col].max()
            assert captured["min_date"] >= pd.to_datetime(settings.date_ini_book_obs)
            assert captured["max_date"] <= pd.to_datetime(settings.date_fin_book_obs)
            return {"figure": rate_fig, "overall_rate": 0.5}

        monkeypatch.setattr(
            preprocessing_module,
            "calculate_and_plot_transformation_rate",
            fake_calculate_and_plot_transformation_rate,
        )

        result = preprocessing_module.run_preprocessing_phase(
            pd.DataFrame({"raw": [1]}),
            settings,
            skip_dq_checks=True,
            output=output,
        )

        assert result.tasa_fin == pytest.approx(0.5)
        assert risk_fig.paths == [output.risk_vs_production_html]
        assert rate_fig.paths == [output.transformation_rate_html]


class TestRunInferencePhase:
    def test_loads_pretrained_model_and_existing_todu_model(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        model_dir = tmp_path / "models" / "model_20240101"
        model_dir.mkdir(parents=True)
        todu_model_path = model_dir.parent / "todu_model.joblib"
        todu_model_path.write_text("placeholder", encoding="utf-8")
        loaded_model = object()
        loaded_todu_model = object()

        monkeypatch.setattr(
            inference_module,
            "load_model_for_prediction",
            lambda path: (
                loaded_model,
                {
                    "cv_mean_r2": 0.42,
                    "cv_std_r2": 0.05,
                    "cv_std_rmse": 0.02,
                    "model_type": "Ridge",
                    "model_variables": ["sc_octroi_new_clus", "new_efx_clus"],
                },
                ["f1", "f2"],
            ),
        )
        # After R0 (#44) the pipeline uses safe_joblib_load, not raw joblib.load,
        # to enforce SHA-256 + trusted-path checks. Patch the safe wrapper directly.
        monkeypatch.setattr(inference_module, "safe_joblib_load", lambda path: loaded_todu_model)

        def fail_todu_average(*args, **kwargs):
            raise AssertionError("Fallback TODU model training should not run when a saved model exists")

        monkeypatch.setattr(inference_module, "todu_average_inference", fail_todu_average)

        risk_inference, reg_todu_amt_pile = inference_module.run_inference_phase(
            pd.DataFrame(),
            settings,
            model_path=str(model_dir),
            output=output,
        )

        assert risk_inference["best_model_info"]["model"] is loaded_model
        assert risk_inference["best_model_info"]["name"] == "Ridge"
        assert risk_inference["features"] == ["f1", "f2"]
        assert risk_inference["model_variables"] == ["sc_octroi_new_clus", "new_efx_clus"]
        assert reg_todu_amt_pile is loaded_todu_model

    def test_trains_new_model_and_todu_regressor(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        captured = {}
        risk_inference_result = {
            "best_model_info": {
                "name": "Ridge",
                "model_type": "ridge",
                "feature_set": "base",
                "cv_mean_r2": 0.61,
                "cv_std_r2": 0.03,
            }
        }

        def fake_inference_pipeline(**kwargs):
            captured["inference_pipeline"] = kwargs
            return risk_inference_result

        def fake_todu_average_inference(**kwargs):
            captured["todu_average_inference"] = kwargs
            return None, "todu-regressor", None

        monkeypatch.setattr(inference_module, "inference_pipeline", fake_inference_pipeline)
        monkeypatch.setattr(inference_module, "todu_average_inference", fake_todu_average_inference)

        risk_inference, reg_todu_amt_pile = inference_module.run_inference_phase(
            pd.DataFrame({"x": [1]}),
            settings,
            output=output,
        )

        assert risk_inference is risk_inference_result
        assert reg_todu_amt_pile == "todu-regressor"
        assert captured["inference_pipeline"]["variables"] == settings.inference_variables
        assert captured["inference_pipeline"]["bins"] == (
            settings.bins["sc_octroi_new_clus"].bin_edges,
            settings.bins["new_efx_clus"].bin_edges,
        )
        assert captured["inference_pipeline"]["model_base_path"] == output.model_base_path
        assert captured["todu_average_inference"]["variables"] == settings.variables
        assert captured["todu_average_inference"]["model_output_path"] == output.todu_model_joblib


class TestRunOptimizationPhase:
    def test_uses_fixed_cutoff_legacy_path(self, monkeypatch, tmp_path):
        settings = make_settings(
            fixed_cutoffs={
                "sc_octroi_new_clus": [1.0, 2.0],
                "new_efx_clus": [1.0, 2.0],
            }
        )
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()
        captured = {}
        summary_desagregado = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2],
                "new_efx_clus": [1, 2],
                "oa_amt_h0": [1000.0, 2000.0],
                "todu_30ever_h6": [10.0, 20.0],
                "todu_amt_pile_h6": [100.0, 200.0],
            }
        )

        monkeypatch.setattr(
            optimization_module,
            "run_optimization_pipeline",
            lambda **kwargs: summary_desagregado.copy(),
        )

        def fake_create_fixed_cutoff_solution(**kwargs):
            captured["create_fixed_cutoff_solution"] = kwargs
            return pd.DataFrame({"sol_fac": [0], "1": [1.0], "2": [2.0]})

        def fake_kpi_of_fact_sol(**kwargs):
            captured["kpi_of_fact_sol"] = kwargs
            return pd.DataFrame(
                {
                    "sol_fac": [0],
                    "oa_amt_h0": [2500.0],
                    "todu_30ever_h6": [25.0],
                    "todu_amt_pile_h6": [250.0],
                    "b2_ever_h6": [70.0],
                }
            )

        def fail_trace_pareto_frontier(**kwargs):
            raise AssertionError("trace_pareto_frontier should not run for fixed cutoffs")

        monkeypatch.setattr(optimization_module, "create_fixed_cutoff_solution", fake_create_fixed_cutoff_solution)
        monkeypatch.setattr(optimization_module, "kpi_of_fact_sol", fake_kpi_of_fact_sol)
        monkeypatch.setattr(optimization_module, "trace_pareto_frontier", fail_trace_pareto_frontier)

        result = optimization_module.run_optimization_phase(
            data_booked=pd.DataFrame(),
            data_demand=pd.DataFrame(),
            risk_inference={},
            reg_todu_amt_pile="reg",
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=output,
        )

        data_summary_desagregado, data_summary, data_summary_sample_no_opt, values_per_var, grid, pareto_masks, _ = (
            result
        )
        assert list(values_per_var["sc_octroi_new_clus"]) == [1, 2]
        assert grid is None
        assert pareto_masks == []
        assert data_summary_sample_no_opt.equals(data_summary)
        assert "b2_ever_h6" in data_summary_desagregado.columns
        assert "text" in data_summary_desagregado.columns
        assert data_summary["sol_fac"].tolist() == [0]
        assert captured["create_fixed_cutoff_solution"]["inv_var1"] is False
        assert Path(output.pareto_solutions_csv).exists()

    def test_uses_trace_pareto_frontier_for_non_fixed_path(self, monkeypatch, tmp_path):
        settings = make_settings(
            max_swapin_production_pct=30.0, max_swapin_risk=5.0, pareto_n_points=7, milp_time_limit=12.0
        )
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()
        captured = {}
        summary_desagregado = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2],
                "new_efx_clus": [1, 2],
                "oa_amt_h0": [1200.0, 1800.0],
                "todu_30ever_h6": [12.0, 18.0],
                "todu_amt_pile_h6": [120.0, 180.0],
            }
        )

        monkeypatch.setattr(
            optimization_module,
            "run_optimization_pipeline",
            lambda **kwargs: summary_desagregado.copy(),
        )

        def fake_trace_pareto_frontier(**kwargs):
            captured["trace_pareto_frontier"] = kwargs
            return pd.DataFrame({"sol_fac": [0], "oa_amt_h0": [2800.0], "b2_ever_h6": [70.0]}), "grid", [[1, 0]]

        def fake_add_bin_columns(df, pareto_masks, grid, inv_vars):
            captured["add_bin_columns"] = {
                "pareto_masks": pareto_masks,
                "grid": grid,
                "inv_vars": inv_vars,
            }
            result = df.copy()
            result["1"] = [2.0]
            return result

        monkeypatch.setattr(optimization_module, "trace_pareto_frontier", fake_trace_pareto_frontier)
        monkeypatch.setattr(optimization_module, "add_bin_columns", fake_add_bin_columns)

        result = optimization_module.run_optimization_phase(
            data_booked=pd.DataFrame(),
            data_demand=pd.DataFrame(),
            risk_inference={},
            reg_todu_amt_pile="reg",
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=output,
        )

        _, data_summary, data_summary_sample_no_opt, values_per_var, grid, pareto_masks, _ = result
        assert list(values_per_var["new_efx_clus"]) == [1, 2]
        assert grid == "grid"
        assert pareto_masks == [[1, 0]]
        assert data_summary_sample_no_opt.empty
        assert data_summary["1"].tolist() == [2.0]
        assert captured["trace_pareto_frontier"]["max_swapin_production_pct"] == pytest.approx(30.0)
        assert captured["trace_pareto_frontier"]["max_swapin_risk"] == pytest.approx(5.0)
        assert captured["trace_pareto_frontier"]["n_points"] == 7
        assert captured["trace_pareto_frontier"]["milp_time_limit"] == pytest.approx(12.0)
        assert Path(output.pareto_solutions_csv).exists()

    def test_legacy_enumeration_runtime_error_falls_back_to_ga(self, monkeypatch, tmp_path):
        """If legacy enumeration would explode, pipeline should try GA instead of crashing."""
        settings = make_settings(pareto_n_points=5)
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()

        # Minimal base surface for values_per_var (used before the failing enumeration)
        summary_desagregado = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2],
                "new_efx_clus": [1, 2],
                "oa_amt_h0": [1200.0, 1800.0],
                "todu_30ever_h6": [12.0, 18.0],
                "todu_amt_pile_h6": [120.0, 180.0],
            }
        )

        monkeypatch.setattr(
            optimization_module,
            "run_optimization_pipeline",
            lambda **kwargs: summary_desagregado.copy(),
        )

        # Force Pareto sweep to return nothing → triggers legacy enumeration.
        dummy_grid = object()
        monkeypatch.setattr(
            optimization_module,
            "trace_pareto_frontier",
            lambda **kwargs: (pd.DataFrame(), dummy_grid, []),
        )

        # Legacy enumeration should fail fast.
        monkeypatch.setattr(
            optimization_module,
            "get_fact_sol",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("too many combinations")),
        )

        # Mock GA fallback and add_bin_columns to avoid depending on pymoo/grid internals.
        fake_pareto_df = pd.DataFrame({"sol_fac": [0], "oa_amt_h0": [999.0], "b2_ever_h6": [1.23]})
        fake_masks = [[1, 0]]

        import src.optimization_utils as optimization_utils_module

        monkeypatch.setattr(
            optimization_utils_module,
            "_ga_pareto_fallback",
            lambda grid, inv_vars, multiplier, indicators, n_points, **kwargs: (fake_pareto_df, dummy_grid, fake_masks),
        )

        def fake_add_bin_columns(df, pareto_masks, grid, inv_vars):
            out = df.copy()
            out["1"] = [2.0]  # backward compat column
            return out

        monkeypatch.setattr(optimization_module, "add_bin_columns", fake_add_bin_columns)

        result = optimization_module.run_optimization_phase(
            data_booked=pd.DataFrame(),
            data_demand=pd.DataFrame(),
            risk_inference={},
            reg_todu_amt_pile="reg",
            stress_factor=1.0,
            tasa_fin=1.0,
            settings=settings,
            annual_coef=1.0,
            output=output,
        )

        _, data_summary, data_summary_sample_no_opt, _, grid, pareto_masks, _ = result
        assert data_summary_sample_no_opt.empty
        assert grid is dummy_grid
        assert pareto_masks == fake_masks
        assert "b2_ever_h6" in data_summary.columns
        assert Path(output.pareto_solutions_csv).exists()


class TestPipelineReportingWrappers:
    def test_generate_segment_report_renders_when_sections_exist(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)
        captured = {}

        monkeypatch.setattr(
            reporting_module,
            "build_segment_report",
            lambda output, settings, scenarios: SimpleNamespace(sections=[object()]),
        )

        def fake_render_report(context, output_path, template_name=None):
            captured["output_path"] = output_path
            captured["template_name"] = template_name
            return Path(output_path)

        monkeypatch.setattr(reporting_module, "render_report", fake_render_report)

        result = reporting_module.generate_segment_report(settings=settings, output=output, scenarios=["base"])

        assert result == Path(output.segment_report_html)
        assert captured["template_name"] is None

    def test_generate_segment_report_returns_none_when_no_sections(self, monkeypatch, tmp_path):
        settings = make_settings()
        output = OutputPaths(base_dir=tmp_path)

        monkeypatch.setattr(
            reporting_module,
            "build_segment_report",
            lambda output, settings, scenarios: SimpleNamespace(sections=[]),
        )

        def fail_render(*args, **kwargs):
            raise AssertionError("render_report should not run when no sections are available")

        monkeypatch.setattr(reporting_module, "render_report", fail_render)

        result = reporting_module.generate_segment_report(settings=settings, output=output, scenarios=["base"])

        assert result is None

    def test_generate_batch_reports_detects_scenarios_and_renders_segment_and_consolidated(self, monkeypatch, tmp_path):
        output_base = tmp_path / "output"
        segment_dir = output_base / "seg_a"
        data_dir = segment_dir / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "risk_production_summary_table_base.csv").write_text("metric,value\nActual,1\n", encoding="utf-8")
        (data_dir / "risk_production_summary_table_optimistic.csv").write_text(
            "metric,value\nActual,2\n", encoding="utf-8"
        )
        (data_dir / "risk_production_summary_table_mr_base.csv").write_text(
            "metric,value\nActual,3\n", encoding="utf-8"
        )
        (segment_dir / "config_segment.toml").write_text(
            """
[preprocessing]
keep_vars = [\"mis_date\"]
indicators = [\"oa_amt\"]
segment_filter = \"seg_a\"
date_ini_book_obs = \"2024-01-01\"
date_fin_book_obs = \"2024-12-31\"
variables = [\"var1\", \"var2\"]
octroi_bins = [0.0, 1.0]
efx_bins = [0.0, 1.0]
""".strip(),
            encoding="utf-8",
        )
        captured = {}

        def fake_build_segment_report(output, settings, scenarios):
            captured["segment_filter"] = settings.segment_filter
            captured["scenarios"] = scenarios
            return SimpleNamespace(sections=[object()])

        def fake_build_consolidated_report(output_base, segments, supersegments):
            captured["consolidated_args"] = (output_base, segments, supersegments)
            return SimpleNamespace(sections=[object()])

        def fake_render_report(context, output_path, template_name=None):
            rendered = captured.setdefault("rendered_paths", [])
            rendered.append((Path(output_path), template_name))
            return Path(output_path)

        monkeypatch.setattr(reporting_module, "build_segment_report", fake_build_segment_report)
        monkeypatch.setattr(reporting_module, "build_consolidated_report", fake_build_consolidated_report)
        monkeypatch.setattr(reporting_module, "render_report", fake_render_report)

        reports = reporting_module.generate_batch_reports(
            output_base=str(output_base),
            segments={"seg_a": {}},
            supersegments={"shared": {}},
            segment_results={"seg_a": True, "seg_b": False},
        )

        assert captured["segment_filter"] == "seg_a"
        assert captured["scenarios"] == ["base", "optimistic"]
        assert reports["seg_a"] == segment_dir / "report.html"
        assert reports["consolidated"] == output_base / "consolidated_report.html"
        assert captured["rendered_paths"][-1] == (output_base / "consolidated_report.html", "consolidated_report.html")
