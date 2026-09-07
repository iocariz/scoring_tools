"""Tests for analyze_logs.py — signal extraction, recommendations, and alt-config generation.

Fixtures use the REAL log message formats emitted by the pipeline (loguru lines copied from
actual runs), so these tests double as a drift guard between the pipeline's logging and the
analyzer's extraction patterns.
"""

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("analyze_logs", Path(__file__).parents[1] / "analyze_logs.py")
analyze_logs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_logs)


def _log(msg, level="INFO"):
    return f"2026-09-07 10:00:00.000 | {level:<8} | src.pipeline.mod:func:42 - {msg}"


def _extract(lines, tmp_path):
    path = tmp_path / "run.log"
    path.write_text("\n".join(lines) + "\n")
    entries = analyze_logs.parse_log_file(path)
    assert len(entries) == len(lines), "every fixture line must parse"
    segments, _ = analyze_logs.extract_signals(entries)
    return segments


def test_extract_core_signals_real_formats(tmp_path):
    segments = _extract(
        [
            _log("PROCESSING SEGMENT: known"),
            _log("Segment filter: known_af"),
            _log(
                "[known] Preprocessing done | clean=1,109,299 booked=88,477 demand=480,718 | "
                "stress=1.0000 tasa_fin=38.14% | 3.5s"
            ),
            _log("Per-loan zero proportion (no-default loans): 97.4%"),
            _log("CV RMSE: 0.0159 ± 0.0008"),
            _log("CV R²: 0.4512 ± 0.0300"),
            _log(
                "[known] Inference done | Tweedie (Optuna Tuned p=1.98, α=3.81) + original (glm) | "
                "features=original | CV R2=0.4512 +/- 0.0300 | 12.3s"
            ),
            _log("Final model: Tweedie (Optuna Tuned p=1.98, α=3.81) + original"),
            _log(
                "[known] Optimization done | mode=milp_pareto | 162 solutions | 20x10 grid | "
                "b2 range: [0.18%, 2.39%] | optimum_risk=3.5% | 3.1s"
            ),
            _log("[known] Scenario base | risk_threshold=2.00% | selected b2=1.99% | production=29,567,129"),
            _log("Overall PSI: 0.2657"),
            _log("[known] Pipeline complete | 3 scenarios | 38.6s total"),
        ],
        tmp_path,
    )
    s = segments["known"]
    assert s.segment_filter == "known_af"
    assert s.n_booked == 88477
    assert s.n_demand == 480718
    assert s.stress_factor == 1.0
    assert s.tasa_fin_pct == pytest.approx(38.14)
    # "Per-loan zero proportion" (lowercase z mid-sentence) must be captured
    assert s.zero_proportion == pytest.approx(97.4)
    assert s.cv_rmse_mean == pytest.approx(0.0159)
    assert s.cv_r2_mean == pytest.approx(0.4512)
    assert s.model_name == "Tweedie (Optuna Tuned p=1.98, α=3.81)"
    assert s.feature_set == "original"
    assert s.grid_dims == "20x10"
    assert s.pareto_n_solutions == 162
    assert s.b2_range_min == pytest.approx(0.18)
    assert s.b2_range_max == pytest.approx(2.39)
    assert s.optimum_risk == pytest.approx(3.5)
    assert s.scenarios["base"].selected_b2 == pytest.approx(1.99)
    assert s.scenarios["base"].production == pytest.approx(29567129)
    assert s.psi_value == pytest.approx(0.2657)


def test_elapsed_times_anchor_to_trailing_seconds(tmp_path):
    """Timing must come from the trailing '| <n>s', never an earlier number — the old
    patterns matched the '1' in '1 scenarios' (via '\\s*s' eating into 'scenarios') and
    could grab a model hyperparameter from the inference line."""
    segments = _extract(
        [
            _log("PROCESSING SEGMENT: seg"),
            _log("[seg] Preprocessing done | clean=100 booked=50 demand=80 | stress=1.0000 tasa_fin=38.14% | 3.5s"),
            _log("[seg] Inference done | Tweedie (Optuna Tuned p=1.98, α=3.81) | features=original | 12.3s"),
            _log("[seg] Optimization done | mode=milp_pareto | 5 solutions | 2x2 grid | 7.9s"),
            _log("[seg] Pipeline complete | 1 scenarios | 38.6s total"),
        ],
        tmp_path,
    )
    s = segments["seg"]
    assert s.elapsed_preprocessing == pytest.approx(3.5)
    assert s.elapsed_inference == pytest.approx(12.3)
    assert s.elapsed_optimization == pytest.approx(7.9)
    assert s.elapsed_total == pytest.approx(38.6)  # not 1 (scenario count), not 1.98


def test_supersegment_metrics_propagate_to_degenerate_members(tmp_path):
    segments = _extract(
        [
            _log("TRAINING SUPERSEGMENT MODEL: direct"),
            _log("CV RMSE: 0.0159 ± 0.0008"),
            _log("Train R² (in-sample): 0.6699"),
            _log("PROCESSING SEGMENT: member"),
            _log("CV R²: 0.0000 ± 0.0000"),
        ],
        tmp_path,
    )
    m = segments["member"]
    assert m.train_r2 == pytest.approx(0.6699)
    assert m.cv_rmse_mean == pytest.approx(0.0159)
    assert m.cv_r2_mean is None  # degenerate placeholder cleared


def test_zero_proportion_recommendation_uses_expert_flag():
    s = analyze_logs.SegmentMetrics(
        name="seg", zero_proportion=97.4, scenarios={"base": analyze_logs.ScenarioResult(name="base")}
    )
    recs = analyze_logs.generate_recommendations({"seg": s})
    hurdle = [r for r in recs if "model_hurdle_per_loan" in (r.setting + r.suggested)]
    assert len(hurdle) == 1
    # above 99.9% the hurdle candidate is skipped as degenerate — no recommendation
    s.zero_proportion = 99.95
    assert not [r for r in analyze_logs.generate_recommendations({"seg": s}) if "model_hurdle_per_loan" in r.setting]


def test_coverage_warning_flags_missing_core_signals():
    empty = analyze_logs.SegmentMetrics(name="seg")
    recs = [r for r in analyze_logs.generate_recommendations({"seg": empty}) if r.category == "Analyzer"]
    assert len(recs) == 1
    for phase in ["preprocessing summary", "model/inference summary", "optimization summary", "scenario selection"]:
        assert phase in recs[0].message

    # a segment with all core signals present raises no coverage warning
    full = analyze_logs.SegmentMetrics(
        name="seg",
        n_booked=1000,
        model_name="Tweedie",
        grid_dims="20x10",
        scenarios={"base": analyze_logs.ScenarioResult(name="base")},
    )
    assert not [r for r in analyze_logs.generate_recommendations({"seg": full}) if r.category == "Analyzer"]

    # errored segments are exempt (the failure itself is already reported)
    errored = analyze_logs.SegmentMetrics(name="seg", errors=["boom"])
    assert not [r for r in analyze_logs.generate_recommendations({"seg": errored}) if r.category == "Analyzer"]


def test_alt_toml_adjusts_unreachable_optimum_risk(tmp_path):
    seg_toml = tmp_path / "segments.toml"
    seg_toml.write_text('[segments.seg]\nsegment_filter = "seg_af"\noptimum_risk = 1.0\nrisk_step = 0.1\n')
    s = analyze_logs.SegmentMetrics(name="seg", optimum_risk=1.0, pareto_fallback_targets=[1.0], pareto_actual_min=1.43)
    toml_text, changes = analyze_logs.generate_alternative_segments_toml({"seg": s}, seg_toml)
    assert len(changes) == 1 and "optimum_risk" in changes[0]
    assert "optimum_risk = 1.45  # CHANGED (was 1.0)" in toml_text
    assert 'segment_filter = "seg_af"' in toml_text  # untouched keys preserved


def test_toml_value_formatting():
    f = analyze_logs._toml_value
    assert f(True) == "true" and f(False) == "false"
    assert f("x") == '"x"'
    assert f([1, 2.5, "a"]) == '[1, 2.5, "a"]'
    assert f(float("inf")) == "inf" and f(float("-inf")) == "-inf"
    assert f([float("-inf"), 3]) == "[-inf, 3]"
    assert f(float("nan")) == "nan"  # must not raise
    assert not math.isnan(0)  # keep math import honest


def test_format_report_smoke(tmp_path):
    segments = _extract(
        [
            _log("PROCESSING SEGMENT: seg"),
            _log("[seg] Scenario base | risk_threshold=2.00% | selected b2=1.99% | production=29,567,129"),
            _log("Actual: risk=2.03% prod=34,138,255"),
        ],
        tmp_path,
    )
    recs = analyze_logs.generate_recommendations(segments)
    report = analyze_logs.format_report(segments, recs)
    assert "PIPELINE LOG ANALYSIS REPORT" in report
    assert "Actual:   risk=2.03%  prod=€34,138,255" in report
