"""Tests for HTML report generation (src/reporting.py & src/pipeline/reporting.py)."""

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.config import OutputPaths, PreprocessingSettings
from src.reporting import (
    ReportContext,
    ReportSection,
    _build_acceptance_matrices,
    _build_acceptance_strip_1d,
    _build_cutoff_comparison_section,
    _build_cutoff_reference_table,
    _build_portfolio_metric_table,
    _build_portfolio_summary_sections,
    _build_scenario_kpi_table,
    _detect_variable_cols,
    _feasibility_banner,
    _portfolio_group_list,
    _read_cutoff_data,
    build_consolidated_report,
    build_segment_report,
    csv_to_html_table,
    extract_plotly_div,
    render_report,
)

# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_PLOTLY_HTML = """<html>
<head><meta charset="utf-8" /></head>
<body>
    <div id="abc12345" class="plotly-graph-div" style="height:100%; width:100%;">
    </div>
    <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        Plotly.newPlot("abc12345", [{"x":[1,2],"y":[3,4],"type":"scatter"}], {}, {responsive: true});
    </script>
</body>
</html>"""


@pytest.fixture
def sample_csv(tmp_path):
    """Write a small CSV and return its path."""
    df = pd.DataFrame(
        {
            "metric": ["Actual", "Optimum selected", "Swap-in", "Swap-out"],
            "production (€)": [100_000.0, 120_000.0, 30_000.0, 10_000.0],
            "todu_30ever_h6": [500.0, 400.0, 100.0, 200.0],
            "todu_amt_pile_h6": [50_000.0, 55_000.0, 20_000.0, 15_000.0],
        }
    )
    path = tmp_path / "summary.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def plotly_html_file(tmp_path):
    """Write a sample Plotly HTML file and return its path."""
    path = tmp_path / "chart.html"
    path.write_text(SAMPLE_PLOTLY_HTML, encoding="utf-8")
    return path


@pytest.fixture
def segment_output_dir(tmp_path, sample_csv, plotly_html_file):
    """Create a mock segment output directory with sample artifacts."""
    base = tmp_path / "seg_a"
    data_dir = base / "data"
    images_dir = base / "images"
    models_dir = base / "models"
    data_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # CSV artifacts
    df = pd.read_csv(sample_csv)
    df.to_csv(data_dir / "risk_production_summary_table_base.csv", index=False)
    df.to_csv(data_dir / "optimal_solution_base.csv", index=False)
    df.to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

    # Plotly HTML artifacts
    html_content = plotly_html_file.read_text()
    (images_dir / "risk_vs_production.html").write_text(html_content)
    (images_dir / "transformation_rate.html").write_text(html_content)
    (images_dir / "b2_ever_h6_vs_octroi_and_risk_score.html").write_text(html_content)
    (images_dir / "risk_production_visualizer_base.html").write_text(html_content)

    return base


@pytest.fixture
def minimal_settings():
    """Return a minimal PreprocessingSettings for testing."""
    return PreprocessingSettings(
        keep_vars=["col_a", "col_b"],
        indicators=["ind_a"],
        segment_filter="seg_a",
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-31",
        variables=["var1", "var2"],
        octroi_bins=[0, 50, 100],
        efx_bins=[0, 50, 100],
    )


# =============================================================================
# Tests: extract_plotly_div
# =============================================================================


class TestExtractPlotlyDiv:
    def test_valid_html(self, plotly_html_file):
        result = extract_plotly_div(plotly_html_file)
        assert result is not None
        assert '<div id="abc12345"' in result
        assert "Plotly.newPlot" in result

    def test_missing_file(self, tmp_path):
        result = extract_plotly_div(tmp_path / "nonexistent.html")
        assert result is None

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.html"
        empty.write_text("")
        result = extract_plotly_div(empty)
        assert result is None

    def test_no_plotly_content(self, tmp_path):
        plain = tmp_path / "plain.html"
        plain.write_text("<html><body><p>Hello</p></body></html>")
        result = extract_plotly_div(plain)
        assert result is None

    def test_bare_script_tag_plotly3(self, tmp_path):
        """Regression: Plotly >= 3.x emits a bare ``<script>`` (no type attr).

        The div-extraction regex must accept it, else every chart silently
        drops out of the report after a Plotly upgrade.
        """
        path = tmp_path / "chart3.html"
        path.write_text(
            "<html><body>"
            '<div id="abc12345" class="plotly-graph-div" style="height:100%;"></div>'
            "            <script>                window.PLOTLYENV=window.PLOTLYENV || {};"
            '                Plotly.newPlot("abc12345", [{"x":[1,2],"y":[3,4]}], {});</script>'
            "</body></html>",
            encoding="utf-8",
        )
        result = extract_plotly_div(path)
        assert result is not None
        assert '<div id="abc12345"' in result
        assert "Plotly.newPlot" in result


# =============================================================================
# Tests: csv_to_html_table
# =============================================================================


class TestCsvToHtmlTable:
    def test_basic_rendering(self, sample_csv):
        result = csv_to_html_table(sample_csv)
        assert result is not None
        assert "<table" in result
        assert "report-table" in result
        assert "Actual" in result

    def test_float_formatting(self, sample_csv):
        result = csv_to_html_table(sample_csv, float_format=",.0f")
        assert result is not None
        assert "100,000" in result

    def test_max_rows(self, tmp_path):
        df = pd.DataFrame({"a": range(100), "b": np.random.randn(100)})
        path = tmp_path / "big.csv"
        df.to_csv(path, index=False)
        result = csv_to_html_table(path, max_rows=5)
        assert result is not None
        # Should contain 5 data rows + header
        assert result.count("<tr>") <= 6

    def test_missing_file(self, tmp_path):
        result = csv_to_html_table(tmp_path / "nonexistent.csv")
        assert result is None

    def test_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n")
        result = csv_to_html_table(path)
        assert result is None

    def test_custom_css_class(self, sample_csv):
        result = csv_to_html_table(sample_csv, css_class="my-table")
        assert result is not None
        assert "my-table" in result


class TestCutoffReferenceEscaping:
    def test_escapes_untrusted_text_cells(self, tmp_path):
        path = tmp_path / "cutoff_summary_wide.csv"
        df = pd.DataFrame(
            {
                "segment": ["seg_a"],
                "scenario": ["base"],
                "var0": ['<script>alert("x")</script>'],
                "risk_pct": [1.23],
                "production": [1000.0],
            }
        )
        df.to_csv(path, index=False)

        html = _build_cutoff_reference_table(path)
        assert html is not None
        assert "<script>alert" not in html
        assert "&lt;script&gt;alert" in html


class TestCutoffReference2VarCellLevel:
    def test_renders_compact_grid_per_scenario(self, tmp_path):
        # 2-var cell-level format (accepted column) must render as colored
        # acceptance pivots per scenario, not a one-row-per-cell listing.
        path = tmp_path / "cutoff_summary_wide.csv"
        rows = []
        for scen in ("base", "optimistic"):
            for v0 in (1.0, 2.0):
                for v1 in (1.0, 2.0, 3.0):
                    rows.append(
                        {
                            "var0": v0,
                            "var1": v1,
                            "accepted": int(v1 >= v0),
                            "segment": "seg_a",
                            "scenario": scen,
                            "risk_pct": 2.0,
                            "production": 1000.0,
                        }
                    )
        pd.DataFrame(rows).to_csv(path, index=False)

        html = _build_cutoff_reference_table(path)
        assert html is not None
        assert "matrix-table" in html  # compact pivot, not the row listing
        assert "cutoff-ref-table" not in html
        assert "badge-base" in html and "badge-optimistic" in html
        assert "Accepted cells: 5/6" in html

    def test_multi_scenario_not_truncated(self, tmp_path):
        # 3 scenarios x 200 cells = 600 rows — the old max_rows=200 head()
        # would have dropped 2 scenarios before pivoting.
        path = tmp_path / "cutoff_summary_wide.csv"
        rows = []
        for scen in ("pessimistic", "base", "optimistic"):
            for v0 in range(1, 21):
                for v1 in range(1, 11):
                    rows.append(
                        {
                            "var0": float(v0),
                            "var1": float(v1),
                            "accepted": 1,
                            "segment": "seg_a",
                            "scenario": scen,
                            "risk_pct": 2.0,
                            "production": 1000.0,
                        }
                    )
        pd.DataFrame(rows).to_csv(path, index=False)

        html = _build_cutoff_reference_table(path)
        assert html is not None
        for badge in ("badge-pessimistic", "badge-base", "badge-optimistic"):
            assert badge in html


# =============================================================================
# Tests: build_segment_report
# =============================================================================


class TestBuildSegmentReport:
    def test_builds_with_artifacts(self, segment_output_dir, minimal_settings):
        output = OutputPaths(base_dir=segment_output_dir)
        context = build_segment_report(output, minimal_settings, scenarios=["base"])

        assert isinstance(context, ReportContext)
        assert context.segment_name == "seg_a"
        assert len(context.sections) > 0

        section_ids = [s.id for s in context.sections]
        assert "executive-summary" in section_ids
        assert "preprocessing" in section_ids
        assert "model-inference" in section_ids

    def test_empty_directory(self, tmp_path, minimal_settings):
        """Report from an empty directory should still produce an executive summary."""
        base = tmp_path / "empty_seg"
        (base / "data").mkdir(parents=True)
        (base / "images").mkdir(parents=True)
        (base / "models").mkdir(parents=True)

        output = OutputPaths(base_dir=base)
        context = build_segment_report(output, minimal_settings, scenarios=["base"])

        assert isinstance(context, ReportContext)
        # Executive summary always has config notes
        assert any(s.id == "executive-summary" for s in context.sections)
        # No lineage artifact present -> lineage is None (M2)
        assert context.lineage is None

    def test_picks_up_run_lineage(self, tmp_path, minimal_settings):
        """M2: build_segment_report surfaces run_lineage.json when present."""
        import json

        base = tmp_path / "lineage_seg"
        (base / "data").mkdir(parents=True)
        (base / "images").mkdir(parents=True)
        (base / "models").mkdir(parents=True)

        output = OutputPaths(base_dir=base)
        Path(output.run_lineage_json).write_text(json.dumps({"run_id": "RID42", "data": {"rows_loaded": 7}}))

        context = build_segment_report(output, minimal_settings, scenarios=["base"])
        assert context.lineage is not None
        assert context.lineage["run_id"] == "RID42"

        # render_report must forward lineage to the template (regression guard:
        # render_report maps fields explicitly, so a new context field is easy to drop).
        out_html = base / "report.html"
        render_report(context, out_html)
        html = out_html.read_text()
        assert "Data Lineage / Provenance" in html
        assert "RID42" in html

    def test_multiple_scenarios(self, segment_output_dir, minimal_settings):
        # Add a pessimistic scenario CSV
        data_dir = segment_output_dir / "data"
        df = pd.read_csv(data_dir / "risk_production_summary_table_base.csv")
        df.to_csv(data_dir / "risk_production_summary_table_pessimistic.csv", index=False)

        output = OutputPaths(base_dir=segment_output_dir)
        context = build_segment_report(output, minimal_settings, scenarios=["pessimistic", "base"])

        section_ids = [s.id for s in context.sections]
        assert "scenario-pessimistic" in section_ids
        assert "scenario-base" in section_ids


# =============================================================================
# Tests: render_report
# =============================================================================


class TestRenderReport:
    def test_renders_valid_html(self, tmp_path):
        context = ReportContext(
            title="Test Report",
            segment_name="test_seg",
            generation_date="2024-01-01 12:00",
            date_range="2023-01-01 to 2023-12-31",
            sections=[
                ReportSection(
                    id="section-1",
                    title="Section One",
                    notes=["Note A", "Note B"],
                    tables=["<table><tr><td>data</td></tr></table>"],
                ),
            ],
        )

        output_path = tmp_path / "report.html"
        result = render_report(context, output_path)

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "Test Report" in content
        assert "Section One" in content
        assert "Note A" in content
        assert "<table>" in content

    def test_contains_expected_sections(self, tmp_path):
        sections = [
            ReportSection(id="a", title="Alpha"),
            ReportSection(id="b", title="Beta", charts=["<div>chart</div>"]),
        ]
        context = ReportContext(
            title="Multi",
            segment_name="s",
            generation_date="2024-01-01",
            date_range="range",
            sections=sections,
        )

        path = tmp_path / "multi.html"
        render_report(context, path)
        content = path.read_text()

        assert 'id="a"' in content
        assert 'id="b"' in content
        assert "Alpha" in content
        assert "Beta" in content

    def test_creates_parent_dirs(self, tmp_path):
        context = ReportContext(
            title="T",
            segment_name="s",
            generation_date="d",
            date_range="r",
            sections=[ReportSection(id="x", title="X")],
        )
        deep_path = tmp_path / "a" / "b" / "report.html"
        result = render_report(context, deep_path)
        assert result.exists()


# =============================================================================
# Tests: build_consolidated_report
# =============================================================================


class TestBuildConsolidatedReport:
    def test_with_consolidated_csv(self, tmp_path):
        """Consolidated report should pick up the consolidated CSV."""
        df = pd.DataFrame(
            {
                "group": ["TOTAL", "segment_a"],
                "period": ["main", "main"],
                "scenario": ["base", "base"],
                "actual_production": [200_000, 100_000],
                "actual_risk_pct": [1.5, 1.2],
                "optimum_production": [250_000, 130_000],
                "optimum_risk_pct": [1.3, 1.0],
                "production_delta_pct": [25.0, 30.0],
            }
        )
        df.to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        context = build_consolidated_report(tmp_path, segments={"seg_a": {}}, supersegments={})
        assert isinstance(context, ReportContext)
        assert context.title == "Consolidated Report"

        section_ids = [s.id for s in context.sections]
        assert "portfolio-a" in section_ids
        assert "portfolio-total" in section_ids

    def test_empty_output(self, tmp_path):
        """Consolidated report with no artifacts should have no sections."""
        context = build_consolidated_report(tmp_path, segments={}, supersegments={})
        assert isinstance(context, ReportContext)
        assert len(context.sections) == 0

    def test_multi_segment(self, tmp_path):
        """Consolidated report with multiple segment dirs picks up cutoff comparison."""
        rng = np.random.RandomState(42)
        for seg in ["seg_a", "seg_b"]:
            data_dir = tmp_path / seg / "data"
            data_dir.mkdir(parents=True)
            rows = []
            for scen in ["pessimistic", "base", "optimistic"]:
                for v1 in range(3):
                    for v2 in range(3):
                        rows.append(
                            {
                                "octroi_bin": v1,
                                "efx_bin": v2,
                                "accepted": int(rng.rand() > 0.4),
                                "segment": seg,
                                "scenario": scen,
                                "risk_pct": 1.0 + rng.rand(),
                                "production": 50_000_000 + rng.rand() * 10_000_000,
                                "risk_ci_lower": 0.5,
                                "risk_ci_upper": 1.5,
                                "production_ci_lower": 48_000_000,
                                "production_ci_upper": 52_000_000,
                            }
                        )
            pd.DataFrame(rows).to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

        context = build_consolidated_report(tmp_path, segments={"seg_a": {}, "seg_b": {}}, supersegments={})

        section_ids = [s.id for s in context.sections]
        assert "cutoff-comparison" in section_ids

        cutoff_section = next(s for s in context.sections if s.id == "cutoff-comparison")
        # KPI summary table + acceptance matrices
        assert len(cutoff_section.tables) == 2
        assert "Scenario KPI Summary" in cutoff_section.tables[0]
        assert "Acceptance Matrices" in cutoff_section.tables[1]


# =============================================================================
# Helpers: shared cutoff DataFrame fixture
# =============================================================================


def _make_cutoff_df(
    segments=("seg_a",),
    scenarios=("pessimistic", "base", "optimistic"),
    grid_size=(3, 4),
    include_accepted=True,
    seed=42,
):
    """Build a synthetic cutoff DataFrame for testing helpers."""
    rng = np.random.RandomState(seed)
    rows = []
    for seg in segments:
        for scen in scenarios:
            for v1 in range(grid_size[0]):
                for v2 in range(grid_size[1]):
                    row = {
                        "octroi_bin": v1,
                        "efx_bin": v2,
                        "segment": seg,
                        "scenario": scen,
                        "risk_pct": round(0.5 + rng.rand(), 2),
                        "production": round(40_000_000 + rng.rand() * 20_000_000, 2),
                        "risk_ci_lower": 0.3,
                        "risk_ci_upper": 1.7,
                        "production_ci_lower": 38_000_000,
                        "production_ci_upper": 52_000_000,
                    }
                    if include_accepted:
                        row["accepted"] = int(rng.rand() > 0.4)
                    rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Tests: _detect_variable_cols
# =============================================================================


class TestDetectVariableCols:
    def test_detects_variable_cols(self):
        df = _make_cutoff_df()
        cols = _detect_variable_cols(df)
        assert "octroi_bin" in cols
        assert "efx_bin" in cols
        assert "segment" not in cols
        assert "accepted" not in cols
        assert "risk_pct" not in cols

    def test_preserves_order(self):
        df = _make_cutoff_df()
        cols = _detect_variable_cols(df)
        assert cols == ["octroi_bin", "efx_bin"]


# =============================================================================
# Tests: _read_cutoff_data
# =============================================================================


class TestReadCutoffData:
    def test_reads_multiple_segments(self, tmp_path):
        for seg in ["seg_a", "seg_b"]:
            data_dir = tmp_path / seg / "data"
            data_dir.mkdir(parents=True)
            _make_cutoff_df(segments=(seg,)).to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

        result = _read_cutoff_data(tmp_path, {"seg_a": {}, "seg_b": {}})
        assert result is not None
        assert set(result["segment"].unique()) == {"seg_a", "seg_b"}

    def test_missing_csv_skipped(self, tmp_path):
        data_dir = tmp_path / "seg_a" / "data"
        data_dir.mkdir(parents=True)
        _make_cutoff_df(segments=("seg_a",)).to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

        result = _read_cutoff_data(tmp_path, {"seg_a": {}, "seg_missing": {}})
        assert result is not None
        assert list(result["segment"].unique()) == ["seg_a"]

    def test_all_missing_returns_none(self, tmp_path):
        result = _read_cutoff_data(tmp_path, {"seg_x": {}})
        assert result is None

    def test_adds_segment_col_if_missing(self, tmp_path):
        data_dir = tmp_path / "seg_a" / "data"
        data_dir.mkdir(parents=True)
        df = _make_cutoff_df(segments=("seg_a",)).drop(columns="segment")
        df.to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

        result = _read_cutoff_data(tmp_path, {"seg_a": {}})
        assert "segment" in result.columns
        assert result["segment"].iloc[0] == "seg_a"


# =============================================================================
# Tests: _build_scenario_kpi_table
# =============================================================================


class TestBuildScenarioKpiTable:
    def test_one_row_per_segment_scenario(self):
        df = _make_cutoff_df(segments=("seg_a", "seg_b"), scenarios=("pessimistic", "base", "optimistic"))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        # 2 segments × 3 scenarios = 6 data rows + 1 header row
        assert html.count("<tr>") == 7
        assert "seg_a" in html
        assert "seg_b" in html

    def test_scenario_ordering(self):
        df = _make_cutoff_df(scenarios=("optimistic", "base", "pessimistic"))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        pess_pos = html.index("pessimistic")
        base_pos = html.index("base")
        opt_pos = html.index("optimistic")
        assert pess_pos < base_pos < opt_pos

    def test_production_in_millions(self):
        df = _make_cutoff_df(scenarios=("base",))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        # Production values are ~40-60M, divided by 1e6 should be ~40-60
        assert "Production (M)" in html
        # Check formatted values are reasonable (not raw large numbers)
        assert "000,000" not in html

    def test_no_acceptance_rate_column(self):
        df = _make_cutoff_df(scenarios=("base",))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        # Acceptance rate columns were removed (misleading for 2-var)
        assert "Acc. Rate" not in html
        assert "Accepted" not in html

    def test_without_accepted_col(self):
        df = _make_cutoff_df(include_accepted=False, scenarios=("base",))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        assert "report-table" in html

    def test_ci_formatting(self):
        df = _make_cutoff_df(scenarios=("base",))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        # Risk CI should be present as [lower, upper]
        assert "[0.30, 1.70]" in html


# =============================================================================
# Tests: _build_acceptance_matrices
# =============================================================================


class TestBuildAcceptanceMatrices:
    def test_renders_base_scenario(self):
        df = _make_cutoff_df()
        html = _build_acceptance_matrices(df, ["octroi_bin", "efx_bin"])
        assert html is not None
        assert "seg_a" in html
        # Contains accept/reject color cells
        assert "m-ok" in html or "m-no" in html

    def test_returns_none_without_accepted(self):
        df = _make_cutoff_df(include_accepted=False)
        result = _build_acceptance_matrices(df, ["octroi_bin", "efx_bin"])
        assert result is None

    def test_returns_none_without_scenario(self):
        df = _make_cutoff_df().drop(columns="scenario")
        result = _build_acceptance_matrices(df, ["octroi_bin", "efx_bin"])
        assert result is None

    def test_returns_none_for_missing_scenario(self):
        df = _make_cutoff_df(scenarios=("pessimistic",))
        result = _build_acceptance_matrices(df, ["octroi_bin", "efx_bin"], scenario="base")
        assert result is None

    def test_multiple_segments(self):
        df = _make_cutoff_df(segments=("seg_a", "seg_b"))
        html = _build_acceptance_matrices(df, ["octroi_bin", "efx_bin"])
        assert html is not None
        assert "seg_a" in html
        assert "seg_b" in html

    def test_three_variable_slices(self):
        rng = np.random.RandomState(42)
        rows = []
        for scen in ["base"]:
            for v1 in range(2):
                for v2 in range(2):
                    for v3 in range(2):
                        rows.append(
                            {
                                "var_a": v1,
                                "var_b": v2,
                                "income_bin": v3,
                                "accepted": int(rng.rand() > 0.5),
                                "segment": "seg_a",
                                "scenario": scen,
                                "risk_pct": 1.0,
                                "production": 50_000_000,
                            }
                        )
        df = pd.DataFrame(rows)
        html = _build_acceptance_matrices(df, ["var_a", "var_b", "income_bin"])
        assert html is not None
        assert "income_bin=" in html

    def test_colored_cells(self):
        df = _make_cutoff_df(grid_size=(2, 2))
        html = _build_acceptance_matrices(df, ["octroi_bin", "efx_bin"])
        assert html is not None
        assert "m-ok" in html or "m-no" in html

    def test_mixed_dimensionality_segments_both_render(self):
        # A 2-var segment (income_bin NaN) consolidated with a 3-var segment. The old global
        # slice-var groupby dropped the 2-var segment's NaN income_bin group -> it vanished.
        # Per-segment vars now render each on its own dimensionality.
        rows = []
        for a in range(2):
            for b in range(2):
                rows.append(
                    {"var_a": a, "var_b": b, "income_bin": None, "accepted": 1, "segment": "seg_2d", "scenario": "base"}
                )
        for a in range(2):
            for b in range(2):
                for inc in range(2):
                    rows.append(
                        {
                            "var_a": a,
                            "var_b": b,
                            "income_bin": inc,
                            "accepted": 0,
                            "segment": "seg_3d",
                            "scenario": "base",
                        }
                    )
        df = pd.DataFrame(rows)
        html = _build_acceptance_matrices(df, ["var_a", "var_b", "income_bin"])
        assert html is not None
        assert "seg_2d" in html  # the 2-var segment did NOT silently vanish
        assert "seg_3d" in html
        assert "income_bin=" in html  # 3-var slice label present for the 3-var segment


class TestFeasibilityBanner:
    def _csv(self, tmp_path, feasible_values):
        p = tmp_path / "rp.csv"
        pd.DataFrame(
            {
                "Metric": ["Actual", "Optimum selected"],
                "Risk (%)": [1.0, 1.5],
                "Feasible": feasible_values,
            }
        ).to_csv(p, index=False)
        return p

    def test_banner_when_infeasible(self, tmp_path):
        p = self._csv(tmp_path, [False, False])
        html = _feasibility_banner(p, "base")
        assert "INFEASIBLE" in html
        assert "optimum_risk" in html

    def test_no_banner_when_feasible(self, tmp_path):
        p = self._csv(tmp_path, [True, True])
        assert _feasibility_banner(p, "base") == ""

    def test_no_banner_when_column_absent(self, tmp_path):
        p = tmp_path / "rp.csv"
        pd.DataFrame({"Metric": ["Actual"], "Risk (%)": [1.0]}).to_csv(p, index=False)
        assert _feasibility_banner(p, "base") == ""

    def test_no_banner_when_file_missing(self, tmp_path):
        assert _feasibility_banner(tmp_path / "nope.csv", "base") == ""


class TestBuildAcceptanceStrip1D:
    def test_three_state_no_crash_and_observed_denominator(self):
        # A 1D strip with an unobserved (NaN) cell: must NOT crash (raw .sum() -> int(NaN)),
        # render the NaN cell as m-na, and compute the rate over OBSERVED cells only.
        df = pd.DataFrame(
            {
                "segment": ["seg_a"] * 4,
                "octroi_bin": [0, 1, 2, 3],
                # 2 accepted, 1 rejected, 1 unobserved -> 2/3 observed accepted = 67%
                "accepted": [1.0, 1.0, 0.0, float("nan")],
            }
        )
        html = _build_acceptance_strip_1d(df, "octroi_bin")
        assert html is not None
        assert "m-na" in html  # the NaN cell rendered as the three-state neutral class
        assert "m-ok" in html and "m-no" in html
        assert "2/3 bins accepted (67%)" in html  # denominator excludes the N/A cell

    def test_all_observed_denominator_matches_total(self):
        df = pd.DataFrame(
            {
                "segment": ["seg_a"] * 3,
                "octroi_bin": [0, 1, 2],
                "accepted": [1.0, 0.0, 1.0],
            }
        )
        html = _build_acceptance_strip_1d(df, "octroi_bin")
        assert "2/3 bins accepted (67%)" in html
        assert "m-na" not in html


# =============================================================================
# Tests: _build_cutoff_comparison_section
# =============================================================================


class TestBuildCutoffComparisonSection:
    def test_full_section(self, tmp_path):
        for seg in ["seg_a"]:
            data_dir = tmp_path / seg / "data"
            data_dir.mkdir(parents=True)
            _make_cutoff_df(segments=(seg,)).to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

        section = _build_cutoff_comparison_section(tmp_path, {"seg_a": {}})
        assert section is not None
        assert section.id == "cutoff-comparison"
        assert len(section.tables) == 2  # KPI table + matrices

    def test_returns_none_when_no_data(self, tmp_path):
        section = _build_cutoff_comparison_section(tmp_path, {"nonexistent": {}})
        assert section is None

    def test_no_matrices_without_accepted(self, tmp_path):
        data_dir = tmp_path / "seg_a" / "data"
        data_dir.mkdir(parents=True)
        _make_cutoff_df(segments=("seg_a",), include_accepted=False).to_csv(
            data_dir / "cutoff_summary_wide.csv", index=False
        )

        section = _build_cutoff_comparison_section(tmp_path, {"seg_a": {}})
        assert section is not None
        # Only KPI table, no matrices
        assert len(section.tables) == 1
        assert "Scenario KPI Summary" in section.tables[0]


# =============================================================================
# Tests: _build_portfolio_metric_table
# =============================================================================


def _make_consolidated_row(**overrides):
    """Build a synthetic consolidated-CSV row (pd.Series)."""
    data = {
        "group": "TOTAL",
        "period": "main",
        "scenario": "base",
        "actual_production": 50_000_000,
        "actual_risk_pct": 1.50,
        "actual_todu_30ever_h6": 1200,
        "actual_todu_amt_pile_h6": 400_000,
        "optimum_production": 55_000_000,
        "optimum_risk_pct": 1.30,
        "optimum_todu_30ever_h6": 1000,
        "optimum_todu_amt_pile_h6": 400_000,
        "swap_in_production": 8_000_000,
        "swap_in_risk_pct": 2.10,
        "swap_in_todu_30ever_h6": 300,
        "swap_in_todu_amt_pile_h6": 80_000,
        "swap_out_production": 3_000_000,
        "swap_out_risk_pct": 3.50,
        "swap_out_todu_30ever_h6": 500,
        "swap_out_todu_amt_pile_h6": 70_000,
        "production_delta_pct": 10.0,
        "production_ci_lower": 53_000_000,
        "production_ci_upper": 57_000_000,
        "risk_ci_lower": 1.10,
        "risk_ci_upper": 1.50,
    }
    data.update(overrides)
    return pd.Series(data)


class TestBuildPortfolioMetricTable:
    def test_renders_four_metric_rows(self):
        row = _make_consolidated_row()
        html = _build_portfolio_metric_table(row)
        assert "<thead>" in html
        assert "<tbody>" in html
        # 4 metric rows + 1 header = 5 <tr (some have class attrs)
        assert html.count("<tr") == 5
        for label in ["Actual", "Optimum", "Swap-in", "Swap-out"]:
            assert label in html

    def test_columns_present(self):
        row = _make_consolidated_row()
        html = _build_portfolio_metric_table(row)
        for col in ["Risk (%)", "Production (\u20ac)", "Production (%)", "Rejection Rate (%)"]:
            assert col in html
        for col in ["todu_30ever_h6", "todu_amt_pile_h6"]:
            assert col not in html
        for col in ["Production CI Lower", "Production CI Upper", "Risk CI Lower", "Risk CI Upper"]:
            assert col in html

    def test_production_delta_only_on_optimum(self):
        row = _make_consolidated_row(production_delta_pct=10.0)
        html = _build_portfolio_metric_table(row)
        assert "+10.0" in html
        # Only one occurrence (Optimum row), rest are dashes
        assert html.count("+10.0") == 1

    def test_cis_only_on_optimum(self):
        row = _make_consolidated_row()
        html = _build_portfolio_metric_table(row)
        assert "53,000,000" in html  # production CI lower
        assert "1.10" in html  # risk CI lower

    def test_handles_missing_columns(self):
        row = pd.Series({"group": "TOTAL", "scenario": "base"})
        html = _build_portfolio_metric_table(row)
        assert "<table" in html
        # All values should be dashes
        assert html.count("\u2014") >= 4 * 8  # 4 rows × 8 numeric columns


# =============================================================================
# Tests: _build_portfolio_summary_sections
# =============================================================================


class TestBuildPortfolioSummarySections:
    def test_supersegment_sections(self, tmp_path):
        rows = []
        for scen in ["pessimistic", "base", "optimistic"]:
            rows.append({"group": "supersegment_premium", "period": "main", "scenario": scen, "actual_production": 100})
            rows.append(
                {"group": "supersegment_standard", "period": "main", "scenario": scen, "actual_production": 200}
            )
            rows.append({"group": "TOTAL", "period": "main", "scenario": scen, "actual_production": 300})
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        ids = [s.id for s in sections]
        assert "portfolio-premium" in ids
        assert "portfolio-standard" in ids
        assert "portfolio-total" in ids
        # TOTAL should be last
        assert ids[-1] == "portfolio-total"

    def test_one_table_per_scenario(self, tmp_path):
        rows = []
        for scen in ["pessimistic", "base", "optimistic"]:
            rows.append({"group": "TOTAL", "period": "main", "scenario": scen, "actual_production": 100})
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        total_section = next(s for s in sections if s.id == "portfolio-total")
        # 3 scenarios = 3 tables (each table = heading + table HTML)
        assert len(total_section.tables) == 3
        assert "Pessimistic" in total_section.tables[0]
        assert "Base" in total_section.tables[1]
        assert "Optimistic" in total_section.tables[2]

    def test_main_and_mr_separate_sections(self, tmp_path):
        rows = [
            {"group": "TOTAL", "period": "main", "scenario": "base", "actual_production": 100},
            {"group": "TOTAL", "period": "mr", "scenario": "base", "actual_production": 50},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        ids = [s.id for s in sections]
        # Main and MR each get their own section
        assert "portfolio-total" in ids
        assert "mr-portfolio-total" in ids
        # Main sections come before MR sections
        assert ids.index("portfolio-total") < ids.index("mr-portfolio-total")

    def test_mr_section_titles(self, tmp_path):
        rows = [
            {"group": "supersegment_premium", "period": "main", "scenario": "base", "actual_production": 100},
            {"group": "supersegment_premium", "period": "mr", "scenario": "base", "actual_production": 50},
            {"group": "TOTAL", "period": "main", "scenario": "base", "actual_production": 100},
            {"group": "TOTAL", "period": "mr", "scenario": "base", "actual_production": 50},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        titles = {s.id: s.title for s in sections}
        assert "Portfolio Summary" in titles["portfolio-premium"]
        assert "MR Validation" in titles["mr-portfolio-premium"]
        assert "Portfolio Summary" in titles["portfolio-total"]
        assert "MR Validation" in titles["mr-portfolio-total"]

    def test_mr_only_when_data_exists(self, tmp_path):
        rows = [
            {"group": "TOTAL", "period": "main", "scenario": "base", "actual_production": 100},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        ids = [s.id for s in sections]
        assert "portfolio-total" in ids
        assert not any(i.startswith("mr-") for i in ids)

    def test_standalone_segments_without_supersegments(self, tmp_path):
        rows = [
            {"group": "segment_a", "period": "main", "scenario": "base", "actual_production": 100},
            {"group": "segment_b", "period": "main", "scenario": "base", "actual_production": 200},
            {"group": "TOTAL", "period": "main", "scenario": "base", "actual_production": 300},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        ids = [s.id for s in sections]
        assert "portfolio-a" in ids
        assert "portfolio-b" in ids
        assert "portfolio-total" in ids

    def test_returns_empty_for_missing_file(self, tmp_path):
        assert _build_portfolio_summary_sections(tmp_path / "nonexistent.csv") == []

    def test_returns_empty_for_empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("group,scenario\n")
        assert _build_portfolio_summary_sections(path) == []

    def test_badge_classes_in_output(self, tmp_path):
        rows = [
            {"group": "TOTAL", "period": "main", "scenario": "pessimistic", "actual_production": 100},
            {"group": "TOTAL", "period": "main", "scenario": "base", "actual_production": 100},
            {"group": "TOTAL", "period": "main", "scenario": "optimistic", "actual_production": 100},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        sections = _build_portfolio_summary_sections(tmp_path / "consolidated_risk_production.csv")
        total_section = next(s for s in sections if s.id == "portfolio-total")
        all_html = "".join(total_section.tables)
        assert "badge-pessimistic" in all_html
        assert "badge-base" in all_html
        assert "badge-optimistic" in all_html


class TestPortfolioGroupList:
    """#45: standalone segments must appear alongside supersegments, not vanish."""

    def test_standalone_segments_listed_alongside_supersegments(self):
        # A reporting supersegment (+ its member detail row) AND a standalone
        # segment outside every supersegment, plus TOTAL.
        df = pd.DataFrame(
            {
                "group": [
                    "supersegment_ecom",
                    "ecom/known",  # member detail row
                    "segment_standalone_pl",  # standalone — must not be dropped
                    "TOTAL",
                ]
            }
        )
        groups = _portfolio_group_list(df)
        names = [g for g, _ in groups]
        assert "supersegment_ecom" in names
        assert "segment_standalone_pl" in names  # was dropped when any supersegment existed
        assert names[-1] == "TOTAL"  # TOTAL stays last

    def test_no_supersegments_lists_all_standalone(self):
        df = pd.DataFrame({"group": ["segment_a", "segment_b", "TOTAL"]})
        names = [g for g, _ in _portfolio_group_list(df)]
        assert names == ["segment_a", "segment_b", "TOTAL"]
