"""Tests for HTML report generation (src/reporting.py & src/pipeline/reporting.py)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.config import OutputPaths, PreprocessingSettings
from src.reporting import (
    ReportContext,
    ReportSection,
    _build_acceptance_matrices,
    _build_cutoff_comparison_section,
    _build_scenario_kpi_table,
    _detect_variable_cols,
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
                "optimum_production": [250_000, 130_000],
            }
        )
        df.to_csv(tmp_path / "consolidated_risk_production.csv", index=False)

        context = build_consolidated_report(tmp_path, segments={"seg_a": {}}, supersegments={})
        assert isinstance(context, ReportContext)
        assert context.title == "Consolidated Report"

        section_ids = [s.id for s in context.sections]
        assert "portfolio-summary" in section_ids

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
        # 2 segments × 3 scenarios = 6 data rows (header row has inline style)
        assert html.count("<tr>") == 6
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

    def test_acceptance_rate(self):
        df = _make_cutoff_df(scenarios=("base",))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        assert "Acc. Rate (%)" in html

    def test_without_accepted_col(self):
        df = _make_cutoff_df(include_accepted=False, scenarios=("base",))
        html = _build_scenario_kpi_table(df, ["octroi_bin", "efx_bin"])
        assert "report-table" in html
        # All cells accepted → rate should be 100.0%
        assert "100.0" in html

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
        # Contains checkmarks and crosses
        assert "&#10003;" in html or "&#10007;" in html

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
        assert "#2ECC71" in html or "#FADBD8" in html


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
