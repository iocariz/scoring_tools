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
        """Consolidated report with multiple segment dirs."""
        # Create segment dirs with cutoff_summary_wide.csv
        for seg in ["seg_a", "seg_b"]:
            data_dir = tmp_path / seg / "data"
            data_dir.mkdir(parents=True)
            df = pd.DataFrame({"var1": [1, 2], "var2": [3, 4]})
            df.to_csv(data_dir / "cutoff_summary_wide.csv", index=False)

        context = build_consolidated_report(tmp_path, segments={"seg_a": {}, "seg_b": {}}, supersegments={})

        section_ids = [s.id for s in context.sections]
        assert "cutoff-comparison" in section_ids

        # Should have tables for both segments
        cutoff_section = next(s for s in context.sections if s.id == "cutoff-comparison")
        assert len(cutoff_section.tables) == 2
