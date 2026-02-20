"""
Pipeline orchestration layer for HTML report generation.

Thin wrappers called from ``main.py`` (single segment) and ``run_batch.py``
(multi-segment).  All heavy lifting is in :mod:`src.reporting`.
"""

from pathlib import Path

from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.reporting import (
    build_consolidated_report,
    build_segment_report,
    render_report,
)


def generate_segment_report(
    settings: PreprocessingSettings,
    output: OutputPaths,
    scenarios: list[str],
) -> Path | None:
    """Generate the single-segment HTML report.

    Returns the path to the written report, or *None* on failure.
    """
    try:
        context = build_segment_report(output, settings, scenarios)
        if not context.sections:
            logger.info(f"[{settings.segment_filter}] No report sections — skipping report")
            return None
        report_path = render_report(context, output.segment_report_html)
        return report_path
    except Exception as e:
        logger.warning(f"[{settings.segment_filter}] Report generation failed: {e}")
        return None


def generate_batch_reports(
    output_base: str,
    segments: dict,
    supersegments: dict,
    segment_results: dict[str, bool],
) -> dict[str, Path]:
    """Generate per-segment and consolidated HTML reports.

    Args:
        output_base: Root output directory (e.g. ``"output"``).
        segments: Segment configurations from ``segments.toml``.
        supersegments: Supersegment configurations.
        segment_results: Mapping of segment name to success boolean.

    Returns:
        Mapping of report name to path for all successfully generated reports.
    """
    reports: dict[str, Path] = {}
    output_base_path = Path(output_base)

    # Per-segment reports
    successful = [name for name, ok in segment_results.items() if ok]
    for seg_name in successful:
        seg_dir = output_base_path / seg_name
        seg_output = OutputPaths(base_dir=seg_dir)

        # Try to load settings from the segment's saved config
        config_path = seg_dir / "config_segment.toml"
        if not config_path.exists():
            logger.debug(f"[{seg_name}] No config_segment.toml — skipping report")
            continue

        try:
            settings = PreprocessingSettings.from_toml(str(config_path))
        except Exception as e:
            logger.warning(f"[{seg_name}] Cannot load config for report: {e}")
            continue

        # Detect scenarios from existing files
        data_dir = seg_dir / "data"
        scenarios = _detect_scenarios(data_dir)

        try:
            context = build_segment_report(seg_output, settings, scenarios)
            if context.sections:
                path = render_report(context, seg_output.segment_report_html)
                reports[seg_name] = path
        except Exception as e:
            logger.warning(f"[{seg_name}] Segment report failed: {e}")

    # Consolidated report
    try:
        context = build_consolidated_report(output_base, segments, supersegments)
        if context.sections:
            consolidated_path = output_base_path / "consolidated_report.html"
            path = render_report(context, consolidated_path, template_name="consolidated_report.html")
            reports["consolidated"] = path
    except Exception as e:
        logger.warning(f"Consolidated report failed: {e}")

    return reports


def _detect_scenarios(data_dir: Path) -> list[str]:
    """Auto-detect scenario names from ``risk_production_summary_table_*.csv`` files."""
    scenarios = []
    if not data_dir.exists():
        return ["base"]

    for f in sorted(data_dir.glob("risk_production_summary_table_*.csv")):
        if "_mr_" in f.name or f.name.endswith("_mr.csv"):
            continue
        # Extract suffix: risk_production_summary_table_{scenario}.csv
        stem = f.stem  # e.g. "risk_production_summary_table_base"
        prefix = "risk_production_summary_table_"
        if stem.startswith(prefix):
            scenario = stem[len(prefix) :]
            if scenario and scenario not in scenarios:
                scenarios.append(scenario)

    return scenarios or ["base"]
