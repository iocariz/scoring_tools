"""Shared data-loading helpers for the dashboards (R2 todo #64).

Before this module existed, ``dashboard.py``, ``gradio_dashboard.py``, and
``interactive_allocator.py`` each redefined near-identical versions of
``get_data_dir`` / ``get_images_dir`` / ``get_available_segments`` /
``get_scenarios`` / ``_resolve_supersegment`` / ``parse_coefficients``. The
audit flagged ~15–20 KB of drift-prone duplication.

Design notes:
  - Every function takes an explicit ``output_base: Path`` parameter rather
    than reading a module-level constant. Each dashboard keeps its own
    ``OUTPUT_BASE`` (mutated at startup from CLI args) and passes it in.
  - Functions are pure: no logging, no I/O other than filesystem reads,
    no side effects. Makes them trivial to unit-test.
  - Static-URL builders (``/static/...``) stay in dashboard.py because
    they're Flask-routing-specific; this module only deals with filesystem
    paths.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

# Scenario ordering used by get_scenarios() for deterministic sort.
# Named constant so both dashboards can share it.
SCENARIO_ORDER: dict[str, int] = {"pessimistic": 0, "base": 1, "optimistic": 2}


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def get_data_dir(output_base: Path, segment: str | None = None) -> Path:
    """Return the ``data/`` directory for *segment* (or the root in single-run mode)."""
    if segment:
        return output_base / segment / "data"
    if (output_base / "data").exists():
        return output_base / "data"
    return Path("data")


def get_images_dir(output_base: Path, segment: str | None = None) -> Path:
    """Return the ``images/`` directory for *segment* (or the root in single-run mode)."""
    if segment:
        return output_base / segment / "images"
    if (output_base / "images").exists():
        return output_base / "images"
    return Path("images")


def get_models_dir(output_base: Path, segment: str | None = None) -> Path:
    """Return the ``models/`` directory for *segment*, with supersegment fallback.

    If the segment has no local models dir but references a supersegment
    that does, the supersegment's models dir is returned instead. This
    matches how ``run_batch.py`` saves supersegment-shared models under
    ``output/_supersegment_<ss>/models/``.
    """
    if segment:
        direct = output_base / segment / "models"
        if direct.exists() and any(direct.iterdir()):
            return direct
        superseg = resolve_supersegment(segment, output_base=output_base)
        if superseg:
            ss_models = output_base / f"_supersegment_{superseg}" / "models"
            if ss_models.exists():
                return ss_models
        return direct
    if (output_base / "models").exists():
        return output_base / "models"
    return Path("models")


# ---------------------------------------------------------------------------
# Segment / scenario discovery
# ---------------------------------------------------------------------------


def get_available_segments(output_base: Path) -> list[str]:
    """Enumerate directories under *output_base* that look like valid segments.

    A directory qualifies if: it is not a dot/underscore-prefixed hidden
    folder, and it contains a ``data/`` subdir. Sorted alphabetically.
    """
    if not output_base.exists():
        return []
    return sorted(
        d.name
        for d in output_base.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and (d / "data").exists()
    )


def get_scenarios(output_base: Path, segment: str | None = None) -> list[str]:
    """Parse scenario names from the CSV files produced by the pipeline.

    Names are either canonical (``pessimistic`` / ``base`` / ``optimistic``)
    or the legacy numeric format (``1.2`` etc.). Returned list is sorted
    by :data:`SCENARIO_ORDER` (pessimistic, base, optimistic) with numeric
    scenarios appended alphabetically after.
    """
    data_dir = get_data_dir(output_base, segment)
    if not data_dir.exists():
        return []

    scenarios: list[str] = []
    for f in data_dir.glob("risk_production_summary_table_*.csv"):
        if "_mr" in f.name:
            # MR files are secondary; discover from the main files only.
            continue
        m = re.search(r"risk_production_summary_table_(pessimistic|base|optimistic)\.csv", f.name)
        if m:
            scenarios.append(m.group(1))
            continue
        m = re.search(r"risk_production_summary_table_(\d+\.\d+)\.csv", f.name)
        if m:
            scenarios.append(m.group(1))

    # Unsuffixed base file fallback.
    if (data_dir / "risk_production_summary_table.csv").exists() and "base" not in scenarios:
        scenarios.append("base")

    return sorted(scenarios, key=lambda x: (SCENARIO_ORDER.get(x, 99), x))


def get_scenario_paths(
    output_base: Path,
    scenario: str,
    segment: str | None = None,
) -> dict[str, Path]:
    """Return the filesystem paths for a scenario's four main CSV artifacts.

    Static-URL path resolution (``/static/<segment>/risk_production...html``)
    is dashboard-specific and NOT included here — that belongs in
    ``dashboard.py``. This module stays framework-agnostic.

    For the base scenario, falls back to the unsuffixed filename when the
    suffixed version is missing.
    """
    data_dir = get_data_dir(output_base, segment)
    suffix = f"_{scenario}"
    paths = {
        "main_csv": data_dir / f"risk_production_summary_table{suffix}.csv",
        "mr_csv": data_dir / f"risk_production_summary_table_mr{suffix}.csv",
        "cutoffs": data_dir / f"optimal_solution{suffix}.csv",
        "summary_data": data_dir / f"data_summary_desagregado{suffix}.csv",
    }
    if scenario == "base":
        fallbacks = {
            "main_csv": data_dir / "risk_production_summary_table.csv",
            "mr_csv": data_dir / "risk_production_summary_table_mr.csv",
            "cutoffs": data_dir / "optimal_solution.csv",
            "summary_data": data_dir / "data_summary_desagregado.csv",
        }
        for key in paths:
            if not paths[key].exists() and fallbacks[key].exists():
                paths[key] = fallbacks[key]
    return paths


# ---------------------------------------------------------------------------
# Supersegment resolution
# ---------------------------------------------------------------------------


def resolve_supersegment(segment: str | None, output_base: Path | None = None) -> str | None:
    """Look up the modelling (or plain) supersegment name for *segment*.

    Searches both the working-directory ``segments.toml`` and (if
    *output_base* is provided) ``<output_base>/segments.toml`` — the
    latter is useful when dashboards open a previously-generated output
    tree that has its own snapshot of segments.toml.

    Returns ``None`` if the segment is unknown or no supersegment is
    configured. Resolution order matches
    :func:`src.utils.resolve_modelling_supersegment`:
    ``modelling_supersegment`` > ``supersegment`` > ``None``.
    """
    if not segment:
        return None

    candidate_paths: list[Path] = [Path("segments.toml")]
    if output_base is not None:
        candidate_paths.append(output_base / "segments.toml")

    for toml_path in candidate_paths:
        if not toml_path.exists():
            continue
        try:
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            seg_config = data.get("segments", {})
            cfg = seg_config.get(segment)
            if cfg is not None:
                return cfg.get("modelling_supersegment") or cfg.get("supersegment")
        except (OSError, tomllib.TOMLDecodeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Model artefact parsing
# ---------------------------------------------------------------------------


def parse_coefficients(model_dir: str | Path) -> list[dict[str, Any]]:
    """Parse ``model_summary.txt`` under *model_dir* for feature coefficients.

    Returns a list of ``{"feature": str, "coefficient": float}`` dicts in
    the order they appear in the summary. An empty list is returned if
    the summary file is missing, unreadable, or has no coefficient section.

    The ``MODEL COEFFICIENTS`` section is delimited by
    ``---``-rules and terminated by the next all-caps header.
    """
    summary_path = Path(model_dir) / "model_summary.txt"
    if not summary_path.exists():
        return []

    coefficients: list[dict[str, Any]] = []
    in_section = False
    try:
        text = summary_path.read_text()
    except OSError:
        return []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "MODEL COEFFICIENTS" in line:
            in_section = True
            continue
        if not in_section:
            continue
        if line.startswith("---") or not line:
            continue
        if line.isupper() and ":" not in line:
            # Next section header — stop.
            break
        if ":" not in line:
            continue
        parts = line.rsplit(":", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        try:
            value = float(parts[1].strip())
        except ValueError:
            continue
        coefficients.append({"feature": name, "coefficient": value})
    return coefficients
