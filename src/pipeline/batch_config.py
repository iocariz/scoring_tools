"""Batch configuration: load ``config.toml`` / ``segments.toml``, merge per-segment overrides, inject learned
bin edges, order segments by cutoff dependencies and write each segment's ``config_segment.toml``
(the persisted run config that consolidation, resimulation, reproducibility and the policy registry read).
"""

import copy
import re
import tomllib
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.utils import resolve_modelling_supersegment, resolve_reporting_supersegment


def load_base_config(config_path: str = "config.toml") -> dict[str, Any]:
    """Load the base configuration from config.toml."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("preprocessing", {})


_VALID_SEGMENT_NAME = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def validate_segment_name(name: str) -> None:
    """Validate segment/supersegment name to prevent path traversal."""
    if not _VALID_SEGMENT_NAME.match(name):
        raise ValueError(
            f"Invalid segment name {name!r}: must match [a-zA-Z0-9_.-]. "
            f"Path separators and special characters are not allowed."
        )


def load_segments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load segment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("segments", {})


def load_supersegments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load supersegment configurations from segments.toml.

    Merges names from ``supersegments``, ``modelling_supersegments``, and
    ``reporting_supersegments`` so that validation sees all defined supersegments.
    """
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    merged: dict[str, dict[str, Any]] = {}
    for key in ("supersegments", "modelling_supersegments", "reporting_supersegments"):
        merged.update(config.get(key, {}))
    return merged


def load_reporting_supersegments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load only reporting supersegment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("reporting_supersegments", {})


def create_output_directories(base_output_dir: Path) -> dict[str, Path]:
    """Create output directory structure for a segment."""
    dirs = {
        "root": base_output_dir,
        "images": base_output_dir / "images",
        "models": base_output_dir / "models",
        "data": base_output_dir / "data",
        "logs": base_output_dir / "logs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def merge_configs(base_config: dict[str, Any], segment_config: dict[str, Any]) -> dict[str, Any]:
    """Merge base config with segment-specific overrides (recursive for nested dicts)."""
    merged = copy.deepcopy(base_config)
    for key, value in segment_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _segment_pins_bin_edges(segment_config: dict[str, Any], var_name: str) -> bool:
    """True if the SEGMENT config explicitly pins ``bin_edges`` for *var_name*
    (form ``[preprocessing.bins.<var>] bin_edges = [...]``, i.e. ``bins.<var>.bin_edges``)."""
    bc = segment_config.get("bins", {}).get(var_name, {})
    return isinstance(bc, dict) and bool(bc.get("bin_edges")) and len(bc["bin_edges"]) >= 2


def _inject_learned_bin_edges(
    merged_config: dict[str, Any],
    segment_config: dict[str, Any],
    segment_name: str,
    global_bin_edges: dict[str, list[float]] | None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None,
) -> dict[str, Any]:
    """Inject learned bin edges into ``merged_config['bins']`` — global first, then
    reporting-supersegment edges override global — but NEVER clobber a variable whose edges the
    SEGMENT explicitly pinned (#60). Precedence: segment pin > supersegment > global. Logs which
    source won so a segment running on a different grid than configured is never silent."""
    bins_section = merged_config.setdefault("bins", {})

    if global_bin_edges:
        for var_name, edges in global_bin_edges.items():
            if var_name not in bins_section:
                continue
            if _segment_pins_bin_edges(segment_config, var_name):
                logger.info(
                    f"Segment '{segment_name}': keeping segment-pinned bin_edges for '{var_name}' "
                    "(NOT injecting the global learned edges)."
                )
                continue
            bins_section[var_name]["bin_edges"] = edges

    if supersegment_bin_edges:
        reporting_ss = resolve_reporting_supersegment(segment_config)
        if reporting_ss and reporting_ss in supersegment_bin_edges:
            for var_name, edges in supersegment_bin_edges[reporting_ss].items():
                if var_name not in bins_section:
                    continue
                if _segment_pins_bin_edges(segment_config, var_name):
                    logger.info(
                        f"Segment '{segment_name}': keeping segment-pinned bin_edges for '{var_name}' "
                        f"(NOT injecting supersegment '{reporting_ss}' edges)."
                    )
                    continue
                bins_section[var_name]["bin_edges"] = edges
                logger.info(f"Using supersegment '{reporting_ss}' bin edges for '{var_name}' (overriding global)")

    return merged_config


def validate_supersegment_edge_consistency(
    base_config: dict[str, Any],
    segments: dict[str, dict[str, Any]],
    global_bin_edges: dict[str, list[float]] | None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None,
) -> None:
    """Batch-start pre-flight (#40c): a segment that REUSES a modelling-supersegment's shared risk
    model must resolve its INFERENCE-variable bin edges to the SAME edges the shared model trains on
    (the global learned / base-config fixed edges). A reporting-supersegment override or a
    segment-level pin that moves an inference variable's edges would apply the shared model on a
    different grid than it was trained on → wrong per-cell risk. Raise at batch start (fail fast)
    rather than only surfacing it mid-run per member (the #40 a/b reuse validation catches it there).

    GRID-only variables (e.g. ``income_bin``) are exempt: they are optimization dimensions, not risk-
    model features, so a per-supersegment income split is legitimate and does not misapply the model.
    """
    inf_vars = base_config.get("inference_variables") or base_config.get("variables") or []
    bins_cfg = base_config.get("bins", {})

    def shared_edges(var: str) -> list[float] | None:
        """The edges the shared model trains on: global learned, else the base config's fixed edges."""
        if global_bin_edges and var in global_bin_edges:
            return [float(e) for e in global_bin_edges[var]]
        bc = bins_cfg.get(var, {})
        return [float(e) for e in bc["bin_edges"]] if bc.get("bin_edges") else None

    problems: list[str] = []
    for name, seg_cfg in segments.items():
        if not resolve_modelling_supersegment(seg_cfg):
            continue  # trains its own model → no shared-model grid to match
        reporting_ss = resolve_reporting_supersegment(seg_cfg)
        for var in inf_vars:
            base_edges = shared_edges(var)
            member_edges = base_edges
            if _segment_pins_bin_edges(seg_cfg, var):
                member_edges = [float(e) for e in seg_cfg["bins"][var]["bin_edges"]]
            elif reporting_ss and supersegment_bin_edges and reporting_ss in supersegment_bin_edges:
                ov = supersegment_bin_edges[reporting_ss].get(var)
                if ov is not None:
                    member_edges = [float(e) for e in ov]
            if base_edges is not None and member_edges is not None and member_edges != base_edges:
                problems.append(
                    f"segment '{name}' inference variable '{var}': edges {member_edges} != "
                    f"shared-model edges {base_edges}"
                )

    if problems:
        raise ValueError(
            "Supersegment model/member INFERENCE bin-edge inconsistency (#40c) — a shared "
            "modelling-supersegment model would be applied on a different grid than it was trained on:\n  "
            + "\n  ".join(problems)
            + "\nGive these segments their own model (remove modelling_supersegment) or align the edges."
        )


def write_segment_config(config: dict[str, Any], output_dir: Path) -> Path:
    """Write a temporary config file for this segment run."""
    import tomli_w

    config_path = output_dir / "config_segment.toml"

    # Keep data source stable if process working directory changes.
    config_for_dump = config.copy()
    data_path = config_for_dump.get("data_path")
    if isinstance(data_path, str):
        data_path_obj = Path(data_path)
        if not data_path_obj.is_absolute():
            config_for_dump["data_path"] = str((Path.cwd() / data_path_obj).resolve())

    # Wrap config in preprocessing section
    full_config = {"preprocessing": config_for_dump}

    with open(config_path, "wb") as f:
        tomli_w.dump(full_config, f)

    return config_path


def _topological_sort_segments(
    segments: dict[str, dict[str, Any]],
) -> list[str]:
    """Sort segment names respecting cutoff_floor_segment dependencies.

    Segments with no dependency come first. Segments that depend on another
    come after their dependency. Raises ValueError on circular dependencies.

    Args:
        segments: Segment configurations keyed by segment name.

    Returns:
        Ordered list of segment names.
    """
    # Build dependency map: seg_name -> floor_segment_name or None
    deps: dict[str, str | None] = {}
    for seg_name, seg_config in segments.items():
        deps[seg_name] = seg_config.get("cutoff_floor_segment")

    ordered: list[str] = []
    resolved: set[str] = set()
    visiting: set[str] = set()

    def _visit(name: str) -> None:
        if name in resolved:
            return
        if name in visiting:
            raise ValueError(f"Circular cutoff_floor_segment dependency involving '{name}'")
        if name not in deps:
            # Dependency references a segment not in the current batch — treat as resolved
            resolved.add(name)
            return
        visiting.add(name)
        dep = deps[name]
        if dep is not None:
            _visit(dep)
        visiting.discard(name)
        resolved.add(name)
        ordered.append(name)

    for seg_name in segments:
        _visit(seg_name)

    return ordered
