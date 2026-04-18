"""Rewrite legacy ``octroi_bins`` / ``efx_bins`` config into the modern
``[preprocessing.bins.*]`` TOML blocks (R2, todo #66).

The 2-variable fields ``octroi_bins`` and ``efx_bins`` with their hardcoded
source columns (``score_rf`` / ``risk_score_rf``) predate the N-variable
pipeline. They still work but emit a ``DeprecationWarning`` at config
load. This script rewrites a config.toml / segments.toml in place so the
warning goes away.

Usage
-----
    # Dry-run: show what would change, write nothing
    uv run python scripts/migrate_legacy_bins.py config.toml --dry-run

    # Apply in place (makes a .bak next to the file)
    uv run python scripts/migrate_legacy_bins.py config.toml

    # Multiple files in one call
    uv run python scripts/migrate_legacy_bins.py config.toml segments.toml

The rewrite preserves all non-binning keys and comments. It drops
``octroi_bins`` / ``efx_bins`` from ``[preprocessing]`` and inserts
``[preprocessing.bins.VAR]`` sub-tables with the same edges plus the
``source_col`` mapping the old code implied:

    octroi_bins → bins.sc_octroi_new_clus (source_col = "score_rf")
    efx_bins    → bins.new_efx_clus       (source_col = "risk_score_rf")

If the config's ``variables`` list names different first-two variables,
those are used instead (positional mapping), matching the legacy
``_auto_populate_bins`` fallback in ``src/config.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_POSITIONAL_SOURCE_COLS = ("score_rf", "risk_score_rf")
_NAME_TO_SOURCE = {
    "sc_octroi_new_clus": "score_rf",
    "new_efx_clus": "risk_score_rf",
}


def _read_toml(path: Path) -> dict:
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8"))


def _write_toml_patch(original_text: str, var_name: str, source_col: str, edges: list[float]) -> str:
    """Append a ``[preprocessing.bins.VAR]`` table to ``original_text``.

    Uses a plain-text append rather than round-tripping through a TOML
    writer so comments and formatting are preserved.
    """
    edges_str = ", ".join(_fmt_edge(e) for e in edges)
    block = (
        f"\n[preprocessing.bins.{var_name}]\n"
        f'source_col = "{source_col}"\n'
        f'output_col = "{var_name}"\n'
        f"bin_edges = [{edges_str}]\n"
    )
    return original_text + block


def _fmt_edge(e: float | int) -> str:
    if e == float("-inf"):
        return '"-inf"'
    if e == float("inf"):
        return '"inf"'
    if isinstance(e, int):
        return str(e)
    return repr(float(e))


def _strip_legacy_lines(text: str) -> str:
    """Remove ``octroi_bins = ...`` / ``efx_bins = ...`` lines (single-line only)."""
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("octroi_bins") or stripped.startswith("efx_bins"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def _resolve_var_and_source(variables: list[str], index: int) -> tuple[str, str]:
    """Return (var_name, source_col) for positional slot *index*."""
    if index < len(variables):
        var = variables[index]
        source = _NAME_TO_SOURCE.get(var, _POSITIONAL_SOURCE_COLS[index])
        return var, source
    # Fall back to the legacy default variable names if `variables` is short.
    default_names = ["sc_octroi_new_clus", "new_efx_clus"]
    return default_names[index], _POSITIONAL_SOURCE_COLS[index]


def migrate_file(path: Path, dry_run: bool) -> bool:
    """Migrate a single file. Returns True if changes were made (or would be)."""
    original_text = path.read_text(encoding="utf-8")
    parsed = _read_toml(path)

    prep = parsed.get("preprocessing", parsed)
    octroi = prep.get("octroi_bins")
    efx = prep.get("efx_bins")
    variables = prep.get("variables") or []

    has_bins_dict = "bins" in prep and isinstance(prep["bins"], dict) and prep["bins"]

    if not (octroi or efx):
        print(f"[skip]    {path}: no legacy octroi_bins/efx_bins found")
        return False

    if has_bins_dict:
        print(
            f"[skip]    {path}: legacy fields present alongside a [preprocessing.bins.*] "
            f"block — migration ambiguous, resolve by hand"
        )
        return False

    new_text = _strip_legacy_lines(original_text)
    added: list[str] = []

    if octroi:
        var, src = _resolve_var_and_source(variables, index=0)
        new_text = _write_toml_patch(new_text, var, src, octroi)
        added.append(f"[preprocessing.bins.{var}] (source_col={src})")

    if efx:
        var, src = _resolve_var_and_source(variables, index=1)
        new_text = _write_toml_patch(new_text, var, src, efx)
        added.append(f"[preprocessing.bins.{var}] (source_col={src})")

    print(f"[migrate] {path}:")
    for a in added:
        print(f"    + {a}")
    print("    - octroi_bins / efx_bins removed")

    if dry_run:
        return True

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    path.write_text(new_text, encoding="utf-8")
    print(f"    wrote {path} (backup: {backup.name})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="TOML config files to rewrite.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes, write nothing.")
    args = parser.parse_args()

    any_changes = False
    for path in args.files:
        if not path.is_file():
            print(f"[error]   {path} is not a file", file=sys.stderr)
            return 2
        if migrate_file(path, dry_run=args.dry_run) and not any_changes:
            any_changes = True

    if args.dry_run and any_changes:
        print("\nDry run — re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
