"""Policy registry + champion/challenger comparison (phase 1).

A *policy* is a frozen base-scenario accepted-cell set for one segment, plus the bin
edges and provenance needed to apply and reproduce it. The **registry** is a committed,
append-only per-segment JSON (``reports/policy_registry/<segment>.json``) carrying one
``champion_policy_id`` — the policy currently designated live. A *challenger* (typically
the latest run's base policy) is scored against the champion on a common, matured
out-of-time cohort, reusing the M4 backtest machinery (``apply_policy`` + ``_realized_metrics``)
and the M5 headline/provenance (``extract_headline``).

Locked design decisions:
- **One champion per segment**, championing the base-scenario accepted set (matches what
  is actually deployed; only the base scenario writes ``accepted_cells_*.csv``).
- **Committed, git-tracked registry** under ``reports/policy_registry/`` (mirrors the M5
  ``reports/validation/reference/`` precedent — diffable in PRs, replayable from a clone).
- **Comparison cohort = the auto-derived matured holdout** (the same window the M4 backtest
  uses), override-able via explicit holdout dates by the runner.

The accepted set is a set of binned coordinate tuples — robust to the positional
``acceptance_mask`` reordering across runs. The risk verdict is **noise-aware** (reuses the
backtest's CI-overlap + ``MIN_DEFAULTS_FOR_DRIFT`` rule): a challenger is only called BETTER
/ WORSE when its realized-risk CI is fully separated from the champion's; otherwise
INCONCLUSIVE. Production is reported alongside (a human weighs both) — it does not drive the
verdict. This module is pure/IO-light; ``run_policy_registry.py`` drives data loading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from src.backtest import (
    MIN_DEFAULTS_FOR_DRIFT,
    _realized_metrics,
    apply_policy,
    load_frozen_policy,
)
from src.reproducibility import REFERENCE_DIR, extract_headline

REGISTRY_DIR = Path("reports/policy_registry")
MRM_SIGNOFF = "reports/validation/mrm_signoff.md"

Cell = tuple[float, ...]


# --------------------------------------------------------------------------- #
# Policy entry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyEntry:
    """A frozen, registrable policy: the accepted set + how to reproduce + evidence pointers."""

    policy_id: str
    segment: str
    scenario: str
    variables: tuple[str, ...]
    bin_edges: dict  # var -> list[float] (frozen edges → reproducible binning)
    accepted_cells: tuple[Cell, ...]  # sorted coordinate tuples
    accepted_set_hash: str
    headline: dict  # risk_pct, production_eur, n_accepted_cells
    provenance: dict  # data_sha256, git_commit, config_hash, created
    evidence: dict  # pointers (lineage, headline_reference, mrm_signoff)
    status: str = "challenger"  # champion | challenger
    effective_from: str | None = None
    signed_off_by: str | None = None

    def accepted_set(self) -> set[Cell]:
        return {tuple(c) for c in self.accepted_cells}

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "segment": self.segment,
            "scenario": self.scenario,
            "variables": list(self.variables),
            "bin_edges": {k: list(v) for k, v in self.bin_edges.items()},
            "accepted_cells": [list(c) for c in self.accepted_cells],
            "accepted_set_hash": self.accepted_set_hash,
            "headline": dict(self.headline),
            "provenance": dict(self.provenance),
            "evidence": dict(self.evidence),
            "status": self.status,
            "effective_from": self.effective_from,
            "signed_off_by": self.signed_off_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PolicyEntry:
        return cls(
            policy_id=d["policy_id"],
            segment=d["segment"],
            scenario=d.get("scenario", "base"),
            variables=tuple(d.get("variables", [])),
            bin_edges={k: list(v) for k, v in (d.get("bin_edges") or {}).items()},
            accepted_cells=tuple(tuple(float(x) for x in c) for c in d.get("accepted_cells", [])),
            accepted_set_hash=d.get("accepted_set_hash", ""),
            headline=dict(d.get("headline") or {}),
            provenance=dict(d.get("provenance") or {}),
            evidence=dict(d.get("evidence") or {}),
            status=d.get("status", "challenger"),
            effective_from=d.get("effective_from"),
            signed_off_by=d.get("signed_off_by"),
        )


def build_policy_entry(
    seg_output_dir: Path,
    settings,
    suffix: str = "_base",
    created: str | None = None,
) -> PolicyEntry:
    """Assemble a :class:`PolicyEntry` from a completed run (reuses M5 ``extract_headline``)."""
    seg_data_dir = Path(seg_output_dir) / "data"
    variables = list(settings.variables)
    headline = extract_headline(seg_output_dir, settings, suffix)
    accepted_set, _ = load_frozen_policy(seg_data_dir, variables, suffix)
    bin_edges = {var: list(bc.bin_edges) for var, bc in (settings.bins or {}).items() if bc.bin_edges}

    return PolicyEntry(
        policy_id=f"{headline.segment}-{headline.accepted_set_hash[:8]}",
        segment=headline.segment,
        scenario=headline.scenario,
        variables=tuple(variables),
        bin_edges=bin_edges,
        accepted_cells=tuple(sorted(accepted_set)),
        accepted_set_hash=headline.accepted_set_hash,
        headline={
            "risk_pct": headline.risk_pct,
            "production_eur": headline.production_eur,
            "n_accepted_cells": headline.n_accepted_cells,
        },
        provenance={
            "data_sha256": headline.data_sha256,
            "git_commit": headline.git_commit,
            "config_hash": headline.config_hash,
            "created": created,
        },
        evidence={
            "lineage": str(seg_data_dir / "run_lineage.json"),
            "headline_reference": str(REFERENCE_DIR / f"{headline.segment}_headline.json"),
            "mrm_signoff": MRM_SIGNOFF,
        },
        status="challenger",
    )


# --------------------------------------------------------------------------- #
# Registry I/O (committed, append-only)
# --------------------------------------------------------------------------- #


def registry_path(segment: str, registry_dir: Path = REGISTRY_DIR) -> Path:
    return Path(registry_dir) / f"{segment}.json"


def load_registry(segment: str, registry_dir: Path = REGISTRY_DIR) -> dict:
    """Load a segment's registry, or an empty skeleton if it does not exist yet."""
    path = registry_path(segment, registry_dir)
    if not path.exists():
        return {"segment": segment, "champion_policy_id": None, "policies": []}
    try:
        reg = json.loads(path.read_text())
    except Exception:
        logger.warning(f"Could not parse registry {path}; treating as empty.")
        return {"segment": segment, "champion_policy_id": None, "policies": []}
    reg.setdefault("segment", segment)
    reg.setdefault("champion_policy_id", None)
    reg.setdefault("policies", [])
    return reg


def save_registry(registry: dict, registry_dir: Path = REGISTRY_DIR) -> Path:
    path = registry_path(registry["segment"], registry_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, default=str))
    return path


def _resync_statuses(registry: dict) -> None:
    """Set exactly the champion policy's status to 'champion', the rest to 'challenger'."""
    champ = registry.get("champion_policy_id")
    for p in registry["policies"]:
        p["status"] = "champion" if p["policy_id"] == champ else "challenger"


def register_policy(entry: PolicyEntry, make_champion: bool = False, registry_dir: Path = REGISTRY_DIR) -> dict:
    """Append a policy (idempotent on ``policy_id``); bootstrap or explicitly set the champion.

    The first policy registered for a segment becomes champion automatically; thereafter a
    policy is added as a challenger unless ``make_champion`` is set. Re-registering an existing
    ``policy_id`` is a no-op for the entry (its original ``created`` is preserved) but can still
    promote it to champion.
    """
    reg = load_registry(entry.segment, registry_dir)
    existing = {p["policy_id"] for p in reg["policies"]}
    if entry.policy_id not in existing:
        reg["policies"].append(entry.to_dict())
    if make_champion or reg["champion_policy_id"] is None:
        reg["champion_policy_id"] = entry.policy_id
    _resync_statuses(reg)
    save_registry(reg, registry_dir)
    return reg


def get_champion(segment: str, registry_dir: Path = REGISTRY_DIR) -> PolicyEntry | None:
    """Return the segment's champion :class:`PolicyEntry`, or None if none is registered."""
    reg = load_registry(segment, registry_dir)
    cid = reg.get("champion_policy_id")
    if not cid:
        return None
    for p in reg["policies"]:
        if p["policy_id"] == cid:
            return PolicyEntry.from_dict(p)
    return None


# --------------------------------------------------------------------------- #
# Champion vs challenger comparison
# --------------------------------------------------------------------------- #


@dataclass
class PolicyComparison:
    """Champion-vs-challenger comparison on one cohort (one segment)."""

    segment: str
    basis: str  # "realized_oot"
    sufficient: bool = True
    message: str = ""
    variables: list = field(default_factory=list)
    champion_policy_id: str | None = None
    challenger_policy_id: str | None = None
    n_champion_cells: int = 0
    n_challenger_cells: int = 0
    added_cells: list = field(default_factory=list)  # accepted by challenger, not champion
    removed_cells: list = field(default_factory=list)  # accepted by champion, not challenger
    champion: dict = field(default_factory=dict)  # realized metrics
    challenger: dict = field(default_factory=dict)
    risk_delta_pp: float | None = None  # challenger − champion (realized risk %)
    production_delta_pct: float | None = None  # challenger vs champion (realized €)
    verdict: str = "INCONCLUSIVE"  # BETTER / WORSE / INCONCLUSIVE (risk, noise-aware)
    window: dict = field(default_factory=dict)

    def aggregate_row(self) -> dict:
        return {
            "segment": self.segment,
            "basis": self.basis,
            "verdict": self.verdict,
            "sufficient": self.sufficient,
            "champion_policy_id": self.champion_policy_id,
            "challenger_policy_id": self.challenger_policy_id,
            "n_champion_cells": self.n_champion_cells,
            "n_challenger_cells": self.n_challenger_cells,
            "n_added_cells": len(self.added_cells),
            "n_removed_cells": len(self.removed_cells),
            "champion_risk_pct": self.champion.get("risk"),
            "champion_risk_ci": [self.champion.get("risk_ci_lower"), self.champion.get("risk_ci_upper")],
            "challenger_risk_pct": self.challenger.get("risk"),
            "challenger_risk_ci": [self.challenger.get("risk_ci_lower"), self.challenger.get("risk_ci_upper")],
            "risk_delta_pp": self.risk_delta_pp,
            "champion_production_eur": self.champion.get("production"),
            "challenger_production_eur": self.challenger.get("production"),
            "production_delta_pct": self.production_delta_pct,
            "n_defaults_champion": self.champion.get("n_defaults"),
            "n_defaults_challenger": self.challenger.get("n_defaults"),
            "holdout_start": self.window.get("start"),
            "holdout_end": self.window.get("end"),
            "message": self.message,
        }


def _risk_verdict(champion: dict, challenger: dict) -> str:
    """Noise-aware risk verdict (lower realized risk = BETTER). Mirrors backtest drift logic.

    BETTER/WORSE only when both policies have ≥ MIN_DEFAULTS_FOR_DRIFT realized defaults AND the
    challenger's risk CI is fully separated from the champion's; otherwise INCONCLUSIVE (within
    sampling noise on a thin window).
    """
    nd = min(champion.get("n_defaults") or 0, challenger.get("n_defaults") or 0)
    if nd < MIN_DEFAULTS_FOR_DRIFT:
        return "INCONCLUSIVE"
    c_lo, c_hi = champion.get("risk_ci_lower"), champion.get("risk_ci_upper")
    h_lo, h_hi = challenger.get("risk_ci_lower"), challenger.get("risk_ci_upper")
    if None in (c_lo, c_hi, h_lo, h_hi):
        return "INCONCLUSIVE"
    if h_hi < c_lo:
        return "BETTER"  # challenger CI entirely below champion's → lower risk
    if h_lo > c_hi:
        return "WORSE"  # challenger CI entirely above champion's → higher risk
    return "INCONCLUSIVE"  # overlapping CIs → within noise


def compare_policies(
    champion_set: set[Cell],
    challenger_set: set[Cell],
    cohort_booked: pd.DataFrame,
    variables: list[str],
    multiplier: float,
    *,
    segment: str = "",
    champion_id: str | None = None,
    challenger_id: str | None = None,
    basis: str = "realized_oot",
) -> PolicyComparison:
    """Score two accepted-cell sets on the SAME booked cohort and diff them.

    ``added`` = cells the challenger accepts but the champion does not; ``removed`` = the
    reverse. Realized risk/production (with bootstrap CIs) come from the shared backtest
    helpers, applied to the identical cohort so the comparison is same-basis. The verdict is
    risk-only and noise-aware; ``production_delta_pct`` is reported for the human trade-off.
    """
    added = sorted(challenger_set - champion_set)
    removed = sorted(champion_set - challenger_set)

    champ = _realized_metrics(cohort_booked, apply_policy(cohort_booked, variables, champion_set), multiplier)
    chal = _realized_metrics(cohort_booked, apply_policy(cohort_booked, variables, challenger_set), multiplier)

    risk_delta = (
        chal["risk"] - champ["risk"] if (chal.get("risk") is not None and champ.get("risk") is not None) else None
    )
    prod_delta_pct = (
        (chal["production"] - champ["production"]) / champ["production"] * 100 if champ.get("production") else None
    )

    return PolicyComparison(
        segment=segment,
        basis=basis,
        variables=list(variables),
        champion_policy_id=champion_id,
        challenger_policy_id=challenger_id,
        n_champion_cells=len(champion_set),
        n_challenger_cells=len(challenger_set),
        added_cells=added,
        removed_cells=removed,
        champion=champ,
        challenger=chal,
        risk_delta_pp=risk_delta,
        production_delta_pct=prod_delta_pct,
        verdict=_risk_verdict(champ, chal),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _celldiff_frame(comparison: PolicyComparison) -> pd.DataFrame:
    """Long-form added/removed cell table for one comparison."""
    rows = []
    cols = list(comparison.variables) or [f"var{i}" for i in range(_n_coords(comparison))]
    for change, cells in (("added", comparison.added_cells), ("removed", comparison.removed_cells)):
        for cell in cells:
            row = {"segment": comparison.segment, "change": change}
            row.update(dict(zip(cols, cell, strict=False)))
            rows.append(row)
    return pd.DataFrame(rows)


def _n_coords(comparison: PolicyComparison) -> int:
    for cells in (comparison.added_cells, comparison.removed_cells):
        if cells:
            return len(cells[0])
    return 0


def write_comparison_report(comparison: PolicyComparison, out_dir: Path, suffix: str = "_base") -> dict[str, Path]:
    """Write the per-segment comparison CSV (+ a cell-diff CSV when cells changed)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = comparison.segment
    paths: dict[str, Path] = {}

    agg_path = out_dir / f"policy_compare_{seg}{suffix}.csv"
    pd.DataFrame([comparison.aggregate_row()]).to_csv(agg_path, index=False)
    paths["aggregate"] = agg_path

    if comparison.added_cells or comparison.removed_cells:
        diff = _celldiff_frame(comparison)
        diff_path = out_dir / f"policy_compare_{seg}{suffix}_celldiff.csv"
        diff.to_csv(diff_path, index=False)
        paths["celldiff"] = diff_path

    return paths


def write_comparison_consolidated(
    comparisons: list[PolicyComparison], out_dir: Path, suffix: str = "_base"
) -> dict[str, Path]:
    """Write the cross-segment comparison CSV + markdown narrative."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"policy_compare_consolidated{suffix}.csv"
    pd.DataFrame([c.aggregate_row() for c in comparisons]).to_csv(csv_path, index=False)

    lines = [
        "# Champion vs challenger — consolidated\n",
        "Each segment's *challenger* (the latest run's base policy) scored against the registered "
        "*champion* on the same auto-derived matured out-of-time cohort. The verdict is risk-only and "
        f"noise-aware: **BETTER/WORSE** only with ≥{MIN_DEFAULTS_FOR_DRIFT} realized defaults in both "
        "policies AND fully separated risk CIs; otherwise **INCONCLUSIVE**. Production delta is reported "
        "for the human trade-off (a lower-risk challenger that also sheds production is not automatically "
        "better).\n",
    ]
    for c in comparisons:
        if not c.sufficient:
            lines.append(f"- **{c.segment}**: skipped — {c.message}")
            continue
        champ_ci = f"[{_fmt(c.champion.get('risk_ci_lower'), 2)}, {_fmt(c.champion.get('risk_ci_upper'), 2)}]"
        chal_ci = f"[{_fmt(c.challenger.get('risk_ci_lower'), 2)}, {_fmt(c.challenger.get('risk_ci_upper'), 2)}]"
        lines.append(
            f"- **{c.segment}** [{c.verdict}]: champion `{c.champion_policy_id}` vs challenger "
            f"`{c.challenger_policy_id}` | +{len(c.added_cells)} / −{len(c.removed_cells)} cells | "
            f"risk {_fmt(c.champion.get('risk'))}% {champ_ci} → {_fmt(c.challenger.get('risk'))}% {chal_ci} "
            f"(Δ {_fmt(c.risk_delta_pp)} pp) | production Δ {_fmt(c.production_delta_pct)}% "
            f"(defaults {c.champion.get('n_defaults')}/{c.challenger.get('n_defaults')})"
        )
    md_path = out_dir / f"policy_compare_summary{suffix}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"consolidated": csv_path, "summary": md_path}


def _fmt(v: float | None, nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"
