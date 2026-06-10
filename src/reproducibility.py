"""M5 — golden-numbers reproducibility check.

Extracts a segment's *headline numbers* (chosen optimum: risk %, production €,
accepted-cell count + accepted-set hash) from a completed run and compares them to a
**committed reference** within tolerance, so an independent party can confirm the
pipeline reproduces the same answer from the pinned (data snapshot, code, config).

The reference records the **data SHA-256** (from the M2 `run_lineage.json`), git commit,
and config hash it was measured on — a mismatch means the inputs changed and the headline
must be re-validated, so it deliberately fails the check (loudly) rather than silently
comparing across snapshots.

This module is pure/IO-light and unit-testable; `run_reproducibility.py` drives the
actual segment re-run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.backtest import load_frozen_policy

if TYPE_CHECKING:
    from src.config import PreprocessingSettings

REFERENCE_DIR = Path("reports/validation/reference")
DEFAULT_RISK_TOL_PP = 0.01  # absolute risk tolerance, percentage points
DEFAULT_PROD_TOL_PCT = 0.1  # relative production tolerance, percent


@dataclass
class Headline:
    """The reproducible headline numbers for one segment/scenario + the inputs they pin to."""

    segment: str
    scenario: str
    risk_pct: float | None
    production_eur: float | None
    n_accepted_cells: int
    accepted_set_hash: str
    data_sha256: str | None = None
    git_commit: str | None = None
    config_hash: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _accepted_set_hash(accepted_set: set[tuple[float, ...]]) -> str:
    """Order-independent hash of the accepted-cell coordinate set (the policy fingerprint)."""
    payload = ";".join(",".join(f"{x:.6f}" for x in cell) for cell in sorted(accepted_set))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _read_lineage(seg_data_dir: Path) -> dict:
    """Read run_lineage.json (M2) if present; return {} otherwise."""
    path = seg_data_dir / "run_lineage.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def extract_headline(seg_output_dir: Path, settings: PreprocessingSettings, suffix: str = "_base") -> Headline:
    """Read the persisted headline numbers + the snapshot/code/config they pin to."""
    seg_data_dir = Path(seg_output_dir) / "data"
    variables = list(settings.variables)
    accepted_set, opt_sol = load_frozen_policy(seg_data_dir, variables, suffix)

    lineage = _read_lineage(seg_data_dir)
    data = lineage.get("data", {}) or {}
    git = lineage.get("git", {}) or {}
    config = lineage.get("config", {}) or {}

    return Headline(
        segment=settings.segment_filter,
        scenario=suffix.lstrip("_") or "base",
        risk_pct=float(opt_sol["b2_ever_h6"].iloc[0]) if "b2_ever_h6" in opt_sol else None,
        production_eur=float(opt_sol["oa_amt_h0"].iloc[0]) if "oa_amt_h0" in opt_sol else None,
        n_accepted_cells=len(accepted_set),
        accepted_set_hash=_accepted_set_hash(accepted_set),
        data_sha256=data.get("sha256"),
        git_commit=git.get("commit"),
        config_hash=config.get("hash"),
    )


def write_reference(path: Path, headline: Headline) -> Path:
    """Write the committed golden reference JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(headline.to_dict(), indent=2, default=str))
    return path


def load_reference(path: Path) -> dict:
    """Load a committed golden reference JSON."""
    return json.loads(Path(path).read_text())


def compare_headline(
    actual: Headline,
    reference: dict,
    risk_tol_pp: float = DEFAULT_RISK_TOL_PP,
    prod_tol_pct: float = DEFAULT_PROD_TOL_PCT,
) -> dict:
    """Compare a fresh headline to the committed reference within tolerance.

    Fail-closed on missing evidence (audit #26): a governance gate must never pass
    because the thing it verifies is absent. `snapshot_match` is False when the data
    SHA-256 differs from the reference's **or when either side lacks one** (lineage is
    best-effort, so a capture hiccup would otherwise silently remove the snapshot pin);
    likewise a headline whose risk/production value is missing on either side FAILS
    instead of passing vacuously on the remaining checks.
    """
    reasons: list[str] = []

    risk_delta_pp = None
    if actual.risk_pct is None or reference.get("risk_pct") is None:
        risk_ok = False
        reasons.append("risk_pct missing on actual/reference side — cannot verify (fail-closed)")
    else:
        risk_delta_pp = actual.risk_pct - reference["risk_pct"]
        risk_ok = abs(risk_delta_pp) <= risk_tol_pp
        if not risk_ok:
            reasons.append(f"risk drift {risk_delta_pp:+.4f}pp > {risk_tol_pp}pp")

    prod_delta_pct = None
    ref_prod = reference.get("production_eur")
    if actual.production_eur is None or not ref_prod:
        prod_ok = False
        reasons.append("production_eur missing on actual/reference side — cannot verify (fail-closed)")
    else:
        prod_delta_pct = (actual.production_eur - ref_prod) / ref_prod * 100
        prod_ok = abs(prod_delta_pct) <= prod_tol_pct
        if not prod_ok:
            reasons.append(f"production drift {prod_delta_pct:+.3f}% > {prod_tol_pct}%")

    cells_match = actual.n_accepted_cells == reference.get("n_accepted_cells")
    if not cells_match:
        reasons.append(f"accepted-cell count {actual.n_accepted_cells} != {reference.get('n_accepted_cells')}")

    accepted_set_match = actual.accepted_set_hash == reference.get("accepted_set_hash")
    if not accepted_set_match:
        reasons.append("accepted-cell set changed (hash mismatch)")

    # Snapshot pin: a missing SHA on either side means the pin cannot be verified —
    # FAIL, never silently True (the old vacuous pass disappeared whenever lineage
    # capture hiccuped at reference or check time). Mismatch fails loudly.
    ref_sha = reference.get("data_sha256")
    if not ref_sha or not actual.data_sha256:
        snapshot_match = False
        missing_side = "reference" if not ref_sha else "current run"
        reasons.append(
            f"DATA SNAPSHOT PIN UNAVAILABLE (no SHA-256 on the {missing_side} side) — "
            "fail-closed; re-establish the reference with lineage enabled"
        )
    elif ref_sha != actual.data_sha256:
        snapshot_match = False
        reasons.append(
            f"DATA SNAPSHOT CHANGED (sha {actual.data_sha256[:12]} != reference {ref_sha[:12]}) — re-validate"
        )
    else:
        snapshot_match = True

    numbers_ok = risk_ok and prod_ok and cells_match and accepted_set_match
    passed = numbers_ok and snapshot_match

    return {
        "passed": passed,
        "numbers_ok": numbers_ok,
        "snapshot_match": snapshot_match,
        "risk_delta_pp": risk_delta_pp,
        "prod_delta_pct": prod_delta_pct,
        "cells_match": cells_match,
        "accepted_set_match": accepted_set_match,
        "reasons": reasons,
        "actual": actual.to_dict(),
        "reference": reference,
    }


def render_report(result: dict, segment: str) -> str:
    """Markdown PASS/FAIL report for a single segment comparison."""
    status = "PASS" if result["passed"] else "FAIL"
    a, r = result["actual"], result["reference"]
    lines = [
        f"# Reproducibility check — {segment} [{status}]\n",
        "| metric | reference | reproduced | tolerance |",
        "|---|---|---|---|",
        f"| risk (%) | {r.get('risk_pct')} | {a.get('risk_pct')} | Δ {result['risk_delta_pp']} pp |",
        f"| production (€) | {r.get('production_eur')} | {a.get('production_eur')} | Δ {result['prod_delta_pct']} % |",
        f"| accepted cells | {r.get('n_accepted_cells')} | {a.get('n_accepted_cells')} | exact |",
        f"| accepted-set hash | {r.get('accepted_set_hash')} | {a.get('accepted_set_hash')} | exact |",
        f"| data SHA-256 | {str(r.get('data_sha256'))[:12]} | {str(a.get('data_sha256'))[:12]} | must match |",
        f"| git commit | {str(r.get('git_commit'))[:12]} | {str(a.get('git_commit'))[:12]} | (informational) |",
        "",
        f"**snapshot_match:** {result['snapshot_match']} | **numbers_ok:** {result['numbers_ok']}",
    ]
    if result["reasons"]:
        lines.append("\n**Reasons:**")
        lines += [f"- {x}" for x in result["reasons"]]
    return "\n".join(lines) + "\n"
