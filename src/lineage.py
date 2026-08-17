"""Per-run data lineage / provenance capture (M2).

Records, for each pipeline run, where the numbers came from: the source data
file (path + size + modification time + SHA-256, the snapshot/extraction proxy)
and its loaded row count, the code revision (git commit + dirty flag), the
resolved config identity, the run timestamp / run-id, and the headline modelling
assumptions (multiplier, stress mode, risk target, MR/RI toggles).

Everything here is **diagnostic, not decisioning**: callers wrap the capture in a
best-effort guard so a lineage failure never aborts the pipeline. Git capture and
file stats degrade gracefully (non-repo, Docker, missing file) rather than raise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from src.config import OutputPaths, PreprocessingSettings

# Cache file provenance by (resolved path, size, mtime_ns) so a large SAS file is
# hashed at most once per process — batch mode shares one file across all segments.
_PROVENANCE_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of *path*'s contents (chunked read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_provenance(path: str | None) -> dict[str, Any]:
    """Provenance for a data file: size, mtime (UTC ISO), SHA-256.

    Returns ``{"exists": False}`` (no raise) when *path* is empty or unreadable.
    Memoized on (path, size, mtime) — repeat calls return the same cached dict.
    """
    if not path:
        return {"path": path, "exists": False}
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {"path": str(path), "exists": False}

    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    cached = _PROVENANCE_CACHE.get(key)
    if cached is not None:
        return cached

    prov: dict[str, Any] = {
        "path": str(p),
        "exists": True,
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
        "sha256": _sha256(p),
    }
    _PROVENANCE_CACHE[key] = prov
    return prov


def git_provenance() -> dict[str, Any]:
    """Current git commit + dirty flag of the CODE's repo. Never raises.

    Returns ``{"commit": None, ...}`` when not in a repo, git is absent, or the
    call times out (e.g. inside a Docker image with no .git).
    """
    # Run git in THIS source tree's directory, not the process CWD — otherwise
    # launching the pipeline from inside another git repo would stamp that repo's
    # commit/dirty flag into the lineage / Excel provenance (a wrong-looking-real value).
    repo_dir = str(Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=repo_dir,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=repo_dir,
        ).stdout
        return {"commit": commit, "short": commit[:12], "dirty": bool(status.strip())}
    except Exception:
        return {"commit": None, "short": None, "dirty": None}


def _config_identity(settings: PreprocessingSettings, config_path: str | None) -> dict[str, Any]:
    """Config path + a stable hash of the resolved settings."""
    cfg_hash: str | None
    try:
        payload = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
        cfg_hash = hashlib.md5(payload.encode()).hexdigest()[:12]
    except Exception:
        cfg_hash = None
    return {"path": config_path, "hash": cfg_hash}


def build_lineage(
    settings: PreprocessingSettings,
    config_path: str | None,
    n_rows: int,
    run_id: str | None = None,
    run_ts_iso: str | None = None,
) -> dict[str, Any]:
    """Assemble the per-run lineage dict.

    ``run_id`` / ``run_ts_iso`` are passed in by batch mode so every segment shares
    one canonical batch run-id; standalone runs generate their own.
    """
    now = datetime.now(UTC)
    ts = run_ts_iso or now.isoformat()
    git = git_provenance()
    if run_id is None:
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{git.get('short') or 'nogit'}"

    return {
        "run_id": run_id,
        "run_timestamp": ts,
        "segment": settings.segment_filter,
        "date_range": f"{settings.date_ini_book_obs} -> {settings.date_fin_book_obs}",
        "data": {**file_provenance(settings.data_path), "rows_loaded": int(n_rows)},
        "git": git,
        "config": _config_identity(settings, config_path),
        "assumptions": {
            "multiplier": settings.multiplier,
            "stress_mode": settings.stress_mode,
            "optimum_risk": settings.optimum_risk,
            "risk_step": settings.risk_step,
            "use_mr_outcomes": settings.use_mr_outcomes,
            "reject_inference_method": settings.reject_inference_method,
        },
    }


def write_lineage(output: OutputPaths, lineage: dict[str, Any]) -> Path:
    """Write *lineage* as JSON to ``output.run_lineage_json``."""
    path = Path(output.run_lineage_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineage, indent=2, default=str))
    return path


def log_run_banner(lineage: dict[str, Any]) -> None:
    """Log a compact provenance banner at run entry."""
    data = lineage.get("data", {})
    git = lineage.get("git", {})
    cfg = lineage.get("config", {})
    a = lineage.get("assumptions", {})
    sha = (data.get("sha256") or "")[:12] or "n/a"
    dirty = " (dirty)" if git.get("dirty") else ""
    logger.info(f"==== RUN {lineage.get('run_id')} | segment={lineage.get('segment')} ====")
    logger.info(f"  config={cfg.get('path')} (hash {cfg.get('hash')}) | git={git.get('short') or 'n/a'}{dirty}")
    logger.info(
        f"  data={data.get('path')} | sha={sha} | mtime={data.get('mtime') or 'n/a'} | rows={data.get('rows_loaded')}"
    )
    logger.info(
        f"  assumptions: multiplier={a.get('multiplier')} stress_mode={a.get('stress_mode')} "
        f"optimum_risk={a.get('optimum_risk')} use_mr_outcomes={a.get('use_mr_outcomes')} "
        f"reject={a.get('reject_inference_method')}"
    )
