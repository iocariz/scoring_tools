"""Sign pre-R0 model artifacts with SHA-256 integrity sidecars.

R0 (todo #44) enforces a SHA-256 sidecar check for every pickle loaded by
the pipeline.  Existing ``.pkl`` / ``.joblib`` files produced before R0 have
no sidecar and will be refused by ``safe_joblib_load``.

This script walks a directory tree and writes a ``<file>.sha256`` sidecar
next to every ``.pkl`` and ``.joblib`` it finds, provided the operator has
explicitly attested that those artifacts are trusted.

Usage
-----
    uv run python scripts/sign_existing_models.py \\
        --root models/ \\
        --yes-i-verified-these-are-safe

    # Dry-run (lists what would be signed, writes nothing):
    uv run python scripts/sign_existing_models.py --root models/ --dry-run

The explicit attestation flag is intentional: signing a malicious pickle
is equivalent to authorizing its execution on the next load.  If you
cannot personally attest that a file is safe, do not sign it — regenerate
it by retraining instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a standalone script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.persistence import ModelIntegrityError, write_integrity_sidecar  # noqa: E402

PICKLE_SUFFIXES = {".pkl", ".joblib"}


def _discover(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in PICKLE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory to scan recursively.")
    parser.add_argument(
        "--yes-i-verified-these-are-safe",
        action="store_true",
        help=(
            "Required attestation. Without this flag the script refuses to write sidecars. "
            "Pass only after you have personally verified every file under --root is trusted."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="List targets without writing sidecars.")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: --root {root} is not a directory", file=sys.stderr)
        return 2

    targets = _discover(root)
    if not targets:
        print(f"No .pkl / .joblib files under {root}")
        return 0

    print(f"Found {len(targets)} pickle file(s) under {root}:")
    for t in targets:
        sidecar = t.with_name(t.name + ".sha256")
        marker = "[exists]" if sidecar.is_file() else "[new]   "
        print(f"  {marker} {t}")

    if args.dry_run:
        print("\nDry run — no sidecars written.")
        return 0

    if not args.yes_i_verified_these_are_safe:
        print(
            "\nERROR: refusing to sign without --yes-i-verified-these-are-safe.\n"
            "Re-run with that flag only after personally verifying every listed file is trusted.",
            file=sys.stderr,
        )
        return 2

    written = 0
    failed: list[tuple[Path, str]] = []
    for t in targets:
        try:
            write_integrity_sidecar(t)
            written += 1
        except ModelIntegrityError as exc:
            failed.append((t, str(exc)))

    print(f"\nSigned {written}/{len(targets)} file(s).")
    if failed:
        print(f"Failed: {len(failed)}", file=sys.stderr)
        for path, msg in failed:
            print(f"  {path}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
