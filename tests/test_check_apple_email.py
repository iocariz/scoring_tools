"""The Apple e-mail draft must match the study outputs, enforced via ``scripts/check_apple_email.py``.

The study outputs (``output/apple_study/``) come from real data and are not committed, so
these tests run only where a study has been run (they skip in CI). They also check that
the script is not vacuous: a single altered cell must fail with a pointed message.
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("check_apple_email", _ROOT / "scripts" / "check_apple_email.py")
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

STUDY_DIR = _ROOT / checker.DEFAULT_STUDY_DIR
EMAIL = _ROOT / checker.DEFAULT_EMAIL

pytestmark = pytest.mark.skipif(
    not (STUDY_DIR / "periodos.json").exists() or not EMAIL.exists(),
    reason="needs a completed run_apple_study.py (real data) and the e-mail draft",
)


def test_email_matches_study_outputs():
    assert checker.EmailChecker(STUDY_DIR, EMAIL).run() == []


def test_altered_cell_is_caught(tmp_path):
    original = EMAIL.read_text(encoding="utf-8")
    needle = "| Apple Online | 51,9% |"
    assert needle in original, "point-1 row changed; update the needle"
    broken = tmp_path / "broken.md"
    broken.write_text(original.replace(needle, "| Apple Online | 52,9% |"), encoding="utf-8")
    fails = checker.EmailChecker(STUDY_DIR, broken).run()
    assert fails == ["P1 Apple Online col0: email='52,9%' expected≈'51,9%'"]


def test_missing_row_is_a_failure_not_a_crash(tmp_path):
    original = EMAIL.read_text(encoding="utf-8")
    needle = "| **Apple Total** |"
    assert needle in original, "point-1 total row changed; update the needle"
    broken = tmp_path / "broken.md"
    broken.write_text(original.replace(needle, "| **Apple Todo** |"), encoding="utf-8")
    fails = checker.EmailChecker(STUDY_DIR, broken).run()
    assert fails and fails[0].startswith("point_1:") and "Apple Total" in fails[0]
