"""Audit 2026-11-27 A17: Apple study tooling that could let a wrong figure through unnoticed.

Unlike ``test_check_apple_email.py`` (which needs a completed study run and skips in CI), these
run on synthetic inputs.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.core.constants import Columns
from src.studies.apple.followup import DRIFT_FACTORS, sensitivity_factors
from src.studies.apple.study import PARRILLA_ECOMMERCE, parse_cutoff_overrides, seasonal_complete_years

_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location("check_apple_email", _ROOT / "scripts" / "check_apple_email.py")
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


# --------------------------------------------------------------------------- #
# --cutoff overrides
# --------------------------------------------------------------------------- #


class TestCutoffOverrides:
    def test_documented_example_used_to_be_silently_ignored(self):
        """`--cutoff new=47` added a phantom key "new": the real "New" group kept 27."""
        cutoffs = parse_cutoff_overrides(["new=47"], PARRILLA_ECOMMERCE)

        assert cutoffs["New"] == 47.0
        assert set(cutoffs) == set(PARRILLA_ECOMMERCE)  # no phantom group

    def test_group_names_match_case_insensitively(self):
        cutoffs = parse_cutoff_overrides(["a-c=18", "INACTIVE=30"], PARRILLA_ECOMMERCE)
        assert cutoffs["A-C"] == 18.0 and cutoffs["Inactive"] == 30.0

    def test_unknown_group_is_an_error_that_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="known_cd.*New.*A-C"):
            parse_cutoff_overrides(["known_cd=11"], PARRILLA_ECOMMERCE)

    @pytest.mark.parametrize("value, expected", [("30", 30.0), ("none", None)])
    def test_the_group_whose_name_contains_an_equals_sign(self, value, expected):
        """`>=G=30` split at the first "=" gave the group ">" — the documented group always failed."""
        assert parse_cutoff_overrides([f">=G={value}"], PARRILLA_ECOMMERCE)[">=G"] == expected
        assert parse_cutoff_overrides([f">=g={value}"], PARRILLA_ECOMMERCE)[">=G"] == expected

    def test_nan_is_not_a_score(self):
        """`score > nan` is False for every row: the whole group would be rejected in silence."""
        with pytest.raises(ValueError, match="nan"):
            parse_cutoff_overrides(["New=nan"], PARRILLA_ECOMMERCE)

    def test_the_same_group_twice_is_an_error(self):
        with pytest.raises(ValueError, match="more than once"):
            parse_cutoff_overrides(["New=30", "new=35"], PARRILLA_ECOMMERCE)

    def test_cli_exits_with_an_error_before_loading_data(self, monkeypatch):
        import run_apple_study

        def must_not_load(*args, **kwargs):
            raise AssertionError("data must not be loaded for an invalid --cutoff")

        monkeypatch.setattr(run_apple_study, "load_study_data", must_not_load)
        assert run_apple_study.main(["--cutoff", "known_cd=11"]) == 2

    @pytest.mark.parametrize("override", ["New", "New=abc", "=27"])
    def test_malformed_override_is_an_error(self, override):
        with pytest.raises(ValueError):
            parse_cutoff_overrides([override], PARRILLA_ECOMMERCE)

    def test_none_removes_the_cutoff_and_the_base_grid_is_not_mutated(self):
        base = dict(PARRILLA_ECOMMERCE)

        cutoffs = parse_cutoff_overrides(["New=none"], base)

        assert cutoffs["New"] is None
        assert base == PARRILLA_ECOMMERCE

    def test_no_overrides_returns_the_base_grid(self):
        assert parse_cutoff_overrides([], PARRILLA_ECOMMERCE) == PARRILLA_ECOMMERCE


# --------------------------------------------------------------------------- #
# e-mail checker: figures are matched as whole numbers
# --------------------------------------------------------------------------- #


class TestEmailFiguresMatchAsWholeNumbers:
    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("14,2%", "4,2%"),  # an extra leading digit
            ("12,3 M€/mes", "2,3 M€/mes"),
            ("54,32%", "4,32%"),
            ("4,25%", "4,2"),  # an extra trailing digit
            ("1.234", "234"),  # the tail of a thousands-separated number
            ("11.234", "1.234"),
        ],
    )
    def test_a_different_number_that_contains_the_expected_one_is_not_a_match(self, cell, expected):
        assert not checker.contains_figure(cell, expected)

    @pytest.mark.parametrize(
        ("cell", "expected"),
        [
            ("4,2%", "4,2%"),
            ("**4,2%**", "4,2%"),  # markdown emphasis
            ("+4,0 pp", "+4,0"),
            ("de 52,4% a 57,2%", "57,2%"),
            ("riesgo 3,85% (antes 3,95%)", "3,85%"),
            ("tienda. 4,2%", "4,2%"),  # a sentence full stop is not a decimal separator
            ("18,4 M€/mes", "18,4 M€/mes"),
            ("18,4 M€/mes", "18,4M€/mes"),  # whitespace inside the figure is free
            # a neighbouring number is not more digits of the same one
            ("4,2, 5,1", "4,2"),
            ("2,5 3,0 3,5", "3,0"),
            ("pasa a 2,3.\n5 más", "2,3"),
            ("en 2023. 4,2%", "4,2%"),
            ("1. 4,2%", "4,2%"),
            ("factor 0,956 para tienda", "0,956"),
        ],
    )
    def test_the_expected_figure_is_found(self, cell, expected):
        assert checker.contains_figure(cell, expected)

    def test_check_row_reports_a_figure_with_an_extra_digit(self):
        email = checker.EmailChecker.__new__(checker.EmailChecker)
        email.fails = []

        email.check_row("fila", ["14,2%", "12,3 M€/mes", "3,85%"], ["4,2%", "2,3 M€/mes", "3,85%"])

        assert len(email.fails) == 2
        assert "col0" in email.fails[0] and "col1" in email.fails[1]

    def test_check_prose_does_not_accept_a_longer_number(self):
        email = checker.EmailChecker.__new__(checker.EmailChecker)
        email.fails, email.body = [], "El factor de nivel es 0,9561 y el riesgo imputado 13,91%."

        email.check_prose("factor", "0,956")
        email.check_prose("riesgo", "3,91%")
        email.check_prose("ok", "13,91%")

        assert [f.split(":")[0] for f in email.fails] == ["factor", "riesgo"]


# --------------------------------------------------------------------------- #
# seasonal index: the years it is built on
# --------------------------------------------------------------------------- #


def _monthly(start: str, months: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=months, freq="MS")
    return pd.DataFrame({Columns.MIS_DATE: dates, Columns.OA_AMT: 100.0})


class TestSeasonalYears:
    def test_only_complete_calendar_years_count(self):
        # Jul-2023 … Feb-2026: 2024 and 2025 are complete, 2023 and 2026 are not
        assert seasonal_complete_years(_monthly("2023-07-01", 32)) == [2024, 2025]

    def test_no_complete_year(self):
        assert seasonal_complete_years(_monthly("2026-08-01", 3)) == []

    def test_each_channel_reports_its_own_years(self):
        """The deck's "Full years …" label came from both channels pooled and SCORED rows only;
        the index is estimated per channel on all its demand."""
        online, stores = _monthly("2024-01-01", 24), _monthly("2025-01-01", 12)

        assert seasonal_complete_years(online) == [2024, 2025]
        assert seasonal_complete_years(stores) == [2025]


# --------------------------------------------------------------------------- #
# drift sensitivity: the study's central factor is always in the table
# --------------------------------------------------------------------------- #


class TestSensitivityFactors:
    def test_default_drift_is_already_one_of_the_factors(self):
        assert sensitivity_factors(1.15) == DRIFT_FACTORS

    def test_another_drift_is_added_so_the_deck_can_find_its_central_row(self):
        """With --online-risk-factor 1.2 the table had no (1.2, 1.0) row and the follow-up deck
        quoted the no-factor row's shift as the central case."""
        factors = sensitivity_factors(1.2)

        assert (1.2, 1.0) in factors
        assert set(DRIFT_FACTORS) <= set(factors)
        assert list(factors) == sorted(factors, key=lambda f: (f[1], f[0]))

    def test_float_noise_does_not_duplicate_the_central_factor(self):
        assert sensitivity_factors(1.15 + 1e-12) == DRIFT_FACTORS


# --------------------------------------------------------------------------- #
# deck labels and the follow-up deck's central case
# --------------------------------------------------------------------------- #


class TestSeasonalYearsLabel:
    def test_one_list_when_the_channels_agree_or_the_run_is_older(self):
        from generate_apple_deck import seasonal_years_label

        assert seasonal_years_label({"estacionalidad_anios": [2024, 2025]}) == "2024 and 2025"
        same = {
            "estacionalidad_anios": [2024, 2025],
            "estacionalidad_anios_online": [2024, 2025],
            "estacionalidad_anios_tienda": [2024, 2025],
        }
        assert seasonal_years_label(same) == "2024 and 2025"

    def test_both_lists_when_the_channels_differ(self):
        from generate_apple_deck import seasonal_years_label

        meta = {
            "estacionalidad_anios": [2024, 2025],
            "estacionalidad_anios_online": [2024, 2025],
            "estacionalidad_anios_tienda": [2025],
        }
        assert seasonal_years_label(meta) == "Online 2024 and 2025; stores 2025"

    def test_no_years_says_so(self):
        from generate_apple_deck import seasonal_years_label

        assert seasonal_years_label({"estacionalidad_anios": []}) == "no seasonal adjustment"


class TestCentralSensitivityRow:
    @staticmethod
    def _table(factors) -> pd.DataFrame:
        index = pd.MultiIndex.from_tuples(factors, names=["factor_online", "factor_tienda"])
        return pd.DataFrame({"desplazamiento": range(len(factors))}, index=index)

    def test_finds_the_central_row_despite_rounding(self):
        """periodos.json stores the factor rounded to 4 decimals; the CSV has it in full."""
        from src.studies.apple.followup_deck import _central_sensitivity_row

        table = self._table([(1.0, 1.0), (1.23456, 1.0), (1.23456, 1.23456)])
        assert _central_sensitivity_row(table, 1.2346)["desplazamiento"] == 1

    def test_never_substitutes_another_row(self):
        from src.studies.apple.followup_deck import _central_sensitivity_row

        with pytest.raises(ValueError, match="central case"):
            _central_sensitivity_row(self._table([(1.0, 1.0), (1.15, 1.0)]), 1.2)
