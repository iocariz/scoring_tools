"""Reglas de negocio del estudio Apple (``run_apple_study.py``).

Cubre las dos funciones donde un error pasaría desapercibido y falsearía el
resultado: el swap de la regla de score y la imputación de riesgo.
"""

import numpy as np
import pandas as pd
import pytest

from run_apple_study import (
    ALWAYS_REJECT,
    effective_months,
    estimate_risk_curve,
    grid_swap_mask,
    imputed_risk,
    warn_partial_rollout,
)


def _stores(rows):
    return pd.DataFrame(
        rows, columns=["segment_cut_off", "risk_score_rf", "se_decision_id", "reject_reason", "oa_amt", "bin"]
    )


class TestGridSwapMask:
    """Aceptada iff supera el corte EFX Y no venía KO por un motivo distinto de score."""

    def test_below_cutoff_is_rejected_even_if_system_said_ok(self):
        df = _stores([["new", 40.0, "ok", None, 100.0, 9.0]])
        assert not grid_swap_mask(df, {"new": 47.0}).iloc[0]

    def test_above_cutoff_recovers_a_score_rejection(self):
        # KO por score pero por encima del corte nuevo -> entra (es el swap-in)
        df = _stores([["new", 60.0, "ko", "09-score", 100.0, 13.0]])
        assert grid_swap_mask(df, {"new": 47.0}).iloc[0]

    def test_above_cutoff_keeps_a_non_score_rejection(self):
        # las demás reglas no cambian: sigue rechazada aunque supere el corte
        df = _stores([["new", 90.0, "ko", "04-customer_profile", 100.0, 19.0]])
        assert not grid_swap_mask(df, {"new": 47.0}).iloc[0]

    def test_segment_without_cutoff_only_filters_other_rules(self):
        df = _stores(
            [
                ["inactive", 1.0, "ok", None, 100.0, 1.0],
                ["inactive", 1.0, "ko", "03-budget", 100.0, 1.0],
            ]
        )
        assert grid_swap_mask(df, {"inactive": None}).tolist() == [True, False]

    def test_always_reject_segment_never_passes(self):
        # known_g: decisión de negocio, se rechaza aunque puntúe 99 y el sistema dijera OK
        df = _stores([["known_g", 99.0, "ok", None, 100.0, 20.0]])
        assert not grid_swap_mask(df, {"known_g": ALWAYS_REJECT}).iloc[0]

    def test_review_above_cutoff_is_accepted(self):
        # 'rv' no es KO, así que por encima del corte entra
        df = _stores([["new", 60.0, "rv", None, 100.0, 13.0]])
        assert grid_swap_mask(df, {"new": 47.0}).iloc[0]


class TestImputedRisk:
    """El riesgo imputado pondera por exposición esperada, no por número de solicitudes."""

    def _curve(self):
        return pd.DataFrame(
            {"b2_pct": [10.0, 2.0], "exposure_ratio": [1.0, 1.0], "booked_eur": [1e6, 1e6]},
            index=pd.Index([1.0, 2.0], name="bin"),
        )

    def test_weights_by_demand_not_by_row_count(self):
        # un solicitante grande en el tramo malo pesa más que muchos pequeños en el bueno
        apps = pd.DataFrame({"bin": [1.0, 2.0, 2.0], "oa_amt": [900.0, 50.0, 50.0]})
        assert imputed_risk(apps, self._curve()) == 10.0 * 0.9 + 2.0 * 0.1

    def test_level_factor_rescales_linearly(self):
        apps = pd.DataFrame({"bin": [1.0], "oa_amt": [100.0]})
        assert imputed_risk(apps, self._curve(), level_factor=1.5) == 15.0

    def test_empty_input_returns_nan_not_zero(self):
        apps = pd.DataFrame({"bin": [], "oa_amt": []})
        assert np.isnan(imputed_risk(apps, self._curve()))


class TestRiskCurve:
    def test_b2_is_multiplier_times_ratio_and_exposure_per_euro(self):
        booked = pd.DataFrame(
            {
                "bin": [1.0, 1.0],
                "acct_booked_h0": [1, 1],
                "todu_30ever_h6": [1.0, 1.0],
                "todu_amt_pile_h6": [50.0, 50.0],
                "oa_amt_h0": [10.0, 10.0],
            }
        )
        curve = estimate_risk_curve(booked, multiplier=7.0)
        assert curve.loc[1.0, "b2_pct"] == 7.0 * 100 * 2.0 / 100.0
        assert curve.loc[1.0, "exposure_ratio"] == 100.0 / 20.0


def _index(august=0.5, september=2.0):
    """Índice estacional sintético con media 1: agosto flojo, septiembre pico."""
    values = dict.fromkeys(range(1, 13), 1.0)
    values[8], values[9] = august, september
    series = pd.Series(values)
    return series / series.mean()


class TestEffectiveMonths:
    """El divisor del run-rate son meses ESTACIONALES, no meses de calendario."""

    def test_full_year_equals_twelve(self):
        # una ventana natural completa no debe alterar el run-rate: el índice tiene media 1
        assert effective_months("2025-01-01", "2026-01-01", _index()) == pytest.approx(12.0)

    def test_single_weak_month_counts_less_than_one(self):
        idx = _index()
        assert effective_months("2026-08-01", "2026-09-01", idx) == pytest.approx(idx.loc[8])
        assert effective_months("2026-08-01", "2026-09-01", idx) < 1.0

    def test_single_peak_month_counts_more_than_one(self):
        idx = _index()
        assert effective_months("2026-09-01", "2026-10-01", idx) == pytest.approx(idx.loc[9])
        assert effective_months("2026-09-01", "2026-10-01", idx) > 1.0

    def test_without_index_falls_back_to_calendar_months(self):
        assert effective_months("2026-07-01", "2026-09-01", None) == pytest.approx(2.0)

    def test_partial_month_is_prorated(self):
        # media ventana de agosto (31 dias) -> 15/31 del indice de agosto
        idx = _index()
        assert effective_months("2026-08-01", "2026-08-16", idx) == pytest.approx(idx.loc[8] * 15 / 31)


class TestWarnPartialRollout:
    def _stores_months(self, july_eur, august_eur):
        return pd.DataFrame(
            {
                "mis_date": [pd.Timestamp("2026-07-15"), pd.Timestamp("2026-08-15")],
                "oa_amt": [july_eur, august_eur],
            }
        )

    def test_flags_a_month_far_below_its_seasonal_profile(self):
        # julio deberia superar a agosto por estacionalidad; si trae el 9% es rollout parcial
        flagged = warn_partial_rollout(self._stores_months(1.0, 10.0), _index())
        assert flagged == ["2026-07"]

    def test_does_not_flag_months_consistent_with_seasonality(self):
        idx = _index()
        # volumenes proporcionales al indice -> ningun mes es sospechoso
        july, august = 100 * idx.loc[7], 100 * idx.loc[8]
        assert warn_partial_rollout(self._stores_months(july, august), idx) == []

    def test_single_month_cannot_be_judged(self):
        one = pd.DataFrame({"mis_date": [pd.Timestamp("2026-08-15")], "oa_amt": [10.0]})
        assert warn_partial_rollout(one, _index()) == []
