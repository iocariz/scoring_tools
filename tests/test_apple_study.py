"""Reglas de negocio del estudio Apple (``run_apple_study.py``).

Cubre las dos funciones donde un error pasaría desapercibido y falsearía el
resultado: el swap de la regla de score y la imputación de riesgo.
"""

import numpy as np
import pandas as pd
import pytest

from run_apple_study import (
    ALWAYS_REJECT,
    apply_reject_inference,
    effective_months,
    estimate_risk_curve,
    grid_fidelity,
    grid_swap_mask,
    imputed_risk,
    measured_status_quo,
    policy_kpis,
    stores_under_grid,
    take_up_by_decision,
    warn_partial_rollout,
)


def _stores(rows):
    return pd.DataFrame(
        rows, columns=["grupo_parrilla", "risk_score_rf", "se_decision_id", "reject_reason", "oa_amt", "bin"]
    )


class TestGridSwapMask:
    """Aceptada iff supera el corte EFX Y no venía KO por un motivo distinto de score."""

    def test_below_cutoff_is_rejected_even_if_system_said_ok(self):
        df = _stores([["New", 20.0, "ok", None, 100.0, 5.0]])
        assert not grid_swap_mask(df, {"New": 27.0}).iloc[0]

    def test_above_cutoff_recovers_a_score_rejection(self):
        # KO por score pero por encima del corte nuevo -> entra (es el swap-in)
        df = _stores([["New", 60.0, "ko", "09-score", 100.0, 13.0]])
        assert grid_swap_mask(df, {"New": 27.0}).iloc[0]

    def test_above_cutoff_keeps_a_non_score_rejection(self):
        # las demás reglas no cambian: sigue rechazada aunque supere el corte
        df = _stores([["New", 90.0, "ko", "04-customer_profile", 100.0, 19.0]])
        assert not grid_swap_mask(df, {"New": 27.0}).iloc[0]

    def test_segment_without_cutoff_only_filters_other_rules(self):
        df = _stores(
            [
                ["Inactive", 1.0, "ok", None, 100.0, 1.0],
                ["Inactive", 1.0, "ko", "03-budget", 100.0, 1.0],
            ]
        )
        assert grid_swap_mask(df, {"Inactive": None}).tolist() == [True, False]

    def test_always_reject_segment_never_passes(self):
        # >=G: decisión de negocio, se rechaza aunque puntúe 99 y el sistema dijera OK
        df = _stores([[">=G", 99.0, "ok", None, 100.0, 20.0]])
        assert not grid_swap_mask(df, {">=G": ALWAYS_REJECT}).iloc[0]

    def test_review_above_cutoff_is_accepted(self):
        # 'rv' no es KO, así que por encima del corte entra
        df = _stores([["New", 60.0, "rv", None, 100.0, 13.0]])
        assert grid_swap_mask(df, {"New": 27.0}).iloc[0]


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


class TestRejectInference:
    """Corrección por selección: más uplift donde menos se acepta, y solo sobre lo nuevo."""

    def _demand(self):
        """Demanda sintética: el tramo 1 casi no se acepta, el 3 se acepta entero."""
        rows = []
        for bin_value, booked, rejected in ((1.0, 5, 95), (2.0, 50, 50), (3.0, 100, 0)):
            rows += [{"bin": bin_value, "status_name": "booked", "reject_reason": None}] * booked
            rows += [{"bin": bin_value, "status_name": "rejected", "reject_reason": "09-score"}] * rejected
        return pd.DataFrame(rows)

    def _curve(self):
        return pd.DataFrame(
            {"b2_pct": [9.0, 6.0, 3.0], "exposure_ratio": [1.0, 1.0, 1.0], "booked_eur": [1e5, 1e5, 1e5]},
            index=pd.Index([1.0, 2.0, 3.0], name="bin"),
        )

    def test_uplift_is_larger_where_acceptance_is_lower(self):
        out = apply_reject_inference(self._curve(), self._demand())
        mult = out["b2_pct_rechazados"] / out["b2_pct"]
        assert mult.loc[1.0] > mult.loc[2.0] > mult.loc[3.0]

    def test_fully_accepted_bin_is_barely_uplifted(self):
        """Si se acepta a todo el mundo, los contratados no son una muestra seleccionada.

        No exactamente 1,00x: el suavizado bayesiano tira la tasa del tramo hacia la
        global, así que queda un residuo. Lo que debe cumplirse es que sea pequeño frente
        al del tramo que apenas se acepta.
        """
        out = apply_reject_inference(self._curve(), self._demand())
        mult = out["b2_pct_rechazados"] / out["b2_pct"]
        assert mult.loc[3.0] < 1.15
        assert mult.loc[1.0] > 2.0

    def test_monotonicity_does_not_flatten_the_multiplier(self):
        """El tramo va de peor a mejor score, así que el multiplicador debe DECRECER.

        Sin declarar la dirección (``inv_vars``), la isotónica del pipeline exige lo
        contrario y aplana el multiplicador a una constante: un recargo plano disfrazado
        de reject inference, y sin error que lo delate.
        """
        out = apply_reject_inference(self._curve(), self._demand())
        mult = (out["b2_pct_rechazados"] / out["b2_pct"]).round(3)
        assert mult.nunique() > 1, f"multiplicador aplanado a una constante: {mult.tolist()}"

    def test_blend_sits_between_booked_and_uplifted(self):
        """La curva aplicada mezcla contratados y nuevos según la tasa de aceptación."""
        out = apply_reject_inference(self._curve(), self._demand())
        for b in (1.0, 2.0, 3.0):
            assert out.loc[b, "b2_pct"] <= out.loc[b, "b2_pct_ri"] <= out.loc[b, "b2_pct_rechazados"] + 1e-9


def _channel(rows):
    """Canal sintético con lo que necesitan la swap mask y el take-up."""
    cols = ["grupo_parrilla", "risk_score_rf", "se_decision_id", "reject_reason", "oa_amt", "bin"]
    cols += ["acct_booked_h0", "oa_amt_h0"]
    return pd.DataFrame(rows, columns=cols)


class TestTakeUp:
    """El take-up va por decisión del motor. Medirlo solo sobre las 'ok' y aplicarlo también a
    las 'rv' —que la swap mask acepta y convierten diez veces menos— hinchaba la producción."""

    def _channel(self):
        # 2 ok (1 contrata), 4 rv (1 contrata), 1 ko por score bajo el corte
        return _channel(
            [
                ["New", 60.0, "ok", None, 100.0, 13.0, 1, 100.0],
                ["New", 60.0, "ok", None, 100.0, 13.0, 0, 0.0],
                ["New", 60.0, "rv", None, 100.0, 13.0, 1, 100.0],
                ["New", 60.0, "rv", None, 100.0, 13.0, 0, 0.0],
                ["New", 60.0, "rv", None, 100.0, 13.0, 0, 0.0],
                ["New", 60.0, "rv", None, 100.0, 13.0, 0, 0.0],
                ["New", 20.0, "ko", "09-score", 100.0, 5.0, 0, 0.0],
            ]
        )

    def _curve(self):
        return pd.DataFrame(
            {"b2_pct": [5.0, 3.0], "b2_pct_contratados": [5.0, 3.0], "exposure_ratio": [1.0, 1.0]},
            index=pd.Index([5.0, 13.0], name="bin"),
        )

    def test_rates_by_decision_and_score_swap_ins_convert_like_ok(self):
        rates = take_up_by_decision(self._channel())
        assert rates["ok"] == 0.5
        assert rates["rv"] == 0.25
        assert rates["ko"] == rates["ok"]

    def test_missing_decision_falls_back(self):
        channel = self._channel()
        only_ok = channel[channel["se_decision_id"] != "rv"]
        assert take_up_by_decision(only_ok, fallback={"ok": 0.9, "rv": 0.1})["rv"] == 0.1

    def test_policy_production_reproduces_booked_under_the_real_policy(self):
        # La parrilla calca al motor (ok/rv por encima del corte, ko-score por debajo): la
        # producción modelada tiene que ser la contratada, euro a euro. Con un take-up
        # medido solo sobre las ok (1,0) salían 600 en vez de 200.
        channel = self._channel()
        kpis = policy_kpis(channel, channel, self._curve(), {"New": 27.0}, 1.0, 1.0, 1.0)
        assert kpis["produccion_mensual_eur"] == pytest.approx(2 * channel["oa_amt_h0"].sum())
        assert kpis["ta_score_pct"] == pytest.approx(100 * 600 / 700)
        assert kpis["ta_efectiva_pct"] == pytest.approx(100 * 200 / 700)

    def test_stores_total_reproduces_booked_under_the_real_policy(self):
        channel = self._channel()
        grid = stores_under_grid(channel, self._curve(), {"New": 27.0}, 1.0)
        total = grid[grid["segmento"] == "TOTAL"].iloc[0]
        assert total["produccion_est_eur"] == pytest.approx(channel["oa_amt_h0"].sum())
        assert total["ta_efectiva_pct"] == pytest.approx(100 * 200 / 700)


class TestGridFidelity:
    """Swap-in / swap-out en % de la demanda en €: lo que separa la parrilla del motor."""

    def _channel(self):
        return TestTakeUp()._channel()

    def test_grid_that_matches_the_engine_has_no_swaps(self):
        fid = grid_fidelity(self._channel(), {"New": 27.0})
        assert fid["aprobado_real_pct"] == pytest.approx(100 * 600 / 700)
        assert fid["parrilla_pct"] == pytest.approx(100 * 600 / 700)
        assert fid["swap_in_pct"] == 0.0
        assert fid["swap_out_pct"] == 0.0

    def test_tightening_is_swap_out_and_loosening_is_swap_in(self):
        tight = grid_fidelity(self._channel(), {"New": 70.0})
        assert tight["swap_out_pct"] == pytest.approx(100 * 600 / 700)
        assert tight["parrilla_pct"] == 0.0
        loose = grid_fidelity(self._channel(), {"New": 10.0})
        assert loose["swap_in_pct"] == pytest.approx(100 * 100 / 700)
        assert loose["swap_out_pct"] == 0.0


class TestMeasuredStatusQuo:
    def test_each_channel_is_normalised_by_its_own_effective_months(self):
        # Online y tienda tienen perfiles estacionales distintos: el 'hoy' medido debe
        # dividir cada canal por sus propios meses efectivos, no por los de Online.
        cols = ["oa_amt", "acct_booked_h0", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]
        ecom = pd.DataFrame([[100.0, 1, 100.0, 1.0, 100.0]], columns=cols)
        stores = pd.DataFrame([[100.0, 1, 100.0, 1.0, 100.0]], columns=cols)
        kpis = measured_status_quo(ecom, stores, months_ecom=1.0, months_stores=2.0, multiplier=7.0)
        assert kpis["produccion_mensual_eur"] == pytest.approx(100.0 + 50.0)
        assert kpis["ta_efectiva_pct"] == pytest.approx(100.0)
