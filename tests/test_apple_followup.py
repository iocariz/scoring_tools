"""Análisis de seguimiento del estudio Apple (``src/studies/apple/followup.py``)."""

import numpy as np
import pandas as pd
import pytest

from src.studies.apple.followup import (
    cesp_score_distribution,
    cesp_summary,
    cesp_swap,
    cesp_trend,
    demand_profile,
    drift_sensitivity,
    income_band,
    income_risk,
    income_score_table,
    risk_drift,
    score_distribution,
)
from src.studies.apple.study import BIN_EDGES, estimate_risk_curve

CUTOFFS = {"New": 27.0, "A-C": 16.0, ">=G": float("inf")}


def _demand(n: int = 600, start: str = "2025-03-01", months: int = 12, bad_rate: float = 0.05, seed: int = 42):
    """Solicitudes sintéticas: score, decisión, contratación y resultados H3/H6 maduros."""
    rng = np.random.RandomState(seed)
    score = rng.randint(1, 100, n).astype(float)
    group = rng.choice(["New", "A-C", ">=G"], n, p=[0.7, 0.25, 0.05])
    ko_score = (score < 35) & (rng.rand(n) < 0.6)
    ko_other = ~ko_score & (rng.rand(n) < 0.15)
    decision = np.where(ko_score | ko_other | (group == ">=G"), "ko", "ok")
    booked = (decision == "ok") & (rng.rand(n) < 0.7)
    amount = rng.uniform(800, 1600, n).round(2)
    bad = booked & (rng.rand(n) < bad_rate)
    month = pd.to_datetime(start) + pd.to_timedelta(rng.randint(0, months, n) * 30.5, unit="D")
    frame = pd.DataFrame(
        {
            "mis_date": month.to_numpy().astype("datetime64[M]").astype("datetime64[ns]"),
            "grupo_parrilla": group,
            "risk_score_rf": score,
            "se_decision_id": decision,
            "reject_reason": np.where(ko_score, "09-score", np.where(decision == "ko", "04-customer_profile", None)),
            "oa_amt": amount,
            "acct_booked_h0": np.where(booked, 1.0, np.nan),
            "oa_amt_h0": np.where(booked, amount, np.nan),
            "todu_30ever_h6": np.where(bad, amount, 0.0) * booked,
            "todu_amt_pile_h6": np.where(booked, amount * 6.5, 0.0),
            "todu_30ever_h3": np.where(bad, amount, 0.0) * booked,
            "todu_amt_pile_h3": np.where(booked, amount * 3.8, 0.0),
            "income_t1_m": rng.choice([900.0, 1000.0, 1700.0, 2500.0, 4000.0], n),
            "cespa": rng.choice(["s", "n"], n),
            "sin_efx": False,
        }
    )
    frame["bin"] = pd.cut(frame["risk_score_rf"], bins=BIN_EDGES, labels=range(1, len(BIN_EDGES))).astype(float)
    return frame


class TestRiskDrift:
    def test_window_reproduces_itself(self):
        data = _demand()
        out = risk_drift(data, ["bin"], "2025-03-01", "2026-03-01", "online")
        h6 = out[out["horizonte"] == "h6"]
        weights = h6["riesgo_esperado_pct"]  # mismo denominador implícito: basta el signo del agregado
        assert set(out["horizonte"]) == {"h6", "h3"}
        assert np.isclose(np.average(h6["indice"], weights=weights * h6["contratos"]), 1.0, atol=0.15)

    def test_later_deterioration_at_equal_score_raises_the_index(self):
        window = _demand(bad_rate=0.05)
        later = _demand(start="2026-03-01", months=3, bad_rate=0.30, seed=7)
        out = risk_drift(pd.concat([window, later]), ["bin"], "2025-03-01", "2026-03-01", "online")
        late = out[(out["horizonte"] == "h6") & (out["trimestre"] >= "2026Q2")]
        assert (late["indice"] > 1.5).all()

    def test_immature_months_are_left_out(self):
        data = _demand()
        fresh = _demand(start="2026-06-01", months=1, seed=3)
        fresh[["todu_amt_pile_h6", "todu_30ever_h6"]] = 0.0
        out = risk_drift(pd.concat([data, fresh]), ["bin"], "2025-03-01", "2026-03-01", "online")
        assert "2026Q2" not in set(out.loc[out["horizonte"] == "h6", "trimestre"])
        assert "2026Q2" in set(out.loc[out["horizonte"] == "h3", "trimestre"])


class TestDriftSensitivity:
    @pytest.fixture
    def ladder(self):
        shift = np.arange(0, 6)
        online, stores = 5.0 - 0.5 * shift, 3.0 - 0.3 * shift
        return pd.DataFrame(
            {
                "desplazamiento": shift,
                "online_riesgo_pct": online,
                "tienda_riesgo_pct": stores,
                "riesgo_total_pct": 0.4 * online + 0.6 * stores,
                "ta_score_pct": 60.0 - 5 * shift,
                "produccion_mensual_eur": 20e6 - 2e6 * shift,
            }
        )

    def test_unit_factors_reproduce_the_ladder(self, ladder):
        out = drift_sensitivity(ladder, [4.0], factors=((1.0, 1.0),))
        assert out.loc[0, "desplazamiento"] == 0
        assert out.loc[0, "riesgo_parrilla_vigente_pct"] == pytest.approx(3.8)

    def test_stress_applies_to_the_channel_value_with_exposure_weights(self, ladder):
        out = drift_sensitivity(ladder, [4.0], factors=((1.5, 1.0),))
        assert out.loc[0, "riesgo_parrilla_vigente_pct"] == pytest.approx(0.4 * 7.5 + 0.6 * 3.0)
        assert out.loc[0, "desplazamiento"] == 2  # 0.4·1.5·4.0 + 0.6·2.4 = 3.84 ≤ 4
        assert out.loc[0, "riesgo_pct"] <= 4.0

    def test_unreachable_target_is_nan_not_a_crash(self, ladder):
        out = drift_sensitivity(ladder, [0.5], factors=((2.0, 2.0),))
        assert np.isnan(out.loc[0, "desplazamiento"])


class TestCesp:
    def test_summary_splits_add_up(self):
        data = _demand()
        out = cesp_summary({"ventana": data}, CUTOFFS).set_index("poblacion")
        assert out.loc["cesp", "demanda_eur"] + out.loc["no_cesp", "demanda_eur"] == pytest.approx(
            out.loc["total", "demanda_eur"]
        )
        assert out.loc["cesp", "pct_demanda"] + out.loc["no_cesp", "pct_demanda"] == pytest.approx(100)

    def test_without_the_flag_only_the_total_is_reported(self):
        out = cesp_summary({"ventana": _demand().drop(columns="cespa")}, CUTOFFS)
        assert list(out["poblacion"]) == ["total"]

    def test_trend_and_score_distribution(self):
        data = _demand()
        trend = cesp_trend(data)
        assert set(trend["poblacion"]) == {"total", "cesp", "no_cesp"}
        dist = cesp_score_distribution(data)
        assert dist[["total_demanda_pct", "cesp_demanda_pct", "no_cesp_demanda_pct"]].sum().round(6).eq(100).all()

    def test_swap_shares_of_each_category_add_up(self):
        data = _demand()
        curve = estimate_risk_curve(data, 7.0)
        out = cesp_swap(data, curve, CUTOFFS, 1.0)
        parts = out[out["poblacion"] != "total"].groupby("categoria")["pct_categoria"].sum()
        assert np.allclose(parts[parts > 0], 100)


class TestIncome:
    def test_band_edges_are_left_closed(self):
        bands = income_band(pd.DataFrame({"income_t1_m": [999.0, 1000.0, 2999.0, 3000.0, np.nan]}))
        assert list(bands[:4]) == ["<1000", "1000-1499", "2000-2999", ">=3000"]
        assert pd.isna(bands[4])

    def test_swap_out_is_approved_today_and_below_the_cutoff(self):
        data = _demand()
        out = income_score_table(data, CUTOFFS)
        total = out[(out["poblacion"] == "total") & (out["ingresos"] == "total")].iloc[0]
        gradable = data[data["grupo_parrilla"] != ">=G"]
        below = gradable["risk_score_rf"] <= gradable["grupo_parrilla"].map(CUTOFFS)
        assert total["swap_out_n"] == int((below & (gradable["se_decision_id"] == "ok")).sum())
        assert total["solicitudes"] == len(gradable)  # >=G se rechaza por letra, no por score
        ranges = [c for c in out.columns if c.startswith("score_") and c.endswith("_pct")]
        assert total[ranges].sum() == pytest.approx(100)

    def test_income_risk_covers_both_channels(self):
        data = _demand()
        out = income_risk(data, data)
        assert set(out["canal"]) == {"online", "tienda"}
        stores_total = out[(out["canal"] == "tienda") & (out["poblacion"] == "total") & (out["ingresos"] == "total")]
        assert stores_total["contratos"].iloc[0] == int((data["acct_booked_h0"] > 0).sum())


def test_score_distribution_and_demand_profile():
    data = _demand()
    curve = estimate_risk_curve(data, 7.0)
    dist = score_distribution(data, data, curve)
    assert dist["tienda_demanda_pct"].sum() == pytest.approx(100)
    assert dist["tienda_produccion_pct"].sum() == pytest.approx(100)
    profile = demand_profile(data, CUTOFFS, "online")
    assert profile["bajo_corte_pct"].between(0, 100).all()
