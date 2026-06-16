"""Coverage tests for src.audit helpers not directly hit by test_audit.py.

Targets the post-classification helpers:
``audit_production_kpis``, ``reconcile_risk_production_summary_with_audit``,
``validate_audit_against_summary``, ``save_audit_tables``,
``_total_demand_from_audit_subset``, and ``primary_amount_column``.
"""

import numpy as np
import pandas as pd

from src import audit


def _audit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "classification": ["keep", "swap_in", "swap_out", "rejected", "rejected_other"],
            "status_name": ["booked", "rejected", "booked", "rejected", "canceled"],
            "oa_amt": [1000.0, 200.0, 300.0, 0.0, 50.0],
            "oa_amt_h0": [950.0, 190.0, 285.0, 0.0, 0.0],
            "oa_amt_adjusted": [950.0, 200.0, 285.0, 0.0, 0.0],
        }
    )


class TestPrimaryAmountColumn:
    def test_prefers_oa_amt_h0(self):
        df = pd.DataFrame({"oa_amt_h0": [1.0], "oa_amt": [2.0]})
        assert audit.primary_amount_column(df) == "oa_amt_h0"

    def test_falls_back_to_oa_amt(self):
        df = pd.DataFrame({"oa_amt": [2.0]})
        assert audit.primary_amount_column(df) == "oa_amt"


class TestAuditProductionKpis:
    def test_returns_zero_dict_for_none(self):
        kpis = audit.audit_production_kpis(None)
        assert kpis == {
            "actual": 0.0,
            "swap_in": 0.0,
            "swap_out": 0.0,
            "optimum": 0.0,
            "total_demand": 0.0,
            "actual_rejections": 0.0,
            "optimum_rejections": None,
            "actual_system_rejections": 0.0,
            "optimum_system_rejections": None,
        }

    def test_returns_zero_dict_for_empty(self):
        kpis = audit.audit_production_kpis(pd.DataFrame())
        assert kpis["actual"] == 0.0

    def test_basic_kpis_use_adjusted_amount(self):
        kpis = audit.audit_production_kpis(_audit_df())
        # actual = booked rows from oa_amt_adjusted = 950 + 285 = 1235
        assert kpis["actual"] > 0
        assert kpis["swap_in"] == 200.0
        assert kpis["swap_out"] == 285.0
        # optimum = actual - swap_out + swap_in
        assert kpis["optimum"] == kpis["actual"] - kpis["swap_out"] + kpis["swap_in"]

    def test_kpis_without_status_name_returns_zero_actual(self):
        df = _audit_df().drop(columns=["status_name"])
        kpis = audit.audit_production_kpis(df)
        assert kpis["actual"] == 0.0


class TestRejectionAmountsFromAudit:
    def test_demand_includes_canceled_numerator_does_not(self):
        # _audit_df: booked 1000+300, rejected 200+0, canceled 50 (oa_amt basis)
        td, actual_rej, _ = audit._rejection_amounts_from_audit(_audit_df(), "oa_amt")
        assert td == 1550.0  # through the door, canceled included
        assert actual_rej == 200.0  # status==rejected only, canceled excluded

    def test_returns_zero_when_status_missing(self):
        df = pd.DataFrame({"oa_amt": [100.0]})
        assert audit._rejection_amounts_from_audit(df, "oa_amt") == (0.0, 0.0, None)

    def test_prefers_undiscounted_demand_column(self):
        df = _audit_df()
        df["oa_amt_demand"] = [950.0, 190.0, 285.0, 0.0, 0.0]
        td, actual_rej, _ = audit._rejection_amounts_from_audit(df, "oa_amt_adjusted")
        # Uses oa_amt_demand over the adjusted basis: 950 + 190 + 285 = 1425
        assert td == 1425.0
        assert actual_rej == 190.0

    def test_demand_invariant_to_swap_in_classification(self):
        # The same applications classified differently (rejected vs swap_in)
        # must yield the same demand AND rejections — neither side of the
        # "Actual" rejection rate can depend on the scenario's accepted set.
        def make(classification: str) -> pd.DataFrame:
            df = pd.DataFrame(
                {
                    "classification": ["keep", classification],
                    "status_name": ["booked", "rejected"],
                    "oa_amt_h0": [1000.0, 500.0],
                }
            )
            df["oa_amt_demand"] = df["oa_amt_h0"] * 1.0
            df["oa_amt_adjusted"] = df["oa_amt_h0"] * 1.0
            df.loc[df["classification"] == "swap_in", "oa_amt_adjusted"] *= 0.34
            return df

        for cls in ("rejected", "swap_in"):
            td, actual_rej, _ = audit._rejection_amounts_from_audit(make(cls), "oa_amt_adjusted")
            assert td == 1500.0
            assert actual_rej == 500.0

    def test_optimum_rejections_from_passes_cut(self):
        df = pd.DataFrame(
            {
                "classification": ["keep", "swap_out", "swap_in", "rejected", "rejected_other", "rejected_other"],
                "status_name": ["booked", "booked", "rejected", "rejected", "rejected", "canceled"],
                "passes_cut": [True, False, True, False, True, True],
                "oa_amt_h0": [1000.0, 100.0, 200.0, 50.0, 80.0, 500.0],
            }
        )
        td, actual_rej, optimum_rej = audit._rejection_amounts_from_audit(df, "oa_amt_h0")
        assert td == 1930.0  # everything through the door
        assert actual_rej == 330.0  # swap_in 200 + rejected 50 + non-score 80
        # Optimum declines: fails-cut (swap_out 100 + rejected 50) + non-score
        # rejections that pass the cut (80). Canceled passing the cut (500) and
        # swap_in (200) are approved — not rejections.
        assert optimum_rej == 230.0

    def test_optimum_rejections_none_without_passes_cut(self):
        _, _, optimum_rej = audit._rejection_amounts_from_audit(_audit_df(), "oa_amt")
        assert optimum_rej is None


class TestSystemRejectionAmounts:
    """System rejection rate (se_decision_id == ko) on the same €-weighted demand basis."""

    def _df(self) -> pd.DataFrame:
        # One €100 application per (se_decision_id, reject_reason, passes_cut) case.
        rows = [
            ("ok", "", True),  # above cut, ok -> ok
            ("rv", "", True),  # above cut, rv -> rv
            ("ok", "", False),  # below cut, ok -> KO (simulated)
            ("rv", "", False),  # below cut, rv -> KO (simulated)
            ("ko", "09-score", True),  # above cut, score-ko -> OK (reversal)
            ("ko", "09-score", False),  # below cut, score-ko -> stays KO
            ("ko", "02-bureaux", True),  # above cut, non-score ko -> stays KO
            ("ko", "02-bureaux", False),  # below cut, non-score ko -> KO
        ]
        return pd.DataFrame(
            {
                "se_decision_id": [r[0] for r in rows],
                "reject_reason": [r[1] for r in rows],
                "passes_cut": [r[2] for r in rows],
                "oa_amt_demand": [100.0] * len(rows),
            }
        )

    def test_actual_is_ko_count_eur(self):
        td, actual, _ = audit._system_rejection_amounts_from_audit(self._df(), "oa_amt_demand")
        assert td == 800.0  # everything through the door
        assert actual == 400.0  # the 4 'ko' rows

    def test_optimum_simulates_cutoff_and_score_reversal(self):
        _, _, optimum = audit._system_rejection_amounts_from_audit(self._df(), "oa_amt_demand")
        # KO = fails-cut (4 rows: the 2 below-cut non-ko + the 2 below-cut ko) OR
        # above-cut non-score ko (1). Above-cut score-ko reverses to OK.
        assert optimum == 500.0

    def test_none_when_se_decision_id_missing(self):
        df = self._df().drop(columns=["se_decision_id"])
        assert audit._system_rejection_amounts_from_audit(df, "oa_amt_demand") == (0.0, 0.0, None)

    def test_optimum_none_without_passes_cut(self):
        df = self._df().drop(columns=["passes_cut"])
        td, actual, optimum = audit._system_rejection_amounts_from_audit(df, "oa_amt_demand")
        assert td == 800.0
        assert actual == 400.0
        assert optimum is None

    def test_audit_production_kpis_includes_system_rejections(self):
        df = self._df()
        df["classification"] = "keep"
        df["status_name"] = "booked"
        df["oa_amt_h0"] = 100.0
        kpis = audit.audit_production_kpis(df)
        assert kpis["actual_system_rejections"] == 400.0
        assert kpis["optimum_system_rejections"] == 500.0


class TestReconcile:
    def test_returns_summary_unchanged_when_audit_none(self):
        summary = pd.DataFrame({"Metric": ["Actual"], "Production (€)": [1000.0]})
        out = audit.reconcile_risk_production_summary_with_audit(summary, None)
        assert out.equals(summary)

    def test_returns_summary_unchanged_when_no_metric_column(self):
        summary = pd.DataFrame({"NotMetric": [1]})
        out = audit.reconcile_risk_production_summary_with_audit(summary, _audit_df())
        # missing 'Metric' returns the original
        assert "NotMetric" in out.columns

    def test_overwrites_production_rows(self):
        summary = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                "Production (€)": [0.0, 0.0, 0.0, 0.0],
                "Production (%)": [0.0, 0.0, 0.0, 0.0],
            }
        )
        out = audit.reconcile_risk_production_summary_with_audit(summary, _audit_df())
        actual = out.loc[out["Metric"] == "Actual", "Production (€)"].iloc[0]
        swap_in = out.loc[out["Metric"] == "Swap-in", "Production (€)"].iloc[0]
        swap_out = out.loc[out["Metric"] == "Swap-out", "Production (€)"].iloc[0]
        assert actual > 0
        assert swap_in == 200.0
        assert swap_out == 285.0


class TestValidateAuditAgainstSummary:
    def _summary(self, swap_in: float, swap_out: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Metric": ["Swap-in", "Swap-out"],
                "Production (€)": [swap_in, swap_out],
            }
        )

    def test_passes_when_within_tolerance(self):
        df = _audit_df()
        # swap_in_audit = 200 (oa_amt_adjusted), swap_out_audit = 285
        ok = audit.validate_audit_against_summary(df, self._summary(200.0, 285.0))
        assert ok is True

    def test_fails_when_diverges_more_than_tolerance(self):
        df = _audit_df()
        ok = audit.validate_audit_against_summary(df, self._summary(500.0, 1000.0))
        assert ok is False

    def test_returns_false_when_summary_missing_swap_in(self):
        df = _audit_df()
        summary = pd.DataFrame({"Metric": ["Other"], "Production (€)": [10.0]})
        assert audit.validate_audit_against_summary(df, summary) is False

    def test_returns_false_when_summary_has_nan(self):
        df = _audit_df()
        summary = pd.DataFrame(
            {
                "Metric": ["Swap-in", "Swap-out"],
                "Production (€)": [np.nan, 1.0],
            }
        )
        assert audit.validate_audit_against_summary(df, summary) is False


class TestSaveAuditTables:
    def test_writes_main_and_mr_csvs(self, tmp_path):
        # Pre-supply audit tables to skip the costly generate_audit_table path
        audit_main = _audit_df()
        audit_mr = _audit_df()
        out = audit.save_audit_tables(
            data_main=pd.DataFrame(),
            data_mr=pd.DataFrame(),
            optimal_solution_df=pd.DataFrame({"sol_fac": [0]}),
            variables=["v0", "v1"],
            scenario_name="base",
            output_dir=str(tmp_path),
            audit_main=audit_main,
            audit_mr=audit_mr,
        )
        assert (tmp_path / "audit_base.csv").exists()
        assert (tmp_path / "audit_base_mr.csv").exists()
        assert "main" in out and "mr" in out
