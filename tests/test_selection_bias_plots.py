"""Smoke tests for selection_bias_plots.

Many of these functions have an "empty → return None" early exit that's cheap
to cover. A handful of happy-path tests render a small plot and verify the
output file lands.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import selection_bias_plots as sbp


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Early-exit paths — cheap coverage
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_thorndike_empty_returns_none(self, tmp_path):
        assert sbp.plot_thorndike_correction(pd.DataFrame(), tmp_path) is None

    def test_simulated_restriction_empty_returns_none(self, tmp_path):
        assert sbp.plot_simulated_restriction(pd.DataFrame(), tmp_path) is None

    def test_temporal_acceptance_empty_rolling_gini_returns_none(self, tmp_path):
        assert sbp.plot_temporal_acceptance_vs_gini(pd.DataFrame(), pd.DataFrame(), tmp_path) is None

    def test_temporal_acceptance_empty_after_merge_returns_none(self, tmp_path):
        rolling_gini = pd.DataFrame(
            {"segment": ["a"], "period": ["main"], "month": ["2024-01"], "score": ["x"], "gini": [0.5]}
        )
        rolling_acceptance = pd.DataFrame(
            {"segment": ["b"], "period": ["main"], "month": ["2024-02"], "acceptance_rate": [0.5]}
        )
        assert sbp.plot_temporal_acceptance_vs_gini(rolling_gini, rolling_acceptance, tmp_path) is None

    def test_rejection_profile_all_empty_returns_none(self, tmp_path):
        assert sbp.plot_rejection_profile({"a": pd.DataFrame()}, "seg", tmp_path) is None

    def test_cross_segment_no_data_returns_none(self, tmp_path):
        assert sbp.plot_cross_segment_scatter([], {"slope": None}, tmp_path) is None

    def test_cross_segment_null_slope_returns_none(self, tmp_path):
        assert sbp.plot_cross_segment_scatter([{"segment": "a"}], {"slope": None}, tmp_path) is None

    def test_booked_vs_rejected_empty_returns_none(self, tmp_path):
        assert sbp.plot_booked_vs_rejected_discriminance(pd.DataFrame(), None, tmp_path) is None

    def test_truncation_comparison_all_empty_returns_none(self, tmp_path):
        assert sbp.plot_truncation_comparison(pd.DataFrame(), pd.DataFrame(), tmp_path) is None

    def test_truncation_comparison_no_overlap_returns_none(self, tmp_path):
        a = pd.DataFrame({"segment": ["a"], "score": ["x"], "u_ratio": [0.5]})
        b = pd.DataFrame({"segment": ["b"], "score": ["y"], "u_ratio": [0.6]})
        assert sbp.plot_truncation_comparison(a, b, tmp_path) is None

    def test_bad_rate_all_empty_returns_none(self, tmp_path):
        assert sbp.plot_bad_rate_by_decile({"a": pd.DataFrame()}, "seg", tmp_path) is None

    def test_ri_gini_empty_returns_none(self, tmp_path):
        assert sbp.plot_ri_gini(pd.DataFrame(), None, tmp_path) is None

    def test_monthly_psi_all_empty_returns_none(self, tmp_path):
        assert sbp.plot_monthly_psi({"a": pd.DataFrame()}, "seg", tmp_path) is None


# ---------------------------------------------------------------------------
# Happy paths — render a small plot
# ---------------------------------------------------------------------------


def _trunc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["seg1", "seg1", "seg2", "seg2"],
            "score": ["Score RF", "Risk Score RF", "Score RF", "Risk Score RF"],
            "observed_gini": [0.55, 0.45, 0.50, 0.40],
            "corrected_gini": [0.60, 0.48, 0.54, 0.42],
            "attenuation_pct": [8.3, 6.3, 7.4, 4.8],
            "u_ratio": [0.85, 0.90, 0.80, 0.88],
        }
    )


class TestPlotHappyPaths:
    def test_distribution_truncation_produces_axes(self):
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame(
            {
                "score": rng.normal(0, 1, n),
                "status_name": rng.choice(["booked", "rejected"], n),
            }
        )
        ax = sbp.plot_distribution_truncation(df, "score", "Test Score", "seg_a")
        assert ax is not None

    def test_all_truncation_panels_writes_file(self, tmp_path):
        rng = np.random.RandomState(42)
        n = 150
        df = pd.DataFrame(
            {
                "score_rf": rng.normal(0, 1, n),
                "risk_score_rf": rng.normal(0, 1, n),
                "status_name": rng.choice(["booked", "rejected"], n),
            }
        )
        score_columns = {
            "Score RF": {"column": "score_rf", "negate": False},
            "Risk Score RF": {"column": "risk_score_rf", "negate": True},
        }
        path = sbp.plot_all_truncation_panels(df, score_columns, "segA", tmp_path)
        assert path.exists()

    def test_thorndike_correction_writes_file(self, tmp_path):
        path = sbp.plot_thorndike_correction(_trunc_df(), tmp_path)
        assert path is not None and path.exists()

    def test_simulated_restriction_writes_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "segment": ["seg1"] * 4,
                "score": ["Score RF"] * 4,
                "pct_retained": [1.0, 0.75, 0.5, 0.25],
                "gini": [0.50, 0.48, 0.42, 0.30],
            }
        )
        path = sbp.plot_simulated_restriction(df, tmp_path)
        assert path is not None and path.exists()

    def test_rejection_profile_writes_file(self, tmp_path):
        profile = pd.DataFrame(
            {
                "decile": [1, 2, 3, 4, 5],
                "n_booked": [100, 90, 70, 50, 30],
                "n_rejected": [10, 20, 30, 50, 70],
                "rejection_rate": [0.10, 0.22, 0.43, 0.66, 0.83],
            }
        )
        path = sbp.plot_rejection_profile({"Score RF": profile}, "segA", tmp_path)
        assert path is not None and path.exists()

    def test_bad_rate_by_decile_writes_file(self, tmp_path):
        profile = pd.DataFrame(
            {
                "decile": [1, 2, 3, 4, 5],
                "bad_rate": [0.01, 0.02, 0.04, 0.06, 0.10],
                "n_bads": [1, 2, 4, 6, 10],
                "n_records": [100, 100, 100, 100, 100],
            }
        )
        path = sbp.plot_bad_rate_by_decile({"Score RF": profile}, "segA", tmp_path)
        assert path is not None and path.exists()

    def test_booked_vs_rejected_discriminance_writes_file(self, tmp_path):
        bvr = pd.DataFrame({"segment": ["a"], "score": ["Score RF"], "gini": [0.4]})
        eb = pd.DataFrame({"segment": ["a"], "score": ["Score RF"], "gini": [0.55]})
        path = sbp.plot_booked_vs_rejected_discriminance(bvr, eb, tmp_path)
        assert path is not None and path.exists()

    def test_booked_vs_rejected_no_early_bad(self, tmp_path):
        bvr = pd.DataFrame({"segment": ["a"], "score": ["Score RF"], "gini": [0.4]})
        path = sbp.plot_booked_vs_rejected_discriminance(bvr, None, tmp_path)
        assert path is not None and path.exists()

    def test_truncation_comparison_writes_file(self, tmp_path):
        df = _trunc_df()[["segment", "score", "u_ratio"]]
        path = sbp.plot_truncation_comparison(df, df, tmp_path)
        assert path is not None and path.exists()

    def test_monthly_psi_writes_file(self, tmp_path):
        psi = pd.DataFrame({"month": ["2024-01", "2024-02"], "psi": [0.05, 0.12]})
        path = sbp.plot_monthly_psi({"Score RF": psi}, "segA", tmp_path)
        assert path is not None and path.exists()

    def test_temporal_acceptance_vs_gini_writes_file(self, tmp_path):
        rolling_gini = pd.DataFrame(
            {
                "segment": ["a"] * 5,
                "period": ["main"] * 5,
                "month": [f"2024-0{i + 1}" for i in range(5)],
                "score": ["Score RF"] * 5,
                "gini": [0.42, 0.45, 0.48, 0.43, 0.46],
            }
        )
        rolling_acceptance = pd.DataFrame(
            {
                "segment": ["a"] * 5,
                "period": ["main"] * 5,
                "month": [f"2024-0{i + 1}" for i in range(5)],
                "acceptance_rate": [0.50, 0.55, 0.60, 0.52, 0.58],
            }
        )
        path = sbp.plot_temporal_acceptance_vs_gini(rolling_gini, rolling_acceptance, tmp_path)
        assert path is not None and path.exists()

    def test_ri_gini_writes_file(self, tmp_path):
        ri_df = pd.DataFrame(
            {
                "segment": ["a"],
                "score": ["Score RF"],
                "ri_gini_mean": [0.48],
                "ri_gini_ci_lo": [0.44],
                "ri_gini_ci_hi": [0.52],
            }
        )
        thorndike_df = pd.DataFrame(
            {
                "segment": ["a"],
                "score": ["Score RF"],
                "observed_gini": [0.40],
                "corrected_gini": [0.50],
            }
        )
        path = sbp.plot_ri_gini(ri_df, thorndike_df, tmp_path)
        assert path is not None and path.exists()

    def test_cross_segment_scatter_writes_file(self, tmp_path):
        segment_data = [
            {"segment": "a", "score": "Score RF", "acceptance_rate": 0.5, "gini": 0.45},
            {"segment": "b", "score": "Score RF", "acceptance_rate": 0.6, "gini": 0.48},
            {"segment": "c", "score": "Score RF", "acceptance_rate": 0.7, "gini": 0.50},
        ]
        regression = {
            "slope": 0.15,
            "intercept": 0.38,
            "r_value": 0.99,
            "p_value": 0.05,
            "n_segments": 3,
        }
        path = sbp.plot_cross_segment_scatter(segment_data, regression, tmp_path)
        assert path is not None and path.exists()
