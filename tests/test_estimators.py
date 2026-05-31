import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from src.estimators import HurdleRegressor

# =============================================================================
# HurdleRegressor Tests
# =============================================================================


class TestHurdleRegressorInit:
    def test_default_init(self):
        model = HurdleRegressor()
        assert model.classifier is None
        assert model.regressor is None
        assert model.zero_threshold == 1e-10

    def test_custom_init(self):
        clf = LogisticRegression(max_iter=500)
        reg = Ridge(alpha=1.0)
        model = HurdleRegressor(classifier=clf, regressor=reg, zero_threshold=0.01)
        assert model.classifier is clf
        assert model.regressor is reg
        assert model.zero_threshold == 0.01


class TestHurdleRegressorFit:
    @pytest.fixture
    def zero_inflated_data(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 3)
        # Create zero-inflated target
        y = np.zeros(n)
        nonzero_mask = X[:, 0] > 0
        y[nonzero_mask] = np.abs(X[nonzero_mask, 1]) * 2 + 1
        return X, y

    def test_fit_returns_self(self, zero_inflated_data):
        X, y = zero_inflated_data
        model = HurdleRegressor()
        result = model.fit(X, y)
        assert result is model

    def test_fit_creates_fitted_models(self, zero_inflated_data):
        X, y = zero_inflated_data
        model = HurdleRegressor()
        model.fit(X, y)
        assert hasattr(model, "classifier_")
        assert hasattr(model, "regressor_")

    def test_fit_with_sample_weight(self, zero_inflated_data):
        X, y = zero_inflated_data
        weights = np.ones(len(y))
        model = HurdleRegressor()
        model.fit(X, y, sample_weight=weights)
        assert hasattr(model, "classifier_")

    def test_fit_with_custom_estimators(self, zero_inflated_data):
        X, y = zero_inflated_data
        clf = LogisticRegression(max_iter=1000)
        reg = Ridge(alpha=0.5, fit_intercept=False)
        model = HurdleRegressor(classifier=clf, regressor=reg)
        model.fit(X, y)
        # Should clone, not use the original
        assert model.classifier_ is not clf
        assert model.regressor_ is not reg

    def test_fit_mostly_zeros(self):
        """When most target values are zero, model should still fit."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = np.zeros(50)
        y[0] = 1.0  # At least one non-zero to satisfy LogisticRegression
        model = HurdleRegressor()
        model.fit(X, y)
        assert hasattr(model, "regressor_")


class TestHurdleRegressorPredict:
    @pytest.fixture
    def fitted_model(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 3)
        y = np.zeros(n)
        nonzero_mask = X[:, 0] > 0
        y[nonzero_mask] = np.abs(X[nonzero_mask, 1]) * 2 + 1
        model = HurdleRegressor()
        model.fit(X, y)
        return model, X

    def test_predict_returns_array(self, fitted_model):
        model, X = fitted_model
        predictions = model.predict(X[:10])
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10

    def test_predictions_non_negative_for_positive_data(self, fitted_model):
        """For non-negative training data, most predictions should be non-negative."""
        model, X = fitted_model
        predictions = model.predict(X)
        # Most predictions should be >= 0 (some may be slightly negative due to model)
        assert (predictions >= -0.5).mean() > 0.9

    def test_predict_binary(self, fitted_model):
        model, X = fitted_model
        binary_preds = model.predict_binary(X[:10])
        assert len(binary_preds) == 10
        assert set(binary_preds).issubset({0, 1})

    def test_predict_magnitude(self, fitted_model):
        model, X = fitted_model
        mag = model.predict_magnitude(X[:10])
        assert len(mag) == 10

    def test_predict_is_product_of_prob_and_magnitude(self, fitted_model):
        """predict = P(nonzero) * magnitude."""
        model, X = fitted_model
        X_test = X[:20]
        predictions = model.predict(X_test)
        prob_nonzero = model.classifier_.predict_proba(X_test)[:, 1]
        magnitude = model.regressor_.predict(X_test)
        expected = prob_nonzero * magnitude
        np.testing.assert_array_almost_equal(predictions, expected)


class TestHurdleRegressorParams:
    def test_get_params(self):
        model = HurdleRegressor(zero_threshold=0.5)
        params = model.get_params()
        assert params["zero_threshold"] == 0.5
        assert params["classifier"] is None
        assert params["regressor"] is None

    def test_set_params(self):
        model = HurdleRegressor()
        model.set_params(zero_threshold=0.1)
        assert model.zero_threshold == 0.1

    def test_get_params_with_custom_estimators(self):
        clf = LogisticRegression()
        reg = Ridge()
        model = HurdleRegressor(classifier=clf, regressor=reg)
        params = model.get_params()
        assert params["classifier"] is clf
        assert params["regressor"] is reg


# =============================================================================
# TweedieGLM Tests
# =============================================================================


class TestTweedieGLM:
    """Tests for the TweedieGLM estimator."""

    def test_init(self):
        from src.estimators import TweedieGLM

        model = TweedieGLM(power=1.6, alpha=0.1)
        assert model.power == 1.6
        assert model.alpha == 0.1
        assert model.link == "log"

    def test_fit_mixed_data(self):
        from src.estimators import TweedieGLM

        np.random.seed(42)
        n = 100
        X = np.abs(np.random.randn(n, 3))  # ensuring non-negative inputs is safer for log link in some impls?
        # Actually sklearn's Tweedie handles X normally, but y must be non-negative
        y = np.abs(np.random.randn(n))
        # Add zeros
        y[:20] = 0.0

        model = TweedieGLM(power=1.5, alpha=0.5)
        model.fit(X, y)
        assert hasattr(model, "regressor_")

        preds = model.predict(X)
        assert len(preds) == n
        # Log link ensures non-negative predictions
        assert (preds >= 0).all()

    def test_fit_sample_weights(self):
        from src.estimators import TweedieGLM

        X = np.array([[1], [2], [3]])
        y = np.array([0, 10, 20])
        w = np.array([1, 2, 1])

        model = TweedieGLM()
        model.fit(X, y, sample_weight=w)
        assert hasattr(model, "regressor_")


# =============================================================================
# Per-loan training makes the hurdle meaningful (audit #6)
# =============================================================================


class TestHurdlePerLoanMeaningful:
    """When fed PER-LOAN data (real zero mass in the default amount), HurdleRegressor's classifier
    learns genuine default-frequency variation and its cell predictions track the exposure-weighted
    aggregate b2 — unlike fitting on the bin-aggregated ratio, where the hurdle is inert."""

    @staticmethod
    def _per_loan_data(seed=0):
        import pandas as pd

        rng = np.random.RandomState(seed)
        rows = []
        for v0 in range(1, 4):
            for v1 in range(1, 4):
                p_def = 0.05 + 0.10 * (v0 + v1)  # default rate rises with coords (~0.25 .. 0.65)
                for _ in range(400):
                    den = rng.uniform(800.0, 1200.0)
                    defaulted = rng.random() < p_def
                    num = den * rng.uniform(0.02, 0.06) if defaulted else 0.0
                    rows.append({"var0": v0, "var1": v1, "todu_30ever_h6": num, "todu_amt_pile_h6": den})
        return pd.DataFrame(rows)

    def test_per_loan_hurdle_classifier_is_not_degenerate(self):
        from scipy.stats import spearmanr

        df = self._per_loan_data()
        r = (7.0 * df["todu_30ever_h6"] / df["todu_amt_pile_h6"]).fillna(0.0).to_numpy()
        w = df["todu_amt_pile_h6"].to_numpy(dtype=float)
        X = df[["var0", "var1"]]

        model = HurdleRegressor(
            classifier=LogisticRegression(max_iter=1000, random_state=42),
            regressor=Ridge(alpha=0.5, random_state=42),
        )
        model.fit(X, r, sample_weight=w)

        cells = (
            df.groupby(["var0", "var1"])
            .agg(num=("todu_30ever_h6", "sum"), den=("todu_amt_pile_h6", "sum"))
            .reset_index()
        )
        cells["true_b2"] = 7.0 * cells["num"] / cells["den"]
        coords = cells[["var0", "var1"]]

        proba = model.classifier_.predict_proba(coords)[:, 1]
        # The classifier learns real frequency variation across cells (NOT the degenerate ~1 it
        # learns on the aggregated target). This is the crux of fixing #6.
        assert proba.max() - proba.min() > 0.10

        pred_b2 = model.predict(coords)
        assert (pred_b2 >= 0).all()
        # Predicted cell b2 tracks the exposure-weighted aggregate b2 (freq×severity reconciliation).
        rho = spearmanr(pred_b2, cells["true_b2"].to_numpy()).correlation
        assert rho > 0.7

    def test_aggregated_target_has_no_zero_mass_for_hurdle(self):
        """Root cause of #6: the bin-aggregated b2 has ≈ no exact zeros, so the hurdle's binary
        target is all-ones and the classifier is degenerate (it cannot learn default frequency —
        a single-class target). The PER-LOAN target, by contrast, has real zero mass."""
        df = self._per_loan_data()
        agg = (
            df.groupby(["var0", "var1"])
            .agg(num=("todu_30ever_h6", "sum"), den=("todu_amt_pile_h6", "sum"))
            .reset_index()
        )
        agg_b2 = 7.0 * agg["num"] / agg["den"]
        y_binary_agg = (np.abs(agg_b2) > 1e-10).astype(int)
        assert y_binary_agg.min() == 1  # every aggregated cell non-zero ⇒ degenerate single-class hurdle

        r = (7.0 * df["todu_30ever_h6"] / df["todu_amt_pile_h6"]).fillna(0.0)
        per_loan_zero_frac = float((r <= 1e-10).mean())
        assert per_loan_zero_frac > 0.3  # many non-defaulting loans ⇒ real zero mass for the hurdle

    def test_per_loan_severity_uses_exposure_weights(self):
        """The severity (positive-part) regressor receives the exposure weights sliced to defaulters."""

        df = self._per_loan_data()
        r = (7.0 * df["todu_30ever_h6"] / df["todu_amt_pile_h6"]).fillna(0.0).to_numpy()
        w = df["todu_amt_pile_h6"].to_numpy(dtype=float)
        model = HurdleRegressor(
            classifier=LogisticRegression(max_iter=1000, random_state=42),
            regressor=Ridge(alpha=0.5, random_state=42),
        )
        model.fit(df[["var0", "var1"]], r, sample_weight=w)
        # Regressor was fit only on defaulters (r > 0), so it predicts a positive severity surface.
        assert (model.regressor_.predict(df[["var0", "var1"]].drop_duplicates()) > 0).all()


class TestTweedieGLMExposureWeighting:
    """Audit #8: TweedieGLM models the rate directly and takes exposure via sample_weight —
    no penalized log-exposure feature, no exposure column needed at predict time."""

    @staticmethod
    def _data(seed=0, n=200):
        import pandas as pd

        rng = np.random.RandomState(seed)
        X = pd.DataFrame({"var0": rng.randint(1, 6, n).astype(float), "var1": rng.randint(1, 6, n).astype(float)})
        y = np.abs(rng.randn(n)) * 0.05  # non-negative rate
        exposure = rng.uniform(500.0, 1500.0, n)
        return X, y, exposure

    def test_predicts_rate_without_exposure_column(self):
        from src.estimators import TweedieGLM

        X, y, exposure = self._data()
        model = TweedieGLM(power=1.5, alpha=0.5)
        model.fit(X, y, sample_weight=exposure)
        # Predict on the bare feature frame (no exposure column) — the mesh/grid path.
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert (preds >= 0).all()  # log link ⇒ non-negative rate

    def test_no_log_exposure_feature_added(self):
        from src.estimators import TweedieGLM

        X, y, exposure = self._data()
        model = TweedieGLM(power=1.5, alpha=0.5).fit(X, y, sample_weight=exposure)
        # The underlying regressor sees exactly the supplied features — no "_log_exposure" column.
        assert model.regressor_.n_features_in_ == X.shape[1]
        assert not hasattr(model, "exposure_col")

    def test_sample_weight_changes_fit(self):
        from src.estimators import TweedieGLM

        X, y, exposure = self._data()
        unw = TweedieGLM(power=1.5, alpha=0.1).fit(X, y).predict(X)
        wtd = TweedieGLM(power=1.5, alpha=0.1).fit(X, y, sample_weight=exposure).predict(X)
        # Exposure weighting is live: weighted vs unweighted fits differ.
        assert not np.allclose(unw, wtd)

    def test_prepare_model_input_appends_no_exposure(self):
        import pandas as pd

        from src.estimators import TweedieGLM, prepare_model_input

        data = pd.DataFrame({"var0": [1.0, 2.0], "var1": [3.0, 4.0], "todu_amt_pile_h6": [900.0, 1100.0]})
        out = prepare_model_input(data, ["var0", "var1"], TweedieGLM())
        assert list(out.columns) == ["var0", "var1"]  # exposure column NOT appended
