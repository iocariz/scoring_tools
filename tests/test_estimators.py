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
