"""
Custom estimators for credit risk modeling.

This module provides specialized machine learning estimators:
- HurdleRegressor: Two-stage model for zero-inflated regression data
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils.validation import check_X_y

from src.constants import DEFAULT_RANDOM_STATE, DEFAULT_ZERO_THRESHOLD


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """
    Hurdle Regression for zero-inflated data.

    Two-stage model:
    1. Classification: Predict if value is zero or non-zero (logistic regression)
    2. Regression: Predict magnitude for non-zero values (linear regression)

    This is ideal when you have many zero values that create a flat regression surface.

    Parameters:
    -----------
    classifier : sklearn classifier, optional
        Model for predicting zero vs non-zero (default: LogisticRegression)
    regressor : sklearn regressor, optional
        Model for predicting non-zero values (default: Ridge)
    zero_threshold : float, optional
        Values below this are considered "zero" (default: 1e-10)

    Attributes:
    -----------
    classifier_ : fitted classifier
    regressor_ : fitted regressor

    Example:
    --------
    >>> from sklearn.linear_model import Ridge, LogisticRegression
    >>> hurdle = HurdleRegressor(
    ...     classifier=LogisticRegression(max_iter=1000),
    ...     regressor=Ridge(alpha=0.5, fit_intercept=True)
    ... )
    >>> hurdle.fit(X_train, y_train, sample_weight=weights_train)
    >>> predictions = hurdle.predict(X_test)
    """

    def __init__(self, classifier=None, regressor=None, zero_threshold=DEFAULT_ZERO_THRESHOLD):
        self.classifier = classifier
        self.regressor = regressor
        self.zero_threshold = zero_threshold

    def fit(self, X, y, sample_weight=None):
        """
        Fit the hurdle model.

        Args:
            X: Features
            y: Target variable
            sample_weight: Optional sample weights

        Returns:
            self
        """
        # Preserve feature names if X is a pandas DataFrame
        feature_names_in_ = None
        if hasattr(X, "columns"):
            feature_names_in_ = X.columns

        # Validate inputs (raises ValueError if NaNs are present)
        X, y = check_X_y(X, y, accept_sparse=True)

        # If we had feature names, optionally restore X to a DataFrame to keep names
        # for sklearn models that warn without them (like Lasso, Ridge)
        if feature_names_in_ is not None:
            import pandas as pd

            X = pd.DataFrame(X, columns=feature_names_in_)

        # Initialize models if not provided
        if self.classifier is None:
            self.classifier_ = LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE)
        else:
            self.classifier_ = clone(self.classifier)

        if self.regressor is None:
            self.regressor_ = Ridge(alpha=0.5, fit_intercept=True)
        else:
            self.regressor_ = clone(self.regressor)

        # Create binary target: 0 if zero, 1 if non-zero
        y_binary = (np.abs(y) > self.zero_threshold).astype(int)

        # Stage 1: Fit classifier (zero vs non-zero)
        self.classifier_.fit(X, y_binary, sample_weight=sample_weight)

        # Stage 2: Fit regressor on non-zero values only
        non_zero_mask = y_binary == 1

        if non_zero_mask.sum() > 0:
            X_nonzero = X[non_zero_mask]
            y_nonzero = y[non_zero_mask]
            weights_nonzero = sample_weight[non_zero_mask] if sample_weight is not None else None

            self.regressor_.fit(X_nonzero, y_nonzero, sample_weight=weights_nonzero)
        else:
            # Fallback: if no non-zero values, just fit on all data
            self.regressor_.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """
        Predict using the hurdle model.

        Process:
        1. Predict probability of non-zero
        2. Predict magnitude if non-zero
        3. Combine: P(non-zero) * predicted_magnitude

        Args:
            X: Features

        Returns:
            predictions: Array of predictions
        """
        # Stage 1: Predict probability of being non-zero
        # Handle single-class case where predict_proba returns 1-column array
        proba = self.classifier_.predict_proba(X)
        if proba.shape[1] == 1:
            # Classifier saw only one class during training
            if self.classifier_.classes_[0] == 1:
                prob_nonzero = np.ones(len(X))
            else:
                prob_nonzero = np.zeros(len(X))
        else:
            prob_nonzero = proba[:, 1]

        # Stage 2: Predict magnitude for all observations
        magnitude = self.regressor_.predict(X)

        # Combine: Expected value = P(non-zero) * E(Y | Y > 0)
        predictions = prob_nonzero * magnitude

        return predictions

    def predict_binary(self, X):
        """Predict binary outcome (zero vs non-zero) only."""
        return self.classifier_.predict(X)

    def predict_magnitude(self, X):
        """Predict magnitude (without zero-inflation adjustment)."""
        return self.regressor_.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        When deep=True, returns nested estimator parameters (e.g., classifier__C)
        for full sklearn compatibility with GridSearchCV/clone.
        """
        params = {"classifier": self.classifier, "regressor": self.regressor, "zero_threshold": self.zero_threshold}
        if deep:
            if self.classifier is not None:
                for key, value in self.classifier.get_params(deep=True).items():
                    params[f"classifier__{key}"] = value
            if self.regressor is not None:
                for key, value in self.regressor.get_params(deep=True).items():
                    params[f"regressor__{key}"] = value
        return params

    def set_params(self, **params):
        """Set parameters for this estimator, including nested estimator params."""
        nested = {}
        for key, value in params.items():
            if "__" in key:
                prefix, sub_key = key.split("__", 1)
                nested.setdefault(prefix, {})[sub_key] = value
            else:
                setattr(self, key, value)
        if "classifier" in nested and self.classifier is not None:
            self.classifier.set_params(**nested["classifier"])
        if "regressor" in nested and self.regressor is not None:
            self.regressor.set_params(**nested["regressor"])
        return self


class TweedieGLM(BaseEstimator, RegressorMixin):
    """
    Tweedie Generalized Linear Model (GLM) for zero-inflated data.

    This model uses the Tweedie distribution with a log-link function, which is
    particularly effective for modeling semi-continuous data with a mass at zero
    (e.g., insurance claims, credit risk exposure).

    When ``exposure_col`` is set, the model internally converts a ratio target
    (``multiplier * numerator / exposure``) back to the numerator, fits the
    Tweedie on the numerator (the natural scale for Compound Poisson-Gamma),
    and converts predictions back to the ratio scale.  This is the standard
    actuarial approach for rate models and properly handles zero-inflation.

    Parameters:
    -----------
    power : float, default=1.5
        Power parameter for the Tweedie distribution.
        - 1 < power < 2: Compound Poisson-Gamma (zero-inflated continuous).
        - power = 0: Normal distribution.
        - power = 1: Poisson distribution.
        - power = 2: Gamma distribution.
        - power = 3: Inverse Gaussian distribution.
    alpha : float, default=0.5
        Constant that multiplies the penalty terms (regularization).
    link : str, default='log'
        Link function to use. 'log' ensures non-negative predictions.
    max_iter : int, default=100
        Maximum number of iterations.
    exposure_col : str or None, default=None
        Name of the exposure column in X.  When set, X must be a DataFrame
        containing this column.  The column is extracted as exposure, removed
        from the features, and used to convert between ratio and numerator
        targets internally.
    multiplier : float, default=7.0
        Risk multiplier used in the ratio target formula
        (``target = multiplier * numerator / exposure``).

    Attributes:
    -----------
    regressor_ : fitted TweedieRegressor from sklearn
    """

    def __init__(self, power=1.5, alpha=0.5, link="log", max_iter=100, exposure_col=None, multiplier=7.0):
        self.power = power
        self.alpha = alpha
        self.link = link
        self.max_iter = max_iter
        self.exposure_col = exposure_col
        self.multiplier = multiplier

    def _split_exposure(self, X):
        """Extract exposure column from X if available.

        Returns (X_features, exposure) where exposure is None when not applicable.
        """
        if self.exposure_col and hasattr(X, "columns") and self.exposure_col in X.columns:
            exposure = X[self.exposure_col].values.astype(float)
            X_features = X.drop(columns=[self.exposure_col])
            return X_features, exposure
        return X, None

    def _add_log_exposure(self, X_features, exposure):
        """Add log(exposure) as a feature to approximate the GLM offset term."""
        import pandas as pd

        if isinstance(X_features, pd.DataFrame):
            X_aug = X_features.copy()
            X_aug["_log_exposure"] = np.log(np.maximum(exposure, 1e-10))
        else:
            log_exp = np.log(np.maximum(exposure, 1e-10)).reshape(-1, 1)
            X_aug = np.hstack([X_features, log_exp])
        return X_aug

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Tweedie GLM model.

        When ``exposure_col`` is set, the model converts the ratio target to
        the numerator (the natural scale for Tweedie compound Poisson-Gamma)
        and adds ``log(exposure)`` as a feature to approximate the GLM offset
        term.  This lets sklearn's TweedieRegressor learn the exposure
        relationship (coefficient should converge to ~1).

        Args:
            X: Features (DataFrame; may include exposure column)
            y: Target variable (ratio scale when using exposure offset)
            sample_weight: Optional sample weights

        Returns:
            self
        """
        from sklearn.linear_model import TweedieRegressor

        self.regressor_ = TweedieRegressor(power=self.power, alpha=self.alpha, link=self.link, max_iter=self.max_iter)

        X_features, exposure = self._split_exposure(X)

        if exposure is not None:
            # Convert ratio target back to numerator:
            #   y_ratio = multiplier * numerator / exposure
            #   => numerator = y_ratio * exposure / multiplier
            y_numerator = np.asarray(y, dtype=float) * exposure / self.multiplier
            # Add log(exposure) as feature to approximate the offset term
            X_aug = self._add_log_exposure(X_features, exposure)
            # Fit on numerator; don't pass sample_weight (exposure is in the offset)
            self.regressor_.fit(X_aug, y_numerator)
            self._fitted_with_exposure = True
            self._median_exposure = float(np.median(exposure))
        else:
            self.regressor_.fit(X, y, sample_weight=sample_weight)
            self._fitted_with_exposure = False
            self._median_exposure = 1.0

        return self

    def predict(self, X):
        """
        Predict using the Tweedie GLM model.

        When fitted with exposure offset, predictions are converted back to the
        ratio scale.  If the exposure column is not present at predict time
        (e.g. for mesh-grid visualization), a median exposure from training is
        used as a fallback.

        Args:
            X: Features (DataFrame; may include exposure column)

        Returns:
            predictions: Array of predictions on the ratio (b2_ever_h6) scale
        """
        X_features, exposure = self._split_exposure(X)

        if self._fitted_with_exposure:
            if exposure is not None:
                X_aug = self._add_log_exposure(X_features, exposure)
                raw_pred = self.regressor_.predict(X_aug)
                return self.multiplier * raw_pred / exposure
            # Fallback: use median exposure from training for approximate rate
            X_aug = self._add_log_exposure(X_features, np.full(len(X_features), self._median_exposure))
            raw_pred = self.regressor_.predict(X_aug)
            return self.multiplier * raw_pred / self._median_exposure

        return self.regressor_.predict(X_features)


def prepare_model_input(data, features, model):
    """Prepare feature DataFrame, appending exposure column for TweedieGLM.

    For non-Tweedie models (or TweedieGLM without exposure_col), this simply
    returns ``data[features]``.  For exposure-aware TweedieGLM, the exposure
    column is appended so that :meth:`TweedieGLM.fit` / :meth:`TweedieGLM.predict`
    can extract it internally.
    """
    cols = list(features)
    if isinstance(model, TweedieGLM) and model.exposure_col:
        if model.exposure_col not in cols and hasattr(data, "columns") and model.exposure_col in data.columns:
            cols = cols + [model.exposure_col]
    return data[cols]
