"""
Custom estimators for credit risk modeling.

This module provides specialized machine learning estimators:
- HurdleRegressor: Two-stage model for zero-inflated regression data
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge

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
    ...     regressor=Ridge(alpha=0.5, fit_intercept=False)
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
        # Initialize models if not provided
        if self.classifier is None:
            self.classifier_ = LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE)
        else:
            self.classifier_ = clone(self.classifier)

        if self.regressor is None:
            self.regressor_ = Ridge(alpha=0.5, fit_intercept=False)
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
        prob_nonzero = self.classifier_.predict_proba(X)[:, 1]

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
        """Get parameters for this estimator."""
        return {"classifier": self.classifier, "regressor": self.regressor, "zero_threshold": self.zero_threshold}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
