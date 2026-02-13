"""Tests for SHAP model interpretability."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge

from src.inference_optimized import _compute_shap_values
from src.plots import plot_shap_summary


def _make_synthetic_model_and_data(n_samples: int = 100, n_features: int = 4, seed: int = 42):
    """Create a fitted Ridge model with synthetic data."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    # Target is a known linear combination + noise
    y = 3 * X["feature_0"] - 2 * X["feature_1"] + 0.5 * X["feature_2"] + rng.randn(n_samples) * 0.1

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return model, X, list(X.columns)


class TestComputeShapValues:
    def test_returns_dict_for_linear_model(self):
        model, X, features = _make_synthetic_model_and_data()
        result = _compute_shap_values(model, X, features)

        assert result is not None
        assert "shap_values" in result
        assert "feature_names" in result
        assert "mean_abs_shap" in result

    def test_shap_values_shape(self):
        model, X, features = _make_synthetic_model_and_data(n_samples=50, n_features=3)
        result = _compute_shap_values(model, X, features)

        assert result is not None
        assert result["shap_values"].shape == (50, 3)
        assert len(result["mean_abs_shap"]) == 3

    def test_feature_names_preserved(self):
        model, X, features = _make_synthetic_model_and_data()
        result = _compute_shap_values(model, X, features)

        assert result is not None
        assert result["feature_names"] == features

    def test_most_important_feature_first(self):
        """Feature_0 has the largest coefficient (3.0), so it should have highest SHAP."""
        model, X, features = _make_synthetic_model_and_data(n_samples=200)
        result = _compute_shap_values(model, X, features)

        assert result is not None
        most_important_idx = np.argmax(result["mean_abs_shap"])
        assert features[most_important_idx] == "feature_0"

    def test_returns_none_on_failure(self):
        """Should return None (not raise) if SHAP fails."""

        class BadModel:
            pass  # No predict, no coef_

        X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _compute_shap_values(BadModel(), X, ["a", "b"])

        assert result is None


class TestPlotShapSummary:
    def test_returns_figure(self):
        shap_values = np.array([[0.1, -0.2, 0.3], [0.4, -0.1, 0.2]])
        features = ["feat_a", "feat_b", "feat_c"]

        fig = plot_shap_summary(shap_values, features)
        assert isinstance(fig, go.Figure)

    def test_saves_html(self, tmp_path):
        shap_values = np.array([[0.1, -0.2], [0.4, -0.1]])
        features = ["feat_a", "feat_b"]
        path = str(tmp_path / "shap.html")

        plot_shap_summary(shap_values, features, output_path=path)
        assert (tmp_path / "shap.html").exists()

    def test_sorted_by_importance(self):
        # feat_a has mean |shap| = 0.25, feat_b = 0.15, feat_c = 0.5
        shap_values = np.array([[0.1, -0.2, 0.6], [0.4, -0.1, 0.4]])
        features = ["feat_a", "feat_b", "feat_c"]

        fig = plot_shap_summary(shap_values, features)
        # The bars are sorted ascending, so the last bar (top) should be feat_c
        bar_y = fig.data[0].y
        assert bar_y[-1] == "feat_c"  # Most important at top
        assert bar_y[0] == "feat_b"  # Least important at bottom
