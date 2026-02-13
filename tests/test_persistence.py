import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from src.persistence import (
    load_model_for_prediction,
    predict_on_new_data,
    save_model_with_metadata,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create simple training data with 3 features and 20 samples."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feature_a": np.random.randn(20),
            "feature_b": np.random.randn(20),
            "feature_c": np.random.randn(20),
        }
    )
    y = 2.0 * X["feature_a"] - 1.5 * X["feature_b"] + 0.5 * X["feature_c"] + np.random.randn(20) * 0.1
    return X, y


@pytest.fixture
def fitted_linear_model(sample_data):
    """Return a fitted LinearRegression model and its training data."""
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def sample_features():
    return ["feature_a", "feature_b", "feature_c"]


@pytest.fixture
def sample_metadata():
    return {
        "cv_mean_r2": 0.95,
        "cv_std_r2": 0.02,
        "full_r2": 0.96,
        "cv_folds": 5,
        "total_samples": 20,
    }


@pytest.fixture
def saved_model_path(tmp_path, fitted_linear_model, sample_features, sample_metadata):
    """Save a model to tmp_path and return the directory path."""
    model, X, y = fitted_linear_model
    model_dir = save_model_with_metadata(
        model=model,
        features=sample_features,
        metadata=sample_metadata,
        base_path=str(tmp_path / "models"),
    )
    return model_dir


# =============================================================================
# save_model_with_metadata Tests
# =============================================================================


class TestSaveModelWithMetadata:
    def test_creates_directory_structure(self, tmp_path, fitted_linear_model, sample_features, sample_metadata):
        """Verify that saving creates model.pkl, metadata.json, features.txt, and model_summary.txt."""
        model, X, y = fitted_linear_model
        model_dir = save_model_with_metadata(
            model=model,
            features=sample_features,
            metadata=sample_metadata,
            base_path=str(tmp_path / "models"),
        )

        model_path = os.path.join(model_dir, "model.pkl")
        metadata_path = os.path.join(model_dir, "metadata.json")
        features_path = os.path.join(model_dir, "features.txt")
        summary_path = os.path.join(model_dir, "model_summary.txt")

        assert os.path.isfile(model_path), "model.pkl not found"
        assert os.path.isfile(metadata_path), "metadata.json not found"
        assert os.path.isfile(features_path), "features.txt not found"
        assert os.path.isfile(summary_path), "model_summary.txt not found"

    def test_returns_string_path(self, tmp_path, fitted_linear_model, sample_features, sample_metadata):
        model, X, y = fitted_linear_model
        result = save_model_with_metadata(
            model=model,
            features=sample_features,
            metadata=sample_metadata,
            base_path=str(tmp_path / "models"),
        )
        assert isinstance(result, str)
        assert os.path.isdir(result)

    def test_features_file_lists_all_features(self, saved_model_path, sample_features):
        """Verify features.txt contains every feature name."""
        with open(os.path.join(saved_model_path, "features.txt")) as f:
            content = f.read()

        for feature in sample_features:
            assert feature in content, f"Feature '{feature}' not found in features.txt"

    def test_model_summary_contains_model_type(self, saved_model_path):
        """Verify model_summary.txt references the correct model type."""
        with open(os.path.join(saved_model_path, "model_summary.txt")) as f:
            content = f.read()

        assert "LinearRegression" in content

    def test_save_with_ridge_model(self, tmp_path, sample_data, sample_features, sample_metadata):
        """Verify saving works for a Ridge model as well."""
        X, y = sample_data
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        model_dir = save_model_with_metadata(
            model=model,
            features=sample_features,
            metadata=sample_metadata,
            base_path=str(tmp_path / "ridge_models"),
        )

        assert os.path.isfile(os.path.join(model_dir, "model.pkl"))

        with open(os.path.join(model_dir, "metadata.json")) as f:
            meta = json.load(f)
        assert meta["model_type"] == "Ridge"


# =============================================================================
# Round-trip save/load Tests
# =============================================================================


class TestRoundTrip:
    def test_predictions_match_after_round_trip(self, saved_model_path, fitted_linear_model, sample_features):
        """Save a model, load it, and verify predictions match the original."""
        original_model, X, y = fitted_linear_model
        original_predictions = original_model.predict(X[sample_features])

        loaded_model, metadata, features = load_model_for_prediction(saved_model_path)
        loaded_predictions = loaded_model.predict(X[features])

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_coefficients_match_after_round_trip(self, saved_model_path, fitted_linear_model):
        """Verify model coefficients survive the round trip."""
        original_model, X, y = fitted_linear_model

        loaded_model, _, _ = load_model_for_prediction(saved_model_path)

        np.testing.assert_array_almost_equal(original_model.coef_, loaded_model.coef_)
        np.testing.assert_almost_equal(original_model.intercept_, loaded_model.intercept_)


# =============================================================================
# load_model_for_prediction Tests
# =============================================================================


class TestLoadModelForPrediction:
    def test_returns_tuple_of_three(self, saved_model_path):
        """Verify the function returns a (model, metadata, features) tuple."""
        result = load_model_for_prediction(saved_model_path)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_correct_types(self, saved_model_path):
        """Verify each element of the returned tuple has the correct type."""
        model, metadata, features = load_model_for_prediction(saved_model_path)

        assert hasattr(model, "predict"), "Loaded model should have a predict method"
        assert isinstance(metadata, dict)
        assert isinstance(features, list)

    def test_features_match_saved(self, saved_model_path, sample_features):
        """Verify the loaded features list matches what was saved."""
        _, _, features = load_model_for_prediction(saved_model_path)
        assert features == sample_features

    def test_metadata_preserves_values(self, saved_model_path, sample_metadata):
        """Verify loaded metadata retains the original metric values."""
        _, metadata, _ = load_model_for_prediction(saved_model_path)

        assert metadata["cv_mean_r2"] == sample_metadata["cv_mean_r2"]
        assert metadata["cv_std_r2"] == sample_metadata["cv_std_r2"]
        assert metadata["cv_folds"] == sample_metadata["cv_folds"]


# =============================================================================
# predict_on_new_data Tests
# =============================================================================


class TestPredictOnNewData:
    def test_returns_predictions(self, saved_model_path, sample_features):
        """Verify predictions are returned as a numpy array."""
        new_data = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [0.5, 1.5, 2.5],
                "feature_c": [-1.0, 0.0, 1.0],
            }
        )
        predictions = predict_on_new_data(saved_model_path, new_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3

    def test_predictions_have_correct_shape(self, saved_model_path):
        """Verify the number of predictions matches the number of input rows."""
        np.random.seed(99)
        n_samples = 10
        new_data = pd.DataFrame(
            {
                "feature_a": np.random.randn(n_samples),
                "feature_b": np.random.randn(n_samples),
                "feature_c": np.random.randn(n_samples),
            }
        )
        predictions = predict_on_new_data(saved_model_path, new_data)
        assert predictions.shape == (n_samples,)

    def test_raises_on_missing_features(self, saved_model_path):
        """Verify ValueError is raised when required features are missing."""
        new_data = pd.DataFrame(
            {
                "feature_a": [1.0],
                "feature_b": [0.5],
                # feature_c is intentionally missing
            }
        )
        with pytest.raises(ValueError, match="Missing features"):
            predict_on_new_data(saved_model_path, new_data)

    def test_extra_columns_are_ignored(self, saved_model_path):
        """Verify that extra columns in new_data do not cause errors."""
        new_data = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0],
                "feature_b": [0.5, 1.5],
                "feature_c": [-1.0, 0.0],
                "extra_col": [99.0, 100.0],
            }
        )
        predictions = predict_on_new_data(saved_model_path, new_data)
        assert len(predictions) == 2


# =============================================================================
# Metadata Integrity Tests
# =============================================================================


class TestMetadataIntegrity:
    def test_metadata_has_expected_keys(self, saved_model_path):
        """Verify the saved metadata JSON contains all expected keys."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        expected_keys = [
            "timestamp",
            "model_type",
            "model_params",
            "features",
            "num_features",
            "python_version",
            "package_versions",
        ]
        for key in expected_keys:
            assert key in metadata, f"Expected key '{key}' not found in metadata"

    def test_num_features_matches_features_list(self, saved_model_path):
        """Verify num_features equals the length of the features list."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        assert metadata["num_features"] == len(metadata["features"])

    def test_model_type_is_correct(self, saved_model_path):
        """Verify model_type reflects the actual model class name."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        assert metadata["model_type"] == "LinearRegression"

    def test_timestamp_format(self, saved_model_path):
        """Verify timestamp follows the YYYYMMDD_HHMMSS format."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        timestamp = metadata["timestamp"]
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS
        assert timestamp[8] == "_"
        # Verify the date and time portions are numeric
        assert timestamp[:8].isdigit()
        assert timestamp[9:].isdigit()

    def test_user_metadata_is_preserved(self, saved_model_path, sample_metadata):
        """Verify user-supplied metadata keys are preserved in the saved file."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        for key, value in sample_metadata.items():
            assert key in metadata, f"User metadata key '{key}' missing"
            assert metadata[key] == value, f"User metadata key '{key}' has wrong value"

    def test_package_versions_present(self, saved_model_path):
        """Verify package_versions contains expected libraries."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        versions = metadata["package_versions"]
        assert isinstance(versions, dict)
        # At minimum scikit-learn and numpy should be tracked
        assert "scikit-learn" in versions
        assert "numpy" in versions

    def test_aggregated_data_flag(self, saved_model_path):
        """Verify the aggregated_data flag is set to True."""
        with open(os.path.join(saved_model_path, "metadata.json")) as f:
            metadata = json.load(f)

        assert metadata["aggregated_data"] is True
