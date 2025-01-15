from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cloudynight.models.predictors import CloudPredictors, calculate_metrics
from cloudynight.visualization.overlay import (
    create_overlay_colors,
    get_colored_regions,
    get_segment_coordinates,
    plot_with_overlay,
)


@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    # Create 500x500 image with some features
    data = np.zeros((500, 500))
    # Add some "clouds" - gaussian blobs
    y, x = np.ogrid[-250:250, -250:250]
    mask = x * x + y * y <= 200 * 200
    data[mask] = 3500  # Above threshold
    return data


@pytest.fixture
def sample_regions(sample_image_data):
    """Create sample regions dictionary."""
    return {
        1: [3500] * 100,  # Above threshold
        2: [3000] * 100,  # Below threshold
        3: [3600] * 100,  # Above threshold
    }


@pytest.fixture
def predictors(tmp_path):
    """Create sample KDE models and initialize predictors."""
    import pickle

    from sklearn.neighbors import KernelDensity

    # Simple KDE models
    X_0 = np.random.normal(3000, 100, 1000).reshape(-1, 1)
    X_1 = np.random.normal(3500, 100, 1000).reshape(-1, 1)

    kde_0 = KernelDensity().fit(X_0)
    kde_1 = KernelDensity().fit(X_1)

    model_path = tmp_path / "test_kde_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"kde_label_0": kde_0, "kde_label_1": kde_1}, f)

    return CloudPredictors(str(model_path))


def test_actual_kde_models_load():
    """Test that the current ver of KDE loads correctly."""
    import os

    from cloudynight.models.predictors import CloudPredictors

    # Get path to the actual models file used in production
    kde_model_path = os.path.join(
        os.path.dirname(__file__), "..", "cloudynight", "models", "kde_models.pkl"
    )

    # Verify file exists
    assert os.path.exists(
        kde_model_path
    ), f"KDE models file not found at {kde_model_path}"

    try:
        # Attempt to load the models
        predictors = CloudPredictors(kde_model_path)

        # Verify the required model attributes exist
        assert hasattr(predictors, "kde_label_0"), "kde_label_0 model not loaded"
        assert hasattr(predictors, "kde_label_1"), "kde_label_1 model not loaded"

        # Test that the models can actually make predictions
        test_value = 3300.0
        percent_0, percent_1 = predictors._get_kde_probability(test_value)

        # Verify the probabilities are valid
        assert isinstance(percent_0, float), "Invalid probability type for label 0"
        assert isinstance(percent_1, float), "Invalid probability type for label 1"
        assert 0 <= percent_0 <= 100, "Invalid probability range for label 0"
        assert 0 <= percent_1 <= 100, "Invalid probability range for label 1"

    except Exception as e:
        pytest.fail(f"Failed to load or use KDE models: {str(e)}")


def test_random_predictor(predictors, sample_regions):
    """Test random predictor always returns zeros."""
    predictions = predictors.predict_random(sample_regions)
    assert len(predictions) == len(sample_regions)
    assert all(p == 0 for p in predictions)


def test_threshold_predictor(predictors, sample_regions):
    """Test threshold-based predictions."""
    predictions = predictors.predict_threshold(sample_regions, threshold=3300)
    assert predictions == [1, 0, 1]  # Based on our sample data


def test_kde_predictor(predictors, sample_regions):
    """Test KDE-based predictions."""
    predictions = predictors.predict_kde(sample_regions)
    assert len(predictions) == len(sample_regions)
    assert all(isinstance(p, int) for p in predictions)
    assert all(p in [0, 1] for p in predictions)


def test_get_regions(predictors, sample_image_data):
    """Test region extraction from image."""
    regions = predictors.get_regions(sample_image_data)
    assert len(regions) == 33  # Expected number of regions
    assert all(isinstance(v, np.ndarray) for v in regions.values())
    assert all(all(isinstance(x, (int, float)) for x in v) for v in regions.values())


def test_segment_coordinates():
    """Test segment coordinate generation."""
    coords = get_segment_coordinates(250, 250, 100, 200, 0, 45)
    assert len(coords) > 0
    assert all(isinstance(c, tuple) and len(c) == 2 for c in coords)
    assert all(0 <= x <= 500 and 0 <= y <= 500 for y, x in coords)


def test_overlay_colors():
    """Test overlay color generation."""
    binary_list = [0, 1, 0, 1]
    colors = create_overlay_colors(binary_list)
    assert len(colors) == len(binary_list)
    assert all(len(c) == 4 for c in colors)  # RGBA format
    assert colors[0] == [0, 0, 0, 0]  # Clear for 0
    assert colors[1] == [255, 100, 100, 64]  # Red-ish for 1


def test_colored_regions(sample_image_data):
    """Test colored region generation."""
    overlay_colors = create_overlay_colors([0, 1, 0])
    colored = get_colored_regions(sample_image_data, overlay_colors)

    assert colored.shape[:2] == sample_image_data.shape
    assert colored.shape[2] == 4  # RGBA
    assert colored.dtype == np.uint8


def test_region_consistency(predictors, sample_image_data):
    """Test consistency between regions and predictions."""
    regions = predictors.get_regions(sample_image_data)
    random_pred = predictors.predict_random(regions)
    thresh_pred = predictors.predict_threshold(regions)
    kde_pred = predictors.predict_kde(regions)

    assert len(random_pred) == len(thresh_pred) == len(kde_pred) == len(regions)


def test_kde_predictor_inf_values(predictors):
    """Test KDE predictions with positive and negative infinity."""
    # Define edge regions with positive and negative infinity
    edge_regions = {
        "positive_inf": [float("inf")] * 100,  # Very large values (positive infinity)
        "negative_inf": [float("-inf")] * 100,  # Very small values (negative infinity)
    }

    # Mock warnings and logging to prevent cluttering test output
    with mock.patch("warnings.warn") as mock_warn, mock.patch(
        "logging.error"
    ) as mock_error:
        predictions = predictors.predict_kde(edge_regions)

        # Assert the number of predictions matches the number of regions
        assert len(predictions) == len(
            edge_regions
        ), "Number of predictions does not match number of regions."
        assert all(
            isinstance(p, int) for p in predictions
        ), "Not all predictions are integers."
        assert all(
            p in [0, 1] for p in predictions
        ), "Predictions contain values other than 0 or 1."

        predictions_dict = dict(zip(edge_regions.keys(), predictions))

        assert (
            predictions_dict["positive_inf"] == 0
        ), "Failed to handle positive infinity correctly."
        assert (
            predictions_dict["negative_inf"] == 0
        ), "Failed to handle negative infinity correctly."

        assert mock_warn.call_count == 2, "Expected two warnings for non-finite values."
        mock_warn.assert_any_call(
            "Inf. value encountered: inf. Assigning zero probabilities."
        )
        mock_warn.assert_any_call(
            "Inf. value encountered: -inf. Assigning zero probabilities."
        )
        assert mock_error.call_count == 0, "Unexpected errors were logged."


def test_visualization_pipeline(sample_image_data, predictors):
    """Test full visualization pipeline."""
    # Get predictions
    regions = predictors.get_regions(sample_image_data)
    predictions = predictors.predict_threshold(regions)

    # Create visualization
    overlay_colors = create_overlay_colors(predictions)
    colored = get_colored_regions(sample_image_data, overlay_colors)

    assert colored.shape[:2] == sample_image_data.shape
    assert colored.shape[2] == 4
    assert np.max(colored) <= 255
    assert np.min(colored) >= 0


def test_visualization_pipeline_changes():
    """Test that visualization changes produce expected output format and properties."""
    image_data = np.zeros((500, 500))  # Base image
    # Adds gaussian blob in center
    y, x = np.ogrid[-250:250, -250:250]
    mask = x * x + y * y <= 200 * 200
    image_data[mask] = 3500

    # Sample predictions (alternating clear/cloudy)
    test_predictions = [1, 0] * 16 + [1]  # 33 regions total

    # Create figure and axis for testing
    fig, ax = plt.subplots()

    # Generate overlay colors
    overlay_colors = create_overlay_colors(test_predictions)

    # Get overlay
    overlay = get_colored_regions(image_data, overlay_colors)

    # Test overlay properties
    assert overlay.shape == (
        *image_data.shape,
        4,
    ), "Overlay should be RGBA with same dimensions as input"
    assert overlay.dtype == np.uint8, "Overlay should be 8-bit unsigned integers"
    assert np.all(overlay[..., 3] <= 64), "Alpha channel should not exceed 64"

    # Test plotting
    plot_with_overlay(image_data, overlay_colors, ax)

    # Get the plotted images from the axis
    plotted_images = ax.get_images()
    assert (
        len(plotted_images) == 2
    ), "Should have exactly 2 image layers (base + overlay)"
    assert (
        plotted_images[0].get_cmap().name == "gray"
    ), "Base image should use gray colormap"
    assert (
        plotted_images[1].get_alpha() == 0.99
    ), "Overlay should have 0.99 alpha in plot"

    plt.close(fig)


def test_metrics_calculation_and_display():
    """Test metrics calculation and display functionality."""
    # Sample data
    test_ground_truth = [1, 0, 1, 1, 0] * 6 + [1, 0, 1]  # 33 values total
    test_predictions = {
        "random": [0] * 33,  # All clear
        "threshold": [1] * 33,  # All cloudy
        "kde": test_ground_truth.copy(),  # Perfect prediction
    }

    # Test metrics calculation for each method
    for method, preds in test_predictions.items():
        metrics = calculate_metrics(test_ground_truth, preds)

        assert isinstance(metrics, dict), f"Metrics for {method} should be a dictionary"
        assert all(
            key in metrics for key in ["accuracy", "precision", "recall", "f1"]
        ), f"Missing metrics for {method}"
        assert all(
            isinstance(v, float) for v in metrics.values()
        ), f"Non-float metrics found for {method}"

        # Verify specific cases with appropriate tolerances
        if method == "random":
            assert metrics["accuracy"] == pytest.approx(0.394, rel=1e-2)
        elif method == "threshold":
            assert metrics["recall"] == pytest.approx(1.0, abs=1e-3)
        elif method == "kde":
            assert metrics["accuracy"] == pytest.approx(1.0, abs=1e-3)
            assert metrics["f1"] == pytest.approx(1.0, abs=1e-3)
