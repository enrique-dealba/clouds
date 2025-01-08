import numpy as np
import pytest

from cloudynight.models.predictors import CloudPredictors
from cloudynight.visualization.overlay import (
    create_overlay_colors,
    get_colored_regions,
    get_segment_coordinates,
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


# TODO: Measure speed and optimize this, since slow
def test_get_regions(predictors, sample_image_data):
    """Test region extraction from image."""
    regions = predictors.get_regions(sample_image_data)
    assert len(regions) == 33  # Expected number of regions
    assert all(isinstance(v, list) for v in regions.values())
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


# TODO: This is also slow due to get_regions func
def test_region_consistency(predictors, sample_image_data):
    """Test consistency between regions and predictions."""
    regions = predictors.get_regions(sample_image_data)
    random_pred = predictors.predict_random(regions)
    thresh_pred = predictors.predict_threshold(regions)
    kde_pred = predictors.predict_kde(regions)

    assert len(random_pred) == len(thresh_pred) == len(kde_pred) == len(regions)


def test_kde_predictor_edge_cases(predictors):
    """Test KDE predictions with edge cases."""
    # Test with extreme values
    edge_regions = {
        1: [float("inf")] * 100,  # Very large values
        2: [float("-inf")] * 100,  # Very small values
        3: [0] * 100,  # Zeros
        4: [3500] * 100,  # Normal values
    }

    predictions = predictors.predict_kde(edge_regions)
    assert len(predictions) == len(edge_regions)
    assert all(isinstance(p, int) for p in predictions)
    assert all(p in [0, 1] for p in predictions)


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
