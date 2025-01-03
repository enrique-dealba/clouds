import numpy as np
import pytest

from cloudynight.cloudynight import AllskyImage
from tests.conftest import create_sample_fits, get_fits_info


def test_image_info(sample_fits_info):
    """Display and verify basic image information"""
    info, data, header = sample_fits_info
    print("\nSample FITS Information:")
    print(f"Shape: {info['shape']}")
    print(f"Data type: {info['dtype']}")
    print(f"Value range: [{info['min']}, {info['max']}]")
    print(f"Mean ± std: {info['mean']:.2f} ± {info['std']:.2f}")
    assert data is not None
    assert len(info["shape"]) == 2  # Ensure 2D image


def test_mask_info(mask_fits_info):
    """Display and verify mask information"""
    info, data, header = mask_fits_info
    print("\nMask FITS Information:")
    print(f"Shape: {info['shape']}")
    print(f"Unique values: {np.unique(data)}")  # Should be [0, 1]
    assert data is not None
    assert len(info["shape"]) == 2  # Ensure 2D mask
    assert np.all(np.unique(data) == np.array([0, 1]))  # Binary mask


def test_image_dimensions_after_crop(sample_image, sample_fits_info):
    """Test that cropping maintains expected dimensions"""
    info, _, _ = sample_fits_info
    sample_image.crop_image()
    assert (
        sample_image.data.shape == info["shape"]
    ), f"Expected shape {info['shape']}, got {sample_image.data.shape}"


def test_subregion_dimension_match(sample_image, sample_subregions):
    """Test that subregions match image dimensions"""
    sample_image.subregions = sample_subregions

    for i, subregion in enumerate(sample_image.subregions):
        assert (
            subregion.shape == sample_image.data.shape
        ), f"Subregion {i} shape {subregion.shape} does not match image shape {sample_image.data.shape}"


def test_create_overlay_dimensions(sample_image, sample_subregions):
    """Test overlay creation maintains correct dimensions"""
    sample_image.subregions = sample_subregions

    overlay = sample_image.create_overlay(overlaytype="subregions")
    assert (
        overlay.shape == sample_image.data.shape
    ), f"Overlay shape {overlay.shape} does not match image shape {sample_image.data.shape}"


def test_resize_to_mask_larger_image(sample_image, mask_fits_info):
    """Test resizing when image is larger than mask"""
    # Make image larger than mask
    sample_image.data = np.pad(sample_image.data, ((0, 100), (0, 100)), mode="constant")
    original_shape = sample_image.data.shape
    mask_shape = mask_fits_info[0]["shape"]

    sample_image.resize_to_mask(mask_shape)

    assert sample_image.data.shape == mask_shape
    assert sample_image.data.shape[0] < original_shape[0]
    assert sample_image.data.shape[1] < original_shape[1]


def test_resize_to_mask_smaller_image(sample_image, mask_fits_info):
    """Test that error is raised when image is smaller than mask"""
    # Make image smaller than mask
    sample_image.data = sample_image.data[:-100, :-100]
    mask_shape = mask_fits_info[0]["shape"]

    with pytest.raises(ValueError) as excinfo:
        sample_image.resize_to_mask(mask_shape)
    assert "Image dimensions" in str(excinfo.value)
    assert "are smaller than mask dimensions" in str(excinfo.value)


def test_apply_mask_dimension_mismatch(sample_image, mask_fits_info):
    """Test applying mask with different dimensions"""
    # Make image larger than mask
    sample_image.data = np.pad(sample_image.data, ((0, 100), (0, 100)), mode="constant")
    original_shape = sample_image.data.shape

    mask = AllskyImage(
        filename="mask", data=mask_fits_info[1], header=mask_fits_info[2]
    )

    sample_image.apply_mask(mask)

    assert sample_image.data.shape == mask.data.shape
    assert sample_image.data.shape[0] < original_shape[0]
    assert sample_image.data.shape[1] < original_shape[1]


@pytest.fixture
def large_sample_fits_info(tmp_path):
    """Fixture to create a larger sample FITS file"""
    sample_path = tmp_path / "large_sample.fits"
    create_sample_fits(
        sample_path, shape=(1240, 1592)
    )  # 200 pixels larger in each dimension
    return get_fits_info(sample_path)


def test_process_larger_fits_file(large_sample_fits_info, mask_fits_info):
    """Test processing a FITS file larger than the mask"""
    large_image = AllskyImage(
        filename="large.fits",
        data=large_sample_fits_info[1],
        header=large_sample_fits_info[2],
    )

    mask = AllskyImage(
        filename="mask", data=mask_fits_info[1], header=mask_fits_info[2]
    )

    # Apply mask to larger image
    large_image.apply_mask(mask)

    assert large_image.data.shape == mask.data.shape


def test_resize_to_mask_centers_properly(sample_image, mask_fits_info):
    """Test that resize_to_mask performs a center crop."""
    # Make image larger with distinct values
    larger_shape = (sample_image.data.shape[0] + 200, sample_image.data.shape[1] + 200)
    larger_data = np.zeros(larger_shape)
    # Put a recognizable pattern in the center
    center_y = larger_shape[0] // 2
    center_x = larger_shape[1] // 2
    larger_data[center_y, center_x] = 100  # Distinctive center value
    sample_image.data = larger_data

    mask_shape = mask_fits_info[0]["shape"]
    sample_image.resize_to_mask(mask_shape)

    # Check if center pixel is preserved
    new_center_y = sample_image.data.shape[0] // 2
    new_center_x = sample_image.data.shape[1] // 2
    assert sample_image.data[new_center_y, new_center_x] == 100
