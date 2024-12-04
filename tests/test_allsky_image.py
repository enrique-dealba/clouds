import numpy as np


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
    original_shape = sample_image.data.shape
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
