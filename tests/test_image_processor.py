import io

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from cloudynight import AllskyImage
from frontend.streamlit_app import ImageProcessor, visualize_image
from tests.conftest import create_sample_fits, get_fits_info


class MockUploadedFile:
    """Mock for StreamLit's UploadedFile."""

    def __init__(self, name, data, header):
        self.name = name
        self.data = data
        self.header = header

    def read(self):
        """Simulate file read by returning FITS data."""
        # Create FITS file in memory
        hdu = fits.PrimaryHDU(self.data)
        for key, value in self.header.items():
            hdu.header[key] = value

        # Write to bytes buffer
        buffer = io.BytesIO()
        hdu.writeto(buffer)
        return buffer.getvalue()


@pytest.fixture
def processor():
    """Create ImageProcessor instance."""
    return ImageProcessor()


@pytest.fixture
def mock_fits_file(sample_fits_info):
    """Create mock FITS file."""
    info, data, header = sample_fits_info
    return MockUploadedFile("test.fits", data, header)


@pytest.fixture
def mock_large_fits_file(large_sample_fits_info):
    """Create mock large FITS file."""
    info, data, header = large_sample_fits_info
    return MockUploadedFile("large.fits", data, header)


@pytest.fixture
def mock_small_fits_file(tmp_path):
    """Create mock small FITS file."""
    small_shape = (800, 900)  # Smaller than standard
    sample_path = tmp_path / "small_sample.fits"
    create_sample_fits(sample_path, shape=small_shape)
    info, data, header = get_fits_info(sample_path)
    return MockUploadedFile("small.fits", data, header)


@pytest.fixture
def large_fits_data():
    """Create a 2048x2048 test FITS data array."""
    data = np.zeros((2048, 2048))
    # Add a test pattern - diagonal gradient
    x, y = np.meshgrid(np.linspace(0, 1, 2048), np.linspace(0, 1, 2048))
    data = x + y
    # Add some "stars"
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    stars = rng.choice(data.size, 1000, replace=False)
    data.ravel()[stars] = rng.uniform(5, 10, 1000)
    return data


def test_load_fits_file_dimensions(processor, mock_fits_file, mock_large_fits_file):
    """Test loading FITS files of different dimensions."""
    # Load standard size file
    data, header = processor.load_fits_file(mock_fits_file)
    assert data is not None
    assert data.shape == (1040, 1392)  # Standard size

    # Load large file
    data, header = processor.load_fits_file(mock_large_fits_file)
    assert data is not None
    assert data.shape == (1240, 1592)  # Large size


def test_process_multiple_images_with_mask(
    processor, mock_fits_file, mock_large_fits_file, mask_fits_info
):
    """Test processing multiple images with mask of different size."""
    # Set up mask first
    processor.mask = AllskyImage("mask", mask_fits_info[1], mask_fits_info[2])
    processor.mask_data = mask_fits_info[1]

    # Process both standard and large files
    images = processor.process_multiple_images([mock_fits_file, mock_large_fits_file])

    assert len(images) == 2
    # Both images should be resized to match mask
    for img in images:
        assert img.data.shape == processor.mask.data.shape


def test_ensure_dimensions_match(processor, mock_large_fits_file, mask_fits_info):
    """Test dimension matching between current image and mask."""
    # Set up mask
    processor.mask = AllskyImage("mask", mask_fits_info[1], mask_fits_info[2])
    processor.mask_data = mask_fits_info[1]

    # Load and set current image (larger than mask)
    data, header = processor.load_fits_file(mock_large_fits_file)
    processor.current_image = AllskyImage("large.fits", data, header)

    # Initial dimensions should be different
    assert processor.current_image.data.shape != processor.mask.data.shape

    # Ensure dimensions match
    assert processor.ensure_dimensions_match()

    # Verify dimensions now match
    assert processor.current_image.data.shape == processor.mask.data.shape


def test_small_image_fails_gracefully(processor, mock_small_fits_file, mask_fits_info):
    """Test that smaller images are handled gracefully."""
    # Set up mask
    processor.mask = AllskyImage("mask", mask_fits_info[1], mask_fits_info[2])
    processor.mask_data = mask_fits_info[1]

    # Process smaller file
    images = processor.process_multiple_images([mock_small_fits_file])

    # Should get empty list as small image should be skipped
    assert len(images) == 0


def test_create_subregions_with_mismatched_dimensions(
    processor, mock_large_fits_file, mask_fits_info
):
    """Test subregion creation with mismatched dimensions."""
    # Set up mask
    processor.mask = AllskyImage("mask", mask_fits_info[1], mask_fits_info[2])
    processor.mask_data = mask_fits_info[1]

    # Load and set current image (larger than mask)
    data, header = processor.load_fits_file(mock_large_fits_file)
    processor.current_image = AllskyImage("large.fits", data, header)

    # Create subregions
    assert processor.create_subregions()

    # Verify dimensions of everything match
    assert processor.current_image.data.shape == processor.mask.data.shape
    assert processor.current_image.data.shape == processor.camera.maskdata.data.shape
    for subregion in processor.current_image.subregions:
        assert subregion.shape == processor.mask.data.shape


def test_visualize_large_fits(large_fits_data):
    """Test basic visualization of 2048x2048 FITS image."""
    buf = visualize_image(large_fits_data, "Large FITS Test")
    assert buf is not None

    # Verify we can open the output image
    img = Image.open(buf)
    assert img.size[0] > 0
    assert img.size[1] > 0

    height, width = large_fits_data.shape
    assert width == 2048  # Verify test data size
    assert height == 2048


def test_visualize_large_fits_with_overlay(large_fits_data):
    """Test visualization with overlay for 2048x2048 FITS image."""
    # Create test overlay same size as data
    overlay = np.zeros((2048, 2048))
    overlay[500:1500, 500:1500] = 1  # Add center square region

    buf = visualize_image(large_fits_data, "Large FITS with Overlay", overlay=overlay)
    assert buf is not None

    # Verify output image
    img = Image.open(buf)
    assert img.size[0] > 0
    assert img.size[1] > 0

    height, width = large_fits_data.shape
    assert width == overlay.shape[1]
    assert height == overlay.shape[0]
