import io

import pytest
from astropy.io import fits

from cloudynight import AllskyImage
from frontend.streamlit_app import ImageProcessor


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
