import numpy as np
import pytest
from astropy.io import fits

from cloudynight import AllskyCamera, AllskyImage


def get_fits_info(fits_path):
    """Get basic information about a FITS file"""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        info = {
            "shape": data.shape,
            "dtype": data.dtype,
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "std": np.std(data),
        }
    return info, data, header


@pytest.fixture
def sample_fits_info(request):
    """Get information about the sample FITS file"""
    sample_path = request.config.getoption("--sample-fits")
    if not sample_path:
        pytest.skip("No sample FITS file provided")
    return get_fits_info(sample_path)


@pytest.fixture
def mask_fits_info(request):
    """Get information about the mask FITS file"""
    mask_path = request.config.getoption("--mask-fits")
    if not mask_path:
        pytest.skip("No mask FITS file provided")
    return get_fits_info(mask_path)


@pytest.fixture
def sample_image(sample_fits_info):
    """Create sample AllskyImage instance from actual FITS file"""
    _, data, header = sample_fits_info
    return AllskyImage(filename="test.fits", data=data, header=header)


@pytest.fixture
def sample_camera():
    """Create sample AllskyCamera instance"""
    return AllskyCamera()


@pytest.fixture
def sample_subregions(sample_fits_info):
    """Create sample subregions matching actual image dimensions"""
    info, _, _ = sample_fits_info
    shape = info["shape"]
    n_regions = 33  # Expected number from original error
    return np.array([np.random.rand(*shape) > 0.5 for _ in range(n_regions)])


def pytest_addoption(parser):
    """Add command line options for test FITS files"""
    parser.addoption(
        "--sample-fits", action="store", default=None, help="Path to sample FITS file"
    )
    parser.addoption(
        "--mask-fits", action="store", default=None, help="Path to mask FITS file"
    )
