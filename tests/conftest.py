import numpy as np
import pytest
from astropy.io import fits

from cloudynight import AllskyCamera, AllskyImage


def create_sample_fits(filename, shape=(1040, 1392)):
    """Generate sample FITS file with random data."""
    data = np.random.random(shape)
    hdu = fits.PrimaryHDU(data)
    hdu.header["DATE-OBS"] = "2024-12-04T00:00:00"
    hdu.writeto(filename, overwrite=True)


def create_mask_fits(filename, shape=(1040, 1392)):
    """Generate binary mask FITS file."""
    data = np.random.choice([0, 1], size=shape)
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(filename, overwrite=True)


def get_fits_info(fits_path):
    """Retrieve basic information from FITS file."""
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
def sample_fits_info(tmp_path):
    """Fixture to create and provide sample FITS file info."""
    sample_path = tmp_path / "sample.fits"
    create_sample_fits(sample_path, shape=(1040, 1392))
    return get_fits_info(sample_path)


@pytest.fixture
def mask_fits_info(tmp_path):
    """Fixture to create and provide mask FITS file info."""
    mask_path = tmp_path / "mask.fits"
    create_mask_fits(mask_path, shape=(1040, 1392))
    return get_fits_info(mask_path)


@pytest.fixture
def sample_image(sample_fits_info):
    """Fixture to create an AllskyImage instance from sample FITS file."""
    _, data, header = sample_fits_info
    return AllskyImage(filename="test.fits", data=data, header=header)


@pytest.fixture
def sample_camera():
    """Fixture to create an AllskyCamera instance."""
    return AllskyCamera()


@pytest.fixture
def sample_subregions(sample_fits_info):
    """Fixture to create subregions matching the image dimensions."""
    info, _, _ = sample_fits_info
    shape = info["shape"]
    n_regions = 33  # Expected num of subreigions
    # Generate random subregions with sparse 1s
    return np.array([np.random.rand(*shape) > 0.95 for _ in range(n_regions)])


def pytest_addoption(parser):
    """Remove command-line options as they are no longer needed."""
    # Prev used to add --sample-fits and --mask-fits
    # Now tests are self-contained
    pass


@pytest.fixture
def large_sample_fits_info(tmp_path):
    """Fixture to create larger sample FITS file."""
    sample_path = tmp_path / "large_sample.fits"
    create_sample_fits(sample_path, shape=(1240, 1592))  # 200 pixels larger in each dim
    return get_fits_info(sample_path)
