import numpy as np

from cloudynight.cloudynight import AllskyImage


def test_generate_subregions_dimensions(sample_camera, sample_image, sample_fits_info):
    """Test that generated subregions match image dimensions"""
    info, _, _ = sample_fits_info
    sample_camera.maskdata = sample_image
    n_regions = sample_camera.generate_subregions()

    assert n_regions > 0, "No subregions were generated"

    for i, subregion in enumerate(sample_camera.subregions):
        assert (
            subregion.shape == info["shape"]
        ), f"Generated subregion {i} shape {subregion.shape} does not match image shape {info['shape']}"


def test_mask_generation(sample_camera, sample_image, mask_fits_info):
    """Test mask generation and verify against provided mask"""
    info, mask_data, _ = mask_fits_info

    # Generate mask
    sample_camera.imgdata = [sample_image]
    generated_mask = sample_camera.generate_mask(mask_lt=3400)

    assert (
        generated_mask.data.shape == info["shape"]
    ), f"Generated mask shape {generated_mask.data.shape} does not match expected shape {info['shape']}"

    # Compare with provided mask (optional)
    correlation = np.corrcoef(generated_mask.data.flatten(), mask_data.flatten())[0, 1]
    print(f"\nMask correlation coefficient: {correlation:.3f}")


def test_camera_process_mismatched_images(
    sample_camera, large_sample_fits_info, mask_fits_info
):
    """Test camera processing with mismatched image and mask dimensions"""
    # Create larger image
    large_image = AllskyImage(
        filename="0001.fits",  # Use a proper numbered filename
        data=large_sample_fits_info[1],
        header=large_sample_fits_info[2],
    )

    # Set mask
    sample_camera.maskdata = AllskyImage(
        filename="mask", data=mask_fits_info[1], header=mask_fits_info[2]
    )

    # Generate subregions before processing
    sample_camera.generate_subregions()

    # Set image data
    sample_camera.imgdata = [large_image]

    # Process data
    file_idx = sample_camera.process_and_upload_data(no_upload=True)

    # Verify processed image matches mask dimensions
    assert sample_camera.imgdata[0].data.shape == sample_camera.maskdata.data.shape
    assert file_idx == 1  # Verify the file index was properly extracted
