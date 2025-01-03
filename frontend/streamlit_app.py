import io
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits

from cloudynight import AllskyCamera, AllskyImage


class ImageProcessor:
    """Handles processing of astronomical images for cloud detection."""

    def __init__(self):
        self.camera = AllskyCamera()
        self.mask: Optional[AllskyImage] = None
        self.mask_data: Optional[np.ndarray] = None
        self.current_image: Optional[AllskyImage] = None

    def load_fits_file(
        self, uploaded_file
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load FITS file safely and return data and header."""
        try:
            # Read the uploaded file's bytes
            file_bytes = uploaded_file.read()

            # For bz2 files, decompress first
            if uploaded_file.name.endswith(".bz2"):
                import bz2

                file_bytes = bz2.decompress(file_bytes)

            # Create file-like object in memory
            file_buffer = io.BytesIO(file_bytes)

            with fits.open(file_buffer) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header

                if data is None or np.all(data == 0):
                    st.error(f"File {uploaded_file.name} appears to be empty")
                    return None, None

                return data, header

        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            return None, None
        finally:
            # Reset the file buffer for potential reuse
            if "file_buffer" in locals():
                file_buffer.close()

    def ensure_dimensions_match(self) -> bool:
        """Checks if current img dimensions match mask dimensions."""
        if self.current_image is None or self.mask is None:
            return False

        try:
            if self.current_image.data.shape != self.mask.data.shape:
                self.current_image.resize_to_mask(self.mask.data.shape)
                return True
        except ValueError as e:
            st.error(f"Cannot match dimensions: {str(e)}")
            return False

        return True

    def process_multiple_images(self, uploaded_files) -> List[AllskyImage]:
        """Process multiple FITS files into AllskyImage objects."""
        images = []
        expected_shape = None

        for file in uploaded_files:
            data, header = self.load_fits_file(file)
            if data is not None and header is not None:
                img = AllskyImage(
                    filename=file.name,
                    data=data,
                    header=header,
                )
                try:
                    # If we have a mask, resize to match mask dimensions
                    if self.mask is not None:
                        img.resize_to_mask(self.mask.data.shape)
                    else:
                        img.crop_image()

                    # Set the expected shape based on the first valid image
                    if expected_shape is None:
                        expected_shape = img.data.shape
                    else:
                        if img.data.shape != expected_shape:
                            st.warning(
                                f"Skipping {file.name}: Image shape {img.data.shape} "
                                f"does not match expected shape {expected_shape}."
                            )
                            continue

                    images.append(img)
                except ValueError as ve:
                    st.warning(f"Skipping {file.name}: {ve}")
                    continue

        return images

    def generate_mask(self, images: List[AllskyImage]) -> Optional[AllskyImage]:
        """Generate mask from multiple images."""
        self.camera.imgdata = images
        try:
            mask = self.camera.generate_mask(
                mask_lt=3400,
                gaussian_blur=10,
                convolve=20,
                filename=None,  # Don't save to disk
            )
            self.mask = mask
            self.mask_data = mask.data  # Store the mask data
            return mask
        except Exception as e:
            st.error(f"Error generating mask: {str(e)}")
            return None

    def create_subregions(self) -> bool:
        """Create and initialize subregions using mask."""
        if self.mask is None or self.mask_data is None:
            st.error("Mask must be generated before creating subregions")
            return False

        try:
            # Ensure dimensions match before proceeding
            if not self.ensure_dimensions_match():
                return False

            self.camera.maskdata = AllskyImage(
                filename="mask", data=self.mask_data, header={}
            )

            num_regions = self.camera.generate_subregions()
            st.success(f"Successfully created {num_regions} subregions")

            # Validate subregions were created properly
            if not self.camera.subregions.any():
                st.warning("Subregions were created but appear to be empty")
                return False

            # Share subregions with current image
            if self.current_image is not None:
                self.current_image.subregions = self.camera.subregions

            return True

        except Exception as e:
            import traceback

            st.error("Error during subregion creation:")
            st.error(str(e))
            st.error(traceback.format_exc())
            return False

    def extract_features(self, image: AllskyImage) -> Optional[pd.DataFrame]:
        """Extract features from an image using generated subregions."""
        try:
            self.camera.imgdata = [image]
            self.camera.extract_features(self.camera.subregions, mask=self.mask.data)

            features = self.camera.imgdata[0].features
            return pd.DataFrame(
                {
                    "Subregion": list(range(len(features["srcdens"]))),
                    "Source Density": features["srcdens"],
                    "Background Median": features["bkgmedian"],
                    "Background Mean": features["bkgmean"],
                    "Background Std": features["bkgstd"],
                }
            )
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
            return None


def visualize_image(
    data: np.ndarray,
    title: str,
    overlay: Optional[np.ndarray] = None,
    overlay_cmap: str = "Oranges",
) -> io.BytesIO:
    # Calculate dimensions
    height, width = data.shape
    scale = min(15 / max(height, width), 1.0)
    figsize = (width * scale / 100, height * scale / 100)

    # Create figure with calculated size
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize display using robust statistics
    vmin, vmax = np.percentile(data[~np.isnan(data)], (1, 99))

    # Display full image
    im = ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Pixel Value")

    if overlay is not None:
        ax.imshow(overlay, origin="lower", cmap=overlay_cmap, alpha=0.3)

    ax.set_title(title)
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")

    # Save at high resolution
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return buf


def main():
    st.title("Cloud Detection")

    processor = ImageProcessor()

    # File upload section
    st.sidebar.header("Image Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload FITS Files", type=["fits", "fits.bz2"], accept_multiple_files=True
    )

    optional_mask = st.sidebar.file_uploader(
        "Upload Optional Mask (FITS)", type=["fits"], key="mask_upload"
    )

    if not uploaded_files:
        st.info("Please upload FITS files to begin processing")
        return

    # Process uploaded files
    images = processor.process_multiple_images(uploaded_files)
    if not images:
        st.error("No valid images found in uploaded files")
        return

    # Set current image
    processor.current_image = images[0]  # Set to first valid image

    # Display sample image
    st.header("Sample Image")
    sample_buf = visualize_image(images[0].data, "Sample Original Image")
    col1, _ = st.columns([3, 1])
    with col1:
        st.image(sample_buf, use_container_width=False)

    # Mask Generation
    st.header("1. Mask Generation")
    if optional_mask:
        mask_data, _ = processor.load_fits_file(optional_mask)
        if mask_data is not None:
            processor.mask = AllskyImage("uploaded_mask", mask_data, {})
            processor.mask_data = mask_data  # Store the mask data
            if processor.ensure_dimensions_match():
                st.success("Using uploaded mask - dimensions matched/adjusted")
            else:
                st.error("Could not adjust image dimensions to match mask")
                return
    else:
        if st.button("Generate Mask from Images"):
            mask = processor.generate_mask(images)
            if mask:
                st.success("Mask generated successfully")
                mask_buf = visualize_image(mask.data, "Generated Mask")
                st.image(mask_buf, use_container_width=True)

    # Subregion Creation
    if processor.mask is not None:
        st.header("2. Subregion Analysis")
        if st.button("Create Subregions"):
            if processor.ensure_dimensions_match():
                if processor.create_subregions():
                    st.success("Subregions created successfully")

                    # Display subregion overlay
                    if (
                        processor.current_image
                        and processor.current_image.subregions is not None
                    ):
                        try:
                            overlay = processor.current_image.create_overlay(
                                overlaytype="subregions",
                                regions=[True]
                                * len(processor.current_image.subregions),
                            )
                            overlay_buf = visualize_image(
                                processor.current_image.data,
                                "Subregions Overlay",
                                overlay=overlay,
                            )
                            st.image(overlay_buf, use_container_width=True)
                        except ValueError as e:
                            st.error(f"Error creating overlay: {str(e)}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cloud Detection System", page_icon="☁️", layout="wide"
    )
    main()
