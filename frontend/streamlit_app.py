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

    def process_multiple_images(self, uploaded_files) -> List[AllskyImage]:
        """Process multiple FITS files into AllskyImage objects."""
        images = []
        for file in uploaded_files:
            data, header = self.load_fits_file(file)
            if data is not None and header is not None:
                images.append(
                    AllskyImage(
                        filename=file.name,  # Use the name directly from UploadedFile
                        data=data,
                        header=header,
                    )
                )
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
            # Create an AllskyImage instance directly from the mask data
            self.camera.maskdata = AllskyImage(
                filename="mask", data=self.mask_data, header={}
            )
            self.camera.generate_subregions()
            return True
        except Exception as e:
            st.error(f"Error creating subregions: {str(e)}")
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
    """Create visualization of image data with optional overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize display
    vmin, vmax = np.percentile(data[~np.isnan(data)], (1, 99))
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    if overlay is not None:
        ax.imshow(overlay, origin="lower", cmap=overlay_cmap, alpha=0.3)

    ax.set_title(title)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
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

    # Display sample image
    st.header("Sample Image")
    sample_buf = visualize_image(images[0].data, "Sample Original Image")
    st.image(sample_buf, use_container_width=True)

    # Mask Generation
    st.header("1. Mask Generation")
    if optional_mask:
        mask_data, _ = processor.load_fits_file(optional_mask)
        if mask_data is not None:
            processor.mask = AllskyImage("uploaded_mask", mask_data, {})
            processor.mask_data = mask_data  # Store the mask data
            st.success("Using uploaded mask")
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
            if processor.create_subregions():
                st.success("Subregions created successfully")

                # Display subregion overlay
                overlay = images[0].create_overlay(overlaytype="subregions")
                overlay_buf = visualize_image(
                    images[0].data, "Subregions Overlay", overlay=overlay
                )
                st.image(overlay_buf, use_container_width=True)

                # Feature Extraction
                st.header("3. Feature Extraction")
                if st.button("Extract Features"):
                    for idx, image in enumerate(images):
                        features_df = processor.extract_features(image)
                        if features_df is not None:
                            st.subheader(f"Features for Image {idx + 1}")
                            st.dataframe(features_df)

                            # Generate and display overlays
                            for overlay_type, params in [
                                ("srcdens", ("Source Density", "Reds")),
                                ("bkgmedian", ("Background Median", "Blues")),
                            ]:
                                overlay = image.create_overlay(overlaytype=overlay_type)
                                overlay_buf = visualize_image(
                                    image.data,
                                    f"{params[0]} Overlay",
                                    overlay=overlay,
                                    overlay_cmap=params[1],
                                )
                                st.image(overlay_buf, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cloud Detection System", page_icon="☁️", layout="wide"
    )
    main()
