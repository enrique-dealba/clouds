import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits

from cloudynight import AllskyCamera, AllskyImage


def load_fits(file):
    """Load FITS file and return data and header."""
    with fits.open(file) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
    return data, header


def display_image(data, title="Image"):
    """Display image using matplotlib and return as Streamlit image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data, origin="lower", cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return buf


def main():
    st.title("Cloud Detection")
    st.sidebar.header("Upload and Process Image")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a FITS file", type=["fits", "fits.bz2"]
    )

    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        # Load FITS data
        data, header = load_fits(uploaded_file)

        # Display Original Image
        st.header("Original Image")
        original_image = display_image(data, title="Original FITS Image")
        st.image(original_image, use_column_width=True)

        # Initialize AllskyImage and AllskyCamera
        image = AllskyImage(filename=uploaded_file.name, data=data, header=header)
        cam = AllskyCamera()

        # Option to generate mask from the single image
        st.subheader("1. Generate Mask")
        if st.button("Generate Mask from This Image"):
            # For single image, use the image itself to create mask
            mask = cam.generate_mask(
                mask_gt=None,
                mask_lt=3400,
                gaussian_blur=10,
                convolve=20,
                filename="mask_single.fits",
            )
            st.success("Mask generated successfully!")
            # Display Mask
            st.header("Mask Image")
            mask_image = display_image(mask.data, title="Generated Mask")
            st.image(mask_image, use_column_width=True)
        else:
            st.info(
                "Click the button above to generate a mask from the uploaded image."
            )

        # Proceed only if mask is generated
        if "mask" in locals():
            # Option to create subregions
            st.subheader("2. Create Subregions")
            if st.button("Create Subregions"):
                cam.read_mask(filename="mask_single.fits")
                cam.generate_subregions()
                st.success("Subregions created successfully!")
                # Display Subregions Overlay
                st.header("Subregions Overlay")
                overlay = image.create_overlay(overlaytype="subregions")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(data, origin="lower", cmap="gray", alpha=0.5)
                ax.imshow(overlay, origin="lower", cmap="Oranges", alpha=0.3)
                ax.set_title("Subregions Overlay")
                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(fig)
                st.image(buf, use_column_width=True)
        else:
            st.warning("Please generate a mask first.")

        # Proceed only if subregions are created
        if "subregions" in cam.__dict__ and cam.subregions is not None:
            st.subheader("3. Extract Features")
            if st.button("Extract Features"):
                cam.imgdata = [image]  # Process the single image
                cam.extract_features(cam.subregions, mask=mask.data)
                st.success("Features extracted successfully!")

                # Display Features
                st.header("Extracted Features")
                features = cam.imgdata[0].features
                feature_df = {
                    "Subregion": list(range(len(features["srcdens"]))),
                    "Source Density": features["srcdens"],
                    "Background Median": features["bkgmedian"],
                    "Background Mean": features["bkgmean"],
                    "Background Std": features["bkgstd"],
                }
                import pandas as pd

                df = pd.DataFrame(feature_df)
                st.dataframe(df)

                # Visualize Source Density Overlay
                st.subheader("Source Density Overlay")
                sourcedens_overlay = image.create_overlay(overlaytype="srcdens")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(data, origin="lower", cmap="gray", alpha=0.5)
                ax.imshow(sourcedens_overlay, origin="lower", cmap="Reds", alpha=0.3)
                ax.set_title("Source Density Overlay")
                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(fig)
                st.image(buf, use_column_width=True)

                # Visualize Background Median Overlay
                st.subheader("Background Median Overlay")
                bkgmedian_overlay = image.create_overlay(overlaytype="bkgmedian")
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(data, origin="lower", cmap="gray", alpha=0.5)
                ax.imshow(bkgmedian_overlay, origin="lower", cmap="Blues", alpha=0.3)
                ax.set_title("Background Median Overlay")
                ax.axis("off")
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(fig)
                st.image(buf, use_column_width=True)
        else:
            st.warning("Please create subregions first.")

    else:
        st.info("Please upload a FITS file to begin processing.")


if __name__ == "__main__":
    main()
