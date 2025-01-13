import io
import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits

from cloudynight.models.predictors import CloudPredictors
from cloudynight.visualization.overlay import create_overlay_colors, get_colored_regions


def load_fits_file(uploaded_file) -> Optional[np.ndarray]:
    """Load FITS file and return image data."""
    try:
        file_bytes = uploaded_file.read()
        file_buffer = io.BytesIO(file_bytes)

        with fits.open(file_buffer, ignore_missing_simple=True) as hdul:
            st.sidebar.write(f"Number of HDUs: {len(hdul)}")
            st.sidebar.write(f"Available HDU names: {[h.name for h in hdul]}")

            # Try to get data from first HDU that has data
            data = None
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None:
                    data = hdu.data.astype(np.float64)
                    break

            if data is None or np.all(data == 0):
                st.error(f"File {uploaded_file.name} appears to be empty")
                return None

            st.sidebar.write(f"Image shape: {data.shape}")
            return data

    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        import traceback

        st.sidebar.text(traceback.format_exc())
        return None


def parse_ground_truth(uploaded_file) -> Optional[List[int]]:
    """Parse ground truth labels from uploaded file."""
    try:
        content = uploaded_file.read().decode()

        # Try JSON format
        try:
            data = json.loads(content)
            if "ground_truth" in data:
                labels = data["ground_truth"]
            else:
                labels = data  # In case the JSON is just an array
        except json.JSONDecodeError:
            labels = eval(content)

        # Convert to list if needed
        labels = list(labels)

        # Validate labels
        if not labels or not all(label in [0, 1] for label in labels):
            st.error("Ground truth must contain only 0s and 1s")
            return None

        return labels

    except Exception as e:
        st.error(f"Error parsing ground truth file: {str(e)}")
        return None


def main():
    st.title("Cloud Detection")

    # Initialize predictors
    kde_model_path = os.path.join(
        os.path.dirname(__file__), "..", "cloudynight", "models", "kde_models.pkl"
    )
    predictors = CloudPredictors(kde_model_path)

    # File upload section
    st.sidebar.header("Upload Files")
    fits_file = st.sidebar.file_uploader("Upload FITS File", type=["fits", "fits.bz2"])
    ground_truth_file = st.sidebar.file_uploader(
        "Upload Ground Truth (optional)", type=["txt", "json"]
    )

    if not fits_file:
        st.info("Please upload a FITS file to begin")
        return

    # Load and process data
    image_data = load_fits_file(fits_file)
    if image_data is None:
        return

    # Get ground truth if provided
    ground_truth = None
    if ground_truth_file:
        ground_truth = parse_ground_truth(ground_truth_file)
        if ground_truth is None:
            return

    # Get regions and predictions
    regions = predictors.get_regions(image_data)
    random_pred = predictors.predict_random(regions)
    thresh_pred = predictors.predict_threshold(regions)
    kde_pred = predictors.predict_kde(regions)

    # Create visualizations
    st.header("Cloud Detection Visualizations")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Plot each prediction method
    predictions = [
        ("All Clear", random_pred),
        ("Mean Threshold", thresh_pred),
        ("KDE", kde_pred),
        ("Ground Truth", ground_truth if ground_truth else random_pred),
    ]

    for idx, (title, pred) in enumerate(predictions):
        row, col = idx // 2, idx % 2
        colored_image = get_colored_regions(image_data, create_overlay_colors(pred))
        axes[row, col].imshow(colored_image)
        axes[row, col].axis("off")
        axes[row, col].set_title(title)

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cloud Detection Visualization", page_icon="☁️", layout="wide"
    )
    main()
