import io
import json
import os
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits

from cloudynight.models.predictors import CloudPredictors, calculate_metrics
from cloudynight.utils import timing_decorator
from cloudynight.visualization.overlay import create_overlay_colors, plot_with_overlay


@timing_decorator
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


def reorder_ground_truth(labels: List[int]) -> List[int]:
    """Reorder ground truth labels to match visualization regions."""
    if len(labels) != 33:
        return labels

    # Keep the center (first value) as is
    center = labels[0]

    # Reorder outer segments
    outer_segments = labels[1:]

    segments_per_ring = 8
    rotation_offset = 4  # Increased from 3 to 4 for ~30° more clockwise rotation

    reordered = [center]  # Start with center

    # Process each ring (4 rings, 8 segments each)
    for ring in range(4):
        start_idx = ring * segments_per_ring
        ring_segments = outer_segments[start_idx : start_idx + segments_per_ring]

        # Rotate segments in this ring
        rotated_segments = (
            ring_segments[rotation_offset:] + ring_segments[:rotation_offset]
        )

        # Reverse the segments to reverse symmetry (left-right symmetry)
        rotated_segments = rotated_segments[::-1]

        reordered.extend(rotated_segments)

    return reordered


@timing_decorator
def parse_ground_truth(uploaded_file) -> Optional[List[int]]:
    """Parse ground truth labels from uploaded file."""
    try:
        content = uploaded_file.read().decode()
        try:
            data = json.loads(content)
            if "ground_truth" in data:
                labels = data["ground_truth"]
            else:
                labels = data
        except json.JSONDecodeError:
            labels = eval(content)

        labels = list(labels)
        if not labels or not all(label in [0, 1] for label in labels):
            st.error("Ground truth must contain only 0s and 1s")
            return None

        # Reorder ground truth to match other sectors
        labels = reorder_ground_truth(labels)
        return labels

    except Exception as e:
        st.error(f"Error parsing ground truth file: {str(e)}")
        return None


def display_metrics_table(
    predictions: Dict[str, List[int]], ground_truth: Optional[List[int]]
):
    """Display metrics table in Streamlit."""
    if not ground_truth:
        st.info("No ground truth data available for metrics calculation")
        return

    metrics_data = []
    for method, preds in predictions.items():
        metrics = calculate_metrics(ground_truth, preds)
        metrics_data.append(
            {
                "Method": method.capitalize(),
                "Accuracy": f"{metrics['accuracy']:.3f}",
                "Precision": f"{metrics['precision']:.3f}",
                "Recall": f"{metrics['recall']:.3f}",
                "F1 Score": f"{metrics['f1']:.3f}",
            }
        )

    df = pd.DataFrame(metrics_data)

    # Table style
    styled_df = (
        df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score"],
            axis=0,
            props="font-weight: bold; background-color: rgba(100, 149, 237, 0.2)",
        )
        .set_properties(
            **{
                "font-size": "60px",  # prev: 20px
                "text-align": "center",
                "padding": "10px",
                "border": "1px solid gray",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("font-size", "66px"),  # prev: 22px
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "10px"),
                        ("background-color", "#262730"),
                    ],
                },
                {"selector": "", "props": [("width", "100%")]},
            ]
        )
    )

    # Display table with custom width
    st.subheader("Metrics")
    st.write(
        """
        <style>
            .stDataFrame {
                width: 100%;
            }
            .stDataFrame td {
                min-width: 150px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(styled_df, use_container_width=True)


def main():
    st.title("Cloud Detection")

    # Debug section in sidebar
    st.sidebar.header("Debug Information")
    debug_mode = st.sidebar.checkbox("Show Debug Info")

    # Reset timing logs at start of each run
    if "timing_logs" not in st.session_state:
        st.session_state.timing_logs = []

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

    process_button = st.button("Process Image")

    if process_button:
        st.session_state.timing_logs = []

        with st.spinner("Processing image..."):
            start_time = time.time()

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
            start_regions = time.time()
            regions = predictors.get_regions(image_data)
            st.session_state.timing_logs.append(
                f"get_regions: {time.time() - start_regions:.2f} seconds"
            )

            start_random = time.time()
            random_pred = predictors.predict_random(regions)
            st.session_state.timing_logs.append(
                f"predict_random: {time.time() - start_random:.2f} seconds"
            )

            start_thresh = time.time()
            thresh_pred = predictors.predict_threshold(regions)
            st.session_state.timing_logs.append(
                f"predict_threshold: {time.time() - start_thresh:.2f} seconds"
            )

            start_kde = time.time()
            kde_pred = predictors.predict_kde(regions)
            st.session_state.timing_logs.append(
                f"predict_kde: {time.time() - start_kde:.2f} seconds"
            )

            # Store results in session state
            st.session_state.image_data = image_data
            st.session_state.predictions = {
                "random": random_pred,
                "threshold": thresh_pred,
                "kde": kde_pred,
                "ground_truth": ground_truth if ground_truth else random_pred,
            }

            total_time = time.time() - start_time
            st.session_state.timing_logs.append(
                f"Total processing time: {total_time:.2f} seconds"
            )
            st.success("Done!")

    # Display timing information if debug mode
    if debug_mode and "timing_logs" in st.session_state:
        st.sidebar.subheader("Performance Metrics")
        for log in st.session_state.timing_logs:
            st.sidebar.text(log)

    if "predictions" in st.session_state:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Plots each prediction method
        predictions = [
            ("All Clear", st.session_state.predictions["random"]),
            ("Mean Threshold", st.session_state.predictions["threshold"]),
            ("KDE", st.session_state.predictions["kde"]),
            ("Ground Truth", st.session_state.predictions["ground_truth"]),
        ]

        for idx, (title, pred) in enumerate(predictions):
            row, col = idx // 2, idx % 2
            overlay_colors = create_overlay_colors(pred)
            plot_with_overlay(
                st.session_state.image_data, overlay_colors, axes[row, col]
            )
            axes[row, col].set_title(title)

        plt.tight_layout()
        st.pyplot(fig)

    if "predictions" in st.session_state:
        display_metrics_table(
            st.session_state.predictions,
            st.session_state.predictions.get("ground_truth"),
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Cloud Detection", page_icon="☁️", layout="wide")
    main()
