import pickle
from typing import Dict, List, Tuple, Union

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cloudynight.utils import timing_decorator


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Calculates classification metrics."""
    try:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    except Exception:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


class CloudPredictors:
    def __init__(self, kde_model_path: str):
        """Initialize predictors with pre-trained KDE model."""
        try:
            # Try loading with joblib first
            models = joblib.load(kde_model_path)
        except Exception:
            try:
                # Fallback to pickle with error handling
                with open(kde_model_path, "rb") as f:
                    models = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load models: {str(e)}")

        self.kde_label_0 = models["kde_label_0"]
        self.kde_label_1 = models["kde_label_1"]

    def get_all_predictions(
        self, regions: Dict[int, np.ndarray]
    ) -> Dict[str, List[int]]:
        """Get predictions from all methods."""
        return {
            "random": self.predict_random(regions),
            "threshold": self.predict_threshold(regions),
            "kde": self.predict_kde(regions),
        }

    @timing_decorator
    def get_regions(self, image_data: np.ndarray) -> Dict[int, np.ndarray]:
        height, width = image_data.shape
        center_x, center_y = width // 2, height // 2
        radii = [height // 10 * i for i in range(1, 6)]
        radii_sq = [r**2 for r in radii]

        # Precompute coordinate grids
        Y, X = np.ogrid[:height, :width]
        dx = X - center_x
        dy = center_y - Y
        R2 = dx * dx + dy * dy
        angles = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

        regions = {}
        # Innermost circle
        mask_region1 = R2 <= radii_sq[0]
        regions[1] = image_data[mask_region1]

        region_number = 2
        # Outer rings
        for i in range(1, len(radii)):
            inner_r_sq = radii_sq[i - 1]
            outer_r_sq = radii_sq[i]
            radial_mask = (R2 > inner_r_sq) & (R2 <= outer_r_sq)

            for j in range(8):
                start_angle = (90 - j * 45) % 360
                end_angle = (start_angle - 45) % 360
                temp_start = end_angle
                temp_end = start_angle

                if temp_start > temp_end:
                    angle_mask = (angles >= temp_start) | (angles < temp_end)
                else:
                    angle_mask = (angles >= temp_start) & (angles < temp_end)

                mask = radial_mask & angle_mask
                regions[region_number] = image_data[mask]
                region_number += 1
        return regions

    def _get_segment_values(
        self,
        image_data: np.ndarray,
        center_x: int,
        center_y: int,
        radius_inner: int,
        radius_outer: int,
        start_angle: float,
        end_angle: float,
    ) -> List[float]:
        """Get pixel values for a circular segment."""
        segment_values = []
        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if radius_inner <= distance <= radius_outer:
                    angle = (
                        np.degrees(np.arctan2(y - center_y, x - center_x)) + 360
                    ) % 360
                    if start_angle < end_angle:
                        if start_angle <= angle < end_angle:
                            segment_values.append(image_data[y, x])
                    else:
                        if angle >= start_angle or angle < end_angle:
                            segment_values.append(image_data[y, x])
        return segment_values

    def predict_random(self, regions: Dict[int, List[float]]) -> List[int]:
        """Generate random predictions (all clear)."""
        return [0] * len(regions)

    def predict_threshold(
        self,
        regions: Dict[int, Union[List[float], np.ndarray]],
        threshold: float = 3300,
    ) -> List[int]:
        """Generate predictions using threshold method."""
        predictions = []
        for _, region in regions.items():
            predictions.append(1 if np.mean(region) > threshold else 0)
        return predictions

    @timing_decorator
    def predict_kde(
        self, regions: Dict[int, Union[List[float], np.ndarray]]
    ) -> List[int]:
        """Generate predictions using KDE method."""
        predictions = []
        for _, region in regions.items():
            percent_0, percent_1 = self._get_kde_probability(np.mean(region))
            predictions.append(1 if percent_1 > percent_0 else 0)
        return predictions

    def _get_kde_probability(self, value: float) -> Tuple[float, float]:
        """Calculate KDE probabilities for a value."""
        if not np.isfinite(value):
            import warnings

            warnings.warn(
                f"Inf. value encountered: {value}. Assigning zero probabilities."
            )
            return 0.0, 0.0

        try:
            log_density_0 = self.kde_label_0.score_samples([[value]])[0]
            log_density_1 = self.kde_label_1.score_samples([[value]])[0]
        except Exception as e:
            import logging

            logging.error(f"KDE scoring failed for value {value}: {e}")
            return 0.0, 0.0

        # Use np.exp with clipping to avoid underflow
        prob_0 = np.clip(np.exp(log_density_0), 1e-10, None)
        prob_1 = np.clip(np.exp(log_density_1), 1e-10, None)

        # Add small epsilon to avoid division by zero
        total_prob = prob_0 + prob_1 + 1e-10

        # Calculate percentages
        percent_0 = (prob_0 / total_prob) * 100
        percent_1 = (prob_1 / total_prob) * 100

        return percent_0, percent_1
