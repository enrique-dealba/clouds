import pickle
from typing import Dict, List, Tuple

import numpy as np


class CloudPredictors:
    def __init__(self, kde_model_path: str):
        """Initialize predictors with pre-trained KDE model."""
        with open(kde_model_path, "rb") as f:
            models = pickle.load(f)
        self.kde_label_0 = models["kde_label_0"]
        self.kde_label_1 = models["kde_label_1"]

    def get_regions(self, image_data: np.ndarray) -> Dict[int, List[float]]:
        """Extract region values from image data."""
        height, width = image_data.shape
        center_x, center_y = width // 2, height // 2
        radii = [height // 10 * i for i in range(1, 6)]

        regions = {}
        regions[1] = [
            image_data[y, x]
            for y in range(height)
            for x in range(width)
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radii[0] ** 2
        ]

        region_number = 2
        for i in range(1, len(radii)):
            for j in range(8):
                start_angle = 90 - j * 45
                end_angle = (start_angle - 45) % 360
                segment_values = self._get_segment_values(
                    image_data,
                    center_x,
                    center_y,
                    radii[i - 1],
                    radii[i],
                    end_angle % 360,
                    start_angle % 360,
                )
                regions[region_number] = segment_values
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
        self, regions: Dict[int, List[float]], threshold: float = 3300
    ) -> List[int]:
        """Generate predictions using threshold method."""
        predictions = []
        for region_num, region in regions.items():
            predictions.append(1 if np.mean(region) > threshold else 0)
        return predictions

    def predict_kde(self, regions: Dict[int, List[float]]) -> List[int]:
        """Generate predictions using KDE method."""
        predictions = []
        for region_num, region in regions.items():
            percent_0, percent_1 = self._get_kde_probability(np.mean(region))
            predictions.append(1 if percent_1 > percent_0 else 0)
        return predictions

    def _get_kde_probability(self, value: float) -> Tuple[float, float]:
        """Calculate KDE probabilities for a value."""
        log_density_0 = self.kde_label_0.score_samples([[value]])[0]
        log_density_1 = self.kde_label_1.score_samples([[value]])[0]
        prob_0 = np.exp(log_density_0)
        prob_1 = np.exp(log_density_1)
        total_prob = prob_0 + prob_1
        return (prob_0 / total_prob) * 100, (prob_1 / total_prob) * 100