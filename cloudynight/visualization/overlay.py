from typing import List, Tuple

import numpy as np


def get_segment_coordinates(
    center_x: int,
    center_y: int,
    radius_inner: int,
    radius_outer: int,
    start_angle: float,
    end_angle: float,
) -> List[Tuple[int, int]]:
    """Get coordinates for a circular segment."""
    coordinates = []
    for y in range(center_y - radius_outer, center_y + radius_outer):
        for x in range(center_x - radius_outer, center_x + radius_outer):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if radius_inner <= distance <= radius_outer:
                angle = (np.degrees(np.arctan2(y - center_y, x - center_x)) + 360) % 360
                if start_angle < end_angle:
                    if start_angle <= angle < end_angle:
                        coordinates.append((y, x))
                else:
                    if angle >= start_angle or angle < end_angle:
                        coordinates.append((y, x))
    return coordinates


def apply_overlay(
    coords: List[Tuple[int, int]], overlay_color: np.ndarray, image: np.ndarray
) -> None:
    """Apply color overlay to specified coordinates."""
    alpha = overlay_color[3] / 255.0
    for y, x in coords:
        original_pixel = image[y, x, :3].astype(np.float32)
        image[y, x, :3] = (
            (1 - alpha) * original_pixel + alpha * overlay_color[:3]
        ).astype(np.uint8)
        image[y, x, 3] = 255


def create_overlay_colors(binary_list: List[int]) -> List[List[int]]:
    """Create overlay colors based on binary predictions."""
    return [
        [255, 100, 100, 64] if value == 1 else [0, 0, 0, 0] for value in binary_list
    ]


def get_colored_regions_prev(
    image_data: np.ndarray, overlay_colors: List[List[int]]
) -> np.ndarray:
    """Generate colored overlay regions for visualization."""
    height, width = image_data.shape
    center_x, center_y = width // 2, height // 2
    radii = [height // 10 * i for i in range(1, 6)]

    normalized_image = (
        (image_data - np.min(image_data))
        / (np.max(image_data) - np.min(image_data))
        * 255
    ).astype(np.uint8)
    colored_image = np.stack(
        [normalized_image] * 3 + [np.full_like(normalized_image, 255)], axis=-1
    )

    region_number = 0

    # Region 1 (innermost circle)
    region_coords = [
        (y, x)
        for y in range(height)
        for x in range(width)
        if (x - center_x) ** 2 + (y - center_y) ** 2 <= radii[0] ** 2
    ]
    overlay_color = np.array(
        overlay_colors[region_number % len(overlay_colors)], dtype=np.float32
    )
    apply_overlay(region_coords, overlay_color, colored_image)
    region_number += 1

    # Remaining regions
    for i in range(1, len(radii)):
        for j in range(8):
            start_angle = 90 - j * 45
            end_angle = (start_angle - 45) % 360
            coordinates = get_segment_coordinates(
                center_x,
                center_y,
                radii[i - 1],
                radii[i],
                end_angle % 360,
                start_angle % 360,
            )
            overlay_color = np.array(
                overlay_colors[region_number % len(overlay_colors)], dtype=np.float32
            )
            apply_overlay(coordinates, overlay_color, colored_image)
            region_number += 1

    return colored_image


def get_colored_regions(
    image_data: np.ndarray, overlay_colors: List[List[int]]
) -> np.ndarray:
    """Generate colored overlay regions using vectorized operations."""
    height, width = image_data.shape
    center_x, center_y = width // 2, height // 2
    radii = [height // 10 * i for i in range(1, 6)]
    radii_sq = [r**2 for r in radii]

    # Normalize image once
    normalized_image = (
        (image_data - np.min(image_data))
        / (np.max(image_data) - np.min(image_data))
        * 255
    ).astype(np.uint8)

    # Create base colored image
    colored_image = np.stack(
        [normalized_image] * 3 + [np.full_like(normalized_image, 255)], axis=-1
    )

    # Precompute coordinate grids (similar to get_regions)
    Y, X = np.ogrid[:height, :width]
    dx = X - center_x
    dy = center_y - Y
    R2 = dx * dx + dy * dy
    angles = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    # Convert overlay_colors to numpy array for vectorization
    overlay_colors = np.array(overlay_colors, dtype=np.float32)

    # Process innermost circle (region 1)
    mask_region1 = R2 <= radii_sq[0]
    alpha = overlay_colors[0][3] / 255.0
    if alpha > 0:  # Only apply if there's any opacity
        colored_image[mask_region1, :3] = (
            (1 - alpha) * colored_image[mask_region1, :3]
            + alpha * overlay_colors[0][:3]
        ).astype(np.uint8)

    region_number = 1
    # Process outer rings
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

            # Combine radial and angular masks
            mask = radial_mask & angle_mask

            # Apply overlay color using vectorized operations
            overlay_color = overlay_colors[region_number % len(overlay_colors)]
            alpha = overlay_color[3] / 255.0

            if alpha > 0:  # Only apply if there's any opacity
                colored_image[mask, :3] = (
                    (1 - alpha) * colored_image[mask, :3] + alpha * overlay_color[:3]
                ).astype(np.uint8)

            region_number += 1

    return colored_image
