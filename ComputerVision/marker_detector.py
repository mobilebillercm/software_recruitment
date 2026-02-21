"""
Marker Detector — Computer Vision Recruitment Task

Implement the MarkerDetector class and the utility functions below.
See README.md for full task description.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class MarkerDetector:
    """
    Detects colored markers in images using classical computer vision techniques.

    Each detection is a dictionary with the following fields:
        - 'color':  str           — one of 'red', 'green', 'blue', 'yellow'
        - 'bbox':   (x, y, w, h) — bounding rectangle of the detected contour
        - 'center': (cx, cy)     — center coordinates of the bounding box
        - 'area':   float        — area of the detected contour

    Optionally, if you attempt the bonus task:
        - 'shape':  str          — one of 'circle', 'triangle', 'rectangle'
    """

    # Define HSV color ranges for each target color.
    # Each entry maps a color name to a list of (lower_bound, upper_bound) tuples.
    # Use np.array([H, S, V]) for bounds. OpenCV uses H: 0-179, S: 0-255, V: 0-255.
    #
    # Hint: Red wraps around the hue spectrum (both ~0-10 and ~170-179 are red),
    # so you will likely need TWO ranges for red.
    COLOR_RANGES = {
        # Example format:
        # "green": [(np.array([35, 80, 80]), np.array([85, 255, 255]))],
    }

    # Minimum contour area to consider (filters noise)
    min_area = 500

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect colored markers in the given BGR image.

        Args:
            image: Input image in BGR format (as loaded by cv2.imread).

        Returns:
            A list of detection dictionaries, each containing:
            'color', 'bbox', 'center', and 'area' keys.
        """
        pass


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Each box is represented as (x, y, w, h) where:
        - (x, y) is the top-left corner
        - (w, h) is the width and height

    Args:
        box_a: First bounding box as (x, y, w, h).
        box_b: Second bounding box as (x, y, w, h).

    Returns:
        IoU value as a float between 0.0 (no overlap) and 1.0 (perfect overlap).
    """
    pass


def filter_detections(
    detections: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Filter overlapping detections using Non-Maximum Suppression (NMS).

    When two detections overlap (IoU > iou_threshold), keep the one with the
    larger area and discard the other.

    Args:
        detections: List of detection dictionaries (each must have 'bbox' and 'area').
        iou_threshold: IoU threshold above which two detections are considered overlapping.

    Returns:
        Filtered list of detections with overlapping duplicates removed.
    """
    pass
