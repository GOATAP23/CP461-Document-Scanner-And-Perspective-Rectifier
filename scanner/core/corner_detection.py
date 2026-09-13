"""
corner_detection.py — Document Corner Detection Module
======================================================
Detects the four corners of a document in an image using:
  - Contour-based polygon fitting
  - Corner ordering algorithm (TL, TR, BR, BL)
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


class CornerDetector:
    """Detects and orders the four corners of a document."""

    def __init__(self, min_area_ratio: float = 0.05, approx_epsilon: float = 0.02):
        """
        Initialize corner detector.

        Args:
            min_area_ratio: Minimum contour area as a ratio of image area.
            approx_epsilon: Epsilon multiplier for contour approximation (relative to perimeter).
        """
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon = approx_epsilon

    def find_document_contour(
        self, edges: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Find the largest quadrilateral contour in the edge map.

        Strategy:
          1. Find all contours.
          2. Sort by area (descending).
          3. Approximate each contour to a polygon.
          4. Return the first polygon with exactly 4 vertices that meets the minimum area.

        Args:
            edges: Binary edge map from Canny.
            image_shape: Shape of the original image (h, w, ...).

        Returns:
            Array of 4 corner points (4, 2), or None if no quadrilateral found.
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Sort contours by area, largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w = image_shape[:2]
        image_area = h * w
        min_area = image_area * self.min_area_ratio

        for contour in contours[:10]:  # Check top 10 largest contours
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * perimeter, True)

            # We need exactly 4 corners for a document
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        return None

    def find_document_contour_aggressive(
        self, edges: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        More aggressive contour detection with multiple epsilon values.
        Used as a secondary attempt when standard detection fails.

        Args:
            edges: Binary edge map.
            image_shape: Shape of the original image.

        Returns:
            Array of 4 corner points, or None if not found.
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w = image_shape[:2]
        image_area = h * w
        min_area = image_area * self.min_area_ratio * 0.5  # Lower threshold

        # Try multiple epsilon values
        for epsilon_mult in [0.01, 0.02, 0.03, 0.04, 0.05]:
            for contour in contours[:15]:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon_mult * perimeter, True)

                if len(approx) == 4:
                    # Verify it's a reasonable quadrilateral (convex)
                    if cv2.isContourConvex(approx):
                        return approx.reshape(4, 2).astype(np.float32)

        return None

    @staticmethod
    def order_corners(pts: np.ndarray) -> np.ndarray:
        """
        Order 4 corner points in consistent order:
          [Top-Left, Top-Right, Bottom-Right, Bottom-Left]

        Algorithm:
          - Top-Left:     smallest sum (x + y)
          - Bottom-Right:  largest sum (x + y)
          - Top-Right:    smallest diff (y - x)
          - Bottom-Left:  largest diff (y - x)

        Args:
            pts: Array of 4 points, shape (4, 2).

        Returns:
            Ordered array of 4 points, shape (4, 2).
        """
        pts = pts.astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Sum and difference
        s = pts.sum(axis=1)     # x + y
        d = np.diff(pts, axis=1).flatten()  # y - x

        ordered[0] = pts[np.argmin(s)]   # Top-Left: smallest sum
        ordered[2] = pts[np.argmax(s)]   # Bottom-Right: largest sum
        ordered[1] = pts[np.argmin(d)]   # Top-Right: smallest diff
        ordered[3] = pts[np.argmax(d)]   # Bottom-Left: largest diff

        return ordered

    def detect(
        self, edges: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Detect and order document corners.

        Args:
            edges: Binary edge map.
            image_shape: Original image shape.

        Returns:
            Ordered corners (4, 2) as [TL, TR, BR, BL], or None if not found.
        """
        # Try standard detection first
        corners = self.find_document_contour(edges, image_shape)

        # If failed, try aggressive detection
        if corners is None:
            corners = self.find_document_contour_aggressive(edges, image_shape)

        if corners is None:
            return None

        # Order corners consistently
        return self.order_corners(corners)

    def draw_corners(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 10,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detected corners and connecting lines on the image.

        Args:
            image: Input image to draw on (will be copied).
            corners: Ordered corner points (4, 2).
            color: BGR color for drawing.
            radius: Circle radius for corner points.
            thickness: Line thickness.

        Returns:
            Image with corners and lines drawn.
        """
        vis = image.copy()
        labels = ["TL", "TR", "BR", "BL"]
        corners_int = corners.astype(int)

        # Draw lines connecting corners
        for i in range(4):
            pt1 = tuple(corners_int[i])
            pt2 = tuple(corners_int[(i + 1) % 4])
            cv2.line(vis, pt1, pt2, color, thickness)

        # Draw corner points with labels
        for i, (corner, label) in enumerate(zip(corners_int, labels)):
            pt = tuple(corner)
            cv2.circle(vis, pt, radius, color, -1)
            cv2.putText(
                vis, label, (pt[0] + 15, pt[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )

        return vis
