"""
fallback.py — Fallback Detection Mechanisms
=============================================
Provides robust document detection when primary methods fail:
  - Multi-strategy edge detection
  - Hough Line-based corner estimation
  - Automatic parameter tuning
  - Full-image fallback (use entire image as document)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from scanner.core.corner_detection import CornerDetector
from scanner.core.preprocessing import Preprocessor


class FallbackDetector:
    """Fallback mechanisms for robust document detection in difficult conditions."""

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.corner_detector = CornerDetector()

    def try_multiple_thresholds(
        self, gray: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Try multiple Canny threshold combinations to find document edges.

        Args:
            gray: Grayscale image.
            image_shape: Shape of the original image.

        Returns:
            Ordered corners (4, 2) or None.
        """
        threshold_pairs = [
            (30, 150),
            (50, 200),
            (75, 250),
            (20, 100),
            (100, 300),
        ]

        for low, high in threshold_pairs:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, low, high)

            # Dilate to close gaps
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)

            corners = self.corner_detector.detect(edges, image_shape)
            if corners is not None:
                return corners

        return None

    def try_morphological_detection(
        self, gray: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Use morphological operations to find document boundary.

        Strategy:
          1. Apply adaptive threshold
          2. Use morphological close to fill gaps
          3. Find contours on the result

        Args:
            gray: Grayscale image.
            image_shape: Shape of the original image.

        Returns:
            Ordered corners or None.
        """
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5,
        )

        # Close gaps with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        corners = self.corner_detector.detect(closed, image_shape)
        return corners

    def try_hough_lines(
        self, edges: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """
        Use Hough Line Transform to detect document edges and compute corners.

        Args:
            edges: Edge map.
            image_shape: Shape of the original image.

        Returns:
            Ordered corners or None.
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(image_shape[:2]) // 4,
            maxLineGap=20,
        )

        if lines is None or len(lines) < 4:
            return None

        # Create a mask from the detected lines
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

        # Dilate lines to connect nearby segments
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

        # Try to find quadrilateral in the line mask
        corners = self.corner_detector.detect(mask, image_shape)
        return corners

    def get_full_image_corners(self, image_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Return corners of the full image as a last resort.
        This assumes the document fills the entire image.

        Args:
            image_shape: Shape of the image (h, w, ...).

        Returns:
            Corner points for the full image boundary.
        """
        h, w = image_shape[:2]
        margin = int(min(h, w) * 0.02)  # 2% margin

        corners = np.array([
            [margin, margin],                 # Top-Left
            [w - 1 - margin, margin],         # Top-Right
            [w - 1 - margin, h - 1 - margin], # Bottom-Right
            [margin, h - 1 - margin],         # Bottom-Left
        ], dtype=np.float32)

        return corners

    def detect(
        self, image: np.ndarray, gray: np.ndarray, edges: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Attempt all fallback strategies in order of preference.

        Args:
            image: Original color image.
            gray: Grayscale version.
            edges: Initial edge detection result.

        Returns:
            Tuple of (corners, method_used_string).
            corners is guaranteed to be non-None (falls back to full image).
        """
        image_shape = image.shape

        # Strategy 1: Try multiple Canny thresholds
        corners = self.try_multiple_thresholds(gray, image_shape)
        if corners is not None:
            return corners, "Fallback: Multi-threshold Canny"

        # Strategy 2: Morphological detection
        corners = self.try_morphological_detection(gray, image_shape)
        if corners is not None:
            return corners, "Fallback: Morphological Detection"

        # Strategy 3: Hough Lines
        corners = self.try_hough_lines(edges, image_shape)
        if corners is not None:
            return corners, "Fallback: Hough Line Detection"

        # Strategy 4: Full image (last resort)
        corners = self.get_full_image_corners(image_shape)
        return corners, "Fallback: Full Image (document not detected)"
