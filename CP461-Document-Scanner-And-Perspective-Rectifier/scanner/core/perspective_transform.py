"""
perspective_transform.py — Perspective Transformation Module
============================================================
Handles Homography Matrix calculation and Perspective Warping
to rectify documents to a standard A4 aspect ratio (1:1.414).
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class PerspectiveTransformer:
    """Calculates homography and warps document to A4 proportions."""

    # Standard A4 aspect ratio: width:height = 1:√2 ≈ 1:1.414
    A4_RATIO = 1.4142135623730951  # √2

    def __init__(self, target_width: int = 595, target_height: Optional[int] = None):
        """
        Initialize transformer.

        Args:
            target_width: Width of the output image in pixels (default 595 ≈ A4 at 72dpi).
            target_height: Height of the output image. If None, calculated from A4 ratio.
        """
        self.target_width = target_width
        self.target_height = target_height or int(target_width * self.A4_RATIO)

    @staticmethod
    def compute_dimensions(corners: np.ndarray) -> Tuple[float, float]:
        """
        Compute the width and height of the document from its 4 corners
        using Euclidean distance.

        Corners must be ordered: [TL, TR, BR, BL].

        Width_A  = dist(BR, BL)  (bottom edge)
        Width_B  = dist(TR, TL)  (top edge)
        MaxWidth = max(Width_A, Width_B)

        Height_A = dist(TR, BR)  (right edge)
        Height_B = dist(TL, BL)  (left edge)
        MaxHeight = max(Height_A, Height_B)

        Args:
            corners: Ordered corners (4, 2) — [TL, TR, BR, BL].

        Returns:
            Tuple of (max_width, max_height).
        """
        tl, tr, br, bl = corners

        # Bottom and top widths
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(width_a, width_b)

        # Right and left heights
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(height_a, height_b)

        return float(max_width), float(max_height)

    def get_destination_points(
        self, corners: np.ndarray, use_a4_ratio: bool = True
    ) -> np.ndarray:
        """
        Compute destination points for the perspective transform.

        If use_a4_ratio is True, forces A4 aspect ratio (1:1.414).
        Otherwise, uses the actual computed dimensions.

        Args:
            corners: Source corners (4, 2) — [TL, TR, BR, BL].
            use_a4_ratio: Whether to enforce A4 aspect ratio.

        Returns:
            Destination points (4, 2) as float32.
        """
        if use_a4_ratio:
            w = self.target_width
            h = self.target_height

            # Determine if the document is landscape or portrait
            actual_w, actual_h = self.compute_dimensions(corners)
            if actual_w > actual_h:
                # Landscape: swap width and height
                w, h = h, w
        else:
            w, h = self.compute_dimensions(corners)
            w, h = int(w), int(h)

        dst = np.array([
            [0, 0],           # Top-Left
            [w - 1, 0],       # Top-Right
            [w - 1, h - 1],   # Bottom-Right
            [0, h - 1],       # Bottom-Left
        ], dtype=np.float32)

        return dst

    def compute_homography(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_threshold: float = 5.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute the Homography matrix H such that:
            P_target = H · P_source

        Args:
            src_points: Source points (N, 2) with N >= 4.
            dst_points: Destination points (N, 2).
            method: Method for computing homography (0, cv2.RANSAC, cv2.LMEDS).
            ransac_threshold: RANSAC reprojection error threshold.

        Returns:
            Tuple of (homography_matrix, inlier_mask).
            Returns (None, None) if computation fails.
        """
        if len(src_points) == 4 and len(dst_points) == 4:
            # Exact 4-point case: use getPerspectiveTransform for exact solution
            H = cv2.getPerspectiveTransform(
                src_points.astype(np.float32),
                dst_points.astype(np.float32),
            )
            return H, np.ones(4, dtype=np.uint8)

        # General case with more points: use findHomography with RANSAC
        H, mask = cv2.findHomography(
            src_points.astype(np.float32),
            dst_points.astype(np.float32),
            method,
            ransac_threshold,
        )
        return H, mask

    def warp_perspective(
        self,
        image: np.ndarray,
        H: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        flags: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: Tuple[int, ...] = (255, 255, 255),
    ) -> np.ndarray:
        """
        Apply perspective warp using the homography matrix.

        Args:
            image: Source image.
            H: 3x3 Homography matrix.
            output_size: (width, height) of output. If None, uses target dimensions.
            flags: Interpolation flags.
            border_mode: Border extrapolation method.
            border_value: Value used for constant border.

        Returns:
            Warped image.
        """
        if output_size is None:
            output_size = (self.target_width, self.target_height)

        return cv2.warpPerspective(
            image, H, output_size,
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value,
        )

    def transform(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        use_a4_ratio: bool = True,
        scale: float = 1.0,
    ) -> dict:
        """
        Full perspective transform pipeline.

        Args:
            image: Original (full-resolution) image.
            corners: Detected corners on the (possibly resized) image, shape (4, 2).
            use_a4_ratio: Whether to enforce A4 aspect ratio.
            scale: Scale factor to map corners back to original resolution.
                   If the image was resized by 0.5, pass scale=0.5 so corners
                   are scaled back by 1/0.5 = 2.0.

        Returns:
            Dictionary with:
                - 'warped': Warped output image
                - 'homography': The 3x3 Homography matrix
                - 'src_points': Source corner points (on original image)
                - 'dst_points': Destination corner points
                - 'dimensions': Computed (width, height) of document
        """
        # Scale corners back to original resolution
        if scale != 1.0:
            src_corners = corners / scale
        else:
            src_corners = corners.copy()

        src_corners = src_corners.astype(np.float32)

        # Compute destination points
        dst_points = self.get_destination_points(src_corners, use_a4_ratio)

        # Compute homography
        H, mask = self.compute_homography(src_corners, dst_points)

        if H is None:
            raise ValueError("Failed to compute homography matrix.")

        # Determine output size from destination points
        w = int(np.max(dst_points[:, 0])) + 1
        h = int(np.max(dst_points[:, 1])) + 1

        # Warp the original image
        warped = self.warp_perspective(image, H, output_size=(w, h))

        return {
            "warped": warped,
            "homography": H,
            "src_points": src_corners,
            "dst_points": dst_points,
            "dimensions": self.compute_dimensions(src_corners),
        }
