"""
preprocessing.py — Image Preprocessing Module
==============================================
Provides functions for preparing raw input images for document detection:
  - Grayscale conversion
  - Gaussian Blur / Bilateral Filter
  - Canny Edge Detection
  - Input validation & resizing
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class Preprocessor:
    """Handles all preprocessing steps before document detection."""

    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (5, 5),
        canny_low: int = 50,
        canny_high: int = 200,
        max_dimension: int = 1500,
    ):
        """
        Initialize preprocessor with configurable parameters.

        Args:
            blur_kernel: Kernel size for Gaussian blur.
            canny_low: Lower threshold for Canny edge detection.
            canny_high: Upper threshold for Canny edge detection.
            max_dimension: Maximum dimension (width or height) for resizing.
        """
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.max_dimension = max_dimension

    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate that the input is a valid image array.

        Args:
            image: Input image array.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        if image is None:
            raise ValueError("Input image is None. Please check the file path.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")
        if image.size == 0:
            raise ValueError("Input image is empty (0 pixels).")
        if len(image.shape) < 2:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
        return True

    def resize_image(
        self, image: np.ndarray, max_dim: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Resize image while maintaining aspect ratio if it exceeds max_dim.

        Args:
            image: Input image.
            max_dim: Maximum allowed dimension. Uses self.max_dimension if None.

        Returns:
            Tuple of (resized_image, scale_factor).
        """
        max_dim = max_dim or self.max_dimension
        h, w = image.shape[:2]
        scale = 1.0

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return image, scale

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale. If already grayscale, return as-is.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Grayscale image.
        """
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 1:
            return image[:, :, 0]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(
        self, image: np.ndarray, kernel: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.

        Args:
            image: Input image.
            kernel: Blur kernel size. Uses self.blur_kernel if None.

        Returns:
            Blurred image.
        """
        kernel = kernel or self.blur_kernel
        return cv2.GaussianBlur(image, kernel, 0)

    def apply_bilateral_filter(
        self, image: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75
    ) -> np.ndarray:
        """
        Apply bilateral filter to reduce noise while preserving edges.

        Args:
            image: Input image.
            d: Diameter of each pixel neighborhood.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.

        Returns:
            Filtered image.
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def detect_edges(
        self,
        image: np.ndarray,
        low: Optional[int] = None,
        high: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply Canny edge detection.

        Args:
            image: Input grayscale image.
            low: Lower threshold. Uses self.canny_low if None.
            high: Upper threshold. Uses self.canny_high if None.

        Returns:
            Edge map (binary image).
        """
        low = low or self.canny_low
        high = high or self.canny_high
        return cv2.Canny(image, low, high)

    def preprocess(
        self, image: np.ndarray, use_bilateral: bool = False
    ) -> dict:
        """
        Run the full preprocessing pipeline.

        Args:
            image: Raw input image (BGR).
            use_bilateral: If True, use bilateral filter instead of Gaussian blur.

        Returns:
            Dictionary containing all intermediate results:
                - 'original': Original image
                - 'resized': Resized image
                - 'scale': Scale factor used
                - 'gray': Grayscale image
                - 'blurred': Blurred image
                - 'edges': Canny edge map
        """
        self.validate_image(image)

        # Resize for performance
        resized, scale = self.resize_image(image.copy())

        # Convert to grayscale
        gray = self.to_grayscale(resized)

        # Apply blur
        if use_bilateral:
            blurred = self.apply_bilateral_filter(gray)
        else:
            blurred = self.apply_gaussian_blur(gray)

        # Edge detection
        edges = self.detect_edges(blurred)

        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        return {
            "original": image,
            "resized": resized,
            "scale": scale,
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "edges_dilated": edges_dilated,
        }
