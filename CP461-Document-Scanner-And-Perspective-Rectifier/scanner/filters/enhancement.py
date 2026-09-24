"""
enhancement.py — Image Enhancement & Post-processing Module
============================================================
Provides multiple enhancement filters for scanned documents:
  - Grayscale Clean
  - Magic Color (vibrant color enhancement)
  - B&W Scanner Effect (Adaptive Thresholding)
  - Shadow Removal (illumination normalization)
  - Contrast Enhancement (CLAHE)
"""

import cv2
import numpy as np
from typing import Tuple


class ImageEnhancer:
    """Post-processing filters to improve scanned document quality."""

    # Available filter modes
    MODES = [
        "original",
        "grayscale",
        "magic_color",
        "bw_scanner",
        "shadow_removal",
        "contrast_enhanced",
    ]

    @staticmethod
    def to_grayscale_clean(image: np.ndarray) -> np.ndarray:
        """
        Convert to a clean grayscale image with slight sharpening.

        Args:
            image: Input BGR image.

        Returns:
            Clean grayscale image (single channel, converted back to BGR for display).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply slight sharpening
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])
        sharpened = cv2.filter2D(gray, -1, kernel)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # Convert back to BGR for consistent display
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def magic_color(image: np.ndarray, saturation_scale: float = 1.5, value_scale: float = 1.2) -> np.ndarray:
        """
        Enhance colors to make the document look vibrant and clear.
        Increases saturation and brightness in HSV space.

        Args:
            image: Input BGR image.
            saturation_scale: Factor to multiply saturation channel.
            value_scale: Factor to multiply value (brightness) channel.

        Returns:
            Color-enhanced BGR image.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)

        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def bw_scanner(
        image: np.ndarray,
        block_size: int = 21,
        c_value: int = 10,
        method: str = "gaussian",
    ) -> np.ndarray:
        """
        Black & White scanner effect using Adaptive Thresholding.

        Args:
            image: Input BGR image.
            block_size: Size of pixel neighborhood for threshold (must be odd).
            c_value: Constant subtracted from mean/Gaussian weighted mean.
            method: Thresholding method — "gaussian" or "mean".

        Returns:
            Binary (B&W) image in BGR format.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply slight blur to reduce noise before thresholding
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Ensure block_size is odd and >= 3
        block_size = max(3, block_size)
        if block_size % 2 == 0:
            block_size += 1

        if method.lower() == "gaussian":
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

        binary = cv2.adaptiveThreshold(
            gray, 255, adaptive_method,
            cv2.THRESH_BINARY, block_size, c_value,
        )

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from document image using illumination normalization.

        Algorithm:
          1. Convert to grayscale
          2. Estimate background illumination using morphological opening
             with a large kernel (dilate then erode)
          3. Divide original by estimated illumination
          4. Normalize to full range

        Args:
            image: Input BGR image.

        Returns:
            Shadow-removed BGR image.
        """
        rgb_planes = cv2.split(image)
        result_planes = []

        for plane in rgb_planes:
            # Estimate background using large morphological dilation
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg = cv2.medianBlur(dilated, 21)

            # Compute difference and normalize
            diff = 255 - cv2.absdiff(plane, bg)
            normalized = cv2.normalize(
                diff, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
            )
            result_planes.append(normalized)

        result = cv2.merge(result_planes)
        return result

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input BGR image.
            clip_limit: Threshold for contrast limiting.
            tile_grid_size: Size of grid for histogram equalization.

        Returns:
            Contrast-enhanced BGR image.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_enhanced = clahe.apply(l_channel)

        # Merge back and convert to BGR
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_filter(self, image: np.ndarray, mode: str, **kwargs) -> np.ndarray:
        """
        Apply a named filter to the image.

        Args:
            image: Input BGR image.
            mode: Filter mode name (see MODES class attribute).
            **kwargs: Additional keyword arguments passed to the filter function.

        Returns:
            Filtered image.
        """
        mode = mode.lower().strip()

        if mode == "original":
            return image.copy()
        elif mode == "grayscale":
            return self.to_grayscale_clean(image)
        elif mode == "magic_color":
            return self.magic_color(image, **kwargs)
        elif mode == "bw_scanner":
            return self.bw_scanner(image, **kwargs)
        elif mode == "shadow_removal":
            return self.remove_shadows(image)
        elif mode == "contrast_enhanced":
            return self.enhance_contrast(image, **kwargs)
        else:
            raise ValueError(
                f"Unknown filter mode: '{mode}'. "
                f"Available modes: {self.MODES}"
            )

    def apply_all_filters(self, image: np.ndarray) -> dict:
        """
        Apply all available filters and return results as a dictionary.

        Args:
            image: Input BGR image.

        Returns:
            Dictionary mapping mode name to filtered image.
        """
        results = {}
        for mode in self.MODES:
            try:
                results[mode] = self.apply_filter(image, mode)
            except Exception as e:
                results[mode] = image.copy()  # Fallback to original
        return results
