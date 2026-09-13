"""
pipeline.py — Main Document Scanner Pipeline
=============================================
Orchestrates the full document scanning workflow:
  1. Preprocessing
  2. Corner/Edge Detection
  3. Feature Matching (optional, for visualization)
  4. Perspective Transformation
  5. Enhancement / Post-processing
  
Provides a single `scan()` method and step-by-step visualization.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging

from scanner.core.preprocessing import Preprocessor
from scanner.core.corner_detection import CornerDetector
from scanner.core.perspective_transform import PerspectiveTransformer
from scanner.features.feature_matching import FeatureMatcher
from scanner.filters.enhancement import ImageEnhancer
from scanner.utils.fallback import FallbackDetector

logger = logging.getLogger(__name__)


class DocumentScanner:
    """
    End-to-end document scanning pipeline.

    Usage:
        scanner = DocumentScanner()
        result = scanner.scan(image)
        warped = result['warped']
        enhanced = result['enhanced']
    """

    def __init__(
        self,
        feature_method: str = "SIFT",
        use_a4_ratio: bool = True,
        enhancement_mode: str = "original",
        canny_low: int = 50,
        canny_high: int = 200,
        blur_kernel: int = 5,
        use_bilateral: bool = False,
        adaptive_block_size: int = 21,
        adaptive_c: int = 10,
        threshold_method: str = "gaussian",
    ):
        """
        Initialize the document scanner with configurable parameters.

        Args:
            feature_method: "SIFT" or "ORB" for feature detection.
            use_a4_ratio: Enforce A4 aspect ratio on output.
            enhancement_mode: Post-processing filter mode.
            canny_low: Lower Canny threshold.
            canny_high: Upper Canny threshold.
            blur_kernel: Gaussian blur kernel size (will be made odd).
            use_bilateral: Use bilateral filter instead of Gaussian.
            adaptive_block_size: Block size for adaptive thresholding.
            adaptive_c: C value for adaptive thresholding.
            threshold_method: "gaussian" or "mean" for adaptive thresholding.
        """
        # Ensure kernel is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        self.preprocessor = Preprocessor(
            blur_kernel=(blur_kernel, blur_kernel),
            canny_low=canny_low,
            canny_high=canny_high,
        )
        self.corner_detector = CornerDetector()
        self.feature_matcher = FeatureMatcher(method=feature_method)
        self.transformer = PerspectiveTransformer()
        self.enhancer = ImageEnhancer()
        self.fallback = FallbackDetector()

        self.use_a4_ratio = use_a4_ratio
        self.enhancement_mode = enhancement_mode
        self.use_bilateral = use_bilateral
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.threshold_method = threshold_method

    def scan(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run the full document scanning pipeline.

        Args:
            image: Input BGR image.

        Returns:
            Dictionary with all results:
                - 'original': Original input image
                - 'preprocessed': Preprocessing results dict
                - 'corners': Detected corner points (4, 2)
                - 'corners_visualization': Image with corners drawn
                - 'detection_method': Method used for detection
                - 'warped': Perspective-corrected image
                - 'enhanced': Enhanced/filtered output image
                - 'homography': 3x3 Homography matrix
                - 'all_filters': Dict of all filter results
                - 'feature_keypoints': Keypoint visualization
                - 'success': Whether scanning succeeded
                - 'message': Status message
        """
        result = {
            "original": image.copy(),
            "success": False,
            "message": "",
        }

        try:
            # ===== Step 1: Preprocessing =====
            logger.info("Step 1: Preprocessing...")
            prep = self.preprocessor.preprocess(image, use_bilateral=self.use_bilateral)
            result["preprocessed"] = prep

            # ===== Step 2: Corner Detection =====
            logger.info("Step 2: Corner Detection...")
            corners = self.corner_detector.detect(
                prep["edges_dilated"], prep["resized"].shape
            )
            detection_method = "Contour Detection"

            # ===== Step 2b: Fallback if primary detection fails =====
            if corners is None:
                logger.warning("Primary detection failed, trying fallback methods...")
                corners, detection_method = self.fallback.detect(
                    prep["resized"], prep["gray"], prep["edges"]
                )

            result["corners"] = corners
            result["detection_method"] = detection_method

            # Draw corners on visualization
            corners_vis = self.corner_detector.draw_corners(
                prep["resized"], corners
            )
            result["corners_visualization"] = corners_vis

            # ===== Step 3: Feature Detection (for visualization) =====
            logger.info("Step 3: Feature Detection...")
            kp, desc = self.feature_matcher.detect_and_compute(prep["gray"])
            kp_vis = self.feature_matcher.visualize_keypoints(
                prep["resized"], kp
            )
            result["feature_keypoints"] = kp_vis
            result["num_keypoints"] = len(kp)

            # ===== Step 4: Perspective Transform =====
            logger.info("Step 4: Perspective Transform...")
            transform_result = self.transformer.transform(
                image, corners,
                use_a4_ratio=self.use_a4_ratio,
                scale=prep["scale"],
            )
            result["warped"] = transform_result["warped"]
            result["homography"] = transform_result["homography"]
            result["src_points"] = transform_result["src_points"]
            result["dst_points"] = transform_result["dst_points"]
            result["dimensions"] = transform_result["dimensions"]

            # ===== Step 5: Enhancement =====
            logger.info("Step 5: Enhancement...")
            if self.enhancement_mode == "bw_scanner":
                enhanced = self.enhancer.apply_filter(
                    transform_result["warped"],
                    self.enhancement_mode,
                    block_size=self.adaptive_block_size,
                    c_value=self.adaptive_c,
                    method=self.threshold_method,
                )
            else:
                enhanced = self.enhancer.apply_filter(
                    transform_result["warped"],
                    self.enhancement_mode,
                )
            result["enhanced"] = enhanced

            # Generate all filter previews
            result["all_filters"] = self.enhancer.apply_all_filters(
                transform_result["warped"]
            )

            result["success"] = True
            result["message"] = f"Document scanned successfully using {detection_method}."
            logger.info(result["message"])

        except Exception as e:
            result["success"] = False
            result["message"] = f"Scanning failed: {str(e)}"
            logger.error(result["message"], exc_info=True)

        return result

    def scan_with_feature_matching(
        self, image: np.ndarray, template: np.ndarray
    ) -> Dict[str, Any]:
        """
        Scan using explicit feature matching against a template image.

        This method is useful for:
          - Documents on backgrounds similar in color
          - Validating detection with a known template
          - Generating SIFT/RANSAC visualizations for the rubric

        Args:
            image: Input document image.
            template: Reference template image (e.g., blank A4).

        Returns:
            Extended result dictionary including match visualization.
        """
        # Run standard scan first
        result = self.scan(image)

        # Run feature matching
        logger.info("Running Feature Matching with template...")
        match_result = self.feature_matcher.match_with_template(image, template)
        result["match_result"] = match_result

        # Generate match visualization
        if match_result["num_matches"] > 0:
            match_vis = self.feature_matcher.visualize_matches(
                image,
                match_result["keypoints_query"],
                template,
                match_result["keypoints_template"],
                match_result["good_matches"],
                match_result["inlier_mask"],
            )
            result["match_visualization"] = match_vis
            result["match_stats"] = self.feature_matcher.get_stats_text(match_result)

        return result

    def get_pipeline_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the pipeline execution.

        Args:
            result: Result dictionary from scan().

        Returns:
            Formatted summary string.
        """
        lines = [
            "═" * 50,
            "  Document Scanner Pipeline Summary",
            "═" * 50,
            f"  Status: {'✓ Success' if result['success'] else '✗ Failed'}",
            f"  Detection Method: {result.get('detection_method', 'N/A')}",
        ]

        if result.get("dimensions"):
            w, h = result["dimensions"]
            lines.append(f"  Document Size: {w:.0f} x {h:.0f} px")

        if result.get("num_keypoints"):
            lines.append(f"  Keypoints Detected: {result['num_keypoints']}")

        if result.get("homography") is not None:
            lines.append(f"  Homography Matrix:")
            H = result["homography"]
            for row in H:
                lines.append(f"    [{row[0]:10.4f}  {row[1]:10.4f}  {row[2]:10.4f}]")

        lines.append(f"  Enhancement: {self.enhancement_mode}")
        lines.append(f"  Message: {result.get('message', '')}")
        lines.append("═" * 50)

        return "\n".join(lines)
