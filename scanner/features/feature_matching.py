"""
feature_matching.py — Feature Matching Module
==============================================
Implements SIFT and ORB feature detection and matching with:
  - Lowe's Ratio Test for filtering good matches
  - RANSAC for outlier rejection
  - Visualization of inliers and outliers
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class FeatureMatcher:
    """Feature-based document detection using SIFT or ORB descriptors."""

    def __init__(
        self,
        method: str = "SIFT",
        ratio_threshold: float = 0.75,
        ransac_threshold: float = 5.0,
        min_matches: int = 10,
    ):
        """
        Initialize feature matcher.

        Args:
            method: Feature detection method — "SIFT" or "ORB".
            ratio_threshold: Lowe's ratio test threshold (lower = stricter).
            ransac_threshold: RANSAC reprojection error threshold in pixels.
            min_matches: Minimum number of good matches required.
        """
        self.method = method.upper()
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches

        # Initialize detector
        if self.method == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=2000)
            self.norm_type = cv2.NORM_L2
        elif self.method == "ORB":
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.norm_type = cv2.NORM_HAMMING
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'SIFT' or 'ORB'.")

    def detect_and_compute(
        self, image: np.ndarray
    ) -> Tuple[list, Optional[np.ndarray]]:
        """
        Detect keypoints and compute descriptors for an image.

        Args:
            image: Grayscale input image.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        """
        Match descriptors using BFMatcher with Lowe's Ratio Test.

        Args:
            desc1: Descriptors from image 1 (query).
            desc2: Descriptors from image 2 (template/train).

        Returns:
            List of good matches that pass the ratio test.
        """
        if desc1 is None or desc2 is None:
            return []

        if len(desc1) < 2 or len(desc2) < 2:
            return []

        # Create BFMatcher
        bf = cv2.BFMatcher(self.norm_type, crossCheck=False)

        # KNN match with k=2 for ratio test
        try:
            raw_matches = bf.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        # Apply Lowe's Ratio Test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def find_homography_from_matches(
        self,
        kp1: list,
        kp2: list,
        matches: List[cv2.DMatch],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Compute Homography matrix from matched keypoints using RANSAC.

        Args:
            kp1: Keypoints from image 1.
            kp2: Keypoints from image 2.
            matches: Good matches between the images.

        Returns:
            Tuple of (homography_matrix, inlier_mask, src_pts, dst_pts).
            Returns (None, None, [], []) if insufficient matches.
        """
        if len(matches) < self.min_matches:
            return None, None, np.array([]), np.array([])

        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Homography with RANSAC
        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold
        )

        return H, mask, src_pts, dst_pts

    def match_with_template(
        self, query_image: np.ndarray, template_image: np.ndarray
    ) -> dict:
        """
        Full feature matching pipeline between a query image and a template.

        Args:
            query_image: The document image to process.
            template_image: Reference template image (e.g., blank A4).

        Returns:
            Dictionary with matching results:
                - 'keypoints_query': Keypoints in query image
                - 'keypoints_template': Keypoints in template image
                - 'descriptors_query': Descriptors for query
                - 'descriptors_template': Descriptors for template
                - 'good_matches': Matches passing ratio test
                - 'homography': Homography matrix (or None)
                - 'inlier_mask': RANSAC inlier mask (or None)
                - 'src_points': Source points
                - 'dst_points': Destination points
                - 'num_inliers': Number of RANSAC inliers
                - 'num_matches': Number of good matches
                - 'method': Feature method used
        """
        # Detect and compute
        kp1, desc1 = self.detect_and_compute(query_image)
        kp2, desc2 = self.detect_and_compute(template_image)

        # Match features
        good_matches = self.match_features(desc1, desc2)

        # Find homography
        H, mask, src_pts, dst_pts = self.find_homography_from_matches(
            kp1, kp2, good_matches
        )

        num_inliers = 0
        if mask is not None:
            num_inliers = int(np.sum(mask))

        return {
            "keypoints_query": kp1,
            "keypoints_template": kp2,
            "descriptors_query": desc1,
            "descriptors_template": desc2,
            "good_matches": good_matches,
            "homography": H,
            "inlier_mask": mask,
            "src_points": src_pts,
            "dst_points": dst_pts,
            "num_inliers": num_inliers,
            "num_matches": len(good_matches),
            "method": self.method,
        }

    def visualize_keypoints(
        self,
        image: np.ndarray,
        keypoints: list,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw keypoints on an image.

        Args:
            image: Input image.
            keypoints: Detected keypoints.
            color: BGR color for keypoints.

        Returns:
            Image with keypoints drawn.
        """
        return cv2.drawKeypoints(
            image, keypoints, None,
            color=color,
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )

    def visualize_matches(
        self,
        img1: np.ndarray,
        kp1: list,
        img2: np.ndarray,
        kp2: list,
        matches: List[cv2.DMatch],
        inlier_mask: Optional[np.ndarray] = None,
        max_matches: int = 100,
    ) -> np.ndarray:
        """
        Visualize feature matches between two images.
        Inliers are drawn in green, outliers in red.

        Args:
            img1: First image (query).
            kp1: Keypoints in first image.
            img2: Second image (template).
            kp2: Keypoints in second image.
            matches: List of matches.
            inlier_mask: RANSAC inlier mask. If None, all matches drawn in green.
            max_matches: Maximum number of matches to draw.

        Returns:
            Visualization image with matches drawn.
        """
        if inlier_mask is not None:
            mask_list = inlier_mask.ravel().tolist()
        else:
            mask_list = [1] * len(matches)

        # Limit number of matches for clarity
        draw_matches = matches[:max_matches]
        draw_mask = mask_list[:max_matches]

        # Draw parameters: green for inliers
        draw_params = dict(
            matchColor=(0, 255, 0),       # Inlier color (green)
            singlePointColor=(255, 0, 0), # Point color (blue)
            matchesMask=draw_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        vis = cv2.drawMatches(img1, kp1, img2, kp2, draw_matches, None, **draw_params)

        # Draw outliers in red
        if inlier_mask is not None:
            outlier_mask = [1 - m for m in draw_mask]
            draw_params_out = dict(
                matchColor=(0, 0, 255),       # Outlier color (red)
                singlePointColor=(255, 0, 0),
                matchesMask=outlier_mask,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            vis = cv2.drawMatches(img1, kp1, img2, kp2, draw_matches, vis, **draw_params_out)

        return vis

    def get_stats_text(self, result: dict) -> str:
        """
        Generate a human-readable statistics string from match results.

        Args:
            result: Dictionary returned by match_with_template().

        Returns:
            Formatted statistics string.
        """
        lines = [
            f"Feature Method: {result['method']}",
            f"Keypoints (Query): {len(result['keypoints_query'])}",
            f"Keypoints (Template): {len(result['keypoints_template'])}",
            f"Good Matches (after Ratio Test): {result['num_matches']}",
            f"RANSAC Inliers: {result['num_inliers']}",
        ]
        if result['num_matches'] > 0:
            inlier_ratio = result['num_inliers'] / result['num_matches'] * 100
            lines.append(f"Inlier Ratio: {inlier_ratio:.1f}%")
        return "\n".join(lines)
