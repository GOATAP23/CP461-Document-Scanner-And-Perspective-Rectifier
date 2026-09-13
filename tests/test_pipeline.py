"""
Unit and integration tests for the Document Scanner pipeline.
"""

import sys
import os
import cv2
import numpy as np

# Ensure root folder is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scanner import DocumentScanner
from scanner.core import Preprocessor, CornerDetector, PerspectiveTransformer
from scanner.features import FeatureMatcher
from scanner.filters import ImageEnhancer
from scanner.utils import FallbackDetector


def create_synthetic_document_image():
    """Create a synthetic tilted document image on a textured/colored desk background."""
    # Background desk (dark gray/brown)
    bg = np.zeros((800, 1000, 3), dtype=np.uint8)
    bg[:] = (45, 55, 65)

    # Document coordinates on background (trapezoid / tilted)
    # Target points: A4 page tilted in 3D perspective
    src_corners = np.array([
        [150, 120],  # TL
        [750, 180],  # TR
        [850, 680],  # BR
        [100, 620],  # BL
    ], dtype=np.float32)

    # Blank white document
    doc = np.ones((500, 700, 3), dtype=np.uint8) * 245

    # Draw some text lines and a header on the document
    cv2.putText(doc, "CONFIDENTIAL REPORT", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.line(doc, (50, 90), (650, 90), (50, 50, 50), 2)
    for y in range(130, 450, 40):
        cv2.line(doc, (50, y), (650, y), (150, 150, 150), 2)

    # Warp the document into the background
    doc_corners = np.array([
        [0, 0],
        [700 - 1, 0],
        [700 - 1, 500 - 1],
        [0, 500 - 1],
    ], dtype=np.float32)

    H_inv = cv2.getPerspectiveTransform(doc_corners, src_corners)
    warped_doc = cv2.warpPerspective(doc, H_inv, (1000, 800))

    # Mask to blend warped document onto background
    mask = np.zeros((800, 1000), dtype=np.uint8)
    cv2.fillConvexPoly(mask, src_corners.astype(int), 255)

    result = bg.copy()
    result[mask > 0] = warped_doc[mask > 0]

    return result, src_corners


def test_imports():
    print("Testing imports... OK")


def test_corner_ordering():
    pts = np.array([[850, 680], [150, 120], [100, 620], [750, 180]], dtype=np.float32)
    ordered = CornerDetector.order_corners(pts)
    assert np.allclose(ordered[0], [150, 120]), "Top-left should be (150, 120)"
    assert np.allclose(ordered[1], [750, 180]), "Top-right should be (750, 180)"
    assert np.allclose(ordered[2], [850, 680]), "Bottom-right should be (850, 680)"
    assert np.allclose(ordered[3], [100, 620]), "Bottom-left should be (100, 620)"
    print("Testing corner ordering algorithm... OK")


def test_full_pipeline():
    img, true_corners = create_synthetic_document_image()

    scanner = DocumentScanner(feature_method="SIFT", use_a4_ratio=True)
    result = scanner.scan(img)

    assert result["success"] is True, f"Scan failed: {result['message']}"
    assert result["warped"] is not None, "Warped image should not be None"
    assert result["enhanced"] is not None, "Enhanced image should not be None"
    assert result["homography"] is not None, "Homography matrix should be computed"
    print(f"Testing full scan pipeline... OK ({result['detection_method']})")
    print(f"  Warped shape: {result['warped'].shape}")
    print(f"  Homography shape: {result['homography'].shape}")


def test_enhancement_filters():
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    enhancer = ImageEnhancer()
    for mode in ImageEnhancer.MODES:
        filtered = enhancer.apply_filter(test_img, mode)
        assert filtered is not None
        assert filtered.shape[:2] == (100, 100)
    print("Testing all enhancement filters... OK")


if __name__ == "__main__":
    test_imports()
    test_corner_ordering()
    test_enhancement_filters()
    test_full_pipeline()
    print("\nALL 4 TEST SUITES PASSED SUCCESSFULLY!")
