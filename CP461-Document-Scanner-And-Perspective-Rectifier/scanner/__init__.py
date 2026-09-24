"""
Automatic Document Scanner & Perspective Rectifier
===================================================
A complete CV pipeline for detecting, rectifying, and enhancing
document images using SIFT/ORB feature matching, Homography,
RANSAC, and Perspective Warping.

Organized Subpackages:
    - scanner.core: Preprocessing, corner detection, perspective transformation
    - scanner.features: SIFT/ORB feature matching & RANSAC
    - scanner.filters: Image enhancement and post-processing filters
    - scanner.utils: Fallback detection algorithms
"""

from scanner.pipeline import DocumentScanner
from scanner.core.preprocessing import Preprocessor
from scanner.core.corner_detection import CornerDetector
from scanner.core.perspective_transform import PerspectiveTransformer
from scanner.features.feature_matching import FeatureMatcher
from scanner.filters.enhancement import ImageEnhancer
from scanner.utils.fallback import FallbackDetector

__all__ = [
    "DocumentScanner",
    "Preprocessor",
    "CornerDetector",
    "PerspectiveTransformer",
    "FeatureMatcher",
    "ImageEnhancer",
    "FallbackDetector",
]

__version__ = "1.0.0"
