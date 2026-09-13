"""
Script to generate sample test images for the project.
"""
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tests.test_pipeline import create_synthetic_document_image

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "test_images")
    os.makedirs(out_dir, exist_ok=True)
    img, _ = create_synthetic_document_image()
    out_path = os.path.join(out_dir, "sample_document.png")
    cv2.imwrite(out_path, img)
    print(f"Sample test document saved to: {out_path}")
