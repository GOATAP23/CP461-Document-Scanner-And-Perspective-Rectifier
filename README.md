# 📄 Automatic Document Scanner & Perspective Rectifier

> **CP461 Introduction to Computer Vision — Semester 1/2026**

A complete end-to-end application for scanning and rectifying document images using computer vision techniques including SIFT/ORB Feature Matching, Homography Matrix calculation, RANSAC outlier rejection, and Perspective Warping.

---

## 🎯 Features

- **Automatic Document Detection** — Detects document boundaries using edge detection and contour fitting
- **Feature Matching** — SIFT and ORB keypoint detection with Lowe's Ratio Test
- **RANSAC Outlier Rejection** — Robust homography estimation
- **Perspective Correction** — Warps documents to standard A4 aspect ratio (1:√2)
- **Multiple Enhancement Filters:**
  - 📷 Original (no filter)
  - 🔲 Grayscale
  - 🌈 Magic Color
  - 📝 B&W Scanner (Adaptive Thresholding)
  - ☀️ Shadow Removal
  - 🔆 Contrast Enhancement (CLAHE)
- **Interactive Web UI** — Built with Streamlit
- **Fallback Detection** — Multiple strategies for difficult images

---

## 🏗️ Architecture

```
[ Input Image ]
       │
       ▼
[ Preprocessing ] ──► (Grayscale, Gaussian Blur, Bilateral Filter)
       │
       ▼
[ Feature & Corner Detection ] ──► (SIFT / ORB Keypoints OR Contour Polygon Fitting)
       │
       ▼
[ Feature Matching & RANSAC ] ──► (Lowe's Ratio Test, RANSAC Outlier Rejection)
       │
       ▼
[ Corner Ordering Algorithm ] ──► (Sort 4 Corners: TL, TR, BR, BL)
       │
       ▼
[ Homography Matrix Calculation ] ──► (cv2.findHomography / cv2.getPerspectiveTransform)
       │
       ▼
[ Perspective Warping ] ──► (Warp to Standard A4 Aspect Ratio)
       │
       ▼
[ Enhancement & Binarization ] ──► (Adaptive Thresholding, Color Cleanup)
       │
       ▼
[ Output Image & UI Render ]
```

---

## 📁 Project Structure

```
├── app.py                          # Streamlit Web Application
├── requirements.txt                # Python Dependencies
├── README.md                       # Project Documentation
├── .gitignore                      # Git Ignore File
│
├── docs/                           # Project Documentation & Plans
│   ├── README.md
│   └── Document_Scanner_Implementation_Plan.md
│
├── test_images/                    # Test Dataset (Normal & Hard cases)
│   └── README.md
│
├── notebooks/                      # Colab / Jupyter Notebooks (Visualizations)
│   └── README.md
│
└── scanner/                        # Modular CV Package
    ├── __init__.py                 # Top-level exports
    ├── pipeline.py                 # Main Pipeline Orchestrator
    │
    ├── core/                       # Core CV & Geometric Transformations
    │   ├── __init__.py
    │   ├── preprocessing.py        # Grayscale, blur, Canny edge detection
    │   ├── corner_detection.py     # Contour polygon fitting & 4-corner ordering
    │   └── perspective_transform.py# Homography matrix & A4 perspective warping
    │
    ├── features/                   # Feature Extraction & Matching
    │   ├── __init__.py
    │   └── feature_matching.py     # SIFT/ORB keypoints, Lowe's ratio test, RANSAC
    │
    ├── filters/                    # Post-Processing Enhancement Filters
    │   ├── __init__.py
    │   └── enhancement.py          # B&W scanner, shadow removal, magic color, CLAHE
    │
    └── utils/                      # Robustness & Fallback Handlers
        ├── __init__.py
        └── fallback.py             # Multi-strategy fallbacks (Hough, multi-threshold)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd document-scanner

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage

1. Open the web app in your browser
2. Upload a photo of a document (JPG, PNG, BMP, TIFF, or WebP)
3. Adjust parameters in the sidebar if needed
4. View the results: Original → Rectified → Enhanced
5. Download the scanned document

---

## 🔧 Technical Details

### Corner Ordering Algorithm
The 4 corners are ordered consistently using sum and difference:
- **Top-Left:** min(x + y)
- **Bottom-Right:** max(x + y)
- **Top-Right:** min(y - x)
- **Bottom-Left:** max(y - x)

### Homography & Perspective Transform
Documents are warped to A4 proportions (1:1.414) using:
```
P_target = H · P_source
```
where H is the 3×3 Homography matrix computed via `cv2.getPerspectiveTransform` or `cv2.findHomography` with RANSAC.

---

## 📦 Deployment

### Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Push the repository to the Space
4. The app will be available at the public URL

### Streamlit Cloud
1. Push the repository to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy

---

## 👥 Team

| Role | Responsibilities |
|------|-----------------|
| Lead CV Engineer | Core SIFT/ORB pipeline, Homography, RANSAC |
| Robustness Specialist | Fallback detection, Edge cases, Post-processing |
| Frontend Developer | Streamlit UI, Visualizations |
| DevOps Lead | GitHub, Deployment, Testing |
| QA & Presentation | Testing, Video Demo, Presentation |

---

## 📜 License

This project is developed for educational purposes as part of CP461 Introduction to Computer Vision.
