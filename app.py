"""
app.py — Streamlit Web Application
====================================
Automatic Document Scanner & Perspective Rectifier

Features:
  - Drag-and-drop image upload
  - Configurable parameters (sidebar)
  - Before/After comparison view
  - SIFT/ORB Keypoint & RANSAC Inlier Visualization
  - Homography Matrix display
  - Download scanned result
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from scanner.pipeline import DocumentScanner
from scanner.filters.enhancement import ImageEnhancer
from scanner.features.feature_matching import FeatureMatcher

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Scanner | Auto Perspective Rectifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-weight: 700;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .info-card h4 {
        color: #2d3748;
        margin: 0 0 0.4rem 0;
        font-weight: 600;
    }
    .info-card p {
        color: #4a5568;
        margin: 0;
        font-size: 0.9rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left-color: #38a169;
    }
    .error-card {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left-color: #e53e3e;
    }
    
    /* Stats */
    .stat-box {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .stat-box .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-box .stat-label {
        font-size: 0.8rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.2rem;
    }
    
    /* Matrix display */
    .matrix-display {
        background: #1a202c;
        color: #68d391;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        line-height: 1.6;
        overflow-x: auto;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header h3 {
        margin: 0;
        color: #2d3748;
        font-weight: 600;
    }
    
    /* Pipeline steps */
    .pipeline-step {
        background: white;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        border-left: 3px solid #667eea;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label {
        color: #cbd5e0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────
def load_image(uploaded_file) -> np.ndarray:
    """Load an uploaded file as a BGR numpy array."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert BGR OpenCV image to RGB PIL Image."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_to_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Convert OpenCV image to bytes for download."""
    success, buffer = cv2.imencode(fmt, image)
    return buffer.tobytes()


def format_homography_html(H: np.ndarray) -> str:
    """Format homography matrix as styled HTML."""
    rows = []
    for row in H:
        formatted = "  ".join(f"{v:12.6f}" for v in row)
        rows.append(formatted)
    matrix_text = "\n".join(rows)
    return f'<div class="matrix-display">H = \n[{rows[0]}]\n[{rows[1]}]\n[{rows[2]}]</div>'


# ─────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Scanner Settings")
    st.markdown("---")

    # Detection settings
    st.markdown("### 🔍 Detection")
    feature_method = st.selectbox(
        "Feature Detection Method",
        ["SIFT", "ORB"],
        index=0,
        help="SIFT is more accurate but slower. ORB is faster but less robust.",
    )

    use_a4 = st.checkbox("Enforce A4 Ratio (1:√2)", value=True)

    st.markdown("---")

    # Preprocessing settings
    st.markdown("### 🖼️ Preprocessing")
    canny_low = st.slider("Canny Low Threshold", 10, 150, 50, step=5)
    canny_high = st.slider("Canny High Threshold", 100, 400, 200, step=10)
    blur_kernel = st.slider("Blur Kernel Size", 3, 15, 5, step=2)
    use_bilateral = st.checkbox("Use Bilateral Filter", value=False)

    st.markdown("---")

    # Enhancement settings
    st.markdown("### ✨ Enhancement")
    enhancement_mode = st.selectbox(
        "Output Filter",
        ["original", "grayscale", "magic_color", "bw_scanner", "shadow_removal", "contrast_enhanced"],
        index=0,
        format_func=lambda x: {
            "original": "📷 Original (No Filter)",
            "grayscale": "🔲 Grayscale",
            "magic_color": "🌈 Magic Color",
            "bw_scanner": "📝 B&W Scanner",
            "shadow_removal": "☀️ Shadow Removal",
            "contrast_enhanced": "🔆 Contrast Enhanced (CLAHE)",
        }.get(x, x),
    )

    # B&W Scanner specific settings
    if enhancement_mode == "bw_scanner":
        st.markdown("#### B&W Scanner Options")
        block_size = st.slider("Block Size", 3, 51, 21, step=2)
        c_value = st.slider("C Value", 1, 30, 10)
        thresh_method = st.selectbox(
            "Threshold Method",
            ["gaussian", "mean"],
            format_func=lambda x: x.capitalize(),
        )
    else:
        block_size = 21
        c_value = 10
        thresh_method = "gaussian"

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#718096; font-size:0.75rem;'>"
        "CP461 — Computer Vision<br>Document Scanner v1.0</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>📄 Automatic Document Scanner</h1>
    <p>Perspective Rectifier with SIFT/ORB Feature Matching, Homography & RANSAC</p>
</div>
""", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    help="Drag and drop or click to upload a photo of a document.",
    key="doc_upload",
)

if uploaded_file is not None:
    # Load image
    image = load_image(uploaded_file)

    if image is None:
        st.markdown(
            '<div class="info-card error-card"><h4>❌ Error</h4>'
            '<p>Could not load the uploaded image. Please try a different file.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        # Initialize scanner
        scanner = DocumentScanner(
            feature_method=feature_method,
            use_a4_ratio=use_a4,
            enhancement_mode=enhancement_mode,
            canny_low=canny_low,
            canny_high=canny_high,
            blur_kernel=blur_kernel,
            use_bilateral=use_bilateral,
            adaptive_block_size=block_size,
            adaptive_c=c_value,
            threshold_method=thresh_method,
        )

        # Run pipeline
        with st.spinner("🔄 Scanning document..."):
            result = scanner.scan(image)

        # Status message
        if result["success"]:
            st.markdown(
                f'<div class="info-card success-card"><h4>✅ Scan Complete</h4>'
                f'<p>{result["message"]}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-card error-card"><h4>⚠️ Scan Issue</h4>'
                f'<p>{result["message"]}</p></div>',
                unsafe_allow_html=True,
            )

        # ─── Statistics Row ───
        if result["success"]:
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-value">'
                    f'{result.get("detection_method", "N/A").split(":")[0]}</div>'
                    f'<div class="stat-label">Detection Method</div></div>',
                    unsafe_allow_html=True,
                )
            with stats_cols[1]:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-value">'
                    f'{result.get("num_keypoints", 0)}</div>'
                    f'<div class="stat-label">Keypoints Detected</div></div>',
                    unsafe_allow_html=True,
                )
            with stats_cols[2]:
                if result.get("dimensions"):
                    w, h = result["dimensions"]
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-value">'
                        f'{w:.0f}×{h:.0f}</div>'
                        f'<div class="stat-label">Document Size (px)</div></div>',
                        unsafe_allow_html=True,
                    )
            with stats_cols[3]:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-value">'
                    f'{feature_method}</div>'
                    f'<div class="stat-label">Feature Method</div></div>',
                    unsafe_allow_html=True,
                )

        # ─── Main Comparison: Original vs Rectified vs Enhanced ───
        if result["success"]:
            st.markdown(
                '<div class="section-header"><h3>📊 Results Comparison</h3></div>',
                unsafe_allow_html=True,
            )

            comp_cols = st.columns(3)
            with comp_cols[0]:
                st.markdown("**📷 Original**")
                st.image(cv2_to_pil(image), use_container_width=True)

            with comp_cols[1]:
                st.markdown("**🔄 Rectified (Warped)**")
                st.image(cv2_to_pil(result["warped"]), use_container_width=True)

            with comp_cols[2]:
                mode_labels = {
                    "original": "📷 Original",
                    "grayscale": "🔲 Grayscale",
                    "magic_color": "🌈 Magic Color",
                    "bw_scanner": "📝 B&W Scanner",
                    "shadow_removal": "☀️ Shadow Removal",
                    "contrast_enhanced": "🔆 CLAHE",
                }
                label = mode_labels.get(enhancement_mode, enhancement_mode)
                st.markdown(f"**{label}**")
                st.image(cv2_to_pil(result["enhanced"]), use_container_width=True)

            # ─── Download Button ───
            download_bytes = image_to_bytes(result["enhanced"])
            st.download_button(
                label="⬇️ Download Scanned Document",
                data=download_bytes,
                file_name="scanned_document.png",
                mime="image/png",
                use_container_width=True,
            )

        # ─── Pipeline Visualization ───
        if result["success"]:
            with st.expander("🔬 Pipeline Visualization — Step by Step", expanded=False):

                st.markdown("#### Step 1: Preprocessing")
                prep_cols = st.columns(3)
                with prep_cols[0]:
                    st.markdown("**Grayscale**")
                    st.image(result["preprocessed"]["gray"], use_container_width=True)
                with prep_cols[1]:
                    st.markdown("**Blurred**")
                    st.image(result["preprocessed"]["blurred"], use_container_width=True)
                with prep_cols[2]:
                    st.markdown("**Canny Edges**")
                    st.image(result["preprocessed"]["edges"], use_container_width=True)

                st.markdown("---")
                st.markdown("#### Step 2: Corner Detection")
                corner_cols = st.columns(2)
                with corner_cols[0]:
                    st.markdown("**Dilated Edges**")
                    st.image(result["preprocessed"]["edges_dilated"], use_container_width=True)
                with corner_cols[1]:
                    st.markdown("**Detected Corners**")
                    st.image(cv2_to_pil(result["corners_visualization"]), use_container_width=True)

                st.markdown("**Corner Coordinates:**")
                corners = result["corners"]
                labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                corner_data = {
                    "Corner": labels,
                    "X": [f"{c[0]:.1f}" for c in corners],
                    "Y": [f"{c[1]:.1f}" for c in corners],
                }
                st.table(corner_data)

                st.markdown("---")
                st.markdown("#### Step 3: Feature Detection (Keypoints)")
                st.image(cv2_to_pil(result["feature_keypoints"]), use_container_width=True)
                st.info(f"Detected **{result.get('num_keypoints', 0)}** keypoints using **{feature_method}**")

                st.markdown("---")
                st.markdown("#### Step 4: Homography Matrix")
                if result.get("homography") is not None:
                    st.markdown(format_homography_html(result["homography"]), unsafe_allow_html=True)
                    st.latex(r"P_{\text{target}} = H \cdot P_{\text{source}}")

        # ─── All Filters Preview ───
        if result["success"] and result.get("all_filters"):
            with st.expander("🎨 All Enhancement Filters Preview", expanded=False):
                filters = result["all_filters"]
                filter_names = {
                    "original": "📷 Original",
                    "grayscale": "🔲 Grayscale",
                    "magic_color": "🌈 Magic Color",
                    "bw_scanner": "📝 B&W Scanner",
                    "shadow_removal": "☀️ Shadow Removal",
                    "contrast_enhanced": "🔆 CLAHE",
                }

                # Display in 3-column grid
                filter_keys = list(filters.keys())
                for i in range(0, len(filter_keys), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(filter_keys):
                            key = filter_keys[idx]
                            with col:
                                st.markdown(f"**{filter_names.get(key, key)}**")
                                st.image(cv2_to_pil(filters[key]), use_container_width=True)

        # ─── Pipeline Summary ───
        if result["success"]:
            with st.expander("📋 Pipeline Summary", expanded=False):
                summary = scanner.get_pipeline_summary(result)
                st.code(summary, language="text")

else:
    # Landing state — show instructions
    st.markdown("""
    <div class="info-card">
        <h4>👋 Welcome to the Document Scanner!</h4>
        <p>Upload a photo of a document to get started. The scanner will automatically:</p>
    </div>
    """, unsafe_allow_html=True)

    step_cols = st.columns(4)
    steps = [
        ("🔍", "Detect Edges", "Find document boundaries using Canny Edge Detection"),
        ("📐", "Find Corners", "Locate the 4 corners of your document"),
        ("🔄", "Rectify", "Warp perspective to a standard A4 ratio"),
        ("✨", "Enhance", "Apply filters for a clean scan output"),
    ]
    for col, (icon, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(
                f'<div class="pipeline-step"><strong>{icon} {title}</strong><br>'
                f'<small style="color:#718096">{desc}</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#a0aec0; font-size:0.85rem;">'
        'CP461 Introduction to Computer Vision — Automatic Document Scanner & Perspective Rectifier<br>'
        'Powered by OpenCV • SIFT/ORB • Homography • RANSAC</p>',
        unsafe_allow_html=True,
    )
