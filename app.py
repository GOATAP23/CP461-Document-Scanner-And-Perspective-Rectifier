"""
app.py — Streamlit Web Application
====================================
Automatic Document Scanner & Perspective Rectifier
Design Style: Modern Dark Studio / High-Contrast SaaS Dashboard

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
    page_title="DocScan — Perspective Rectifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Modern High-Contrast Studio Styling
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Typography & Deep Slate Canvas */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #f1f5f9;
        background: #090d16;
    }
    
    /* Top Navigation / App Header */
    .app-header {
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.85) 0%, rgba(15, 23, 42, 0.7) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-header-left {
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }
    .app-logo-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }
    .app-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .app-badge {
        font-size: 0.7rem;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.2rem 0 0 0;
        font-weight: 400;
    }

    /* Status Banners */
    .status-banner {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        line-height: 1.4;
    }
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
    }
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
    }

    /* Metrics Strip */
    .metric-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #38bdf8);
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }

    /* Section Headings */
    .section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Scanned Document Canvas Frame */
    .doc-canvas {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.85rem;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.5);
    }

    /* Empty State / Welcome Guide */
    .welcome-container {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        position: relative;
    }
    .welcome-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
    }
    .welcome-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
        max-width: 800px;
    }
    
    /* Process Step Cards with Distinct Vibrant Color Accents */
    .step-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.25rem 1.1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: all 0.2s ease;
    }
    .step-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .step-icon-badge {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        margin-bottom: 0.85rem;
    }
    .step-badge-1 {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .step-badge-2 {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .step-badge-3 {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .step-badge-4 {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .step-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
    }
    .step-desc {
        font-size: 0.825rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* Homography Matrix Box */
    .matrix-display {
        background: #090d16;
        color: #38bdf8;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        padding: 1.1rem 1.35rem;
        border-radius: 10px;
        font-size: 0.85rem;
        line-height: 1.7;
        overflow-x: auto;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #0b111e;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .sidebar-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        background: #111827 !important;
        margin-bottom: 0.85rem;
    }
    div[data-testid="stExpander"] details summary {
        font-weight: 600;
        color: #e2e8f0 !important;
    }

    /* Primary Action Buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1.25rem !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
    }
    
    /* Hide Default Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    """Format homography matrix as styled clean HTML."""
    rows = []
    for row in H:
        formatted = "  ".join(f"{v:12.6f}" for v in row)
        rows.append(formatted)
    return f'<div class="matrix-display">H = \n[{rows[0]}]\n[{rows[1]}]\n[{rows[2]}]</div>'


# ─────────────────────────────────────────────────────────
# Internationalization (i18n) / ระบบสลับภาษา
# ─────────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "page_title": "DocScan — Perspective Rectifier",
        "header_title": "Document Scanner",
        "header_tag": "Auto Rectifier",
        "header_subtitle": "Intelligent perspective correction powered by SIFT/ORB, Homography & RANSAC",
        "lang_select": "Language",
        "settings_title": "Scanner Settings",
        "detection_title": "Detection & Geometry",
        "feature_method_label": "Feature Method",
        "feature_method_help": "SIFT is more accurate but slower. ORB is faster.",
        "enforce_a4_label": "Enforce A4 Ratio (1:√2)",
        "preprocessing_title": "Preprocessing & Edges",
        "canny_low_label": "Canny Low Threshold",
        "canny_high_label": "Canny High Threshold",
        "blur_kernel_label": "Blur Kernel Size",
        "bilateral_label": "Bilateral Filtering (Edge-preserving)",
        "enhancement_title": "Enhancement Filter",
        "output_filter_label": "Select Filter",
        "bw_options_title": "B&W Parameters",
        "block_size_label": "Adaptive Block Size",
        "c_value_label": "C Constant (Light Offset)",
        "thresh_method_label": "Threshold Type",
        "footer_text": "CP461 Computer Vision • Document Scanner v1.0",
        "uploader_label": "Upload Document Photo",
        "uploader_help": "Drag and drop or click to upload a photo of a document (JPG, PNG, BMP, TIFF, WebP).",
        "error_title": "Error Loading Image",
        "error_load_msg": "Could not read the uploaded image. Please try a different file.",
        "scanning_msg": "Processing document pipeline...",
        "scan_complete_title": "Scan Completed Successfully",
        "scan_issue_title": "Scan Completed with Notice",
        "stat_detection": "Method",
        "stat_keypoints": "Keypoints",
        "stat_size": "Output Size (px)",
        "stat_feature": "Algorithm",
        "results_title": "Scan Comparison",
        "img_original": "Input Original",
        "img_rectified": "Rectified (Warped)",
        "download_btn": "Download Scanned Document (PNG)",
        "pipeline_expander": "Pipeline Inspection (Steps 1 – 4)",
        "step1_title": "Step 1: Preprocessing & Edge Detection",
        "step1_gray": "Grayscale",
        "step1_blurred": "Gaussian / Bilateral Blur",
        "step1_edges": "Canny Edges",
        "step2_title": "Step 2: Contour & Corner Detection",
        "step2_dilated": "Dilated Edges",
        "step2_corners": "Detected Document Corners",
        "step2_coords": "Corner Coordinates:",
        "corner_tl": "Top-Left",
        "corner_tr": "Top-Right",
        "corner_br": "Bottom-Right",
        "corner_bl": "Bottom-Left",
        "step3_title": "Step 3: Feature Keypoints Extraction",
        "step3_info": "Detected **{count}** keypoints using **{method}**",
        "step4_title": "Step 4: Homography Matrix Estimation",
        "filters_expander": "Enhancement Filter Gallery",
        "summary_expander": "Pipeline Execution Summary",
        "welcome_title": "Intelligent Document Scanner",
        "welcome_desc": "Upload a skewed document photo to automatically detect boundaries, rectify perspective to a flat top-down view, and apply professional enhancement filters.",
        "step_detect_title": "1. Edge Detection",
        "step_detect_desc": "Locates paper boundaries using dual-threshold Canny edge detection.",
        "step_corner_title": "2. Corner Fitting",
        "step_corner_desc": "Extracts exact 4 corners via contour polygon approximation.",
        "step_rectify_title": "3. Perspective Warp",
        "step_rectify_desc": "Calculates homography matrix to transform skewed perspective.",
        "step_enhance_title": "4. Document Enhance",
        "step_enhance_desc": "Cleans noise, removes casting shadows, and enhances text readability.",
        "landing_footer": "CP461 Introduction to Computer Vision — Powered by OpenCV • SIFT/ORB • Homography • RANSAC",
        "filter_names": {
            "original": "Original (Warped)",
            "grayscale": "Grayscale",
            "magic_color": "Magic Color",
            "bw_scanner": "B&W Scanner",
            "shadow_removal": "Shadow Removal",
            "contrast_enhanced": "Contrast Enhanced (CLAHE)",
        },
    },
    "th": {
        "page_title": "DocScan — ระบบปรับระนาบและสแกนเอกสาร",
        "header_title": "ระบบสแกนเอกสาร",
        "header_tag": "ปรับมุมมองอัตโนมัติ",
        "header_subtitle": "ปรับระนาบหน้าตรงด้วย SIFT/ORB Feature Matching, เมทริกซ์ Homography และ RANSAC",
        "lang_select": "ภาษา / Language",
        "settings_title": "การตั้งค่าระบบ",
        "detection_title": "การตรวจจับและสัดส่วน",
        "feature_method_label": "อัลกอริทึมสกัดจุดเด่น",
        "feature_method_help": "SIFT ให้ความแม่นยำสูงในมุมเอียงมาก ส่วน ORB ประมวลผลได้รวดเร็วกว่า",
        "enforce_a4_label": "บังคับสัดส่วนกระดาษ A4 (1:√2)",
        "preprocessing_title": "การตรวจหาเส้นขอบ",
        "canny_low_label": "Canny Low Threshold",
        "canny_high_label": "Canny High Threshold",
        "blur_kernel_label": "ขนาด Blur Kernel",
        "bilateral_label": "Bilateral Filter (ลด Noise คงขอบคม)",
        "enhancement_title": "ฟิลเตอร์ปรับปรุงภาพ",
        "output_filter_label": "เลือกฟิลเตอร์แสดงผล",
        "bw_options_title": "พารามิเตอร์ B&W Scanner",
        "block_size_label": "Block Size (ขนาดบริเวณรอบจุด)",
        "c_value_label": "C Value (ค่าชดเชยแสง)",
        "thresh_method_label": "วิธีคำนวณ Threshold",
        "footer_text": "CP461 Computer Vision • Document Scanner v1.0",
        "uploader_label": "อัปโหลดภาพถ่ายเอกสาร",
        "uploader_help": "ลากไฟล์มาวาง หรือคลิกเพื่อเลือกภาพเอกสาร (รองรับ JPG, PNG, BMP, TIFF, WebP)",
        "error_title": "ข้อผิดพลาดในการโหลดภาพ",
        "error_load_msg": "ไม่สามารถโหลดภาพที่อัปโหลดได้ กรุณาลองใช้ไฟล์อื่น",
        "scanning_msg": "กำลังประมวลผล Pipeline เอกสาร...",
        "scan_complete_title": "ประมวลผลเอกสารสำเร็จ",
        "scan_issue_title": "พบข้อจำกัดในการประมวลผล",
        "stat_detection": "วิธีตรวจจับขอบ",
        "stat_keypoints": "Keypoints",
        "stat_size": "ขนาดภาพผลลัพธ์ (px)",
        "stat_feature": "โมเดล Feature",
        "results_title": "เปรียบเทียบผลลัพธ์เอกสาร",
        "img_original": "ภาพต้นฉบับ (Original)",
        "img_rectified": "ปรับระนาบหน้าตรง (Rectified)",
        "download_btn": "ดาวน์โหลดภาพสแกนเอกสาร (PNG)",
        "pipeline_expander": "ตรวจสอบการทำงานของ Pipeline (สเต็ป 1 – 4)",
        "step1_title": "สเต็ป 1: การประมวลผลเบื้องต้นและตรวจจับขอบ",
        "step1_gray": "ภาพเฉดสีเทา (Grayscale)",
        "step1_blurred": "ภาพเบลอลด Noise (Blurred)",
        "step1_edges": "เส้นขอบ Canny Edges",
        "step2_title": "สเต็ป 2: ตรวจหาตำแหน่งมุมทั้ง 4",
        "step2_dilated": "ขยายเส้นขอบ (Dilated Edges)",
        "step2_corners": "ตำแหน่งมุมทั้ง 4 ที่ตรวจพบ",
        "step2_coords": "พิกัดของมุมทั้ง 4 (Corner Coordinates):",
        "corner_tl": "บน-ซ้าย (Top-Left)",
        "corner_tr": "บน-ขวา (Top-Right)",
        "corner_br": "ล่าง-ขวา (Bottom-Right)",
        "corner_bl": "ล่าง-ซ้าย (Bottom-Left)",
        "step3_title": "สเต็ป 3: การสกัดจุดเด่นภาพ (Feature Keypoints)",
        "step3_info": "ตรวจพบ Keypoints ทั้งหมด **{count}** จุด ด้วยอัลกอริทึม **{method}**",
        "step4_title": "สเต็ป 4: เมทริกซ์การแปลงพิกัด (Homography Matrix)",
        "filters_expander": "แกลเลอรีตัวอย่างทุกฟิลเตอร์ปรับแต่งภาพ",
        "summary_expander": "สรุปข้อมูลการประมวลผล (Pipeline Summary)",
        "welcome_title": "ระบบสแกนเอกสารอัจฉริยะ",
        "welcome_desc": "อัปโหลดภาพถ่ายเอกสารที่ถ่ายมุมเอียง เพื่อตรวจจับขอบเขต ดัดมุมมองเป็นหน้าตรง และปรับคุณภาพเอกสารให้อ่านง่าย คมชัดอัตโนมัติ",
        "step_detect_title": "1. ตรวจจับขอบเขต",
        "step_detect_desc": "ค้นหาขอบเอกสารด้วย Canny Edge Detection สองระดับความเข้ม",
        "step_corner_title": "2. คำนวณมุมทั้ง 4",
        "step_corner_desc": "ระบุพิกัด 4 มุมของเอกสารด้วย Contour Polygon Fitting",
        "step_rectify_title": "3. ปรับระนาบตรง",
        "step_rectify_desc": "คำนวณ Homography Matrix เพื่อ Perspective Warping สู่มุมมองหน้าตรง",
        "step_enhance_title": "4. ปรับปรุงคุณภาพ",
        "step_enhance_desc": "ลบเงา ปรับขาวดำ คมชัด ลบสัญญาณรบกวนให้อ่านตัวหนังสือชัดเจน",
        "landing_footer": "CP461 Introduction to Computer Vision — พัฒนาด้วย OpenCV • SIFT/ORB • Homography • RANSAC",
        "filter_names": {
            "original": "ต้นฉบับปรับระนาบ (Original)",
            "grayscale": "ขาวดำเกรย์สเกล (Grayscale)",
            "magic_color": "เร่งสีสดใส (Magic Color)",
            "bw_scanner": "สแกนขาว-ดำ คมชัด (B&W Scanner)",
            "shadow_removal": "ลบเงาตกกระทบ (Shadow Removal)",
            "contrast_enhanced": "เพิ่มคอนทราสต์ (CLAHE)",
        },
    },
}

# Initialize language in session state (default: Thai)
if "language" not in st.session_state:
    st.session_state["language"] = "th"

lang_code = st.session_state["language"]
t = TRANSLATIONS[lang_code]


# ─────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    # Language Switcher
    st.markdown(f'<div class="sidebar-section-header">{t["lang_select"]}</div>', unsafe_allow_html=True)
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button(
            "🇹🇭 ไทย",
            use_container_width=True,
            type="primary" if lang_code == "th" else "secondary",
            key="btn_lang_th_side",
        ):
            st.session_state["language"] = "th"
            st.rerun()
    with lang_col2:
        if st.button(
            "🇬🇧 English",
            use_container_width=True,
            type="primary" if lang_code == "en" else "secondary",
            key="btn_lang_en_side",
        ):
            st.session_state["language"] = "en"
            st.rerun()

    # Section 1: Detection & Geometry
    st.markdown(f'<div class="sidebar-section-header">{t["detection_title"]}</div>', unsafe_allow_html=True)
    feature_method = st.selectbox(
        t["feature_method_label"],
        ["SIFT", "ORB"],
        index=0,
        help=t["feature_method_help"],
    )
    use_a4 = st.checkbox(t["enforce_a4_label"], value=True)

    # Section 2: Preprocessing
    st.markdown(f'<div class="sidebar-section-header">{t["preprocessing_title"]}</div>', unsafe_allow_html=True)
    canny_low = st.slider(t["canny_low_label"], 10, 150, 50, step=5)
    canny_high = st.slider(t["canny_high_label"], 100, 400, 200, step=10)
    blur_kernel = st.slider(t["blur_kernel_label"], 3, 15, 5, step=2)
    use_bilateral = st.checkbox(t["bilateral_label"], value=False)

    # Section 3: Enhancement
    st.markdown(f'<div class="sidebar-section-header">{t["enhancement_title"]}</div>', unsafe_allow_html=True)
    enhancement_mode = st.selectbox(
        t["output_filter_label"],
        ["original", "grayscale", "magic_color", "bw_scanner", "shadow_removal", "contrast_enhanced"],
        index=0,
        format_func=lambda x: t["filter_names"].get(x, x),
    )

    # B&W Scanner options conditionally displayed
    if enhancement_mode == "bw_scanner":
        st.markdown(f'<div class="sidebar-section-header">{t["bw_options_title"]}</div>', unsafe_allow_html=True)
        block_size = st.slider(t["block_size_label"], 3, 51, 21, step=2)
        c_value = st.slider(t["c_value_label"], 1, 30, 10)
        thresh_method = st.selectbox(
            t["thresh_method_label"],
            ["gaussian", "mean"],
            format_func=lambda x: x.capitalize(),
        )
    else:
        block_size = 21
        c_value = 10
        thresh_method = "gaussian"

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    st.caption(f'<center style="color:#64748b;">{t["footer_text"]}</center>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────

# Sleek Glassmorphic App Bar
st.markdown(f"""
<div class="app-header">
    <div class="app-header-left">
        <div class="app-logo-icon">📄</div>
        <div>
            <div class="app-title">
                <span>{t['header_title']}</span>
                <span class="app-badge">{t['header_tag']}</span>
            </div>
            <p class="app-subtitle">{t['header_subtitle']}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader(
    t["uploader_label"],
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    help=t["uploader_help"],
    key="doc_upload",
)

if uploaded_file is not None:
    # Load image
    image = load_image(uploaded_file)

    if image is None:
        st.markdown(
            f'<div class="status-banner status-error">'
            f'<div><strong>{t["error_title"]}:</strong> {t["error_load_msg"]}</div>'
            f'</div>',
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
        with st.spinner(t["scanning_msg"]):
            result = scanner.scan(image)

        # Status Banner
        if result["success"]:
            st.markdown(
                f'<div class="status-banner status-success">'
                f'<div><strong>✓ {t["scan_complete_title"]}</strong> — {result["message"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-banner status-error">'
                f'<div><strong>⚠ {t["scan_issue_title"]}</strong> — {result["message"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ─── Metrics Strip ───
        if result["success"]:
            stat_cols = st.columns(4)
            with stat_cols[0]:
                det_method = result.get("detection_method", "N/A").split(":")[0]
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{t["stat_detection"]}</div>'
                    f'<div class="metric-value">{det_method}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with stat_cols[1]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{t["stat_keypoints"]}</div>'
                    f'<div class="metric-value">{result.get("num_keypoints", 0):,}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with stat_cols[2]:
                dims = result.get("dimensions")
                dim_str = f"{dims[0]:.0f} × {dims[1]:.0f}" if dims else "—"
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{t["stat_size"]}</div>'
                    f'<div class="metric-value">{dim_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with stat_cols[3]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{t["stat_feature"]}</div>'
                    f'<div class="metric-value">{feature_method}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ─── Main Comparison & Download ───
        if result["success"]:
            st.markdown(f'<div class="section-title"><span>📊</span> {t["results_title"]}</div>', unsafe_allow_html=True)

            # Two-panel comparison: Input Original vs Final Enhanced Document
            comp_cols = st.columns([1, 1.2], gap="medium")
            with comp_cols[0]:
                st.caption(f"**{t['img_original']}**")
                st.image(cv2_to_pil(image), use_container_width=True)

            with comp_cols[1]:
                filter_label = t["filter_names"].get(enhancement_mode, enhancement_mode)
                st.caption(f"**{t['img_rectified']} ({filter_label})**")
                st.image(cv2_to_pil(result["enhanced"]), use_container_width=True)

                # Direct download action
                download_bytes = image_to_bytes(result["enhanced"])
                st.download_button(
                    label=t["download_btn"],
                    data=download_bytes,
                    file_name="scanned_document.png",
                    mime="image/png",
                    type="primary",
                    use_container_width=True,
                )

        # ─── Detailed Technical Expanders (Clean Accordions) ───
        if result["success"]:
            st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
            
            # 1. Pipeline Inspection
            with st.expander(f"🔬 {t['pipeline_expander']}", expanded=False):
                # Step 1
                st.markdown(f"**{t['step1_title']}**")
                prep_cols = st.columns(3)
                with prep_cols[0]:
                    st.caption(t["step1_gray"])
                    st.image(result["preprocessed"]["gray"], use_container_width=True)
                with prep_cols[1]:
                    st.caption(t["step1_blurred"])
                    st.image(result["preprocessed"]["blurred"], use_container_width=True)
                with prep_cols[2]:
                    st.caption(t["step1_edges"])
                    st.image(result["preprocessed"]["edges"], use_container_width=True)

                st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1rem 0;">', unsafe_allow_html=True)

                # Step 2
                st.markdown(f"**{t['step2_title']}**")
                corner_cols = st.columns([1, 1])
                with corner_cols[0]:
                    st.caption(t["step2_dilated"])
                    st.image(result["preprocessed"]["edges_dilated"], use_container_width=True)
                with corner_cols[1]:
                    st.caption(t["step2_corners"])
                    st.image(cv2_to_pil(result["corners_visualization"]), use_container_width=True)

                st.caption(f"**{t['step2_coords']}**")
                corners = result["corners"]
                labels = [t["corner_tl"], t["corner_tr"], t["corner_br"], t["corner_bl"]]
                corner_col_name = "ตำแหน่งมุม" if lang_code == "th" else "Corner Position"
                corner_data = {
                    corner_col_name: labels,
                    "X (px)": [f"{c[0]:.1f}" for c in corners],
                    "Y (px)": [f"{c[1]:.1f}" for c in corners],
                }
                st.dataframe(corner_data, use_container_width=True, hide_index=True)

                st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1rem 0;">', unsafe_allow_html=True)

                # Step 3
                st.markdown(f"**{t['step3_title']}**")
                st.image(cv2_to_pil(result["feature_keypoints"]), use_container_width=True)
                st.info(t["step3_info"].format(count=result.get("num_keypoints", 0), method=feature_method))

                st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1rem 0;">', unsafe_allow_html=True)

                # Step 4
                st.markdown(f"**{t['step4_title']}**")
                if result.get("homography") is not None:
                    st.markdown(format_homography_html(result["homography"]), unsafe_allow_html=True)
                    st.latex(r"P_{\text{target}} = H \cdot P_{\text{source}}")

            # 2. Filter Gallery
            if result.get("all_filters"):
                with st.expander(f"🎨 {t['filters_expander']}", expanded=False):
                    filters = result["all_filters"]
                    filter_keys = list(filters.keys())
                    for i in range(0, len(filter_keys), 3):
                        cols = st.columns(3)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(filter_keys):
                                key = filter_keys[idx]
                                with col:
                                    filter_display = t["filter_names"].get(key, key)
                                    st.caption(f"**{filter_display}**")
                                    st.image(cv2_to_pil(filters[key]), use_container_width=True)

            # 3. Pipeline Summary
            with st.expander(f"📋 {t['summary_expander']}", expanded=False):
                summary = scanner.get_pipeline_summary(result)
                st.code(summary, language="text")

else:
    # High-Contrast Visual Guide & Process Steps
    st.markdown(f"""
    <div class="welcome-container">
        <h3 class="welcome-title">{t['welcome_title']}</h3>
        <p class="welcome-desc">{t['welcome_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    step_cols = st.columns(4)
    step_data = [
        ("🔍", "step-badge-1", t["step_detect_title"], t["step_detect_desc"]),
        ("📐", "step-badge-2", t["step_corner_title"], t["step_corner_desc"]),
        ("🔄", "step-badge-3", t["step_rectify_title"], t["step_rectify_desc"]),
        ("✨", "step-badge-4", t["step_enhance_title"], t["step_enhance_desc"]),
    ]
    for col, (icon, badge_class, title, desc) in zip(step_cols, step_data):
        with col:
            st.markdown(
                f'<div class="step-card">'
                f'<div class="step-icon-badge {badge_class}">{icon}</div>'
                f'<div class="step-title">{title}</div>'
                f'<div class="step-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height: 2.5rem;"></div>', unsafe_allow_html=True)
    st.caption(f'<center style="color:#64748b;">{t["landing_footer"]}</center>', unsafe_allow_html=True)
