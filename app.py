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
# Internationalization (i18n) / ระบบสลับภาษา
# ─────────────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "page_title": "Document Scanner | Auto Perspective Rectifier",
        "header_title": "📄 Automatic Document Scanner",
        "header_subtitle": "Perspective Rectifier with SIFT/ORB Feature Matching, Homography & RANSAC",
        "lang_select": "🌐 Language / ภาษา",
        "settings_title": "⚙️ Scanner Settings",
        "detection_title": "🔍 Detection",
        "feature_method_label": "Feature Detection Method",
        "feature_method_help": "SIFT is more accurate but slower. ORB is faster but less robust.",
        "enforce_a4_label": "Enforce A4 Ratio (1:√2)",
        "preprocessing_title": "🖼️ Preprocessing",
        "canny_low_label": "Canny Low Threshold",
        "canny_high_label": "Canny High Threshold",
        "blur_kernel_label": "Blur Kernel Size",
        "bilateral_label": "Use Bilateral Filter",
        "enhancement_title": "✨ Enhancement",
        "output_filter_label": "Output Filter",
        "bw_options_title": "B&W Scanner Options",
        "block_size_label": "Block Size",
        "c_value_label": "C Value",
        "thresh_method_label": "Threshold Method",
        "footer_text": "CP461 — Computer Vision<br>Document Scanner v1.0",
        "uploader_label": "Upload a document image",
        "uploader_help": "Drag and drop or click to upload a photo of a document (JPG, PNG, BMP, TIFF, WebP).",
        "error_title": "❌ Error",
        "error_load_msg": "Could not load the uploaded image. Please try a different file.",
        "scanning_msg": "🔄 Scanning document...",
        "scan_complete_title": "✅ Scan Complete",
        "scan_issue_title": "⚠️ Scan Issue",
        "stat_detection": "Detection Method",
        "stat_keypoints": "Keypoints Detected",
        "stat_size": "Document Size (px)",
        "stat_feature": "Feature Method",
        "results_title": "📊 Results Comparison",
        "img_original": "📷 Original",
        "img_rectified": "🔄 Rectified (Warped)",
        "download_btn": "⬇️ Download Scanned Document",
        "pipeline_expander": "🔬 Pipeline Visualization — Step by Step",
        "step1_title": "Step 1: Preprocessing",
        "step1_gray": "Grayscale",
        "step1_blurred": "Blurred",
        "step1_edges": "Canny Edges",
        "step2_title": "Step 2: Corner Detection",
        "step2_dilated": "Dilated Edges",
        "step2_corners": "Detected Corners",
        "step2_coords": "Corner Coordinates:",
        "corner_tl": "Top-Left",
        "corner_tr": "Top-Right",
        "corner_br": "Bottom-Right",
        "corner_bl": "Bottom-Left",
        "step3_title": "Step 3: Feature Detection (Keypoints)",
        "step3_info": "Detected **{count}** keypoints using **{method}**",
        "step4_title": "Step 4: Homography Matrix",
        "filters_expander": "🎨 All Enhancement Filters Preview",
        "summary_expander": "📋 Pipeline Summary",
        "welcome_title": "👋 Welcome to the Document Scanner!",
        "welcome_desc": "Upload a photo of a document to get started. The scanner will automatically:",
        "step_detect_title": "Detect Edges",
        "step_detect_desc": "Find document boundaries using Canny Edge Detection",
        "step_corner_title": "Find Corners",
        "step_corner_desc": "Locate the 4 corners of your document",
        "step_rectify_title": "Rectify",
        "step_rectify_desc": "Warp perspective to a standard A4 ratio",
        "step_enhance_title": "Enhance",
        "step_enhance_desc": "Apply filters for a clean scan output",
        "landing_footer": "CP461 Introduction to Computer Vision — Automatic Document Scanner & Perspective Rectifier<br>Powered by OpenCV • SIFT/ORB • Homography • RANSAC",
        "filter_names": {
            "original": "📷 Original (No Filter)",
            "grayscale": "🔲 Grayscale",
            "magic_color": "🌈 Magic Color",
            "bw_scanner": "📝 B&W Scanner",
            "shadow_removal": "☀️ Shadow Removal",
            "contrast_enhanced": "🔆 Contrast Enhanced (CLAHE)",
        },
    },
    "th": {
        "page_title": "ระบบสแกนและปรับมุมมองเอกสารอัตโนมัติ | Document Scanner",
        "header_title": "📄 ระบบสแกนเอกสารและปรับมุมมองภาพอัตโนมัติ",
        "header_subtitle": "ปรับระนาบหน้าตรงด้วย SIFT/ORB Feature Matching, เมทริกซ์ Homography และ RANSAC",
        "lang_select": "🌐 ภาษา / Language",
        "settings_title": "⚙️ การตั้งค่าระบบ (Settings)",
        "detection_title": "🔍 การตรวจจับขอบเขต (Detection)",
        "feature_method_label": "อัลกอริทึมสกัดจุดเด่น (Feature Method)",
        "feature_method_help": "SIFT ให้ความแม่นยำสูงกว่าในมุมเอียงมาก ส่วน ORB ประมวลผลได้รวดเร็วกว่า",
        "enforce_a4_label": "บังคับสัดส่วน A4 (1:√2)",
        "preprocessing_title": "🖼️ การปรับแต่งก่อนประมวลผล (Preprocessing)",
        "canny_low_label": "Canny Low Threshold",
        "canny_high_label": "Canny High Threshold",
        "blur_kernel_label": "ขนาด Blur Kernel",
        "bilateral_label": "ใช้ Bilateral Filter (ลด Noise คงขอบภาพ)",
        "enhancement_title": "✨ ฟิลเตอร์ปรับปรุงภาพ (Enhancement)",
        "output_filter_label": "เลือกฟิลเตอร์ผลลัพธ์",
        "bw_options_title": "ตัวเลือก B&W Scanner",
        "block_size_label": "Block Size (ขนาดพื้นที่รอบจุด)",
        "c_value_label": "C Value (ค่าชดเชยแสง)",
        "thresh_method_label": "วิธีคำนวณ Threshold",
        "footer_text": "CP461 — Computer Vision<br>Document Scanner v1.0",
        "uploader_label": "อัปโหลดภาพถ่ายเอกสาร",
        "uploader_help": "ลากไฟล์มาวาง หรือคลิกเพื่อเลือกภาพเอกสาร (รองรับ JPG, PNG, BMP, TIFF, WebP)",
        "error_title": "❌ ข้อผิดพลาด",
        "error_load_msg": "ไม่สามารถโหลดภาพที่อัปโหลดได้ กรุณาลองใช้ไฟล์อื่น",
        "scanning_msg": "🔄 กำลังสแกนและประมวลผลภาพเอกสาร...",
        "scan_complete_title": "✅ สแกนสำเร็จ",
        "scan_issue_title": "⚠️ พบข้อจำกัดในการสแกน",
        "stat_detection": "วิธีตรวจจับขอบ",
        "stat_keypoints": "Keypoints ที่ตรวจพบ",
        "stat_size": "ขนาดเอกสาร (พิกเซล)",
        "stat_feature": "โมเดล Feature",
        "results_title": "📊 เปรียบเทียบผลลัพธ์ภาพ",
        "img_original": "📷 ภาพต้นฉบับ (Original)",
        "img_rectified": "🔄 ปรับมุมระนาบตรง (Rectified)",
        "download_btn": "⬇️ ดาวน์โหลดภาพสแกนเอกสาร (PNG)",
        "pipeline_expander": "🔬 ขั้นตอนการทำงานของ Pipeline ทีละสเต็ป",
        "step1_title": "สเต็ป 1: การประมวลผลเบื้องต้น (Preprocessing)",
        "step1_gray": "ภาพเฉดสีเทา (Grayscale)",
        "step1_blurred": "ภาพเบลอลด Noise (Blurred)",
        "step1_edges": "เส้นขอบ Canny Edges",
        "step2_title": "สเต็ป 2: ตรวจจับมุมเอกสารทั้ง 4 (Corner Detection)",
        "step2_dilated": "ขยายเส้นขอบ (Dilated Edges)",
        "step2_corners": "ตำแหน่งมุมทั้ง 4 (Detected Corners)",
        "step2_coords": "พิกัดของมุมทั้ง 4 (Corner Coordinates):",
        "corner_tl": "บน-ซ้าย (Top-Left)",
        "corner_tr": "บน-ขวา (Top-Right)",
        "corner_br": "ล่าง-ขวา (Bottom-Right)",
        "corner_bl": "ล่าง-ซ้าย (Bottom-Left)",
        "step3_title": "สเต็ป 3: จุดเด่นภาพ (Feature Keypoints)",
        "step3_info": "ตรวจพบ Keypoints ทั้งหมด **{count}** จุด ด้วยอัลกอริทึม **{method}**",
        "step4_title": "สเต็ป 4: เมทริกซ์การแปลงพิกัด (Homography Matrix)",
        "filters_expander": "🎨 แสดงตัวอย่างผลลัพธ์ทุกฟิลเตอร์แต่งภาพ",
        "summary_expander": "📋 สรุปข้อมูล Pipeline Summary",
        "welcome_title": "👋 ยินดีต้อนรับสู่ระบบสแกนเอกสารอัตโนมัติ!",
        "welcome_desc": "อัปโหลดภาพถ่ายเอกสารเพื่อเริ่มต้นใช้งาน ระบบจะประมวลผลอัตโนมัติตามขั้นตอน:",
        "step_detect_title": "ตรวจจับขอบกระดาษ",
        "step_detect_desc": "ค้นหาขอบเขตของเอกสารด้วย Canny Edge Detection",
        "step_corner_title": "ค้นหามุมทั้ง 4",
        "step_corner_desc": "คำนวณตำแหน่ง 4 มุมของเอกสารด้วย Contour Polygon",
        "step_rectify_title": "ดัดมุมระนาบตรง",
        "step_rectify_desc": "แก้ไขมุมเอียง Perspective Warping สู่สัดส่วนมาตรฐาน A4",
        "step_enhance_title": "ปรับแต่งคุณภาพภาพ",
        "step_enhance_desc": "ตกแต่งให้อ่านง่าย คมชัด ลบเงา และเร่งสีสัน",
        "landing_footer": "CP461 Introduction to Computer Vision — Automatic Document Scanner & Perspective Rectifier<br>พัฒนาด้วย OpenCV • SIFT/ORB • Homography • RANSAC",
        "filter_names": {
            "original": "📷 ภาพต้นฉบับ (Original)",
            "grayscale": "🔲 ขาวดำเกรย์สเกล (Grayscale)",
            "magic_color": "🌈 เร่งสีสันสดใส (Magic Color)",
            "bw_scanner": "📝 สแกนขาว-ดำ คมชัด (B&W Scanner)",
            "shadow_removal": "☀️ ลบเงาตกกระทบ (Shadow Removal)",
            "contrast_enhanced": "🔆 เพิ่มคอนทราสต์ (CLAHE)",
        },
    },
}


# ─────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    # Language Selector
    lang_code = st.selectbox(
        "🌐 ภาษา / Language",
        ["th", "en"],
        index=0,
        format_func=lambda x: "🇹🇭 ภาษาไทย" if x == "th" else "🇬🇧 English",
        key="selected_language",
    )
    t = TRANSLATIONS[lang_code]

    st.markdown(f"## {t['settings_title']}")
    st.markdown("---")

    # Detection settings
    st.markdown(f"### {t['detection_title']}")
    feature_method = st.selectbox(
        t["feature_method_label"],
        ["SIFT", "ORB"],
        index=0,
        help=t["feature_method_help"],
    )

    use_a4 = st.checkbox(t["enforce_a4_label"], value=True)

    st.markdown("---")

    # Preprocessing settings
    st.markdown(f"### {t['preprocessing_title']}")
    canny_low = st.slider(t["canny_low_label"], 10, 150, 50, step=5)
    canny_high = st.slider(t["canny_high_label"], 100, 400, 200, step=10)
    blur_kernel = st.slider(t["blur_kernel_label"], 3, 15, 5, step=2)
    use_bilateral = st.checkbox(t["bilateral_label"], value=False)

    st.markdown("---")

    # Enhancement settings
    st.markdown(f"### {t['enhancement_title']}")
    enhancement_mode = st.selectbox(
        t["output_filter_label"],
        ["original", "grayscale", "magic_color", "bw_scanner", "shadow_removal", "contrast_enhanced"],
        index=0,
        format_func=lambda x: t["filter_names"].get(x, x),
    )

    # B&W Scanner specific settings
    if enhancement_mode == "bw_scanner":
        st.markdown(f"#### {t['bw_options_title']}")
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

    st.markdown("---")
    st.markdown(
        f"<p style='text-align:center; color:#718096; font-size:0.75rem;'>"
        f"{t['footer_text']}</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="main-header">
    <h1>{t['header_title']}</h1>
    <p>{t['header_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# File Upload
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
            f'<div class="info-card error-card"><h4>{t["error_title"]}</h4>'
            f'<p>{t["error_load_msg"]}</p></div>',
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

        # Status message
        if result["success"]:
            st.markdown(
                f'<div class="info-card success-card"><h4>{t["scan_complete_title"]}</h4>'
                f'<p>{result["message"]}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-card error-card"><h4>{t["scan_issue_title"]}</h4>'
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
                    f'<div class="stat-label">{t["stat_detection"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with stats_cols[1]:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-value">'
                    f'{result.get("num_keypoints", 0)}</div>'
                    f'<div class="stat-label">{t["stat_keypoints"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with stats_cols[2]:
                if result.get("dimensions"):
                    w, h = result["dimensions"]
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-value">'
                        f'{w:.0f}×{h:.0f}</div>'
                        f'<div class="stat-label">{t["stat_size"]}</div></div>',
                        unsafe_allow_html=True,
                    )
            with stats_cols[3]:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-value">'
                    f'{feature_method}</div>'
                    f'<div class="stat-label">{t["stat_feature"]}</div></div>',
                    unsafe_allow_html=True,
                )

        # ─── Main Comparison: Original vs Rectified vs Enhanced ───
        if result["success"]:
            st.markdown(
                f'<div class="section-header"><h3>{t["results_title"]}</h3></div>',
                unsafe_allow_html=True,
            )

            comp_cols = st.columns(3)
            with comp_cols[0]:
                st.markdown(f"**{t['img_original']}**")
                st.image(cv2_to_pil(image), use_container_width=True)

            with comp_cols[1]:
                st.markdown(f"**{t['img_rectified']}**")
                st.image(cv2_to_pil(result["warped"]), use_container_width=True)

            with comp_cols[2]:
                label = t["filter_names"].get(enhancement_mode, enhancement_mode)
                st.markdown(f"**{label}**")
                st.image(cv2_to_pil(result["enhanced"]), use_container_width=True)

            # ─── Download Button ───
            download_bytes = image_to_bytes(result["enhanced"])
            st.download_button(
                label=t["download_btn"],
                data=download_bytes,
                file_name="scanned_document.png",
                mime="image/png",
                use_container_width=True,
            )

        # ─── Pipeline Visualization ───
        if result["success"]:
            with st.expander(t["pipeline_expander"], expanded=False):

                st.markdown(f"#### {t['step1_title']}")
                prep_cols = st.columns(3)
                with prep_cols[0]:
                    st.markdown(f"**{t['step1_gray']}**")
                    st.image(result["preprocessed"]["gray"], use_container_width=True)
                with prep_cols[1]:
                    st.markdown(f"**{t['step1_blurred']}**")
                    st.image(result["preprocessed"]["blurred"], use_container_width=True)
                with prep_cols[2]:
                    st.markdown(f"**{t['step1_edges']}**")
                    st.image(result["preprocessed"]["edges"], use_container_width=True)

                st.markdown("---")
                st.markdown(f"#### {t['step2_title']}")
                corner_cols = st.columns(2)
                with corner_cols[0]:
                    st.markdown(f"**{t['step2_dilated']}**")
                    st.image(result["preprocessed"]["edges_dilated"], use_container_width=True)
                with corner_cols[1]:
                    st.markdown(f"**{t['step2_corners']}**")
                    st.image(cv2_to_pil(result["corners_visualization"]), use_container_width=True)

                st.markdown(f"**{t['step2_coords']}**")
                corners = result["corners"]
                labels = [t["corner_tl"], t["corner_tr"], t["corner_br"], t["corner_bl"]]
                corner_col_name = "มุม (Corner)" if lang_code == "th" else "Corner"
                corner_data = {
                    corner_col_name: labels,
                    "X": [f"{c[0]:.1f}" for c in corners],
                    "Y": [f"{c[1]:.1f}" for c in corners],
                }
                st.table(corner_data)

                st.markdown("---")
                st.markdown(f"#### {t['step3_title']}")
                st.image(cv2_to_pil(result["feature_keypoints"]), use_container_width=True)
                st.info(t["step3_info"].format(count=result.get("num_keypoints", 0), method=feature_method))

                st.markdown("---")
                st.markdown(f"#### {t['step4_title']}")
                if result.get("homography") is not None:
                    st.markdown(format_homography_html(result["homography"]), unsafe_allow_html=True)
                    st.latex(r"P_{\text{target}} = H \cdot P_{\text{source}}")

        # ─── All Filters Preview ───
        if result["success"] and result.get("all_filters"):
            with st.expander(t["filters_expander"], expanded=False):
                filters = result["all_filters"]

                # Display in 3-column grid
                filter_keys = list(filters.keys())
                for i in range(0, len(filter_keys), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(filter_keys):
                            key = filter_keys[idx]
                            with col:
                                filter_display = t["filter_names"].get(key, key)
                                st.markdown(f"**{filter_display}**")
                                st.image(cv2_to_pil(filters[key]), use_container_width=True)

        # ─── Pipeline Summary ───
        if result["success"]:
            with st.expander(t["summary_expander"], expanded=False):
                summary = scanner.get_pipeline_summary(result)
                st.code(summary, language="text")

else:
    # Landing state — show instructions
    st.markdown(f"""
    <div class="info-card">
        <h4>{t['welcome_title']}</h4>
        <p>{t['welcome_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    step_cols = st.columns(4)
    steps = [
        ("🔍", t["step_detect_title"], t["step_detect_desc"]),
        ("📐", t["step_corner_title"], t["step_corner_desc"]),
        ("🔄", t["step_rectify_title"], t["step_rectify_desc"]),
        ("✨", t["step_enhance_title"], t["step_enhance_desc"]),
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
        f'<p style="text-align:center; color:#a0aec0; font-size:0.85rem;">'
        f'{t["landing_footer"]}</p>',
        unsafe_allow_html=True,
    )

