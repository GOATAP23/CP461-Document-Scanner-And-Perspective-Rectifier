"""
Generate rich, realistic sample test images for Document Scanner demo.
"""
import os
import cv2
import numpy as np

def create_wood_desk(width=1200, height=900):
    """Generate a realistic dark wooden desk background."""
    desk = np.zeros((height, width, 3), dtype=np.uint8)
    desk[:] = (35, 45, 60) # Dark rich mahogany/slate tone
    # Add subtle wood grain / noise
    noise = np.random.normal(0, 8, (height, width, 3)).astype(np.int16)
    desk_noisy = np.clip(desk.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return desk_noisy

def warp_and_blend(doc_img, src_pts, dst_pts, bg_img):
    """Warp document into background using perspective transform."""
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    h_bg, w_bg = bg_img.shape[:2]
    warped = cv2.warpPerspective(doc_img, H, (w_bg, h_bg), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    # Create mask with smooth anti-aliased edge
    mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    
    # Add subtle drop shadow under the document
    shadow_mask = np.zeros((h_bg, w_bg), dtype=np.float32)
    shadow_offset_pts = dst_pts + np.array([[12, 12]], dtype=np.float32)
    cv2.fillConvexPoly(shadow_mask, shadow_offset_pts.astype(np.int32), 0.45)
    shadow_blur = cv2.GaussianBlur(shadow_mask, (31, 31), 10)
    
    out = bg_img.astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - shadow_blur * 0.7)
    
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[mask > 0] = warped[mask > 0]
    return out

def generate_standard_document():
    """Sample 1: Standard A4 report with clear typography."""
    w, h = 700, 990 # A4 proportion
    doc = np.ones((h, w, 3), dtype=np.uint8) * 248
    
    # Blue branding header band
    cv2.rectangle(doc, (0, 0), (w, 80), (180, 100, 30), -1)
    cv2.putText(doc, "PROJECT RESEARCH & ANALYSIS REPORT", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
    
    cv2.putText(doc, "1. Executive Summary", (40, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 40, 40), 2)
    for y in range(165, 290, 25):
        cv2.line(doc, (40, y), (640, y), (140, 140, 140), 1)
        
    cv2.putText(doc, "2. Key Evaluation Metrics", (40, 340), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 40, 40), 2)
    # Table headers
    cv2.rectangle(doc, (40, 370), (640, 405), (220, 220, 220), -1)
    cv2.putText(doc, "Metric ID       Feature Description          Score (%)", (50, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    
    # Table rows
    for i, y in enumerate(range(440, 600, 40)):
        cv2.line(doc, (40, y), (640, y), (200, 200, 200), 1)
        cv2.putText(doc, f"M-0{i+1}          Corner Precision Test          {94.5 + i*1.2:.1f}%", (50, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    cv2.putText(doc, "3. Signature & Authorization", (40, 690), cv2.FONT_HERSHEY_DUPLEX, 0.7, (40, 40, 40), 2)
    cv2.line(doc, (40, 780), (280, 780), (30, 30, 30), 2)
    cv2.putText(doc, "Authorized Signature", (40, 810), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # Stamp
    cv2.circle(doc, (520, 780), 55, (60, 60, 200), 3)
    cv2.putText(doc, "APPROVED", (470, 785), cv2.FONT_HERSHEY_DUPLEX, 0.65, (60, 60, 200), 2)

    # Warp into desk
    bg = create_wood_desk(1200, 950)
    src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    dst_pts = np.array([[220, 110], [920, 160], [1010, 850], [140, 790]], dtype=np.float32)
    return warp_and_blend(doc, src_pts, dst_pts, bg)

def generate_receipt():
    """Sample 2: Store receipt with barcode and itemized list."""
    w, h = 420, 840
    doc = np.ones((h, w, 3), dtype=np.uint8) * 252
    
    cv2.putText(doc, "SUPERMARKET PLUS", (75, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(doc, "Tax Invoice / Receipt", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.line(doc, (30, 95), (390, 95), (60, 60, 60), 1)
    
    items = [
        ("A4 Printing Paper (500s)", "145.00"),
        ("Ballpoint Pen 0.5 Black", "25.00"),
        ("Sticky Notes Pastel", "35.00"),
        ("Clear File Folder A4", "45.00"),
        ("Stainless Scissors 7-in", "69.00"),
        ("Correction Tape 12m", "38.00"),
    ]
    y = 135
    for item, price in items:
        cv2.putText(doc, item, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        cv2.putText(doc, price, (320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        y += 35
        
    cv2.line(doc, (30, y+10), (390, y+10), (100, 100, 100), 1)
    y += 45
    cv2.putText(doc, "SUBTOTAL:", (30, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(doc, "357.00", (310, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 20, 20), 1)
    y += 30
    cv2.putText(doc, "VAT 7%:", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(doc, "24.99", (325, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    y += 35
    cv2.putText(doc, "TOTAL AMOUNT:", (30, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(doc, "$381.99", (295, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
    
    # Barcode simulation
    np.random.seed(42)
    for bx in range(60, 360, 5):
        thick = np.random.choice([1, 2, 3])
        cv2.line(doc, (bx, y+60), (bx, y+130), (0, 0, 0), thick)
    cv2.putText(doc, "* 8 8 5 9 1 0 4 2 3 7 *", (100, y+155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    bg = create_wood_desk(1100, 950)
    src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    dst_pts = np.array([[360, 90], [800, 140], [700, 890], [240, 830]], dtype=np.float32)
    return warp_and_blend(doc, src_pts, dst_pts, bg)

def generate_shadowed_document():
    """Sample 3: Document with diagonal shadow across it (ideal for Shadow Removal & B&W demo)."""
    img = generate_standard_document()
    
    # Create smooth shadow gradient diagonally across the image
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # Diagonal shadow from top-right to bottom-left
    diag = (X + Y) / 2.0
    shadow = np.clip(1.0 - np.exp(-((diag - 0.48)**2) / 0.08) * 0.6, 0.35, 1.0)
    
    img_shadowed = img.astype(np.float32)
    for c in range(3):
        img_shadowed[:, :, c] = img_shadowed[:, :, c] * shadow
        
    return np.clip(img_shadowed, 0, 255).astype(np.uint8)

def generate_heavy_perspective():
    """Sample 4: Document taken from a strong perspective angle (>50 deg tilt)."""
    w, h = 650, 900
    doc = np.ones((h, w, 3), dtype=np.uint8) * 245
    cv2.rectangle(doc, (20, 20), (w-20, h-20), (40, 60, 120), 4)
    cv2.putText(doc, "CERTIFICATE OF ACHIEVEMENT", (55, 100), cv2.FONT_HERSHEY_DUPLEX, 0.7, (30, 40, 90), 2)
    cv2.putText(doc, "THIS ACKNOWLEDGES THAT", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(doc, "ADVANCED COMPUTER VISION", (80, 280), cv2.FONT_HERSHEY_DUPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(doc, "Has successfully completed practical evaluation in", (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(doc, "Homography, Feature Matching & RANSAC", (95, 410), cv2.FONT_HERSHEY_DUPLEX, 0.65, (40, 80, 160), 2)
    
    # Gold seal
    cv2.circle(doc, (w//2, 600), 70, (40, 160, 220), -1)
    cv2.circle(doc, (w//2, 600), 62, (20, 120, 180), 2)
    cv2.putText(doc, "HONOR", (w//2 - 40, 608), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

    bg = create_wood_desk(1200, 900)
    src_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    # Steep perspective trapezoid
    dst_pts = np.array([[380, 120], [790, 130], [1050, 810], [90, 780]], dtype=np.float32)
    return warp_and_blend(doc, src_pts, dst_pts, bg)

if __name__ == "__main__":
    out_dir = os.path.dirname(__file__)
    
    samples = [
        ("sample_1_standard_doc.png", generate_standard_document()),
        ("sample_2_receipt.png", generate_receipt()),
        ("sample_3_with_shadow.png", generate_shadowed_document()),
        ("sample_4_heavy_skew.png", generate_heavy_perspective()),
    ]
    
    for filename, img in samples:
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, img)
        print(f"Generated: {path} ({img.shape[1]}x{img.shape[0]})")
