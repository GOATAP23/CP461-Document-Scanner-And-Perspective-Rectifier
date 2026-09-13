# แผนการดำเนินงานโครงการ (Implementation Plan)
## โครงการ: Automatic Document Scanner & Perspective Rectifier
**วิชา:** CP461 Introduction to Computer Vision (Semester 1/2026)  
**เป้าหมาย:** พัฒนาแอปพลิเคชันสแกนและจัดปรับมุมมองเอกสารแบบ End-to-End พร้อมระบบปรับปรุงภาพ เพื่อยื่นขอคะแนนเต็ม **Tier 3 (10/10 คะแนน)**

---

## 1. ภาพรวมโครงการและเป้าหมาย (Project Overview & Objectives)

### 1.1 วัตถุประสงค์
1. **การประมวลผลทางคอมพิวเตอร์วิชัน (CV Pipeline):** สร้างระบบตรวจจับขอบเขตและมุมเอกสารกระดาษที่เอียงหรือถ่ายจากมุมต่างๆ โดยใช้สถาปัตยกรรม Feature Matching / Corner Detection ร่วมกับ Geometric Transformation (SIFT/ORB, Homography Matrix, RANSAC และ Perspective Warping)
2. **ระบบการปรับปรุงคุณภาพภาพ (Post-processing):** ปรับแก้ภาพให้อยู่ในสัดส่วนมาตรฐาน A4 (210 x 297 mm) พร้อมฟังก์ชันลบเงา ปรับความสว่าง และแปลงเป็นภาพสแกนแบบ Binarized / Adaptive Thresholding
3. **ระบบเว็บแอปพลิเคชันและการ Deploy (Tier 3 Target):** พัฒนาด้วย Streamlit หรือ Gradio และทำการ Deploy บนระบบคลาวด์สาธารณะ (Hugging Face Spaces หรือ Streamlit Cloud) เพื่อให้เปิดใช้งานผ่าน URL สาธารณะได้ทันที

### 1.2 ตัวชี้วัดความสำเร็จ (Key Success Indicators)
* **เกณฑ์คะแนนเป้าหมาย:** 10 / 10 คะแนน
  * Algorithmic Correctness & Robustness (4.0/4.0 คะแนน)
  * Engineering & UI Implementation - Tier 3 Deployed Public App (3.0/3.0 คะแนน)
  * 10-Minute Presentation & Demo Clip (3.0/3.0 คะแนน)

---

## 2. โครงสร้างสถาปัตยกรรมระบบ (Technical Architecture)

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
[ Corner Ordering Algorithm ] ──► (Sort 4 Corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left)
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

## 3. การแบ่งหน้าที่และความรับผิดชอบ (Team Roles & Task Allocation - 5 Members)

| ลำดับ | บทบาท (Role) | สมาชิกที่รับผิดชอบ | ขอบเขตงานหลัก (Key Responsibilities) |
|---|---|---|---|
| 1 | **Lead CV Engineer (Pipeline Core)** | สมาชิกคนที่ 1 | - พัฒนา Algorithm การสกัด Keypoint (SIFT/ORB)<br>- คำนวณ Homography Matrix และ RANSAC<br>- ทำการ Warp Perspective ภาพเข้าสู่สัดส่วน A4 |
| 2 | **Robustness & Edge-Case Specialist** | สมาชิกคนที่ 2 | - พัฒนาระบบสำรอง (Fallback Contour/Canny Detection)<br>- จัดการกรณีเอกสารขอบไม่ชัดเจน แสงสะท้อน หรือภาพเอียงมาก<br>- ทำ Post-processing (Adaptive Thresholding, Shadow Removal) |
| 3 | **Frontend & UI Developer** | สมาชิกคนที่ 3 | - ออกแบบและพัฒนา Web UI ด้วย Streamlit / Gradio<br>- ทำระบบ Drag-and-Drop ภาพ และ Interactive Corner Adjustment (ถ้ามี)<br>- จัดทำหน้าแสดงผลเปรียบเทียบ Before/After และ Inlier Matching Visualizations |
| 4 | **DevOps & Integration Lead** | สมาชิกคนที่ 4 | - จัดตั้ง GitHub Repository ให้เป็นระเบียบตามมาตรฐาน<br>- จัดทำ `requirements.txt` และทำ Docker/Deployment สู่ Hugging Face Spaces<br>- ทดสอบ Performance และ Load Speed ของ Public App |
| 5 | **QA, Video & Presentation Director** | สมาชิกคนที่ 5 | - ทดสอบระบบและรวบรวม Test Dataset (เคสปกติ / เคสยาก)<br>- อัดคลิปวิดีโอเดโมความยาวไม่เกิน 10 นาที (ตามข้อกำหนด)<br>- ควบคุมสไลด์นำเสนอและการแบ่งพูดของสมาชิกทุกคนให้สมดุล |

---

## 4. แผนการดำเนินงานรายสัปดาห์ (Sprint Schedule & Timeline)

```
+-------------------------------------------------------+
| Week 1: Core CV Pipeline Development                 |
| Week 2: Robustness, Post-Processing & Edge Cases      |
| Week 3: Web UI & Tier 3 Cloud Deployment              |
| Week 4: QA Testing, Video Production & Submission     |
+-------------------------------------------------------+
```

### สัปดาห์ที่ 1: การพัฒนา Core CV Pipeline (Week 1)
* [ ] ศึกษาเปรียบเทียบ SIFT vs. ORB สำหรับงานจัดมุมมองเอกสาร
* [ ] Implement ฟังก์ชัน Grayscale, Gaussian Blur และ Edge Detection (Canny)
* [ ] Implement การหา 4 มุมเอกสารและเรียงลำดับพิกัด (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
* [ ] พัฒนาฟังก์ชันคำนวณ Homography Matrix ด้วย RANSAC และ `cv2.warpPerspective`
* [ ] **Deliverable สัปดาห์ที่ 1:** Colab Notebook ที่สามารถรับภาพเอกสารเอียงแล้ว Warp เป็นภาพตรง A4 ได้สำเร็จ

### สัปดาห์ที่ 2: ปรับปรุงความสมบูรณ์และแก้ Edge Cases (Week 2)
* [ ] เพิ่มอัลกอริทึม Feature Matching (SIFT/ORB) กับ Template A4 เพื่อหาพื้นที่กระดาษในกรณีพื้นหลังกลมกลืน
* [ ] พัฒนา Fallback Mechanism: หาก Feature Matching ล้มเหลว ให้ใช้วิธี Contour Detection
* [ ] พัฒนาฟังก์ชันปรับแต่งคุณภาพภาพสแกน (Post-processing Filters):
  * Gray Scale / Magic Color
  * B&W Scanner Effect (Adaptive Thresholding)
  * Shadow Removal / Contrast Enhancement
* [ ] **Deliverable สัปดาห์ที่ 2:** Python Class/Module ที่ครอบคลุมทุก Edge Cases พร้อมระบบ Visualization (Inlier Plotting)

### สัปดาห์ที่ 3: พัฒนา Web App และ Deploy ขึ้น Tier 3 (Week 3)
* [ ] พัฒนา Web Application ด้วย Streamlit หรือ Gradio
  * โหมด Upload ภาพเอกสาร
  * โหมดปรับแต่งพารามิเตอร์ (Threshold, Filter Type, Aspect Ratio)
  * หน้าแสดงภาพเปรียบเทียบ Original vs. Rectified vs. Enhanced
  * แสดงกราฟ/ภาพการทำงานของ SIFT/RANSAC Inliers เพื่อตอบโจทย์ Rubric
* [ ] สร้าง GitHub Repository (โครงสร้างสะอาด มี `requirements.txt` และ `README.md` ที่สมบูรณ์)
* [ ] Deploy Web App บน Hugging Face Spaces หรือ Streamlit Cloud ( Tier 3 Validation )
* [ ] **Deliverable สัปดาห์ที่ 3:** Public URL สำหรับใช้งาน Web App ที่ทุกคนเข้าถึงได้

### สัปดาห์ที่ 4: การทดสอบ วิดีโอเดโม และจัดส่งงาน (Week 4)
* [ ] ทดสอบแอปพลิเคชันอย่างเข้มงวดด้วยภาพทดสอบหลากหลายสภาวะแสงและมุมถ่าย
* [ ] จัดทำสไลด์นำเสนอ (เน้นทฤษฎี SIFT/ORB, RANSAC, Homography และสถาปัตยกรรมระบบ)
* [ ] บันทึกวิดีโอความยาว **ไม่เกิน 10 นาที** (ตามข้อกำหนด):
  * อธิบายหลักการทางเทคนิคและการตัดสินใจเชิงวิศวกรรม (1.5 คะแนน)
  * สาธิตการใช้งานแอปพลิเคชันจริงแบบ Live Demo + เคสยาก/Edge Cases (1.0 คะแนน)
  * การมีส่วนร่วมของสมาชิกทุกคนในทีมอย่างสมดุล (0.5 คะแนน)
* [ ] อัปโหลดวิดีโอขึ้น YouTube (Unlisted) หรือ Google Drive
* [ ] ตรวจสอบแพ็กเกจส่งงาน (Source Code, Executable Link, Video Link)
* [ ] **Deliverables สัปดาห์ที่ 4:** แพ็กเกจฉบับสมบูรณ์พร้อมยื่นส่ง

---

## 5. รายละเอียดทางเทคนิคของอัลกอริทึม (Technical Pipeline Details)

### 5.1 ขั้นตอนการเรียงลำดับมุม (Corner Ordering Algorithm)
เพื่อให้การ Warp ภาพแม่นยำ ต้องจัดเรียงพิกัดทั้ง 4 มุม $(x, y)$ ให้อยู่ในลำดับคงที่เสมอ:
1. **Top-Left (มุมซ้ายบน):** จุดที่มีผลรวม $(x + y)$ น้อยที่สุด
2. **Bottom-Right (มุมขวาใน):** จุดที่มีผลรวม $(x + y)$ มากที่สุด
3. **Top-Right (มุมขวาบน):** จุดที่มีผลต่าง $(y - x)$ น้อยที่สุด
4. **Bottom-Left (มุมซ้ายล่าง):** จุดที่มีผลต่าง $(y - x)$ มากที่สุด

### 5.2 การคำนวณขนาดและสัดส่วน A4 (Perspective Transformation)
คำนวณความกว้างและความสูงใหม่ของภาพจากระยะทางยูคลิด (Euclidean Distance):
$$	ext{Width}_A = \sqrt{(x_{br} - x_{bl})^2 + (y_{br} - y_{bl})^2}$$
$$	ext{Width}_B = \sqrt{(x_{tr} - x_{tl})^2 + (y_{tr} - y_{tl})^2}$$
$$	ext{MaxWidth} = \max(	ext{Width}_A, 	ext{Width}_B)$$

หลังจากนั้นแมปไปยังพิกัดเป้าหมายตามสัดส่วน $1 : 1.414$ (สัดส่วนกระดาษ A4) แล้วทำการ Warp ด้วย Homography Matrix ($H$):
$$P_{	ext{target}} = H \cdot P_{	ext{source}}$$

---

## 6. แผนการรับมือความเสี่ยงและ Edge Cases (Risk Management)

| ความเสี่ยง / Edge Case | ผลกระทบ | วิธีการแก้ไขและบรรเทาปัญหา (Mitigation Strategy) |
|---|---|---|
| **พื้นหลังสีใกล้เคียงกับกระดาษ** | ตรวจจับขอบกระดาษผิดพลาด | ใช้ SIFT Descriptor Matching กับ Template กระดาษ A4 หรือปรับพารามิเตอร์ Bilateral Filter |
| **เงาตกกระทบและแสงสะท้อน (Glare)** | เกิดรอยเงาในภาพสแกน | ใช้ Illumination Normalization และ Adaptive Thresholding (Gaussian/Mean) |
| **มุมภาพเอียง extreme (> 60 องศา)** | Keypoints จับไม่พอ | เพิ่ม RANSAC Reprojection Error Threshold และมีระบบ Interactive Manual Point Adjustment บน UI |
| **การ Deploy บน Cloud ล้มเหลว** | เสียคะแนน Tier 3 | ทำ Docker Container และเริ่ม Deploy บน Hugging Face Spaces ตั้งแต่สัปดาห์ที่ 3 |

---

## 7. ตารางตรวจสอบก่อนส่งงาน (Submission Final Checklist)

- [ ] **Source Code Repo:** Clean GitHub Repository พร้อม `requirements.txt` และ `README.md`
- [ ] **Tier 3 Executable Link:** Public URL ของ Web App ที่เปิดใช้งานได้จริงบน Cloud
- [ ] **Demonstration Video:** 
  - [ ] ความยาวไม่เกิน 10 นาทีพอดี
  - [ ] มีเสียงพากย์อธิบายหลักการ CV / SIFT / RANSAC / Homography
  - [ ] แสดงการใช้งาน Live Demo กับภาพทดสอบของตัวเอง
  - [ ] สมาชิกทั้ง 5 คนมีบทบาทในการนำเสนอ
- [ ] **Colab Notebook:** มีขั้นตอนการทำงาน Visualization ของ RANSAC/Keypoints ชัดเจน

