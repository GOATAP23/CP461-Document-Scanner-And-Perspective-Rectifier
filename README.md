# 📄 Automatic Document Scanner & Perspective Rectifier
### ระบบสแกนเอกสารและปรับมุมมองภาพอัตโนมัติ

> **CP461 Introduction to Computer Vision — ภาคการศึกษา 1/2569 (Semester 1/2026)**

แอปพลิเคชันสำหรับสแกนและแก้ไขมุมมองภาพถ่ายเอกสารให้เป็นมุมมองหน้าตรง (Top-down view) แบบครบวงจร (End-to-End) ด้วยเทคนิค Computer Vision ขั้นสูง ประกอบด้วยการสกัดและจับคู่จุดเด่น (SIFT / ORB Feature Matching), การคำนวณ Homography Matrix, การตัดจุดรบกวนด้วย RANSAC และการดัดสัดส่วนภาพ Perspective Warping สู่ขนาดมาตรฐาน A4

---

## 🎯 ฟีเจอร์หลัก (Key Features)

- **ตรวจจับขอบเขตเอกสารอัตโนมัติ (Automatic Document Detection)** — ค้นหาขอบกระดาษด้วย Canny Edge Detection และการประมาณรูปหลายเหลี่ยมจาก Contour
- **การจับคู่จุดเด่นภาพ (Feature Matching)** — สกัดและจับคู่จุด Keypoint ด้วย SIFT และ ORB พร้อมคัดกรองคู่แมตช์ด้วย Lowe's Ratio Test
- **ตัด Outlier อย่างแม่นยำด้วย RANSAC** — ประเมินเมทริกซ์การแปลงพิกัด (Homography) ที่ทนทานต่อสัญญาณรบกวน
- **แก้ไขมุมมองภาพ (Perspective Correction)** — ปรับมุมเอียงของภาพให้ตรงตามสัดส่วนกระดาษ A4 มาตรฐาน (อัตราส่วน 1:√2) หรือตามสัดส่วนจริงของวัตถุ
- **ฟิลเตอร์ปรับแต่งคุณภาพเอกสาร (Multiple Enhancement Filters):**
  - 📷 **Original** — แสดงภาพสีจริงโดยไม่ผ่านฟิลเตอร์
  - 🔲 **Grayscale** — แปลงเป็นภาพขาว-ดำระดับเฉดสีเทา
  - 🌈 **Magic Color** — ปรับสีสันและเร่งความสดใสของเนื้อหา (คล้ายโหมด Magic Color ใน CamScanner)
  - 📝 **B&W Scanner** — แปลงเป็นเอกสารสแกน 2 สี คมชัด อ่านง่าย ด้วย Adaptive Thresholding
  - ☀️ **Shadow Removal** — ลบเงาตกกระทบและปรับความสว่างพื้นหลังให้สม่ำเสมอ
  - 🔆 **Contrast Enhancement (CLAHE)** — เพิ่มคอนทราสต์เฉพาะจุดเพื่อให้อ่านข้อความชัดเจนยิ่งขึ้น
- **เว็บแอปพลิเคชันใช้งานง่าย (Interactive Web UI)** — พัฒนาด้วย Streamlit รองรับการลากวางไฟล์และปรับจูนค่าได้แบบเรียลไทม์
- **ระบบตรวจจับสำรอง (Fallback Detection)** — มีกลยุทธ์สำรองหลายระดับเพื่อรองรับภาพที่มีคอนทราสต์ต่ำหรือมีสิ่งรบกวนสูง

---

## 🏗️ สถาปัตยกรรมระบบ (System Architecture)

```
[ ภาพถ่ายอินพุต (Input Image) ]
              │
              ▼
[ การประมวลผลเบื้องต้น (Preprocessing) ]
  ├── แปลงภาพเป็น Grayscale
  ├── ลดสัญญาณรบกวนด้วย Gaussian Blur / Bilateral Filter
  └── ตรวจจับเส้นขอบด้วย Canny Edge Detection
              │
              ▼
[ การตรวจจับจุดเด่นและมุมเอกสาร (Feature & Corner Detection) ]
  ├── สกัด Keypoints ด้วย SIFT หรือ ORB
  └── หรือหาขอบเขต 4 มุมด้วย Contour Polygon Fitting
              │
              ▼
[ การจับคู่จุดเด่นและคัดกรองด้วย RANSAC ]
  ├── กรองคู่แมตช์ด้วย Lowe's Ratio Test
  └── ตัดจุดผิดปกติ (Outliers) ด้วย RANSAC
              │
              ▼
[ การจัดเรียงลำดับมุมทั้ง 4 (Corner Ordering) ]
  └── จัดเรียงตำแหน่ง: บน-ซ้าย (TL), บน-ขวา (TR), ล่าง-ขวา (BR), ล่าง-ซ้าย (BL)
              │
              ▼
[ การคำนวณ Homography Matrix ]
  └── คำนวณเมทริกซ์การแปลงพิกัด 3×3 (cv2.findHomography / cv2.getPerspectiveTransform)
              │
              ▼
[ การดัดมุมมองภาพ (Perspective Warping) ]
  └── ทำ Warp Perspective ปรับระนาบเอกสารสู่ขนาดมาตรฐาน (A4 Ratio หรือ Real Aspect Ratio)
              │
              ▼
[ การปรับปรุงคุณภาพและแปลงภาพ (Enhancement & Binarization) ]
  └── ตกแต่งภาพด้วย Adaptive Thresholding, Color Enhancement, Shadow Removal
              │
              ▼
[ แสดงผลลัพธ์บน Web UI & ดาวน์โหลดเอกสาร ]
```

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
├── app.py                          # เว็บแอปพลิเคชันหลัก (Streamlit Web UI)
├── requirements.txt                # รายการ Dependencies / Libraries ของ Python
├── README.md                       # เอกสารคู่มือโปรเจกต์
├── .gitignore                      # กำหนดไฟล์ที่ไม่ต้อง Track ใน Git
│
├── docs/                           # เอกสารรายงานและแผนงานโปรเจกต์
│   ├── README.md
│   └── Document_Scanner_Implementation_Plan.md
│
├── test_images/                    # ชุดภาพทดสอบ (เคสปกติและเคสยาก)
│   ├── README.md
│   ├── generate_sample.py          # สคริปต์สร้างภาพทดสอบสังเคราะห์
│   └── sample_document.png         # ภาพตัวอย่างเอกสารสังเคราะห์
│
├── notebooks/                      # Colab / Jupyter Notebooks สำหรับแสดงขั้นตอนการทำงาน
│   └── README.md
│
└── scanner/                        # แพ็กเกจโมดูล Computer Vision
    ├── __init__.py                 # โมดูลนำเข้าหลัก (Top-level exports)
    ├── pipeline.py                 # Pipeline Orchestrator ควบคุมลำดับการประมวลผลทั้งหมด
    │
    ├── core/                       # ฟังก์ชันหลักด้านการคำนวณเรขาคณิตและภาพ
    │   ├── __init__.py
    │   ├── preprocessing.py        # แปลง Grayscale, Blur, Canny Edge Detection
    │   ├── corner_detection.py     # Contour Fitting & การจัดเรียงมุมทั้ง 4
    │   └── perspective_transform.py# คำนวณ Homography Matrix & Perspective Warp
    │
    ├── features/                   # โมดูลสกัดและจับคู่จุดเด่นภาพ
    │   ├── __init__.py
    │   └── feature_matching.py     # SIFT/ORB Keypoints, Lowe's Ratio Test, RANSAC
    │
    ├── filters/                    # โมดูลฟิลเตอร์ปรับปรุงคุณภาพภาพ
    │   ├── __init__.py
    │   └── enhancement.py          # B&W Scanner, Shadow Removal, Magic Color, CLAHE
    │
    └── utils/                      # ฟังก์ชันช่วยเหลือและระบบ Fallback
        ├── __init__.py
        └── fallback.py             # กลยุทธ์ตรวจจับสำรอง (Hough Lines, Multi-threshold)
```

---

## 🚀 การติดตั้งและเริ่มต้นใช้งาน (Quick Start)

### 1. สิ่งที่ต้องมีในเครื่องก่อนติดตั้ง (Prerequisites)

- **Python 3.9 - 3.12** (แนะนำ 3.10 หรือ 3.11)
  - ตรวจสอบเวอร์ชันใน Terminal / Command Prompt:
    ```bash
    python --version   # หรือ python3 --version
    ```
  - หากยังไม่ได้ติดตั้ง สามารถดาวน์โหลดได้ที่ [python.org](https://www.python.org/downloads/) *(สำหรับ Windows แนะนำให้ติ๊ก **"Add Python to PATH"** ขณะติดตั้ง)*
- **Git** สำหรับ Clone โค้ด ([git-scm.com](https://git-scm.com/))
- *(สำหรับ Windows)* **Microsoft Visual C++ Redistributable** (มักมีในเครื่องอยู่แล้ว หากไม่มีดาวน์โหลดได้จาก [Microsoft Official](https://aka.ms/vs/17/release/vc_redist.x64.exe))

---

### 2. ขั้นตอนการติดตั้งทีละสเต็ป (Installation Steps)

#### ขั้นตอนที่ 1: Clone Repository
```bash
git clone https://github.com/GOATAP23/CP461-Document-Scanner-And-Perspective-Rectifier.git
cd CP461-Document-Scanner-And-Perspective-Rectifier
```

#### ขั้นตอนที่ 2: สร้างและเปิดใช้งาน Virtual Environment (แนะนำอย่างยิ่ง)
เพื่อป้องกันไม่ให้เวอร์ชันของแพ็กเกจชนกับโปรเจกต์อื่นในเครื่อง:

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(หากพบข้อความแจ้งเตือน Execution Policy ให้รัน `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ก่อน)*

- **Windows (Command Prompt / CMD):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

เมื่อเปิดใช้งานสำเร็จ จะมีคำว่า `(venv)` ขึ้นที่หน้าบรรทัดคำสั่ง Terminal

#### ขั้นตอนที่ 3: ติดตั้ง Dependencies
```bash
# อัปเดต pip เป็นเวอร์ชันล่าสุด
python -m pip install --upgrade pip

# ติดตั้งไลบรารีทั้งหมดตาม requirements.txt
pip install -r requirements.txt
```

**รายการแพ็กเกจสำคัญที่ใช้งาน:**
| แพ็กเกจ (Package) | เวอร์ชัน | หน้าที่การทำงาน |
|---|---|---|
| `opencv-python-headless` | 4.10.0.84 | แกนหลักการประมวลผลภาพ, SIFT/ORB, Canny, Homography & Perspective Warp |
| `streamlit` | >= 1.38.0 | สร้าง Web Application UI และ Interactive Controls |
| `numpy` | >= 1.24.0 | การคำนวณทางคณิตศาสตร์ เมทริกซ์ และการจัดการพิกเซล |
| `scikit-image` | >= 0.22.0 | การวิเคราะห์โครงสร้างภาพและฟิลเตอร์ขั้นสูง |
| `Pillow` | >= 10.0.0 | จัดการและโหลดไฟล์ภาพรูปแบบต่าง ๆ |

---

### 3. การรันแอปพลิเคชัน (Run Application)

เมื่อติดตั้งเสร็จเรียบร้อย เริ่มต้นรันเว็บแอปได้ทันทีด้วยคำสั่ง:

```bash
streamlit run app.py
```
*(หรือใช้คำสั่ง: `python -m streamlit run app.py`)*

ระบบจะเปิดหน้าต่างเบราว์เซอร์ให้อัตโนมัติที่: **`http://localhost:8501`**

---

### 4. การทดสอบความถูกต้องของระบบ (Verification & Testing)

* **สร้างภาพเอกสารสังเคราะห์สำหรับทดสอบ:**
  ```bash
  python test_images/generate_sample.py
  ```
  ภาพตัวอย่างจะถูกบันทึกไว้ที่ `test_images/sample_document.png`

* **รันชุดการทดสอบ Unit Tests ทั้งหมด:**
  ```bash
  python -m unittest discover tests
  ```

---

### 5. วิธีการใช้งานบนหน้าเว็บ (Usage Instructions)

1. **อัปโหลดภาพ:** ลากและวางไฟล์ภาพถ่ายเอกสาร (รองรับ JPG, PNG, BMP, TIFF, WebP) ลงในกล่องอัปโหลด
2. **ปรับแต่งพารามิเตอร์ (แถบด้านซ้าย - Sidebar):**
   - **Feature Detection Method:** เลือกใช้อัลกอริทึมจับคู่จุดเด่น `SIFT` (เน้นความแม่นยำสูง) หรือ `ORB` (เน้นความเร็วในการประมวลผล)
   - **Enforce A4 Ratio:** ติ๊กเลือกหากเป็นเอกสาร A4 หรือเอาออกหากต้องการสัดส่วนจริง (เช่น ใบเสร็จ, สลิป, นามบัตร)
   - **Output Filter:** เลือกฟิลเตอร์แต่งภาพตามต้องการ (เช่น B&W Scanner, Magic Color, Shadow Removal)
3. **ตรวจสอบผลลัพธ์:** ดูการแสดงผลเปรียบเทียบภาพต้นฉบับกับภาพที่ถูกดัดมุมมองและปรับแต่งแล้ว พร้อมแสดงค่า Homography Matrix
4. **ดาวน์โหลดเอกสาร:** คลิกปุ่ม Download Scanned Image เพื่อบันทึกภาพผลลัพธ์

---

### 6. การแก้ปัญหาที่พบบ่อย (Troubleshooting)

- **ปัญหา `streamlit: command not found`:**
  - เกิดจากการที่ Terminal ไม่ได้เปิด Virtual Environment หรือไม่ได้ตั้งค่า PATH
  - ให้รันผ่าน Python แทน: `python -m streamlit run app.py`
- **ปัญหา PowerShell ปิดกั้นการรันสคริปต์ (`Activate.ps1 cannot be loaded`):**
  - ให้พิมพ์คำสั่ง: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ใน PowerShell ก่อนรัน Activate
- **ปัญหา `ImportError: DLL load failed` ของ OpenCV (บน Windows):**
  - ติดตั้ง [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) เพิ่มเติมในเครื่อง

---

## 🔧 รายละเอียดทางเทคนิค (Technical Details)

### 1. อัลกอริทึมการเรียงลำดับมุมทั้ง 4 (Corner Ordering Algorithm)
เพื่อให้การ Warp Perspective ได้ระนาบที่ถูกต้อง สี่เหลี่ยมจะต้องถูกจัดเรียงตำแหน่งมุมอย่างสอดคล้องกันเสมอ โดยใช้ผลรวมและผลต่างของพิกัด $(x, y)$:
- **บน-ซ้าย (Top-Left):** จุดที่มีค่าผลรวมน้อยที่สุด $\min(x + y)$
- **ล่าง-ขวา (Bottom-Right):** จุดที่มีค่าผลรวมมากที่สุด $\max(x + y)$
- **บน-ขวา (Top-Right):** จุดที่มีค่าผลต่างน้อยที่สุด $\min(y - x)$
- **ล่าง-ซ้าย (Bottom-Left):** จุดที่มีค่าผลต่างมากที่สุด $\max(y - x)$

### 2. การคำนวณ Homography & Perspective Transform
การแปลงพิกัดจากภาพถ่ายเอียง $(P_{\text{source}})$ ไปยังภาพเอกสารระนาบตรง $(P_{\text{target}})$ ทำได้โดยใช้สมการความสัมพันธ์โปรเจกทีฟ:
$$P_{\text{target}} = H \cdot P_{\text{source}}$$
โดยที่ $H$ คือ เมทริกซ์โฮโมกราฟีขนาด $3 \times 3$ ที่คำนวณผ่าน `cv2.getPerspectiveTransform` หรือ `cv2.findHomography` ร่วมกับ RANSAC เพื่อตัดคู่จุดที่ไม่สอดคล้องกันออก

---

## 📦 การนำไปติดตั้งใช้งานบนคลาวด์ (Cloud Deployment)

### บน Hugging Face Spaces
1. สร้าง Space ใหม่บน [Hugging Face](https://huggingface.co/spaces)
2. เลือก SDK เป็น **Streamlit**
3. Push ซอร์สโค้ดใน Repository นี้ขึ้นไปยัง Space
4. แอปจะเปิดใช้งานผ่าน Public URL อัตโนมัติ

### บน Streamlit Cloud
1. Push ซอร์สโค้ดขึ้น GitHub
2. ไปที่ [Streamlit Community Cloud](https://share.streamlit.io)
3. เชื่อมต่อบัญชีกับ GitHub Repository แล้วเลือกไฟล์หลักเป็น `app.py`
4. กด Deploy

---

## 👥 สมาชิกในทีมและหน้าที่รับผิดชอบ (Team Roles)

| บทบาทหน้าที่ | รายละเอียดความรับผิดชอบ |
|---|---|
| **Lead CV Engineer** | พัฒนาอัลกอริทึมหลัก SIFT/ORB Pipeline, Homography Matrix และ RANSAC |
| **Robustness Specialist** | พัฒนาระบบตรวจจับสำรอง (Fallback), จัดการ Edge Cases และฟิลเตอร์แต่งภาพ |
| **Frontend Developer** | ออกแบบและพัฒนาหน้าเว็บส่วนติดต่อผู้ใช้ด้วย Streamlit พร้อม Data Visualization |
| **DevOps Lead** | ดูแล Git Repository, CI/CD, การรัน Unit Tests และการ Deploy บน Cloud |
| **QA & Presentation** | ทดสอบระบบ, จัดทำชุดข้อมูลทดสอบ (Test Dataset), จัดทำวิดีโอเดโมและสไลด์นำเสนอ |

---

## 📜 ลิขสิทธิ์และการใช้งาน (License)

โครงงานนี้พัฒนาขึ้นเพื่อการศึกษา เป็นส่วนหนึ่งของรายวิชา **CP461 Introduction to Computer Vision**
