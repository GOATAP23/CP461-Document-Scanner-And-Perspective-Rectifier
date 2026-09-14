# Automatic Document Scanner & Perspective Rectifier (Prototype)
### ระบบสแกนเอกสารและปรับมุมมองภาพอัตโนมัติ

แอปพลิเคชันสำหรับสแกนและแก้ไขมุมมองภาพถ่ายเอกสารให้เป็นมุมมองหน้าตรง (Top-down view) พร้อมฟิลเตอร์ปรับปรุงคุณภาพเอกสาร

---

## ฟีเจอร์หลัก (Features)

- **ตรวจจับขอบเขตเอกสารอัตโนมัติ (Automatic Document Detection)** — ค้นหาขอบกระดาษด้วย Canny Edge Detection และ Contour Polygon Fitting
- **แก้ไขมุมมองภาพ (Perspective Correction)** — ปรับมุมเอียงของภาพให้ตรงตามสัดส่วนกระดาษ A4 มาตรฐาน หรือตามสัดส่วนจริงของเอกสาร
- **การจับคู่จุดเด่นภาพ (Feature Matching & Homography)** — รองรับอัลกอริทึม SIFT และ ORB พร้อมประเมินระนาบภาพด้วย RANSAC Homography Matrix
- **ฟิลเตอร์ปรับแต่งคุณภาพเอกสาร (Enhancement Filters):**
  - **Original** — ภาพสีต้นฉบับ
  - **Grayscale** — แปลงเป็นภาพขาว-ดำระดับเฉดสีเทา
  - **Magic Color** — ปรับสีสันและความคมชัดให้สดใส
  - **B&W Scanner** — แปลงเป็นเอกสารขาว-ดำ คมชัด ด้วย Adaptive Thresholding
  - **Shadow Removal** — ลบเงาตกกระทบและเกลี่ยพื้นหลังให้สว่างสม่ำเสมอ
  - **Contrast Enhancement (CLAHE)** — เพิ่มคอนทราสต์เฉพาะจุดให้อ่านตัวหนังสือชัดเจน
- **เว็บแอปพลิเคชันใช้งานง่าย (Interactive Web UI)** — พัฒนาด้วย Streamlit รองรับการสลับภาษา (🇹🇭 ภาษาไทย / 🇬🇧 English)

---

## สิ่งที่ต้องติดตั้งก่อนเริ่ม (Prerequisites & Installation)

### 1. โปรแกรมที่ต้องมีในเครื่อง
- **Python 3.9 - 3.12** (ดาวน์โหลดจาก [python.org](https://www.python.org/downloads/) — *สำหรับ Windows แนะนำให้ติ๊ก **"Add Python to PATH"** ขณะติดตั้ง*)
- **Git** สำหรับดาวน์โหลดโปรเจกต์ ([git-scm.com](https://git-scm.com/))

---

### 2. ขั้นตอนการติดตั้งลงในเครื่อง (Step-by-Step Setup)

#### ขั้นตอนที่ 1: Clone Repository
เปิด Terminal / PowerShell แล้วดาวน์โหลดโค้ด:
```bash
git clone https://github.com/GOATAP23/CP461-Document-Scanner-And-Perspective-Rectifier.git
cd CP461-Document-Scanner-And-Perspective-Rectifier
```

#### ขั้นตอนที่ 2: สร้างและเปิดใช้งาน Virtual Environment (แนะนำอย่างยิ่ง)
เพื่อป้องกันแพ็กเกจชนกับโปรเจกต์อื่น:

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(หากพบแจ้งเตือนสิทธิ์ ให้พิมพ์ `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ก่อนเปิดใช้งาน)*
  
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

#### ขั้นตอนที่ 3: ติดตั้ง Dependencies
```bash
# อัปเดต pip และติดตั้งแพ็กเกจทั้งหมดตาม requirements.txt
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(กรณีบน Windows ที่คำสั่ง `python` มีปัญหา ให้ใช้คำสั่ง `py -m pip install -r requirements.txt` แทน)*

---

## วิธีการเปิดใช้งานแอปพลิเคชัน (How to Run)

> **หมายเหตุสำคัญ:** หากพิมพ์ `streamlit run app.py` แล้วพบปัญหา `command not found` หรือไม่สามารถเรียกคำสั่งได้ **ให้รันผ่านโมดูลของ Python โดยตรง** ซึ่งเป็นวิธีที่ถูกต้องและแน่นอนที่สุด:

```bash
# วิธีที่ถูกต้องและแนะนำที่สุด:
python -m streamlit run app.py
```
*(หรือสำหรับ Windows: `py -m streamlit run app.py`)*

เมื่อรันสำเร็จ ระบบจะเปิดหน้าเว็บเบราว์เซอร์ให้อัตโนมัติที่: **`http://localhost:8501`**

---

## ขั้นตอนการใช้งานบนหน้าเว็บ (Web UI Usage)

1. **เลือกภาษา (Language):** ที่แถบเมนูด้านซ้าย (Sidebar) สามารถสลับภาษาระหว่าง 🇹🇭 ภาษาไทย หรือ 🇬🇧 English ได้
2. **อัปโหลดภาพเอกสาร:** ลากและวางไฟล์ภาพถ่ายเอกสารลงในช่องอัปโหลด (รองรับ JPG, PNG, BMP, TIFF, WebP)
3. **ปรับแต่งการประมวลผล (Sidebar):**
   - **Feature Detection Method:** เลือกใช้อัลกอริทึม `SIFT` (เน้นความแม่นยำ) หรือ `ORB` (เน้นความเร็ว)
   - **Enforce A4 Ratio:** เปิดใช้งานหากต้องการบังคับสัดส่วนกระดาษ A4 หรือปิดเพื่อรักษาสัดส่วนจริงของเอกสาร (เช่น ใบเสร็จ, นามบัตร)
   - **Output Filter:** เลือกฟิลเตอร์สำหรับปรับแต่งภาพ (Original, Magic Color, B&W Scanner ฯลฯ)
4. **ตรวจสอบผลลัพธ์และดาวน์โหลด:** เปรียบเทียบภาพก่อน-หลังการประมวลผล และคลิกปุ่ม **Download Scanned Image** เพื่อบันทึกภาพลงเครื่อง
