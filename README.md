
# Thai Text Cleaning Pipeline — README

เอกสารนี้อธิบายวิธีใช้งานสคริปต์ **`Pipeline_clean.py`** สำหรับทำความสะอาดข้อความภาษาไทยพร้อมกลไก PDPA, Thai NER, ตรวจจับสคริปต์ผสม, และแก้เครื่องหมายสระเลื่อนสระลอย

> เคล็ดลับ: บน Bash/PowerShell แนะนำให้ใส่ชื่อไฟล์ที่มีวงเล็บใน `"` เสมอ เช่น `python "Pipeline_clean.py" ...`

---

## การติดตั้ง (ย่อ)
- Python 3.9+
- แพ็กเกจที่ใช้ (บางส่วน): `tqdm`, `regex`, `transformers`, `torch` (ถ้าใช้ Thai NER), `fasttext` (ถ้าจะใช้ตัวกรองเบื้องต้น)
- โมเดล Thai NER แบบ local (ระบุตำแหน่งผ่าน `THAI_NER_PATH` หรือค่าเริ่มต้นในโค้ด)

---

## อินพุต/เอาต์พุต
- **อินพุต**: โฟลเดอร์ที่มีไฟล์ `*.jsonl` (กำหนดลายแพทเทิร์นด้วย `--pattern`)
- **ฟิลด์ข้อความ**: ถ้าไม่ระบุ `--text-keys` ระบบจะพยายามเดา key ทั่วไป เช่น `text`, `content`, `body`, `ocr_text` ฯลฯ; หรือใช้ `--combine-keys` เพื่อรวมหลายคีย์เป็นหนึ่งเดียว
- **เอาต์พุต**: โฟลเดอร์ปลายทาง `--out` จะมีผลลัพธ์ชุดใหญ่ เช่น
  - `legal.jsonl` / `illegal.jsonl` (ข้อมูลที่ผ่าน/ไม่ผ่านเงื่อนไขเบื้องต้น)
  - ไฟล์ย่อย/ไฟล์ส่วน และ `summary.txt` (ตัวเลขสรุปการประมวลผล)
- มีระบบ **checkpoint/resume** ทำงานอัตโนมัติ สามารถสั่ง `--no-resume` หรือ `--clear-ckpt` เพื่อเริ่มใหม่

---

## Quickstart

```bash
python "Pipeline_clean.py" \
  --input data/ \
  --pattern "*.jsonl" \
  --out runs/exp1 \
  --workers 4 --chunk-size 200 --save-every 5 \
  --clean-steps "html,bbcode,emoticon,links,social" \
  --pdpa-policy mask \
  --pdpa-steps "name,email,phone,links,card,account,address" \
  --keep-last 4 \
  --addr-gazetteer thai_provinces_districts.json \
  --thai-ner-full \
  --thai-ner-policy mask \
  --thai-ner-cats "PERSON,PHONE,EMAIL,ADDRESS,NATIONAL_ID,HOSPITAL_IDS" \
  --detect-mixed-script --mixed-policy tag \
  --thai-mark-fix --floating-policy drop --normalize NFC \
  --flags-key "__flags"
```

### เปิดใช้งาน Thai NER แบบแมปชื่อแทน
```bash
python "Pipeline_clean.py" \
  -i data/ -o runs/exp2 --pattern "*.jsonl" \
  --thai-ner-full --thai-ner-policy anonymize \
  --thai-ner-map path/to/token_map.json
```

### เริ่มใหม่/ต่อจากเดิม และการเขียนทับผลลัพธ์
```bash
# เริ่มใหม่แบบล้าง checkpoint
python "Pipeline_clean.py" -i data -o runs/exp3 --clear-ckpt --no-resume

# เขียน output ต่อท้ายไฟล์เดิม (append)
python "Pipeline_clean.py" -i data -o runs/exp4 --append-out
```

---

## อาร์กิวเมนต์หลัก

### ไฟล์และระบบประมวลผล
- `-i, --input`: โฟลเดอร์อินพุต
- `--pattern`: ลายไฟล์ (เช่น `"*.jsonl"`) เริ่มต้น `*.jsonl`
- `-o, --out`: โฟลเดอร์เอาต์พุต (จำเป็น)
- `--append-out`: เขียนต่อท้ายไฟล์เอาต์พุตเดิม
- `--workers`: จำนวน worker ประมวลผล
- `--chunk-size`: จำนวนรายการต่อหนึ่งชิ้นงาน
- `--save-every`: บันทึกผลทุก ๆ N ชิ้นงาน (ช่วยให้ resume ได้ละเอียดขึ้น)
- `--resume` / `--no-resume`: เปิด/ปิดโหมดทำต่อจาก checkpoint (ค่าเริ่มต้นคือเปิด)
- `--clear-ckpt`: ลบ checkpoint ก่อนเริ่ม
- `--dry-run`: โหมดลองรัน (ไม่เขียนผลลัพธ์จริง)
- `--debug-first-n`: ประมวลผลเฉพาะ N แถวแรก (ดีมากสำหรับลองของ)
- `--debug-show-diff`: แสดง diff ก่อน-หลังทำความสะอาด

### การเลือกคีย์ข้อความ
- `--text-keys "k1,k2,..."`: ระบุคีย์ที่เป็นข้อความชัดเจน
- `--combine-keys "k1,k2,..."`: รวมข้อความจากหลายคีย์เข้าด้วยกัน

### ขั้นตอน sanitize/clean เบื้องต้น
- `--clean-steps`: รายการขั้นตอน เช่น `html,bbcode,emoticon,links,social`
  - รองรับชื่อเก่าแนว `remove_*` เพื่อความเข้ากันได้ย้อนหลัง เช่น `remove_html`, `remove_emails`

### PDPA
- `--pdpa-policy`: `skip | mask | anonymize | drop`  
- `--pdpa-steps`: เลือกสิ่งที่จะจัดการ เช่น `name,email,phone,links,card,account,address,id_card`
- ตัวเลือกเสริม:
  - `--keep-last`: จำนวนเลขท้ายที่คงไว้ในโหมด mask (เช่น บัตร/บัญชี)
  - `--card-mode`, `--account-mode`: โหมดการแทนค่า/สุ่มของบัตรและบัญชี
  - `--pdpa-salt`: seed noise สำหรับการทำ anonymize ให้คงที่
  - `--addr-gazetteer`: ไฟล์ gazetteer ไทยสำหรับตรวจจับ/แทนที่ที่อยู่
  - `--no-addr-keep-province`: อนุญาตให้สุ่มจังหวัดใหม่เมื่อ anonymize
  - `--addr-tag`: แท็กสำหรับ mask ที่อยู่ (ค่าเริ่มต้น `[ADDRESS]`)

### Thai NER (เต็มระบบ)
- `--thai-ner-full`: เปิดใช้การทำ NER เต็มรูปแบบ
- `--thai-ner-policy`: `mask | anonymize`
- `--thai-ner-cats`: หมวดที่ต้องการจัดการ เช่น `PERSON,PHONE,EMAIL,ADDRESS,NATIONAL_ID,HOSPITAL_IDS`
- `--thai-ner-map`: ไฟล์แมปโทเค็น (เช่น mapping ของชื่อบุคคล) สำหรับโหมด anonymize

> หมายเหตุ: หากไม่ได้ติดตั้งโมดูล/โมเดล NER ตามที่กำหนด สคริปต์จะข้ามส่วนนี้อัตโนมัติ

### การตรวจจับสคริปต์ผสม + ตัวช่วยแก้เครื่องหมายไทย
- `--detect-mixed-script`: ตรวจจับอักขระไทยติดกับ Latin/ญี่ปุ่น/จีน
- `--mixed-policy`: `skip | tag` (ถ้า `tag` ระบบจะจดธงไว้ใน `--flags-key`)
- `--thai-mark-fix`: เปิดการแก้ **สระลอย**, ดึงสระกลับ, ลบ/แทนที่สระที่ลอย, และจัดกลุ่มเครื่องหมายซ้ำ
- `--floating-policy`: วิธีจัดการสระที่ลอย `drop | mask | keep`
- `--pull-back-gap`: รูปแบบช่องว่างที่จะดึงกลับ `any | tight`
- `--normalize`: ปรับ normalization `NFC | NFKC | none`
- `--flags-key`: ชื่อคีย์ในเอาต์พุตสำหรับเก็บธง เช่น `"__flags"`

### ตัวกรอง fastText (ถ้ามี)
- `--fasttext-model PATH`: ใช้งานโมเดล fastText เพื่อช่วยแยก legal/illegal
- `--no-fasttext`: ปิดการใช้ fastText

### รายชื่อบุคคลสาธารณะ (ถ้ามี)
- `--public-figure-files`: ไฟล์รายชื่อบุคคลสาธารณะเพื่อช่วยลด false positive
- `--no-public-figure-skip`: ปิดการข้ามชื่อที่อยู่ในรายชื่อสาธารณะ

---

## ตัวอย่างการเลือกขั้นตอน PDPA

```bash
# โหมด mask แบบเก็บท้าย 4 ตัว และแท็กที่อยู่เป็น [ADDRESS]
python "Pipeline_clean.py" -i data -o runs/mask \
  --pdpa-policy mask \
  --pdpa-steps "name,email,phone,links,card,account,address" \
  --keep-last 4 --addr-tag "[ADDRESS]"

# โหมด anonymize โดยคงจังหวัดเดิมไว้เมื่อพบในข้อความ
python "Pipeline_clean.py" -i data -o runs/anon \
  --pdpa-policy anonymize \
  --pdpa-steps "name,email,phone,card,account,address,id_card" \
  --pdpa-salt mysecret --addr-gazetteer thai_provinces_districts.json

# โหมด drop (ลบทิ้ง) สำหรับอีเมล/โทรศัพท์/ลิงก์ + ลบแท็กการ์ด/บัญชีที่ถูก mask ออก
python "Pipeline_clean.py" -i data -o runs/drop \
  --pdpa-policy drop \
  --pdpa-steps "email,phone,links,card,account,address,id_card"
```

---

## เคล็ดลับประสิทธิภาพ
- ใช้หลาย GPU: ตั้งค่า `CUDA_VISIBLE_DEVICES=0,1` และปรับ `--workers`
- ปรับ `--chunk-size` ให้เหมาะกับขนาดข้อมูล
- ใช้ `--debug-first-n` เพื่อลองรันอย่างรวดเร็วก่อนประมวลผลงานจริง

---

## โครงสร้างผลลัพธ์โดยย่อ
- `legal.jsonl` / `illegal.jsonl` — บรรทัดละ 1 เร็กคอร์ด (JSON) พร้อมข้อความที่ถูกทำความสะอาด
- `summary.txt` — ตัวเลขสรุป (จำนวนเร็กคอร์ด, สัดส่วน, ฯลฯ)
- อาจมีโฟลเดอร์ชิ้นงานชั่วคราว เพื่อรองรับการ resume/merge อัตโนมัติ

---

## ตัวอย่างข้อมูลอินพุต (JSONL)

```json
{"id": "001", "text": "ติดต่อ พงศกร ได้ที่ 081-234-5678 หรืออีเมล test@example.com"}
{"id": "002", "content": "ที่อยู่: 123 ถ.สาทร กรุงเทพฯ 10110"}
```

## คำถามที่พบบ่อย
- **Thai NER ช้าไป?** ปิดด้วยการไม่ใช้ `--thai-ner-full` หรือรันบน GPU
- **mask ไม่ทับตรงชื่อพอดี?** ตรวจสอบ `--thai-ner-cats` และลองเปิด `--thai-mark-fix` เพื่อให้การแบ่ง token/สระถูกต้องขึ้น

---

# สรุปผลการทดลอง: การหา Context Length และ Overlap ที่เหมาะสมที่สุด

## วัตถุประสงค์
ทดสอบหาค่า context length และ overlap ratio ที่เหมาะสมที่สุดสำหรับการประเมิน Perplexity (PPL) ของโมเดลภาษาไทย โดยพิจารณาจากความสมดุลระหว่าง **เวลาในการประมวลผล (runtime)** และ **ค่า PPL ที่อยู่ในช่วงปกติ** (normal distribution)

## ข้อมูลการทดลอง
- **จำนวน tokens ที่ประเมิน**: 28,510,680 tokens ทุก configuration
- **จำนวน samples**: 143 samples
- **Context lengths ที่ทดสอบ**: 3200, 4000, 6400, 8000, 16000, 32000
- **Overlap ratios ที่ทดสอบ**: 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8

## การจำแนกกลุ่ม Domain

### 📊 **กลุ่มที่ 1: No Outliers (ค่า PPL ใกล้เคียงกัน)**

กลุ่มนี้ประกอบด้วย 6 domains ที่มีค่าสถิติ PPL ใกล้เคียงกัน:

| Domain | MIN | MAX | AVG | STDEV |
|--------|-----|-----|-----|-------|
| **การศึกษา** (Wikipedia) | 8.34 | 32.65 | 22.44 | 5.99 |
| **การแพทย์** | 8.69 | 37.09 | 27.11 | 6.59 |
| **ข่าว** (TPBS + Prachathai) | 27.79 | 109.23 | 81.26 | 19.59 |
| **โซเชียลมีเดีย** (Wongnai) | 7.55 | 28.99 | 21.53 | 5.31 |
| **ท่องเที่ยว** | 8.06 | 30.99 | 22.70 | 5.54 |
| **ธุรกิจและการเงิน** (SCB) | 8.42 | 32.34 | 23.88 | 5.76 |
| **ราชการ** (Parliament) | 11.22 | 39.10 | 28.79 | 6.83 |

### ⚠️ **กลุ่มที่ 2: Outliers (ค่า PPL สูงผิดปกติ)**

กลุ่มนี้ประกอบด้วย 2 domains ที่มีค่า PPL สูงกว่ากลุ่มอื่นอย่างมีนัยสำคัญ:

| Domain | MIN | MAX | AVG | STDEV |
|--------|-----|-----|-----|-------|
| **กฎหมาย** (SupremeCourt) | 477.58 | 1804.53 | 1342.59 | 323.90 |
| **Code** (CodeGen) | 529.39 | 2004.35 | 1491.86 | 359.87 |

---

## 🎯 ค่า Configuration ที่แนะนำ

### **สำหรับกลุ่ม No Outliers**

#### ✅ **Configuration ที่เหมาะสมที่สุด** (Balanced: Speed + PPL Quality)

| Context Length | Overlap | Avg Runtime | PPL Range (Macro) | หมายเหตุ |
|----------------|---------|-------------|-------------------|----------|
| **6400** | **0.8** | **32.58 sec** | **18.48 - 21.77** | ⭐ **แนะนำ**: เร็วที่สุด + PPL ต่ำในช่วงปกติ |
| **8000** | **0.2-0.6** | **18-45 sec** | **15.47 - 17.60** | เร็วมาก + PPL ต่ำที่สุด แต่ overlap น้อย |
| **6400** | **0.2-0.7** | **31-36 sec** | **18.48 - 19.81** | สมดุลดี |

#### 🐌 **Configurations ที่ควรหลีกเลี่ยง** (ช้าเกินไป)

- **Context 3200-4000** + **Overlap ≥ 0.5**: runtime > 200 sec
- **Context 8000** + **Overlap ≥ 0.65**: runtime > 340 sec  
- **Context 16000**: runtime 211-632 sec (ช้ามาก)
- **Context 32000**: runtime 178-237 sec แต่ PPL ไม่เสถียร (มี outliers มาก)

---

### **สำหรับกลุ่ม Outliers (กฎหมาย & Code)**

#### ✅ **Configuration ที่เหมาะสม**

| Context Length | Overlap | Avg Runtime | PPL Range (Macro) | หมายเหตุ |
|----------------|---------|-------------|-------------------|----------|
| **6400** | **0.8** | **36-38 sec** | **1118-1244** | ⭐ เร็ว + PPL ต่ำที่สุดในกลุ่ม |
| **8000** | **0.6-0.8** | **40-592 sec** | **986-1097** | PPL ต่ำที่สุด แต่อาจช้า |
| **4000** | **0.7-0.8** | **32-47 sec** | **1256-1405** | เร็วมาก แต่ PPL สูงกว่าเล็กน้อย |

#### ⚠️ **Configurations ที่ควรระวัง**

- **Context 32000** + **Overlap 0.4-0.55**: PPL ต่ำผิดปกติ (529-746) แต่ไม่เสถียร อาจเกิดจาก noise
- **Context 16000-32000**: PPL สูงมาก (1400-2000+) ไม่แนะนำ

---

## 📈 สรุปผลการวิเคราะห์

### **Normal Distribution Analysis**

ใช้ Z-score เพื่อจำแนกช่วง PPL:
- **−1 ≤ Z ≤ +1**: ค่าปกติ (สีเขียว) ✅
- **Z < -2**: Outlier ต่ำผิดปกติ (สีแดง) ❌  
- **Z > +2**: Outlier สูงผิดปกติ (สีแดง) ❌

### **Key Findings**

1. **Context Length 6400-8000** ให้ผลลัพธ์ที่ดีที่สุดทั้งด้านความเร็วและคุณภาพ PPL
2. **Overlap 0.2-0.6** เหมาะสำหรับการประมวลผลเร็ว
3. **Overlap 0.7-0.8** ให้ PPL ที่เสถียรกว่า แต่ใช้เวลานานขึ้น
4. **Context 32000** มักให้ผลลัพธ์ที่ไม่เสถียร โดยเฉพาะ overlap 0.4-0.75
5. **Domain กฎหมายและ Code** ต้องการ context ที่สูงกว่า เนื่องจากโครงสร้างประโยคซับซ้อน

---

## 💡 คำแนะนำการใช้งาน

### **สำหรับ Production Use**
```
Context Length: 6400
Overlap: 0.8 (80%)
Expected Runtime: ~33 sec per 28M tokens
Expected PPL Range: 18-22 (general domains), 1100-1250 (law/code)
```

### **สำหรับ Quick Evaluation**
```
Context Length: 8000
Overlap: 0.2-0.4 (20-40%)
Expected Runtime: ~18-40 sec per 28M tokens
Expected PPL Range: 15-18 (general domains), 986-1120 (law/code)
```

### **สำหรับ High Accuracy**
```
Context Length: 8000
Overlap: 0.75-0.8 (75-80%)
Expected Runtime: ~435-531 sec per 28M tokens
Expected PPL Range: 15.47-16.61 (most stable)
```

---

## 📝 หมายเหตุ

- ค่า runtime อาจแตกต่างกันตามฮาร์ดแวร์ที่ใช้
- PPL ที่ต่ำเกินไปใน context 32000 อาจเกิดจาก overfitting หรือ noise ในข้อมูล
- แนะนำให้ทดสอบกับข้อมูลจริงก่อนนำไปใช้งาน production

