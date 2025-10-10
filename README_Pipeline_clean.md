
# Thai Text Cleaning Pipeline — README

เอกสารนี้อธิบายวิธีใช้งานสคริปต์ **`Pipeline_clean.py`** สำหรับทำความสะอาดข้อความภาษาไทยพร้อมกลไก PDPA, Thai NER, ตรวจจับสคริปต์ผสม, และแก้เครื่องหมายสระลอย ุ/ู

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
  - `--pdpa-salt`: เกลือสำหรับการทำ anonymize ให้คงเสถียร
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
- `--thai-mark-fix`: เปิดการแก้ **สระลอย ุ/ู**, ดึงสระกลับ, ลบ/แทนที่สระที่ลอย, และจัดกลุ่มเครื่องหมายซ้ำ
- `--floating-policy`: วิธีจัดการ ุ/ู ที่ลอย `drop | mask | keep`
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
- `legal.jsonl` / `illegal.jsonl` — บรรทัดละ 1 ระเบียน (JSON) พร้อมข้อความที่ถูกทำความสะอาด
- `summary.txt` — ตัวเลขสรุป (จำนวนระเบียน, สัดส่วน, ฯลฯ)
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

