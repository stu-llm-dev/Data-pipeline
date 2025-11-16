
# Thai Text Cleaning Pipeline
เอกสารนี้อธิบายวิธีใช้งานสคริปต์ **`Pipeline_clean.py`** สำหรับทำความสะอาดข้อความภาษาไทยพร้อมกลไก PDPA, Thai NER, ตรวจจับ gibberish(ข้อความขยะ), และแก้เครื่องหมายสระเลื่อนสระลอย

---

## การติดตั้ง
- Python 3.9+
- แพ็กเกจที่ใช้ (บางส่วน): `tqdm`, `regex`, `transformers`, `torch` (ถ้าใช้ Thai NER), `fasttext` (ถ้าจะใช้ตัวกรองเบื้องต้น)
```bash
pip install beautifulsoup4 fasttext https://github.com/PyThaiNLP/Thai-Data-Privacy/archive/master.zip 
```
- โมเดล Thai NER แบบ local (ระบุตำแหน่งผ่าน `THAI_NER_PATH` หรือค่าเริ่มต้นในโค้ด)
- สามารถดาวน์โหลดโมเดลจากลิงก์นี้
  
## fasttext
```bash
https://huggingface.co/ThanadolSav/fasttext-filter/blob/main/fasttext.bin
```
## loolootech/no-name-ner-th
```bash
https://huggingface.co/loolootech/no-name-ner-th
```
- เมื่อดาวน์โหลดโมเดลมาเรียบร้อยแล้วให้นำไปวางไว้ที่โฟลเดอร์ model

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

เพื่อให้ระบบสามารถโหลด Thai NER model ได้อย่างถูกต้อง
ให้ตั้งค่า THAI_NER_PATH ชี้ไปยังโฟลเดอร์โมเดลบนเครื่องของคุณ

### Linux / macOS

```bash
export THAI_NER_PATH="model/NERmodel"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
จากนั้น
```bash
python Pipeline_clean.py \
  --input data/ \
  --pattern "*.jsonl" \
  --out runs/ \
  --workers 4 \
  --chunk-size 200 \
  --save-every 5 \
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
  --flags-key "__flags" \
  --text-keys "text" \
  --fasttext-model "model/fasttext.bin"
```
### Windows PowerShell

```bash
$env:THAI_NER_PATH="model\NERmodel"
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```
จากนั้น
```bash
python .\Pipeline_clean.py `
  --input "data" `
  --pattern "*.jsonl" `
  --out "runs" `
  --workers 4 `
  --chunk-size 200 `
  --save-every 5 `
  --clean-steps "html,bbcode,emoticon,links,social" `
  --pdpa-policy mask `
  --pdpa-steps "name,email,phone,links,card,account,address" `
  --keep-last 4 `
  --addr-gazetteer "thai_provinces_districts.json" `
  --thai-ner-full `
  --thai-ner-policy mask `
  --thai-ner-cats "PERSON,PHONE,EMAIL,ADDRESS,NATIONAL_ID,HOSPITAL_IDS" `
  --detect-mixed-script `
  --mixed-policy tag `
  --thai-mark-fix `
  --floating-policy drop `
  --normalize NFC `
  --flags-key "__flags" `
  --text-keys "text" `
  --fasttext-model "model\fasttext.bin"
```

### เปิดใช้งาน Thai NER แบบแมปชื่อแทน
```bash
python Pipeline_clean.py \
  -i data/ -o runs/exp2 --pattern "*.jsonl" \
  --thai-ner-full --thai-ner-policy anonymize \
  --thai-ner-map path/to/token_map.json
```

### เริ่มใหม่/ต่อจากเดิม และการเขียนทับผลลัพธ์
```bash
# เริ่มใหม่แบบล้าง checkpoint
python Pipeline_clean.py -i data -o runs/exp3 --clear-ckpt --no-resume

# เขียน output ต่อท้ายไฟล์เดิม (append)
python Pipeline_clean.py -i data -o runs/exp4 --append-out
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
# โหมด mask แบบเก็บท้าย 4 ตัว หรือสามารถปรับได้และแท็กที่อยู่เป็น [ADDRESS]
python Pipeline_clean.py -i data -o runs/mask \
  --pdpa-policy mask \
  --pdpa-steps "name,email,phone,links,card,account,address" \
  --keep-last 4 --addr-tag "[ADDRESS]"

# โหมด anonymize โดยคงจังหวัดเดิมไว้เมื่อพบในข้อความ
python Pipeline_clean.py -i data -o runs/anon \
  --pdpa-policy anonymize \
  --pdpa-steps "name,email,phone,card,account,address,id_card" \
  --pdpa-salt mysecret --addr-gazetteer thai_provinces_districts.json

# โหมด drop (ลบทิ้ง) สำหรับอีเมล/โทรศัพท์/ลิงก์ + ลบแท็กการ์ด/บัญชีที่ถูก mask ออก
python Pipeline_clean.py -i data -o runs/drop \
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
---


<br>
<br>

# Perplexity — คู่มือการใช้งาน (CLI)  

### โหมดไฟล์เดี่ยว หรือ ข้อความเดียว (default; ไม่ใช้ subcommand)
```bash 
  python Perplexity.py --model <MODEL_OR_PATH> [--file <PATH> | --text "ข้อความ"] [options...]
```

### โหมดทั้งโฟลเดอร์ (ใช้ subcommand 'folder')
```bash
  python Perplexity.py folder --folder <DIR> --model <MODEL_OR_PATH> [options...]
```
---

## อาร์กิวเมนต์สำคัญ
- --model               ชื่อโมเดลบน HuggingFace หรือ path โมเดลโลคัล
- --file                ไฟล์อินพุต (.txt .md .markdown .csv .jsonl/.ndjson)
- --text                ข้อความสั้น ๆ (ซ้ำ flag นี้ได้หลายครั้งถ้าแก้โค้ดรับซ้ำเอง; เวอร์ชันนี้รับครั้งเดียว)
- --context_length      ความยาวหน้าต่างโทเค็นต่อชิ้น (ค่าเริ่มต้น 32000)
- --overlap_ratio       สัดส่วนโทเค็นที่ซ้อนทับระหว่างชิ้น (0..1, ค่าเริ่มต้น 0.25)
- --overlap             จำนวนโทเค็นซ้อนทับแบบกำหนดตรง (ถ้าระบุจะ override overlap_ratio)
- --batch_size          จำนวนชิ้นต่อ batch ตอน forward (เริ่มต้น 4)
- --use_chat_template   ใช้ chat template ของ tokenizer หรือไม่ (true/false/none)
- --md_handling         การจัดการ Markdown: auto|force|off
- --md_strip_code_blocks ตัด code blocks ออกจาก Markdown ก่อนวัด (true/false)
- --file_format         บังคับชนิดไฟล์: auto|text|md|csv|jsonl
- --csv_text_col        ชื่อคอลัมน์ข้อความใน CSV (ถ้าไม่ระบุจะเดาให้)
- --csv_sep             ตัวคั่น CSV (ถ้าไม่ระบุจะเดาให้)
- --csv_encoding        เอนโค้ดดิ้งของ CSV (เริ่มต้น utf-8-sig)
- --jsonl_text_field    ชื่อฟิลด์ข้อความใน JSONL (ถ้าไม่ระบุจะเดาให้)
- --jsonl_encoding      เอนโค้ดดิ้งของ JSONL (เริ่มต้น utf-8)
- --max_rows            จำกัดจำนวนแถวสูงสุดที่อ่านจาก CSV/JSONL
- --skip_empty          ข้ามบรรทัดว่าง (true/false)
- --verbose             โชว์ log รายละเอียด
---

## ตัวอย่างการใช้งาน
1) วัด PPL จากไฟล์เดี่ยว (.txt)
Windows (PowerShell):
```bash
  python .\Perplexity.py `
    --model "C:\path\to\model" `
    --file  "C:\path\to\file.txt" `
    --context_length 32000 `
    --overlap_ratio 0.25 `
    --md_handling auto `
    --md_strip_code_blocks true `
    --verbose
```

Linux/Mac/WSL (Bash):
```bash
  python Perplexity.py \
    --model "/path/to/model" \
    --file  "/path/to/file.txt" \
    --context_length 32000 \
    --overlap_ratio 0.25 \
    --md_handling auto \
    --md_strip_code_blocks true \
    --verbose
```

2) วัด PPL จากข้อความโดยตรง
Windows:
```bash
  python .\Perplexity.py `
    --model "C:\path\to\model" `
    --text "ประเทศไทยมีความหลากหลายทางวัฒนธรรม" `
    --context_length 32000 --overlap_ratio 0.25 --verbose
```

Bash:
```bash
  python Perplexity.py \
    --model "/path/to/model" \
    --text "ประเทศไทยมีความหลากหลายทางวัฒนธรรม" \
    --context_length 32000 --overlap_ratio 0.25 --verbose
```

3) วัด PPL ทั้งโฟลเดอร์ และสรุปลง JSON
Windows:
```bash
  python .\Perplexity.py folder `
    --folder "D:\datasets\my_texts" `
    --model  "C:\path\to\model" `
    --context_length 32000 `
    --overlap_ratio 0.25 `
    --output_json "D:\datasets\my_texts\ppl_summary.json" `
    --md_handling auto --md_strip_code_blocks true
```

Bash:
```bash
  python Perplexity.py folder \
    --folder "/data/my_texts" \
    --model  "/path/to/model" \
    --context_length 32000 \
    --overlap_ratio 0.25 \
    --output_json "/data/my_texts/ppl_summary.json" \
    --md_handling auto --md_strip_code_blocks true
```
---

## ผลลัพธ์และไฟล์สรุป
- โหมดไฟล์เดี่ยว/ข้อความเดี่ยว: จะแสดง PPL_macro, PPL_micro, tokens ใน log
- โหมดโฟลเดอร์: ถ้าระบุ --output_json จะสร้างไฟล์ JSON สรุปเดียวที่มีคีย์ "files" เก็บผลรายไฟล์
  ตัวอย่างส่วน "files" ภายใน ppl_summary.json:
  ```bash
    "files": [
      {
        "file": "a.txt",
        "absolute_path": "D:/datasets/my_texts/a.txt",
        "PPL_micro": 5.01,
        "PPL_macro": 5.08,
        "tokens": 12345,
        "context_length": 32000,
        "overlap_tokens": 256,
        "overlap_ratio": 0.25
      },
      ...
    ]
  ```
---

## เคล็ดลับการตั้งค่า
จากการทดลองด้วยโมเดล gemma-3-270m พบว่า
- context_length: 32,000 เหมาะสมที่สุดภายใต้ขีดความสามารถของโมเดล
(หากเกินขนาดที่รองรับ สคริปต์จะทำการแบ่งอัตโนมัติ)
- overlap_ratio: 0.25
ให้สมดุลที่ดีระหว่างคุณภาพการประเมินและเวลาในการประมวลผล
(ถือเป็นค่าแนะนำสำหรับโมเดลนี้)
- ภาษา: ถ้าโมเดลไม่ได้เทรนบนภาษาไทย ค่า PPL อาจสูงมากเป็นปกติ
---

## คำถามที่พบบ่อย
- **Thai NER ช้า?** แก้ด้วยการไม่ใช้ `--thai-ner-full` หรือรันบน GPU
- **mask ไม่ทับตรงชื่อพอดี?** ตรวจสอบ `--thai-ner-cats` และลองเปิด `--thai-mark-fix` เพื่อให้การแบ่ง token/สระถูกต้องขึ้น
---
