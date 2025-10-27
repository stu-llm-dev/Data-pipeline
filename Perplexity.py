#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perplexity.py — วัด PPL สำหรับโมเดล Causal LM (เช่น Llama, Qwen, Gemma, Mistral)
- รองรับ: ไฟล์ .txt .md .markdown .csv .jsonl/.ndjson และสตริงโดยตรง
- Sliding window + overlap
- ป้องกัน error 512/530 ด้วยการ cap context ต่อชิ้นไม่เกินเพดานโมเดล/โทเค็นไนเซอร์
- ไม่ส่ง token_type_ids เข้าโมเดล
- รองรับโหมดประมวลผลทั้งโฟลเดอร์ + บันทึก JSON สรุปผล
หมายเหตุ:
- ถ้าใช้โมเดล encoder-only (BERT/DeBERTa/Roberta) จะไม่รองรับ PPL แบบ causal — ไฟล์นี้ไม่ได้ทำ pseudo-PPL ให้
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import os
import re
import html
import csv
import json
import math
from tqdm.auto import tqdm
import unicodedata
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any, Iterable

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== optional deps for markdown handling ======
try:
    import markdown as mdlib
    from bs4 import BeautifulSoup
    _HAVE_MD_STACK = True
except Exception:
    _HAVE_MD_STACK = False


# -------------------- Basic Utils --------------------
def is_markdown_path(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in (".md", ".markdown")


def md_to_plain_fallback(s: str, strip_code_blocks: bool = True) -> str:
    """
    Fallback หากไม่มี markdown+bs4:
    - ตัดรั้วโค้ด (```...```) เมื่อ strip_code_blocks=True
    - ตัด inline code `...`
    - ลบ markdown markers พื้นฐาน
    """
    text = s
    if strip_code_blocks:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]+`", "", text)
    else:
        text = re.sub(r"```", "\n", text)

    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # headings
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)             # images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)         # links -> label
    text = re.sub(r">\s?", "", text)                             # blockquote
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)               # bold **
    text = re.sub(r"\*([^*]+)\*", r"\1", text)                   # italic *
    text = re.sub(r"__([^_]+)__", r"\1", text)                   # bold __
    text = re.sub(r"_([^_]+)_", r"\1", text)                     # italic _
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def md_to_plain_external(s: str, strip_code_blocks: bool = True) -> str:
    """ ใช้ markdown → HTML แล้ว BeautifulSoup ดึงข้อความล้วน """
    if not _HAVE_MD_STACK:
        return md_to_plain_fallback(s, strip_code_blocks)

    html_doc = mdlib.markdown(
        s,
        extensions=[
            "extra",
            "fenced_code",
            "sane_lists",
            "codehilite",
            "toc",
            "smarty",
        ],
        output_format="html5",
    )
    soup = BeautifulSoup(html_doc, "html.parser")

    # แทนรูปภาพด้วย alt
    for img in soup.find_all("img"):
        alt = img.get("alt") or ""
        img.replace_with(alt)

    if strip_code_blocks:
        for tag in soup.find_all(["pre", "code"]):
            tag.decompose()
    else:
        for pre in soup.find_all("pre"):
            code_text = pre.get_text("\n")
            pre.replace_with("\n```\n" + code_text + "\n```\n")
        for code in soup.find_all("code"):
            if code.parent and code.parent.name != "pre":
                code.replace_with("`" + code.get_text() + "`")

    text = soup.get_text("\n")
    text = html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_file_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".markdown"):
        return "md"
    if ext == ".csv":
        return "csv"
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    return "text"  # .txt หรืออื่น ๆ


def autodetect_text_field(keys: List[str]) -> Optional[str]:
    candidates = ["ข้อความ", "text", "Text", "message", "content", "utterance"]
    for c in candidates:
        if c in keys:
            return c
    return None


def read_jsonl_texts(
    path: str,
    field: Optional[str] = None,
    max_rows: Optional[int] = None,
    skip_empty: bool = True,
    encoding: str = "utf-8",
) -> Tuple[List[str], Dict[str, Any]]:
    texts: List[str] = []
    num_lines = 0
    used_field = field
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if used_field is None:
                used_field = autodetect_text_field(list(obj.keys()))
            val = obj.get(used_field) if used_field else None
            if val is None:
                # เลือกฟิลด์แรกที่เป็นสตริง
                for k, v in obj.items():
                    if isinstance(v, str):
                        val = v
                        used_field = k
                        break
            if isinstance(val, str):
                s = val.strip()
                if (not skip_empty) or s:
                    texts.append(s)
                    num_lines += 1
            if max_rows is not None and num_lines >= max_rows:
                break
    meta = {"detected_text_field": used_field, "num_rows": num_lines}
    return texts, meta


def read_csv_texts(
    path: str,
    text_col: Optional[str] = None,
    sep: Optional[str] = None,
    encoding: str = "utf-8-sig",
    max_rows: Optional[int] = None,
    skip_empty: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    texts: List[str] = []
    used_sep = sep
    used_col = text_col
    num_rows = 0
    with open(path, "r", encoding=encoding, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if used_sep is None:
            try:
                used_sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                used_sep = ","
        reader = csv.DictReader(f, delimiter=used_sep)
        if reader.fieldnames is None:
            return [], {"detected_text_col": None, "num_rows": 0, "delimiter": used_sep}
        if used_col is None:
            used_col = autodetect_text_field(reader.fieldnames)
        for row in reader:
            val = row.get(used_col) if used_col else None
            if isinstance(val, str):
                s = val.strip()
                if (not skip_empty) or s:
                    texts.append(s)
                    num_rows += 1
            if max_rows is not None and num_rows >= max_rows:
                break
    meta = {"detected_text_col": used_col, "num_rows": num_rows, "delimiter": used_sep}
    return texts, meta


def sanitize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    return unicodedata.normalize("NFKC", s)


def safe_harmonic_mean(values):
    """Harmonic mean แบบปลอดภัย: รับเฉพาะค่าที่เป็นตัวเลข บวก และ finite"""
    vals = [p for p in values if isinstance(p, (int, float)) and p > 0 and math.isfinite(p)]
    if not vals:
        return None
    denom = sum(1.0 / p for p in vals)
    if denom <= 0:
        return None
    return len(vals) / denom


def get_all_special_ids(tokenizer) -> set:
    ids = set()
    for tid in [tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id]:
        if tid is not None:
            ids.add(tid)
    if hasattr(tokenizer, "all_special_ids") and tokenizer.all_special_ids:
        ids.update(tokenizer.all_special_ids)
    return ids


def mask_special_tokens(labels: torch.Tensor, special_ids: set) -> torch.Tensor:
    if not special_ids:
        return labels
    masked = labels.clone()
    for tid in special_ids:
        masked[masked == tid] = -100
    return masked


def supports_chat_template(tokenizer) -> bool:
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None


def batch_iterator(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# -------------------- Sliding-window Encoding --------------------
def encode_with_overlap(
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int,
    overlap_tokens: int,
    use_chat_template: bool,
    eval_role: str = "assistant",
    assistant_user_prompt: Optional[str] = None,
    verbose: bool = False,
    chunk_head: int = 3,
    chunk_tail: int = 1,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    แบ่งข้อความด้วยหน้าต่างความยาว max_length และซ้อนกัน overlap_tokens โทเค็น
    โดยทุกชังก์จะถูกแพดให้ความยาวเท่ากับ max_length เสมอ เพื่อให้ concat เป็น batch ได้ปลอดภัย
    """
    # กันค่า overlap ผิดปกติ
    if overlap_tokens < 0:
        overlap_tokens = 0
    if overlap_tokens >= max_length:
        overlap_tokens = max_length - 1
    step = max(1, max_length - overlap_tokens)

    encoded_per_line: List[List[Dict[str, torch.Tensor]]] = []

    for idx, text in enumerate(texts):
        if len(text.strip()) == 0:
            encoded_per_line.append([])
            continue

        # Apply chat template ถ้าต้องใช้
        if use_chat_template:
            if eval_role == "assistant":
                user_msg = assistant_user_prompt or "โปรดพิมพ์ข้อความต่อไปนี้ซ้ำตามตัวอักษรโดยไม่แก้ไข"
                conv = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": text},
                ]
            else:
                conv = [{"role": "user", "content": text}]
            try:
                rendered = tokenizer.apply_chat_template(
                    conv, add_generation_prompt=False, tokenize=False
                )
            except Exception:
                rendered = text
        else:
            rendered = text

        if verbose:
            logger.info(f"  Text {idx+1}: {len(text)} chars -> {len(rendered)} chars (after template)")

        # เตรียม token ลิสต์ 1 มิติสำหรับสไลด์ด้วยมือ
        token_ids: List[int] = tokenizer.encode(rendered, add_special_tokens=True)
        total_tokens = len(token_ids)
        if verbose:
            logger.info(f"  Total tokens: {total_tokens}")

        chunks: List[Dict[str, torch.Tensor]] = []

        if total_tokens <= max_length:
            # เคสสั้นพอ: เข็มขัดนิรภัยด้วยการ pad ให้เต็ม window เสมอ
            enc = tokenizer(
                rendered,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",          # << สำคัญ: pad ให้เท่ากันทุกชิ้น
                return_token_type_ids=False,
            )
            chunks.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }
            )
            if verbose:
                logger.info(f"  No sliding needed: 1 chunk of {enc['input_ids'].size(1)} tokens")
        else:
            if verbose:
                logger.info("  Using manual sliding window...")

            # id สำหรับแพด
            pad_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
            )

            debug_chunks: List[Tuple[int, int, int]] = []
            start_idx = 0
            while start_idx < total_tokens:
                end_idx = min(start_idx + max_length, total_tokens)
                sub_ids = token_ids[start_idx:end_idx]
                orig_len = len(sub_ids)

                # แพดให้ครบ max_length เสมอ + attention_mask ให้ตรง
                if orig_len < max_length:
                    sub_ids = sub_ids + [pad_id] * (max_length - orig_len)
                    attn = [1] * orig_len + [0] * (max_length - orig_len)
                else:
                    attn = [1] * max_length

                chunk = {
                    "input_ids": torch.tensor([sub_ids], dtype=torch.long),
                    "attention_mask": torch.tensor([attn], dtype=torch.long),
                }
                chunks.append(chunk)
                debug_chunks.append((orig_len, start_idx, end_idx))

                if end_idx >= total_tokens:
                    break
                start_idx += step

            if verbose:
                total_chunks = len(debug_chunks)
                logger.info(f"  Total chunks: {total_chunks}")
                if total_chunks <= (chunk_head + chunk_tail):
                    for i, (tok, st, ed) in enumerate(debug_chunks, 1):
                        logger.info(f"    Chunk {i}: {tok} tokens (pos {st}:{ed})")
                else:
                    for i, (tok, st, ed) in enumerate(debug_chunks[:chunk_head], 1):
                        logger.info(f"    Chunk {i}: {tok} tokens (pos {st}:{ed})")
                    skipped = total_chunks - (chunk_head + chunk_tail)
                    logger.info(f"    ... {skipped} more chunks ...")
                    start_num = total_chunks - chunk_tail + 1
                    for j, (tok, st, ed) in enumerate(debug_chunks[-chunk_tail:], start_num):
                        logger.info(f"    Chunk {j}: {tok} tokens (pos {st}:{ed})")

        encoded_per_line.append(chunks)

    return encoded_per_line


# -------------------- Core: compute PPL per line --------------------
@torch.no_grad()
def compute_ppl_per_line(
    model_name: str,
    texts: List[str],
    batch_size: int = 4,
    max_length: int = 1024,
    overlap_tokens: Optional[int] = None,
    overlap_ratio: Optional[float] = 0.25,
    use_chat_template: Optional[bool] = None,
    eval_role: str = "assistant",
    assistant_user_prompt: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[float], List[int]]:
    """
    คืน: (ppl_list, token_counts_list)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # โหลดโมเดล/โทเค็นไนเซอร์
    is_local_path = os.path.exists(model_name) and os.path.isdir(model_name)
    load_kwargs = {"use_fast": True}
    model_kwargs = {
        "torch_dtype": (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if is_local_path:
        load_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True
        if verbose:
            logger.info(f"Loading local model from: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
    except OSError as e:
        if "couldn't connect" in str(e) or "Network is unreachable" in str(e):
            if verbose:
                logger.info("Network unavailable, trying offline mode...")
            load_kwargs["local_files_only"] = True
            model_kwargs["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
        else:
            raise e

    # pad token กัน error
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # cap ความยาวต่อชิ้นให้ไม่เกินเพดานจริง
    max_pos_model = getattr(model.config, "max_position_embeddings", None)
    if (max_pos_model is None) or (max_pos_model <= 0):
        max_pos_model = tokenizer.model_max_length
    if (tokenizer.model_max_length is None) or (tokenizer.model_max_length <= 0):
        tokenizer_max = 10**9
    else:
        tokenizer_max = tokenizer.model_max_length
    effective_max_length = min(max_length, max_pos_model, tokenizer_max)

    # ตัดสินใจใช้ chat template?
    template_supported = supports_chat_template(tokenizer)
    if use_chat_template is None:
        model_name_lower = model_name.lower()
        likely_instruct = any(
            k in model_name_lower
            for k in ["instruct", "chat", "it", "sft", "alpaca", "vicuna", "llama-2-chat", "llama-3-instruct", "gemma"]
        )
        will_use_template = template_supported and likely_instruct
    else:
        will_use_template = bool(use_chat_template) and template_supported

    if verbose:
        logger.info(f"Model: {model_name}")
        logger.info(f"Chat template supported: {template_supported}")
        logger.info(f"Will use chat template: {will_use_template}")
        logger.info(f"Processing {len(texts)} texts...")

    # overlap tokens
    if overlap_tokens is None:
        r = 0.0 if overlap_ratio is None else float(overlap_ratio)
        r = max(0.0, min(1.0, r))
        overlap_tokens = int(round(effective_max_length * r))
    if overlap_tokens >= effective_max_length:
        overlap_tokens = effective_max_length - 1
    if overlap_tokens < 0:
        overlap_tokens = 0

    special_ids = get_all_special_ids(tokenizer)
    clean_texts = [sanitize_text(t) for t in texts]

    encoded_per_line = encode_with_overlap(
        tokenizer,
        clean_texts,
        effective_max_length,
        overlap_tokens,
        will_use_template,
        eval_role,
        assistant_user_prompt,
        verbose=verbose,
        chunk_head=3,
        chunk_tail=1,
    )

    ppl_list: List[float] = []
    tok_list: List[int] = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    for line_idx, chunks in enumerate(encoded_per_line):
        if not chunks:
            ppl_list.append(float("inf"))
            tok_list.append(0)
            continue

        sum_nll = 0.0
        sum_tokens = 0

        for chunk_batch in batch_iterator(chunks, batch_size):
            # คำนวณความยาวสูงสุดใน batch แล้วแพดให้เท่ากัน (เผื่อกรณี edge)
            batch_max = max(c["input_ids"].size(1) for c in chunk_batch)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
                tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            )

            def _pad_to(t: torch.Tensor, tgt_len: int, value: int):
                pad = tgt_len - t.size(1)
                if pad <= 0:
                    return t
                return torch.nn.functional.pad(t, (0, pad), value=value)

            input_ids = torch.cat(
                [_pad_to(c["input_ids"], batch_max, pad_id) for c in chunk_batch], dim=0
            ).to(device)
            attention_mask = torch.cat(
                [_pad_to(c["attention_mask"], batch_max, 0) for c in chunk_batch], dim=0
            ).to(device)

            labels = input_ids.clone()
            labels = mask_special_tokens(labels, special_ids)

            with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                                dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = ((shift_labels != -100) & (attention_mask[:, 1:] == 1)).to(torch.float32)

            vocab_size = shift_logits.size(-1)
            token_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            token_loss = token_loss.view(shift_labels.size(0), shift_labels.size(1))

            masked_loss = token_loss * shift_mask
            nll_per_seq = masked_loss.sum(dim=1)
            tokens_per_seq = shift_mask.sum(dim=1)

            sum_nll += float(nll_per_seq.sum().item())
            sum_tokens += int(tokens_per_seq.sum().item())

        if sum_tokens > 0:
            ppl = math.exp(sum_nll / sum_tokens)
        else:
            ppl = float("inf")

        ppl_list.append(ppl)
        tok_list.append(sum_tokens)

        if verbose and (line_idx + 1) % 10 == 0:
            logger.info(f"Processed {line_idx + 1}/{len(texts)} texts")

    return ppl_list, tok_list


# -------------------- High-level: compute_ppl --------------------
@torch.no_grad()
def compute_ppl(
    model_name: str,
    text_or_path: Union[str, List[str]],
    context_length: int = 1024,
    overlap_ratio: Optional[float] = 0.25,
    overlap: Optional[int] = None,
    batch_size: int = 4,
    use_chat_template: Optional[bool] = None,
    eval_role: str = "assistant",
    assistant_user_prompt: Optional[str] = None,
    verbose: bool = False,
    is_file: bool = True,
    md_handling: str = "auto",
    md_strip_code_blocks: bool = True,
    file_format: str = "auto",
    csv_text_col: Optional[str] = None,
    csv_sep: Optional[str] = None,
    csv_encoding: str = "utf-8-sig",
    jsonl_text_field: Optional[str] = None,
    jsonl_encoding: str = "utf-8",
    max_rows: Optional[int] = None,
    skip_empty: bool = True,
) -> Dict[str, Any]:
    """
    Main entry — รับไฟล์เดี่ยว/ลิสต์ข้อความ → คืน dict ผลลัพธ์
    """
    # เตรียมข้อความ
    if isinstance(text_or_path, list):
        texts = text_or_path
        if verbose:
            logger.info(f"Processing {len(texts)} texts from list")
    elif is_file:
        if verbose:
            logger.info(f"Reading file: {text_or_path}")
        if not os.path.exists(text_or_path):
            raise FileNotFoundError(f"File not found: {text_or_path}")
        fmt = file_format
        if fmt == "auto":
            fmt = detect_file_format(text_or_path)

        if fmt == "csv":
            texts, meta = read_csv_texts(
                text_or_path,
                text_col=csv_text_col,
                sep=csv_sep,
                encoding=csv_encoding,
                max_rows=max_rows,
                skip_empty=skip_empty,
            )
            if verbose:
                logger.info(f"CSV loaded: rows={meta.get('num_rows', 0)} | col={meta.get('detected_text_col')} | sep={meta.get('delimiter')}")
        elif fmt == "jsonl":
            texts, meta = read_jsonl_texts(
                text_or_path,
                field=jsonl_text_field,
                max_rows=max_rows,
                skip_empty=skip_empty,
                encoding=jsonl_encoding,
            )
            if verbose:
                logger.info(f"JSONL loaded: rows={meta.get('num_rows', 0)} | field={meta.get('detected_text_field')}")
        else:
            with open(text_or_path, "r", encoding="utf-8") as f:
                content = f.read()
            will_md = (md_handling == "force") or (md_handling == "auto" and is_markdown_path(text_or_path))
            if will_md:
                if verbose:
                    logger.info(f"Detected Markdown → converting to plain text using {'markdown+bs4' if _HAVE_MD_STACK else 'regex fallback'} ...")
                    before_len = len(content)
                content = md_to_plain_external(content, strip_code_blocks=md_strip_code_blocks)
                if verbose:
                    logger.info(f"Markdown converted: {before_len} → {len(content)} chars")
            content = content.strip()
            texts = [content] if content else []

        if verbose:
            logger.info(f"Prepared {len(texts)} text(s) for processing")
            if texts and len(texts) == 1:
                logger.info(f"Text length: {len(texts[0])} characters")
            elif texts and len(texts) > 1:
                avg_len = sum(len(t) for t in texts) / len(texts)
                logger.info(f"Average text length: {avg_len:.1f} characters")
    else:
        texts = [str(text_or_path)]
        if verbose:
            logger.info(f"Processing direct text input: {len(texts[0])} characters")

    if not texts:
        return {
            "context_length": context_length,
            "overlap_tokens": 0 if (overlap is None and overlap_ratio is None) else (overlap or int(round(context_length * float(overlap_ratio or 0)))),
            "overlap_ratio": (float(overlap_ratio) if overlap is None and overlap_ratio is not None else ((overlap or 0) / context_length if context_length > 0 else None)),
            "num_texts": 0,
            "PPL_micro": None,
            "PPL_macro": None,
            "tokens_evaluated": 0,
            "used_chat_template": False,
        }

    # คำนวณ overlap ที่ใช้จริง
    if overlap is not None:
        overlap_tokens = int(overlap)
    else:
        r = 0.0 if overlap_ratio is None else float(overlap_ratio)
        r = max(0.0, min(1.0, r))
        overlap_tokens = int(round(context_length * r))

    if overlap_tokens >= context_length:
        overlap_tokens = context_length - 1
    if overlap_tokens < 0:
        overlap_tokens = 0

    if verbose:
        eff_ratio = (overlap_tokens / context_length) if context_length > 0 else 0.0
        step = max(1, context_length - overlap_tokens)
        logger.info(f"Context length: {context_length}, Overlap: {overlap_tokens} ({eff_ratio:.2%}), Step: {step}")

    # คำนวณ PPL
    ppl_list, tok_list = compute_ppl_per_line(
        model_name=model_name,
        texts=texts,
        batch_size=batch_size,
        max_length=context_length,      # จะถูก cap ภายใน compute_ppl_per_line
        overlap_tokens=overlap_tokens,
        overlap_ratio=None,
        use_chat_template=use_chat_template,
        eval_role=eval_role,
        assistant_user_prompt=assistant_user_prompt,
        verbose=verbose,
    )

    if verbose:
        logger.info(f"Token counts per text: {tok_list}")
        logger.info(f"PPL per text: {[f'{p:.3f}' if not math.isinf(p) else 'inf' for p in ppl_list]}")

    valid_ppls = [p for p in ppl_list if not math.isinf(p)]
    total_tokens = sum(tok_list)

    if not valid_ppls:
        ppl_macro = None
        ppl_micro = None
    else:
        ppl_macro = sum(valid_ppls) / len(valid_ppls)
        if total_tokens > 0:
            weighted_sum = sum(p * t for p, t in zip(ppl_list, tok_list) if not math.isinf(p))
            ppl_micro = weighted_sum / total_tokens
        else:
            ppl_micro = ppl_macro

    result = {
        "context_length": context_length,
        "overlap_tokens": overlap_tokens,
        "overlap_ratio": (overlap_tokens / context_length) if context_length > 0 else None,
        "num_texts": len(texts),
        "PPL_micro": ppl_micro,
        "PPL_macro": ppl_macro,
        "tokens_evaluated": total_tokens,
        "used_chat_template": use_chat_template,
    }
    if len(texts) > 1:
        result["per_text_ppls"] = ppl_list
        result["per_text_tokens"] = tok_list
    return result


# -------------------- Folder-level Runner --------------------
def process_folder(
    folder_path: str,
    model_name: str,
    context_length: int = 1024,
    overlap_ratio: float = 0.25,
    output_json: Optional[str] = None,
    file_extensions: Tuple[str, ...] = (".txt", ".md", ".markdown", ".csv", ".jsonl", ".ndjson"),
    verbose: bool = True,
    **compute_ppl_kwargs,
) -> Dict[str, Any]:
    """
    ประมวลผล PPL สำหรับทุกไฟล์ในโฟลเดอร์ด้วย compute_ppl(...)
    """
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Folder not found: {root}")

    files: List[Path] = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in file_extensions
    ]
    if not files:
        if verbose:
            logger.info(f"⚠️ No files with extensions {file_extensions} found in {root}")
        return {
            "folder": str(root),
            "model": model_name,
            "num_files": 0,
            "num_successful": 0,
            "PPL_macro_folder": None,
            "PPL_micro_folder": None,
            "PPL_dataset": None,
            "PPL_dataset_harmonic": None,
            "total_tokens": 0,
            "context_length": context_length,
            "overlap_ratio": overlap_ratio,
            "files": [],
        }

    if verbose:
        logger.info(f"📁 Processing folder: {root}")
        logger.info(f"📄 Found {len(files)} file(s)")
        logger.info("=" * 60)

    file_results: List[Dict[str, Any]] = []
    total_weighted_ppl = 0.0
    total_tokens = 0
    valid_macro_ppls: List[float] = []

    bar = tqdm(
        sorted(files),
        desc="Processing files",
        total=len(files),
        unit="file",
        dynamic_ncols=True,
        disable=verbose  # ถ้าอยากให้โชว์ bar แม้ verbose=True ให้เปลี่ยนเป็น False
    )

    for i, file_path in enumerate(bar, 1):
        rel = file_path.relative_to(root)
        if verbose:
            logger.info(f"\n[{i}/{len(files)}] Processing: {rel}")
            logger.info("-" * 60)

        try:
            result = compute_ppl(
                model_name=model_name,
                text_or_path=str(file_path),
                context_length=context_length,
                overlap_ratio=overlap_ratio,
                is_file=True,
                verbose=verbose,
                **compute_ppl_kwargs,
            )

            ppl_micro = result.get("PPL_micro")
            ppl_macro = result.get("PPL_macro")
            tokens = int(result.get("tokens_evaluated", 0))

            file_results.append({
                "file": str(rel),
                "absolute_path": str(file_path),
                "PPL_micro": ppl_micro,
                "PPL_macro": ppl_macro,
                "tokens": tokens,
                "context_length": result.get("context_length"),
                "overlap_tokens": result.get("overlap_tokens"),
                "overlap_ratio": result.get("overlap_ratio"),
            })

            if isinstance(ppl_micro, (int, float)) and math.isfinite(ppl_micro) and tokens > 0:
                total_weighted_ppl += ppl_micro * tokens
                total_tokens += tokens

            if isinstance(ppl_macro, (int, float)) and math.isfinite(ppl_macro):
                valid_macro_ppls.append(ppl_macro)

            # อัปเดตข้อความด้านขวาของ progress bar ให้เห็นสรุปไฟล์ล่าสุด
            if not verbose:
                bar.set_postfix({
                    "PPL_micro": (f"{ppl_micro:.4f}" if isinstance(ppl_micro, (int, float)) else "N/A"),
                    "Tokens": tokens
                })

            if verbose:
                logger.info(f"✓ PPL_micro: {ppl_micro:.4f}" if isinstance(ppl_micro, (int, float)) else "✓ PPL_micro: N/A")
                logger.info(f"✓ PPL_macro: {ppl_macro:.4f}" if isinstance(ppl_macro, (int, float)) else "✓ PPL_macro: N/A")
                logger.info(f"✓ Tokens: {tokens}")

        except Exception as e:
            if verbose:
                logger.error(f"❌ Error processing {rel}: {e}")
            file_results.append({
                "file": str(rel),
                "absolute_path": str(file_path),
                "error": str(e),
                "PPL_micro": None,
                "PPL_macro": None,
                "tokens": 0,
            })
            if not verbose:
                bar.set_postfix({"error": str(e)[:40] + "..."})

    ppl_micro_folder = (total_weighted_ppl / total_tokens) if total_tokens > 0 else None
    ppl_macro_folder = (sum(valid_macro_ppls) / len(valid_macro_ppls)) if valid_macro_ppls else None
    ppl_dataset_weighted = ppl_micro_folder
    harmonic_mean = safe_harmonic_mean(valid_macro_ppls)

    summary: Dict[str, Any] = {
        "folder": str(root),
        "model": model_name,
        "num_files": len(files),
        "num_successful": len([r for r in file_results if "error" not in r]),
        "PPL_macro_folder": ppl_macro_folder,
        "PPL_micro_folder": ppl_micro_folder,
        "PPL_dataset": ppl_dataset_weighted,
        "PPL_dataset_harmonic": harmonic_mean,
        "total_tokens": total_tokens,
        "context_length": context_length,
        "overlap_ratio": overlap_ratio,
        "files": file_results,
    }

    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info("📊 FOLDER SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {summary['num_files']}")
        logger.info(f"Successful: {summary['num_successful']}")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info("\n📁 Folder-level metrics:")
        logger.info(f"  PPL_macro (folder): {ppl_macro_folder:.4f}" if isinstance(ppl_macro_folder, (int, float)) else "  PPL_macro (folder): N/A")
        logger.info(f"  PPL_micro (folder): {ppl_micro_folder:.4f}" if isinstance(ppl_micro_folder, (int, float)) else "  PPL_micro (folder): N/A")
        logger.info("\n🎯 Dataset-level PPL (recommended):")
        logger.info(f"  PPL_dataset (weighted): {ppl_dataset_weighted:.4f}" if isinstance(ppl_dataset_weighted, (int, float)) else "  PPL_dataset (weighted): N/A")
        logger.info(f"  PPL_dataset (harmonic): {harmonic_mean:.4f}" if isinstance(harmonic_mean, (int, float)) else "  PPL_dataset (harmonic): N/A")

    if output_json:
        outp = Path(output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if verbose:
            logger.info(f"\n💾 Results saved to: {outp}")

    return summary


# -------------------- CLI Entry --------------------
def _str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    x = x.lower()
    if x in ("none", "null"):
        return None
    return x == "true"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute Perplexity for text/markdown/csv/jsonl files or entire folder.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # single file / direct text mode (default)
    parser.add_argument("--model", type=str, help="HF model name or local model path")
    parser.add_argument("--text", type=str, help="Direct text input (if not using --file)")
    parser.add_argument("--file", type=str, help="Path to a single file")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--overlap_ratio", type=float, default=0.25)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_chat_template", type=str, default=None, choices=["true", "false", "none"])
    parser.add_argument("--eval_role", type=str, default="assistant", choices=["assistant", "user"])
    parser.add_argument("--assistant_user_prompt", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--md_handling", type=str, default="auto", choices=["auto", "force", "off"])
    parser.add_argument("--md_strip_code_blocks", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--file_format", type=str, default="auto", choices=["auto", "text", "md", "csv", "jsonl"])
    parser.add_argument("--csv_text_col", type=str, default=None)
    parser.add_argument("--csv_sep", type=str, default=None)
    parser.add_argument("--csv_encoding", type=str, default="utf-8-sig")
    parser.add_argument("--jsonl_text_field", type=str, default=None)
    parser.add_argument("--jsonl_encoding", type=str, default="utf-8")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--skip_empty", type=str, default="true", choices=["true", "false"])

    # folder mode
    p_folder = sub.add_parser("folder", help="Process an entire folder")
    p_folder.add_argument("--folder", required=True, help="Path to folder")
    p_folder.add_argument("--model", required=True, help="HF model name or local model path")
    p_folder.add_argument("--context_length", type=int, default=1024)
    p_folder.add_argument("--overlap_ratio", type=float, default=0.25)
    p_folder.add_argument("--output_json", type=str, default=None)
    p_folder.add_argument("--use_chat_template", type=str, default=None, choices=["true", "false", "none"])
    p_folder.add_argument("--md_handling", type=str, default="auto", choices=["auto", "force", "off"])
    p_folder.add_argument("--md_strip_code_blocks", type=str, default="true", choices=["true", "false"])
    p_folder.add_argument("--batch_size", type=int, default=4)
    p_folder.add_argument("--csv_text_col", type=str, default=None)
    p_folder.add_argument("--csv_sep", type=str, default=None)
    p_folder.add_argument("--csv_encoding", type=str, default="utf-8-sig")
    p_folder.add_argument("--jsonl_text_field", type=str, default=None)
    p_folder.add_argument("--jsonl_encoding", type=str, default="utf-8")
    p_folder.add_argument("--max_rows", type=int, default=None)
    p_folder.add_argument("--skip_empty", type=str, default="true", choices=["true", "false"])

    args = parser.parse_args()

    if args.cmd == "folder":
        results = process_folder(
            folder_path=args.folder,
            model_name=args.model,
            context_length=args.context_length,
            overlap_ratio=args.overlap_ratio,
            output_json=args.output_json,
            file_extensions=(".txt", ".md", ".markdown", ".csv", ".jsonl", ".ndjson"),
            verbose=True,
            use_chat_template=_str2bool(args.use_chat_template),
            md_handling=args.md_handling,
            md_strip_code_blocks=_str2bool(args.md_strip_code_blocks) if args.md_strip_code_blocks is not None else True,
            batch_size=args.batch_size,
            csv_text_col=args.csv_text_col,
            csv_sep=args.csv_sep,
            csv_encoding=args.csv_encoding,
            jsonl_text_field=args.jsonl_text_field,
            jsonl_encoding=args.jsonl_encoding,
            max_rows=args.max_rows,
            skip_empty=_str2bool(args.skip_empty),
        )
        ds = results.get("PPL_dataset")
        logger.info(f"\n✅ Done! PPL_dataset = {ds:.4f}" if isinstance(ds, (int, float)) else "\n✅ Done! PPL_dataset = N/A")
    else:
        # single shot
        if not args.model:
            raise SystemExit("❌ Please provide --model (HF name or local path)")
        if not args.text and not args.file:
            raise SystemExit("❌ Please provide either --text or --file")

        if args.text:
            input_payload: Union[str, List[str]] = args.text
            is_file = False
        else:
            input_payload = args.file
            is_file = True

        res = compute_ppl(
            model_name=args.model,
            text_or_path=input_payload,
            context_length=args.context_length,
            overlap_ratio=args.overlap_ratio,
            overlap=args.overlap,
            batch_size=args.batch_size,
            use_chat_template=_str2bool(args.use_chat_template),
            eval_role=args.eval_role,
            assistant_user_prompt=args.assistant_user_prompt,
            verbose=args.verbose,
            is_file=is_file,
            md_handling=args.md_handling,
            md_strip_code_blocks=_str2bool(args.md_strip_code_blocks) if args.md_strip_code_blocks is not None else True,
            file_format=args.file_format,
            csv_text_col=args.csv_text_col,
            csv_sep=args.csv_sep,
            csv_encoding=args.csv_encoding,
            jsonl_text_field=args.jsonl_text_field,
            jsonl_encoding=args.jsonl_encoding,
            max_rows=args.max_rows,
            skip_empty=_str2bool(args.skip_empty),
        )
        pm = res.get("PPL_macro")
        pmicro = res.get("PPL_micro")
        logger.info("\n=== RESULT ===")
        logger.info(f"PPL_macro: {pm:.4f}" if isinstance(pm, (int, float)) else "PPL_macro: N/A")
        logger.info(f"PPL_micro: {pmicro:.4f}" if isinstance(pmicro, (int, float)) else "PPL_micro: N/A")
        logger.info(f"Tokens evaluated: {res.get('tokens_evaluated', 0)}")