#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Clean (MODEL, augmented)
=================================
Drop‑in pipeline for Thai text cleaning with **PDPA + Thai NER** + **checkpoint/resume**
+ **Mixed‑script detection** + **Thai mark fixer (ุ/ู, spacing, combining)** + **basic sanitizers**.

Base: Pipeline_clean_model.py (your file) — extended to include the requested features.
"""
from __future__ import annotations
import os, sys, io, json, re, argparse, logging, unicodedata, multiprocessing as mp, pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # reduce TF CUDA noise

# ========= Optional fastText =========
_FT_MODEL = None
_TOK = None
# ปรับพาธให้ตรงของคุณ หรืออ่านจาก ENV
_TOKENIZER_DIR = os.environ.get(
    "TY_TOKENIZER_DIR",
    "/model/tokenizermodel"
)
THRESH_POS = float(os.environ.get("FT_POS_THRESHOLD", "0.95"))
_PRED_POS = "__label__1"
_PRED_NEG = "__label__0"
public_figure_skip: bool = False

def _load_fasttext(model_path: str):
    global _FT_MODEL
    if not model_path:
        return None
    try:
        import fasttext  # type: ignore
        _FT_MODEL = fasttext.load_model(model_path)
        logging.info(f"[fastText] loaded: {model_path}")
    except Exception as e:
        logging.warning(f"[fastText] disabled: {e}")
        _FT_MODEL = None
    return _FT_MODEL

def _load_tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(_TOKENIZER_DIR, trust_remote_code=True, local_files_only=True)
    return _TOK

def _tokenize_like_llm(text: str, space_replacement: str = "▁") -> str:
    if not text:
        return ""
    tok = _load_tokenizer()
    ids = tok.encode(str(text), add_special_tokens=False)
    pieces = [tok.convert_ids_to_tokens(i) for i in ids]
    pieces = [p.replace(" ", space_replacement) for p in pieces]
    return " ".join(pieces)

def _strip_bpe_space_prefix(s: str) -> str:
    # "▁เว็บ ▁พนันฟุตบอล" -> "เว็บ พนันฟุตบอล"
    return s.replace("▁", "")

# ========= f_text_cleaner integration =========
_F_CLEAN_AVAILABLE = False
_THAI_NER_AVAILABLE = False
try:
    from src.f_text_cleaner import (
        remove_html_tags, remove_bbcode_tags, remove_emoticons,
        remove_emails, remove_web_links, remove_phone_numbers,
        remove_social_media_mentions,
        clean_name, clean_phone, clean_national_id, clean_email,
        anonymize_credit_cards, anonymize_bank_accounts,
        thai_ner_full_apply, _load_thai_ner, address_mask, address_anonymize,
        reset_romanize_cache, configure_public_figure_paths
    )
    _F_CLEAN_AVAILABLE = True
    _THAI_NER_AVAILABLE = True
except Exception as e:
    logging.warning(f"[import] f_text_cleaner not available: {e}")
    _F_CLEAN_AVAILABLE = False
    _THAI_NER_AVAILABLE = False

# ========= Clean step registry =========
_CLEAN_STEP_FUNCS = {
    "html":     (lambda s: remove_html_tags(s))                if _F_CLEAN_AVAILABLE else (lambda s: s),
    "bbcode":   (lambda s: remove_bbcode_tags(s))              if _F_CLEAN_AVAILABLE else (lambda s: s),
    "emoticon": (lambda s: remove_emoticons(s))                if _F_CLEAN_AVAILABLE else (lambda s: s),
    "links":    (lambda s: remove_web_links(s))                if _F_CLEAN_AVAILABLE else (lambda s: s),
    "social":   (lambda s: remove_social_media_mentions(s))    if _F_CLEAN_AVAILABLE else (lambda s: s),
}

_FTEXT_STEP_FUNCS = {
    # backward-compat names
    "remove_html":      _CLEAN_STEP_FUNCS["html"],
    "remove_bbcode":    _CLEAN_STEP_FUNCS["bbcode"],
    "remove_emoticons": _CLEAN_STEP_FUNCS["emoticon"],
    "remove_emails":    (lambda s: remove_emails(s)) if _F_CLEAN_AVAILABLE else (lambda s: s),
    "remove_links":     _CLEAN_STEP_FUNCS["links"],
    "remove_phones":    (lambda s: remove_phone_numbers(s)) if _F_CLEAN_AVAILABLE else (lambda s: s),
    "remove_social":    _CLEAN_STEP_FUNCS["social"],
}

# ========= Text key heuristics =========
_DEFAULT_TEXT_KEYS = [
    "text","context","content","contents","document","body","article",
    "passage","paragraph","raw_text","rawtext","source_text",
    "ocr_text","text_clean","segment_text","content_thai"
]
_INSTR_KEYS = ["instruction","input","question","prompt"]
_OUT_KEYS   = ["output","response","answer","completion"]

_ZW = "\u200b\u200c\u200d\ufeff\u2060"
_CTRL_SAFE_PATTERN = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]")

# ========= Mixed-script detection (Thai ↔ LATN/Hira/Kata/Han adjacent) =========
_RE_THAI = r"\u0E00-\u0E7F"
_RE_LATN = r"A-Za-z"
_RE_HIRA = r"\u3040-\u309F"
_RE_KATA = r"\u30A0-\u30FF\u31F0-\u31FF"
_RE_HAN  = r"\u4E00-\u9FFF"
_MIXED_ADJACENT = re.compile(
    fr"([{_RE_THAI}])([{_RE_LATN}{_RE_HIRA}{_RE_KATA}{_RE_HAN}])|"
    fr"([{_RE_LATN}{_RE_HIRA}{_RE_KATA}{_RE_HAN}])([{_RE_THAI}])"
)

def has_mixed_script_adjacent(s: str) -> bool:
    return bool(_MIXED_ADJACENT.search(s or ""))

# ========= Thai floating below-vowels (ุ/ู) detector & fixer + mark normalization =========
_ZW_CHARS = ["\u200b","\u200c","\u200d","\ufeff","\u2060"]
TH_FLOATING_BELOW = {"\u0E38", "\u0E39"}  # ุ, ู

def _is_thai_base_char(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (0x0E00 <= cp <= 0x0E7F) and unicodedata.category(ch).startswith("L")

def _prev_nonspace_index(s: str, i: int) -> int:
    j = i - 1
    while j >= 0 and (s[j].isspace() or s[j] in _ZW_CHARS):
        j -= 1
    return j

def detect_floating_below(text: str):
    spans = []
    if not text:
        return spans
    for i, ch in enumerate(text):
        if ch not in TH_FLOATING_BELOW:
            continue
        j = _prev_nonspace_index(text, i)
        if j < 0:
            spans.append({"start": i, "end": i+1, "char": ch, "reason": "start_of_string"}); continue
        separated = any(c.isspace() or c in _ZW_CHARS for c in text[j+1:i])
        prev_base = _is_thai_base_char(text[j])
        if (not prev_base) or separated:
            spans.append({"start": i, "end": i+1, "char": ch, "reason": "no_base_or_separated"})
    return spans

def fix_floating_below(text: str, policy: str = "drop"):
    spans = detect_floating_below(text)
    if not spans or policy == "keep":
        return text, spans
    floating_idx = {sp["start"] for sp in spans}
    if policy == "drop":
        new_text = "".join(ch for idx, ch in enumerate(text) if idx not in floating_idx)
        return new_text, spans
    if policy == "mask":
        out = []
        for idx, ch in enumerate(text):
            out.append("<fv>" if idx in floating_idx else ch)
        return "".join(out), spans
    return text, spans

# Combining mark normalization (collapse duplicates per cluster)
_TH_BASE = r"[ก-ฮ]"
_TH_COMBINING = (
    "\u0E31"                      # ั
    "\u0E34\u0E35\u0E36\u0E37"    # ิ ี ึ ื
    "\u0E38\u0E39"                # ุ ู
    "\u0E3A"                      # ฺ
    "\u0E47"                      # ็
    "\u0E48\u0E49\u0E4A\u0E4B"    # ่ ้ ๊ ๋
    "\u0E4C\u0E4D\u0E4E"          # ์ ํ ๎
)
_TH_COMB_SET = set(_TH_COMBINING)
_GROUPS = {
    "upper": set("\u0E34\u0E35\u0E36\u0E37\u0E31"),
    "lower": set("\u0E38\u0E39"),
    "tone":  set("\u0E48\u0E49\u0E4A\u0E4B"),
    "taikhu": set("\u0E47"),
    "nikhahit": set("\u0E4D"),
    "thanthakhat": set("\u0E4C"),
    "yamakkan": set("\u0E4E"),
    "phinthu": set("\u0E3A"),
}

def _collapse_dup_groups(cluster: str) -> str:
    if not cluster:
        return cluster
    base = cluster[0]
    keep, seen = [], {k: False for k in _GROUPS}
    for ch in cluster[1:]:
        placed = False
        for g, s in _GROUPS.items():
            if ch in s:
                if not seen[g]:
                    keep.append(ch); seen[g] = True
                placed = True; break
        if not placed:
            if ch in _TH_COMB_SET and ch not in keep:
                keep.append(ch)
    return base + "".join(keep)

def _fix_clusters_once(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if re.match(_TH_BASE, ch):
            j = i + 1
            while j < len(s) and s[j] in _TH_COMB_SET:
                j += 1
            out.append(_collapse_dup_groups(s[i:j])); i = j
        else:
            out.append(ch); i += 1
    return "".join(out)

# ========= Helpers =========

def _parse_csv_list(csv_str: Optional[str]) -> List[str]:
    if not csv_str:
        return []
    return [x.strip() for x in csv_str.split(',') if x.strip()]

def _sanitize_str(s: str) -> str:
    # remove private-use and control
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cs")
    s = _CTRL_SAFE_PATTERN.sub("", s)
    # strip zero-widths early
    s = re.sub(r"[\u200B\u200C\u200D\u2060\ufeff]", "", s)
    return s

def _coerce_text_to_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x
    elif isinstance(x, (list, tuple)):
        s = " ".join(e if isinstance(e, str) else str(e) for e in x)
    elif isinstance(x, dict):
        vals = [v for v in x.values() if isinstance(v, str)]
        s = " ".join(vals) if vals else json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)
    s = s.strip()
    return s if s else None

def extract_text_like(obj: dict, text_keys: Optional[List[str]]=None, combine_keys: Optional[List[str]]=None) -> Tuple[Optional[str], Optional[str]]:
    # (0) combine forced
    if combine_keys:
        parts, used = [], []
        for k in combine_keys:
            if k in obj and obj[k] not in (None, ""):
                s = _coerce_text_to_str(obj[k])
                if s:
                    parts.append(s); used.append(k)
        if parts:
            return "\n\n".join(parts), f"+combine({','.join(used)})"
    # (1) standard keys
    keys = text_keys or (_DEFAULT_TEXT_KEYS + _INSTR_KEYS + _OUT_KEYS)
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            s = _coerce_text_to_str(obj[k])
            if s:
                return s, k
    # (2) dialogue-like
    for k in ("messages","conversations","dialogue"):
        v = obj.get(k)
        if isinstance(v, list):
            parts = []
            for m in v:
                vv = None
                if isinstance(m, dict):
                    vv = m.get("content") or m.get("text") or m.get("value")
                else:
                    vv = m
                vv = _coerce_text_to_str(vv)
                if vv:
                    parts.append(vv)
            if parts:
                return "\n\n".join(parts), k
    # (3) nested
    for container in ("data","payload","meta","sample"):
        vv = obj.get(container)
        if isinstance(vv, dict):
            for k in keys:
                if k in vv and vv[k] not in (None,""):
                    s = _coerce_text_to_str(vv[k])
                    if s:
                        return s, f"{container}.{k}"
    # (4) fallback
    if isinstance(obj, dict):
        vals = []
        for k in ("title","abstract","summary","body"):
            if k in obj:
                s = _coerce_text_to_str(obj[k])
                if s:
                    vals.append(s)
        if vals:
            return "\n\n".join(vals), "+fallback"
    return None, None

# ========= PDPA passes =========

def apply_pdpa(text: str, policy: str, steps: List[str], *,
               keep_last: int=4, card_mode: str="randomize", account_mode: str="randomize",
               salt: str="", addr_gazetteer: Optional[str] = None,
               addr_keep_province: bool = True, addr_tag: str = "[ADDRESS]",) -> str:
    if not _F_CLEAN_AVAILABLE:
        return text
    st = set(x.lower() for x in steps)

    if policy == "mask":
        if "name"  in st: text = clean_name(text)
        if "email" in st: text = remove_emails(text)
        if "phone" in st: text = remove_phone_numbers(text)
        if "links" in st: text = remove_web_links(text)
        if "card"  in st: text = anonymize_credit_cards(text, mode="mask", keep_last=keep_last)
        if "account" in st: text = anonymize_bank_accounts(text, mode="mask", keep_last=keep_last)
        if "address" in st: text = address_mask(text, gazetteer_path=addr_gazetteer, tag=addr_tag)
        return text

    if policy == "anonymize":
        if "name"  in st: text = clean_name(text)
        if "phone" in st: text = clean_phone(text)
        if "email" in st: text = clean_email(text)
        if "id_card" in st: text = clean_national_id(text, salt)
        if "card"   in st:
            text = anonymize_credit_cards(text, mode=card_mode, keep_last=keep_last, salt=salt)
        if "account" in st:
            text = anonymize_bank_accounts(text, mode=account_mode, keep_last=keep_last, salt=salt)
        if "address" in st:
            text = address_anonymize(text, gazetteer_path=addr_gazetteer, salt=salt, keep_province=addr_keep_province)
        return text

    # === NEW: โหมดลบทิ้ง (drop) — reuse ฟังก์ชันเดิม, ไม่ใช้ regex ใหม่ ===
    if policy == "drop":
        # 1) ตัวลบตรง ๆ ที่มีอยู่เดิม
        if "email" in st:  text = remove_emails(text)
        if "phone" in st:  text = remove_phone_numbers(text)
        if "links" in st:  text = remove_web_links(text)

        # 2) บัตรประชาชน (Thai National ID) — ลบทิ้งด้วย filter_national_id(token="")
        if "id_card" in st:
            try:
                text = filter_national_id(text, token="")
            except Exception:
                pass  # ถ้า import/func ไม่พร้อม ให้ข้ามไป (ไม่พัง pipeline)

        # 3) ที่อยู่ — ใช้ address_mask แต่ตั้ง tag="" = ลบทิ้ง
        if "address" in st:
            try:
                text = address_mask(text, gazetteer_path=addr_gazetteer, tag="")
            except Exception:
                pass

        # 4) บัตรเครดิต / บัญชีธนาคาร — ให้ anonymize เป็น "mask" แล้วลบแท็กที่มาส์กออก
        if "card" in st:
            try:
                tmp = anonymize_credit_cards(text, mode="mask", keep_last=0, salt=salt)
                # ลบ token ที่ถูกมาส์ก (รองรับชื่อแท็กแพร่หลาย)
                tmp = re.sub(r"\[(?:CREDIT_)?CARD\]|<(?:CREDIT_)?CARD>", "", tmp)
                text = tmp
            except Exception:
                pass
        if "account" in st:
            try:
                tmp = anonymize_bank_accounts(text, mode="mask", keep_last=0, salt=salt)
                tmp = re.sub(r"\[(?:BANK_)?ACCOUNT\]|<(?:BANK_)?ACCOUNT>", "", tmp)
                text = tmp
            except Exception:
                pass

        # 5) ชื่อบุคคล — ถ้ามี Thai NER ให้มาส์กเป็นค่าว่าง (token_map) เพื่อให้เท่ากับ drop
        try:
            if "name" in st and _THAI_NER_AVAILABLE:
                text = thai_ner_full_apply(
                    text,
                    policy="mask",
                    keep_categories="PERSON",
                    token_map={"PERSON": ""},
                    salt=salt or "",
                )
        except Exception:
            pass

        return text
    return text


# ========= NER pass =========

def apply_thai_ner_full(text: str, *, enabled: bool, policy: str,
                        cats: List[str], token_map_path: Optional[str], salt: str) -> str:
    if not enabled:
        return text
    if not _THAI_NER_AVAILABLE:
        logging.warning("[thai-ner] requested but f_text_cleaner is unavailable; skipping")
        return text

    tmap = None
    if token_map_path:
        try:
            with io.open(token_map_path, "r", encoding="utf-8") as f:
                tmap = json.load(f)
        except Exception as e:
            logging.warning(f"[thai-ner] token-map ignored: {e}")

    keep = ",".join(cats) if cats else "PERSON,PHONE,EMAIL"

    try:
        if policy == "drop":
            # ‘drop’ = ใช้กลไก mask แต่แทน token เป็นค่าว่างผ่าน token_map
            # ถ้า user มี token_map เอง จะ merge ตรงนี้ได้ (ให้ค่าว่างเฉพาะ cat ที่ต้องการ)
            # ดีฟอลต์: ค่าว่างกับทุก cat ที่เลือก
            default_token_map = {c.strip().upper(): "" for c in keep.split(",") if c.strip()}
            if tmap:
                default_token_map.update({k.upper(): v for k, v in tmap.items()})
            return thai_ner_full_apply(
                text,
                policy="mask",
                keep_categories=keep,
                token_map=default_token_map,
                salt=salt or "",
                public_figure_skip=public_figure_skip,
            )
        else:
            # policy == "mask" | "anonymize" ทำงานเดิม
            return thai_ner_full_apply(
                text,
                policy=policy,
                keep_categories=keep,
                token_map=tmap,
                salt=salt or "",
                public_figure_skip=public_figure_skip,
            )
    except Exception as e:
        logging.warning(f"[thai-ner] failed; keeping text: {e}")
        return text


# ========= Classifier routing (optional) =========

def _bpe_tokenize(text: str) -> str:
    bpe = _tokenize_like_llm((text or "").replace("\n", " "))
    return _strip_bpe_space_prefix(bpe)

def _chunk_words(s: str, chunk_size=200, overlap=100):
    # สไลซ์เป็น "โทเคนเวิร์ด" ที่คั่นด้วย space (เพราะเราตัดเป็น BPE แล้ว)
    ws = s.split()
    if not ws:
        return []
    out = []
    i = 0
    while i < len(ws):
        j = min(i + chunk_size, len(ws))
        out.append(" ".join(ws[i:j]))
        if j >= len(ws):
            break
        i += max(1, chunk_size - overlap)
    return out


def route_legal_illegal(text: str, *, enable_ft: bool,
                        chunk_size=200, overlap=100, agg="maxpos") -> str:
    """
    ตัดสินผลแบบเข้ม: จะเป็น 'illegal' ได้ก็ต่อเมื่อ
    สกอร์ฝั่งบวก (pos prob) รวมระดับเอกสาร >= THRESH_POS
    otherwise => 'legal'

    agg:
      - 'maxpos'  : ใช้ max ของ prob ฝั่งบวกในทุก chunk (แนะนำ)
      - 'meanpos' : ใช้ค่าเฉลี่ยของ prob ฝั่งบวก
      - 'majority': เข้ากันได้กับของเดิม แต่จะ 'illegal' ต่อเมื่อ
                    ทั้งได้เสียงข้างมากเป็น __label__1 และ
                    maxpos >= THRESH_POS
    """
    if not enable_ft or _FT_MODEL is None:
        return "legal"
    try:
        # 1) tokenize BPE (คุณเพิ่งเพิ่มการ strip ▁ ใน _bpe_tokenize แล้ว)
        t_tok = _bpe_tokenize(text)

        # 2) chunk ตามโปรไฟล์
        chunks = _chunk_words(t_tok, chunk_size, overlap) or [t_tok[:2000]]

        # 3) เอา prob ของ 'ฝั่งบวก' ด้วย k=2 เพื่อเข้าถึง prob ทั้งสองคลาส
        labels, probs = _FT_MODEL.predict(chunks, k=2)  # batch predict (k=2)

        # 4) ทำเป็นรายการ prob ฝั่งบวก (__label__1)
        pos_scores = []
        maj_labels = []  # ไว้ใช้กรณี agg='majority'
        for lb_row, pb_row in zip(labels, probs):
            # แปลงเป็น dict {label: prob}
            row = {}
            for l, p in zip(lb_row, pb_row):
                lab = l.decode() if isinstance(l, bytes) else l
                row[lab] = float(p)
            # เก็บ prob ฝั่งบวก
            pos_scores.append(row.get(_PRED_POS, 0.0))
            # เก็บ top-1 label ไว้สำหรับ majority
            if lb_row:
                top1 = lb_row[0].decode() if isinstance(lb_row[0], bytes) else lb_row[0]
                maj_labels.append(top1)
            else:
                maj_labels.append(_PRED_NEG)

        # 5) รวมเป็นคะแนนระดับเอกสาร
        if not pos_scores:
            doc_pos = 0.0
        elif agg == "meanpos":
            doc_pos = float(sum(pos_scores) / len(pos_scores))
        else:
            # ค่าเริ่มต้น: ใช้ max ของ prob ฝั่งบวก (เหมาะกับเอกสารยาว)
            doc_pos = max(pos_scores)

        # 6) เกณฑ์ตัดสินเข้ม: ต้องมั่นใจ >= THRESH_POS ถึงจะ 'illegal'
        if agg == "majority":
            # ของเดิมคุณใช้ majority → ให้คงเงื่อนไข majority + threshold
            from collections import Counter
            top_label, _ = Counter(maj_labels).most_common(1)[0]
            if top_label == _PRED_POS and doc_pos >= THRESH_POS:
                return "illegal"
            return "legal"
        else:
            # maxpos / meanpos: ใช้ threshold ตรง ๆ
            return "illegal" if doc_pos >= THRESH_POS else "legal"

    except Exception as e:
        logging.warning(f"[fastText] routing failed: {e}")
        return "legal"

# ========= Options + processing =========
@dataclass
class Options:
    clean_steps: List[str]
    pdpa_policy: str
    pdpa_steps: List[str]
    keep_last: int
    card_mode: str
    account_mode: str
    salt: str
    thai_ner_full: bool
    thai_ner_policy: str
    thai_ner_cats: List[str]
    thai_ner_map: Optional[str]
    enable_fasttext: bool
    text_keys: Optional[List[str]]
    combine_keys: Optional[List[str]]
    debug_first_n: int
    debug_show_diff: bool
    # new features
    detect_mixed_script: bool
    mixed_policy: str  # skip|tag|reroute
    thai_mark_fix: bool
    floating_policy: str  # drop|mask|keep
    pull_back_gap: str    # any|tight
    normalize: str        # NFC|NFKC|none
    flags_key: Optional[str]
    addr_gazetteer: Optional[str] = None
    addr_keep_province: bool = True
    addr_tag: str = "[ADDRESS]"
    public_figure_skip: bool = True


def process_text(text: str, opt: Options, flags: List[str]) -> str:
    s0 = text or ""
    s0 = _sanitize_str(s0)

    # 1) Sanitizers
    for st in opt.clean_steps:
        fn = _CLEAN_STEP_FUNCS.get(st) or _FTEXT_STEP_FUNCS.get(st)
        if fn:
            try:
                s0 = fn(s0)
            except Exception as e:
                logging.warning(f"[clean-step:{st}] {e}")

    # 2) PDPA
    s0 = apply_pdpa(
        s0,
        policy=opt.pdpa_policy,
        steps=opt.pdpa_steps,
        keep_last=opt.keep_last,
        card_mode=opt.card_mode,
        account_mode=opt.account_mode,
        salt=opt.salt,
        addr_gazetteer=opt.addr_gazetteer,
        addr_keep_province=opt.addr_keep_province,
        addr_tag=opt.addr_tag,
    )

    # 3) Thai NER (optional)
    s0 = apply_thai_ner_full(
        s0,
        enabled=opt.thai_ner_full,
        policy=opt.thai_ner_policy,
        cats=opt.thai_ner_cats,
        token_map_path=opt.thai_ner_map,
        salt=opt.salt,
    )

    # 4) Thai mark fixer / floating below vowels
    if opt.thai_mark_fix:
        s1, spans = fix_floating_below(s0, policy=opt.floating_policy)
        if spans:
            flags.append("floating_below_detected")
        s1 = _fix_clusters_once(s1)
        if opt.normalize and opt.normalize.upper() in ("NFC","NFKC"):
            s1 = unicodedata.normalize(opt.normalize.upper(), s1)
        s0 = s1

    return s0

# ========= Parent/worker preload =========

def parent_preload(opt: Options, model_path: Optional[str]):
    if opt.thai_ner_full and _THAI_NER_AVAILABLE:
        try:
            logging.info("[parent-preload] preloading Thai NER…")
            _load_thai_ner()
        except Exception as e:
            logging.warning(f"[parent-preload] thai-ner preload skip: {e}")
    if opt.enable_fasttext and model_path:
        _load_fasttext(model_path)


def _worker_init(opt_dict: dict, model_path: Optional[str]):
    opt = Options(**opt_dict)
    global public_figure_skip
    public_figure_skip = bool(opt_dict.get("public_figure_skip", False))
    try:
        if opt.thai_ner_full and _THAI_NER_AVAILABLE:
            _load_thai_ner()
        if opt.enable_fasttext and model_path and _FT_MODEL is None:
            _load_fasttext(model_path)
        if opt.thai_ner_full and not _THAI_NER_AVAILABLE:
            raise RuntimeError("thai-ner requested but f_text_cleaner is unavailable in worker")
    except Exception as e:
        logging.error(f"[worker-init] fatal: {e}")
        raise


def _worker_init_gpu(opt_dict: dict, model_path: Optional[str], gpu_token: str):
    global public_figure_skip
    public_figure_skip = bool(opt_dict.get("public_figure_skip", False))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _worker_init(opt_dict, model_path)

# ========= I/O helpers =========

def iter_files_from_input(input_path: Path, pattern: str = "*.jsonl") -> List[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob(pattern) if p.is_file())
    return [input_path.resolve()]

def read_lines_in_chunks(path: Path, chunk_size: int):
    buf = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= chunk_size:
                yield buf; buf = []
    if buf:
        yield buf

# ========= Checkpoint manager =========
class CheckpointManager:
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.ckpt_file = temp_dir / "checkpoint.pkl"
        temp_dir.mkdir(parents=True, exist_ok=True)
    def load(self):
        if not self.ckpt_file.exists():
            return {"completed_parts": set(), "stats_total": {}}
        try:
            obj = pickle.loads(self.ckpt_file.read_bytes())
            obj["completed_parts"] = set(obj.get("completed_parts", []))
            obj["stats_total"] = dict(obj.get("stats_total", {}))
            return obj
        except Exception:
            return {"completed_parts": set(), "stats_total": {}}
    def save(self, state: dict):
        state = {
            "completed_parts": sorted(list(state.get("completed_parts", set()))),
            "stats_total": dict(state.get("stats_total", {})),
        }
        self.ckpt_file.write_bytes(pickle.dumps(state))
    def clear(self):
        removed = 0
        try:
            for p in self.temp_dir.glob("part_*.*"):
                try: p.unlink(); removed += 1
                except Exception: pass
            for p in self.temp_dir.glob("*.tmp"):
                try: p.unlink(); removed += 1
                except Exception: pass
            if self.ckpt_file.exists():
                try: self.ckpt_file.unlink(); removed += 1
                except Exception: pass
        finally:
            logging.info(f"[ckpt] cleared {removed} temp files in {self.temp_dir}")

# ========= Chunk worker =========

def _log_debug_diff(before: str, after: str, idx: int):
    if before == after:
        logging.info(f"[debug {idx}] unchanged: {after[:120]!r}")
    else:
        logging.info(f"[debug {idx}] BEFORE: {before}")
        logging.info(f"[debug {idx}] AFTER : {after}")


def process_chunk(args):
    chunk_lines, part_id, temp_dir, opt_dict = args
    opt = Options(**opt_dict)
    
    try:
        reset_romanize_cache()
    except Exception:
        pass
    try:
        reset_person_cache()  
    except Exception:
        pass
        
    stats = defaultdict(int)
    legal_tmp   = Path(temp_dir) / f"part_{part_id:08d}.legal.jsonl"
    illegal_tmp = Path(temp_dir) / f"part_{part_id:08d}.illegal.jsonl"

    with legal_tmp.open("w", encoding="utf-8") as f_legal, \
         illegal_tmp.open("w", encoding="utf-8") as f_illegal:
        for idx, line in enumerate(chunk_lines):
            try:
                line = (line or "").strip()
                if not line:
                    continue
                obj = json.loads(line)
            except Exception:
                stats["errors_json"] += 1
                continue

            text, src_key = extract_text_like(obj, opt.text_keys, opt.combine_keys)
            if not text:
                stats["skipped_no_text"] += 1
                continue

            flags: List[str] = []
            cleaned = process_text(text, opt, flags)

            # Mixed script detection (post-clean)
            mixed_hit = False
            if opt.detect_mixed_script and has_mixed_script_adjacent(cleaned):
                mixed_hit = True
                flags.append("mixed_adjacent")

            if opt.debug_first_n and idx < opt.debug_first_n:
                _log_debug_diff(text, cleaned, idx)

            out_obj = dict(obj)
            key = (src_key if src_key and not src_key.startswith("+") else "text")
            out_obj[key] = cleaned
            if opt.flags_key and flags:
                out_obj[opt.flags_key] = sorted(set(flags))

            # Routing: fastText only
            route = route_legal_illegal(cleaned, enable_ft=opt.enable_fasttext)

            line_out = json.dumps(out_obj, ensure_ascii=False) + "\n"
            if route == "illegal":
                f_illegal.write(line_out); stats["illegal"] += 1
            else:
                f_legal.write(line_out); stats["legal"] += 1
            stats["processed"] += 1

    # write done marker for resume correctness
    try:
        done_path = Path(temp_dir) / f"part_{part_id:08d}.done.json"
        with done_path.open("w", encoding="utf-8") as fd:
            json.dump({"part_id": part_id, "stats": dict(stats)}, fd, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"[part-done] cannot write marker for part {part_id}: {e}")

    return dict(stats)

# ========= Merge helpers =========

def stream_merge_parts(tmp_dir: Path, fo_legal, fo_illegal):
    done_markers = sorted(tmp_dir.glob("part_*.done.json"))
    for dm in done_markers:
        prefix = dm.name.replace(".done.json", "")
        p_legal = tmp_dir / f"{prefix}.legal.jsonl"
        p_illegal = tmp_dir / f"{prefix}.illegal.jsonl"
        if p_legal.exists():
            with p_legal.open("r", encoding="utf-8") as fi:
                for line in fi:
                    fo_legal.write(line)
        if p_illegal.exists():
            with p_illegal.open("r", encoding="utf-8") as fi:
                for line in fi:
                    fo_illegal.write(line)


def _write_summary(out_dir: Path, stats: Counter, opt: Options):
    summary = out_dir / "summary.txt"
    with summary.open("w", encoding="utf-8") as fo:
        fo.write("# Pipeline Summary\n")
        for k, v in sorted(stats.items()):
            fo.write(f"{k}: {v}\n")
        fo.write("\n# Options\n")
        fo.write(json.dumps({
            "clean_steps": opt.clean_steps,
            "pdpa_policy": opt.pdpa_policy,
            "pdpa_steps": opt.pdpa_steps,
            "thai_ner_full": opt.thai_ner_full,
            "thai_ner_policy": opt.thai_ner_policy,
            "thai_ner_cats": opt.thai_ner_cats,
            "enable_fasttext": opt.enable_fasttext,
            "detect_mixed_script": opt.detect_mixed_script,
            "mixed_policy": opt.mixed_policy,
            "thai_mark_fix": opt.thai_mark_fix,
            "floating_policy": opt.floating_policy,
            "pull_back_gap": opt.pull_back_gap,
            "normalize": opt.normalize,
            "flags_key": opt.flags_key,
        }, ensure_ascii=False, indent=2))

# ========= Per-file runner with checkpoint =========

def run_one_input_file(input_file: Path, out_dir: Path, args, opt: Options, opt_dict: dict, stats_total: Counter, pool: mp.pool.Pool):
    tmp_dir = out_dir / f"tmp_{input_file.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ckpt = CheckpointManager(tmp_dir)
    if getattr(args, "clear_ckpt", False):
        ckpt.clear()
    state = ckpt.load() if getattr(args, "resume", True) else {"completed_parts": set(), "stats_total": {}}
    completed_parts = set(state.get("completed_parts", set()))

    # build chunk jobs
    part_id = 0
    jobs = []
    for chunk in read_lines_in_chunks(input_file, getattr(args, "chunk_size", 50000)):
        done_json  = tmp_dir / f"part_{part_id:08d}.done.json"
        legal_tmp  = tmp_dir / f"part_{part_id:08d}.legal.jsonl"
        illegal_tmp= tmp_dir / f"part_{part_id:08d}.illegal.jsonl"

        already_done = False
        if getattr(args, "resume", True) and done_json.exists():
            try:
                meta = json.loads(done_json.read_text(encoding="utf-8"))
                if int(meta.get("stats", {}).get("processed", 0)) > 0:
                    already_done = True
            except Exception:
                already_done = False

        if already_done and getattr(args, "resume", True):
            part_id += 1
            continue

        # clean stale tmp before re-run
        for pp in (legal_tmp, illegal_tmp, done_json):
            try:
                if pp.exists(): pp.unlink()
            except Exception:
                pass

        jobs.append((chunk, part_id, tmp_dir, opt_dict))
        part_id += 1

    # dispatch
    async_jobs = []
    for ji, j in enumerate(jobs):
        if isinstance(pool, list):
            sel = pool[ji % len(pool)]
            async_jobs.append(sel.apply_async(process_chunk, (j,)))
        else:
            async_jobs.append(pool.apply_async(process_chunk, (j,)))
    pbar = tqdm(total=len(async_jobs), desc=f"Process parts: {input_file.name}", unit="part")

    save_every = max(1, getattr(args, "save_every", 5))
    for idx, aj in enumerate(async_jobs, 1):
        st = aj.get() or {}
        for k, v in st.items():
            stats_total[k] = stats_total.get(k, 0) + int(v)
        completed_parts.add(jobs[idx-1][1])
        if (len(completed_parts) % save_every) == 0:
            ckpt.save({"completed_parts": completed_parts, "stats_total": dict(stats_total)})
        pbar.update(1)
    pbar.close()

    ckpt.save({"completed_parts": completed_parts, "stats_total": dict(stats_total)})
    return tmp_dir

# ========= Main =========

def _self_test():
    if not _THAI_NER_AVAILABLE:
        print("f_text_cleaner not available. Self-test skipped.")
        return 0
    _load_thai_ner()
    sample = "ติดต่อ สมหญิง กล้าหาญ ได้ที่ 091-234-5678 ดีจัง555lol"
    out = thai_ner_full_apply(sample, policy="mask", keep_categories="PERSON,PHONE,EMAIL,ADDRESS,NATIONAL_ID,HOSPITAL_IDS")
    print("INPUT :", sample)
    print("OUTPUT:", out)
    print("MIXED?", has_mixed_script_adjacent(sample))
    fx, spans = fix_floating_below("กา ุแฟ", policy="drop")
    print("FLOAT FIX:", fx, spans)
    return 0


def _doctor():
    print("=== ENV DOCTOR ===")
    print("python:", sys.executable)
    print("version:", sys.version)
    try:
        import bs4  # type: ignore
        print("bs4: OK")
    except Exception as e:
        print("bs4: MISSING:", e)
    try:
        from src import f_text_cleaner as m
        print("f_text_cleaner: OK at", getattr(m, "__file__", "<unknown>"))
        try:
            _load_thai_ner(); print("thai_ner: preload OK")
        except Exception as e:
            print("thai_ner: preload FAIL:", e)
    except Exception as e:
        print("f_text_cleaner: IMPORT FAIL:", e)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    try:
        import torch  # type: ignore
        print("torch.cuda.device_count():", torch.cuda.device_count())
    except Exception as e:
        print("torch: not available (", e, ")")
    print("PYTHONPATH:")
    for p in sys.path:
        print(" -", p)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Thai text cleaning with PDPA + Thai NER + mixed-script + mark fixer + checkpoint")
    ap.add_argument("-i","--input", type=Path, required=False, help="JSONL file or directory")
    ap.add_argument("-o","--out", type=Path, default=Path("./runs"))
    ap.add_argument("--pattern", type=str, default="*.jsonl")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4)//2))
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--single-process", action="store_true", help="Run without multiprocessing (debug/compat)")

    # Multi‑GPU (devices auto via CUDA_VISIBLE_DEVICES or torch)
    ap.add_argument("--gpu-workers-per-gpu", type=int, default=1, help="Processes per GPU (each loads the model)")

    # output merge mode
    ap.add_argument("--append-out", action="store_true", help="append to existing outputs instead of overwrite")

    # Cleaning
    ap.add_argument("--clean-steps", type=str, default="html,emoji,emoticon")

    # PDPA
    ap.add_argument("--pdpa-policy", type=str, default="skip", choices=["skip","mask","anonymize","drop"])
    ap.add_argument("--pdpa-steps", type=str, default="")
    ap.add_argument("--keep-last", type=int, default=0)
    ap.add_argument("--card-mode", type=str, default="randomize")
    ap.add_argument("--account-mode", type=str, default="randomize")
    ap.add_argument("--pdpa-salt", type=str, default="")
    ap.add_argument("--addr-gazetteer", default="thai_provinces_districts.json", help="Thai gazetteer JSON for address detection (province/amphoe/tambon).")
    ap.add_argument("--no-addr-keep-province", action="store_true", help="If set, anonymized addresses may change province.")
    ap.add_argument("--addr-tag", default="[ADDRESS]", help="Mask tag for address in PDPA mask mode.")

    # Thai NER full
    ap.add_argument("--thai-ner-full", action="store_true")
    ap.add_argument("--thai-ner-policy", type=str, default="mask", choices=["mask","anonymize"])
    ap.add_argument("--thai-ner-cats", type=str, default="PERSON,PHONE,EMAIL,ADDRESS,NATIONAL_ID,HOSPITAL_IDS")
    ap.add_argument("--thai-ner-map", type=Path, default=None)

    # fastText (optional)
    ap.add_argument("--fasttext-model", type=Path, default=None)
    ap.add_argument("--no-fasttext", action="store_true")

    # Text keys
    ap.add_argument("--text-keys", type=str, default=None)
    ap.add_argument("--combine-keys", type=str, default=None)

    # Debug
    ap.add_argument("--debug-first-n", type=int, default=0)
    ap.add_argument("--debug-show-diff", action="store_true")

    # Checkpoint
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--clear-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=5)


    # NEW: Mixed-script + Thai mark fixer + flags
    ap.add_argument("--detect-mixed-script", action="store_true")
    ap.add_argument("--mixed-policy", type=str, default="skip", choices=["skip","tag"], help="When mixed-script adjacent is found: skip=do nothing, tag=record flag")
    ap.add_argument("--thai-mark-fix", action="store_true", help="Enable Thai mark fixer (pull back separated marks, drop/mask floating ุ/ู, collapse duplicates)")
    ap.add_argument("--floating-policy", type=str, default="drop", choices=["drop","mask","keep"])
    ap.add_argument("--pull-back-gap", type=str, default="any", choices=["any","tight"], help="What gaps to pull back (any=spaces/ZW, tight=ZW only)")
    ap.add_argument("--normalize", type=str, default="NFC", choices=["NFC","NFKC","none"])
    ap.add_argument("--flags-key", type=str, default=None, help="If set, store detection flags under this key in each output record")

    ap.add_argument("--public-figure-files", nargs="+", default=[], help="หนึ่งหรือหลายไฟล์ .txt รายชื่อบุคคลสาธารณะ (หนึ่งชื่อ/บรรทัด)")
    ap.add_argument("--no-public-figure-skip", action="store_true", help="ปิดการข้ามชื่อบุคคลสาธารณะ(debug)")

    args = ap.parse_args(argv)
    global public_figure_skip
    public_figure_skip = not args.no_public_figure_skip

    if getattr(args, "self_test", False):
        return _self_test()
    if getattr(args, "doctor", False):
        return _doctor()

    # Fail fast if user asked for NER but f_text_cleaner deps are missing
    if args.thai_ner_full and not _THAI_NER_AVAILABLE:
        logging.error(
            "[fatal] --thai-ner-full requested but f_text_cleaner is unavailable. "
            "Install 'beautifulsoup4' and ensure PYTHONPATH includes repo root.")
        return 3

    if not args.input:
        print("--input is required unless --self-test is used", file=sys.stderr)
        return 2
    if _F_CLEAN_AVAILABLE:
        if args.no_public_figure_skip:
            configure_public_figure_paths([])
        else:
            configure_public_figure_paths(args.public_figure_files or [])

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    fasttext_model_path = str(args.fasttext_model) if (args.fasttext_model and args.fasttext_model.exists()) else ""
    enable_ft = (not args.no_fasttext) and bool(fasttext_model_path)

    opt = Options(
        clean_steps=[s.strip().lower() for s in _parse_csv_list(args.clean_steps)],
        pdpa_policy=args.pdpa_policy,
        pdpa_steps=[s.strip().lower() for s in _parse_csv_list(args.pdpa_steps)],
        keep_last=args.keep_last,
        card_mode=args.card_mode,
        account_mode=args.account_mode,
        salt=args.pdpa_salt or "",
        thai_ner_full=bool(args.thai_ner_full),
        thai_ner_policy=args.thai_ner_policy,
        thai_ner_cats=[s.strip().upper() for s in _parse_csv_list(args.thai_ner_cats)] or ["PERSON","PHONE","EMAIL"],
        thai_ner_map=str(args.thai_ner_map) if args.thai_ner_map else None,
        enable_fasttext=enable_ft,
        text_keys=[s.strip() for s in _parse_csv_list(args.text_keys)] if args.text_keys else None,
        combine_keys=[s.strip() for s in _parse_csv_list(args.combine_keys)] if args.combine_keys else None,
        debug_first_n=max(0, int(args.debug_first_n)),
        debug_show_diff=bool(args.debug_show_diff),
        detect_mixed_script=bool(args.detect_mixed_script),
        mixed_policy=args.mixed_policy,
        thai_mark_fix=bool(args.thai_mark_fix),
        floating_policy=args.floating_policy,
        pull_back_gap=args.pull_back_gap,
        normalize=args.normalize,
        flags_key=(args.flags_key if args.flags_key else None),
    )

    # ==== GPU discovery (no explicit device CLI needed) ====
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    env_wpg = os.environ.get("THCLEAN_WORKERS_PER_GPU")
    if env_wpg:
        try: args.gpu_workers_per_gpu = int(env_wpg)
        except Exception: pass

    # Preload once in parent (skip NER here if multi-GPU)
    # Build per-worker CUDA masks
    try:
        import torch  # type: ignore
        n = int(torch.cuda.device_count()) if not cvd else len([t for t in cvd.split(',') if t.strip()])
    except Exception:
        n = len([t for t in cvd.split(',') if t.strip()]) if cvd else 0
    gpu_tokens = [t.strip() for t in cvd.split(',') if t.strip()] if cvd else [str(i) for i in range(n)]
    use_multi_gpu = bool(gpu_tokens and opt.thai_ner_full)
    if use_multi_gpu:
        if enable_ft and fasttext_model_path:
            _load_fasttext(fasttext_model_path)
        logging.info("[CFG] multi-GPU active: visible=%s map=%s workers/gpu=%d", (cvd or "<all>" if n>0 else "<none>"), ",".join(gpu_tokens), args.gpu_workers_per_gpu)
    else:
        parent_preload(opt, fasttext_model_path)

    logging.info("[CFG] thai_ner_full=%s policy=%s cats=%s", opt.thai_ner_full, opt.thai_ner_policy, ",".join(opt.thai_ner_cats))

    # ----- name outputs by input path (file or folder) -----
    if args.input.is_file():
        base_name = args.input.stem
    else:
        base_name = args.input.name or "merged"

    legal_out   = out_dir / f"GOOD_{base_name}.jsonl"
    illegal_out = out_dir / f"ฺBAD_{base_name}.jsonl" 
    mode = "a" if getattr(args, "append_out", False) else "w"

    stats_total: Counter = Counter()
    opt_dict = opt.__dict__.copy()
    opt_dict["public_figure_skip"] = public_figure_skip

    # Single-process mode
    if args.single_process:
        with legal_out.open(mode, encoding="utf-8") as fo_legal, \
             illegal_out.open(mode, encoding="utf-8") as fo_illegal:
            for path in iter_files_from_input(args.input, args.pattern):
                logging.info(f"[INPUT] {path} (single-process)")
                tmp_dir = out_dir / f"tmp_{path.stem}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                part_id = 0
                for chunk in read_lines_in_chunks(path, args.chunk_size):
                    res = process_chunk((chunk, part_id, tmp_dir, opt.__dict__.copy()))
                    for k, v in (res or {}).items():
                        stats_total[k] = stats_total.get(k, 0) + int(v)
                    stream_merge_parts(tmp_dir, fo_legal, fo_illegal)
                    part_id += 1
        # summary
        summary = out_dir / "summary.txt"
        with summary.open("w", encoding="utf-8") as fo:
            for k, v in sorted(stats_total.items()):
                fo.write(f"{k}: {v}\n")
        if stats_total.get("processed", 0) == 0:
            logging.warning("[warn] No records were processed. Check text keys or input format.")
        logging.info(f"[DONE] merged outputs: {legal_out}, {illegal_out}")
        return 0

    # Multi / CPU pools
    with legal_out.open(mode, encoding="utf-8") as fo_legal, \
         illegal_out.open(mode, encoding="utf-8") as fo_illegal:
        if use_multi_gpu:
            ctx = mp.get_context("spawn")
            pools = []
            try:
                for tok in gpu_tokens:
                    p = ctx.Pool(processes=max(1, args.gpu_workers_per_gpu), initializer=_worker_init_gpu, initargs=(opt_dict, fasttext_model_path, str(tok)))
                    pools.append(p)
                for path in iter_files_from_input(args.input, args.pattern):
                    logging.info(f"[INPUT] {path} (multi-GPU)")
                    tmp_dir = run_one_input_file(path, out_dir, args, opt, opt_dict, stats_total, pools)
                    stream_merge_parts(tmp_dir, fo_legal, fo_illegal)
            finally:
                for p in pools:
                    try: p.close(); p.join()
                    except Exception: pass
        else:
            workers = max(1, args.workers)
            ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
            with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(opt_dict, fasttext_model_path)) as pool:
                for path in iter_files_from_input(args.input, args.pattern):
                    logging.info(f"[INPUT] {path}")
                    tmp_dir = run_one_input_file(path, out_dir, args, opt, opt_dict, stats_total, pool)
                    stream_merge_parts(tmp_dir, fo_legal, fo_illegal)

    _write_summary(out_dir, stats_total, opt)
    if stats_total.get("processed", 0) == 0:
        logging.warning("[warn] No records were processed. Check text keys or input format.")
    logging.info("[DONE] merged outputs: legal.jsonl, illegal.jsonl")
    return 0


if __name__ == "__main__":
    try:
        if sys.platform.startswith("linux"):
            mp.set_start_method("fork")
        else:
            mp.set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())
