# -*- coding: utf-8 -*-
"""
Text Cleaning Utilities
-----------------------
A collection of functions for cleaning and normalizing text data.
"""

import re
import string
import warnings
from bs4 import BeautifulSoup
import unicodedata
from functools import lru_cache
import random, hashlib, datetime
import json
import logging
from typing import List, Dict, Tuple, Optional, Sequence, Set
import os

# Import libraries for anonymization features
#import ollama
from thaidp.filter import *
from thaidp.gen import *
from thaidp.clean import _replace, random_bool
from dataclasses import dataclass

# --- romanize + fuzzy match helpers ---
from collections import defaultdict

try:
    from pythainlp.transliterate import romanize as _romanize
    _HAS_THAI2ROM = True
    def _thai2rom(text: str) -> str | None:
        return _romanize(text, engine="thai2rom")
except Exception:
    _HAS_THAI2ROM = False
    def _thai2rom(text: str) -> str | None:
        return None

try:
    # optional, better speed/quality if available
    from rapidfuzz.fuzz import ratio as _rf_ratio
    def _sim(a: str, b: str) -> float:
        return _rf_ratio(a, b) / 100.0
except Exception:
    from difflib import SequenceMatcher
    def _sim(a: str, b: str) -> float:
        try:
            return SequenceMatcher(None, a, b).ratio()
        except Exception:
            return 0.0


# --- Global storages ---
# ไทย -> โรมัน (string เดียว เช่น "Somying Klaharn")
NEW_NAMES_ROMA: dict[str, str] = {}
# โรมัน lowercase -> เซ็ตชื่อไทย (รองรับแมปรูปแบบเดียวกันหลายชื่อไทย ถ้าจำเป็น)
ROMA2TH: defaultdict[str, set[str]] = defaultdict(set)
# ===== Per-document PERSON alias cache (Thai) =====
PERSON_ALIAS: dict[str, str] = {}
# สำรอง token ของ [PERSON]
try:
    _PERSON_TOKEN_DEFAULT = "[PERSON]" if "ENTITY_TO_ANONYMIZED_TOKEN_MAP" not in globals() \
        else globals()["ENTITY_TO_ANONYMIZED_TOKEN_MAP"].get("PERSON", "[PERSON]")
except Exception:
    _PERSON_TOKEN_DEFAULT = "[PERSON]"

# regex: ดึงคำตัวอักษรละติน (รวม ' และ -)
_LATIN_WORD_RX = re.compile(r"[A-Za-z][A-Za-z'\-]*")

_GENERIC_LATIN_NAME_RX = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,}|[a-z]{3,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|[a-z]{3,})){1,2}\b"
)

_URL_OR_EMAIL_HINT_RX = re.compile(r"(https?://|www\.|@)")

def reset_person_cache():
    """ล้างแคชชื่อบุคคล (ไทย) ต่อเอกสาร/ชังก์ เรียกก่อนเริ่มประมวลผลข้อความชุดใหม่"""
    PERSON_ALIAS.clear()

def _looks_like_url_or_email(s: str) -> bool:
    return bool(_URL_OR_EMAIL_HINT_RX.search(s))

_ZW = {0x200B:None, 0x200C:None, 0x200D:None, 0xFEFF:None}
_DASH_SPLIT = re.compile(r"\s*[-–—]\s*")  # -, – , —
_ASCII_ONLY = re.compile(r"^[\x00-\x7F]+$")

def _strip_z(s: str) -> str:
    return s.translate(_ZW)

_TH_PREFIXES = [
    "นาย","นาง","นางสาว","น.ส.","นส.","คุณ",
    "ดร.","ดร","ศ.","รศ.","ผศ.",
    "ม.ล.","ม.ร.ว.","ม.จ.",
    "พระ","หม่อม","จอมพล",
    "พล.อ.","พลเอก","พล.ท.","พลโท","พล.ต.","พลตรี",
    "ร.อ.","ร้อยเอก","พ.ต.","พันตรี","พ.อ.","พันเอก",
]
_PREFIX_PATTERN = r"^(?:(?:%s)\s+)+" % "|".join(map(re.escape, _TH_PREFIXES))
_PREFIX = re.compile(_PREFIX_PATTERN)


def normalize_person_key(name: str) -> str:
    s = str(name or "")
    # ตัด zero-width +ช่องว่างซ้ำ
    ss = _strip_z(s)  
    s = re.sub(r"\s+", " ", s).strip()
    # เอาคำนำหน้าออก (ต้นสตริงเท่านั้น)
    for hon in _TH_PREFIXES:
        if s.startswith(hon):
            s = s[len(hon):].lstrip()
            break
    # ตัวอย่าง: ตัดอักขระคั่นชื่อทั่วไปออกปลาย/หัว
    s = re.sub(r"^[\-–—•\s]+|[\-–—•\s]+$", "", s)
    return s  # ไทยไม่ต้อง lower-case
    
def normalize_person_name(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFC", s)
    s = _strip_z(s).strip()
    s = re.sub(r"\([^)]*\)", "", s).strip()
    s = _DASH_SPLIT.split(s, maxsplit=1)[0]
    s = _PREFIX.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

_PUBLIC_FIGURE_CFG = type("PF_CFG", (), {"paths": []})
public_figure_skip: bool = False      # ดีฟอลต์: ไม่ข้าม (จึง anonymize ได้ทุกคน)
_PUBLIC_FIGURE_SET: Set[str] = set()

def _build_pf_set(paths):
    names = set()
    for p in paths or []:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = _DASH_SPLIT.split(line, maxsplit=1)[0].strip()
                base = normalize_person_name(line)
                if base:
                    names.add(base)
    names |= {n.replace(" ", "") for n in list(names)}
    return names

def configure_public_figure_paths(paths: Sequence[str]) -> None:
    """กำหนดรายชื่อบุคคลสาธารณะ; ว่าง = ปิดโหมด skip"""
    global public_figure_skip, _PUBLIC_FIGURE_SET
    _PUBLIC_FIGURE_SET.clear()
    if not paths:
        public_figure_skip = False
        logging.info("[pubfig] disabled (no list)")
        return
    public_figure_skip = True
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        _PUBLIC_FIGURE_SET.add(name)
        except Exception as e:
            logging.warning("[pubfig] cannot read %s: %s", p, e)
    logging.info("[pubfig] enabled: %d names", len(_PUBLIC_FIGURE_SET))
    is_public_figure.cache_clear()
    

def _ensure_pf():
    global _PUBLIC_FIGURE_SET
    if _PUBLIC_FIGURE_SET is None:
        _PUBLIC_FIGURE_SET = _build_pf_set(_PUBLIC_FIGURE_CFG.paths)

@lru_cache(maxsize=131072)
def is_public_figure(name_text: str) -> bool:
    if not name_text:
        return False
    _ensure_pf()
    norm = normalize_person_name(name_text)
    if _ASCII_ONLY.match(norm):
        # skip roman at whitelist stage (roman handled per text via aliases)
        return False
    return (norm in _PUBLIC_FIGURE_SET) or (norm.replace(" ", "") in _PUBLIC_FIGURE_SET)


def _build_local_public_figure_roman_aliases(names_iterable) -> set[str]:
    """From Thai public-figure names matched in *this* text, derive roman aliases to skip
       (normalized and no-space variants) for per-text roman masking/anonymizing."""
    if not _HAS_THAI2ROM:
        return set()
    out = set()
    for n in names_iterable:
        name = (n.get("text","") if isinstance(n, dict) else str(n) if n is not None else "")
        if not name:
            continue
        if not is_public_figure(name):
            continue
        try:
            r = normalize_person_name(_thai2rom(name))
            if r:
                out.add(r)
                out.add(r.replace(" ", ""))
        except Exception:
            pass
    return out
    
# cache สำหรับ anonymize ชื่อโรมัน (คงเดิมทุกครั้งที่เจอซ้ำ)
ROMA_ANON_CACHE: dict[str, str] = {}

#------------------------------------------------------------------------------
# Load Model
#------------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification, pipeline

_TH_NER_PATH = os.environ.get(
    "THAI_NER_PATH",
    "/project/lt200393-foullm/QP/Pipeline_clean/model/models--loolootech--no-name-ner-th"
)

_thai_ner = None

def _load_thai_ner():
    global _thai_ner
    if _thai_ner is not None:
        return _thai_ner
    try:
        use_cpu = (os.environ.get("CUDA_VISIBLE_DEVICES", "") == "") or (not torch.cuda.is_available())
        tok = AutoTokenizer.from_pretrained(_TH_NER_PATH, local_files_only=True)
        _thai_ner = pipeline("token-classification", model=_TH_NER_PATH,
                             device = (-1 if use_cpu else 0),
                             aggregation_strategy="simple")
    except Exception as e:
        logging.warning("[thai_ner] disabled: %s", e)
        _thai_ner = None
    return _thai_ner
    
#------------------------------------------------------------------------------
# HTML and Markup Removal
#------------------------------------------------------------------------------

def remove_html_tags(text):
    """
    Remove HTML tags from text using BeautifulSoup.
    
    Args:
        text (str): Text containing HTML tags
        
    Returns:
        str: Clean text without HTML tags
    """
    # Suppress BeautifulSoup warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
    try:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        return text
    except Exception as e:
        print(f"Error in remove_html_tags: {str(e)}")
        return text

def remove_bbcode_tags(text):
    """
    Remove BBCode tags like [url], [img], etc. from text.
    
    Args:
        text (str): Text containing BBCode tags
        
    Returns:
        str: Clean text without BBCode tags
    """
    text = re.sub(r'\[/?(?:youtube|b|i|u|url|img|quote|code|size|color|list|.*?)\]', '', text)
    return text


#------------------------------------------------------------------------------
# Emoticons and Emoji Removal
#------------------------------------------------------------------------------

def remove_emoticons(text):
    """
    Remove ASCII emoticons and Thai emoticons from text.
    
    Args:
        text (str): Text containing emoticons
        
    Returns:
        str: Text with emoticons removed
    """
    patterns = [
        r'T\s*\^?\s*T',      # T T, T^T
        r'[-=\^_T]\s*[-=\^_T]',  # = =
        r'[Tㅠㅜπ]\s*[ㅠㅜ^~_.]\s*[Tㅠㅜπ]',
        r'\s*=\s*=\s*',      # = =
        r'=_=',              # =_=
        r'T\s*A\s*T',        # TAT
        r'T[\s_\^]*T',       # T_T
    ]
    combined_pattern = '|'.join(f'(?:{p})' for p in patterns)
    text = re.sub(combined_pattern, '', text)
    return text

def remove_emojis(text):
    """
    Remove Unicode emojis from text.
    
    Args:
        text (str): Text containing emojis
        
    Returns:
        str: Text with emojis removed
    """
    emoji_pattern = re.compile(
        pattern = "["
        u"\u231A-\u231B"
        u"\u23E9-\u23EC"
        u"\u25FD-\u25FE"
        u"\u2614-\u2615"
        u"\u2648-\u2653"
        u"\u26AA-\u26AB"
        u"\u26BD-\u26BE"
        u"\u26C4-\u26C5"
        u"\u26F2-\u26F3"
        u"\u270A-\u270B"
        u"\u2753-\u2755"
        u"\u2795-\u2797"
        u"\u2B1B-\u2B1C"
        u"\U0001F191-\U0001F19A"
        u"\U0001F232-\U0001F236"
        u"\U0001F238-\U0001F23A"
        u"\U0001F250-\U0001F251"
        u"\U0001F300-\U0001F30C"
        u"\U0001F30D-\U0001F30E"
        u"\U0001F313-\U0001F315"
        u"\U0001F316-\U0001F318"
        u"\U0001F31D-\U0001F31E"
        u"\U0001F31F-\U0001F320"
        u"\U0001F32D-\U0001F32F"
        u"\U0001F330-\U0001F331"
        u"\U0001F332-\U0001F333"
        u"\U0001F334-\U0001F335"
        u"\U0001F337-\U0001F34A"
        u"\U0001F34C-\U0001F34F"
        u"\U0001F351-\U0001F37B"
        u"\U0001F37E-\U0001F37F"
        u"\U0001F380-\U0001F393"
        u"\U0001F3A0-\U0001F3C4"
        u"\U0001F3CF-\U0001F3D3"
        u"\U0001F3E0-\U0001F3E3"
        u"\U0001F3E5-\U0001F3F0"
        u"\U0001F3F8-\U0001F407"
        u"\U0001F409-\U0001F40B"
        u"\U0001F40C-\U0001F40E"
        u"\U0001F40F-\U0001F410"
        u"\U0001F411-\U0001F412"
        u"\U0001F417-\U0001F429"
        u"\U0001F42B-\U0001F43E"
        u"\U0001F442-\U0001F464"
        u"\U0001F466-\U0001F46B"
        u"\U0001F46C-\U0001F46D"
        u"\U0001F46E-\U0001F4AC"
        u"\U0001F4AE-\U0001F4B5"
        u"\U0001F4B6-\U0001F4B7"
        u"\U0001F4B8-\U0001F4EB"
        u"\U0001F4EC-\U0001F4ED"
        u"\U0001F4F0-\U0001F4F4"
        u"\U0001F4F6-\U0001F4F7"
        u"\U0001F4F9-\U0001F4FC"
        u"\U0001F4FF-\U0001F502"
        u"\U0001F504-\U0001F507"
        u"\U0001F50A-\U0001F514"
        u"\U0001F516-\U0001F52B"
        u"\U0001F52C-\U0001F52D"
        u"\U0001F52E-\U0001F53D"
        u"\U0001F54B-\U0001F54E"
        u"\U0001F550-\U0001F55B"
        u"\U0001F55C-\U0001F567"
        u"\U0001F595-\U0001F596"
        u"\U0001F5FB-\U0001F5FF"
        u"\U0001F601-\U0001F606"
        u"\U0001F607-\U0001F608"
        u"\U0001F609-\U0001F60D"
        u"\U0001F612-\U0001F614"
        u"\U0001F61C-\U0001F61E"
        u"\U0001F620-\U0001F625"
        u"\U0001F626-\U0001F627"
        u"\U0001F628-\U0001F62B"
        u"\U0001F62E-\U0001F62F"
        u"\U0001F630-\U0001F633"
        u"\U0001F637-\U0001F640"
        u"\U0001F641-\U0001F644"
        u"\U0001F645-\U0001F64F"
        u"\U0001F681-\U0001F682"
        u"\U0001F683-\U0001F685"
        u"\U0001F68A-\U0001F68B"
        u"\U0001F691-\U0001F693"
        u"\U0001F699-\U0001F69A"
        u"\U0001F69B-\U0001F6A1"
        u"\U0001F6A4-\U0001F6A5"
        u"\U0001F6A7-\U0001F6AD"
        u"\U0001F6AE-\U0001F6B1"
        u"\U0001F6B3-\U0001F6B5"
        u"\U0001F6B7-\U0001F6B8"
        u"\U0001F6B9-\U0001F6BE"
        u"\U0001F6C1-\U0001F6C5"
        u"\U0001F6D1-\U0001F6D2"
        u"\U0001F6D6-\U0001F6D7"
        u"\U0001F6DD-\U0001F6DF"
        u"\U0001F6EB-\U0001F6EC"
        u"\U0001F6F4-\U0001F6F6"
        u"\U0001F6F7-\U0001F6F8"
        u"\U0001F6FB-\U0001F6FC"
        u"\U0001F7E0-\U0001F7EB"
        u"\U0001F90D-\U0001F90F"
        u"\U0001F910-\U0001F918"
        u"\U0001F919-\U0001F91E"
        u"\U0001F920-\U0001F927"
        u"\U0001F928-\U0001F92F"
        u"\U0001F931-\U0001F932"
        u"\U0001F933-\U0001F93A"
        u"\U0001F93C-\U0001F93E"
        u"\U0001F940-\U0001F945"
        u"\U0001F947-\U0001F94B"
        u"\U0001F94D-\U0001F94F"
        u"\U0001F950-\U0001F95E"
        u"\U0001F95F-\U0001F96B"
        u"\U0001F96C-\U0001F970"
        u"\U0001F973-\U0001F976"
        u"\U0001F977-\U0001F978"
        u"\U0001F97C-\U0001F97F"
        u"\U0001F980-\U0001F984"
        u"\U0001F985-\U0001F991"
        u"\U0001F992-\U0001F997"
        u"\U0001F998-\U0001F9A2"
        u"\U0001F9A3-\U0001F9A4"
        u"\U0001F9A5-\U0001F9AA"
        u"\U0001F9AB-\U0001F9AD"
        u"\U0001F9AE-\U0001F9AF"
        u"\U0001F9B0-\U0001F9B9"
        u"\U0001F9BA-\U0001F9BF"
        u"\U0001F9C1-\U0001F9C2"
        u"\U0001F9C3-\U0001F9CA"
        u"\U0001F9CD-\U0001F9CF"
        u"\U0001F9D0-\U0001F9E6"
        u"\U0001F9E7-\U0001F9FF"
        u"\U0001FA70-\U0001FA73"
        u"\U0001FA78-\U0001FA7A"
        u"\U0001FA7B-\U0001FA7C"
        u"\U0001FA80-\U0001FA82"
        u"\U0001FA83-\U0001FA86"
        u"\U0001FA90-\U0001FA95"
        u"\U0001FA96-\U0001FAA8"
        u"\U0001FAA9-\U0001FAAC"
        u"\U0001FAB0-\U0001FAB6"
        u"\U0001FAB7-\U0001FABA"
        u"\U0001FAC0-\U0001FAC2"
        u"\U0001FAC3-\U0001FAC5"
        u"\U0001FAD0-\U0001FAD6"
        u"\U0001FAD7-\U0001FAD9"
        u"\U0001FAE0-\U0001FAE7"
        u"\U0001FAF0-\U0001FAF6"
        u"\u23F0"
        u"\u23F3"
        u"\u267F"
        u"\u2693"
        u"\u26A1"
        u"\u26CE"
        u"\u26D4"
        u"\u26EA"
        u"\u26F5"
        u"\u26FA"
        u"\u26FD"
        u"\u2705"
        u"\u2728"
        u"\u274C"
        u"\u274E"
        u"\u2757"
        u"\u27B0"
        u"\u27BF"
        u"\u2B50"
        u"\u2B55"
        u"\U0001F004"
        u"\U0001F0CF"
        u"\U0001F18E"
        u"\U0001F201"
        u"\U0001F21A"
        u"\U0001F22F"
        u"\U0001F30F"
        u"\U0001F310"
        u"\U0001F311"
        u"\U0001F312"
        u"\U0001F319"
        u"\U0001F31A"
        u"\U0001F31B"
        u"\U0001F31C"
        u"\U0001F34B"
        u"\U0001F350"
        u"\U0001F37C"
        u"\U0001F3C5"
        u"\U0001F3C6"
        u"\U0001F3C7"
        u"\U0001F3C8"
        u"\U0001F3C9"
        u"\U0001F3CA"
        u"\U0001F3E4"
        u"\U0001F3F4"
        u"\U0001F408"
        u"\U0001F413"
        u"\U0001F414"
        u"\U0001F415"
        u"\U0001F416"
        u"\U0001F42A"
        u"\U0001F440"
        u"\U0001F465"
        u"\U0001F4AD"
        u"\U0001F4EE"
        u"\U0001F4EF"
        u"\U0001F4F5"
        u"\U0001F4F8"
        u"\U0001F503"
        u"\U0001F508"
        u"\U0001F509"
        u"\U0001F515"
        u"\U0001F57A"
        u"\U0001F5A4"
        u"\U0001F600"
        u"\U0001F60E"
        u"\U0001F60F"
        u"\U0001F610"
        u"\U0001F611"
        u"\U0001F615"
        u"\U0001F616"
        u"\U0001F617"
        u"\U0001F618"
        u"\U0001F619"
        u"\U0001F61A"
        u"\U0001F61B"
        u"\U0001F61F"
        u"\U0001F62C"
        u"\U0001F62D"
        u"\U0001F634"
        u"\U0001F635"
        u"\U0001F636"
        u"\U0001F680"
        u"\U0001F686"
        u"\U0001F687"
        u"\U0001F688"
        u"\U0001F689"
        u"\U0001F68C"
        u"\U0001F68D"
        u"\U0001F68E"
        u"\U0001F68F"
        u"\U0001F690"
        u"\U0001F694"
        u"\U0001F695"
        u"\U0001F696"
        u"\U0001F697"
        u"\U0001F698"
        u"\U0001F6A2"
        u"\U0001F6A3"
        u"\U0001F6A6"
        u"\U0001F6B2"
        u"\U0001F6B6"
        u"\U0001F6BF"
        u"\U0001F6C0"
        u"\U0001F6CC"
        u"\U0001F6D0"
        u"\U0001F6D5"
        u"\U0001F6F9"
        u"\U0001F6FA"
        u"\U0001F7F0"
        u"\U0001F90C"
        u"\U0001F91F"
        u"\U0001F930"
        u"\U0001F93F"
        u"\U0001F94C"
        u"\U0001F971"
        u"\U0001F972"
        u"\U0001F979"
        u"\U0001F97A"
        u"\U0001F97B"
        u"\U0001F9C0"
        u"\U0001F9CB"
        u"\U0001F9CC"
        u"\U0001FA74"
        u"\u00A9"
        u"\uFE0F"
        u"\u00AE"
        u"\uFE0F"
        u"\u203C"
        u"\uFE0F"
        u"\u2049"
        u"\uFE0F"
        u"\u2122"
        u"\uFE0F"
        u"\u2139"
        u"\uFE0F"
        u"\u2194"
        u"\uFE0F"
        u"\u2195"
        u"\uFE0F"
        u"\u2196"
        u"\uFE0F"
        u"\u2197"
        u"\uFE0F"
        u"\u2198"
        u"\uFE0F"
        u"\u2199"
        u"\uFE0F"
        u"\u21A9"
        u"\uFE0F"
        u"\u21AA"
        u"\uFE0F"
        u"\u2328"
        u"\uFE0F"
        u"\u23CF"
        u"\uFE0F"
        u"\u23ED"
        u"\uFE0F"
        u"\u23EE"
        u"\uFE0F"
        u"\u23EF"
        u"\uFE0F"
        u"\u23F1"
        u"\uFE0F"
        u"\u23F2"
        u"\uFE0F"
        u"\u23F8"
        u"\uFE0F"
        u"\u23F9"
        u"\uFE0F"
        u"\u23FA"
        u"\uFE0F"
        u"\u24C2"
        u"\uFE0F"
        u"\u25AA"
        u"\uFE0F"
        u"\u25AB"
        u"\uFE0F"
        u"\u25B6"
        u"\uFE0F"
        u"\u25C0"
        u"\uFE0F"
        u"\u25FB"
        u"\uFE0F"
        u"\u25FC"
        u"\uFE0F"
        u"\u2600"
        u"\uFE0F"
        u"\u2601"
        u"\uFE0F"
        u"\u2602"
        u"\uFE0F"
        u"\u2603"
        u"\uFE0F"
        u"\u2604"
        u"\uFE0F"
        u"\u260E"
        u"\uFE0F"
        u"\u2611"
        u"\uFE0F"
        u"\u2618"
        u"\uFE0F"
        u"\u261D"
        u"\uFE0F"
        u"\u2620"
        u"\uFE0F"
        u"\u2622"
        u"\uFE0F"
        u"\u2623"
        u"\uFE0F"
        u"\u2626"
        u"\uFE0F"
        u"\u262A"
        u"\uFE0F"
        u"\u262E"
        u"\uFE0F"
        u"\u262F"
        u"\uFE0F"
        u"\u2638"
        u"\uFE0F"
        u"\u2639"
        u"\uFE0F"
        u"\u263A"
        u"\uFE0F"
        u"\u2640"
        u"\uFE0F"
        u"\u2642"
        u"\uFE0F"
        u"\u265F"
        u"\uFE0F"
        u"\u2660"
        u"\uFE0F"
        u"\u2663"
        u"\uFE0F"
        u"\u2665"
        u"\uFE0F"
        u"\u2666"
        u"\uFE0F"
        u"\u2668"
        u"\uFE0F"
        u"\u267B"
        u"\uFE0F"
        u"\u267E"
        u"\uFE0F"
        u"\u2692"
        u"\uFE0F"
        u"\u2694"
        u"\uFE0F"
        u"\u2695"
        u"\uFE0F"
        u"\u2696"
        u"\uFE0F"
        u"\u2697"
        u"\uFE0F"
        u"\u2699"
        u"\uFE0F"
        u"\u269B"
        u"\uFE0F"
        u"\u269C"
        u"\uFE0F"
        u"\u26A0"
        u"\uFE0F"
        u"\u26A7"
        u"\uFE0F"
        u"\u26B0"
        u"\uFE0F"
        u"\u26B1"
        u"\uFE0F"
        u"\u26C8"
        u"\uFE0F"
        u"\u26CF"
        u"\uFE0F"
        u"\u26D1"
        u"\uFE0F"
        u"\u26D3"
        u"\uFE0F"
        u"\u26E9"
        u"\uFE0F"
        u"\u26F0"
        u"\uFE0F"
        u"\u26F1"
        u"\uFE0F"
        u"\u26F4"
        u"\uFE0F"
        u"\u26F7"
        u"\uFE0F"
        u"\u26F8"
        u"\uFE0F"
        u"\u26F9"
        u"\uFE0F"
        u"\u2702"
        u"\uFE0F"
        u"\u2708"
        u"\uFE0F"
        u"\u2709"
        u"\uFE0F"
        u"\u270C"
        u"\uFE0F"
        u"\u270D"
        u"\uFE0F"
        u"\u270F"
        u"\uFE0F"
        u"\u2712"
        u"\uFE0F"
        u"\u2714"
        u"\uFE0F"
        u"\u2716"
        u"\uFE0F"
        u"\u271D"
        u"\uFE0F"
        u"\u2721"
        u"\uFE0F"
        u"\u2733"
        u"\uFE0F"
        u"\u2734"
        u"\uFE0F"
        u"\u2744"
        u"\uFE0F"
        u"\u2747"
        u"\uFE0F"
        u"\u2763"
        u"\uFE0F"
        u"\u2764"
        u"\uFE0F"
        u"\u27A1"
        u"\uFE0F"
        u"\u2934"
        u"\uFE0F"
        u"\u2935"
        u"\uFE0F"
        u"\u2B05"
        u"\uFE0F"
        u"\u2B06"
        u"\uFE0F"
        u"\u2B07"
        u"\uFE0F"
        u"\u3030"
        u"\uFE0F"
        u"\u303D"
        u"\uFE0F"
        u"\u3297"
        u"\uFE0F"
        u"\u3299"
        u"\uFE0F"
        u"\U0001F170"
        u"\uFE0F"
        u"\U0001F171"
        u"\uFE0F"
        u"\U0001F17E"
        u"\uFE0F"
        u"\U0001F17F"
        u"\uFE0F"
        u"\U0001F202"
        u"\uFE0F"
        u"\U0001F237"
        u"\uFE0F"
        u"\U0001F321"
        u"\uFE0F"
        u"\U0001F324"
        u"\uFE0F"
        u"\U0001F325"
        u"\uFE0F"
        u"\U0001F326"
        u"\uFE0F"
        u"\U0001F327"
        u"\uFE0F"
        u"\U0001F328"
        u"\uFE0F"
        u"\U0001F329"
        u"\uFE0F"
        u"\U0001F32A"
        u"\uFE0F"
        u"\U0001F32B"
        u"\uFE0F"
        u"\U0001F32C"
        u"\uFE0F"
        u"\U0001F336"
        u"\uFE0F"
        u"\U0001F37D"
        u"\uFE0F"
        u"\U0001F396"
        u"\uFE0F"
        u"\U0001F397"
        u"\uFE0F"
        u"\U0001F399"
        u"\uFE0F"
        u"\U0001F39A"
        u"\uFE0F"
        u"\U0001F39B"
        u"\uFE0F"
        u"\U0001F39E"
        u"\uFE0F"
        u"\U0001F39F"
        u"\uFE0F"
        u"\U0001F3CB"
        u"\uFE0F"
        u"\U0001F3CC"
        u"\uFE0F"
        u"\U0001F3CD"
        u"\uFE0F"
        u"\U0001F3CE"
        u"\uFE0F"
        u"\U0001F3D4"
        u"\uFE0F"
        u"\U0001F3D5"
        u"\uFE0F"
        u"\U0001F3D6"
        u"\uFE0F"
        u"\U0001F3D7"
        u"\uFE0F"
        u"\U0001F3D8"
        u"\uFE0F"
        u"\U0001F3D9"
        u"\uFE0F"
        u"\U0001F3DA"
        u"\uFE0F"
        u"\U0001F3DB"
        u"\uFE0F"
        u"\U0001F3DC"
        u"\uFE0F"
        u"\U0001F3DD"
        u"\uFE0F"
        u"\U0001F3DE"
        u"\uFE0F"
        u"\U0001F3DF"
        u"\uFE0F"
        u"\U0001F3F3"
        u"\uFE0F"
        u"\U0001F3F5"
        u"\uFE0F"
        u"\U0001F3F7"
        u"\uFE0F"
        u"\U0001F43F"
        u"\uFE0F"
        u"\U0001F441"
        u"\uFE0F"
        u"\U0001F4FD"
        u"\uFE0F"
        u"\U0001F549"
        u"\uFE0F"
        u"\U0001F54A"
        u"\uFE0F"
        u"\U0001F56F"
        u"\uFE0F"
        u"\U0001F570"
        u"\uFE0F"
        u"\U0001F573"
        u"\uFE0F"
        u"\U0001F574"
        u"\uFE0F"
        u"\U0001F575"
        u"\uFE0F"
        u"\U0001F576"
        u"\uFE0F"
        u"\U0001F577"
        u"\uFE0F"
        u"\U0001F578"
        u"\uFE0F"
        u"\U0001F579"
        u"\uFE0F"
        u"\U0001F587"
        u"\uFE0F"
        u"\U0001F58A"
        u"\uFE0F"
        u"\U0001F58B"
        u"\uFE0F"
        u"\U0001F58C"
        u"\uFE0F"
        u"\U0001F58D"
        u"\uFE0F"
        u"\U0001F590"
        u"\uFE0F"
        u"\U0001F5A5"
        u"\uFE0F"
        u"\U0001F5A8"
        u"\uFE0F"
        u"\U0001F5B1"
        u"\uFE0F"
        u"\U0001F5B2"
        u"\uFE0F"
        u"\U0001F5BC"
        u"\uFE0F"
        u"\U0001F5C2"
        u"\uFE0F"
        u"\U0001F5C3"
        u"\uFE0F"
        u"\U0001F5C4"
        u"\uFE0F"
        u"\U0001F5D1"
        u"\uFE0F"
        u"\U0001F5D2"
        u"\uFE0F"
        u"\U0001F5D3"
        u"\uFE0F"
        u"\U0001F5DC"
        u"\uFE0F"
        u"\U0001F5DD"
        u"\uFE0F"
        u"\U0001F5DE"
        u"\uFE0F"
        u"\U0001F5E1"
        u"\uFE0F"
        u"\U0001F5E3"
        u"\uFE0F"
        u"\U0001F5E8"
        u"\uFE0F"
        u"\U0001F5EF"
        u"\uFE0F"
        u"\U0001F5F3"
        u"\uFE0F"
        u"\U0001F5FA"
        u"\uFE0F"
        u"\U0001F6CB"
        u"\uFE0F"
        u"\U0001F6CD"
        u"\uFE0F"
        u"\U0001F6CE"
        u"\uFE0F"
        u"\U0001F6CF"
        u"\uFE0F"
        u"\U0001F6E0"
        u"\uFE0F"
        u"\U0001F6E1"
        u"\uFE0F"
        u"\U0001F6E2"
        u"\uFE0F"
        u"\U0001F6E3"
        u"\uFE0F"
        u"\U0001F6E4"
        u"\uFE0F"
        u"\U0001F6E5"
        u"\uFE0F"
        u"\U0001F6E9"
        u"\uFE0F"
        u"\U0001F6F0"
        u"\uFE0F"
        u"\U0001F6F3"
        u"\uFE0F"
        u"\u0023"
        u"\uFE0F"
        u"\u20E3"
        u"\u002A"
        u"\uFE0F"
        u"\u20E3"
        u"\u0030"
        u"\uFE0F"
        u"\u20E3"
        u"\u0031"
        u"\uFE0F"
        u"\u20E3"
        u"\u0032"
        u"\uFE0F"
        u"\u20E3"
        u"\u0033"
        u"\uFE0F"
        u"\u20E3"
        u"\u0034"
        u"\uFE0F"
        u"\u20E3"
        u"\u0035"
        u"\uFE0F"
        u"\u20E3"
        u"\u0036"
        u"\uFE0F"
        u"\u20E3"
        u"\u0037"
        u"\uFE0F"
        u"\u20E3"
        u"\u0038"
        u"\uFE0F"
        u"\u20E3"
        u"\u0039"
        u"\uFE0F"
        u"\u20E3"
        u"\U0001F1E6"
        u"\U0001F1E8"
        u"\U0001F1E6"
        u"\U0001F1E9"
        u"\U0001F1E6"
        u"\U0001F1EA"
        u"\U0001F1E6"
        u"\U0001F1EB"
        u"\U0001F1E6"
        u"\U0001F1EC"
        u"\U0001F1E6"
        u"\U0001F1EE"
        u"\U0001F1E6"
        u"\U0001F1F1"
        u"\U0001F1E6"
        u"\U0001F1F2"
        u"\U0001F1E6"
        u"\U0001F1F4"
        u"\U0001F1E6"
        u"\U0001F1F6"
        u"\U0001F1E6"
        u"\U0001F1F7"
        u"\U0001F1E6"
        u"\U0001F1F8"
        u"\U0001F1E6"
        u"\U0001F1F9"
        u"\U0001F1E6"
        u"\U0001F1FA"
        u"\U0001F1E6"
        u"\U0001F1FC"
        u"\U0001F1E6"
        u"\U0001F1FD"
        u"\U0001F1E6"
        u"\U0001F1FF"
        u"\U0001F1E7"
        u"\U0001F1E6"
        u"\U0001F1E7"
        u"\U0001F1E7"
        u"\U0001F1E7"
        u"\U0001F1E9"
        u"\U0001F1E7"
        u"\U0001F1EA"
        u"\U0001F1E7"
        u"\U0001F1EB"
        u"\U0001F1E7"
        u"\U0001F1EC"
        u"\U0001F1E7"
        u"\U0001F1ED"
        u"\U0001F1E7"
        u"\U0001F1EE"
        u"\U0001F1E7"
        u"\U0001F1EF"
        u"\U0001F1E7"
        u"\U0001F1F1"
        u"\U0001F1E7"
        u"\U0001F1F2"
        u"\U0001F1E7"
        u"\U0001F1F3"
        u"\U0001F1E7"
        u"\U0001F1F4"
        u"\U0001F1E7"
        u"\U0001F1F6"
        u"\U0001F1E7"
        u"\U0001F1F7"
        u"\U0001F1E7"
        u"\U0001F1F8"
        u"\U0001F1E7"
        u"\U0001F1F9"
        u"\U0001F1E7"
        u"\U0001F1FB"
        u"\U0001F1E7"
        u"\U0001F1FC"
        u"\U0001F1E7"
        u"\U0001F1FE"
        u"\U0001F1E7"
        u"\U0001F1FF"
        u"\U0001F1E8"
        u"\U0001F1E6"
        u"\U0001F1E8"
        u"\U0001F1E8"
        u"\U0001F1E8"
        u"\U0001F1E9"
        u"\U0001F1E8"
        u"\U0001F1EB"
        u"\U0001F1E8"
        u"\U0001F1EC"
        u"\U0001F1E8"
        u"\U0001F1ED"
        u"\U0001F1E8"
        u"\U0001F1EE"
        u"\U0001F1E8"
        u"\U0001F1F0"
        u"\U0001F1E8"
        u"\U0001F1F1"
        u"\U0001F1E8"
        u"\U0001F1F2"
        u"\U0001F1E8"
        u"\U0001F1F3"
        u"\U0001F1E8"
        u"\U0001F1F4"
        u"\U0001F1E8"
        u"\U0001F1F5"
        u"\U0001F1E8"
        u"\U0001F1F7"
        u"\U0001F1E8"
        u"\U0001F1FA"
        u"\U0001F1E8"
        u"\U0001F1FB"
        u"\U0001F1E8"
        u"\U0001F1FC"
        u"\U0001F1E8"
        u"\U0001F1FD"
        u"\U0001F1E8"
        u"\U0001F1FE"
        u"\U0001F1E8"
        u"\U0001F1FF"
        u"\U0001F1E9"
        u"\U0001F1EA"
        u"\U0001F1E9"
        u"\U0001F1EC"
        u"\U0001F1E9"
        u"\U0001F1EF"
        u"\U0001F1E9"
        u"\U0001F1F0"
        u"\U0001F1E9"
        u"\U0001F1F2"
        u"\U0001F1E9"
        u"\U0001F1F4"
        u"\U0001F1E9"
        u"\U0001F1FF"
        u"\U0001F1EA"
        u"\U0001F1E6"
        u"\U0001F1EA"
        u"\U0001F1E8"
        u"\U0001F1EA"
        u"\U0001F1EA"
        u"\U0001F1EA"
        u"\U0001F1EC"
        u"\U0001F1EA"
        u"\U0001F1ED"
        u"\U0001F1EA"
        u"\U0001F1F7"
        u"\U0001F1EA"
        u"\U0001F1F8"
        u"\U0001F1EA"
        u"\U0001F1F9"
        u"\U0001F1EA"
        u"\U0001F1FA"
        u"\U0001F1EB"
        u"\U0001F1EE"
        u"\U0001F1EB"
        u"\U0001F1EF"
        u"\U0001F1EB"
        u"\U0001F1F0"
        u"\U0001F1EB"
        u"\U0001F1F2"
        u"\U0001F1EB"
        u"\U0001F1F4"
        u"\U0001F1EB"
        u"\U0001F1F7"
        u"\U0001F1EC"
        u"\U0001F1E6"
        u"\U0001F1EC"
        u"\U0001F1E7"
        u"\U0001F1EC"
        u"\U0001F1E9"
        u"\U0001F1EC"
        u"\U0001F1EA"
        u"\U0001F1EC"
        u"\U0001F1EB"
        u"\U0001F1EC"
        u"\U0001F1EC"
        u"\U0001F1EC"
        u"\U0001F1ED"
        u"\U0001F1EC"
        u"\U0001F1EE"
        u"\U0001F1EC"
        u"\U0001F1F1"
        u"\U0001F1EC"
        u"\U0001F1F2"
        u"\U0001F1EC"
        u"\U0001F1F3"
        u"\U0001F1EC"
        u"\U0001F1F5"
        u"\U0001F1EC"
        u"\U0001F1F6"
        u"\U0001F1EC"
        u"\U0001F1F7"
        u"\U0001F1EC"
        u"\U0001F1F8"
        u"\U0001F1EC"
        u"\U0001F1F9"
        u"\U0001F1EC"
        u"\U0001F1FA"
        u"\U0001F1EC"
        u"\U0001F1FC"
        u"\U0001F1EC"
        u"\U0001F1FE"
        u"\U0001F1ED"
        u"\U0001F1F0"
        u"\U0001F1ED"
        u"\U0001F1F2"
        u"\U0001F1ED"
        u"\U0001F1F3"
        u"\U0001F1ED"
        u"\U0001F1F7"
        u"\U0001F1ED"
        u"\U0001F1F9"
        u"\U0001F1ED"
        u"\U0001F1FA"
        u"\U0001F1EE"
        u"\U0001F1E8"
        u"\U0001F1EE"
        u"\U0001F1E9"
        u"\U0001F1EE"
        u"\U0001F1EA"
        u"\U0001F1EE"
        u"\U0001F1F1"
        u"\U0001F1EE"
        u"\U0001F1F2"
        u"\U0001F1EE"
        u"\U0001F1F3"
        u"\U0001F1EE"
        u"\U0001F1F4"
        u"\U0001F1EE"
        u"\U0001F1F6"
        u"\U0001F1EE"
        u"\U0001F1F7"
        u"\U0001F1EE"
        u"\U0001F1F8"
        u"\U0001F1EE"
        u"\U0001F1F9"
        u"\U0001F1EF"
        u"\U0001F1EA"
        u"\U0001F1EF"
        u"\U0001F1F2"
        u"\U0001F1EF"
        u"\U0001F1F4"
        u"\U0001F1EF"
        u"\U0001F1F5"
        u"\U0001F1F0"
        u"\U0001F1EA"
        u"\U0001F1F0"
        u"\U0001F1EC"
        u"\U0001F1F0"
        u"\U0001F1ED"
        u"\U0001F1F0"
        u"\U0001F1EE"
        u"\U0001F1F0"
        u"\U0001F1F2"
        u"\U0001F1F0"
        u"\U0001F1F3"
        u"\U0001F1F0"
        u"\U0001F1F5"
        u"\U0001F1F0"
        u"\U0001F1F7"
        u"\U0001F1F0"
        u"\U0001F1FC"
        u"\U0001F1F0"
        u"\U0001F1FE"
        u"\U0001F1F0"
        u"\U0001F1FF"
        u"\U0001F1F1"
        u"\U0001F1E6"
        u"\U0001F1F1"
        u"\U0001F1E7"
        u"\U0001F1F1"
        u"\U0001F1E8"
        u"\U0001F1F1"
        u"\U0001F1EE"
        u"\U0001F1F1"
        u"\U0001F1F0"
        u"\U0001F1F1"
        u"\U0001F1F7"
        u"\U0001F1F1"
        u"\U0001F1F8"
        u"\U0001F1F1"
        u"\U0001F1F9"
        u"\U0001F1F1"
        u"\U0001F1FA"
        u"\U0001F1F1"
        u"\U0001F1FB"
        u"\U0001F1F1"
        u"\U0001F1FE"
        u"\U0001F1F2"
        u"\U0001F1E6"
        u"\U0001F1F2"
        u"\U0001F1E8"
        u"\U0001F1F2"
        u"\U0001F1E9"
        u"\U0001F1F2"
        u"\U0001F1EA"
        u"\U0001F1F2"
        u"\U0001F1EB"
        u"\U0001F1F2"
        u"\U0001F1EC"
        u"\U0001F1F2"
        u"\U0001F1ED"
        u"\U0001F1F2"
        u"\U0001F1F0"
        u"\U0001F1F2"
        u"\U0001F1F1"
        u"\U0001F1F2"
        u"\U0001F1F2"
        u"\U0001F1F2"
        u"\U0001F1F3"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F1F2"
        u"\U0001F1F5"
        u"\U0001F1F2"
        u"\U0001F1F6"
        u"\U0001F1F2"
        u"\U0001F1F7"
        u"\U0001F1F2"
        u"\U0001F1F8"
        u"\U0001F1F2"
        u"\U0001F1F9"
        u"\U0001F1F2"
        u"\U0001F1FA"
        u"\U0001F1F2"
        u"\U0001F1FB"
        u"\U0001F1F2"
        u"\U0001F1FC"
        u"\U0001F1F2"
        u"\U0001F1FD"
        u"\U0001F1F2"
        u"\U0001F1FE"
        u"\U0001F1F2"
        u"\U0001F1FF"
        u"\U0001F1F3"
        u"\U0001F1E6"
        u"\U0001F1F3"
        u"\U0001F1E8"
        u"\U0001F1F3"
        u"\U0001F1EA"
        u"\U0001F1F3"
        u"\U0001F1EB"
        u"\U0001F1F3"
        u"\U0001F1EC"
        u"\U0001F1F3"
        u"\U0001F1EE"
        u"\U0001F1F3"
        u"\U0001F1F1"
        u"\U0001F1F3"
        u"\U0001F1F4"
        u"\U0001F1F3"
        u"\U0001F1F5"
        u"\U0001F1F3"
        u"\U0001F1F7"
        u"\U0001F1F3"
        u"\U0001F1FA"
        u"\U0001F1F3"
        u"\U0001F1FF"
        u"\U0001F1F4"
        u"\U0001F1F2"
        u"\U0001F1F5"
        u"\U0001F1E6"
        u"\U0001F1F5"
        u"\U0001F1EA"
        u"\U0001F1F5"
        u"\U0001F1EB"
        u"\U0001F1F5"
        u"\U0001F1EC"
        u"\U0001F1F5"
        u"\U0001F1ED"
        u"\U0001F1F5"
        u"\U0001F1F0"
        u"\U0001F1F5"
        u"\U0001F1F1"
        u"\U0001F1F5"
        u"\U0001F1F2"
        u"\U0001F1F5"
        u"\U0001F1F3"
        u"\U0001F1F5"
        u"\U0001F1F7"
        u"\U0001F1F5"
        u"\U0001F1F8"
        u"\U0001F1F5"
        u"\U0001F1F9"
        u"\U0001F1F5"
        u"\U0001F1FC"
        u"\U0001F1F5"
        u"\U0001F1FE"
        u"\U0001F1F6"
        u"\U0001F1E6"
        u"\U0001F1F7"
        u"\U0001F1EA"
        u"\U0001F1F7"
        u"\U0001F1F4"
        u"\U0001F1F7"
        u"\U0001F1F8"
        u"\U0001F1F7"
        u"\U0001F1FA"
        u"\U0001F1F7"
        u"\U0001F1FC"
        u"\U0001F1F8"
        u"\U0001F1E6"
        u"\U0001F1F8"
        u"\U0001F1E7"
        u"\U0001F1F8"
        u"\U0001F1E8"
        u"\U0001F1F8"
        u"\U0001F1E9"
        u"\U0001F1F8"
        u"\U0001F1EA"
        u"\U0001F1F8"
        u"\U0001F1EC"
        u"\U0001F1F8"
        u"\U0001F1ED"
        u"\U0001F1F8"
        u"\U0001F1EE"
        u"\U0001F1F8"
        u"\U0001F1EF"
        u"\U0001F1F8"
        u"\U0001F1F0"
        u"\U0001F1F8"
        u"\U0001F1F1"
        u"\U0001F1F8"
        u"\U0001F1F2"
        u"\U0001F1F8"
        u"\U0001F1F3"
        u"\U0001F1F8"
        u"\U0001F1F4"
        u"\U0001F1F8"
        u"\U0001F1F7"
        u"\U0001F1F8"
        u"\U0001F1F8"
        u"\U0001F1F8"
        u"\U0001F1F9"
        u"\U0001F1F8"
        u"\U0001F1FB"
        u"\U0001F1F8"
        u"\U0001F1FD"
        u"\U0001F1F8"
        u"\U0001F1FE"
        u"\U0001F1F8"
        u"\U0001F1FF"
        u"\U0001F1F9"
        u"\U0001F1E6"
        u"\U0001F1F9"
        u"\U0001F1E8"
        u"\U0001F1F9"
        u"\U0001F1E9"
        u"\U0001F1F9"
        u"\U0001F1EB"
        u"\U0001F1F9"
        u"\U0001F1EC"
        u"\U0001F1F9"
        u"\U0001F1ED"
        u"\U0001F1F9"
        u"\U0001F1EF"
        u"\U0001F1F9"
        u"\U0001F1F0"
        u"\U0001F1F9"
        u"\U0001F1F1"
        u"\U0001F1F9"
        u"\U0001F1F2"
        u"\U0001F1F9"
        u"\U0001F1F3"
        u"\U0001F1F9"
        u"\U0001F1F4"
        u"\U0001F1F9"
        u"\U0001F1F7"
        u"\U0001F1F9"
        u"\U0001F1F9"
        u"\U0001F1F9"
        u"\U0001F1FB"
        u"\U0001F1F9"
        u"\U0001F1FC"
        u"\U0001F1F9"
        u"\U0001F1FF"
        u"\U0001F1FA"
        u"\U0001F1E6"
        u"\U0001F1FA"
        u"\U0001F1EC"
        u"\U0001F1FA"
        u"\U0001F1F2"
        u"\U0001F1FA"
        u"\U0001F1F3"
        u"\U0001F1FA"
        u"\U0001F1F8"
        u"\U0001F1FA"
        u"\U0001F1FE"
        u"\U0001F1FA"
        u"\U0001F1FF"
        u"\U0001F1FB"
        u"\U0001F1E6"
        u"\U0001F1FB"
        u"\U0001F1E8"
        u"\U0001F1FB"
        u"\U0001F1EA"
        u"\U0001F1FB"
        u"\U0001F1EC"
        u"\U0001F1FB"
        u"\U0001F1EE"
        u"\U0001F1FB"
        u"\U0001F1F3"
        u"\U0001F1FB"
        u"\U0001F1FA"
        u"\U0001F1FC"
        u"\U0001F1EB"
        u"\U0001F1FC"
        u"\U0001F1F8"
        u"\U0001F1FD"
        u"\U0001F1F0"
        u"\U0001F1FE"
        u"\U0001F1EA"
        u"\U0001F1FE"
        u"\U0001F1F9"
        u"\U0001F1FF"
        u"\U0001F1E6"
        u"\U0001F1FF"
        u"\U0001F1F2"
        u"\U0001F1FF"
        u"\U0001F1FC"
        u"\U0001F3F4"
        u"\U000E0067"
        u"\U000E0062"
        u"\U000E0065"
        u"\U000E006E"
        u"\U000E0067"
        u"\U000E007F"
        u"\U0001F3F4"
        u"\U000E0067"
        u"\U000E0062"
        u"\U000E0073"
        u"\U000E0063"
        u"\U000E0074"
        u"\U000E007F"
        u"\U0001F3F4"
        u"\U000E0067"
        u"\U000E0062"
        u"\U000E0077"
        u"\U000E006C"
        u"\U000E0073"
        u"\U000E007F"
        u"\u261D"
        u"\U0001F3FB"
        u"\u261D"
        u"\U0001F3FC"
        u"\u261D"
        u"\U0001F3FD"
        u"\u261D"
        u"\U0001F3FE"
        u"\u261D"
        u"\U0001F3FF"
        u"\u26F9"
        u"\U0001F3FB"
        u"\u26F9"
        u"\U0001F3FC"
        u"\u26F9"
        u"\U0001F3FD"
        u"\u26F9"
        u"\U0001F3FE"
        u"\u26F9"
        u"\U0001F3FF"
        u"\u270A"
        u"\U0001F3FB"
        u"\u270A"
        u"\U0001F3FC"
        u"\u270A"
        u"\U0001F3FD"
        u"\u270A"
        u"\U0001F3FE"
        u"\u270A"
        u"\U0001F3FF"
        u"\u270B"
        u"\U0001F3FB"
        u"\u270B"
        u"\U0001F3FC"
        u"\u270B"
        u"\U0001F3FD"
        u"\u270B"
        u"\U0001F3FE"
        u"\u270B"
        u"\U0001F3FF"
        u"\u270C"
        u"\U0001F3FB"
        u"\u270C"
        u"\U0001F3FC"
        u"\u270C"
        u"\U0001F3FD"
        u"\u270C"
        u"\U0001F3FE"
        u"\u270C"
        u"\U0001F3FF"
        u"\u270D"
        u"\U0001F3FB"
        u"\u270D"
        u"\U0001F3FC"
        u"\u270D"
        u"\U0001F3FD"
        u"\u270D"
        u"\U0001F3FE"
        u"\u270D"
        u"\U0001F3FF"
        u"\U0001F385"
        u"\U0001F3FB"
        u"\U0001F385"
        u"\U0001F3FC"
        u"\U0001F385"
        u"\U0001F3FD"
        u"\U0001F385"
        u"\U0001F3FE"
        u"\U0001F385"
        u"\U0001F3FF"
        u"\U0001F3C2"
        u"\U0001F3FB"
        u"\U0001F3C2"
        u"\U0001F3FC"
        u"\U0001F3C2"
        u"\U0001F3FD"
        u"\U0001F3C2"
        u"\U0001F3FE"
        u"\U0001F3C2"
        u"\U0001F3FF"
        u"\U0001F3C3"
        u"\U0001F3FB"
        u"\U0001F3C3"
        u"\U0001F3FC"
        u"\U0001F3C3"
        u"\U0001F3FD"
        u"\U0001F3C3"
        u"\U0001F3FE"
        u"\U0001F3C3"
        u"\U0001F3FF"
        u"\U0001F3C4"
        u"\U0001F3FB"
        u"\U0001F3C4"
        u"\U0001F3FC"
        u"\U0001F3C4"
        u"\U0001F3FD"
        u"\U0001F3C4"
        u"\U0001F3FE"
        u"\U0001F3C4"
        u"\U0001F3FF"
        u"\U0001F3C7"
        u"\U0001F3FB"
        u"\U0001F3C7"
        u"\U0001F3FC"
        u"\U0001F3C7"
        u"\U0001F3FD"
        u"\U0001F3C7"
        u"\U0001F3FE"
        u"\U0001F3C7"
        u"\U0001F3FF"
        u"\U0001F3CA"
        u"\U0001F3FB"
        u"\U0001F3CA"
        u"\U0001F3FC"
        u"\U0001F3CA"
        u"\U0001F3FD"
        u"\U0001F3CA"
        u"\U0001F3FE"
        u"\U0001F3CA"
        u"\U0001F3FF"
        u"\U0001F3CB"
        u"\U0001F3FB"
        u"\U0001F3CB"
        u"\U0001F3FC"
        u"\U0001F3CB"
        u"\U0001F3FD"
        u"\U0001F3CB"
        u"\U0001F3FE"
        u"\U0001F3CB"
        u"\U0001F3FF"
        u"\U0001F3CC"
        u"\U0001F3FB"
        u"\U0001F3CC"
        u"\U0001F3FC"
        u"\U0001F3CC"
        u"\U0001F3FD"
        u"\U0001F3CC"
        u"\U0001F3FE"
        u"\U0001F3CC"
        u"\U0001F3FF"
        u"\U0001F442"
        u"\U0001F3FB"
        u"\U0001F442"
        u"\U0001F3FC"
        u"\U0001F442"
        u"\U0001F3FD"
        u"\U0001F442"
        u"\U0001F3FE"
        u"\U0001F442"
        u"\U0001F3FF"
        u"\U0001F443"
        u"\U0001F3FB"
        u"\U0001F443"
        u"\U0001F3FC"
        u"\U0001F443"
        u"\U0001F3FD"
        u"\U0001F443"
        u"\U0001F3FE"
        u"\U0001F443"
        u"\U0001F3FF"
        u"\U0001F446"
        u"\U0001F3FB"
        u"\U0001F446"
        u"\U0001F3FC"
        u"\U0001F446"
        u"\U0001F3FD"
        u"\U0001F446"
        u"\U0001F3FE"
        u"\U0001F446"
        u"\U0001F3FF"
        u"\U0001F447"
        u"\U0001F3FB"
        u"\U0001F447"
        u"\U0001F3FC"
        u"\U0001F447"
        u"\U0001F3FD"
        u"\U0001F447"
        u"\U0001F3FE"
        u"\U0001F447"
        u"\U0001F3FF"
        u"\U0001F448"
        u"\U0001F3FB"
        u"\U0001F448"
        u"\U0001F3FC"
        u"\U0001F448"
        u"\U0001F3FD"
        u"\U0001F448"
        u"\U0001F3FE"
        u"\U0001F448"
        u"\U0001F3FF"
        u"\U0001F449"
        u"\U0001F3FB"
        u"\U0001F449"
        u"\U0001F3FC"
        u"\U0001F449"
        u"\U0001F3FD"
        u"\U0001F449"
        u"\U0001F3FE"
        u"\U0001F449"
        u"\U0001F3FF"
        u"\U0001F44A"
        u"\U0001F3FB"
        u"\U0001F44A"
        u"\U0001F3FC"
        u"\U0001F44A"
        u"\U0001F3FD"
        u"\U0001F44A"
        u"\U0001F3FE"
        u"\U0001F44A"
        u"\U0001F3FF"
        u"\U0001F44B"
        u"\U0001F3FB"
        u"\U0001F44B"
        u"\U0001F3FC"
        u"\U0001F44B"
        u"\U0001F3FD"
        u"\U0001F44B"
        u"\U0001F3FE"
        u"\U0001F44B"
        u"\U0001F3FF"
        u"\U0001F44C"
        u"\U0001F3FB"
        u"\U0001F44C"
        u"\U0001F3FC"
        u"\U0001F44C"
        u"\U0001F3FD"
        u"\U0001F44C"
        u"\U0001F3FE"
        u"\U0001F44C"
        u"\U0001F3FF"
        u"\U0001F44D"
        u"\U0001F3FB"
        u"\U0001F44D"
        u"\U0001F3FC"
        u"\U0001F44D"
        u"\U0001F3FD"
        u"\U0001F44D"
        u"\U0001F3FE"
        u"\U0001F44D"
        u"\U0001F3FF"
        u"\U0001F44E"
        u"\U0001F3FB"
        u"\U0001F44E"
        u"\U0001F3FC"
        u"\U0001F44E"
        u"\U0001F3FD"
        u"\U0001F44E"
        u"\U0001F3FE"
        u"\U0001F44E"
        u"\U0001F3FF"
        u"\U0001F44F"
        u"\U0001F3FB"
        u"\U0001F44F"
        u"\U0001F3FC"
        u"\U0001F44F"
        u"\U0001F3FD"
        u"\U0001F44F"
        u"\U0001F3FE"
        u"\U0001F44F"
        u"\U0001F3FF"
        u"\U0001F450"
        u"\U0001F3FB"
        u"\U0001F450"
        u"\U0001F3FC"
        u"\U0001F450"
        u"\U0001F3FD"
        u"\U0001F450"
        u"\U0001F3FE"
        u"\U0001F450"
        u"\U0001F3FF"
        u"\U0001F466"
        u"\U0001F3FB"
        u"\U0001F466"
        u"\U0001F3FC"
        u"\U0001F466"
        u"\U0001F3FD"
        u"\U0001F466"
        u"\U0001F3FE"
        u"\U0001F466"
        u"\U0001F3FF"
        u"\U0001F467"
        u"\U0001F3FB"
        u"\U0001F467"
        u"\U0001F3FC"
        u"\U0001F467"
        u"\U0001F3FD"
        u"\U0001F467"
        u"\U0001F3FE"
        u"\U0001F467"
        u"\U0001F3FF"
        u"\U0001F468"
        u"\U0001F3FB"
        u"\U0001F468"
        u"\U0001F3FC"
        u"\U0001F468"
        u"\U0001F3FD"
        u"\U0001F468"
        u"\U0001F3FE"
        u"\U0001F468"
        u"\U0001F3FF"
        u"\U0001F469"
        u"\U0001F3FB"
        u"\U0001F469"
        u"\U0001F3FC"
        u"\U0001F469"
        u"\U0001F3FD"
        u"\U0001F469"
        u"\U0001F3FE"
        u"\U0001F469"
        u"\U0001F3FF"
        u"\U0001F46B"
        u"\U0001F3FB"
        u"\U0001F46B"
        u"\U0001F3FC"
        u"\U0001F46B"
        u"\U0001F3FD"
        u"\U0001F46B"
        u"\U0001F3FE"
        u"\U0001F46B"
        u"\U0001F3FF"
        u"\U0001F46C"
        u"\U0001F3FB"
        u"\U0001F46C"
        u"\U0001F3FC"
        u"\U0001F46C"
        u"\U0001F3FD"
        u"\U0001F46C"
        u"\U0001F3FE"
        u"\U0001F46C"
        u"\U0001F3FF"
        u"\U0001F46D"
        u"\U0001F3FB"
        u"\U0001F46D"
        u"\U0001F3FC"
        u"\U0001F46D"
        u"\U0001F3FD"
        u"\U0001F46D"
        u"\U0001F3FE"
        u"\U0001F46D"
        u"\U0001F3FF"
        u"\U0001F46E"
        u"\U0001F3FB"
        u"\U0001F46E"
        u"\U0001F3FC"
        u"\U0001F46E"
        u"\U0001F3FD"
        u"\U0001F46E"
        u"\U0001F3FE"
        u"\U0001F46E"
        u"\U0001F3FF"
        u"\U0001F470"
        u"\U0001F3FB"
        u"\U0001F470"
        u"\U0001F3FC"
        u"\U0001F470"
        u"\U0001F3FD"
        u"\U0001F470"
        u"\U0001F3FE"
        u"\U0001F470"
        u"\U0001F3FF"
        u"\U0001F471"
        u"\U0001F3FB"
        u"\U0001F471"
        u"\U0001F3FC"
        u"\U0001F471"
        u"\U0001F3FD"
        u"\U0001F471"
        u"\U0001F3FE"
        u"\U0001F471"
        u"\U0001F3FF"
        u"\U0001F472"
        u"\U0001F3FB"
        u"\U0001F472"
        u"\U0001F3FC"
        u"\U0001F472"
        u"\U0001F3FD"
        u"\U0001F472"
        u"\U0001F3FE"
        u"\U0001F472"
        u"\U0001F3FF"
        u"\U0001F473"
        u"\U0001F3FB"
        u"\U0001F473"
        u"\U0001F3FC"
        u"\U0001F473"
        u"\U0001F3FD"
        u"\U0001F473"
        u"\U0001F3FE"
        u"\U0001F473"
        u"\U0001F3FF"
        u"\U0001F474"
        u"\U0001F3FB"
        u"\U0001F474"
        u"\U0001F3FC"
        u"\U0001F474"
        u"\U0001F3FD"
        u"\U0001F474"
        u"\U0001F3FE"
        u"\U0001F474"
        u"\U0001F3FF"
        u"\U0001F475"
        u"\U0001F3FB"
        u"\U0001F475"
        u"\U0001F3FC"
        u"\U0001F475"
        u"\U0001F3FD"
        u"\U0001F475"
        u"\U0001F3FE"
        u"\U0001F475"
        u"\U0001F3FF"
        u"\U0001F476"
        u"\U0001F3FB"
        u"\U0001F476"
        u"\U0001F3FC"
        u"\U0001F476"
        u"\U0001F3FD"
        u"\U0001F476"
        u"\U0001F3FE"
        u"\U0001F476"
        u"\U0001F3FF"
        u"\U0001F477"
        u"\U0001F3FB"
        u"\U0001F477"
        u"\U0001F3FC"
        u"\U0001F477"
        u"\U0001F3FD"
        u"\U0001F477"
        u"\U0001F3FE"
        u"\U0001F477"
        u"\U0001F3FF"
        u"\U0001F478"
        u"\U0001F3FB"
        u"\U0001F478"
        u"\U0001F3FC"
        u"\U0001F478"
        u"\U0001F3FD"
        u"\U0001F478"
        u"\U0001F3FE"
        u"\U0001F478"
        u"\U0001F3FF"
        u"\U0001F47C"
        u"\U0001F3FB"
        u"\U0001F47C"
        u"\U0001F3FC"
        u"\U0001F47C"
        u"\U0001F3FD"
        u"\U0001F47C"
        u"\U0001F3FE"
        u"\U0001F47C"
        u"\U0001F3FF"
        u"\U0001F481"
        u"\U0001F3FB"
        u"\U0001F481"
        u"\U0001F3FC"
        u"\U0001F481"
        u"\U0001F3FD"
        u"\U0001F481"
        u"\U0001F3FE"
        u"\U0001F481"
        u"\U0001F3FF"
        u"\U0001F482"
        u"\U0001F3FB"
        u"\U0001F482"
        u"\U0001F3FC"
        u"\U0001F482"
        u"\U0001F3FD"
        u"\U0001F482"
        u"\U0001F3FE"
        u"\U0001F482"
        u"\U0001F3FF"
        u"\U0001F483"
        u"\U0001F3FB"
        u"\U0001F483"
        u"\U0001F3FC"
        u"\U0001F483"
        u"\U0001F3FD"
        u"\U0001F483"
        u"\U0001F3FE"
        u"\U0001F483"
        u"\U0001F3FF"
        u"\U0001F485"
        u"\U0001F3FB"
        u"\U0001F485"
        u"\U0001F3FC"
        u"\U0001F485"
        u"\U0001F3FD"
        u"\U0001F485"
        u"\U0001F3FE"
        u"\U0001F485"
        u"\U0001F3FF"
        u"\U0001F486"
        u"\U0001F3FB"
        u"\U0001F486"
        u"\U0001F3FC"
        u"\U0001F486"
        u"\U0001F3FD"
        u"\U0001F486"
        u"\U0001F3FE"
        u"\U0001F486"
        u"\U0001F3FF"
        u"\U0001F487"
        u"\U0001F3FB"
        u"\U0001F487"
        u"\U0001F3FC"
        u"\U0001F487"
        u"\U0001F3FD"
        u"\U0001F487"
        u"\U0001F3FE"
        u"\U0001F487"
        u"\U0001F3FF"
        u"\U0001F48F"
        u"\U0001F3FB"
        u"\U0001F48F"
        u"\U0001F3FC"
        u"\U0001F48F"
        u"\U0001F3FD"
        u"\U0001F48F"
        u"\U0001F3FE"
        u"\U0001F48F"
        u"\U0001F3FF"
        u"\U0001F491"
        u"\U0001F3FB"
        u"\U0001F491"
        u"\U0001F3FC"
        u"\U0001F491"
        u"\U0001F3FD"
        u"\U0001F491"
        u"\U0001F3FE"
        u"\U0001F491"
        u"\U0001F3FF"
        u"\U0001F4AA"
        u"\U0001F3FB"
        u"\U0001F4AA"
        u"\U0001F3FC"
        u"\U0001F4AA"
        u"\U0001F3FD"
        u"\U0001F4AA"
        u"\U0001F3FE"
        u"\U0001F4AA"
        u"\U0001F3FF"
        u"\U0001F574"
        u"\U0001F3FB"
        u"\U0001F574"
        u"\U0001F3FC"
        u"\U0001F574"
        u"\U0001F3FD"
        u"\U0001F574"
        u"\U0001F3FE"
        u"\U0001F574"
        u"\U0001F3FF"
        u"\U0001F575"
        u"\U0001F3FB"
        u"\U0001F575"
        u"\U0001F3FC"
        u"\U0001F575"
        u"\U0001F3FD"
        u"\U0001F575"
        u"\U0001F3FE"
        u"\U0001F575"
        u"\U0001F3FF"
        u"\U0001F57A"
        u"\U0001F3FB"
        u"\U0001F57A"
        u"\U0001F3FC"
        u"\U0001F57A"
        u"\U0001F3FD"
        u"\U0001F57A"
        u"\U0001F3FE"
        u"\U0001F57A"
        u"\U0001F3FF"
        u"\U0001F590"
        u"\U0001F3FB"
        u"\U0001F590"
        u"\U0001F3FC"
        u"\U0001F590"
        u"\U0001F3FD"
        u"\U0001F590"
        u"\U0001F3FE"
        u"\U0001F590"
        u"\U0001F3FF"
        u"\U0001F595"
        u"\U0001F3FB"
        u"\U0001F595"
        u"\U0001F3FC"
        u"\U0001F595"
        u"\U0001F3FD"
        u"\U0001F595"
        u"\U0001F3FE"
        u"\U0001F595"
        u"\U0001F3FF"
        u"\U0001F596"
        u"\U0001F3FB"
        u"\U0001F596"
        u"\U0001F3FC"
        u"\U0001F596"
        u"\U0001F3FD"
        u"\U0001F596"
        u"\U0001F3FE"
        u"\U0001F596"
        u"\U0001F3FF"
        u"\U0001F645"
        u"\U0001F3FB"
        u"\U0001F645"
        u"\U0001F3FC"
        u"\U0001F645"
        u"\U0001F3FD"
        u"\U0001F645"
        u"\U0001F3FE"
        u"\U0001F645"
        u"\U0001F3FF"
        u"\U0001F646"
        u"\U0001F3FB"
        u"\U0001F646"
        u"\U0001F3FC"
        u"\U0001F646"
        u"\U0001F3FD"
        u"\U0001F646"
        u"\U0001F3FE"
        u"\U0001F646"
        u"\U0001F3FF"
        u"\U0001F647"
        u"\U0001F3FB"
        u"\U0001F647"
        u"\U0001F3FC"
        u"\U0001F647"
        u"\U0001F3FD"
        u"\U0001F647"
        u"\U0001F3FE"
        u"\U0001F647"
        u"\U0001F3FF"
        u"\U0001F64B"
        u"\U0001F3FB"
        u"\U0001F64B"
        u"\U0001F3FC"
        u"\U0001F64B"
        u"\U0001F3FD"
        u"\U0001F64B"
        u"\U0001F3FE"
        u"\U0001F64B"
        u"\U0001F3FF"
        u"\U0001F64C"
        u"\U0001F3FB"
        u"\U0001F64C"
        u"\U0001F3FC"
        u"\U0001F64C"
        u"\U0001F3FD"
        u"\U0001F64C"
        u"\U0001F3FE"
        u"\U0001F64C"
        u"\U0001F3FF"
        u"\U0001F64D"
        u"\U0001F3FB"
        u"\U0001F64D"
        u"\U0001F3FC"
        u"\U0001F64D"
        u"\U0001F3FD"
        u"\U0001F64D"
        u"\U0001F3FE"
        u"\U0001F64D"
        u"\U0001F3FF"
        u"\U0001F64E"
        u"\U0001F3FB"
        u"\U0001F64E"
        u"\U0001F3FC"
        u"\U0001F64E"
        u"\U0001F3FD"
        u"\U0001F64E"
        u"\U0001F3FE"
        u"\U0001F64E"
        u"\U0001F3FF"
        u"\U0001F64F"
        u"\U0001F3FB"
        u"\U0001F64F"
        u"\U0001F3FC"
        u"\U0001F64F"
        u"\U0001F3FD"
        u"\U0001F64F"
        u"\U0001F3FE"
        u"\U0001F64F"
        u"\U0001F3FF"
        u"\U0001F6A3"
        u"\U0001F3FB"
        u"\U0001F6A3"
        u"\U0001F3FC"
        u"\U0001F6A3"
        u"\U0001F3FD"
        u"\U0001F6A3"
        u"\U0001F3FE"
        u"\U0001F6A3"
        u"\U0001F3FF"
        u"\U0001F6B4"
        u"\U0001F3FB"
        u"\U0001F6B4"
        u"\U0001F3FC"
        u"\U0001F6B4"
        u"\U0001F3FD"
        u"\U0001F6B4"
        u"\U0001F3FE"
        u"\U0001F6B4"
        u"\U0001F3FF"
        u"\U0001F6B5"
        u"\U0001F3FB"
        u"\U0001F6B5"
        u"\U0001F3FC"
        u"\U0001F6B5"
        u"\U0001F3FD"
        u"\U0001F6B5"
        u"\U0001F3FE"
        u"\U0001F6B5"
        u"\U0001F3FF"
        u"\U0001F6B6"
        u"\U0001F3FB"
        u"\U0001F6B6"
        u"\U0001F3FC"
        u"\U0001F6B6"
        u"\U0001F3FD"
        u"\U0001F6B6"
        u"\U0001F3FE"
        u"\U0001F6B6"
        u"\U0001F3FF"
        u"\U0001F6C0"
        u"\U0001F3FB"
        u"\U0001F6C0"
        u"\U0001F3FC"
        u"\U0001F6C0"
        u"\U0001F3FD"
        u"\U0001F6C0"
        u"\U0001F3FE"
        u"\U0001F6C0"
        u"\U0001F3FF"
        u"\U0001F6CC"
        u"\U0001F3FB"
        u"\U0001F6CC"
        u"\U0001F3FC"
        u"\U0001F6CC"
        u"\U0001F3FD"
        u"\U0001F6CC"
        u"\U0001F3FE"
        u"\U0001F6CC"
        u"\U0001F3FF"
        u"\U0001F90C"
        u"\U0001F3FB"
        u"\U0001F90C"
        u"\U0001F3FC"
        u"\U0001F90C"
        u"\U0001F3FD"
        u"\U0001F90C"
        u"\U0001F3FE"
        u"\U0001F90C"
        u"\U0001F3FF"
        u"\U0001F90F"
        u"\U0001F3FB"
        u"\U0001F90F"
        u"\U0001F3FC"
        u"\U0001F90F"
        u"\U0001F3FD"
        u"\U0001F90F"
        u"\U0001F3FE"
        u"\U0001F90F"
        u"\U0001F3FF"
        u"\U0001F918"
        u"\U0001F3FB"
        u"\U0001F918"
        u"\U0001F3FC"
        u"\U0001F918"
        u"\U0001F3FD"
        u"\U0001F918"
        u"\U0001F3FE"
        u"\U0001F918"
        u"\U0001F3FF"
        u"\U0001F919"
        u"\U0001F3FB"
        u"\U0001F919"
        u"\U0001F3FC"
        u"\U0001F919"
        u"\U0001F3FD"
        u"\U0001F919"
        u"\U0001F3FE"
        u"\U0001F919"
        u"\U0001F3FF"
        u"\U0001F91A"
        u"\U0001F3FB"
        u"\U0001F91A"
        u"\U0001F3FC"
        u"\U0001F91A"
        u"\U0001F3FD"
        u"\U0001F91A"
        u"\U0001F3FE"
        u"\U0001F91A"
        u"\U0001F3FF"
        u"\U0001F91B"
        u"\U0001F3FB"
        u"\U0001F91B"
        u"\U0001F3FC"
        u"\U0001F91B"
        u"\U0001F3FD"
        u"\U0001F91B"
        u"\U0001F3FE"
        u"\U0001F91B"
        u"\U0001F3FF"
        u"\U0001F91C"
        u"\U0001F3FB"
        u"\U0001F91C"
        u"\U0001F3FC"
        u"\U0001F91C"
        u"\U0001F3FD"
        u"\U0001F91C"
        u"\U0001F3FE"
        u"\U0001F91C"
        u"\U0001F3FF"
        u"\U0001F91D"
        u"\U0001F3FB"
        u"\U0001F91D"
        u"\U0001F3FC"
        u"\U0001F91D"
        u"\U0001F3FD"
        u"\U0001F91D"
        u"\U0001F3FE"
        u"\U0001F91D"
        u"\U0001F3FF"
        u"\U0001F91E"
        u"\U0001F3FB"
        u"\U0001F91E"
        u"\U0001F3FC"
        u"\U0001F91E"
        u"\U0001F3FD"
        u"\U0001F91E"
        u"\U0001F3FE"
        u"\U0001F91E"
        u"\U0001F3FF"
        u"\U0001F91F"
        u"\U0001F3FB"
        u"\U0001F91F"
        u"\U0001F3FC"
        u"\U0001F91F"
        u"\U0001F3FD"
        u"\U0001F91F"
        u"\U0001F3FE"
        u"\U0001F91F"
        u"\U0001F3FF"
        u"\U0001F926"
        u"\U0001F3FB"
        u"\U0001F926"
        u"\U0001F3FC"
        u"\U0001F926"
        u"\U0001F3FD"
        u"\U0001F926"
        u"\U0001F3FE"
        u"\U0001F926"
        u"\U0001F3FF"
        u"\U0001F930"
        u"\U0001F3FB"
        u"\U0001F930"
        u"\U0001F3FC"
        u"\U0001F930"
        u"\U0001F3FD"
        u"\U0001F930"
        u"\U0001F3FE"
        u"\U0001F930"
        u"\U0001F3FF"
        u"\U0001F931"
        u"\U0001F3FB"
        u"\U0001F931"
        u"\U0001F3FC"
        u"\U0001F931"
        u"\U0001F3FD"
        u"\U0001F931"
        u"\U0001F3FE"
        u"\U0001F931"
        u"\U0001F3FF"
        u"\U0001F932"
        u"\U0001F3FB"
        u"\U0001F932"
        u"\U0001F3FC"
        u"\U0001F932"
        u"\U0001F3FD"
        u"\U0001F932"
        u"\U0001F3FE"
        u"\U0001F932"
        u"\U0001F3FF"
        u"\U0001F933"
        u"\U0001F3FB"
        u"\U0001F933"
        u"\U0001F3FC"
        u"\U0001F933"
        u"\U0001F3FD"
        u"\U0001F933"
        u"\U0001F3FE"
        u"\U0001F933"
        u"\U0001F3FF"
        u"\U0001F934"
        u"\U0001F3FB"
        u"\U0001F934"
        u"\U0001F3FC"
        u"\U0001F934"
        u"\U0001F3FD"
        u"\U0001F934"
        u"\U0001F3FE"
        u"\U0001F934"
        u"\U0001F3FF"
        u"\U0001F935"
        u"\U0001F3FB"
        u"\U0001F935"
        u"\U0001F3FC"
        u"\U0001F935"
        u"\U0001F3FD"
        u"\U0001F935"
        u"\U0001F3FE"
        u"\U0001F935"
        u"\U0001F3FF"
        u"\U0001F936"
        u"\U0001F3FB"
        u"\U0001F936"
        u"\U0001F3FC"
        u"\U0001F936"
        u"\U0001F3FD"
        u"\U0001F936"
        u"\U0001F3FE"
        u"\U0001F936"
        u"\U0001F3FF"
        u"\U0001F937"
        u"\U0001F3FB"
        u"\U0001F937"
        u"\U0001F3FC"
        u"\U0001F937"
        u"\U0001F3FD"
        u"\U0001F937"
        u"\U0001F3FE"
        u"\U0001F937"
        u"\U0001F3FF"
        u"\U0001F938"
        u"\U0001F3FB"
        u"\U0001F938"
        u"\U0001F3FC"
        u"\U0001F938"
        u"\U0001F3FD"
        u"\U0001F938"
        u"\U0001F3FE"
        u"\U0001F938"
        u"\U0001F3FF"
        u"\U0001F939"
        u"\U0001F3FB"
        u"\U0001F939"
        u"\U0001F3FC"
        u"\U0001F939"
        u"\U0001F3FD"
        u"\U0001F939"
        u"\U0001F3FE"
        u"\U0001F939"
        u"\U0001F3FF"
        u"\U0001F93D"
        u"\U0001F3FB"
        u"\U0001F93D"
        u"\U0001F3FC"
        u"\U0001F93D"
        u"\U0001F3FD"
        u"\U0001F93D"
        u"\U0001F3FE"
        u"\U0001F93D"
        u"\U0001F3FF"
        u"\U0001F93E"
        u"\U0001F3FB"
        u"\U0001F93E"
        u"\U0001F3FC"
        u"\U0001F93E"
        u"\U0001F3FD"
        u"\U0001F93E"
        u"\U0001F3FE"
        u"\U0001F93E"
        u"\U0001F3FF"
        u"\U0001F977"
        u"\U0001F3FB"
        u"\U0001F977"
        u"\U0001F3FC"
        u"\U0001F977"
        u"\U0001F3FD"
        u"\U0001F977"
        u"\U0001F3FE"
        u"\U0001F977"
        u"\U0001F3FF"
        u"\U0001F9B5"
        u"\U0001F3FB"
        u"\U0001F9B5"
        u"\U0001F3FC"
        u"\U0001F9B5"
        u"\U0001F3FD"
        u"\U0001F9B5"
        u"\U0001F3FE"
        u"\U0001F9B5"
        u"\U0001F3FF"
        u"\U0001F9B6"
        u"\U0001F3FB"
        u"\U0001F9B6"
        u"\U0001F3FC"
        u"\U0001F9B6"
        u"\U0001F3FD"
        u"\U0001F9B6"
        u"\U0001F3FE"
        u"\U0001F9B6"
        u"\U0001F3FF"
        u"\U0001F9B8"
        u"\U0001F3FB"
        u"\U0001F9B8"
        u"\U0001F3FC"
        u"\U0001F9B8"
        u"\U0001F3FD"
        u"\U0001F9B8"
        u"\U0001F3FE"
        u"\U0001F9B8"
        u"\U0001F3FF"
        u"\U0001F9B9"
        u"\U0001F3FB"
        u"\U0001F9B9"
        u"\U0001F3FC"
        u"\U0001F9B9"
        u"\U0001F3FD"
        u"\U0001F9B9"
        u"\U0001F3FE"
        u"\U0001F9B9"
        u"\U0001F3FF"
        u"\U0001F9BB"
        u"\U0001F3FB"
        u"\U0001F9BB"
        u"\U0001F3FC"
        u"\U0001F9BB"
        u"\U0001F3FD"
        u"\U0001F9BB"
        u"\U0001F3FE"
        u"\U0001F9BB"
        u"\U0001F3FF"
        u"\U0001F9CD"
        u"\U0001F3FB"
        u"\U0001F9CD"
        u"\U0001F3FC"
        u"\U0001F9CD"
        u"\U0001F3FD"
        u"\U0001F9CD"
        u"\U0001F3FE"
        u"\U0001F9CD"
        u"\U0001F3FF"
        u"\U0001F9CE"
        u"\U0001F3FB"
        u"\U0001F9CE"
        u"\U0001F3FC"
        u"\U0001F9CE"
        u"\U0001F3FD"
        u"\U0001F9CE"
        u"\U0001F3FE"
        u"\U0001F9CE"
        u"\U0001F3FF"
        u"\U0001F9CF"
        u"\U0001F3FB"
        u"\U0001F9CF"
        u"\U0001F3FC"
        u"\U0001F9CF"
        u"\U0001F3FD"
        u"\U0001F9CF"
        u"\U0001F3FE"
        u"\U0001F9CF"
        u"\U0001F3FF"
        u"\U0001F9D1"
        u"\U0001F3FB"
        u"\U0001F9D1"
        u"\U0001F3FC"
        u"\U0001F9D1"
        u"\U0001F3FD"
        u"\U0001F9D1"
        u"\U0001F3FE"
        u"\U0001F9D1"
        u"\U0001F3FF"
        u"\U0001F9D2"
        u"\U0001F3FB"
        u"\U0001F9D2"
        u"\U0001F3FC"
        u"\U0001F9D2"
        u"\U0001F3FD"
        u"\U0001F9D2"
        u"\U0001F3FE"
        u"\U0001F9D2"
        u"\U0001F3FF"
        u"\U0001F9D3"
        u"\U0001F3FB"
        u"\U0001F9D3"
        u"\U0001F3FC"
        u"\U0001F9D3"
        u"\U0001F3FD"
        u"\U0001F9D3"
        u"\U0001F3FE"
        u"\U0001F9D3"
        u"\U0001F3FF"
        u"\U0001F9D4"
        u"\U0001F3FB"
        u"\U0001F9D4"
        u"\U0001F3FC"
        u"\U0001F9D4"
        u"\U0001F3FD"
        u"\U0001F9D4"
        u"\U0001F3FE"
        u"\U0001F9D4"
        u"\U0001F3FF"
        u"\U0001F9D5"
        u"\U0001F3FB"
        u"\U0001F9D5"
        u"\U0001F3FC"
        u"\U0001F9D5"
        u"\U0001F3FD"
        u"\U0001F9D5"
        u"\U0001F3FE"
        u"\U0001F9D5"
        u"\U0001F3FF"
        u"\U0001F9D6"
        u"\U0001F3FB"
        u"\U0001F9D6"
        u"\U0001F3FC"
        u"\U0001F9D6"
        u"\U0001F3FD"
        u"\U0001F9D6"
        u"\U0001F3FE"
        u"\U0001F9D6"
        u"\U0001F3FF"
        u"\U0001F9D7"
        u"\U0001F3FB"
        u"\U0001F9D7"
        u"\U0001F3FC"
        u"\U0001F9D7"
        u"\U0001F3FD"
        u"\U0001F9D7"
        u"\U0001F3FE"
        u"\U0001F9D7"
        u"\U0001F3FF"
        u"\U0001F9D8"
        u"\U0001F3FB"
        u"\U0001F9D8"
        u"\U0001F3FC"
        u"\U0001F9D8"
        u"\U0001F3FD"
        u"\U0001F9D8"
        u"\U0001F3FE"
        u"\U0001F9D8"
        u"\U0001F3FF"
        u"\U0001F9D9"
        u"\U0001F3FB"
        u"\U0001F9D9"
        u"\U0001F3FC"
        u"\U0001F9D9"
        u"\U0001F3FD"
        u"\U0001F9D9"
        u"\U0001F3FE"
        u"\U0001F9D9"
        u"\U0001F3FF"
        u"\U0001F9DA"
        u"\U0001F3FB"
        u"\U0001F9DA"
        u"\U0001F3FC"
        u"\U0001F9DA"
        u"\U0001F3FD"
        u"\U0001F9DA"
        u"\U0001F3FE"
        u"\U0001F9DA"
        u"\U0001F3FF"
        u"\U0001F9DB"
        u"\U0001F3FB"
        u"\U0001F9DB"
        u"\U0001F3FC"
        u"\U0001F9DB"
        u"\U0001F3FD"
        u"\U0001F9DB"
        u"\U0001F3FE"
        u"\U0001F9DB"
        u"\U0001F3FF"
        u"\U0001F9DC"
        u"\U0001F3FB"
        u"\U0001F9DC"
        u"\U0001F3FC"
        u"\U0001F9DC"
        u"\U0001F3FD"
        u"\U0001F9DC"
        u"\U0001F3FE"
        u"\U0001F9DC"
        u"\U0001F3FF"
        u"\U0001F9DD"
        u"\U0001F3FB"
        u"\U0001F9DD"
        u"\U0001F3FC"
        u"\U0001F9DD"
        u"\U0001F3FD"
        u"\U0001F9DD"
        u"\U0001F3FE"
        u"\U0001F9DD"
        u"\U0001F3FF"
        u"\U0001FAC3"
        u"\U0001F3FB"
        u"\U0001FAC3"
        u"\U0001F3FC"
        u"\U0001FAC3"
        u"\U0001F3FD"
        u"\U0001FAC3"
        u"\U0001F3FE"
        u"\U0001FAC3"
        u"\U0001F3FF"
        u"\U0001FAC4"
        u"\U0001F3FB"
        u"\U0001FAC4"
        u"\U0001F3FC"
        u"\U0001FAC4"
        u"\U0001F3FD"
        u"\U0001FAC4"
        u"\U0001F3FE"
        u"\U0001FAC4"
        u"\U0001F3FF"
        u"\U0001FAC5"
        u"\U0001F3FB"
        u"\U0001FAC5"
        u"\U0001F3FC"
        u"\U0001FAC5"
        u"\U0001F3FD"
        u"\U0001FAC5"
        u"\U0001F3FE"
        u"\U0001FAC5"
        u"\U0001F3FF"
        u"\U0001FAF0"
        u"\U0001F3FB"
        u"\U0001FAF0"
        u"\U0001F3FC"
        u"\U0001FAF0"
        u"\U0001F3FD"
        u"\U0001FAF0"
        u"\U0001F3FE"
        u"\U0001FAF0"
        u"\U0001F3FF"
        u"\U0001FAF1"
        u"\U0001F3FB"
        u"\U0001FAF1"
        u"\U0001F3FC"
        u"\U0001FAF1"
        u"\U0001F3FD"
        u"\U0001FAF1"
        u"\U0001F3FE"
        u"\U0001FAF1"
        u"\U0001F3FF"
        u"\U0001FAF2"
        u"\U0001F3FB"
        u"\U0001FAF2"
        u"\U0001F3FC"
        u"\U0001FAF2"
        u"\U0001F3FD"
        u"\U0001FAF2"
        u"\U0001F3FE"
        u"\U0001FAF2"
        u"\U0001F3FF"
        u"\U0001FAF3"
        u"\U0001F3FB"
        u"\U0001FAF3"
        u"\U0001F3FC"
        u"\U0001FAF3"
        u"\U0001F3FD"
        u"\U0001FAF3"
        u"\U0001F3FE"
        u"\U0001FAF3"
        u"\U0001F3FF"
        u"\U0001FAF4"
        u"\U0001F3FB"
        u"\U0001FAF4"
        u"\U0001F3FC"
        u"\U0001FAF4"
        u"\U0001F3FD"
        u"\U0001FAF4"
        u"\U0001F3FE"
        u"\U0001FAF4"
        u"\U0001F3FF"
        u"\U0001FAF5"
        u"\U0001F3FB"
        u"\U0001FAF5"
        u"\U0001F3FC"
        u"\U0001FAF5"
        u"\U0001F3FD"
        u"\U0001FAF5"
        u"\U0001F3FE"
        u"\U0001FAF5"
        u"\U0001F3FF"
        u"\U0001FAF6"
        u"\U0001F3FB"
        u"\U0001FAF6"
        u"\U0001F3FC"
        u"\U0001FAF6"
        u"\U0001F3FD"
        u"\U0001FAF6"
        u"\U0001F3FE"
        u"\U0001FAF6"
        u"\U0001F3FF"
        "]+", flags = re.UNICODE
    )
    return emoji_pattern.sub(r'',text)

#------------------------------------------------------------------------------
# PDPA (compact)
#------------------------------------------------------------------------------
ENTITY_TO_ANONYMIZED_TOKEN_MAP = {
    "PERSON":       "[PERSON]",
    "PHONE":        "[PHONE]",
    "EMAIL":        "[EMAIL]",
    "ADDRESS":      "[LOCATION]",
    "NATIONAL_ID":  "[NATIONAL_ID]",
    "HOSPITAL_IDS": "[HOSPITAL_IDS]",
}


# ---------------------------------------------------------------------
# 2) โครงสร้าง span + ตัวช่วย
# ---------------------------------------------------------------------
@dataclass
class _Span:
    start: int
    end: int
    text: str
    cat: str  # PERSON/ADDRESS/DATE/PHONE/EMAIL/NATIONAL_ID/HOSPITAL_IDS

def _resolve_overlaps(spans: list[_Span]) -> list[_Span]:
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept, last_end = [], -1
    for s in spans:
        if s.start >= last_end:
            kept.append(s)
            last_end = s.end
    return kept

def _replace_with_tokens(text: str, spans: list[_Span], token_map: dict[str, str]) -> str:
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        token = token_map.get(s.cat, f"[{s.cat}]")
        text = text[:s.start] + token + text[s.end:]
    return text

def _merge_adjacent_person_spans(spans: list[_Span], text: str) -> list[_Span]:
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    out = []
    i = 0
    while i < len(spans):
        s = spans[i]
        if s.cat != "PERSON":
            out.append(s); i += 1; continue
        j = i + 1
        end = s.end
        while j < len(spans) and spans[j].cat == "PERSON":
            gap = text[end:spans[j].start]
            # รวมเมื่อชิดกัน หรือคั่นด้วยช่องว่าง/ zero-width /ขีดสั้นยาว
            if gap == "" or gap.isspace() or re.fullmatch(r"[\u200B\u200C\u200D\u00A0\-\u2010-\u2015]*", gap):
                end = spans[j].end
                j += 1
            else:
                break
        if j > i + 1:
            merged_text = text[s.start:end]
            out.append(_Span(s.start, end, merged_text, "PERSON"))
            i = j
        else:
            out.append(s); i += 1
    return out

# ---------------------------------------------------------------------
# 3) NER + REGEX detector
# ---------------------------------------------------------------------
_NER2CAT = {
    "PER": "PERSON", "PERSON": "PERSON",
    "LOC": "ADDRESS", "LOCATION": "ADDRESS", "GPE": "ADDRESS", "ADDR": "ADDRESS",
}

EMAIL_RX  = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RX  = re.compile(r"(?:\+?66[-\s]?)?0?\d(?:[-\s]?\d){7,9}")
TH_ID_RX  = re.compile(r"\b\d{1}[- ]?\d{4}[- ]?\d{5}[- ]?\d{2}[- ]?\d{1}\b|\b\d{13}\b")
HOSPITAL_ID_RX = re.compile(r"\b(?:HN|AN)[- ]?\d{6,10}\b", re.IGNORECASE)

def _spans_from_ner(text: str, enabled: set[str]) -> list[_Span]:
    spans: list[_Span] = []
    if not enabled & {"PERSON", "ADDRESS", "DATE"}:
        return spans
    ents = _load_thai_ner()(text)
    for e in ents:
        grp = (e.get("entity_group") or "").upper()
        cat = _NER2CAT.get(grp)
        if cat and cat in enabled:
            s, t = int(e["start"]), int(e["end"])
            spans.append(_Span(s, t, text[s:t], cat))
    return spans

def _spans_from_regex(text: str, enabled: set[str]) -> list[_Span]:
    spans: list[_Span] = []
    def add(rx, cat):
        for m in rx.finditer(text):
            spans.append(_Span(m.start(), m.end(), m.group(0), cat))
    if "EMAIL" in enabled:        add(EMAIL_RX, "EMAIL")
    if "PHONE" in enabled:        add(PHONE_RX, "PHONE")
    if "NATIONAL_ID" in enabled:  add(TH_ID_RX, "NATIONAL_ID")
    if "HOSPITAL_IDS" in enabled: add(HOSPITAL_ID_RX, "HOSPITAL_IDS")
    return spans

# ---------------------------------------------------------------------
# 4) ตัวช่วย anonymize (ใช้ยูทิลเดิมเมื่อมี, fallback เมื่อไม่มี)
# ---------------------------------------------------------------------
def _hash_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()

def _digits_from_hash(seed: str, n: int) -> str:
    d = _hash_bytes(seed)
    out = []
    i = 0
    while len(out) < n:
        for b in d:
            out.append(str(b % 10))
            if len(out) == n:
                break
        i += 1
        d = _hash_bytes(seed + f"#{i}")
    return "".join(out)

def _preserve_separators(original: str, digits: str) -> str:
    # คงตำแหน่งขีด/ช่องว่างของต้นฉบับ
    clean = re.sub(r"\D", "", original)
    it = iter(digits)
    res = []
    idx = 0
    for ch in original:
        if ch.isdigit():
            res.append(next(it))
            idx += 1
        else:
            res.append(ch)
    # ถ้า original มีขีด/เว้นวรรค แต่ digits สั้นกว่า ให้เติมเลขที่เหลือท้าย
    remain = "".join(list(it))
    if remain:
        res.append(remain)
    return "".join(res)


def _anonymize_hospital_id(text: str, salt: str | None) -> str:
    def _rep(m):
        raw = m.group(0)
        prefix = "HN" if raw.upper().startswith("HN") else "AN"
        body = re.sub(r"\D", "", raw)
        syn = _digits_from_hash((salt or "") + body, len(body))
        # คงรูปแบบ prefix+digits (ไม่ใส่ขีดเพื่อความง่าย)
        return f"{prefix}{syn}"
    return HOSPITAL_ID_RX.sub(_rep, text)

# ---------------------------------------------------------------------
# 5) ฟังก์ชันหลัก: thai_ner_full_apply(policy="mask"/"anonymize")
# ---------------------------------------------------------------------
_ORIG_thai_ner_full_apply = globals().get("thai_ner_full_apply", None)

def thai_ner_full_apply(
    text: str,
    policy: str = "mask",  # "mask" | "anonymize"
    keep_categories: str = "PERSON,PHONE,EMAIL,ADDRESS,NATIONAL_ID,HOSPITAL_IDS",
    token_map: dict | None = None,              # ใช้เมื่อ policy="mask"
    salt: str | None = None,                    # ใช้สร้างค่าแบบ deterministic ถ้าฟังก์ชันภายในรองรับ
    return_spans: bool = False,
    public_figure_skip: bool = True,            # ใหม่: ข้ามชื่อบุคคลสาธารณะ (Thai)
):
    """
    ใช้ Thai NER + regex ครอบคลุม: PERSON, PHONE, EMAIL, ADDRESS, NATIONAL_ID, HOSPITAL_IDS

    - mask: แทนด้วย token_map (หรือ ENTITY_TO_ANONYMIZED_TOKEN_MAP) แล้ว 'มาส์กชื่อโรมัน' ต่อ
    - anonymize: แทนค่าจริงแบบย้อนกลับไม่ได้ + จัดการชื่อโรมันต่อ

    หมายเหตุ: ต้องมี is_public_figure(...) / configure_public_figure_paths(...) ถูกตั้งค่าก่อนรัน
    """
    def _unify_person_label(cat: str) -> str:
        c = (cat or "").upper()
        return "PERSON" if c in {"PERSON","PER","B-PER","I-PER","B-PERSON","I-PERSON"} else c

    PERSON_LABELS = {"PERSON","PER","B-PER","I-PER","B-PERSON","I-PERSON"}
    enabled = {x.strip().upper() for x in keep_categories.split(",") if x.strip()}

    # 1) หา spans จาก NER + regex
    spans = []
    try:
        spans += _spans_from_ner(text, enabled)
    except Exception:
        pass
    spans += _spans_from_regex(text, enabled)
    try:
        # ปรับ label ให้เป็น PERSON เดียวกัน และแก้ overlap
        for s in spans:
            if hasattr(s, "cat"):
                s.cat = _unify_person_label(getattr(s, "cat", ""))
            elif isinstance(s, dict):
                s["cat"] = _unify_person_label(s.get("cat") or s.get("type") or s.get("label", ""))
        spans = _resolve_overlaps(spans)
        spans = _merge_adjacent_person_spans(spans, text)  # รวม PERSON ที่ติดๆ กันก่อน
    except Exception:
        spans = sorted(spans, key=lambda s: (getattr(s, "start", 0), getattr(s, "end", 0)))

    # 2) เก็บชื่อ PERSON (ไทย) ที่เจอในข้อความนี้ไว้ก่อน (เพื่อทำ roman alias ต่อ)
    _persons_in_text = [
        (getattr(s, "text", "") or (s.get("text","") if isinstance(s, dict) else ""))
        for s in spans
        if ((getattr(s,"cat",None) or getattr(s,"type",None) or getattr(s,"label","")).upper() in PERSON_LABELS)
    ]

    # 3) ถ้าขอข้าม “บุคคลสาธารณะ (ไทย)” ให้กรองสแปน PERSON ออกก่อน
    if public_figure_skip:
        _filtered = []
        for s in spans:
            cat = (getattr(s,"cat",None) or getattr(s,"type",None) or getattr(s,"label","")).upper()
            if cat in PERSON_LABELS:
                name_txt = getattr(s,"text","") or (s.get("text","") if isinstance(s,dict) else "")
                try:
                    if is_public_figure(name_txt):
                        continue
                except Exception:
                    pass
            _filtered.append(s)
        spans = _filtered

    # 4) สร้าง roman alias skip-set สำหรับคนดังที่ “พบในข้อความนี้”
    roman_skip_set = _build_local_public_figure_roman_aliases(_persons_in_text)

    # 5) เตรียม mapping ไทย->โรมัน จากชื่อไทยที่ยังเหลือใน spans (ช่วย downstream)
    try:
        if any(lbl in enabled for lbl in ("PERSON","PER")):
            thai_person_names = [getattr(s, "text", "") or (s.get("text","") if isinstance(s, dict) else "")
                                 for s in spans
                                 if ((getattr(s,"cat",None) or getattr(s,"type",None) or getattr(s,"label","")).upper() in PERSON_LABELS)]
            if thai_person_names and callable(globals().get("register_romanized_names")):
                register_romanized_names(tuple(thai_person_names))
    except Exception:
        pass

    # 6) ปรับผลลัพธ์ตาม policy
    if policy == "mask":
        tmap = token_map or globals().get("ENTITY_TO_ANONYMIZED_TOKEN_MAP", {})
        out = _replace_with_tokens(text, spans, tmap)
        # มาส์ก “ชื่อโรมัน” เพิ่ม แต่เว้นชื่อโรมันของคนดังที่เจอในข้อความนี้
        try:
            person_token = tmap.get("PERSON", "[PERSON]") if isinstance(tmap, dict) else "[PERSON]"
            out = mask_romanized_person_names(out, threshold=0.5, token=person_token,
                                              skip_roman_set=roman_skip_set)
        except Exception:
            pass
        try:
            out = re.sub(r"\]\s*(?=\[)", "] ", out)  # fix token sticking
            out = re.sub(r"\](?=[ก-๙A-Za-z0-9])", "] ", out)
        except Exception:
            pass

    elif policy == "anonymize":
        out = text
        if ("PERSON" in enabled) or ("PER" in enabled):
            # มี PERSON spans ไหม?
            person_spans = [s for s in spans if (getattr(s, "cat", None) or getattr(s, "type", None) or getattr(s, "label", "")).upper() in {"PERSON","PER","B-PER","I-PER","B-PERSON","I-PERSON"}]
            try:
                if person_spans:
                    out = clean_name(out, spans=spans, public_figure_skip=public_figure_skip)
                else:
                    # ไม่มี PERSON -> บังคับใช้ regex fallback เหมือนเวอร์ชันเก่า
                    out = clean_name(out, spans=None, public_figure_skip=public_figure_skip)
            except Exception:
                pass
            try:
                out = anonymize_romanized_person_names(out, threshold=0.5)
            except Exception:
                pass
        if "EMAIL" in enabled:
            out = clean_email(out)
        if "PHONE" in enabled:
            out = clean_phone(out)
        if "NATIONAL_ID" in enabled:
            out = clean_national_id(out, salt)
        if "HOSPITAL_IDS" in enabled:
            out = _anonymize_hospital_id(out, salt)
        if "ADDRESS" in enabled:
            try:
                # ใช้ anonymize จริง พร้อมเก็บ province เดิม (ปรับตามต้องการ)
                out = address_anonymize(out, gazetteer_path=None, salt=salt, keep_province=True)
            except Exception:
                # fallback เผื่อ resource ไม่พร้อม
                out = _replace_with_tokens(
                    out, [s for s in spans if getattr(s,"cat","") == "ADDRESS"],
                    globals().get("ENTITY_TO_ANONYMIZED_TOKEN_MAP", {}))
    else:
        raise ValueError("policy must be 'mask' or 'anonymize'")

    if not return_spans:
        return out

    # (option) คืน spans ที่ใช้จริง
    found = {}
    for s in spans:
        cat = (getattr(s,"cat",None) or getattr(s,"type",None) or getattr(s,"label","")).upper()
        txt = getattr(s,"text","") or (s.get("text","") if isinstance(s,dict) else "")
        found.setdefault(cat, []).append(txt)
    return out, found

#------------------------------------------------------------------------------
# Personal Information Removal (Basic)
#------------------------------------------------------------------------------

# ตัวคั่นที่มักเจอ: เว้นวรรค, ขีดหลายชนิด, จุด, zero-width
_PHONE_SEP = r"[\s\-\.\u2010-\u2015\u2212\u00A0\u200B\u200C\u200D]*"

# ผู้ต้องสงสัย: (+66)? หรือ 0 นำหน้า แล้วตามด้วยกลุ่มเลขที่มีตัวคั่นแทรกได้
# ใช้ callback เพื่อตรวจ “นับจำนวนหลักจริง” ลด false positive
_PHONE_CANDIDATE = re.compile(
    rf"""(?<!\d)
        (?:\+?66{_PHONE_SEP})?           # ประเทศ
        (?:\(?0\)?{_PHONE_SEP})?         # ศูนย์นำหน้า (บางเคสหลัง +66 จะไม่มี)
        (?:\d{{1,4}}\)?{_PHONE_SEP})?    # กลุ่มหน้าสุด (พื้นที่/โอเปอเรเตอร์)
        \d{{2,4}}{_PHONE_SEP}\d{{3}}{_PHONE_SEP}\d{{3,4}}  # กลุ่มหลัก ๆ 3-3/4
        (?:{_PHONE_SEP}(?:x|ext\.?|[#]){_PHONE_SEP}\d{{1,5}})? # ต่อ
        (?!\d)
    """,
    re.VERBOSE | re.UNICODE,
)


def _digits_only_phone(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

def _looks_like_th_phone(digits: str) -> bool:
    """เงื่อนไขหยาบ ๆ: 9–12 หลัก และขึ้นต้นด้วย 0 หรือ 66"""
    if len(digits) < 9 or len(digits) > 12:
        return False
    return digits.startswith("0") or digits.startswith("66")

def _rand_string(length=6, alphabet=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(alphabet) for _ in range(length))

def remove_emails(text):
    """
    Replace email addresses with <email> placeholder.
    
    Args:
        text (str): Text containing email addresses
        
    Returns:
        str: Text with emails replaced by placeholder
    """
    domains = ["example.com", "mail.com", "test.org", "demo.net"]
    replace = f"{_rand_string()}@{random.choice(domains)}"
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', replace, text)
    return text

def _gen_th_mobile10() -> str:
    """สุ่มเบอร์มือถือไทย 10 หลัก (06/08/09)"""
    pfx = random.choice(["06", "08", "09"])
    tail = "".join(random.choice(string.digits) for _ in range(8))
    return pfx + tail

def _format_like(src: str, new_digits: str) -> str:
    """
    จัดรูปเลขสุ่มให้ 'เลียนแบบลวดลายตัวคั่น' ของต้นฉบับ
    เช่น 091-234-5678 -> 0xy-zzz-zzzz, (02) 123 4567 -> (0x) xxx xxxx
    """
    # ตำแหน่งคั่นของต้นฉบับ
    parts = re.split(r"[\d]+", src)
    slots = re.findall(r"\d+", src)
    idx = 0
    out_chunks = []
    for i, slot in enumerate(slots):
        n = len(slot)
        out_chunks.append(new_digits[idx:idx+n])
        idx += n
    # สอดส่วนคั่นกลับไป
    out = []
    for i in range(max(len(parts), len(out_chunks))):
        if i < len(parts): out.append(parts[i])
        if i < len(out_chunks): out.append(out_chunks[i])
    return "".join(out)


_ALNUM = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def _digits_only_bank(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

# ---------------- Luhn for credit cards ----------------
def _luhn_check(num: str) -> bool:
    if not num or not num.isdigit(): 
        return False
    s = 0
    alt = False
    for d in reversed(num):
        n = ord(d) - 48
        if alt:
            n *= 2
            if n > 9: n -= 9
        s += n
        alt = not alt
    return (s % 10) == 0

def _luhn_complete(prefix: str, total_len: int) -> str:
    """Given prefix length total_len-1, compute check digit to make valid Luhn."""
    assert len(prefix) == total_len - 1
    s = 0
    alt = True  # last digit (check) is position 1; we start from position 2
    for d in reversed(prefix):
        n = ord(d) - 48
        if alt:
            n *= 2
            if n > 9: n -= 9
        s += n
        alt = not alt
    check = (10 - (s % 10)) % 10
    return prefix + str(check)

def _keep_sep_like(orig: str, newdigits: str) -> str:
    """Format newdigits to mimic separators (space/dash) of orig."""
    out = []
    it = iter(newdigits)
    for ch in orig:
        if ch.isdigit():
            out.append(next(it, ""))
        elif ch in "- ":
            out.append(ch)
        else:
            # unexpected char inside; drop it
            pass
    # If orig had fewer digit slots than newdigits (unlikely), append remaining digits
    rem = "".join(it)
    if rem:
        # keep last separator kind if exists
        sep = " " if " " in orig and "-" not in orig else "-"
        # chunk by 4
        chunks = [rem[i:i+4] for i in range(0, len(rem), 4)]
        out.extend([sep + c for c in chunks])
    return "".join(out) if out else newdigits

def _tokenize_number(raw_digits: str, salt: str = "", keep_last: int = 4, label: str = "tok"):
    tail = raw_digits[-keep_last:] if keep_last > 0 else ""
    h = hashlib.sha256((salt + raw_digits).encode("utf-8")).hexdigest()[:12]
    return f"<{label}:{h}{('-'+tail) if tail else ''}>"

# ---------------- Credit card anonymizer ----------------
_CARD_CAND = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

def _randomize_digits_like(sample: str, keep_last: int = 0, seed: str | None = None) -> str:
    raw = re.sub(r"\D", "", sample)
    n = len(raw)
    if n == 0:
        return sample
    keep_last = max(0, min(keep_last, n))
    kept = raw[-keep_last:] if keep_last else ""
    rnd = random.Random(seed) if seed is not None else random
    prefix_len = n - keep_last
    prefix = []
    for i in range(prefix_len):
        prefix.append(str(rnd.randint(0, 9)))
    new_digits = "".join(prefix) + kept
    out, it = [], iter(new_digits)
    for ch in sample:
        out.append(next(it) if ch.isdigit() else ch)
    return "".join(out)

# ตัวคั่นใช้ “ขีด” เป็นหลัก; อนุโลมเว้นวรรครอบขีดได้
_BANK_HSEP = r"\s*-\s*"

# ผู้สมัครเลขบัญชี:
#   - กรณีมาตรฐาน 4 กลุ่ม:   d{-}d{-}d{-}d  (มีขีด 3 ตัว เช่น 123-4-56789-0)
#   - อนุโลม 3 กลุ่ม:         d{-}d{-}d      (บางธนาคารเขียน 3 กลุ่ม)
# กรองซ้ำใน callback อีกชั้นด้วยจำนวนหลักรวม 9–12 เพื่อไม่ชนบัตรเครดิต (16) หรือบัตร ปชช. (13)
_BANK_CANDIDATE = re.compile(
    rf"""(?<!\d)(
            (?:\d{{1,4}}{_BANK_HSEP}){{3}}\d{{1,4}}      # exactly 3 hyphens → 4 groups
          | (?:\d{{2,6}}{_BANK_HSEP}){{2}}\d{{2,6}}      # 2 hyphens → 3 groups
        )(?!\d)""",
    re.VERBOSE,
)

def anonymize_credit_cards(text: str,
                           mode: str = "randomize",  # mask | tokenize | randomize
                           keep_last: int = 0,
                           salt: str | None = None) -> str:
    """
    Discover and anonymize credit-card-like sequences (13–19 digits).
    - Luhn-validate to reduce false positives.
    - Preserve original separators (space/dash) when randomizing.
    """
    if not text: 
        return text

    def _rand_card_like(length: int, first_digit: str | None = None) -> str:
        # keep 1st digit (brand-ish) if given; fill random until length-1; then compute check
        if first_digit and first_digit.isdigit():
            prefix = first_digit + "".join(str(random.randint(0,9)) for _ in range(length-2))
        else:
            prefix = str(random.randint(3,6)) + "".join(str(random.randint(0,9)) for _ in range(length-2))
        return _luhn_complete(prefix, length)

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        digits = _digits_only_bank(raw)
        if len(digits) < 13 or len(digits) > 19:
            return raw
        if not _luhn_check(digits):
            return raw  # not a card → keep
        if mode == "mask":
            tail = digits[-keep_last:] if keep_last > 0 else ""
            return f"<card{(':'+tail) if tail else ''}>"
        elif mode == "tokenize":
            return _tokenize_number(digits, salt or "", keep_last, "card")
        else:  # randomize
            newd = _rand_card_like(len(digits), first_digit=digits[0])
            return _keep_sep_like(raw, newd)

    return _CARD_CAND.sub(repl, text)

# ---------------- Bank account / IBAN anonymizer ----------------

def anonymize_bank_accounts(text: str, mode: str = "randomize", keep_last: int = 0, salt: str = "") -> str:
    """แทนที่เลขบัญชีธนาคารไทย
    - mode = "randomize": สุ่มเลขใหม่ตามความยาวเดิม คงรูปแบบขีด/เว้นวรรคเดิม
    - mode = "mask": แทนด้วย <account>
    - keep_last: คงท้าย n หลัก
    - salt: ทำ deterministic random ต่ออินพุตเดิมได้
    """
    def _cb(m: re.Match) -> str:
        s = m.group(0)
        # จำนวนหลักรวม (ตัด non-digit ออก)
        digits = re.sub(r"\D", "", s)
        L = len(digits)

        # กันชนกับบัตรเครดิต/บัตรประชาชน
        if L < 9 or L > 12:
            return s  # ไม่ใช่เลขบัญชีไทยทั่วไป ก็ข้าม

        # (ป้องกันชนบัตรประชาชน 13 หลักรูปแบบ 1-4-5-2-1 ซึ่งเราไม่ได้แมตช์อยู่แล้วเพราะต้อง 2–3 ขีด)
        if mode == "randomize":
            seed = (salt + digits) if salt else None
            return _randomize_digits_like(s, keep_last=keep_last, seed=seed)
        else:
            return "<account>"

    return _BANK_CANDIDATE.sub(_cb, text)

# Convenience wrappers for "mask" phase (placeholder-like)
def remove_credit_cards(text: str, keep_last: int = 0) -> str:
    return anonymize_credit_cards(text, mode="mask", keep_last=keep_last)

# (เผื่ออยากเรียกชื่อแบบ "remove_*" ให้สอดคล้อง)
def remove_bank_accounts(text: str, mode: str = "mask", keep_last: int = 0, salt: str = "") -> str:
    return anonymize_bank_accounts(text, mode=mode, keep_last=keep_last, salt=salt)
    
def remove_web_links(text):
    """
    Replace web URLs with <web> placeholder.
    
    Args:
        text (str): Text containing web links
        
    Returns:
        str: Text with links replaced by placeholder
    """
    tlds = [".com", ".net", ".org", ".io"]
    replace = f"https://www.{_rand_string(8)}{random.choice(tlds)}"
    text = re.sub(r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+(?:/\S*)?\b', replace, text)
    return text

def remove_phone_numbers(text: str, placeholder: str = "<phone>") -> str:
    """
    Replace Thai-like phone numbers with <phone>, รองรับรูปแบบมีตัวคั่น/วงเล็บ/+66/ext/#.
    """
    def _repl(m: re.Match) -> str:
        digits = _digits_only_phone(m.group(0))
        return _gen_th_mobile10() if _looks_like_th_phone(digits) else m.group(0)
    return _PHONE_CANDIDATE.sub(_repl, text)


def remove_social_media_mentions(text):
    """
    Remove social media profile references.
    
    Args:
        text (str): Text containing social media mentions
        
    Returns:
        str: Text with social media mentions removed
    """
    text = re.sub(r'(?:Twitter|Instagram|Facebook|Club|Fanpage|Line|YouTube|Wikipedia|E-Mail)\s*:?[^:\n]*', '', text)
    return text


#------------------------------------------------------------------------------
# Advanced Privacy Protection/Anonymization
#------------------------------------------------------------------------------

def _filter(regex, text, output_tag=False):
    """
    Helper function to filter content based on regex patterns.
    
    Args:
        regex: Compiled regex pattern
        text (str): Text to filter
        output_tag (bool): Whether to return tagged items
        
    Returns:
        str or tuple: Filtered text or (text, found items)
    """
    list_item = regex.findall(text)
    
    item_list_2 = []
    for item in list_item:
        if item not in item_list_2:
            item_list_2.append(item)
    
    if output_tag:
        return (text, item_list_2)
    return text

# Define phone number regex pattern
phone_number = r"\d{9}|\d{2}\-\d{3}\-\d{4}|\d{1}\s\d{4}\s\d{4}|\d{3}\-\d{6}|\+\d{2}\s\d{4}\s\d{4}|\+\d{2}\s\d{2}\s\d{6}|\+\d{3}\s\d{4}\s\d{4}|\+\d{11}|\+\d{10}"
phone_number_re = re.compile(phone_number)
#------------------------------------------------------------------------------
# Entity Filtering Functions
#------------------------------------------------------------------------------

def filter_personname(text, output_tag=False):
    ner = _load_thai_ner()
    ents = ner(text)
    names, seen = [], set()
    for e in ents:
        grp = (e.get("entity_group") or "").upper()
        if grp in ("PER","PERSON","B-PER","I-PER"):
            s, t = int(e["start"]), int(e["end"])
            span_txt = text[s:t].strip()
            if span_txt and span_txt not in seen:
                seen.add(span_txt); names.append(span_txt)
    if output_tag: return (text, names)
    return _replace(text, names, ["[PERSON]"]*len(names))


def filter_phone(text, token: str = "[PHONE]", output_tag=False):
    spans = [ _Span(m.start(), m.end(), m.group(0), "PHONE") for m in PHONE_RX.finditer(text) ]
    if output_tag:
        return (_replace_with_tokens(text, _resolve_overlaps(spans), {"PHONE": token}),
                [s.text for s in _resolve_overlaps(spans)])
    return _replace_with_tokens(text, _resolve_overlaps(spans), {"PHONE": token})

def filter_email(text, token: str = "[EMAIL]", output_tag=False):
    spans = [ _Span(m.start(), m.end(), m.group(0), "EMAIL") for m in EMAIL_RX.finditer(text) ]
    if output_tag:
        return (_replace_with_tokens(text, _resolve_overlaps(spans), {"EMAIL": token}),
                [s.text for s in _resolve_overlaps(spans)])
    return _replace_with_tokens(text, _resolve_overlaps(spans), {"EMAIL": token})

def filter_national_id(text, token: str = "[NATIONAL_ID]", output_tag=False):
    spans = [ _Span(m.start(), m.end(), m.group(0), "NATIONAL_ID") for m in TH_ID_RX.finditer(text) ]
    if output_tag:
        return (_replace_with_tokens(text, _resolve_overlaps(spans), {"NATIONAL_ID": token}),
                [s.text for s in _resolve_overlaps(spans)])
    return _replace_with_tokens(text, _resolve_overlaps(spans), {"NATIONAL_ID": token})

def filter_hospital_ids(text, token: str = "[HOSPITAL_IDS]", output_tag=False):
    spans = [ _Span(m.start(), m.end(), m.group(0), "HOSPITAL_IDS") for m in HOSPITAL_ID_RX.finditer(text) ]
    if output_tag:
        return (_replace_with_tokens(text, _resolve_overlaps(spans), {"HOSPITAL_IDS": token}),
                [s.text for s in _resolve_overlaps(spans)])
    return _replace_with_tokens(text, _resolve_overlaps(spans), {"HOSPITAL_IDS": token})

def filter_address(text, token: str = "[LOCATION]", output_tag=False):
    spans = _resolve_overlaps(_spans_from_ner_for(text, {"ADDRESS"}))
    if output_tag:
        return (_replace_with_tokens(text, spans, {"ADDRESS": token}), [s.text for s in spans])
    return _replace_with_tokens(text, spans, {"ADDRESS": token})

#------------------------------------------------------------------------------
# Entity Cleaning Functions (Anonymization)
#------------------------------------------------------------------------------


def reset_romanize_cache():
    """ล้างสถานะ mapping/แคชของระบบจับชื่อโรมัน (เรียกก่อนประมวลผลเอกสารใหม่)"""
    NEW_NAMES_ROMA.clear()
    ROMA2TH.clear()
    ROMA_ANON_CACHE.clear()
    

def register_romanized_names(names: list[str]) -> dict[str, str]:
    added = {}
    if not names or _thai2rom is None:
        return added
    for th in names:
        if not th:
            continue
        if th in NEW_NAMES_ROMA:
            added[th] = NEW_NAMES_ROMA[th]
            continue
        try:
            roma = _thai2rom(th) or ""
            roma = re.sub(r"\s+", " ", roma).strip()
        except Exception:
            roma = ""
        if roma:
            NEW_NAMES_ROMA[th] = roma
            ROMA2TH[roma.lower()].add(th)
            added[th] = roma
    return added

def _find_generic_latin_name_spans(text: str):
    spans = []
    for m in _GENERIC_LATIN_NAME_RX.finditer(text):
        g = m.group(0)
        if _looks_like_url_or_email(g):
            continue
        spans.append((m.start(), m.end(), g))
    return spans

    
def _find_romanized_person_matches(text: str, threshold: float = 0.5):
    """
    คืน list ของ (start, end, matched_text) สำหรับชื่อ 'โรมัน' ที่น่าจะเป็นชื่อคน
    อ้างอิงจาก mapping NEW_NAMES_ROMA (ที่ได้มาหลัง NER พบชื่อไทย)
    - ตรวจ exact ก่อน
    - ไม่เจอค่อยตรวจ fuzzy แบบเลื่อน window ตามจำนวนคำของรูปโรมัน
    - กัน span ซ้อนทับแบบง่าย
    """
    if not NEW_NAMES_ROMA:
        return []

    s = text
    low = s.lower()
    matches = []

    # 1) exact
    for roma in NEW_NAMES_ROMA.values():
        r = (roma or "").strip()
        if not r:
            continue
        rl = r.lower()
        idx = low.find(rl)
        if idx != -1:
            matches.append((idx, idx + len(r), s[idx:idx+len(r)]))

    # 2) fuzzy (ถ้ายังว่าง)
    if not matches:
        words = _LATIN_WORD_RX.findall(s)
        if words:
            for r in [v for v in NEW_NAMES_ROMA.values() if v]:
                rl = r.lower().strip()
                n_words = max(1, len(r.split()))
                for k in range(0, max(0, len(words) - n_words + 1)):
                    cand = " ".join(words[k:k + n_words])
                    if _sim(cand.lower(), rl) >= threshold:
                        pos = low.find(cand.lower())
                        if pos != -1:
                            matches.append((pos, pos + len(cand), s[pos:pos+len(cand)]))
                            break

    # กันซ้อนทับ
    matches = sorted(matches, key=lambda x: x[0])
    non_overlap = []
    last_end = -1
    for st, ed, m in matches:
        if st >= last_end:
            non_overlap.append((st, ed, m))
            last_end = ed
    return non_overlap


def mask_romanized_person_names(text: str, threshold: float = 0.5, token: str | None = None) -> str:
    if token is None:
        token = _PERSON_TOKEN_DEFAULT
    matches = _find_romanized_person_matches(text, threshold=threshold)
    if not matches:  # fallback: จับชื่อโรมันแบบ generic (รองรับ lowercase/allcaps)
        matches = _find_generic_latin_name_spans(text)
    if not matches:
        return text
    s = text
    for st, ed, _src in sorted(matches, key=lambda x: x[0], reverse=True):
        s = s[:st] + token + s[ed:]
    return s


def _copy_casing(src: str, repl: str) -> str:
    """คงรูปแบบตัวพิมพ์ให้เหมือน src (UPPER / lower / Title)"""
    if src.isupper():
        return repl.upper()
    if src.islower():
        return repl.lower()
    if src.istitle():
        return repl.title()
    return repl


# ชื่อฝรั่งสำรอง หากไม่มี pythainlp ให้ romanize
_WEST_FIRST = ["Alex", "Jamie", "Taylor", "Jordan", "Casey", "Chris", "Sam", "Sky"]
_WEST_LAST  = ["Smith", "Carter", "Lee", "Morgan", "Parker", "Reed", "Gray", "Wong"]

def _gen_western_name() -> str:
    import random
    return f"{random.choice(_WEST_FIRST)} {random.choice(_WEST_LAST)}"


def anonymize_romanized_person_names(text: str, threshold: float = 0.5) -> str:
    matches = _find_romanized_person_matches(text, threshold=threshold)
    if not matches:  # ← fallback generic
        matches = _find_generic_latin_name_spans(text)
    if not matches:
        return text
    out = text
    gen_name_fn = globals().get("gen_name", None)
    for st, ed, src in sorted(matches, key=lambda x: x[0], reverse=True):
        rep = ROMA_ANON_CACHE.get(src)
        if not rep:
            if callable(gen_name_fn):
                fake_th = gen_name_fn(True)
            else:
                fake_th = None
            if _thai2rom is not None and fake_th:
                try:
                    rep = _thai2rom(fake_th) or _gen_western_name()
                except Exception:
                    rep = _gen_western_name()
            else:
                rep = _gen_western_name()
            rep = re.sub(r"\s+", " ", rep).strip()
            ROMA_ANON_CACHE[src] = rep
        out = out[:st] + _copy_casing(src, rep) + out[ed:]
    return out


_ORIG_clean_name = globals().get("clean_name", None)

def clean_name(text: str, *, spans=None, public_figure_skip: bool = True):
    """
    Anonymize ชื่อบุคคลภาษาไทยแบบ 'คง alias เดิมทั้งเอกสาร'
    - ใช้ PERSON_ALIAS ให้ชื่อเดิมคนเดียวกันได้ชื่อปลอมเดิม
    - รองรับ input จาก thai_ner_full_apply(spans=...) หากส่งมา
    """
    filter_personname = globals().get("filter_personname")
    gen_name_fn = globals().get("gen_name")
    _replace_fn = globals().get("_replace")

    if not callable(gen_name_fn) or not callable(_replace_fn):
        if callable(_ORIG_clean_name):
            return _ORIG_clean_name(text)
        return text

    # 1) หา spans PERSON (ถ้า caller ไม่ได้ส่งมา)
    found_names = []
    if spans:
        # ดึงเฉพาะสแปนที่เป็น PERSON
        for s in spans:
            cat = (getattr(s, "cat", None) or getattr(s, "type", None) or getattr(s, "label", "")).upper()
            if cat in {"PERSON","PER","B-PER","I-PER","B-PERSON","I-PERSON"}:
                span_txt = getattr(s,"text","") or (s.get("text","") if isinstance(s, dict) else "")
                if span_txt:
                    found_names.append(span_txt)
    else:
        if not callable(filter_personname):
            if callable(_ORIG_clean_name):
                return _ORIG_clean_name(text)
            return text
        try:
            if callable(filter_personname):
                _tmp, more_names = filter_personname(text, output_tag=True)
                # ถ้าเปิดโหมด skip คนดัง ให้กรองออกก่อน
                if public_figure_skip:
                    more_names = [n for n in more_names if not is_public_figure(n)]
                for n in more_names:
                    if n and (n not in found_names):
                        found_names.append(n)
        except Exception:
            pass
    if not found_names and callable(filter_personname):
        _tmp, found_names = filter_personname(text, output_tag=True)
    if not found_names:
        return text

    # 2) เตรียม mapping ไทย->โรมัน (ของเดิมคุณใช้ต่อให้ชื่อโรมันคงรูป)
    try:
        register_romanized_names(found_names)
    except Exception:
        pass

    # 3) สร้างรายการแทนที่แบบอ้าง position เพื่อกันทับซ้อน
    #    เดินจากซ้ายไปขวา แล้วค่อยแทนที่ย้อนจากขวาไปซ้าย
    replace_ops = []
    # หา position ของแต่ละชื่อจากข้อความจริง (กรณี spans มี start/end จะดีกว่า)
    pairs = []  # [(orig_full, fake_full)]
    for op in replace_ops:
        if isinstance(op, tuple) and len(op) >= 2:
            orig, fake = op[0], op[1]
            if isinstance(orig, str) and isinstance(fake, str):
                pairs.append((orig, fake))
    
    # แตกเป็นชื่อหน้า/นามสกุล แล้วเพิ่ม "ตัวเสริม" เฉพาะเอนทิตีเดียวกันเท่านั้น
    import re
    TH = r"\u0E00-\u0E7F"
    def _split_name_th(s: str):
        parts = [p for p in s.strip().split() if p]
        if len(parts) >= 2: return parts[0], parts[-1]
        if len(parts) == 1: return parts[0], ""
        return "", ""
    
    for orig_full, fake_full in pairs:
        o_first, o_last = _split_name_th(orig_full)
        f_first, f_last = _split_name_th(fake_full)
    
        # เฉพาะเคสชื่อหน้าที่ยาวพอ และไม่ใช่คำสามัญ
        if len(o_first) >= 3 and len(f_first) >= 3:
            # ขอบเขตไทย: ไม่ชนตัวอักษรไทยติดกันซ้าย/ขวา (ลด false positive)
            pat = re.compile(rf"(?<![{TH}]){re.escape(o_first)}(?![{TH}])")
            replace_ops.append((pat, f_first))  # ใช้ regex pattern เป็น key
    
        if o_last and f_last and len(o_last) >= 3 and len(f_last) >= 3:
            pat = re.compile(rf"(?<![{TH}]){re.escape(o_last)}(?![{TH}])")
            replace_ops.append((pat, f_last))
    # ถ้าไม่มีตำแหน่ง ขอใช้การค้นหาแบบ find-all
    candidates = []
    if spans:
        for s in spans:
            cat = (getattr(s, "cat", None) or getattr(s, "type", None) or getattr(s, "label", "")).upper()
            if cat in {"PERSON","PER","B-PER","I-PER","B-PERSON","I-PERSON"}:
                st = getattr(s, "start", None) or s.get("start")
                ed = getattr(s, "end", None) or s.get("end")
                txt = getattr(s, "text", "") or (s.get("text","") if isinstance(s, dict) else "")
                if isinstance(st, int) and isinstance(ed, int) and txt:
                    candidates.append((st, ed, txt))
    else:
        # fallback: หาโดย string match ทุกตำแหน่งแบบ non-overlap (อย่างง่าย)
        used_ranges = []
        for name in found_names:
            start = 0
            while True:
                idx = text.find(name, start)
                if idx == -1:
                    break
                rng = (idx, idx+len(name))
                # กัน overlap แบบง่าย
                if not any(not (rng[1] <= a or rng[0] >= b) for a,b,_ in used_ranges):
                    used_ranges.append((*rng, name))
                start = idx + len(name)
        used_ranges.sort(key=lambda x: x[0])
        candidates = [(a, b, n) for a,b,n in used_ranges]

    # 4) เดินตามลำดับซ้าย→ขวา กำหนด alias ตาม key ปกติ
    for st, ed, src in sorted(candidates, key=lambda x: x[0]):
        key = normalize_person_key(src)
        if not key:
            continue
        fake = PERSON_ALIAS.get(key)
        if not fake:
            # ตัดสินว่าควรสุ่มชื่อเต็มไหม
            is_full = len(key.split()) >= 2
            fake = gen_name_fn(is_full)
            PERSON_ALIAS[key] = fake
        replace_ops.append((st, ed, fake))

    # 5) แทนที่จากขวา→ซ้าย เพื่อไม่ให้ตำแหน่งเลื่อน
    out = text
    for st, ed, rep in sorted(replace_ops, key=lambda x: x[0], reverse=True):
        out = out[:st] + rep + out[ed:]

    return out


def clean_phone(text, salt: str | None = None, keep_last: int = 0):
    def _rep(m):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        n = len(digits)
        syn = _digits_from_hash((salt or "") + digits, n)
        if keep_last:
            syn = syn[:-keep_last] + digits[-keep_last:]
        return _preserve_separators(raw, syn)
    return PHONE_RX.sub(_rep, text)

def clean_email(text, salt: str | None = None):
    def _rep(m):
        raw = m.group(0)
        # สร้างอีเมลใหม่ deterministic: userXXXX@example.com (คง TLD เดิมถ้าต้องการ)
        suffix = _digits_from_hash((salt or "") + raw, 6)
        return f"user{suffix}@example.com"
    return EMAIL_RX.sub(_rep, text)

def clean_national_id(text, salt: str | None = None, keep_last: int = 0):
    def _rep(m):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        n = len(digits)
        syn = _digits_from_hash((salt or "") + digits, n)
        if keep_last:
            syn = syn[:-keep_last] + digits[-keep_last:]
        return _preserve_separators(raw, syn)
    return TH_ID_RX.sub(_rep, text)

def clean_hospital_ids(text, salt: str | None = None):
    def _rep(m):
        raw = m.group(0)
        prefix = "HN" if raw.upper().startswith("HN") else "AN"
        body = re.sub(r"\D", "", raw)
        syn = _digits_from_hash((salt or "") + body, len(body))
        return f"{prefix}{syn}"
    return HOSPITAL_ID_RX.sub(_rep, text)

def clean_address(text):
    # สำหรับ anonymize address (จริง ๆ ต้องการ generator ที่ซับซ้อน)
    # ตอนนี้ใช้ mask แทนเป็นค่าเริ่มต้น เพื่อลดความเสี่ยง PDPA
    return filter_address(text)

def clean_url(text):
    """
    Replace real URLs with generated fake URLs.
    
    Args:
        text (str): Text containing real URLs
        
    Returns:
        str: Text with URLs replaced by fake URLs
    """
    temp = filter_url(text, output_tag=True)
    text_temp = temp[0]
    list_index = temp[1]
    list_url = [gen_url() for i in range(0, len(list_index)) if len(list_index) != 0]
    return _replace(text_temp, list_index, list_url)

#------------------------------------------------------------------------------
# Website-specific Content Removal
#------------------------------------------------------------------------------

# List of patterns to remove from website content
PARTIAL_REMOVAL_KEYWORDS = [
    # from pantip
    "\[Spoil\].*?(?=\n|$)",
    "\[How to\]",
    "คลิกเพื่อซ่อนข้อความ",
    # from oscar_23
    "Posted on",
    "Posted by",
    "Posted by:",
    "Posted By:",
    "สมาชิกหมายเลข [0-9,]+",
    "อ่าน [0-9,]+ ครั้ง",
    "เปิดดู [0-9,]+ ครั้ง",
    "ดู [0-9,]+ ครั้ง",
    "คะแนนสะสม: [0-9,]+ แต้ม",
    "ความคิดเห็น: [0-9,]+",
    "[0-9,]+ บุคคลทั่วไป กำลังดูบอร์ดนี้",
    "หน้าที่แล้ว ต่อไป",
    "ความคิดเห็นที่ [0-9,]+",
    "[0-9,]+ สมาชิก และ [0-9,]+ บุคคลทั่วไป",
    "กำลังดูหัวข้อนี้",
    "เข้าสู่ระบบด้วยชื่อผู้ใช้",
    "แสดงกระทู้จาก:",
    "กระทู้: [0-9,]+",
    "เว็บไซต์เรามีการใช้คุกกี้และเก็บข้อมูลผู้ใช้งาน โปรดศึกษาและยอมรับ นโยบายคุ้มครองข้อมูลส่วนบุคคล ก่อนใช้งาน",
    "Privacy & Cookies: This site uses cookies. By continuing to use this website, you agree to their use\.",
    "Previous\t\nNext\nLeave a Reply Cancel reply\nYou must be logged in to post a comment.\nSearch for:\nFeatured Post\n",
    "Click to read more\nYou must be logged in to view or write comments\.",
    "[0-9,]+ Views",
    "Skip to content",
    "Last Modified Posts",
    "Last Updated:",
    "\(อ่าน [0-9,]+ ครั้ง\)",
    "Recent Comments",
    "«.*?»",
    "< --แสดงทั้งหมด-- >",
    "นโยบายความเป็นส่วนตัว",
    "เงื่อนไขการใช้เว็บไซต์",
    "ตั้งค่าคุกกี้",
    "ท่านยอมรับให้เว็บไซต์นี้จัดเก็บคุกกี้เพื่อประสบการณ์การใช้งานเว็บไซต์ที่ดียิ่งขึ้น",
    "รวมถึงช่วยให้ท่านมีโอกาสได้รับข้อเสนอหรือเนื้อหาที่ตรงตามความสนใจของท่าน",
    "ท่านสามารถดู Privacy Notice ของเว็บไซต์เรา ได้ที่นี่",
    "You may be trying to access this site from a secured browser on the server. Please enable scripts and reload this page.",
    "เผยแพร่: \d\d [ก-๙]+ \d\d\d\d \d\d:\d\d น\.",
    "Last updated: \d\d [ก-๙]+\.[ก-๙]+\. \d\d\d\d \d\d:\d\d น\.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit\.",
    "Search for:",
    "Save my name, email, and website in this browser for the next time I comment",
    "Your email address will not be published. Required fields are marked",
    "Leave a Reply Cancel reply",
    "((?:หน้าหลัก|เข้าสู่ระบบ|หน้าแรก) \|(?: [^\s]+(?:(?: \|)|$|\s))+)",
    "กลับหน้าแรก",
    "ติดต่อเรา",
    "Contact Us",
    "#\w+",
    "ติดต่อผู้ดูแลเว็บไซต์",
    "หากท่านพบว่ามีข้อมูลใดๆที่ละเมิดทรัพย์สินทางปัญญาปรากฏอยู่ในเว็บไซต์โปรดแจ้งให้ทราบ",
    "No related posts",
    "Posted in",
    "((?:Tags:|Tagged|Tag) (?:.{1,40}(?:,|\n|$))+)",
    "ตอบ:",
    "Sort by:",
    "All rights reserved",
    "ความยาวอย่างน้อย",
    "ระบบได้ดำเนินการส่ง OTP",
    "เป็นสมาชิกอยู่แล้ว\?",
    "We use cookies",
    "Cookie Settings",
    "Homeหน้าหลัก",
    "Home หน้าหลัก",
    "ข่าวสารล่าสุด",
    "ปัญหา การใช้งาน",
    "ปัญหาการใช้งาน" "ผู้เขียน",
    "หัวข้อ:",
    "\*\* พร้อมส่ง \*\*"
]

def remove_partial_removal_keywords(text):
    """
    Remove website-specific content patterns.
    
    Args:
        text (str): Text containing website-specific patterns
        
    Returns:
        str: Text with website-specific patterns removed
    """
    return re.sub("|".join(PARTIAL_REMOVAL_KEYWORDS), "", text)


#------------------------------------------------------------------------------
# Text Normalization
#------------------------------------------------------------------------------

def lower_text(text):
    """
    Convert text to lowercase.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lowercase text
    """
    return text.lower()

def replace_text(text):
    """
    Replace various special characters and whitespace.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with normalized characters
    """
    text = text.replace('\x93', '"')
    text = text.replace('\x94', '"')
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    #text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    return text

# Dictionary of common Thai mistypes and their corrections
MISTYPE_WORD_DICT = {
    'เเ': 'แ',
    'กฏ': 'กฎ',
    'มงกุฏ': 'มงกุฎ',
    'กวยเตี๋ยว': 'ก๋วยเตี๋ยว',
    'ชำรุจ': 'ชำรุด',
    'บ้องวงจรปิด': 'กล้องวงจรปิด',
    'กงศุล': 'กงสุล',
    'คลิกเบท': 'คลิกเบต',
    'กัลปพกฤกษ์': 'กัลปพฤกษ์',
    'มิฉฉาชีพ': 'มิจฉาชีพ',
    'วักคณิกา': 'วัดคณิกา',
    'สวัสดิการณ์': 'สวัสดิการ',
    'ทศกัณฑ์': 'ทศกัณฐ์',
    'กลางคืม': 'กลางคืน',
    'มังคล': 'มงคล',
    'มานั้ง': 'ม้านั่ง',
    'พุกพ่าน': 'พลุกพล่าน',
    'รับผิดสอบ': 'รับผิดชอบ',
    'โอกาศ': 'โอกาส',
    'นำ้': 'น้ำ',
    'สันจร': 'สัญจร',
    'กาญจนภิเษก': 'กาญจนาภิเษก',
    'เขรยว': 'เขียว',
    'มอเตอไซด์': 'มอเตอร์ไซค์',
    'มอเตอร์ไซด์': 'มอเตอร์ไซค์',
    'มอเตอร์ไซต์': 'มอเตอร์ไซค์',
    'มอเตอร์ไซร์': 'มอเตอร์ไซค์',
    'ฟุทบาท':'ฟุตบาท',
    'ฟุตปาธ':'ฟุตบาท',
    'ฟุตพาธ': 'ฟุตบาท',
    'ฟุบาท': 'ฟุตบาท',
    'ฟุตบาต': 'ฟุตบาท',
    'แบร์ริเออร์': 'แบริเออร์',
    'พิจจารณา': 'พิจารณา',
    'ทางอดิน': 'ทางเดิน',
    'บริิหาร': 'บริหาร',
    'น้อมจิตร์': 'น้อมจิตต์',
    'ถรน': 'ถนน',
    'ประสพปัญหา': 'ประสบปัญหา',
    'บริเวน': 'บริเวณ',
    'วิดิโอ': 'วิดีโอ',
    'เซนทรัล': 'เซ็นทรัล',
    'บริเวร': 'บริเวณ',
    'วิลแชร์': 'วีลแชร์',
}

def replace_mistype_text(text):
    """
    Replace common Thai misspelled words with correct versions.
    
    Args:
        text (str): Text with possible misspellings
        
    Returns:
        str: Text with corrected spellings
    """
    for mistype_word, correct_word in MISTYPE_WORD_DICT.items():
        text = text.replace(mistype_word, correct_word)
    return text

def transform_thai_num_text_to_arabic_num_text(text):
    """
    Convert Thai numerals to Arabic numerals.
    
    Args:
        text (str): Text with Thai numerals
        
    Returns:
        str: Text with Arabic numerals
    """
    thai_to_arabic = {
        '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
        '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
    }
    
    for thai, arabic in thai_to_arabic.items():
        text = text.replace(thai, arabic)
    return text

def remove_arabic_num_text(text):
    """
    Remove Arabic numerals from text.
    
    Args:
        text (str): Text containing Arabic numerals
        
    Returns:
        str: Text with Arabic numerals removed
    """
    text = re.sub(r'[0-9]', '', text)
    return text


#------------------------------------------------------------------------------
# Punctuation Processing
#------------------------------------------------------------------------------

# Create escaped punctuation for regex patterns
ESCAPED_PUNCTUATION = re.escape(string.punctuation)

def remove_duplicate_punctuation(text):
    """
    Replace duplicate punctuation marks with single ones.
    
    Args:
        text (str): Text with possible duplicate punctuation
        
    Returns:
        str: Text with normalized punctuation
    """
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    text = re.sub(f'([{ESCAPED_PUNCTUATION}])\\1+', r'\1', text)
    return text

def isolate_punctuation(text):
    """
    Add spaces around punctuation marks.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with spaces around punctuation
    """
    text = re.sub(f'([{ESCAPED_PUNCTUATION}])', r' \1 ', text)
    return text

def remove_punctuation(text):
    """
    Remove all punctuation marks from text.
    
    Args:
        text (str): Text containing punctuation
        
    Returns:
        str: Text without punctuation
    """
    text = re.sub(f'[{ESCAPED_PUNCTUATION}]', ' ', text)
    return text
    
#------------------------------------------------------------------------------
# Whitespace Normalization
#------------------------------------------------------------------------------

def normalize_whitespace(text):
    """
    Normalize multiple spaces, newlines, tabs to single space.
    
    Args:
        text (str): Text with irregular spacing
        
    Returns:
        str: Text with normalized whitespace
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# === BEGIN: AddressCleaner (gazetteer-based) ===

import json as _json
import unicodedata as _unicodedata
import re as _re

def load_thai_gazetteer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = _json.load(f)
    if isinstance(raw, dict) and raw:
        first_val = next(iter(raw.values()))
    else:
        first_val = raw
    prov_set, amph_set, tamb_set = set(), set(), set()
    if isinstance(first_val, dict):
        for prov_label, items in first_val.items():
            prov_name_th = str(prov_label).split("-")[0].strip()
            if prov_name_th:
                prov_set.add(prov_name_th)
                if prov_name_th in ("กรุงเทพมหานคร","กรุงเทพฯ"):
                    prov_set.update({"กรุงเทพฯ","กทม.","กทม"})
            if isinstance(items, list):
                for it in items:
                    th = str(it).split("-")[0].strip()
                    th = _re.sub(r"\s+", " ", th)
                    th_no_prefix = _re.sub(r"^(อำเภอ|เขต|ตำบล|แขวง)\s*", "", th)
                    if th.startswith(("อำเภอ","เขต")):
                        amph_set.add(th_no_prefix)
                    elif th.startswith(("ตำบล","แขวง")):
                        tamb_set.add(th_no_prefix)
    return {"provinces": prov_set, "amphoe": amph_set, "tambon": tamb_set}

def _alts_from_set(_names:set):
    al = sorted((_re.escape(x) for x in _names if x), key=len, reverse=True)
    return r"(?:%s)" % r"|".join(al) if al else r"(?!x)x"

def build_addr_regex(gz: dict):
    TOKS = [
        r"บ้านเลขที่", r"เลขที่", r"หมู่บ้าน", r"หมู่ที่", r"หมู่",
        r"ซอย", r"ซ\.", r"ตรอก", r"ถนน", r"ถ\.", r"สะพาน", r"อาคาร", r"ตึก", r"ชั้น", r"ห้อง",
        r"แขวง", r"เขต", r"ตำบล", r"ต\.", r"อำเภอ", r"อ\.", r"จังหวัด", r"จ\.",
    ]
    ALTS_PROV  = _alts_from_set(gz.get("provinces", set()))
    ALTS_AMPH  = _alts_from_set(gz.get("amphoe", set()))
    ALTS_TAMB  = _alts_from_set(gz.get("tambon", set()))
    TOKS_ALT   = r"(?:%s)" % "|".join(TOKS)
    STOP = r"(?:โทร|เบอร์|มือถือ|เบอร์ติดต่อ|email|อีเมล|e-?mail|https?://|บัญชี|บัตร|เลขประชาชน|\d{9,}|\Z)"
    rx_list = [
        _re.compile(
            rf"(?P<addr>(?:ที่อยู่|อยู่ที่|บ้านเลขที่|เลขที่)\s*\d{{1,6}}(?:/\d{{1,6}})?(?:-\d{{1,4}})?)"
            rf"(?=(?:\s|$|[.,;:)\]])|{STOP})"
        ),
        _re.compile(rf"(?P<addr>(?:ที่อยู่|อยู่ที่|บ้านเลขที่|เลขที่)?\s*.*?"
                    rf"(?:แขวง\s*{ALTS_TAMB}|ตำบล\s*{ALTS_TAMB}|เขต\s*{ALTS_AMPH}|อำเภอ\s*{ALTS_AMPH}|จังหวัด\s*{ALTS_PROV})"rf".*?)(?=\s*{STOP})"),
        _re.compile(rf"(?P<addr>.*?(?:{TOKS_ALT})\s*(?:{ALTS_TAMB}|{ALTS_AMPH}|{ALTS_PROV}).*?)(?=\s*{STOP})"),
        _re.compile(rf"(?P<addr>\b\d{{1,4}}(?:/\d{{1,4}})?\s*.*?(?:{TOKS_ALT}|{ALTS_TAMB}|{ALTS_AMPH}|{ALTS_PROV}).*?)(?=\s*{STOP})"),
    ]
    return rx_list

def address_find_gaz(text: str, rx_list):
    if not text: return []
    spans = []
    for rx in rx_list:
        for m in rx.finditer(text):
            s, e = m.start("addr"), m.end("addr")
            val = text[s:e].strip()
            if len(val) < 8: 
                continue
            if not _re.search(r"(แขวง|เขต|ตำบล|ต\.|อำเภอ|อ\.|จังหวัด|จ\.|ซอย|ถนน|เลขที่|บ้านเลขที่)", val):
                continue
            spans.append({"start": s, "end": e, "value": val})
    spans.sort(key=lambda x: (x["start"], x["end"]))
    merged = []
    for sp in spans:
        if merged and sp["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"]  = max(merged[-1]["end"], sp["end"])
            if len(sp["value"]) > len(merged[-1]["value"]):
                merged[-1]["value"] = sp["value"]
        else:
            merged.append(sp)
    return merged

_FAKE_ROADS   = ["สุขุมวิท","เพลินจิต","สีลม","พหลโยธิน","ลาดพร้าว","รัชดาภิเษก","พระราม 9","สาทร","ราชดำริ","เอกมัย"]
_FAKE_KWANG   = ["คลองตัน","ลุมพินี","บางรัก","จอมพล","จตุจักร","ลาดพร้าว"]
_FAKE_KHET    = ["วัฒนา","ปทุมวัน","บางรัก","พญาไท","จตุจักร","ลาดพร้าว"]
_FAKE_ZIP_BKK = ["10110","10330","10500","10400","10900","10230"]

def _h_i(s: str, salt: str = "") -> int:
    import hashlib as _hashlib
    return int(_hashlib.sha1((salt+"§"+s).encode("utf-8")).hexdigest()[:12], 16)

def _pick(seq, h: int):
    seq = list(seq)
    return seq[h % max(1, len(seq))] if seq else ""

def anonymize_address_gaz(text: str, spans, *, salt: str, keep_province: bool, gz: dict) -> str:
    s = text
    for sp in sorted(spans, key=lambda x: x["start"], reverse=True):
        raw = sp["value"]
        h = _h_i(raw, salt)
        prov = None
        for p in gz.get("provinces", set()):
            if p and p in raw:
                prov = p; break
        if keep_province and prov and prov in {"กรุงเทพมหานคร","กรุงเทพฯ","กทม.","กทม"}:
            fake = f"เลขที่ {1+(h%299)}/{1+((h//19)%50)} ถนน{_pick(_FAKE_ROADS,h)} แขวง{_pick(_FAKE_KWANG,h//7)} เขต{_pick(_FAKE_KHET,h//11)} กรุงเทพฯ {_pick(_FAKE_ZIP_BKK,h//13)}"
        else:
            prov_use = prov if (keep_province and prov) else _pick(sorted(list(gz.get('provinces', []))), h)
            tamb = _pick(sorted(list(gz.get('tambon', []))), h//5)
            amph = _pick(sorted(list(gz.get('amphoe', []))), h//9)
            zipcode = 10000 + (h % 90000)
            fake = f"เลขที่ {1+(h%299)}/{1+((h//19)%50)} ตำบล{tamb} อำเภอ{amph} จังหวัด{prov_use} {zipcode}"
        s = s[:sp["start"]] + fake + s[sp["end"]:]
    return s

def mask_address_gaz(text: str, spans, tag: str="[ADDRESS]") -> str:
    s = text
    for sp in sorted(spans, key=lambda x: x["start"], reverse=True):
        s = s[:sp["start"]] + tag + s[sp["end"]:]
    return s

__ADDR_CLEANER_CACHE = {"inst": None, "path": None}

class AddressCleaner:
    def __init__(self, gazetteer_path: str | None):
        self.gz = {"provinces": set(), "amphoe": set(), "tambon": set()}
        if gazetteer_path and os.path.exists(gazetteer_path):
            self.gz = load_thai_gazetteer(gazetteer_path)
        self.rx_list = build_addr_regex(self.gz)
    def find(self, text: str):
        return address_find_gaz(text, self.rx_list)
    def clean(self, text: str, policy: str="mask", *, salt: str="", keep_province: bool=True, tag: str="[ADDRESS]"):
        spans = self.find(text)
        if not spans: return text, []
        if policy == "anonymize":
            return anonymize_address_gaz(text, spans, salt=salt, keep_province=keep_province, gz=self.gz), spans
        return mask_address_gaz(text, spans, tag=tag), spans

def _get_address_cleaner(gazetteer_path: str | None):
    if __ADDR_CLEANER_CACHE["inst"] is None or __ADDR_CLEANER_CACHE["path"] != gazetteer_path:
        __ADDR_CLEANER_CACHE["inst"] = AddressCleaner(gazetteer_path)
        __ADDR_CLEANER_CACHE["path"] = gazetteer_path
    return __ADDR_CLEANER_CACHE["inst"]

def address_detect(text: str, gazetteer_path: str | None = None):
    return _get_address_cleaner(gazetteer_path).find(text or "")

def address_mask(text: str, gazetteer_path: str | None = None, tag: str = "[ADDRESS]"):
    cleaner = _get_address_cleaner(gazetteer_path)
    out, _ = cleaner.clean(text or "", policy="mask", tag=tag)
    return out

def address_anonymize(text: str, gazetteer_path: str | None = None, salt: str = "", keep_province: bool = True):
    cleaner = _get_address_cleaner(gazetteer_path)
    out, _ = cleaner.clean(text or "", policy="anonymize", salt=salt, keep_province=keep_province)
    return out