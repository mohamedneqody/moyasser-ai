"""
Moyasser AI — Killer Neuro-Symbolic Grounding Layer:
Explainable Egyptian Dialect X-Ray (`الأشعة الدلالية للعامية المصرية`)
Solves the hardest problem in Legal/Financial AI ("The Neuro-Symbolic Grounding Gap"):
  1. Compiles spoken colloquial Egyptian Arabic numbers (`مليون ونص`, `نص مليون`, `ميتين ألف`, `تلاتة مليون وربع`) into exact integers.
  2. Runs a trained PyTorch Bidirectional GRU Sequence Tagger (`ArabicSlotBiGRU` — `models/arabic_slot_bigru.pt`)
     reading sentences in BOTH directions (RTL + LTR) to bind numbers to their true financial role even when
     numbers precede labels (`بعمل 4 مليون مبيعات`) or multiple clauses appear in one sentence.
  3. Automatically converts Monthly (`×12`), Weekly (`×52`), and Daily (`×300` working days) figures into Annual EGP
     with explicit transparent formulas.
  4. Honors Egyptian Arabic Negation Scope (`معليش غرامات`, `مفيش نزاع`, `بدون مصاريف`) so negated terms never trigger false flags.
  5. Computes Statutory Cliff-Edge Sensitivity Alerts (e.g. proximity to Law 6 brackets & the 24M EGP Theorem-9 Grace Buffer).
"""

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

SLOT_TAGS = ["O", "REV", "EXP", "PEN", "PRINC", "RE_VAL", "VAT", "NEG", "FREQ"]

SLOT_LABELS_AR = {
    "REV": "حجم الأعمال / الإيراد السنوي",
    "EXP": "المصروفات والتكاليف السنوية",
    "PEN": "مقابل التأخير والغرامات",
    "PRINC": "أصل الضريبة المتنازع عليها",
    "RE_VAL": "قيمة التصرف العقاري",
    "VAT": "رصيد رد القيمة المضافة",
}

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")


def _norm_token(tok: str) -> str:
    t = tok.translate(_AR_DIGITS).lower().strip()
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"ة", "ه", t)
    t = re.sub(r"[ًٌٍَُِّْ،؛؟!.,()\[\]{}]", "", t)
    return t


class ArabicSlotBiGRU(nn.Module):
    """2-Layer Bidirectional GRU Token-Slot Tagger for Egyptian Arabic Financial Grounding."""

    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 96, num_tags: int = len(SLOT_TAGS)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(0.20)
        self.bigru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.drop(self.emb(x))
        h, _ = self.bigru(e)
        return self.head(self.norm(h))


@lru_cache(maxsize=1)
def _load_slot_bigru():
    pt_path = os.path.join(MODEL_DIR, "arabic_slot_bigru.pt")
    if not os.path.exists(pt_path):
        return None
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    model = ArabicSlotBiGRU(
        vocab_size=int(ckpt["vocab_size"]),
        emb_dim=int(ckpt["emb_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        num_tags=len(ckpt["tags"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, dict(ckpt["vocab"]), list(ckpt["tags"])


def compile_egyptian_spoken_numbers(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Rewrites spoken Egyptian Arabic number expressions into explicit numeric strings
    while recording a transparent trace of every spoken-to-numeric conversion!
    Examples:
      - `مليون ونص` -> `1500000`
      - `نص مليون` -> `500000`
      - `ميتين ألف` / `مائتين الف` -> `200000`
      - `تلاتة مليون وربع` -> `3250000`
    """
    s = (text or "").translate(_AR_DIGITS)
    conversions: List[Dict[str, Any]] = []

    spoken_rules: List[Tuple[str, int]] = [
        # Compound Millions + Fractions
        (r"\b(?:مليونين|اتنين\s*مليون|اثنين\s*مليون)\s*و\s*(?:نص|نصف)\b", 2_500_000),
        (r"\b(?:مليونين|اتنين\s*مليون|اثنين\s*مليون)\s*و\s*ربع\b", 2_250_000),
        (r"\b(?:تلات[ةه]|ثلاث[ةه])\s*(?:مليون|ملايين)\s*و\s*(?:نص|نصف)\b", 3_500_000),
        (r"\b(?:تلات[ةه]|ثلاث[ةه])\s*(?:مليون|ملايين)\s*و\s*ربع\b", 3_250_000),
        (r"\b(?:اربع[ةه]|أربع[ةه])\s*(?:مليون|ملايين)\s*و\s*(?:نص|نصف)\b", 4_500_000),
        (r"\b(?:خمس[ةه])\s*(?:مليون|ملايين)\s*و\s*(?:نص|نصف)\b", 5_500_000),
        (r"\bمليون\s*و\s*(?:تلت\s*تربع|ثلاث[ةه]\s*ارباع|ثلاثة\s*أرباع)\b", 1_750_000),
        (r"\bمليون\s*و\s*(?:نص|نصف)\b", 1_500_000),
        (r"\bمليون\s*و\s*ربع\b", 1_250_000),
        (r"\b(?:تلت\s*تربع|ثلاثة\s*أرباع)\s*مليون\b", 750_000),
        (r"\b(?:نص|نصف)\s*مليون\b", 500_000),
        (r"\bربع\s*مليون\b", 250_000),
        (r"\b(?:مليونين|اتنين\s*مليون|اثنين\s*مليون)\b", 2_000_000),
        # Word-count Millions
        (r"\b(?:تلات[ةه]|ثلاث[ةه])\s*(?:مليون|ملايين)\b", 3_000_000),
        (r"\b(?:اربع[ةه]|أربع[ةه])\s*(?:مليون|ملايين)\b", 4_000_000),
        (r"\b(?:خمس[ةه])\s*(?:مليون|ملايين)\b", 5_000_000),
        (r"\b(?:ست[ةه])\s*(?:مليون|ملايين)\b", 6_000_000),
        (r"\b(?:سبع[ةه])\s*(?:مليون|ملايين)\b", 7_000_000),
        (r"\b(?:تمني[ةه]|ثماني[ةه])\s*(?:مليون|ملايين)\b", 8_000_000),
        (r"\b(?:تسع[ةه])\s*(?:مليون|ملايين)\b", 9_000_000),
        (r"\b(?:عشر[ةه])\s*(?:مليون|ملايين)\b", 10_000_000),
        (r"\b(?:خمستاشر|خمسة\s*عشر)\s*مليون\b", 15_000_000),
        (r"\bعشرين\s*مليون\b", 20_000_000),
        # Spoken Thousands (Egyptian dialect)
        (r"\b(?:ميتين|مائتين|مائتي|٢٠٠)\s*و\s*خمسين\s*(?:ألف|الف|آلاف|الاف)\b", 250_000),
        (r"\b(?:مية|مائة|ميت)\s*و\s*خمسين\s*(?:ألف|الف|آلاف|الاف)\b", 150_000),
        (r"\b(?:ميتين|مائتين|مائتي)\s*(?:ألف|الف|آلاف|الاف)\b", 200_000),
        (r"\b(?:تلتمية|تلتميت|ثلاثمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 300_000),
        (r"\b(?:ربعمية|ربعميت|أربعمائة|اربعمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 400_000),
        (r"\b(?:خمسمية|خمسميت|خمسمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 500_000),
        (r"\b(?:ستمية|ستميت|ستمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 600_000),
        (r"\b(?:سبعمية|سبعميت|سبعمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 700_000),
        (r"\b(?:تمنمية|تمنميت|ثمانمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 800_000),
        (r"\b(?:تسعمية|تسعميت|تسعمائة)\s*(?:ألف|الف|آلاف|الاف)\b", 900_000),
        (r"\b(?:مية|مائة|ميت)\s*(?:ألف|الف)\b", 100_000),
        (r"\bخمسين\s*(?:ألف|الف)\b", 50_000),
        (r"\bخمس[ةه]\s*و\s*عشرين\s*(?:ألف|الف)\b", 25_000),
        (r"\bتلاتين\s*(?:ألف|الف)\b", 30_000),
        (r"\b(?:أربعين|اربعين)\s*(?:ألف|الف)\b", 40_000),
        (r"\bستين\s*(?:ألف|الف)\b", 60_000),
        (r"\bسبعين\s*(?:ألف|الف)\b", 70_000),
        (r"\bتمانين\s*(?:ألف|الف)\b", 80_000),
        (r"\bتسعين\s*(?:ألف|الف)\b", 90_000),
    ]

    for pat, val in spoken_rules:
        # Allow optional attached Arabic preposition/conjunction prefix `ب` / `بـ` / `و` (e.g. `بميتين ألف`, `ومليون ونص`)
        flex_pat = pat.replace(r"\b", "")
        full_pat = rf"(?:(?<=\s)|^)(?:[بـ]+)?({flex_pat})(?=\s|$|[،.؟!])"
        for m in list(re.finditer(full_pat, s, flags=re.IGNORECASE)):
            conversions.append({"spoken_phrase": m.group(0).strip(), "numeric_egp": val})
        s = re.sub(full_pat, f" {val} ", s, flags=re.IGNORECASE)

    # Also handle `<digit> مليون ونص` -> `<digit + 0.5> مليون`
    def _num_and_half(match: re.Match) -> str:
        base = float(match.group(1).replace(",", ""))
        unit = match.group(2)
        frac = 0.5 if ("نص" in match.group(3)) else 0.25
        mult = 1_000_000 if any(k in unit for k in ("مليون", "ملايين")) else 1_000
        val = int(round((base + frac) * mult))
        conversions.append({"spoken_phrase": match.group(0), "numeric_egp": val})
        return f" {val} "

    s = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*(مليون|ملايين|ألف|الف)\s*و\s*(نص|نصف|ربع)",
        _num_and_half,
        s,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", s).strip(), conversions


def _predict_token_slots(tokens: List[str]) -> List[Tuple[str, float]]:
    bundle = _load_slot_bigru()
    if not bundle or not tokens:
        return [("O", 0.0) for _ in tokens]
    model, vocab, tags = bundle
    ids = []
    for tok in tokens:
        nt = _norm_token(tok)
        if re.fullmatch(r"\d+(?:[.,]\d+)?", nt):
            nt = "<NUM>"
        elif nt not in vocab and nt.startswith("و") and len(nt) >= 4 and nt[1:] in vocab:
            nt = nt[1:]
        ids.append(vocab.get(nt, vocab.get("<UNK>", 1)))
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0].numpy()
    out = []
    for p in probs:
        idx = int(np.argmax(p))
        out.append((tags[idx], round(float(p[idx]) * 100.0, 1)))
    return out


_NEG_WORDS = {"معليش", "معنديش", "مفيش", "بدون", "معلش", "لايوجد", "مدفعتش", "مبعتش", "مش", "عدم", "zero", "no", "without"}

_ROLE_LEXICON = {
    "REV": {"مبيعات", "مبيعاتي", "مبيعاته", "مبيعاتها", "مبيعاتنا", "ايراد", "ايرادات", "ايراداتي", "ايراداته", "ايراداتها", "دخل", "دخلي", "حجم", "اعمال", "ببيع", "بعمل", "باكسب", "شغلي", "يومي", "يوميتي", "revenue", "turnover", "sales"},
    "EXP": {"مصاريف", "مصاريفي", "مصاريفه", "مصاريفها", "مصاريفنا", "مصروفات", "مصروفاتي", "مصروفاته", "مصروفاتها", "تكاليف", "تكاليفي", "تكاليفها", "بدفع", "ادفع", "ايجار", "بضاعه", "خامات", "رواتب", "مرتبات", "نفقات", "expenses", "costs"},
    "PEN": {"غرامه", "غرامات", "تأخير", "تاخير", "فوائد", "عقوبات", "penalty", "penalties"},
    "PRINC": {"اصل", "أصل", "دين", "مطالبه", "نزاع", "principal", "dispute"},
    "RE_VAL": {"شقه", "عقار", "فيلا", "ارض", "بيت", "محل", "بعت", "بيع", "تصرف", "تصرفات", "عقاريه", "apartment", "property", "villa"},
    "VAT": {"مضافه", "المضافه", "رد", "دائن", "vat", "refund"},
}


def analyze_egyptian_dialect_xray(message: str) -> Dict[str, Any]:
    """
    Runs the full Neuro-Symbolic Dialect Grounding X-Ray on an Arabic/English query:
      1. Compiles spoken Egyptian numbers (`ميتين ألف`, `مليون ونص`, `نص مليون`).
      2. Tokenizes & runs the trained PyTorch BiGRU Sequence Slot Tagger across all tokens.
      3. Resolves negation scope (`معليش غرامات` -> negated PEN), temporal frequency (`في الشهر` -> ×12),
         and bidirectional role binding (works whether label is BEFORE or AFTER the number!).
      4. Returns rich explainable `spans`, `negations_honored`, and extracted `slots_egp`.
    """
    compiled_text, spoken_conversions = compile_egyptian_spoken_numbers(message or "")
    tokens = compiled_text.split()
    slot_preds = _predict_token_slots(tokens)

    # Detect negated roles (e.g. `معليش أي غرامات` or `بدون مصاريف` or `no penalties`)
    negated_roles: List[str] = []
    negation_notes: List[str] = []
    norm_tokens = [_norm_token(t) for t in tokens]

    for i, nt in enumerate(norm_tokens):
        nt_base = nt[1:] if (nt.startswith("و") and len(nt) >= 4) else nt
        if nt in _NEG_WORDS or nt_base in _NEG_WORDS or (nt in {"لا", "ولا"} and i + 1 < len(norm_tokens) and norm_tokens[i + 1] == "يوجد"):
            window = norm_tokens[i + 1 : min(len(norm_tokens), i + 5)]
            # Stop negation scope if a number immediately follows (e.g. `مش 20 مليون`)
            if any(re.fullmatch(r"\d+(?:[.,]\d+)?", w) for w in window[:2]):
                continue
            for role, lex in _ROLE_LEXICON.items():
                if any(w in lex for w in window):
                    if role not in negated_roles:
                        negated_roles.append(role)
                        phrase = " ".join(tokens[i : min(len(tokens), i + 4)])
                        negation_notes.append(f"تم رصد نفي صريح «{phrase}» ← تم تصفير ({SLOT_LABELS_AR.get(role, role)})")

    # Find all numeric mentions (with optional trailing unit `مليون / ألف / k / m`) and bind each to its semantic slot!
    unit_words = {
        "مليون": 1_000_000, "ملايين": 1_000_000, "million": 1_000_000, "mn": 1_000_000, "m": 1_000_000,
        "الف": 1_000, "ألف": 1_000, "الاف": 1_000, "آلاف": 1_000, "thousand": 1_000, "k": 1_000,
    }

    spans: List[Dict[str, Any]] = []
    slots_egp: Dict[str, int] = {}
    clause_boundaries = {"و", "لكن", "بس", "اما", "أو", "او", "بينما", "and", "but", "while"}

    def _is_clause_start_token(idx_tok: int) -> bool:
        nt = norm_tokens[idx_tok]
        if nt in clause_boundaries:
            return True
        # Arabic attached `و` conjunction starting a new financial clause (`ومصاريفي`, `وعليا`, `وليا`, `وبدفع`, `ومعليش`)
        if nt.startswith("و") and len(nt) >= 4:
            stem = nt[1:]
            if any(
                stem.startswith(root)
                for root in (
                    "مصاريف", "مصروف", "تكاليف", "نفقات", "مبيعات", "ايراد",
                    "عليا", "علي", "ليا", "بدفع", "بعمل", "معليش", "معنديش",
                    "مفيش", "بدون", "اصل", "غرام",
                )
            ):
                return True
        return False
    i = 0
    while i < len(tokens):
        clean_num = tokens[i].replace(",", "")
        m_num = re.fullmatch(r"(\d+(?:\.\d+)?)", clean_num)
        if not m_num:
            # Also check attached prefix like `ب4` or `بـ3`
            m_pref = re.fullmatch(r"[بـ]+(\d+(?:\.\d+)?)", clean_num)
            if m_pref:
                m_num = m_pref
            else:
                i += 1
                continue

        raw_val = float(m_num.group(1))
        end_i = i
        if i + 1 < len(tokens) and _norm_token(tokens[i + 1]) in unit_words:
            raw_val *= unit_words[_norm_token(tokens[i + 1])]
            end_i = i + 1

        # Skip statutory references like `قانون 6 لسنة 2025` or `151 لسنة 2020` or `7 أيام` or `5 سنوات`!
        prev_tok = _norm_token(tokens[i - 1]) if i > 0 else ""
        next_tok = _norm_token(tokens[end_i + 1]) if end_i + 1 < len(tokens) else ""
        if prev_tok in {"قانون", "رقم", "لسنه", "سنه", "ماده", "الماده", "عام", "منذ", "law", "no"} or next_tok in {"لسنه", "سنه", "ايام", "يوم", "يوما", "سنوات", "سنين", "اشهر", "شهور", "days", "years"}:
            i = end_i + 1
            continue
        if 1980 <= raw_val <= 2035 and prev_tok in {"سنه", "لسنه", "عام", "في", "سار", "من"}:
            i = end_i + 1
            continue
        # Skip negated contrast numbers (`مش 20 مليون` / `not 20 million`)
        if prev_tok in {"مش", "ليس", "ليست", "غير", "not"}:
            spans.append(
                {
                    "raw_phrase": " ".join(tokens[max(0, i - 1) : end_i + 1]),
                    "role_code": "NEGATED_NUMBER",
                    "role_ar": "رقم منفي (تم استبعاده تلقائياً)",
                    "extracted_egp": 0,
                    "formula_note": "نفي عددي صريح",
                    "confidence_pct": 99.0,
                    "negated": True,
                }
            )
            i = end_i + 1
            continue

        base_egp = int(round(raw_val))
        if base_egp < 500:
            i = end_i + 1
            continue

        # Determine semantic role using BiGRU Contextual Tagging + Bidirectional Clause Window!
        left_start = i
        while left_start > max(0, i - 4):
            if _is_clause_start_token(left_start):
                break
            if norm_tokens[left_start - 1] in clause_boundaries:
                break
            left_start -= 1
        right_end = end_i
        while right_end + 1 < min(len(tokens), end_i + 4):
            if _is_clause_start_token(right_end + 1):
                break
            right_end += 1

        # Score candidate roles from BiGRU predictions + lexicon cues in [left_start .. right_end]
        role_scores: Dict[str, float] = {r: 0.0 for r in SLOT_LABELS_AR}
        best_conf = 86.0

        # 1. BiGRU contextual predictions on the number token and immediate neighbors
        for pos in range(left_start, right_end + 1):
            pred_tag, pred_conf = slot_preds[pos]
            dist = 0 if (i <= pos <= end_i) else min(abs(pos - i), abs(pos - end_i))
            weight = 1.2 if dist == 0 else (1.0 / dist)
            if pred_tag in role_scores:
                role_scores[pred_tag] += (pred_conf / 100.0) * weight
                best_conf = max(best_conf, pred_conf)

        # 2. Bidirectional lexicon cues (handles BOTH `مبيعاتي 4 مليون` AND `4 مليون مبيعات`!)
        for pos in range(left_start, right_end + 1):
            if i <= pos <= end_i:
                continue
            nt = norm_tokens[pos]
            nt_stripped = nt[1:] if (nt.startswith("و") and len(nt) >= 4) else nt
            dist = min(abs(pos - i), abs(pos - end_i))
            prox_bonus = 3.2 / max(1, dist)
            for role, lex in _ROLE_LEXICON.items():
                if nt in lex or nt_stripped in lex:
                    role_scores[role] += prox_bonus
                    best_conf = max(best_conf, 97.5)

        chosen_role = max(role_scores.items(), key=lambda kv: kv[1])[0]
        if role_scores[chosen_role] < 0.25:
            chosen_role = "REV" if "REV" not in slots_egp else "EXP"
            best_conf = 78.0

        # Check Temporal Frequency Multiplier (`في الشهر / شهرياً` -> ×12, `في اليوم / يومياً` -> ×300, `أسبوعياً` -> ×52)
        freq_mult = 1
        freq_note = "قيمة سنوية مباشرة"
        freq_window = [norm_tokens[p] for p in range(left_start, right_end + 1)]
        has_explicit_annual = any(w in {"سنه", "السنه", "سنويا", "سنوي", "year", "annual", "annually"} for w in freq_window)
        if chosen_role in ("REV", "EXP") and not has_explicit_annual:
            if any(w in {"شهر", "الشهر", "شهري", "شهريا", "بالشهر", "month", "monthly"} for w in freq_window):
                freq_mult = 12
                freq_note = f"{base_egp:,} شهرياً × 12 شهر = {base_egp * 12:,} ج.م/سنة"
            elif any(w in {"اسبوع", "الاسبوع", "اسبوعيا", "اسبوعي", "week", "weekly"} for w in freq_window):
                freq_mult = 52
                freq_note = f"{base_egp:,} أسبوعياً × 52 أسبوع = {base_egp * 52:,} ج.م/سنة"
            elif any(w in {"يوم", "اليوم", "يوميا", "يومي", "يوميتي", "يوميه", "day", "daily"} for w in freq_window):
                freq_mult = 300
                freq_note = f"{base_egp:,} يومياً × 300 يوم عمل = {base_egp * 300:,} ج.م/سنة"

        final_egp = base_egp * freq_mult
        if chosen_role in negated_roles:
            final_egp = 0
            freq_note = "ملغى لوجود نفي صريح في الجملة"

        raw_span_text = " ".join(tokens[left_start : right_end + 1])
        spans.append(
            {
                "raw_phrase": raw_span_text,
                "role_code": chosen_role,
                "role_ar": SLOT_LABELS_AR.get(chosen_role, chosen_role),
                "extracted_egp": final_egp,
                "formula_note": freq_note,
                "confidence_pct": round(min(99.4, best_conf), 1),
                "negated": chosen_role in negated_roles,
            }
        )

        if chosen_role not in negated_roles:
            slots_egp[chosen_role] = slots_egp.get(chosen_role, 0) + final_egp

        i = end_i + 1

    return {
        "engine": "PyTorch BiGRU Sequence Slot Tagger + Spoken Egyptian Compiler",
        "bigru_loaded": _load_slot_bigru() is not None,
        "spoken_conversions": spoken_conversions,
        "negations_honored": negation_notes,
        "spans": spans,
        "slots_egp": slots_egp,
    }


def compute_cliff_edge_sensitivity_alert(annual_revenue_egp: int, annual_expenses_egp: int) -> Optional[str]:
    """
    Statutory Cliff-Edge & What-If Sensitivity Radar:
    Detects when a taxpayer sits near a critical statutory bracket threshold or the 20M/24M ceiling!
    """
    rev = int(annual_revenue_egp or 0)
    if rev <= 0:
        return None
    if 18_000_000 <= rev < 20_000_000:
        gap = 20_000_000 - rev
        return (
            f"📡 رادار الحوافز والحدود الحرجة: حجم أعمالك على بُعد {gap:,} ج.م فقط من سقف الـ 20 مليون جنيه — "
            "ميزة هامة مبرهنة في النظرية رقم (9) بنواة Lean 4: يمنحك القانون 6 لسنة 2025 سماح تجاوز لمرة واحدة بنسبة 20% "
            "(حتى 24,000,000 ج.م) مع الاحتفاظ بشريحة 1.5% دون فقدان النظام المبسط!"
        )
    if 20_000_000 <= rev <= 24_000_000:
        return (
            "📡 رادار الحوافز والحدود الحرجة: حجم أعمالك بين 20 و24 مليون جنيه — إذا كنت منضماً مسبقاً للنظام المبسط "
            "وتجاوزت السقف لأول مرة هذا العام، يحق لك البقاء في شريحة 1.5% لسنة إضافية وفق قاعدة الـ 20% (النظرية 9 في Lean 4)!"
        )
    thresholds = [(500_000, "0.4%", "0.5%"), (2_000_000, "0.5%", "0.75%"), (3_000_000, "0.75%", "1.0%"), (10_000_000, "1.0%", "1.5%")]
    for bound, cur_r, nxt_r in thresholds:
        if int(bound * 0.90) <= rev < bound:
            return (
                f"📡 رادار الشرائح الحرجة: أنت حالياً في شريحة {cur_r} وعلى بُعد {bound - rev:,} ج.م من الانتقال لشريحة {nxt_r} ({bound:,} ج.م)."
            )
    return None
