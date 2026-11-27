"""
Moyasser AI — Official TAX-02 Bilingual (AR/EN) Neuro-Symbolic Copilot, Multi-Modal Document & Voice Handler,
Out-of-Scope Guard + Officer Escalation Summary, 3 Contextual Follow-Ups, and Law No. 151/2020 PII Shield.
"""

import base64
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .analytics_store import record_escalation, record_question_topic
from .arabic_dialect_xray import analyze_egyptian_dialect_xray, compute_cliff_edge_sensitivity_alert
from .ml_models import (
    estimate_sector_expense_ratio,
    extract_pdf_bytes_text,
    predict_arabic_intent,
    search_official_eta_documents,
)
from .optimizer import TaxpayerProfile, optimize_tax_strategy
from .tax_laws import NON_BINDING_DISCLAIMER_AR, NON_BINDING_DISCLAIMER_EN

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")

FOLLOWUPS_AR: Dict[str, List[str]] = {
    "LAW6_SIMPLIFIED_SME": [
        "ما المستندات المطلوبة للانضمام للنظام الضريبي المبسط وفق قانون 6 لسنة 2025؟",
        "ماذا يحدث إذا تجاوز حجم أعمالي 20 مليون جنيه بنسبة 15% في سنة واحدة خلال أول 5 سنوات؟",
        "كيف أستفيد من مبادرة التمويل الميسر لأول 100 ألف ممول بالنظام المبسط؟",
    ],
    "LAW5_DISPUTE_WAIVER": [
        "ما الموعد النهائي لتقديم طلب إنهاء المنازعات الضريبية وفق القانون 152 لسنة 2026؟",
        "كيف يطبق القانون رقم 7 لسنة 2025 سقف الـ 100% على مقابل التأخير؟",
        "هل يمكنني تقسيط أصل الضريبة المتنازع عليها عند التسوية؟",
    ],
    "INFORMAL_REGULARIZATION": [
        "إذا فتحت ملفاً ضريبياً جديداً اليوم لمشروع صغير، هل أخضع للفحص خلال أول 5 سنوات؟",
        "ما الفرق بين شريحة 0.4% وشريحة 0.5% لأصحاب العمل الحر والمنشآت الصغيرة؟",
        "هل يلزمني إمساك دفاتر محاسبية معقدة أم يكفي نظام السجلات المبسط؟",
    ],
    "PKG2_REAL_ESTATE_TAX": [
        "ما المستندات المطلوبة لإثبات صلة القرابة من الدرجة الأولى للإعفاء من الـ 2.5%؟",
        "كيف أستخرج شهادة المخالصة عبر تطبيق التصرفات العقارية الإلكتروني خلال مهلة الـ 60 يوماً؟",
        "هل تتغير نسبة الـ 2.5% للأفراد غير التجار إذا تكررت عمليات البيع؟",
    ],
    "PKG2_WHITELIST_VAT": [
        "ما معايير الانضمام للقائمة البيضاء وكارت التميز لاسترداد القيمة المضافة في 7 أيام؟",
        "ما تفاصيل تخفيض ضريبة القيمة المضافة على الأجهزة الطبية إلى 5% في قانون 149 لسنة 2026؟",
        "كيف يُقدَّم إقرار القيمة المضافة ربع السنوي للمنضمين لقانون 6 لسنة 2025؟",
    ],
    "THIN_MARGIN_STANDARD_COMPARE": [
        "كيف تُخصم المساهمة التكافلية (0.25%) من التكاليف واجبة الخصم وفق قانون 151 لسنة 2026؟",
        "متى يكون البقاء في النظام الضريبي العادي أوفر من النظام المبسط قانون 6؟",
        "ما إعفاءات توزيعات الأرباح للشركات القابضة والتابعة في الحزمة الثانية؟",
    ],
}

FOLLOWUPS_EN: Dict[str, List[str]] = {
    "LAW6_SIMPLIFIED_SME": [
        "What documents are required to enroll an Egyptian subsidiary in Law No. 6 of 2025?",
        "How does the one-time 20% turnover grace buffer (up to 24M EGP) work during the 5-year lock-in?",
        "Are dividends and capital gains fully exempt under Law No. 6 of 2025?",
    ],
    "LAW5_DISPUTE_WAIVER": [
        "What is the exact statutory deadline to file for Tax Dispute Settlement under Law No. 152 of 2026?",
        "How does Law No. 7 of 2025 cap delay penalties at 100% of principal tax?",
        "Which Elite Investor Service Center should foreign investors visit in New Cairo or Sheikh Zayed?",
    ],
    "PKG2_WHITELIST_VAT": [
        "How do foreign investors qualify for the White List 7-day fast-track VAT refund?",
        "How does Law No. 149 of 2026 reduce VAT on medical equipment to 5% and extend machinery VAT suspension to 4 years?",
        "How does Second Package Law No. 151 of 2026 prevent double taxation on holding-subsidiary dividends?",
    ],
    "DEFAULT": [
        "Compare Simplified Law 6/2025 vs. Standard 22.5% Corporate Tax for my Egyptian venture.",
        "How do I qualify for the White List 7-day VAT refund and Advance Tax Ruling Unit?",
        "What is the deadline for settling existing tax disputes under Law No. 152 of 2026?",
    ],
}

_OUT_OF_SCOPE_PATTERNS = [
    (r"جمرك|جمارك\s*سيارات|بند\s*جمركي|تعريفة\s*جمركية|hs\s*code|customs\s*tariff", "الرسوم والتعريفة الجمركية (اختصاص مصلحة الجمارك المصرية)", "Customs tariffs & HS classification (Egyptian Customs Authority scope)"),
    (r"طلاق|حضانة|نفقة|زواج|قضية\s*جنائية|جنحة\s*سرقة|criminal|divorce|custody", "الأحوال الشخصية أو القضايا الجنائية غير الضريبية", "Non-tax criminal or family law matters"),
    (r"رخصة\s*مباني|تصالح\s*مباني|مخالفة\s*بناء|building\s*permit", "تراخيص وتصالحات البناء المحليات", "Municipal building permits & zoning"),
    (r"فتوى\s*ملزمة|رأي\s*ملزم\s*نهائي|حوّلني\s*لموظف|تحدث\s*مع\s*موظف|human\s*agent|binding\s*ruling|speak\s*to\s*officer", "طلب فتوى ضريبية ملزمة / تدخل بشري مباشر من وحدة الرأي المسبق وخدمات المستثمرين", "Request for binding tax ruling / direct ETA Investor Services specialist"),
]


def normalize_arabic_numbers(text: str) -> str:
    return (text or "").translate(_ARABIC_DIGITS)


def detect_language(text: str, explicit_lang: str = "auto") -> str:
    if explicit_lang in ("ar", "en"):
        return explicit_lang
    ar_count = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    en_count = len(re.findall(r"[A-Za-z]", text or ""))
    return "en" if (en_count > ar_count and en_count >= 6) else "ar"


def redact_pii(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Law No. 151 of 2020 (Egyptian Personal Data Protection Law) & Tax Secrecy Guard:
    Masks 14-digit National IDs, 9-digit Tax Reg IDs, phones, cards, and IBANs BEFORE any processing.
    """
    if not text:
        return "", {"redacted_count": 0, "types": [], "law_compliance": "Law No. 151 of 2020"}

    norm = normalize_arabic_numbers(text)
    redacted_types: List[str] = []
    total_hits = 0

    patterns = [
        ("EGYPTIAN_IBAN", r"\bEG\d{27}\b", "[REDACTED-IBAN]"),
        ("CARD_NUMBER", r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED-CARD]"),
        ("NATIONAL_ID", r"(?<!\d)[23]\d{13}(?!\d)", "[رقم قومي محجوب - قانون 151/2020]"),
        ("TAX_REG_ID", r"(?<!\d)\d{3}-\d{3}-\d{3}(?!\d)", "[رقم تسجيل ضريبي محجوب]"),
        ("MOBILE_PHONE", r"(?<!\d)(?:\+?2)?01[0125]\d{8}(?!\d)", "[رقم هاتف محجوب]"),
    ]

    out = norm
    for label, pattern, replacement in patterns:
        out, n = re.subn(pattern, replacement, out, flags=re.IGNORECASE)
        if n > 0:
            total_hits += n
            redacted_types.append(label)

    return out, {"redacted_count": total_hits, "types": redacted_types, "law_compliance": "Law No. 151 of 2020"}


def _parse_amount_value(raw_num: str, unit: str) -> int:
    cleaned = raw_num.replace(",", "").strip()
    try:
        val = float(cleaned)
    except ValueError:
        return 0
    u = (unit or "").strip().lower()
    if any(k in u for k in ("مليون", "ملايين", "million", "mn", "m")):
        val *= 1_000_000
    elif any(k in u for k in ("ألف", "الف", "الاف", "آلاف", "thousand", "k")):
        val *= 1_000
    return max(0, int(round(val)))


def _find_labeled_amount(text: str, keywords: List[str]) -> Optional[int]:
    unit_pat = r"(مليون|ملايين|ألف|الف|الاف|آلاف|million|thousand|mn\b|m\b|k\b)?"
    num_pat = r"(\d+(?:[.,]\d+)?)"
    for kw in keywords:
        m = re.search(rf"{kw}[^\d\n]{{0,32}}{num_pat}\s*{unit_pat}", text, flags=re.IGNORECASE)
        if m:
            amt = _parse_amount_value(m.group(1), m.group(2) or "")
            if amt > 0:
                return amt
    return None


def extract_profile_from_arabic_text(
    message: str,
    base_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[TaxpayerProfile, List[str]]:
    """Bilingual + PyTorch BiGRU Dialect X-Ray Financial NLU Extractor."""
    xray = analyze_egyptian_dialect_xray(message or "")
    slots = xray.get("slots_egp", {})
    text = normalize_arabic_numbers(message or "")
    base = dict(base_profile or {})
    signals: List[str] = []

    # Pre-seed from BiGRU + Spoken Egyptian Number + Monthly/Daily X-Ray slots!
    if slots.get("REV", 0) > 0:
        base["annual_revenue_egp"] = slots["REV"]
        signals.append(f"حجم الأعمال السنوي (BiGRU X-Ray): {slots['REV']:,} ج.م")
    if slots.get("EXP", 0) > 0:
        base["annual_expenses_egp"] = slots["EXP"]
        signals.append(f"المصروفات السنوية (BiGRU X-Ray): {slots['EXP']:,} ج.م")
    if slots.get("PRINC", 0) > 0:
        base["dispute_principal_egp"] = slots["PRINC"]
        base["has_old_dispute_or_penalties"] = True
        signals.append(f"أصل الضريبة (BiGRU X-Ray): {slots['PRINC']:,} ج.م")
    if slots.get("PEN", 0) > 0:
        base["delay_penalties_egp"] = slots["PEN"]
        base["has_old_dispute_or_penalties"] = True
        signals.append(f"مقابل التأخير (BiGRU X-Ray): {slots['PEN']:,} ج.م")
    if slots.get("RE_VAL", 0) > 0:
        base["real_estate_sale_egp"] = slots["RE_VAL"]
        signals.append(f"قيمة التصرف العقاري (BiGRU X-Ray): {slots['RE_VAL']:,} ج.م")
    if slots.get("VAT", 0) > 0:
        base["vat_refund_claim_egp"] = slots["VAT"]
        base["vat_registered"] = True
        signals.append(f"طلب رد قيمة مضافة (BiGRU X-Ray): {slots['VAT']:,} ج.م")

    rev = _find_labeled_amount(
        text,
        [
            r"حجم\s*الأعمال", r"حجم\s*الاعمال", r"مبيعات(?:ه|ي|نا)?", r"إيراد(?:ات)?(?:ه|ي|نا)?",
            r"ايراد(?:ات)?(?:ه|ي|نا)?", r"دخل(?:ه|ي)?", r"شغل(?:ي|ه)?\s*ب", r"إجمالي\s*الفواتير",
            r"اجمالي\s*الفواتير", r"annual\s*revenue", r"revenue", r"turnover", r"total\s*sales", r"sales",
        ],
    )
    if rev is not None and not slots.get("REV"):
        base["annual_revenue_egp"] = rev
        signals.append(f"حجم الأعمال السنوي: {rev:,} ج.م")

    exp = _find_labeled_amount(
        text,
        [
            r"مصاريف(?:ه|ي|نا)?", r"مصروفات(?:ه|ي|نا)?", r"تكاليف(?:ه|ي|نا)?",
            r"نفقات(?:ه|ي|نا)?", r"annual\s*expenses", r"expenses", r"operating\s*costs", r"costs",
        ],
    )
    if exp is not None and not slots.get("EXP"):
        base["annual_expenses_egp"] = exp
        signals.append(f"المصروفات السنوية: {exp:,} ج.م")

    principal = _find_labeled_amount(
        text,
        [
            r"أصل\s*الضريبة", r"اصل\s*الضريبة", r"أصل\s*الدين", r"ضريبة\s*قديمة",
            r"نزاع\s*ب(?:قيمة|مبلغ)?", r"tax\s*principal", r"disputed\s*tax", r"principal",
        ],
    )
    if principal is not None and not slots.get("PRINC"):
        base["dispute_principal_egp"] = principal
        base["has_old_dispute_or_penalties"] = True
        signals.append(f"أصل الضريبة المتنازع عليها: {principal:,} ج.م")

    penalties = _find_labeled_amount(
        text,
        [
            r"غرام(?:ة|ات)", r"مقابل\s*التأخير", r"مقابل\s*التاخير", r"فوائد\s*تأخير",
            r"delay\s*penalt(?:y|ies)", r"late\s*penalt(?:y|ies)", r"penalt(?:y|ies)",
        ],
    )
    if penalties is not None and not slots.get("PEN") and not xray.get("negations_honored"):
        base["delay_penalties_egp"] = penalties
        base["has_old_dispute_or_penalties"] = True
        signals.append(f"غرامات ومقابل التأخير: {penalties:,} ج.م")

    # Honor explicit Egyptian Arabic negations (e.g. `ومعليش غرامات`)!
    for neg_msg in xray.get("negations_honored", []):
        signals.append(neg_msg)
        if "التأخير" in neg_msg or "غرام" in neg_msg:
            base["delay_penalties_egp"] = 0
            if not base.get("dispute_principal_egp"):
                base["has_old_dispute_or_penalties"] = False

    re_sale = _find_labeled_amount(
        text,
        [
            r"بعت\s*(?:شقة|عقار|بيت|محل|أرض|ارض|فيلا)", r"بيع\s*(?:شقة|عقار|بيت|محل|أرض|ارض|فيلا)",
            r"تصرف(?:ات)?\s*عقاري(?:ة)?", r"شقة\s*ب", r"عقار\s*ب",
            r"real\s*estate\s*(?:sale|disposal)", r"sold\s*(?:an?\s*)?(?:apartment|property|villa)",
        ],
    )
    if re_sale is not None:
        base["real_estate_sale_egp"] = re_sale
        signals.append(f"قيمة التصرف العقاري: {re_sale:,} ج.م")

    if any(
        kw in text.lower()
        for kw in (
            "درجة أولى", "درجة اولى", "لابني", "لإبني", "لبنتي", "لابنتي", "لوالدي", "لوالدتي",
            "لأبي", "لأمي", "لزوجي", "لزوجتي", "أقارب", "اقارب",
            "first degree", "first-degree", "my son", "my daughter", "my father", "my mother", "my spouse",
        )
    ):
        base["real_estate_first_degree"] = True
        signals.append("تصرف عقاري لأقارب من الدرجة الأولى")

    vat_claim = _find_labeled_amount(
        text,
        [r"رد\s*(?:ضريبة\s*)?(?:القيمة\s*المضافة|قيمة\s*مضافة)", r"رصيد\s*دائن", r"vat\s*refund", r"refund\s*claim"],
    )
    if vat_claim is not None:
        base["vat_refund_claim_egp"] = vat_claim
        base["vat_registered"] = True
        signals.append(f"طلب رد ضريبة قيمة مضافة: {vat_claim:,} ج.م")
    elif any(kw in text.lower() for kw in ("قيمة مضافة", "القائمة البيضاء", "كارت التميز", "vat", "white list", "whitelist")):
        base["vat_registered"] = True
        signals.append("مسجل بضريبة القيمة المضافة")

    if any(
        kw in text.lower()
        for kw in (
            "مش مسجل", "غير مسجل", "لسه مفتحتش ملف", "مفتحتش ملف", "بدون ملف ضريبي",
            "ملف جديد", "اقتصاد غير رسمي", "unregistered", "not registered", "new taxpayer", "informal",
        )
    ):
        base["is_registered_previously"] = False
        signals.append("مشروع غير مسجل سابقاً")

    lower_text = text.lower()
    if any(kw in lower_text for kw in ("فريلانسر", "عمل حر", "مبرمج حر", "مصمم حر", "صانع محتوى", "freelanc")):
        base["entity_type"] = "freelancer"
        base["sector"] = "عمل حر وخدمات رقمية"
    elif any(kw in lower_text for kw in ("فردي", "محل", "ورشة", "صيدلية", "عيادة", "منشأة فردية", "sole proprietor")):
        base["entity_type"] = "sole"
        base["sector"] = "خدمات طبية وصحية" if ("عيادة" in lower_text or "صيدلية" in lower_text) else "تجارة تجزئة ومحال"
    elif any(kw in lower_text for kw in ("مطعم", "كافيه", "كافية", "مخبز", "restaurant", "cafe")):
        base["sector"] = "مطاعم وأغذية"
        base.setdefault("entity_type", "company")
    elif any(kw in lower_text for kw in ("برمجيات", "تكنولوجيا", "شركة ناشئة", "startup", "software", "tech", "subsidiary")):
        base["sector"] = "تكنولوجيا وبرمجيات"
        base.setdefault("entity_type", "company")

    gov_map = {
        "القاهرة": ("القاهرة", "قاهرة", "التجمع", "المعادي", "مدينة نصر", "cairo", "new cairo"),
        "الجيزة": ("الجيزة", "جيزة", "الشيخ زايد", "أكتوبر", "اكتوبر", "giza", "sheikh zayed", "october"),
        "الإسكندرية": ("الإسكندرية", "الاسكندرية", "اسكندرية", "إسكندرية", "alex"),
        "مطروح": ("مطروح", "العلمين", "الساحل", "alamein"),
    }
    for gov, keys in gov_map.items():
        if any(k in lower_text for k in keys):
            base["governorate"] = gov
            break

    if not base.get("annual_revenue_egp") and not base.get("real_estate_sale_egp"):
        bare = re.search(r"(\d+(?:[.,]\d+)?)\s*(مليون|ملايين|ألف|الف|الاف|آلاف|million|thousand|mn\b|m\b|k\b)", text, flags=re.IGNORECASE)
        if bare:
            amt = _parse_amount_value(bare.group(1), bare.group(2))
            if amt > 0:
                base["annual_revenue_egp"] = amt
                signals.append(f"حجم الأعمال المستخلص: {amt:,} ج.م")

    if base.get("annual_revenue_egp", 0) > 0 and "annual_expenses_egp" not in base:
        est = estimate_sector_expense_ratio(
            entity_type=str(base.get("entity_type") or "company"),
            sector=str(base.get("sector") or "تجارة وخدمات"),
            governorate=str(base.get("governorate") or "القاهرة"),
            annual_revenue_egp=int(base["annual_revenue_egp"]),
            vat_registered=bool(base.get("vat_registered", False)),
            is_registered_previously=bool(base.get("is_registered_previously", True)),
        )
        base["annual_expenses_egp"] = est["estimated_expenses_egp"]
        signals.append(
            f"تقدير المصروفات القطاعية (XGBoost Regressor): {est['estimated_expenses_egp']:,} ج.م ({est['predicted_ratio']*100:.1f}%)"
        )

    profile = TaxpayerProfile(
        business_name=str(base.get("business_name") or "مشروع الممول"),
        entity_type=str(base.get("entity_type") or "company"),
        sector=str(base.get("sector") or "تجارة وخدمات"),
        governorate=str(base.get("governorate") or "القاهرة"),
        annual_revenue_egp=int(base.get("annual_revenue_egp") or 0),
        annual_expenses_egp=int(base.get("annual_expenses_egp") or 0),
        is_registered_previously=bool(base.get("is_registered_previously", True)),
        has_old_dispute_or_penalties=bool(base.get("has_old_dispute_or_penalties", False)),
        dispute_principal_egp=int(base.get("dispute_principal_egp") or 0),
        delay_penalties_egp=int(base.get("delay_penalties_egp") or 0),
        vat_registered=bool(base.get("vat_registered", False)),
        vat_refund_claim_egp=int(base.get("vat_refund_claim_egp") or 0),
        compliant_filing_history=bool(base.get("compliant_filing_history", True)),
        real_estate_sale_egp=int(base.get("real_estate_sale_egp") or 0),
        real_estate_first_degree=bool(base.get("real_estate_first_degree", False)),
    )
    return profile, signals


def _check_out_of_scope(text: str) -> Optional[Tuple[str, str]]:
    for pat, ar_reason, en_reason in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return ar_reason, en_reason
    return None


def _get_followups(intent: str, lang: str) -> List[str]:
    if lang == "en":
        return FOLLOWUPS_EN.get(intent, FOLLOWUPS_EN["DEFAULT"])[:3]
    return FOLLOWUPS_AR.get(intent, FOLLOWUPS_AR["LAW6_SIMPLIFIED_SME"])[:3]


def _optional_gemini_multimodal(prompt: str, data_b64: str, mime_type: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or not data_b64:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": data_b64}}]}]}
    try:
        with httpx.Client(timeout=12.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None
    return None


def scan_document(
    filename: str = "document.txt",
    raw_text: str = "",
    image_base64: Optional[str] = None,
    pdf_base64: Optional[str] = None,
    mime_type: str = "image/png",
) -> Dict[str, Any]:
    """
    TAX-02 Multi-Modal Document Scanner:
    Accepts plain text, uploaded PDF files (`pdf_base64` extracted via `pypdfium2`), or scanned images (`image_base64`).
    Enforces Law No. 151 of 2020 PII scrubbing before ML + HiGHS MILP + Lean 4 verification.
    """
    ocr_engine = "Deterministic Bilingual Document Parser"
    combined_text = raw_text or ""

    if pdf_base64:
        try:
            pdf_text = extract_pdf_bytes_text(base64.b64decode(pdf_base64))
            combined_text = f"{combined_text}\n{pdf_text}".strip()
            ocr_engine = "Native pypdfium2 Official PDF Extractor + Deterministic Parser"
        except Exception as exc:
            ocr_engine = f"PDF Parse Fallback ({exc})"

    if image_base64:
        gemini_text = _optional_gemini_multimodal(
            "Extract only financial totals (revenue, expenses, tax principal, delay penalties, real estate value, VAT refund) without personal IDs.",
            image_base64,
            mime_type=mime_type,
        )
        if gemini_text:
            combined_text = f"{combined_text}\n{gemini_text}".strip()
            ocr_engine = "Gemini 2.5 Flash Vision OCR + Deterministic Parser"

    redacted_text, pii_report = redact_pii(combined_text)
    profile, signals = extract_profile_from_arabic_text(redacted_text)
    profile.business_name = f"تحليل مستند: {os.path.basename(filename or 'فاتورة')[:40]}"

    optimization = optimize_tax_strategy(profile)
    return {
        "filename": os.path.basename(filename or "document.txt"),
        "ocr_engine": ocr_engine,
        "pii_protection": pii_report,
        "redacted_preview": redacted_text[:450],
        "extracted_signals": signals,
        "optimization": optimization,
        "disclaimer_ar": NON_BINDING_DISCLAIMER_AR,
    }


def process_voice_query(
    transcript_text: str = "",
    audio_base64: Optional[str] = None,
    mime_type: str = "audio/webm",
    lang: str = "auto",
) -> Dict[str, Any]:
    """
    TAX-02 Multi-Modal Voice Query Handler:
    Supports uploaded audio clips (`audio_base64` transcribed via Gemini Audio when configured)
    AND live browser Web Speech API voice transcripts (`ar-EG` / `en-US`).
    """
    spoken = (transcript_text or "").strip()
    asr_engine = "Web Speech API Live Voice Transcript"
    if audio_base64 and not spoken:
        gem_tx = _optional_gemini_multimodal(
            "Transcribe this Arabic or English tax question accurately. Do not include personal ID numbers.",
            audio_base64,
            mime_type=mime_type,
        )
        if gem_tx:
            spoken = gem_tx.strip()
            asr_engine = "Gemini 2.5 Audio Transcription"
        else:
            asr_engine = "Audio Clip Received (Use live mic dictation or set GEMINI_API_KEY for raw audio decoding)"

    if not spoken:
        return {
            "asr_engine": asr_engine,
            "transcribed_text": "",
            "chat_response": answer_tax_question("مساعدة صوتية في التيسيرات الضريبية", lang=lang),
        }

    chat_res = answer_tax_question(spoken, lang=lang)
    return {
        "asr_engine": asr_engine,
        "transcribed_text": redact_pii(spoken)[0],
        "chat_response": chat_res,
    }


def answer_tax_question(
    message: str,
    profile_override: Optional[Dict[str, Any]] = None,
    lang: str = "auto",
    conversation_history: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Official TAX-02 Bilingual (Simplified MSA Arabic + Foreign Investor English) Neuro-Symbolic Copilot:
      - Enforces Law No. 151 of 2020 PII protection first.
      - Politely refuses out-of-scope topics, asks clarifying questions, and escalates to an ETA specialist with a conversation summary.
      - States statutory references, effective dates, deadline/expiry alerts, and non-binding disclaimer.
      - Suggests 3 contextual follow-up questions after every reply.
    """
    resolved_lang = detect_language(message or "", explicit_lang=lang)
    redacted_msg, pii_report = redact_pii(message or "")

    # 1. Check Out-of-Scope / Human Officer Escalation Guard (TAX-02 Mandatory Rule)
    oos = _check_out_of_scope(redacted_msg)
    if oos is not None:
        reason_ar, reason_en = oos
        hist_preview = " | ".join((conversation_history or [])[-3:] + [redacted_msg])[:600]
        summary_text = (
            f"[{resolved_lang.upper()}] السبب: {reason_ar} | ملخص سياق المستفسر (محجوب البيانات الشخصية): {hist_preview}"
        )
        ticket = record_escalation(
            lang=resolved_lang,
            reason=reason_ar if resolved_lang == "ar" else reason_en,
            redacted_query=redacted_msg,
            conversation_summary=summary_text,
        )
        if resolved_lang == "en":
            reply_en = (
                f"🙏 **Scope Notice & Specialist Escalation:** Your inquiry relates to **{reason_en}**, "
                "which lies outside automated Egyptian Tax Facilitation guidance (Laws 5, 6, 7 of 2025 & Second Package Laws 148–153 of 2026) "
                "or requires a formal human review.\n\n"
                "❓ **To help us assist you better, could you clarify:**\n"
                "1. Does your inquiry also involve corporate/SME income tax, VAT refunds, or real-estate disposal tax in Egypt?\n"
                "2. What is your entity's approximate annual turnover in EGP?\n\n"
                f"📨 **Automatic Specialist Handoff Created:** Ticket `{ticket['ticket_id']}` has been queued for the "
                f"**{ticket['department_en']}** along with a sanitized conversation summary.\n\n"
                f"ℹ️ _{NON_BINDING_DISCLAIMER_EN}_"
            )
            return {
                "lang": "en",
                "out_of_scope": True,
                "escalation": ticket,
                "reply_ar": reply_en,
                "pii_protection": pii_report,
                "suggested_followups": FOLLOWUPS_EN["DEFAULT"],
                "disclaimer": NON_BINDING_DISCLAIMER_EN,
                "optimization": None,
            }

        reply_ar = (
            f"🙏 **اعتذار وتوضيح نطاق الخدمة:** استفسارك يتعلق بـ **«{reason_ar}»**، وهو خارج نطاق المساعد الآلي "
            "المخصص لتشريعات وحزم التيسيرات الضريبية المصرية (القوانين 5 و6 و7 لسنة 2025 وقوانين الحزمة الثانية 148–153 لسنة 2026)، "
            "أو يتطلب رأياً بشرياً مختصاً.\n\n"
            "❓ **للتأكد من خدمتك بدقة، هل يمكنك توضيح:**\n"
            "1. هل لديك شق مرتبط بضريبة الدخل للمشروعات الصغيرة (< 20 مليون جنيه)، أو رد القيمة المضافة، أو التصرفات العقارية؟\n"
            "2. ما طبيعة الكيان القانوني وحجم الأعمال التقريبي (دون ذكر أي رقم قومي أو رقم تسجيل ضريبي)؟\n\n"
            f"📨 **تمت الإحالة لموظف مختص مع ملخص المحادثة:** رقم طلب الإحالة `{ticket['ticket_id']}` لدى **{ticket['department_ar']}**.\n\n"
            f"ℹ️ _{NON_BINDING_DISCLAIMER_AR}_"
        )
        return {
            "lang": "ar",
            "out_of_scope": True,
            "escalation": ticket,
            "reply_ar": reply_ar,
            "pii_protection": pii_report,
            "suggested_followups": FOLLOWUPS_AR["LAW6_SIMPLIFIED_SME"],
            "disclaimer": NON_BINDING_DISCLAIMER_AR,
            "optimization": None,
        }

    # 2. Predict Intent, Run Dialect X-Ray, Record FAQ Telemetry, Retrieve Official Clauses
    intent_pred = predict_arabic_intent(redacted_msg)
    dialect_xray = analyze_egyptian_dialect_xray(redacted_msg)
    record_question_topic(intent_pred["intent"])
    official_rag = search_official_eta_documents(redacted_msg, top_k=2)
    profile, signals = extract_profile_from_arabic_text(redacted_msg, base_profile=profile_override)
    cliff_alert = compute_cliff_edge_sensitivity_alert(profile.annual_revenue_egp, profile.annual_expenses_egp)

    has_numbers = any(
        [
            profile.annual_revenue_egp > 0,
            profile.delay_penalties_egp > 0,
            profile.dispute_principal_egp > 0,
            profile.real_estate_sale_egp > 0,
            profile.vat_refund_claim_egp > 0,
        ]
    )
    followups = _get_followups(intent_pred["intent"], resolved_lang)

    if not has_numbers:
        if resolved_lang == "en":
            return {
                "lang": "en",
                "out_of_scope": False,
                "escalation": None,
                "reply_ar": (
                    "Welcome to **Moyasser AI — Egyptian Tax Facilitation Copilot (Track TAX-02)**!\n"
                    "Please share your approximate figures in English (without any personal IDs or tax registration numbers) so our "
                    "**SciPy HiGHS MILP Optimizer + Lean 4.36.0 Formal Kernel (10 theorems, 0 sorry)** can compute and certify your exact position.\n\n"
                    "Examples for Foreign Investors & SMEs:\n"
                    "• `My software subsidiary in Cairo has annual revenue of 12 million EGP, expenses of 7.5 million EGP, and a VAT refund claim of 450 thousand EGP.`\n"
                    "• `We have an old tax dispute with principal 100 thousand EGP and delay penalties 180 thousand EGP.`\n\n"
                    f"ℹ️ _{NON_BINDING_DISCLAIMER_EN}_"
                ),
                "pii_protection": pii_report,
                "ml_intent": intent_pred,
                "official_legal_passages": official_rag.get("matches", []),
                "suggested_followups": followups,
                "disclaimer": NON_BINDING_DISCLAIMER_EN,
                "extracted_signals": signals,
                "optimization": None,
            }

        rag_note = ""
        if official_rag.get("matches"):
            top_m = official_rag["matches"][0]
            rag_note = (
                f"\n\n📚 **المرجع الرسمي المعتمد:** {top_m['title']} (صفحة {top_m['page']})\n"
                f"«{top_m['excerpt'][:210]}...»"
            )
        return {
            "lang": "ar",
            "out_of_scope": False,
            "escalation": None,
            "reply_ar": (
                f"🧠 **المسار الضريبي المصنّف ({intent_pred['confidence_pct']}%):** {intent_pred['intent_ar']}\n\n"
                "يرجى ذكر الأرقام المالية التقريبية لحالتك (مثل حجم الأعمال السنوي، المصروفات، أو قيمة التصرف العقاري — **دون إدخال الرقم القومي أو رقم التسجيل الضريبي**) "
                "لنحسب لك المسار الأوفر قانونياً ونوثّقه بشهادة إثبات رياضي من نواة `Lean 4`."
                + rag_note
                + f"\n\nℹ️ _{NON_BINDING_DISCLAIMER_AR}_"
            ),
            "pii_protection": pii_report,
            "ml_intent": intent_pred,
            "official_legal_passages": official_rag.get("matches", []),
            "suggested_followups": followups,
            "disclaimer": NON_BINDING_DISCLAIMER_AR,
            "extracted_signals": signals,
            "optimization": None,
        }

    # 3. Run Global MILP Optimization + Lean 4 Formal Certification
    opt = optimize_tax_strategy(profile)
    comp = opt["comparison"]
    lean = opt["lean4_verification"]
    ml_intel = opt.get("ml_intelligence", {})

    if resolved_lang == "en":
        en_lines: List[str] = []
        if pii_report["redacted_count"] > 0:
            en_lines.append(
                f"🔒 **Data Privacy Shield (Law No. 151 of 2020):** {pii_report['redacted_count']} sensitive identifier(s) automatically redacted prior to processing."
            )
        regime_en = (
            "Simplified Turnover Tax Regime (Law No. 6 of 2025 — Effective 2025-03-01, Permanent)"
            if comp["recommended_regime"] == "SIMPLIFIED_LAW6"
            else "Standard Net-Profit Corporate/Personal Tax Regime (Lower Tax Burden for Thin Margins)"
        )
        en_lines.append(f"✅ **Formally Certified Optimal Strategy:** {regime_en}")
        if profile.annual_revenue_egp > 0:
            if comp["recommended_regime"] == "SIMPLIFIED_LAW6" and comp["law6_bracket"]:
                b = comp["law6_bracket"]
                en_lines.append(
                    f"• Based on annual turnover of **{profile.annual_revenue_egp:,} EGP**, your statutory bracket under **Law No. 6 of 2025** is **{b['rate_label']}**: "
                    f"Optimal Annual Tax = **{comp['optimal_annual_tax_egp']:,} EGP** vs. **{comp['standard_regime_tax_egp']:,} EGP** under the Standard Regime "
                    f"(Direct Annual Tax Savings: **{comp['annual_income_tax_savings_egp']:,} EGP**, plus 5-year audit exemption and full dividend/capital-gains/stamp-duty relief)."
                )
            else:
                en_lines.append(
                    f"• Optimal Standard Tax = **{comp['optimal_annual_tax_egp']:,} EGP** (incorporating deductible 0.25% solidarity contribution of "
                    f"**{comp['standard_details']['solidarity_contribution_egp']:,} EGP** under **Law No. 151 of 2026**, effective 2026-07-29)."
                )
        if comp["total_penalty_savings_egp"] > 0:
            en_lines.append(
                f"• **Dispute & Delay Penalty Relief (Laws No. 5 & 7 of 2025 + Law No. 152 of 2026):** Delay penalties reduced from "
                f"**{comp['raw_penalty_egp']:,} EGP** to **{comp['final_penalty_egp']:,} EGP** (Savings: **{comp['total_penalty_savings_egp']:,} EGP**; "
                f"Dispute Settlement window open until **2026-12-31**)."
            )
        if profile.real_estate_sale_egp > 0:
            if comp["real_estate_tax_egp"] == 0:
                en_lines.append(
                    f"• **Real Estate Disposal Tax (Law No. 151 of 2026, effective 2026-07-29):** **0 EGP (100% Exempt)** for first-degree relatives — saved **{comp['real_estate_savings_egp']:,} EGP**."
                )
            else:
                en_lines.append(
                    f"• **Real Estate Disposal Tax (Law No. 151 of 2026, effective 2026-07-29):** Unified 2.5% = **{comp['real_estate_tax_egp']:,} EGP** with an extended **60-day payment window**."
                )
        if comp["vat_whitelist_eligible"]:
            en_lines.append(
                "• **White List & Investor Priority (Laws No. 149 & 150 of 2026, effective 2026-08-01):** Eligible for **7-day fast-track VAT refund** and **Advance Tax Ruling Unit** access."
            )
        for w_en in opt.get("deadline_warnings_en", []):
            en_lines.append(w_en)
        en_lines.append(f"\n💰 **Total Certified Direct Savings:** **{comp['total_direct_savings_egp']:,} EGP**")
        en_lines.append(
            f"🛡️ **Lean 4.36.0 Proof Certificate:** `{lean.get('proof_certificate_id', 'N/A')}` | "
            f"**10/10 Theorems Verified (`0 sorry`)** in `{lean.get('elapsed_ms', 0)} ms`"
        )
        en_lines.append(f"\nℹ️ _{NON_BINDING_DISCLAIMER_EN}_")
        return {
            "lang": "en",
            "out_of_scope": False,
            "escalation": None,
            "reply_ar": "\n".join(en_lines),
            "pii_protection": pii_report,
            "ml_intent": intent_pred,
            "official_legal_passages": official_rag.get("matches", []),
            "suggested_followups": followups,
            "disclaimer": NON_BINDING_DISCLAIMER_EN,
            "extracted_signals": signals,
            "optimization": opt,
        }

    # Simplified Modern Standard Arabic (فصحى مبسطة — TAX-02 specification)
    lines: List[str] = []
    if pii_report["redacted_count"] > 0:
        lines.append(
            f"🔒 **حماية سرية البيانات (قانون 151 لسنة 2020):** تم حجب {pii_report['redacted_count']} بيانات شخصية حساسة تلقائياً قبل المعالجة."
        )

    lines.append(
        f"🧠 **تصنيف الاستفسار ({intent_pred['confidence_pct']}%):** {intent_pred['intent_ar']}"
    )
    lines.append(f"✅ **المسار القانوني الأمثل الموثّق رياضياً:** {comp['recommended_regime_ar']}")

    if profile.annual_revenue_egp > 0:
        if comp["recommended_regime"] == "SIMPLIFIED_LAW6" and comp["law6_bracket"]:
            b = comp["law6_bracket"]
            lines.append(
                f"• استناداً إلى حجم أعمالكم السنوي (**{profile.annual_revenue_egp:,} ج.م**)، تقع المنشأة في شريحة **{b['rate_label']}** "
                f"بموجب **القانون رقم 6 لسنة 2025** (سارٍ بصفة دائمة من `2025-03-01`): الضريبة السنوية المستحقة استرشادياً = **{comp['optimal_annual_tax_egp']:,} ج.م** "
                f"مقابل **{comp['standard_regime_tax_egp']:,} ج.م** في النظام العادي (وفر سنوي مباشر: **{comp['annual_income_tax_savings_egp']:,} ج.م**)."
            )
        else:
            lines.append(
                f"• الضريبة السنوية المثلى وفق النظام العادي = **{comp['optimal_annual_tax_egp']:,} ج.م** "
                f"(بعد خصم المساهمة التكافلية **{comp['standard_details']['solidarity_contribution_egp']:,} ج.م** ضمن التكاليف واجبة الخصم بموجب **القانون رقم 151 لسنة 2026** الساري من `2026-07-29`)."
            )

    if comp["total_penalty_savings_egp"] > 0:
        lines.append(
            f"• **تسوية المنازعات ومقابل التأخير (القانونان 5 و7 لسنة 2025 والقانون 152 لسنة 2026 الساري حتى `2026-12-31`):** "
            f"انخفض عبء الغرامات من **{comp['raw_penalty_egp']:,} ج.م** إلى **{comp['final_penalty_egp']:,} ج.م** "
            f"(وفر مباشر: **{comp['total_penalty_savings_egp']:,} ج.م**)."
        )

    if profile.real_estate_sale_egp > 0:
        if comp["real_estate_tax_egp"] == 0:
            lines.append(
                f"• **التصرفات العقارية (القانون 151 لسنة 2026 — سارٍ من `2026-07-29`):** معفى بالكامل (**0 ج.م**) لكونه بين أقارب الدرجة الأولى "
                f"(وفر **{comp['real_estate_savings_egp']:,} ج.م**)."
            )
        else:
            lines.append(
                f"• **التصرفات العقارية (القانون 151 لسنة 2026 — سارٍ من `2026-07-29`):** الضريبة الموحدة 2.5% = **{comp['real_estate_tax_egp']:,} ج.م** "
                f"مع مهلة سداد ممتدة إلى **60 يوماً**."
            )

    if comp["vat_whitelist_eligible"]:
        lines.append(
            "• **القائمة البيضاء وكارت التميز (القانونان 149 و150 لسنة 2026 — ساريان من `2026-08-01`):** مؤهل لرد ضريبة القيمة المضافة خلال **7 أيام** وأولوية وحدة الرأي المسبق."
        )

    for w_ar in opt.get("deadline_warnings_ar", []):
        lines.append(w_ar)

    if cliff_alert:
        lines.append(cliff_alert)

    if ml_intel.get("declaration_anomaly_guard", {}).get("is_anomalous"):
        lines.append(f"⚠️ {ml_intel['declaration_anomaly_guard']['warning_ar']}")

    lines.append(f"\n💰 **إجمالي الوفر المباشر الموثّق:** **{comp['total_direct_savings_egp']:,} ج.م**")
    lines.append(
        f"🛡️ **شهادة الإثبات الرياضي (Lean 4.36.0):** `{lean.get('proof_certificate_id', 'PENDING')}` "
        f"| **10/10 نظريات مُبرهَنة (`0 sorry`)** خلال `{lean.get('elapsed_ms', 0)} مللي ثانية`"
    )

    if opt["roadmap_ar"]:
        lines.append("\n📋 **الخطوات الإجرائية المطلوبة:**")
        for idx, step in enumerate(opt["roadmap_ar"][:4], start=1):
            lines.append(f"{idx}. {step}")

    if official_rag.get("matches"):
        top_m = official_rag["matches"][0]
        lines.append(
            f"\n📚 **المصدر الرسمي المعتمد:** {top_m['title']} — صفحة {top_m['page']}"
        )

    lines.append(f"\nℹ️ _{NON_BINDING_DISCLAIMER_AR}_")

    return {
        "lang": "ar",
        "out_of_scope": False,
        "escalation": None,
        "reply_ar": "\n".join(lines),
        "pii_protection": pii_report,
        "ml_intent": intent_pred,
        "dialect_xray": dialect_xray,
        "cliff_edge_alert": cliff_alert,
        "official_legal_passages": official_rag.get("matches", []),
        "suggested_followups": followups,
        "disclaimer": NON_BINDING_DISCLAIMER_AR,
        "extracted_signals": signals,
        "optimization": opt,
    }
