"""
Official Egyptian Tax Facilitation Legal Catalog (2025–2026)
Grounded strictly in:
  - Law No. 5 of 2025 (Regularization of taxpayers & delay-penalty waiver)
  - Law No. 6 of 2025 (Simplified SME tax regime <= 20M EGP turnover)
  - Law No. 7 of 2025 (Unified Tax Procedures Law amendment: 100% penalty ceiling)
  - Second Tax Facilitation Package (White List fast VAT refund, Dispute Settlement
    extended to Dec 31, 2026, Real Estate Disposal 2.5% + 1st-degree exemption,
    subsidized financing for first 100,000 simplified-regime SMEs, Elite Service Centers).
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple

NON_BINDING_DISCLAIMER_AR = (
    "تنويه رسمي (وفق ضوابط مصلحة الضرائب المصرية وقانون حماية البيانات الشخصية رقم 151 لسنة 2020): "
    "هذه الإجابات والحسابات استرشادية فقط استناداً إلى التشريعات والأدلة الرسمية المنشورة، ولا تُعد رأياً ملزماً "
    "لمصلحة الضرائب المصرية أو فتوى ضريبية. لا يطلب النظام ولا يعالج الرقم القومي أو رقم التسجيل الضريبي."
)

NON_BINDING_DISCLAIMER_EN = (
    "Official Disclaimer (Egyptian Tax Authority & Personal Data Protection Law No. 151 of 2020): "
    "All answers and calculations are indicative guidance based strictly on published Egyptian tax legislation "
    "and do not constitute a binding tax ruling or legal opinion. Do not share National IDs or Tax Registration Numbers."
)

LAW6_CEILING_EGP = 20_000_000

# (upper_bound_exclusive, rate_bps, rate_label_ar, bracket_label_ar)
LAW6_BRACKETS: List[Tuple[int, int, str, str]] = [
    (500_000, 40, "0.4%", "أقل من 500 ألف جنيه"),
    (2_000_000, 50, "0.5%", "من 500 ألف إلى أقل من 2 مليون جنيه"),
    (3_000_000, 75, "0.75%", "من 2 مليون إلى أقل من 3 ملايين جنيه"),
    (10_000_000, 100, "1.0%", "من 3 ملايين إلى أقل من 10 ملايين جنيه"),
    (20_000_000, 150, "1.5%", "من 10 ملايين إلى أقل من 20 مليون جنيه"),
]

# Standard Egyptian Income Tax rates for comparison baseline
CORPORATE_STANDARD_RATE = 0.225  # 22.5% corporate income tax (Law 91/2005)
SOLIDARITY_CONTRIBUTION_RATE = 0.0025  # 0.25% healthcare solidarity contribution (deductible under Package #2!)

# Standard Personal/Sole-Proprietor progressive brackets (Law 91/2005 as amended)
PERSONAL_BRACKETS: List[Tuple[int, float]] = [
    (40_000, 0.0),
    (55_000, 0.10),
    (70_000, 0.15),
    (200_000, 0.20),
    (400_000, 0.225),
    (1_200_000, 0.25),
    (10_000_000_000, 0.275),
]

LEGAL_CITATIONS: Dict[str, Dict[str, str]] = {
    "LAW_6_2025_RATES": {
        "code": "LAW_6_2025",
        "law": "قانون رقم 6 لسنة 2025",
        "law_en": "Law No. 6 of 2025 (Simplified SME Tax Regime)",
        "effective_date": "2025-03-01",
        "expiry_date": "PERMANENT",
        "title": "النظام الضريبي المبسط والدائم للمشروعات حتى 20 مليون جنيه",
        "summary": "ضريبة نسبية مبسطة على حجم الأعمال السنوي (0.4% إلى 1.5%) بدلاً من الدفاتر المعقدة والفحص التقديري (مع سماح تجاوز لمرة واحدة حتى 20% / 24 مليون جنيه خلال 5 سنوات).",
        "official_url": "https://www.eta.gov.eg/sites/default/files/2025-02/law_no.6.of_.2025.pdf",
    },
    "LAW_6_2025_EXEMPTIONS": {
        "code": "LAW_6_2025_EXEMPTIONS",
        "law": "قانون رقم 6 لسنة 2025",
        "law_en": "Law No. 6 of 2025 — Statutory Exemptions",
        "effective_date": "2025-03-01",
        "expiry_date": "PERMANENT",
        "title": "إعفاءات الدمغة وتنمية الموارد والأرباح الرأسمالية وتوزيعات الأرباح + استقرار 5 سنوات",
        "summary": "إعفاء كامل من ضريبة الدمغة، رسم تنمية الموارد، توثيق عقود التأسيس والتمويل، ضريبة الأرباح الرأسمالية وتوزيعات الأرباح، عدم الخضوع للخصم والتحصيل، تأجيل الفحص 5 سنوات، وإقرار قيمة مضافة ربع سنوي.",
        "official_url": "https://www.eta.gov.eg/sites/default/files/2025-02/law_no.6.of_.2025.pdf",
    },
    "LAW_5_2025_SETTLEMENT": {
        "code": "LAW_5_2025",
        "law": "قانون رقم 5 لسنة 2025 (وقرار وزير المالية 167 لسنة 2025)",
        "law_en": "Law No. 5 of 2025 & Ministerial Decree No. 167 of 2025",
        "effective_date": "2025-02-13",
        "expiry_date": "2025-08-12",
        "title": "تسوية أوضاع الممولين وفتح ملفات جديدة والتجاوز عن مقابل التأخير",
        "summary": "عدم المحاسبة بأثر رجعي لغير المسجلين (اعتبار 13 فبراير 2025 تاريخ بدء النشاط بشرط التقديم في المهلة المحددة)، والتجاوز حتى 100% عن مقابل التأخير وفق ضوابط ومواعيد القانون وقرار المد 167 لسنة 2025.",
        "official_url": "https://www.eta.gov.eg/sites/default/files/2025-02/law_no.5.of_.2025.pdf",
    },
    "LAW_7_2025_PENALTY_CAP": {
        "code": "LAW_7_2025",
        "law": "قانون رقم 7 لسنة 2025",
        "law_en": "Law No. 7 of 2025 (Unified Tax Procedures Amendment)",
        "effective_date": "2025-02-13",
        "expiry_date": "PERMANENT",
        "title": "سقف مقابل التأخير (تعديل قانون الإجراءات الضريبية الموحد 206 لسنة 2020)",
        "summary": "وضع حد أقصى قانوني دائم لمقابل التأخير والضريبة الإضافية بحيث لا يتجاوز 100% من أصل الضريبة المستحقة.",
        "official_url": "https://www.eta.gov.eg/sites/default/files/2025-02/law_no.7.of_.2025.pdf",
    },
    "PKG2_WHITELIST": {
        "code": "LAW_149_150_2026",
        "law": "الحزمة الثانية — القانونان 149 و150 لسنة 2026",
        "law_en": "Second Package — Laws No. 149 & 150 of 2026 (VAT & Procedures)",
        "effective_date": "2026-08-01",
        "expiry_date": "PERMANENT",
        "title": "القائمة البيضاء وكارت التميز ورد القيمة المضافة خلال أسبوع + تخفيض ضريبة الأجهزة الطبية لـ 5%",
        "summary": "رد فوري لضريبة القيمة المضافة خلال 7 أيام للممولين الملتزمين في القائمة البيضاء، وتخفيض ضريبة الأجهزة الطبية من 14% إلى 5% مع إعفاء مستلزمات الغسيل الكلوي.",
        "official_url": "https://eta.gov.eg/sites/default/files/2026-08/law.no_.149.of_.2026.pdf",
    },
    "PKG2_DISPUTE_2026": {
        "code": "LAW_152_2026",
        "law": "القانون رقم 152 لسنة 2026",
        "law_en": "Law No. 152 of 2026 (Dispute Settlement Renewal)",
        "effective_date": "2026-08-01",
        "expiry_date": "2026-12-31",
        "title": "تجديد العمل بقانون 79 لسنة 2016 لإنهاء المنازعات الضريبية حتى 31 ديسمبر 2026",
        "summary": "إتاحة التسوية السريعة للمنازعات الضريبية القائمة قبل لجان الطعن والمحاكم حتى موعد أقصاه 31 ديسمبر 2026.",
        "official_url": "https://eta.gov.eg/sites/default/files/2026-08/law.no_.152.of_.2026.pdf",
    },
    "PKG2_REAL_ESTATE": {
        "code": "LAW_151_2026_RE",
        "law": "القانون رقم 151 لسنة 2026 (ودليل ETA أكتوبر 2026)",
        "law_en": "Law No. 151 of 2026 (Real Estate Disposal Tax Facilitation)",
        "effective_date": "2026-07-29",
        "expiry_date": "PERMANENT",
        "title": "توحيد ضريبة التصرفات العقارية عند 2.5% + إعفاء أقارب الدرجة الأولى + مهلة 60 يوماً",
        "summary": "تثبيت النسبة عند 2.5% للأفراد غير التجار مهما تكررت التصرفات مع مد مهلة السداد إلى 60 يوماً، وإعفاء كامل (0%) للبيع بين أقارب الدرجة الأولى.",
        "official_url": "https://eta.gov.eg/sites/default/files/2026-10/dlyl_alasylt_alshayt_-_mnzwmt_altsrfat_alqaryt_alasdar_alawl.pdf",
    },
    "PKG2_SME_FINANCING": {
        "code": "PKG2_SME_FINANCING",
        "law": "الحزمة الثانية للتيسيرات الضريبية 2026",
        "law_en": "Second Package 2026 — Subsidized SME Financing (First 100k Taxpayers)",
        "effective_date": "2026-08-01",
        "expiry_date": "CAP_100000_TAXPAYERS",
        "title": "تمويل ميسر ومنخفض التكلفة لأول 100 ألف ممول بالنظام المبسط",
        "summary": "تسهيلات تمويلية ميسرة بالتنسيق مع جهاز تنمية المشروعات للمنضمين للنظام المبسط بحد أقصى أول 100 ألف ممول.",
        "official_url": "https://www.eta.gov.eg/ar/content/dlyl-almstthmryn",
    },
    "PKG2_SOLIDARITY_DEDUCTIBLE": {
        "code": "LAW_151_2026_SOLIDARITY",
        "law": "القانون رقم 151 لسنة 2026 (الحزمة الثانية)",
        "law_en": "Law No. 151 of 2026 — Deductible Solidarity Contribution",
        "effective_date": "2026-07-29",
        "expiry_date": "PERMANENT",
        "title": "اعتبار المساهمة التكافلية من التكاليف واجبة الخصم",
        "summary": "خصم المساهمة التكافلية (0.25% من الإيرادات) من الوعاء الضريبي لتقليل العبء الفعلي على الشركات.",
        "official_url": "https://eta.gov.eg/sites/default/files/2026-08/law.no_.151.of_.2026.pdf",
    },
}


def evaluate_citation_deadlines(
    citations: List[Dict[str, str]],
    reference_date: Optional[datetime.date] = None,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Evaluates statutory effective & expiry dates against `reference_date` (defaults to today).
    Returns (enriched_citations, warnings_ar, warnings_en) to satisfy TAX-02 deadline rules!
    """
    ref = reference_date or datetime.date.today()
    enriched: List[Dict[str, Any]] = []
    warnings_ar: List[str] = []
    warnings_en: List[str] = []

    for cit in citations:
        c = dict(cit)
        exp = c.get("expiry_date", "PERMANENT")
        eff = c.get("effective_date", "2025-02-13")
        status = "ACTIVE_PERMANENT"
        days_remaining: Optional[int] = None

        if exp not in ("PERMANENT", "CAP_100000_TAXPAYERS"):
            try:
                exp_dt = datetime.date.fromisoformat(exp)
                days_remaining = (exp_dt - ref).days
                if days_remaining < 0:
                    status = "EXPIRED_WINDOW"
                    warnings_ar.append(
                        f"⏰ تنبيه مهلة قانونية منتهية ({c['law']}): المهلة الأصلية لتقديم طلبات التسوية/الإقرارات القديمة انتهت في {exp} "
                        f"(تاريخ السريان كان {eff}) — يُرجى الاستفادة بدلاً منها من القانون الدائم 6/2025 أو قانون إنهاء المنازعات 152/2026 الساري حتى 31 ديسمبر 2026."
                    )
                    warnings_en.append(
                        f"⏰ Expired Statutory Window ({c.get('law_en', c['code'])}): The initial filing window closed on {exp} "
                        f"(effective {eff}). Rely instead on permanent Law 6/2025 or Dispute Settlement Law 152/2026 (open until 2026-12-31)."
                    )
                elif days_remaining <= 60:
                    status = "EXPIRING_SOON"
                    warnings_ar.append(
                        f"⚠️ تنبيه مهلة قريبة الانتهاء ({c['law']}): يتبقى {days_remaining} يوماً فقط قبل انتهاء المهلة في {exp} (سارٍ منذ {eff})."
                    )
                    warnings_en.append(
                        f"⚠️ Approaching Statutory Deadline ({c.get('law_en', c['code'])}): Only {days_remaining} days left until {exp} (effective {eff})."
                    )
                else:
                    status = "ACTIVE_TIME_BOUND"
            except ValueError:
                status = "ACTIVE_PERMANENT"
        elif exp == "CAP_100000_TAXPAYERS":
            status = "ACTIVE_QUOTA_BOUND"

        c["deadline_status"] = status
        c["days_remaining"] = days_remaining
        enriched.append(c)

    return enriched, warnings_ar, warnings_en

ELITE_SERVICE_CENTERS: Dict[str, str] = {
    "القاهرة": "مركز الخدمات الضريبية المتميزة — القاهرة الجديدة (أو بوابة مصلحة الضرائب المصرية ETA / E-Tax)",
    "الجيزة": "مركز الخدمات الضريبية المتميزة — الشيخ زايد (أو بوابة ETA الإلكترونية)",
    "الإسكندرية": "مركز الخدمات الضريبية المتميزة — العلمين الجديدة / منطقة الإسكندرية الضريبية (أو بوابة ETA)",
    "أخرى": "بوابة مصلحة الضرائب المصرية (eta.gov.eg) + مراكز الخدمات المتميزة (القاهرة الجديدة / الشيخ زايد / العلمين)",
}


def get_law6_bracket(annual_revenue: int) -> Optional[Dict[str, Any]]:
    """Returns exact Law No. 6 of 2025 bracket metadata or None if >= 20,000,000 EGP."""
    rev = max(0, int(annual_revenue))
    if rev >= LAW6_CEILING_EGP:
        return None
    for upper, bps, rate_str, label in LAW6_BRACKETS:
        if rev < upper:
            return {
                "eligible": True,
                "upper_bound": upper,
                "rate_bps": bps,
                "rate_pct": bps / 100.0,
                "rate_label": rate_str,
                "bracket_label": label,
                "tax_egp": (rev * bps) // 10_000,
            }
    return None


def compute_standard_income_tax(
    annual_revenue: int,
    annual_expenses: int,
    entity_type: str = "company",
) -> Dict[str, Any]:
    """
    Computes baseline Standard Regime annual tax (Law 91/2005 + Package #2 solidarity deductibility).
    Uses integer EGP floor matching the Lean 4 formal kernel's integer arithmetic.
    """
    rev = max(0, int(annual_revenue))
    exp = max(0, int(annual_expenses))
    solidarity_fee = int(rev * SOLIDARITY_CONTRIBUTION_RATE)
    # Package #2: Solidarity contribution is now a deductible expense!
    taxable_profit = max(0, rev - exp - solidarity_fee)

    if entity_type in ("sole", "freelancer", "individual", "مهني", "فردي"):
        tax = 0.0
        prev = 0
        for limit, rate in PERSONAL_BRACKETS:
            if taxable_profit <= prev:
                break
            slice_amt = min(taxable_profit, limit) - prev
            tax += slice_amt * rate
            prev = limit
        income_tax = int(tax)
        method_label = "ضريبة دخل الأفراد والمهنيين التصاعدية (قانون 91 لسنة 2005)"
    else:
        income_tax = int(taxable_profit * CORPORATE_STANDARD_RATE)
        method_label = "ضريبة أرباح الأشخاص الاعتبارية 22.5% (مع خصم المساهمة التكافلية بالحزمة الثانية)"

    return {
        "taxable_profit_egp": taxable_profit,
        "solidarity_contribution_egp": solidarity_fee,
        "standard_tax_egp": income_tax,
        "method_label": method_label,
    }
