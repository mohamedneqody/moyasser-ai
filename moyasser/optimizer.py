"""
Moyasser AI — Industrial MILP Tax Optimization Engine (SciPy HiGHS Solver)
Formulates Egyptian Tax Facilitation decisions (Laws 5, 6, 7 of 2025 + Package #2) as a
Mixed-Integer Linear Program (MILP), solves globally to proven optimality with HiGHS,
and cross-verifies every solution through the compiled Lean 4 Formal Proof Kernel.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .lean_bridge import verify_solution_with_lean4
from .ml_models import predict_formalization_and_anomaly
from .tax_laws import (
    ELITE_SERVICE_CENTERS,
    LAW6_CEILING_EGP,
    LEGAL_CITATIONS,
    NON_BINDING_DISCLAIMER_AR,
    NON_BINDING_DISCLAIMER_EN,
    compute_standard_income_tax,
    evaluate_citation_deadlines,
    get_law6_bracket,
)


@dataclass
class TaxpayerProfile:
    business_name: str = "مشروع تجريبي"
    entity_type: str = "company"          # "company" | "sole" | "freelancer"
    sector: str = "تجارة وخدمات"
    governorate: str = "القاهرة"
    annual_revenue_egp: int = 0
    annual_expenses_egp: int = 0
    is_registered_previously: bool = True
    has_old_dispute_or_penalties: bool = False
    dispute_principal_egp: int = 0
    delay_penalties_egp: int = 0
    vat_registered: bool = False
    vat_refund_claim_egp: int = 0
    compliant_filing_history: bool = True
    real_estate_sale_egp: int = 0
    real_estate_first_degree: bool = False


def optimize_tax_strategy(profile: TaxpayerProfile) -> Dict[str, Any]:
    """
    Jointly optimizes Egyptian tax facilitation decisions using SciPy's HiGHS MILP solver,
    then mathematically certifies the optimal solution via the Lean 4 formal kernel.

    Decision variables x = [x_sim, x_std, x_waiver, x_re_exempt, x_whitelist]:
      - x_sim       ∈ {0,1}: Adopt Law 6/2025 Simplified Turnover Regime (0 if rev >= 20M)
      - x_std       ∈ {0,1}: Stay in Standard Net-Profit Regime (x_sim + x_std = 1)
      - x_waiver    ∈ {0,1}: Elect Law 5/2025 Regularization / Dispute Settlement (100% penalty waiver)
      - x_re_exempt ∈ {0,1}: Claim Package #2 First-Degree Relative Real-Estate Exemption (0%)
      - x_whitelist ∈ {0,1}: Apply for Package #2 White List / Excellence Card (7-day VAT refund)
    """
    rev = max(0, int(profile.annual_revenue_egp))
    exp = max(0, int(profile.annual_expenses_egp))
    principal = max(0, int(profile.dispute_principal_egp))
    raw_pen = max(0, int(profile.delay_penalties_egp))
    re_val = max(0, int(profile.real_estate_sale_egp))
    vat_claim = max(0, int(profile.vat_refund_claim_egp))

    law6 = get_law6_bracket(rev)
    law6_eligible = law6 is not None
    std_info = compute_standard_income_tax(rev, exp, profile.entity_type)
    std_tax = std_info["standard_tax_egp"]
    sim_tax = law6["tax_egp"] if law6_eligible else std_tax

    # Law 7 of 2025 statutory cap: delay penalty <= 100% of original tax principal
    law7_capped_penalty = min(principal, raw_pen) if principal > 0 else raw_pen
    waiver_allowed = bool(profile.has_old_dispute_or_penalties and (raw_pen > 0 or principal > 0))

    # Package #2 Real Estate disposal (2.5% = 250 bps, or 0% for 1st-degree relatives)
    re_standard_tax = (re_val * 250) // 10_000
    re_exempt_allowed = bool(re_val > 0 and profile.real_estate_first_degree)

    # Package #2 White List fast 7-day VAT refund liquidity benefit (estimated financing value ~2% of unlocked cash)
    whitelist_allowed = bool(profile.compliant_filing_history and (profile.vat_registered or vat_claim > 0))
    whitelist_liquidity_bonus = int(vat_claim * 0.02) if (whitelist_allowed and vat_claim > 0) else (100 if whitelist_allowed else 0)

    # Objective: minimize total EGP cash outflow
    # Note: slight tie-breaker (-0.5 on x_sim) favors Law 6 when taxes are exactly tied
    # because Law 6 also grants stamp/capital-gains/5-year-audit exemptions.
    c = np.array(
        [
            float(sim_tax) - 0.5,                           # x_sim
            float(std_tax),                                 # x_std
            -float(law7_capped_penalty),                    # x_waiver saves capped penalty
            -float(re_standard_tax),                        # x_re_exempt saves 2.5% real-estate tax
            -float(whitelist_liquidity_bonus),              # x_whitelist unlocks 7-day VAT liquidity
        ],
        dtype=float,
    )

    ub = np.array(
        [
            1.0 if law6_eligible else 0.0,
            1.0,
            1.0 if waiver_allowed else 0.0,
            1.0 if re_exempt_allowed else 0.0,
            1.0 if whitelist_allowed else 0.0,
        ],
        dtype=float,
    )
    lb = np.zeros(5, dtype=float)
    integrality = np.ones(5, dtype=int)

    # Mutual exclusion constraint: x_sim + x_std == 1
    A_eq = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]], dtype=float)
    constraints = LinearConstraint(A_eq, lb=np.array([1.0]), ub=np.array([1.0]))

    res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints)
    if not res.success or res.x is None:
        raise RuntimeError(f"HiGHS MILP optimization failed: {res.message}")

    x_sim, x_std, x_waiver, x_re_exempt, x_whitelist = [int(round(v)) for v in res.x]

    optimal_regime = "SIMPLIFIED_LAW6" if x_sim == 1 else "STANDARD"
    optimal_income_tax = sim_tax if x_sim == 1 else std_tax
    income_tax_savings = max(0, std_tax - optimal_income_tax)

    final_penalty = 0 if x_waiver == 1 else law7_capped_penalty
    penalty_saved_law7 = max(0, raw_pen - law7_capped_penalty)
    penalty_saved_law5 = law7_capped_penalty if x_waiver == 1 else 0
    total_penalty_savings = max(0, raw_pen - final_penalty)

    final_re_tax = 0 if x_re_exempt == 1 else re_standard_tax
    re_tax_savings = re_standard_tax if x_re_exempt == 1 else 0

    total_direct_savings = income_tax_savings + total_penalty_savings + re_tax_savings

    # Formal Verification via Compiled Native Lean 4 Proof Kernel!
    lean_cert = verify_solution_with_lean4(
        annual_revenue=rev,
        standard_tax=std_tax,
        dispute_principal=principal,
        raw_penalty=raw_pen,
        waiver_eligible=bool(x_waiver == 1),
        real_estate_value=re_val,
        re_first_degree=bool(x_re_exempt == 1),
        claimed_optimal_tax=optimal_income_tax,
        claimed_final_penalty=final_penalty,
        claimed_real_estate_tax=final_re_tax,
    )

    # Assemble Official Citations & Actionable Roadmap
    citations: List[Dict[str, str]] = []
    benefits: List[str] = []
    roadmap: List[str] = []
    documents: List[str] = [
        "بطاقة الرقم القومي للمالك أو الممثل القانوني (سارية)",
        "عقد الإيجار أو التمليك لمقر النشاط (موثق أو ثابت التاريخ)",
    ]

    if not profile.is_registered_previously:
        citations.append(LEGAL_CITATIONS["LAW_5_2025_SETTLEMENT"])
        benefits.append("إعفاء كامل من المحاسبة عن الفترات السابقة قبل تاريخ التسجيل الجديد وفق قانون 5 لسنة 2025.")
        roadmap.append("1. فتح ملف ضريبي إلكتروني جديد على بوابة مصلحة الضرائب (eta.gov.eg) بدون أي غرامات بأثر رجعي.")
    else:
        documents.append("البطاقة الضريبية وشهادة التسجيل الضريبي (رقم التسجيل الموحد)")

    if optimal_regime == "SIMPLIFIED_LAW6":
        citations.append(LEGAL_CITATIONS["LAW_6_2025_RATES"])
        citations.append(LEGAL_CITATIONS["LAW_6_2025_EXEMPTIONS"])
        citations.append(LEGAL_CITATIONS["PKG2_SME_FINANCING"])
        benefits.extend([
            f"الانضمام للنظام الضريبي المبسط بنسبة {law6['rate_label']} فقط من حجم الأعمال ({law6['bracket_label']}) بدلاً من الضريبة العادية.",
            "إعفاء تام من ضريبة الدمغة، ورسم تنمية الموارد، ورسوم توثيق عقود التأسيس والائتمان، وضريبة الأرباح الرأسمالية وتوزيعات الأرباح.",
            "عدم الخضوع لنظام الخصم والتحصيل تحت حساب الضريبة + تأجيل الفحص الضريبي لمدة 5 سنوات.",
            "تقديم إقرار القيمة المضافة كل 3 أشهر (ربع سنوي) بدلاً من شهرياً + إقرار مرتبات سنوي مبسط.",
            "أولوية الحصول على تمويل ميسر منخفض التكلفة ضمن مبادرة أول 100 ألف ممول بالحزمة الثانية.",
        ])
        roadmap.append(f"تقديم طلب الانضمام للنظام الضريبي المبسط (قانون 6 لسنة 2025 — شريحة {law6['rate_label']}) عبر بوابة ETA.")
        roadmap.append("الانضمام لمنظومة الفاتورة الإلكترونية أو الإيصال الإلكتروني كشرط استمرار التمتع بالمزايا.")
        documents.append("ملخص الإيرادات السنوية أو الإقرار الضريبي لآخر فترة (لإثبات عدم تجاوز سقف 20 مليون جنيه)")
    else:
        citations.append(LEGAL_CITATIONS["PKG2_SOLIDARITY_DEDUCTIBLE"])
        if law6_eligible:
            benefits.append(
                f"محرك التحسين الرياضي أثبت أن البقاء في النظام العادي ({std_tax:,} ج.م) أوفر لك حالياً من النظام المبسط ({sim_tax:,} ج.م) نظراً لانخفاض هامش الربح الصافي بعد خصم المصروفات والمساهمة التكافلية!"
            )
            roadmap.append("الاستمرار في النظام العادي (دفاتر منتظمة) مع خصم المساهمة التكافلية من التكاليف واجبة الخصم وفق الحزمة الثانية.")
        else:
            benefits.append("نظراً لتجاوز حجم الأعمال 20 مليون جنيه، يُطبق النظام العادي مع الاستفادة من خصم المساهمة التكافلية ومزايا الحزمة الثانية.")
            roadmap.append("الاستمرار بالنظام العادي والاستفادة من مزايا الممولين الملتزمين في الحزمة الثانية.")
        documents.append("القوائم المالية / ميزان المراجعة ومستندات المصروفات المؤيدة")

    if raw_pen > 0 or principal > 0:
        citations.append(LEGAL_CITATIONS["LAW_7_2025_PENALTY_CAP"])
        if x_waiver == 1:
            citations.append(LEGAL_CITATIONS["LAW_5_2025_SETTLEMENT"])
            citations.append(LEGAL_CITATIONS["PKG2_DISPUTE_2026"])
            benefits.append(
                f"التجاوز الكامل (100%) عن مقابل التأخير بقيمة {total_penalty_savings:,} ج.م عبر تسوية قانون 5 لسنة 2025 وقانون إنهاء المنازعات الممتد حتى 31 ديسمبر 2026."
            )
            roadmap.append("تقديم طلب إنهاء النزاع / تسوية أصل الضريبة قبل 31 ديسمبر 2026 لإسقاط مقابل التأخير بالكامل.")
            documents.append("إخطار المطالبة أو ملف النزاع الضريبي القائم")
        elif penalty_saved_law7 > 0:
            benefits.append(f"تطبيق سقف قانون 7 لسنة 2025 خفّض مقابل التأخير تلقائياً بمقدار {penalty_saved_law7:,} ج.م بحيث لا يتجاوز 100% من أصل الضريبة.")

    if x_whitelist == 1:
        citations.append(LEGAL_CITATIONS["PKG2_WHITELIST"])
        benefits.append(
            "مؤهل للانضمام لـ «القائمة البيضاء وكارت التميز» — رد ضريبة القيمة المضافة خلال 7 أيام فقط لتعزيز السيولة النقدية."
        )
        roadmap.append("تفعيل طلب الانضمام للقائمة البيضاء وكارت التميز للاستفادة من رد القيمة المضافة الفوري.")

    if re_val > 0:
        citations.append(LEGAL_CITATIONS["PKG2_REAL_ESTATE"])
        if x_re_exempt == 1:
            benefits.append(f"إعفاء كامل (0%) من ضريبة التصرفات العقارية (وفر {re_tax_savings:,} ج.م) لأن التصرف بين أقارب من الدرجة الأولى وفق الحزمة الثانية.")
            roadmap.append("تقديم مستند إثبات صلة القرابة من الدرجة الأولى عبر التطبيق الرسمي لإصدار شهادة الإعفاء.")
        else:
            benefits.append(f"ضريبة التصرفات العقارية الموحدة 2.5% = {final_re_tax:,} ج.م مع مهلة سداد ممتدة إلى 60 يوماً بدلاً من 30 يوماً.")
            roadmap.append("سداد ضريبة التصرفات العقارية (2.5%) خلال 60 يوماً وإصدار المخالصة آلياً عبر التطبيق الإلكتروني.")

    # Deduplicate citations by code & evaluate effective/expiry dates (TAX-02 compliance)
    uniq_citations = list({c["code"]: c for c in citations}.values())
    enriched_citations, deadline_warnings_ar, deadline_warnings_en = evaluate_citation_deadlines(uniq_citations)
    service_center = ELITE_SERVICE_CENTERS.get(profile.governorate, ELITE_SERVICE_CENTERS["أخرى"])
    roadmap.append(f"للدعم الفني الفوري: التوجه إلى {service_center}.")

    # Trained ML Intelligence Layer (XGBoost Formalization Propensity + IsolationForest Anomaly Guard)
    ml_intel = predict_formalization_and_anomaly(
        entity_type=profile.entity_type,
        sector=profile.sector,
        governorate=profile.governorate,
        annual_revenue_egp=rev,
        annual_expenses_egp=exp,
        is_registered_previously=profile.is_registered_previously,
        has_old_dispute=bool(profile.has_old_dispute_or_penalties or raw_pen > 0),
        delay_penalties_egp=raw_pen,
        vat_registered=profile.vat_registered,
        law6_eligible=law6_eligible,
        total_direct_savings_egp=total_direct_savings,
    )

    return {
        "profile": asdict(profile),
        "ml_intelligence": ml_intel,
        "solver": {
            "engine": "SciPy HiGHS Mixed-Integer Linear Programming (MILP)",
            "status": "OPTIMAL",
            "message": str(res.message),
            "decision_vector": {
                "adopt_simplified_law6": bool(x_sim),
                "stay_standard_regime": bool(x_std),
                "elect_law5_penalty_waiver": bool(x_waiver),
                "claim_real_estate_exemption": bool(x_re_exempt),
                "apply_whitelist_fast_vat": bool(x_whitelist),
            },
        },
        "comparison": {
            "law6_eligible": law6_eligible,
            "law6_ceiling_egp": LAW6_CEILING_EGP,
            "law6_bracket": law6,
            "simplified_law6_tax_egp": sim_tax if law6_eligible else None,
            "standard_regime_tax_egp": std_tax,
            "standard_details": std_info,
            "recommended_regime": optimal_regime,
            "recommended_regime_ar": (
                "النظام الضريبي المبسط (قانون 6 لسنة 2025)"
                if optimal_regime == "SIMPLIFIED_LAW6"
                else "النظام الضريبي العادي (الأوفر حسب هامش الربح)"
            ),
            "optimal_annual_tax_egp": optimal_income_tax,
            "annual_income_tax_savings_egp": income_tax_savings,
            "raw_penalty_egp": raw_pen,
            "law7_capped_penalty_egp": law7_capped_penalty,
            "final_penalty_egp": final_penalty,
            "total_penalty_savings_egp": total_penalty_savings,
            "real_estate_tax_egp": final_re_tax,
            "real_estate_savings_egp": re_tax_savings,
            "total_direct_savings_egp": total_direct_savings,
            "vat_whitelist_eligible": bool(x_whitelist),
            "sme_financing_eligible": bool(optimal_regime == "SIMPLIFIED_LAW6"),
        },
        "lean4_verification": lean_cert,
        "benefits_ar": benefits,
        "roadmap_ar": roadmap,
        "required_documents_ar": documents,
        "legal_citations": enriched_citations,
        "deadline_warnings_ar": deadline_warnings_ar,
        "deadline_warnings_en": deadline_warnings_en,
        "disclaimer_ar": NON_BINDING_DISCLAIMER_AR,
        "disclaimer_en": NON_BINDING_DISCLAIMER_EN,
        "nearest_service_center": service_center,
    }
