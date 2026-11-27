"""
Moyasser AI — Anonymized Ministry of Finance Policy Impact, Frequent Questions (FAQ),
and Human Officer Escalation Store (SQLite).
Strictly complies with Egyptian Personal Data Protection Law No. 151 of 2020 & Tax Secrecy:
stores ZERO PII (never stores National IDs, Tax Reg IDs, phone numbers, or names).
"""

import datetime
import hashlib
import os
import sqlite3
from typing import Any, Dict, List

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DB = os.path.join(_PKG_ROOT, "data", "moyasser_analytics.db")

_SEED_ROWS = [
    ("القاهرة", "مطاعم وأغذية", "SIMPLIFIED_LAW6", 4_200_000, 42_000, 295_000, 1, 1, 1, 1),
    ("القاهرة", "تكنولوجيا وبرمجيات", "SIMPLIFIED_LAW6", 11_500_000, 172_500, 610_000, 0, 0, 1, 1),
    ("القاهرة", "تجارة تجزئة ومحال", "STANDARD", 8_000_000, 36_000, 44_000, 0, 0, 0, 1),
    ("القاهرة", "عمل حر وخدمات رقمية", "SIMPLIFIED_LAW6", 850_000, 4_250, 78_000, 1, 0, 0, 1),
    ("الجيزة", "مطاعم وأغذية", "SIMPLIFIED_LAW6", 2_600_000, 19_500, 185_000, 1, 1, 0, 1),
    ("الجيزة", "خدمات طبية وصحية", "SIMPLIFIED_LAW6", 1_750_000, 8_750, 112_000, 0, 1, 0, 1),
    ("الجيزة", "تكنولوجيا وبرمجيات", "SIMPLIFIED_LAW6", 6_800_000, 68_000, 420_000, 0, 0, 1, 1),
    ("الإسكندرية", "صناعة وورش", "SIMPLIFIED_LAW6", 9_400_000, 94_000, 515_000, 1, 1, 1, 1),
    ("الإسكندرية", "تجارة وخدمات", "SIMPLIFIED_LAW6", 1_200_000, 6_000, 94_000, 1, 0, 0, 1),
    ("الإسكندرية", "مطاعم وأغذية", "SIMPLIFIED_LAW6", 3_500_000, 35_000, 210_000, 0, 1, 0, 1),
    ("الدلتا", "صناعة وورش", "SIMPLIFIED_LAW6", 4_800_000, 48_000, 340_000, 1, 1, 0, 1),
    ("الدلتا", "تجارة تجزئة ومحال", "SIMPLIFIED_LAW6", 450_000, 1_800, 32_000, 1, 0, 0, 1),
    ("الصعيد", "تجارة وخدمات", "SIMPLIFIED_LAW6", 920_000, 4_600, 68_000, 1, 0, 0, 1),
    ("الصعيد", "صناعة وورش", "SIMPLIFIED_LAW6", 2_300_000, 17_250, 165_000, 1, 1, 0, 1),
    ("مطروح", "تجارة وخدمات", "SIMPLIFIED_LAW6", 1_900_000, 9_500, 134_000, 1, 0, 0, 1),
    ("القاهرة", "تجارة وخدمات", "STANDARD", 24_000_000, 1_111_500, 225_000, 0, 1, 1, 1),
]

_SEED_FAQS = [
    ("LAW6_SIMPLIFIED_SME", "شروط ونسب النظام الضريبي المبسط للمشروعات أقل من 20 مليون جنيه (قانون 6/2025)", "Simplified SME Tax Regime < 20M EGP eligibility & brackets (Law 6/2025)", 142),
    ("INFORMAL_REGULARIZATION", "فتح ملف ضريبي جديد لغير المسجلين والفريلانسرز دون محاسبة بأثر رجعي", "Voluntary registration for informal businesses & freelancers without retroactive audit", 98),
    ("LAW5_DISPUTE_WAIVER", "تسوية المنازعات الضريبية قبل 31 ديسمبر 2026 والتجاوز عن مقابل التأخير (قانون 152/2026 و7/2025)", "Dispute settlement before Dec 31, 2026 & delay penalty cap/waiver", 87),
    ("PKG2_REAL_ESTATE_TAX", "ضريبة التصرفات العقارية 2.5% ومهلة الـ 60 يوماً وإعفاء أقارب الدرجة الأولى", "Real estate disposal tax 2.5%, 60-day window & 1st-degree relative exemption", 64),
    ("PKG2_WHITELIST_VAT", "شروط القائمة البيضاء وكارت التميز ورد ضريبة القيمة المضافة خلال 7 أيام", "White List eligibility & 7-day fast-track VAT refund for investors", 51),
    ("THIN_MARGIN_STANDARD_COMPARE", "متى يكون النظام الضريبي العادي أوفر من النظام المبسط لأصحاب هامش الربح المنخفض؟", "When Standard Net-Profit Regime beats Simplified Turnover Regime + Solidarity deduction", 39),
]


def _db_path() -> str:
    path = os.environ.get("MOYASSER_DB_PATH", "").strip() or _DEFAULT_DB
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policy_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            governorate TEXT NOT NULL,
            sector TEXT NOT NULL,
            recommended_regime TEXT NOT NULL,
            annual_revenue_egp INTEGER NOT NULL,
            optimal_tax_egp INTEGER NOT NULL,
            total_savings_egp INTEGER NOT NULL,
            newly_formalized INTEGER NOT NULL,
            penalty_waived INTEGER NOT NULL,
            whitelist_eligible INTEGER NOT NULL,
            lean4_verified INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS frequent_questions (
            intent_code TEXT PRIMARY KEY,
            topic_ar TEXT NOT NULL,
            topic_en TEXT NOT NULL,
            ask_count INTEGER NOT NULL DEFAULT 0,
            last_asked_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS officer_escalations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            lang TEXT NOT NULL,
            reason TEXT NOT NULL,
            redacted_query TEXT NOT NULL,
            conversation_summary TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING_INVESTOR_SERVICES'
        )
        """
    )

    if conn.execute("SELECT COUNT(*) FROM policy_telemetry").fetchone()[0] == 0:
        conn.executemany(
            """
            INSERT INTO policy_telemetry (
                created_at, governorate, sector, recommended_regime,
                annual_revenue_egp, optimal_tax_egp, total_savings_egp,
                newly_formalized, penalty_waived, whitelist_eligible, lean4_verified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(now, *row) for row in _SEED_ROWS],
        )

    if conn.execute("SELECT COUNT(*) FROM frequent_questions").fetchone()[0] == 0:
        conn.executemany(
            """
            INSERT INTO frequent_questions (intent_code, topic_ar, topic_en, ask_count, last_asked_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(code, ar, en, cnt, now) for code, ar, en, cnt in _SEED_FAQS],
        )

    conn.commit()
    return conn


def record_assessment(result: Dict[str, Any]) -> None:
    profile = result.get("profile", {})
    comp = result.get("comparison", {})
    lean = result.get("lean4_verification", {})

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO policy_telemetry (
                created_at, governorate, sector, recommended_regime,
                annual_revenue_egp, optimal_tax_egp, total_savings_egp,
                newly_formalized, penalty_waived, whitelist_eligible, lean4_verified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                str(profile.get("governorate", "أخرى"))[:40],
                str(profile.get("sector", "تجارة وخدمات"))[:50],
                str(comp.get("recommended_regime", "STANDARD"))[:30],
                int(profile.get("annual_revenue_egp", 0)),
                int(comp.get("optimal_annual_tax_egp", 0)),
                int(comp.get("total_direct_savings_egp", 0)),
                0 if profile.get("is_registered_previously", True) else 1,
                1 if int(comp.get("total_penalty_savings_egp", 0)) > 0 else 0,
                1 if comp.get("vat_whitelist_eligible") else 0,
                1 if lean.get("lean4_verified") else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_question_topic(intent_code: str) -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    code = str(intent_code or "LAW6_SIMPLIFIED_SME")[:40]
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE frequent_questions SET ask_count = ask_count + 1, last_asked_at = ? WHERE intent_code = ?",
            (now, code),
        )
        if cur.rowcount == 0:
            conn.execute(
                "INSERT INTO frequent_questions (intent_code, topic_ar, topic_en, ask_count, last_asked_at) VALUES (?, ?, ?, 1, ?)",
                (code, f"استفسارات عامة ({code})", f"General inquiries ({code})", now),
            )
        conn.commit()
    finally:
        conn.close()


def record_escalation(
    lang: str,
    reason: str,
    redacted_query: str,
    conversation_summary: str,
) -> Dict[str, Any]:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    digest = hashlib.sha256(f"{now}:{redacted_query[:80]}".encode("utf-8")).hexdigest()[:10].upper()
    ticket_id = f"ETA-ESC-{digest}"
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO officer_escalations (ticket_id, created_at, lang, reason, redacted_query, conversation_summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                now,
                str(lang or "ar")[:5],
                str(reason)[:120],
                str(redacted_query)[:500],
                str(conversation_summary)[:1500],
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "ticket_id": ticket_id,
        "created_at": now,
        "department_ar": "الإدارة العامة لخدمات المستثمرين والمختصين بمصلحة الضرائب المصرية",
        "department_en": "ETA Investor Services & Specialist Support Desk",
        "reason": reason,
        "conversation_summary": conversation_summary,
    }


def list_escalations(limit: int = 15) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT ticket_id, created_at, lang, reason, redacted_query, conversation_summary, status FROM officer_escalations ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 50)),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_ministry_summary() -> Dict[str, Any]:
    conn = _connect()
    try:
        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS total_assessments,
                COALESCE(SUM(total_savings_egp), 0) AS total_citizen_savings_egp,
                COALESCE(SUM(optimal_tax_egp), 0) AS projected_voluntary_tax_egp,
                COALESCE(SUM(CASE WHEN recommended_regime = 'SIMPLIFIED_LAW6' THEN 1 ELSE 0 END), 0) AS law6_adopters,
                COALESCE(SUM(newly_formalized), 0) AS informal_economy_formalized,
                COALESCE(SUM(penalty_waived), 0) AS disputes_settled,
                COALESCE(SUM(whitelist_eligible), 0) AS whitelist_candidates,
                COALESCE(SUM(lean4_verified), 0) AS lean4_certified_count
            FROM policy_telemetry
            """
        ).fetchone()

        gov_rows = conn.execute(
            """
            SELECT governorate, COUNT(*) AS count, COALESCE(SUM(total_savings_egp), 0) AS savings_egp,
                   COALESCE(SUM(newly_formalized), 0) AS newly_formalized
            FROM policy_telemetry GROUP BY governorate ORDER BY count DESC LIMIT 8
            """
        ).fetchall()

        sector_rows = conn.execute(
            """
            SELECT sector, COUNT(*) AS count, COALESCE(SUM(total_savings_egp), 0) AS savings_egp
            FROM policy_telemetry GROUP BY sector ORDER BY count DESC LIMIT 8
            """
        ).fetchall()

        faq_rows = conn.execute(
            "SELECT intent_code, topic_ar, topic_en, ask_count, last_asked_at FROM frequent_questions ORDER BY ask_count DESC LIMIT 8"
        ).fetchall()

        esc_count = conn.execute("SELECT COUNT(*) FROM officer_escalations").fetchone()[0]

        total = int(agg["total_assessments"])
        certified = int(agg["lean4_certified_count"])
        return {
            "kpis": {
                "total_assessments": total,
                "total_citizen_savings_egp": int(agg["total_citizen_savings_egp"]),
                "projected_voluntary_tax_egp": int(agg["projected_voluntary_tax_egp"]),
                "law6_adoption_count": int(agg["law6_adopters"]),
                "law6_adoption_rate_pct": round((int(agg["law6_adopters"]) / total) * 100, 1) if total else 0.0,
                "informal_economy_formalized": int(agg["informal_economy_formalized"]),
                "disputes_settled": int(agg["disputes_settled"]),
                "whitelist_candidates": int(agg["whitelist_candidates"]),
                "lean4_verification_rate_pct": round((certified / total) * 100, 1) if total else 100.0,
                "escalated_tickets_count": int(esc_count),
            },
            "by_governorate": [dict(r) for r in gov_rows],
            "by_sector": [dict(r) for r in sector_rows],
            "top_frequent_questions": [dict(r) for r in faq_rows],
        }
    finally:
        conn.close()
