"""
Comprehensive End-to-End Test Suite for Moyasser AI v2.0 (Official Track TAX-02 Complete Edition):
  - Native Lean 4.36.0 Formal Proof Kernel (`moyasser_verifier.exe` — 10 theorems, 0 sorry)
  - SciPy HiGHS MILP Optimization Engine (Laws 5, 6, 7 of 2025 + Laws 148-153 of 2026)
  - Bilingual (Simplified MSA Arabic + Foreign Investor English) Copilot
  - 3 Contextual Follow-Up Questions + Statutory Effective/Expiry Warnings + Law 151/2020 Disclaimer
  - Out-of-Scope Polite Refusal + Human Officer Escalation Summary Ticket
  - Multi-Modal Inputs: Real PDF Upload (`pdf_base64` via pypdfium2) + Voice Query Endpoint
  - Dynamic Hot-Reload Decree Ingestion WITHOUT Rebuilding (`POST /api/knowledge/ingest-decree`)
  - Top Frequent Questions (FAQ) Analytics + Real MoF 2027/28 Fiscal Forecaster + OWASP Security Headers
"""

import base64
import datetime
import os
import tempfile

from fastapi.testclient import TestClient

_TMP_DB = os.path.join(tempfile.gettempdir(), "moyasser_tax02_test.db")
_TMP_DYN = os.path.join(tempfile.gettempdir(), "moyasser_tax02_dyn_decrees.jsonl")
for _p in (_TMP_DB, _TMP_DYN):
    if os.path.exists(_p):
        os.remove(_p)
os.environ["MOYASSER_DB_PATH"] = _TMP_DB
os.environ["MOYASSER_DYNAMIC_DECREES_PATH"] = _TMP_DYN

from moyasser.ai_assistant import answer_tax_question, redact_pii, scan_document
from moyasser.app import app
from moyasser.lean_bridge import find_lean_verifier_binary, verify_solution_with_lean4
from moyasser.optimizer import TaxpayerProfile, optimize_tax_strategy
from moyasser.pdf_export import build_arabic_roadmap_html, export_roadmap_pdf
from moyasser.tax_laws import LEGAL_CITATIONS, evaluate_citation_deadlines

client = TestClient(app)


def test_lean4_native_10_theorems_and_verification():
    assert find_lean_verifier_binary() is not None
    cert = verify_solution_with_lean4(
        annual_revenue=4_000_000,
        standard_tax=335_250,
        dispute_principal=100_000,
        raw_penalty=180_000,
        waiver_eligible=True,
        real_estate_value=2_000_000,
        re_first_degree=True,
        claimed_optimal_tax=40_000,
        claimed_final_penalty=0,
        claimed_real_estate_tax=0,
    )
    assert cert["lean4_verified"] is True
    assert cert["sorry_count"] == 0
    assert cert["theorems_proven"] == 10
    assert cert["proof_certificate_id"].startswith("LEAN4-PROOF-")


def test_lean4_rejects_1_egp_tampered_claim():
    cert = verify_solution_with_lean4(
        annual_revenue=4_000_000,
        standard_tax=335_250,
        dispute_principal=100_000,
        raw_penalty=180_000,
        waiver_eligible=True,
        real_estate_value=0,
        re_first_degree=False,
        claimed_optimal_tax=39_999,
        claimed_final_penalty=0,
        claimed_real_estate_tax=0,
    )
    assert cert["lean4_verified"] is False
    assert cert["exit_code"] == 2


def test_milp_simplified_law6_and_thin_margin_protection():
    sim = optimize_tax_strategy(
        TaxpayerProfile(
            annual_revenue_egp=4_000_000,
            annual_expenses_egp=2_500_000,
            has_old_dispute_or_penalties=True,
            dispute_principal_egp=100_000,
            delay_penalties_egp=180_000,
        )
    )
    assert sim["comparison"]["recommended_regime"] == "SIMPLIFIED_LAW6"
    assert sim["comparison"]["optimal_annual_tax_egp"] == 40_000
    assert sim["comparison"]["final_penalty_egp"] == 0
    assert "151 لسنة 2020" in sim["disclaimer_ar"]
    assert sim["lean4_verification"]["lean4_verified"] is True

    thin = optimize_tax_strategy(
        TaxpayerProfile(annual_revenue_egp=10_000_000, annual_expenses_egp=9_800_000)
    )
    assert thin["comparison"]["recommended_regime"] == "STANDARD"
    assert thin["comparison"]["optimal_annual_tax_egp"] < 150_000
    assert thin["lean4_verification"]["lean4_verified"] is True


def test_statutory_effective_dates_and_expired_window_warnings():
    cits = [LEGAL_CITATIONS["LAW_5_2025_SETTLEMENT"], LEGAL_CITATIONS["PKG2_DISPUTE_2026"]]
    enriched, warn_ar, warn_en = evaluate_citation_deadlines(cits, reference_date=datetime.date(2026, 11, 27))
    statuses = {c["code"]: c["deadline_status"] for c in enriched}
    assert statuses["LAW_5_2025"] == "EXPIRED_WINDOW"
    assert statuses["LAW_152_2026"] == "EXPIRING_SOON"
    assert len(warn_ar) == 2 and len(warn_en) == 2


def test_bilingual_english_foreign_investor_mode_and_3_followups():
    r = client.post(
        "/api/chat",
        json={
            "message": "My software subsidiary in Cairo has annual revenue of 12 million EGP, expenses of 7.5 million EGP, and VAT refund claim of 450 thousand EGP.",
            "lang": "auto",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["lang"] == "en"
    assert len(data["suggested_followups"]) == 3
    assert data["optimization"]["comparison"]["vat_whitelist_eligible"] is True
    assert "LEAN4-PROOF-" in data["reply_ar"]
    assert "Law No. 151 of 2020" in data["disclaimer"]


def test_out_of_scope_refusal_and_officer_escalation_summary():
    r = client.post(
        "/api/chat",
        json={
            "message": "ما البند والتعريفة الجمركية لاستيراد سيارة كهربائية للرقم القومي 29805150102345؟",
            "conversation_history": ["لدي استفسار عام"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["out_of_scope"] is True
    assert data["escalation"]["ticket_id"].startswith("ETA-ESC-")
    assert "29805150102345" not in data["escalation"]["conversation_summary"]

    q = client.get("/api/officer/escalations")
    assert q.status_code == 200
    assert any(t["ticket_id"] == data["escalation"]["ticket_id"] for t in q.json()["escalations"])


def test_real_pdf_upload_scanning_and_voice_query():
    # Generate a certified PDF first, then upload it back as base64 PDF to test pypdfium2 extraction!
    opt = optimize_tax_strategy(TaxpayerProfile(annual_revenue_egp=3_500_000, annual_expenses_egp=2_000_000))
    pdf_bytes = export_roadmap_pdf(opt)
    r_scan = client.post(
        "/api/scan-document",
        json={
            "filename": "uploaded_tax_certificate.pdf",
            "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            "mime_type": "application/pdf",
        },
    )
    assert r_scan.status_code == 200
    scan_data = r_scan.json()
    assert "pypdfium2" in scan_data["ocr_engine"]
    assert scan_data["optimization"]["lean4_verification"]["lean4_verified"] is True

    r_voice = client.post(
        "/api/voice-query",
        json={"transcript_text": "بعت شقة بـ 3 مليون لابني، عليا ضريبة تصرفات عقارية كام؟", "lang": "ar"},
    )
    assert r_voice.status_code == 200
    v_data = r_voice.json()
    assert v_data["chat_response"]["optimization"]["comparison"]["real_estate_tax_egp"] == 0


def test_hot_reload_decree_ingestion_without_rebuild():
    r_ing = client.post(
        "/api/knowledge/ingest-decree",
        json={
            "doc_id": "ETA_CIRCULAR_99_2026",
            "title": "كتاب دوري رقم 99 لسنة 2026 بشأن تيسيرات مصدري البرمجيات السحابية",
            "effective_date": "2026-11-27",
            "raw_text": "يُمنح مصدرو البرمجيات السحابية والحوسبة الفائقة مساراً فورياً لرد ضريبة القيمة المضافة خلال خمسة أيام عمل دون فحص ميداني مسبق.",
        },
    )
    assert r_ing.status_code == 200
    ing = r_ing.json()
    assert ing["hot_reloaded"] is True
    assert ing["rebuild_required"] is False

    # Immediately search for the newly added decree WITHOUT restarting app!
    r_search = client.post("/api/official-data/search", json={"query": "مصدرو البرمجيات السحابية خمسة أيام عمل", "top_k": 3})
    assert r_search.status_code == 200
    matches = r_search.json()["matches"]
    assert any("ETA_CIRCULAR_99_2026" in m["doc_id"] for m in matches)


def test_frequent_questions_dashboard_and_official_manifest():
    r_mof = client.get("/api/ministry/analytics")
    assert r_mof.status_code == 200
    mof = r_mof.json()
    assert len(mof["top_frequent_questions"]) >= 6

    r_man = client.get("/api/official-data/manifest")
    assert r_man.status_code == 200
    assert r_man.json()["official_pdfs_count"] >= 18

    r_fc = client.get("/api/official-data/fiscal-forecast")
    assert r_fc.status_code == 200
    assert r_fc.json()["fy_2027_2028_projected_tax_billion_egp"] > 3000.0


def test_trained_6_models_and_security_headers_and_pitch_deck():
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    h = r_health.json()
    assert h["lean4_theorems"] == 10
    assert h["ml_models_ready"] is True
    assert len(h["ml_models"]) == 7

    r_deck = client.get("/pitch-deck")
    assert r_deck.status_code == 200
    assert "TAX-02" in r_deck.text
    assert r_deck.headers.get("x-frame-options") == "DENY"
    assert "content-security-policy" in r_deck.headers

    r_html = client.post("/api/report/html", json={"business_name": "<img src=x onerror=alert(1)>", "annual_revenue_egp": 1_500_000})
    assert "<img src=x" not in r_html.text


def test_killer_egyptian_dialect_xray_spoken_monthly_reversed_and_negation():
    # 1. Spoken Egyptian numbers + Monthly (×12) auto-annualization + Negation scope!
    r1 = client.post(
        "/api/chat",
        json={"message": "ببيع في المطعم بحوالي ميتين ألف في الشهر ومصاريفي مليون ونص في السنة ومعليش أي غرامات"},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    prof1 = d1["optimization"]["profile"]
    assert prof1["annual_revenue_egp"] == 2_400_000
    assert prof1["annual_expenses_egp"] == 1_500_000
    assert prof1["delay_penalties_egp"] == 0
    assert len(d1["dialect_xray"]["negations_honored"]) >= 1
    assert d1["optimization"]["lean4_verification"]["lean4_verified"] is True

    # 2. Reversed Egyptian word order (numbers BEFORE labels!)
    r2 = client.post(
        "/api/chat",
        json={"message": "بعمل 4 مليون مبيعات وبدفع 2 مليون مصاريف"},
    )
    assert r2.status_code == 200
    prof2 = r2.json()["optimization"]["profile"]
    assert prof2["annual_revenue_egp"] == 4_000_000
    assert prof2["annual_expenses_egp"] == 2_000_000

    # 3. Cliff-edge radar near 20M EGP ceiling (Lean 4 Theorem 9 24M EGP buffer)
    r3 = client.post("/api/chat", json={"message": "شركتي مبيعاتها 19.5 مليون ومصاريفها 14 مليون"})
    assert r3.status_code == 200
    assert "24,000,000" in (r3.json().get("cliff_edge_alert") or "")


def test_contradictory_inputs_zero_id_and_lean4_numeric_firewall():
    from moyasser.sovereign_llm import verify_llm_numeric_alignment

    # 1. Anonymous-by-Design (no National ID / Tax ID) + Contradictory inputs (Expenses > Revenue & Penalty w/o Principal)
    r_bad = client.post(
        "/api/chat",
        json={
            "message": "مبيعاتي 800 ألف في السنة ومصاريفي 3 مليون وعليا غرامات 90 ألف من غير ما أكتب أصل الضريبة",
            "llm_mode": "instant",
        },
    )
    assert r_bad.status_code == 200
    d_bad = r_bad.json()
    assert d_bad["pii_protection"]["redacted_count"] == 0
    assert len(d_bad.get("data_quality_alerts") or []) >= 2
    assert d_bad["optimization"]["comparison"]["standard_regime_tax_egp"] == 0
    assert d_bad["optimization"]["lean4_verification"]["lean4_verified"] is True

    # 2. Lean 4 Post-Generation Numeric Hallucination Firewall intercepts wrong LLM math (e.g., 6,750 instead of 18,000)
    hallucinated = "الضريبة المستحقة هي 6,750 ج.م وفق حساب صافي الربح.\nيرجى التسجيل عبر البوابة الإلكترونية لمصلحة الضرائب."
    cleaned, intercepted = verify_llm_numeric_alignment(hallucinated, {2_400_000, 1_500_000, 18_000, 164_500}, lang="ar")
    assert intercepted is True
    assert "6,750" not in cleaned
    assert "البوابة الإلكترونية" in cleaned

    # 3. Sovereign Local LLM Status endpoint confirms strict loopback sovereignty
    r_llm = client.get("/api/local-llm/status")
    assert r_llm.status_code == 200
    st = r_llm.json()
    assert "127.0.0.1:11434" in st["endpoint"]
    assert "151/2020" in st["sovereignty"]


def test_qa_certification_regressions_all_fixed():
    # 1. Cafe Arabic query (`ومصاريفه 2 مليون،`) separates 3M revenue from 2M expenses & cites SME law (not real estate!)
    r_cafe = client.post(
        "/api/chat",
        json={"message": "عندي كافيه مبيعاته 3 مليون ومصاريفه 2 مليون، إيه أهم خطوات التسجيل؟", "llm_mode": "instant"},
    )
    assert r_cafe.status_code == 200
    d_cafe = r_cafe.json()
    assert d_cafe["optimization"]["profile"]["annual_revenue_egp"] == 3_000_000
    assert d_cafe["optimization"]["profile"]["annual_expenses_egp"] == 2_000_000
    assert "التصرفات العقارية" not in (d_cafe["official_legal_passages"][0]["title"] if d_cafe["official_legal_passages"] else "")

    # 2. Foreign Investor English query must NEVER trigger Real Estate Disposal Tax (`real_estate_sale_egp == 0`)
    r_en = client.post(
        "/api/chat",
        json={
            "message": "My software subsidiary in Cairo has annual revenue of 12 million EGP, expenses of 7.5 million EGP, and a VAT refund claim of 450 thousand EGP."
        },
    )
    assert r_en.status_code == 200
    d_en = r_en.json()
    assert d_en["optimization"]["profile"]["annual_revenue_egp"] == 12_000_000
    assert d_en["optimization"]["profile"]["annual_expenses_egp"] == 7_500_000
    assert d_en["optimization"]["profile"]["real_estate_sale_egp"] == 0
    assert "Real Estate Disposal Tax" not in d_en["reply_ar"]

    # 3. Sample tax notice scanner separates 2.4M revenue, 1.3M expenses, 60k principal, 95k penalties -> 18,000 EGP Law 6 tax!
    r_doc = client.post(
        "/api/scan-document",
        json={
            "filename": "notice.txt",
            "raw_text": "إخطار ضريبي للرقم القومي 29805150102345 — إجمالي المبيعات السنوية 2.4 مليون جنيه، المصروفات 1.3 مليون جنيه، أصل الضريبة 60 ألف، مقابل التأخير 95 ألف جنيه.",
        },
    )
    assert r_doc.status_code == 200
    d_doc = r_doc.json()
    assert d_doc["optimization"]["profile"]["annual_revenue_egp"] == 2_400_000
    assert d_doc["optimization"]["profile"]["annual_expenses_egp"] == 1_300_000
    assert d_doc["optimization"]["profile"]["dispute_principal_egp"] == 60_000
    assert d_doc["optimization"]["profile"]["delay_penalties_egp"] == 95_000
    assert d_doc["optimization"]["comparison"]["optimal_annual_tax_egp"] == 18_000

    # 4. Hot-reload decree title search ranks #1 & Analytics sanitizes fuzz governorates
    client.post(
        "/api/knowledge/ingest-decree",
        json={
            "doc_id": "ETA_CIRCULAR_30_2026",
            "title": "تعليمات تنفيذية رقم 30 لسنة 2026 بتيسير فحص مستثمري التكنولوجيا",
            "raw_text": "تُعفى الشركات التكنولوجية المصدّرة للبرمجيات المنضمة للقائمة البيضاء من تقديم مستندات الفحص الورقية.",
        },
    )
    r_title = client.post("/api/official-data/search", json={"query": "تعليمات تنفيذية رقم 30 لسنة 2026", "top_k": 3})
    assert r_title.status_code == 200
    assert any("ETA_CIRCULAR_30_2026" in m["doc_id"] for m in r_title.json()["matches"])

    client.post("/api/optimize", json={"governorate": "1' OR '1'='1", "annual_revenue_egp": 1_000_000})
    mof = client.get("/api/ministry/analytics").json()
    assert all("'" not in g["governorate"] for g in mof["by_governorate"])

    # 5. Model metrics includes all 7 models including ArabicSlotBiGRU
    r_met = client.get("/api/models/metrics")
    assert r_met.status_code == 200
    assert "arabic_slot_bigru_tagger" in r_met.json()


