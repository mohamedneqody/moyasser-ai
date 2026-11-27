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
    assert len(h["ml_models"]) == 6

    r_deck = client.get("/pitch-deck")
    assert r_deck.status_code == 200
    assert "TAX-02" in r_deck.text
    assert r_deck.headers.get("x-frame-options") == "DENY"
    assert "content-security-policy" in r_deck.headers

    r_html = client.post("/api/report/html", json={"business_name": "<img src=x onerror=alert(1)>", "annual_revenue_egp": 1_500_000})
    assert "<img src=x" not in r_html.text
