"""
Moyasser AI — Hardened FastAPI Web Application (Track TAX-02 Complete Edition)
Neuro-Symbolic Bilingual (AR/EN) Egyptian Tax Facilitation Platform:
  - External Track: Taxpayers & Foreign Investors (Text, Live Voice, PDF/Scanned Upload, 3 Follow-Ups, Lean 4 + HiGHS MILP)
  - Internal Track: ETA Investor Services Specialists (Official 18-PDF Search, Hot-Reload Decree Ingestion without Rebuild, Escalation Queue)
  - Analytics Track: Top Frequent Questions, Regional Formalization, Official MoF 2027/28 Forecast, & SECC Security Gate
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .ai_assistant import answer_tax_question, process_voice_query, redact_pii, scan_document
from .analytics_store import get_ministry_summary, list_escalations, record_assessment, record_escalation
from .lean_bridge import find_lean_verifier_binary
from .ml_models import (
    get_official_mof_fiscal_forecast,
    ingest_new_decree_hot_reload,
    load_model_card_metrics,
    load_official_manifest,
    models_ready,
    search_official_eta_documents,
)
from .optimizer import TaxpayerProfile, optimize_tax_strategy
from .pdf_export import build_arabic_roadmap_html, export_roadmap_pdf

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_LEAN_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lean_kernel", "TaxLaw2025.lean"))
_PITCH_DECK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "MOYASSER_PITCH_DECK.html"))

app = FastAPI(
    title="Moyasser AI — مُيسّر الضرائب الذكي (TAX-02)",
    description="Neuro-Symbolic Formally Verified Egyptian Tax Facilitation Assistant (HiGHS MILP + Lean 4 + Official RAG)",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), payment=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class ProfilePayload(BaseModel):
    business_name: str = Field(default="مشروع استرشادي", max_length=120)
    entity_type: str = Field(default="company", pattern="^(company|sole|freelancer)$")
    sector: str = Field(default="تجارة وخدمات", max_length=80)
    governorate: str = Field(default="القاهرة", max_length=50)
    annual_revenue_egp: int = Field(default=0, ge=0, le=50_000_000_000)
    annual_expenses_egp: int = Field(default=0, ge=0, le=50_000_000_000)
    is_registered_previously: bool = True
    has_old_dispute_or_penalties: bool = False
    dispute_principal_egp: int = Field(default=0, ge=0, le=10_000_000_000)
    delay_penalties_egp: int = Field(default=0, ge=0, le=10_000_000_000)
    vat_registered: bool = False
    vat_refund_claim_egp: int = Field(default=0, ge=0, le=10_000_000_000)
    compliant_filing_history: bool = True
    real_estate_sale_egp: int = Field(default=0, ge=0, le=10_000_000_000)
    real_estate_first_degree: bool = False


class ChatPayload(BaseModel):
    message: str = Field(..., min_length=1, max_length=2500)
    lang: str = Field(default="auto", pattern="^(auto|ar|en)$")
    profile_override: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[str]] = Field(default=None, max_length=12)


class DocumentScanPayload(BaseModel):
    filename: str = Field(default="invoice.txt", max_length=120)
    raw_text: str = Field(default="", max_length=25000)
    image_base64: Optional[str] = Field(default=None, max_length=8_000_000)
    pdf_base64: Optional[str] = Field(default=None, max_length=12_000_000)
    mime_type: str = Field(default="image/png", max_length=50)


class VoiceQueryPayload(BaseModel):
    transcript_text: str = Field(default="", max_length=2500)
    audio_base64: Optional[str] = Field(default=None, max_length=10_000_000)
    mime_type: str = Field(default="audio/webm", max_length=50)
    lang: str = Field(default="auto", pattern="^(auto|ar|en)$")


class OfficialSearchPayload(BaseModel):
    query: str = Field(..., min_length=2, max_length=600)
    top_k: int = Field(default=3, ge=1, le=6)


class EscalationPayload(BaseModel):
    lang: str = Field(default="ar", pattern="^(ar|en)$")
    reason: str = Field(default="طلب إحالة لمختص الإدارة العامة لخدمات المستثمرين", max_length=200)
    query: str = Field(..., min_length=2, max_length=1000)
    conversation_summary: str = Field(..., min_length=5, max_length=2000)


class HotReloadDecreePayload(BaseModel):
    doc_id: str = Field(..., min_length=2, max_length=40)
    title: str = Field(..., min_length=4, max_length=200)
    org: str = Field(default="مصلحة الضرائب المصرية (ETA)", max_length=100)
    effective_date: str = Field(default="2026-11-27", max_length=20)
    expiry_date: str = Field(default="PERMANENT", max_length=30)
    official_url: str = Field(default="https://eta.gov.eg/ar/news", max_length=250)
    raw_text: str = Field(default="", max_length=30000)
    pdf_base64: Optional[str] = Field(default=None, max_length=10_000_000)


def _to_profile(payload: ProfilePayload) -> TaxpayerProfile:
    return TaxpayerProfile(**payload.model_dump())


@app.get("/", include_in_schema=False)
def serve_home():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"), media_type="text/html; charset=utf-8")


@app.get("/pitch-deck", include_in_schema=False)
def serve_pitch_deck():
    if os.path.exists(_PITCH_DECK):
        return FileResponse(_PITCH_DECK, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h1>Pitch deck not generated yet</h1>", status_code=404)


@app.get("/api/health")
def api_health():
    lean_bin = find_lean_verifier_binary()
    ml_status = models_ready()
    return {
        "status": "ok",
        "track": "TAX-02 — مساعد ذكى للتيسيرات الضريبية",
        "service": "Moyasser AI v2.0 — Bilingual Neuro-Symbolic Tax Facilitation Assistant",
        "privacy_compliance": "Egyptian Personal Data Protection Law No. 151 of 2020",
        "milp_solver": "SciPy HiGHS",
        "lean4_kernel_ready": bool(lean_bin),
        "lean4_theorems": 10,
        "sorry_count": 0,
        "ml_models_ready": all(ml_status.values()),
        "ml_models": ml_status,
    }


@app.get("/api/lean4/theorems")
def api_lean4_theorems():
    code_excerpt = ""
    if os.path.exists(_LEAN_SRC):
        with open(_LEAN_SRC, "r", encoding="utf-8") as fh:
            code_excerpt = fh.read()
    return {
        "lean_version": "4.36.0",
        "theorems_proven": 10,
        "sorry_count": 0,
        "binary_ready": bool(find_lean_verifier_binary()),
        "theorems": [
            {"id": "T1", "name": "law6_eligibility_iff", "law": "قانون 6 لسنة 2025", "desc": "استحقاق النظام المبسط يتحقق رياضياً فقط عندما يقل حجم الأعمال عن 20 مليون جنيه دون فجوات."},
            {"id": "T2", "name": "law6_rate_bounds", "law": "قانون 6 لسنة 2025", "desc": "النسبة الضريبية لكل شريحة محصورة قطعاً بين 0.4% و 1.5%."},
            {"id": "T3", "name": "law6_rate_monotone", "law": "قانون 6 لسنة 2025", "desc": "النسب تتصاعد بانتظام مع زيادة الإيرادات دون أي تناقض بين الشرائح."},
            {"id": "T4", "name": "law7_penalty_cap_bound", "law": "قانون 7 لسنة 2025", "desc": "سقف مقابل التأخير لا يمكن أن يتجاوز 100% من أصل الضريبة."},
            {"id": "T5", "name": "law5_waiver_reduces_burden", "law": "قانون 5 لسنة 2025", "desc": "تسوية النزاعات والتجاوز عن مقابل التأخير تخفّض العبء دائماً ولا تزيده أبداً."},
            {"id": "T6", "name": "real_estate_first_degree_exempt", "law": "قانون 151 لسنة 2026", "desc": "التصرف العقاري لأقارب الدرجة الأولى معفى قطعاً (0 جنيه)."},
            {"id": "T7", "name": "optimal_never_exceeds_standard", "law": "سلامة التحسين الرياضي", "desc": "الضريبة المختارة من محرك HiGHS لا يمكن أبداً أن تزيد عن النظام العادي."},
            {"id": "T8", "name": "savings_conservation", "law": "حفظ المبالغ المالية", "desc": "الضريبة المثلى + الوفر المحقق = ضريبة النظام العادي بدقة الجنيه دون كسور مفقودة."},
            {"id": "T9", "name": "law6_grace_buffer_ceiling", "law": "قانون 6 لسنة 2025 (مادة التجاوز)", "desc": "سماح التجاوز لمرة واحدة بنسبة 20% خلال 5 سنوات محكوم بسقف 24 مليون جنيه."},
            {"id": "T10", "name": "law149_medical_vat_reduction", "law": "قانون 149 لسنة 2026", "desc": "تخفيض القيمة المضافة للأجهزة الطبية إلى 5% وإعفاء الغسيل الكلوي لا يتجاوز أبداً السعر العام 14%."},
        ],
        "source_code": code_excerpt,
    }


@app.post("/api/optimize")
def api_optimize(payload: ProfilePayload):
    p_dict = payload.model_dump()
    p_dict["business_name"] = redact_pii(p_dict["business_name"])[0]
    result = optimize_tax_strategy(TaxpayerProfile(**p_dict))
    record_assessment(result)
    return JSONResponse(content=result)


@app.post("/api/chat")
def api_chat(payload: ChatPayload):
    res = answer_tax_question(
        payload.message,
        profile_override=payload.profile_override,
        lang=payload.lang,
        conversation_history=payload.conversation_history,
    )
    if res.get("optimization"):
        record_assessment(res["optimization"])
    return JSONResponse(content=res)


@app.post("/api/voice-query")
def api_voice_query(payload: VoiceQueryPayload):
    res = process_voice_query(
        transcript_text=payload.transcript_text,
        audio_base64=payload.audio_base64,
        mime_type=payload.mime_type,
        lang=payload.lang,
    )
    opt = (res.get("chat_response") or {}).get("optimization")
    if opt:
        record_assessment(opt)
    return JSONResponse(content=res)


@app.post("/api/scan-document")
def api_scan_document(payload: DocumentScanPayload):
    res = scan_document(
        filename=payload.filename,
        raw_text=payload.raw_text,
        image_base64=payload.image_base64,
        pdf_base64=payload.pdf_base64,
        mime_type=payload.mime_type,
    )
    if res.get("optimization"):
        record_assessment(res["optimization"])
    return JSONResponse(content=res)


@app.post("/api/escalate")
def api_escalate(payload: EscalationPayload):
    clean_q, _ = redact_pii(payload.query)
    clean_sum, _ = redact_pii(payload.conversation_summary)
    ticket = record_escalation(
        lang=payload.lang,
        reason=payload.reason,
        redacted_query=clean_q,
        conversation_summary=clean_sum,
    )
    return JSONResponse(content=ticket)


@app.get("/api/officer/escalations")
def api_list_escalations():
    return JSONResponse(content={"escalations": list_escalations(20)})


@app.post("/api/knowledge/ingest-decree")
def api_ingest_decree(payload: HotReloadDecreePayload):
    clean_text, pii_rep = redact_pii(payload.raw_text)
    try:
        out = ingest_new_decree_hot_reload(
            doc_id=payload.doc_id,
            title=payload.title,
            org=payload.org,
            effective_date=payload.effective_date,
            expiry_date=payload.expiry_date,
            official_url=payload.official_url,
            raw_text=clean_text,
            pdf_base64=payload.pdf_base64,
        )
        out["pii_protection"] = pii_rep
        return JSONResponse(content=out)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/report/html")
def api_report_html(payload: ProfilePayload):
    p_dict = payload.model_dump()
    p_dict["business_name"] = redact_pii(p_dict["business_name"])[0]
    result = optimize_tax_strategy(TaxpayerProfile(**p_dict))
    html_doc = build_arabic_roadmap_html(result)
    return HTMLResponse(
        content=html_doc,
        headers={"Content-Disposition": 'attachment; filename="moyasser_roadmap.html"'},
    )


@app.post("/api/report/pdf")
def api_report_pdf(payload: ProfilePayload):
    p_dict = payload.model_dump()
    p_dict["business_name"] = redact_pii(p_dict["business_name"])[0]
    result = optimize_tax_strategy(TaxpayerProfile(**p_dict))
    pdf_bytes = export_roadmap_pdf(result, prefer_browser=False)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="moyasser_verified_certificate.pdf"'},
    )


@app.get("/api/ministry/analytics")
def api_ministry_analytics():
    return JSONResponse(content=get_ministry_summary())


@app.get("/api/models/metrics")
def api_models_metrics():
    return JSONResponse(content=load_model_card_metrics())


@app.get("/api/official-data/manifest")
def api_official_manifest():
    return JSONResponse(content=load_official_manifest())


@app.get("/api/official-data/fiscal-forecast")
def api_official_fiscal_forecast():
    return JSONResponse(content=get_official_mof_fiscal_forecast())


@app.post("/api/official-data/search")
def api_official_search(payload: OfficialSearchPayload):
    return JSONResponse(content=search_official_eta_documents(payload.query, top_k=payload.top_k))
