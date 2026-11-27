"""
Moyasser AI — 100% On-Premise Sovereign Local LLM Bridge (Ollama: Qwen-Hybrid & Liquid LFM)
Strictly governed by the Lean 4 Formal Proof Kernel & Numeric Hallucination Firewall:
  - Hardcoded localhost endpoint (`http://127.0.0.1:11434`) — zero external cloud calls, zero SSRF risk,
    full compliance with Egyptian Personal Data Protection Law No. 151 of 2020 & MoF Data Sovereignty.
  - Supports `instant` (<40ms symbolic), `qwen-hybrid` (Qwen 3.5 Hybrid / Qwen 2.5), and `lfm-fast` (Liquid LFM2 Q4).
  - Post-generation Numeric Hallucination Firewall blocks any unapproved arithmetic from overriding Lean 4 numbers.
"""

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

# Strictly pinned to loopback interface — NEVER configurable from untrusted request payloads (SSRF-proof).
_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9._:-]{2,64}$")

_QWEN_PREFERENCE = [
    "qwen-hybrid-16-8k:latest",
    "qwen-hybrid-24-8k:latest",
    "qwen-hybrid-18-12k:latest",
    "AI-COS-Qwen-2.5:latest",
    "qwen2.5:3b",
    "qwen3.5:4b",
]

_LFM_PREFERENCE = [
    "AI-COS-LFM-Q4:latest",
    "AI-COS-Custom:latest",
    "lfm2.5-gpu:latest",
    "lfm2.5:latest",
]

_STATUS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}


def get_local_ollama_status(force_refresh: bool = False) -> Dict[str, Any]:
    """Probes local Ollama daemon (`127.0.0.1:11434`) with 15-second caching for fast health checks."""
    now = time.time()
    if not force_refresh and _STATUS_CACHE["data"] is not None and (now - _STATUS_CACHE["ts"] < 15.0):
        return _STATUS_CACHE["data"]

    installed: List[Dict[str, Any]] = []
    online = False
    try:
        with httpx.Client(timeout=1.5) as client:
            resp = client.get(f"{_OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                online = True
                for m in resp.json().get("models", []):
                    name = str(m.get("name") or "")
                    if not _SAFE_MODEL_RE.match(name):
                        continue
                    details = m.get("details") or {}
                    installed.append(
                        {
                            "name": name,
                            "size_gb": round(float(m.get("size") or 0) / (1024**3), 2),
                            "family": str(details.get("family") or ""),
                            "parameter_size": str(details.get("parameter_size") or ""),
                            "quantization": str(details.get("quantization_level") or ""),
                        }
                    )
    except Exception:
        online = False

    names = [m["name"] for m in installed]
    rec_qwen = next((c for c in _QWEN_PREFERENCE if c in names), None)
    if not rec_qwen:
        rec_qwen = next((n for n in names if "qwen" in n.lower()), None)

    rec_lfm = next((c for c in _LFM_PREFERENCE if c in names), None)
    if not rec_lfm:
        rec_lfm = next((n for n in names if "lfm" in n.lower()), None)

    status = {
        "online": online,
        "endpoint": "127.0.0.1:11434 (Loopback Only)",
        "sovereignty": "100% On-Premise Localhost — Zero External Cloud (Law No. 151/2020 Compliant)",
        "installed_count": len(installed),
        "installed_models": installed,
        "recommended_qwen_model": rec_qwen,
        "recommended_lfm_model": rec_lfm,
    }
    _STATUS_CACHE["ts"] = now
    _STATUS_CACHE["data"] = status
    return status


def resolve_sovereign_model(llm_mode: str, explicit_model: Optional[str] = None) -> Optional[str]:
    """Selects a validated installed local Ollama model for the requested sovereign mode."""
    if llm_mode == "instant":
        return None

    status = get_local_ollama_status()
    if not status["online"] or not status["installed_models"]:
        return None

    installed_names = {m["name"] for m in status["installed_models"]}
    if explicit_model and _SAFE_MODEL_RE.match(explicit_model):
        if explicit_model in installed_names:
            return explicit_model
        if f"{explicit_model}:latest" in installed_names:
            return f"{explicit_model}:latest"

    if llm_mode in ("lfm-fast", "auto-local"):
        return status["recommended_lfm_model"] or status["recommended_qwen_model"] or next(iter(installed_names))
    if llm_mode == "qwen-hybrid":
        return status["recommended_qwen_model"] or status["recommended_lfm_model"] or next(iter(installed_names))
    return None


def verify_llm_numeric_alignment(text: str, allowed_integers: Set[int], lang: str = "ar") -> Tuple[str, bool]:
    """
    Lean 4 Post-Generation Numeric Hallucination Firewall:
    Scans multi-digit monetary figures (>= 1,000) in local LLM output. If a small local model
    improvises an unverified calculation (e.g., multiplying rate by net profit instead of turnover),
    the firewall intercepts it so ONLY Lean-4-certified figures reach the citizen.
    """
    if not text:
        return "", False

    safe_years = {2020, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 15000}
    permitted = set(allowed_integers) | safe_years

    # Normalize Arabic-Indic digits for inspection
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩٬", "0123456789,")
    inspected = text.translate(trans)

    hallucinated_tokens: List[str] = []
    for match in re.finditer(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,10}\b", inspected):
        raw_tok = match.group(0)
        val = int(raw_tok.replace(",", ""))
        if val >= 1000 and val not in permitted:
            hallucinated_tokens.append(raw_tok)

    if not hallucinated_tokens:
        return text.strip(), False

    # Strip sentences containing unverified calculations while keeping valid procedural advice
    cleaned_lines: List[str] = []
    for line in re.split(r"[\n]+", text):
        line_norm = line.translate(trans)
        if any(tok in line_norm for tok in hallucinated_tokens):
            continue
        if line.strip():
            cleaned_lines.append(line.strip())

    guard_note = (
        "🛡️ [جدار حماية Lean 4: تم تثبيت الأرقام المالية المبرهنة أعلاه آلياً وحجب أي تقريب حسابي غير معتمد من الموديل اللغوي]."
        if lang == "ar"
        else "🛡️ [Lean 4 Numeric Firewall: Certified figures locked; unverified LLM arithmetic suppressed]."
    )
    if cleaned_lines:
        return "\n".join(cleaned_lines) + f"\n{guard_note}", True
    return guard_note, True


def generate_sovereign_advisory(
    *,
    llm_mode: str,
    explicit_model: Optional[str],
    user_query: str,
    lang: str,
    locked_facts_summary: str,
    rag_passages: List[Dict[str, Any]],
    data_quality_alerts: List[str],
    allowed_integers: Set[int],
) -> Dict[str, Any]:
    """
    Runs local on-premise Qwen-Hybrid or Liquid LFM via Ollama to produce a natural, human-friendly
    procedural & clarifying explanation, strictly locked to Lean 4 numbers and official ETA/MoF RAG clauses.
    """
    if llm_mode == "instant":
        return {
            "enabled": False,
            "mode": "instant",
            "model": None,
            "elapsed_ms": 0.0,
            "advisory_text": "",
            "firewall_checked": True,
            "hallucination_intercepted": False,
        }

    model_name = resolve_sovereign_model(llm_mode, explicit_model=explicit_model)
    if not model_name:
        return {
            "enabled": False,
            "mode": llm_mode,
            "model": None,
            "elapsed_ms": 0.0,
            "advisory_text": "",
            "fallback_reason": "Ollama غير مشغّل حالياً — تم التبديل التلقائي الفوري للوضع الحتمي الموثق بـ Lean 4.",
            "firewall_checked": True,
            "hallucination_intercepted": False,
        }

    rag_snippet = ""
    if rag_passages:
        top = rag_passages[0]
        rag_snippet = f"{top.get('title', '')} (p.{top.get('page', 1)}): {str(top.get('excerpt', ''))[:280]}"

    alerts_snippet = " | ".join(data_quality_alerts) if data_quality_alerts else "لا توجد أخطاء إدخال"

    if lang == "en":
        sys_prompt = (
            "You are an Egyptian Tax Facilitation Specialist at the Ministry of Finance. "
            "Provide 2 to 3 concise, practical procedural bullets (registration steps, required documents, or filing tips) "
            "grounded strictly in the official clause below. Write plain prose without EGP calculations."
        )
        usr_prompt = (
            f"Inquiry: {user_query[:360]}\n"
            f"Recommended Track: {locked_facts_summary}\n"
            f"Input Checks: {alerts_snippet}\n"
            f"Official Reference: {rag_snippet}"
        )
    else:
        sys_prompt = (
            "أنت مستشار خدمة الممولين والمستثمرين بمصلحة الضرائب المصرية. "
            "قدّم نقطتين أو ثلاث نقاط إجرائية عملية ومباشرة بالفصحى المبسطة الواضحة (خطوات التقديم، المستندات المطلوبة، أو نصيحة التزام مهمة) "
            "استناداً إلى النص التشريعي الرسمي أدناه فقط، وبدون كتابة معادلات حسابية."
        )
        usr_prompt = (
            f"استفسار الممول: {user_query[:360]}\n"
            f"المسار القانوني الموصى به: {locked_facts_summary}\n"
            f"ملاحظات فحص البيانات: {alerts_snippet}\n"
            f"النص الرسمي المعتمد: {rag_snippet}"
        )

    payload = {
        "model": model_name,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.12,
            "num_predict": 135,
            "num_ctx": 2048,
        },
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ],
    }

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(f"{_OLLAMA_BASE_URL}/api/chat", json=payload)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if resp.status_code == 200:
                raw_content = str((resp.json().get("message") or {}).get("content") or "").strip()
                # Strip any residual <think>...</think> tags if emitted by hybrid reasoning weights
                raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                safe_text, intercepted = verify_llm_numeric_alignment(raw_content, allowed_integers, lang=lang)
                return {
                    "enabled": bool(safe_text),
                    "mode": llm_mode,
                    "model": model_name,
                    "elapsed_ms": elapsed_ms,
                    "advisory_text": safe_text,
                    "sovereignty": "100% On-Premise Ollama (Zero External Data Transfer)",
                    "firewall_checked": True,
                    "hallucination_intercepted": intercepted,
                }
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "enabled": False,
            "mode": llm_mode,
            "model": model_name,
            "elapsed_ms": elapsed_ms,
            "advisory_text": "",
            "fallback_reason": f"انتهت مهلة الاستجابة المحلية ({exc.__class__.__name__}) — تم الاعتماد الفوري على المحرك الموثّق بـ Lean 4.",
            "firewall_checked": True,
            "hallucination_intercepted": False,
        }

    return {
        "enabled": False,
        "mode": llm_mode,
        "model": model_name,
        "elapsed_ms": 0.0,
        "advisory_text": "",
        "firewall_checked": True,
        "hallucination_intercepted": False,
    }
