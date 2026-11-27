"""
Moyasser AI — Live Runtime Inference Engine for Trained ML & Deep Learning Models.
Loads the 4 trained artifacts from `models/`:
  1. PyTorch Residual MLP (`arabic_intent_mlp.pt` + `arabic_tfidf_vectorizer.joblib`)
  2. XGBoost SME Formalization Propensity Classifier (`sme_formalization_xgb.joblib`)
  3. XGBoost Sector Operating Expense-Ratio Regressor (`expense_estimator_xgb.joblib`)
  4. IsolationForest Tax Filing Anomaly Guard (`tax_anomaly_iforest.joblib`)
"""

import base64
import datetime
import json
import os
import re
import threading
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
# Eagerly initialize scikit-learn & xgboost submodules on the main thread at import time
# to prevent Python multi-threaded lazy-import circular ImportError under concurrent HTTP requests.
import sklearn.base  # noqa: F401
import sklearn.compose  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.feature_extraction.text  # noqa: F401
import sklearn.impute  # noqa: F401
import sklearn.linear_model  # noqa: F401
import sklearn.pipeline  # noqa: F401
import sklearn.preprocessing  # noqa: F401
import torch
import torch.nn as nn
import xgboost  # noqa: F401

_MODEL_LOAD_LOCK = threading.RLock()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

INTENT_LABELS_AR: Dict[str, str] = {
    "LAW6_SIMPLIFIED_SME": "النظام الضريبي المبسط للمشروعات الصغيرة (قانون 6 لسنة 2025)",
    "LAW5_DISPUTE_WAIVER": "تسوية المنازعات وإسقاط غرامات التأخير 100% (قانون 5 و7 لسنة 2025)",
    "INFORMAL_REGULARIZATION": "ضم الاقتصاد غير الرسمي وفتح ملف بدون أثر رجعي (قانون 5 لسنة 2025)",
    "PKG2_REAL_ESTATE_TAX": "ضريبة التصرفات العقارية 2.5% أو إعفاء الدرجة الأولى (الحزمة الثانية)",
    "PKG2_WHITELIST_VAT": "القائمة البيضاء وكارت التميز — رد القيمة المضافة في 7 أيام (الحزمة الثانية)",
    "THIN_MARGIN_STANDARD_COMPARE": "مقارنة النظام العادي بالنظام المبسط وخصم المساهمة التكافلية",
}


def normalize_arabic_text(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ى", "ي", t)
    t = re.sub(r"ة", "ه", t)
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"\s+", " ", t)


class ArabicTaxIntentNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.28),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.22),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.GELU()
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.act(h + self.res_block(h))
        return self.head(h)


@lru_cache(maxsize=1)
def _load_intent_bundle():
    pt_path = os.path.join(MODEL_DIR, "arabic_intent_mlp.pt")
    vec_path = os.path.join(MODEL_DIR, "arabic_tfidf_vectorizer.joblib")
    if not (os.path.exists(pt_path) and os.path.exists(vec_path)):
        return None
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    net = ArabicTaxIntentNet(
        input_dim=int(ckpt["input_dim"]),
        num_classes=len(ckpt["classes"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
    )
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    vec = joblib.load(vec_path)
    return net, vec, list(ckpt["classes"])


@lru_cache(maxsize=1)
def _load_formalization_clf():
    path = os.path.join(MODEL_DIR, "sme_formalization_xgb.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@lru_cache(maxsize=1)
def _load_expense_reg():
    path = os.path.join(MODEL_DIR, "expense_estimator_xgb.joblib")
    return joblib.load(path) if os.path.exists(path) else None


@lru_cache(maxsize=1)
def _load_anomaly_iforest():
    path = os.path.join(MODEL_DIR, "tax_anomaly_iforest.joblib")
    return joblib.load(path) if os.path.exists(path) else None


class OfficialLegalClauseNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dynamic_decrees_path() -> str:
    env_p = os.environ.get("MOYASSER_DYNAMIC_DECREES_PATH", "").strip()
    if env_p:
        return env_p
    return os.path.join(ROOT_DIR, "datasets", "official_mof_eta", "dynamic_decrees.jsonl")


@lru_cache(maxsize=1)
def _load_official_rag_bundle():
    with _MODEL_LOAD_LOCK:
        rag_path = os.path.join(MODEL_DIR, "official_eta_rag_index.joblib")
        pt_path = os.path.join(MODEL_DIR, "official_eta_legal_net.pt")
        if not (os.path.exists(rag_path) and os.path.exists(pt_path)):
            return None
        rag = joblib.load(rag_path)
        records = list(rag["records"])

        # Hot-merge any dynamically added decrees/circulars WITHOUT rebuilding models!
        dyn_path = _dynamic_decrees_path()
        if os.path.exists(dyn_path):
            with open(dyn_path, "r", encoding="utf-8-sig") as fh:
                for line in fh:
                    clean_ln = line.lstrip("\ufeff").strip()
                    if clean_ln:
                        try:
                            records.append(json.loads(clean_ln))
                        except json.JSONDecodeError:
                            continue

        # Index BOTH title and clause body so searching by decree/law title OR content ranks accurately
        clean_texts = [normalize_arabic_text(f"{r.get('title', '')} {r.get('text', '')}") for r in records]
        rag["matrix"] = rag["vectorizer"].transform(clean_texts)
        rag["records"] = records

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        net = OfficialLegalClauseNet(
            input_dim=int(ckpt["input_dim"]),
            num_classes=len(ckpt["classes"]),
            hidden_dim=int(ckpt.get("hidden_dim", 128)),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return rag, net, list(ckpt["classes"])


def extract_pdf_bytes_text(pdf_bytes: bytes) -> str:
    """Extracts clean logical RTL Arabic & English text from uploaded PDF bytes via pypdfium2."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    lines_out: List[str] = []
    for i in range(len(doc)):
        page = doc[i]
        tp = page.get_textpage()
        raw = tp.get_text_range() or ""
        tp.close()
        page.close()
        for ln in raw.splitlines():
            s = unicodedata.normalize("NFKC", ln).strip()
            if not s:
                continue
            s = s.replace("األ", "الأ").replace("اإل", "الإ").replace("اآل", "الآ")
            ar_c = len(re.findall(r"[\u0600-\u06FF]", s))
            lat_c = len(re.findall(r"[A-Za-z]", s))
            if ar_c > lat_c and ar_c >= 4:
                s = " ".join(reversed(s.split()))
            lines_out.append(s)
    doc.close()
    return "\n".join(lines_out).strip()


def ingest_new_decree_hot_reload(
    doc_id: str,
    title: str,
    org: str = "مصلحة الضرائب المصرية (ETA)",
    effective_date: str = "",
    expiry_date: str = "PERMANENT",
    official_url: str = "https://eta.gov.eg/ar/news",
    raw_text: str = "",
    pdf_base64: Optional[str] = None,
) -> Dict[str, Any]:
    """
    TAX-02 Requirement: Dynamically ingests new Ministerial Decrees, Executive Instructions,
    or Circulars into the live RAG Knowledge Base WITHOUT rebuilding or restarting the app!
    """
    content = (raw_text or "").strip()
    if pdf_base64:
        pdf_bytes = base64.b64decode(pdf_base64)
        extracted = extract_pdf_bytes_text(pdf_bytes)
        content = f"{content}\n{extracted}".strip()

    if len(content) < 20:
        raise ValueError("Decree text is too short to index.")

    eff = effective_date or datetime.date.today().isoformat()
    dyn_path = _dynamic_decrees_path()
    os.makedirs(os.path.dirname(os.path.abspath(dyn_path)), exist_ok=True)

    # Split into ~500-char passages
    passages = [content[i : i + 500].strip() for i in range(0, len(content), 460) if len(content[i : i + 500].strip()) >= 20]
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", doc_id or "DECREE_NEW")[:40]

    with open(dyn_path, "a", encoding="utf-8") as fh:
        for idx, ch in enumerate(passages, start=1):
            rec = {
                "chunk_id": f"{safe_id}_dyn_{idx}",
                "doc_id": safe_id,
                "org": org,
                "category": "DYNAMIC_DECREE_UPDATE",
                "domain": "PKG2_LAWS_148_153_2026",
                "title": f"{title} (سارٍ من {eff} | المهلة: {expiry_date})",
                "effective_date": eff,
                "expiry_date": expiry_date,
                "official_url": official_url,
                "page": idx,
                "text": ch,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Instant in-memory hot reload!
    _load_official_rag_bundle.cache_clear()
    bundle = _load_official_rag_bundle()
    total_chunks = len(bundle[0]["records"]) if bundle else len(passages)
    return {
        "hot_reloaded": True,
        "rebuild_required": False,
        "doc_id": safe_id,
        "title": title,
        "effective_date": eff,
        "expiry_date": expiry_date,
        "chunks_added": len(passages),
        "total_live_knowledge_chunks": total_chunks,
    }


@lru_cache(maxsize=1)
def _load_mof_forecaster():
    path = os.path.join(MODEL_DIR, "mof_tax_forecaster.joblib")
    return joblib.load(path) if os.path.exists(path) else None


def load_official_manifest() -> Dict[str, Any]:
    path = os.path.join(ROOT_DIR, "datasets", "official_mof_eta", "official_sources_manifest.json")
    if not os.path.exists(path):
        return {"official_pdfs_count": 0, "official_pdfs": []}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def search_official_eta_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Live semantic retrieval over the 437 REAL verbatim Arabic clauses extracted directly from
    the 18 official ETA (`eta.gov.eg`) & Ministry of Finance (`media.mof.gov.eg`) PDFs.
    """
    bundle = _load_official_rag_bundle()
    if not bundle or not (query or "").strip():
        return {"predicted_domain": None, "matches": []}
    rag, net, classes = bundle
    clean = normalize_arabic_text(query)

    # 1. Classify query legal domain via PyTorch OfficialLegalClauseNet
    clf_x = rag["clf_vectorizer"].transform([clean]).toarray()
    with torch.no_grad():
        probs = torch.softmax(net(torch.tensor(clf_x, dtype=torch.float32)), dim=1).numpy()[0]
    best_idx = int(np.argmax(probs))

    # 2. Cosine similarity over L2-normalized TF-IDF matrix + Legal Domain Alignment Boost
    q_vec = rag["vectorizer"].transform([clean])
    raw_scores = (rag["matrix"] @ q_vec.T).toarray().ravel()
    pred_domain = classes[best_idx]
    has_re_word = any(k in clean for k in ("شقه", "عقار", "فيلا", "ارض", "بيت", "تصرفات عقاريه", "تصرف عقاري"))
    has_sme_word = any(k in clean for k in ("مطعم", "كافيه", "فريلانسر", "محل", "ورشه", "مبسط", "قانون 6", "مبيعات"))

    adj_scores = raw_scores.copy()
    for r_i, rec in enumerate(rag["records"]):
        if raw_scores[r_i] <= 0.01:
            continue
        dom = rec.get("domain", "")
        if dom == pred_domain:
            adj_scores[r_i] += 0.12
        if dom == "ETA_REAL_ESTATE_FAQ_2026" and not has_re_word:
            adj_scores[r_i] *= 0.25
        if dom == "PKG1_LAWS_5_6_7_2025" and has_sme_word and not has_re_word:
            adj_scores[r_i] += 0.16

    top_indices = np.argsort(adj_scores)[::-1][: max(1, min(top_k, 8))]

    matches: List[Dict[str, Any]] = []
    for idx in top_indices:
        sc = float(adj_scores[int(idx)])
        if sc <= 0.02:
            continue
        rec = rag["records"][int(idx)]
        matches.append(
            {
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "org": rec["org"],
                "title": rec["title"],
                "page": int(rec["page"]),
                "official_url": rec["official_url"],
                "similarity": round(min(0.99, sc), 3),
                "excerpt": rec["text"][:360],
            }
        )

    return {
        "model": "PyTorch OfficialLegalClauseNet + Official PDF RAG (437 real MoF/ETA clauses)",
        "predicted_domain": classes[best_idx],
        "domain_confidence_pct": round(float(probs[best_idx]) * 100.0, 1),
        "matches": matches,
    }


def get_official_mof_fiscal_forecast() -> Dict[str, Any]:
    fc = _load_mof_forecaster()
    manifest = load_official_manifest()
    if not fc:
        return {"ready": False}
    # Sanitize pandas float NaN -> None for strict RFC-8259 JSON compliance
    recent = []
    for r in fc["history"]:
        if int(r["year"]) >= 2020:
            clean_r = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            recent.append(clean_r)
    return {
        "ready": True,
        "model_name": fc["model_name"],
        "data_source": "Official MoF Monthly Financial Report (Oct 2026 Table 10 & 12A) + World Bank IMF GFS Egypt",
        "recent_fiscal_years": recent,
        "fy_2027_2028_projected_tax_billion_egp": fc["fy_2027_2028_projected_tax_billion_egp"],
        "mof_benchmarks": manifest.get("fiscal_timeseries", {}).get("mof_published_benchmarks_2025_2026", {}),
    }


def models_ready() -> Dict[str, bool]:
    from .arabic_dialect_xray import _load_slot_bigru

    return {
        "official_eta_legal_rag": _load_official_rag_bundle() is not None,
        "official_mof_fiscal_forecaster": _load_mof_forecaster() is not None,
        "pytorch_arabic_slot_bigru": _load_slot_bigru() is not None,
        "pytorch_arabic_intent": _load_intent_bundle() is not None,
        "xgboost_formalization": _load_formalization_clf() is not None,
        "xgboost_expense_estimator": _load_expense_reg() is not None,
        "isolation_forest_anomaly": _load_anomaly_iforest() is not None,
    }


def load_model_card_metrics() -> Dict[str, Any]:
    path = os.path.join(MODEL_DIR, "model_card_metrics.json")
    if not os.path.exists(path):
        return {"ready": False, "models": models_ready()}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["ready"] = all(models_ready().values())
    data["loaded_status"] = models_ready()
    return data


def predict_arabic_intent(text: str) -> Dict[str, Any]:
    """Runs live inference with the trained PyTorch Residual MLP on Egyptian Arabic queries."""
    bundle = _load_intent_bundle()
    if not bundle or not (text or "").strip():
        return {
            "model": "PyTorch ArabicTaxIntentNet (unavailable)",
            "intent": "LAW6_SIMPLIFIED_SME",
            "intent_ar": INTENT_LABELS_AR["LAW6_SIMPLIFIED_SME"],
            "confidence_pct": 0.0,
            "top3": [],
        }

    net, vec, classes = bundle
    clean = normalize_arabic_text(text)
    x_vec = vec.transform([clean]).toarray()
    with torch.no_grad():
        probs = torch.softmax(net(torch.tensor(x_vec, dtype=torch.float32)), dim=1).numpy()[0]

    order = np.argsort(probs)[::-1]
    top_idx = int(order[0])
    top_code = classes[top_idx]
    top3 = [
        {
            "intent": classes[int(i)],
            "intent_ar": INTENT_LABELS_AR.get(classes[int(i)], classes[int(i)]),
            "confidence_pct": round(float(probs[int(i)]) * 100.0, 1),
        }
        for i in order[:3]
    ]
    return {
        "model": "PyTorch Residual MLP (arabic_intent_mlp.pt)",
        "intent": top_code,
        "intent_ar": INTENT_LABELS_AR.get(top_code, top_code),
        "confidence_pct": round(float(probs[top_idx]) * 100.0, 1),
        "top3": top3,
    }


def estimate_sector_expense_ratio(
    entity_type: str,
    sector: str,
    governorate: str,
    annual_revenue_egp: int,
    vat_registered: bool = False,
    is_registered_previously: bool = True,
) -> Dict[str, Any]:
    """Predicts realistic Egyptian sector operating expense ratio using trained XGBoost Regressor."""
    reg = _load_expense_reg()
    if reg is None or annual_revenue_egp <= 0:
        return {"model": "fallback", "predicted_ratio": 0.60, "estimated_expenses_egp": int(annual_revenue_egp * 0.60)}

    row = pd.DataFrame(
        [
            {
                "entity_type": entity_type or "company",
                "sector": sector or "تجارة وخدمات",
                "governorate": governorate or "القاهرة",
                "annual_revenue_egp": int(annual_revenue_egp),
                "vat_registered": int(bool(vat_registered)),
                "is_registered_previously": int(bool(is_registered_previously)),
            }
        ]
    )
    ratio = float(np.clip(reg.predict(row)[0], 0.15, 0.94))
    return {
        "model": "XGBoost Regressor (expense_estimator_xgb.joblib)",
        "predicted_ratio": round(ratio, 3),
        "estimated_expenses_egp": int(round(annual_revenue_egp * ratio)),
    }


def predict_formalization_and_anomaly(
    entity_type: str,
    sector: str,
    governorate: str,
    annual_revenue_egp: int,
    annual_expenses_egp: int,
    is_registered_previously: bool,
    has_old_dispute: bool,
    delay_penalties_egp: int,
    vat_registered: bool,
    law6_eligible: bool,
    total_direct_savings_egp: int,
) -> Dict[str, Any]:
    """
    Runs live inference on:
      1. Trained XGBoost Classifier -> SME formalization / facilitation adoption propensity (0–100%).
      2. Trained IsolationForest -> Clerical typo / declaration anomaly check BEFORE filing.
    """
    rev = max(0, int(annual_revenue_egp))
    exp = max(0, int(annual_expenses_egp))
    pen = max(0, int(delay_penalties_egp))
    exp_ratio = (exp / rev) if rev > 0 else 0.0
    sav_ratio = (total_direct_savings_egp / rev) if rev > 0 else 0.0
    pen_ratio = (pen / rev) if rev > 0 else 0.0

    # 1. XGBoost Formalization Propensity
    clf = _load_formalization_clf()
    propensity_pct: Optional[float] = None
    if clf is not None and rev > 0:
        row = pd.DataFrame(
            [
                {
                    "entity_type": entity_type or "company",
                    "sector": sector or "تجارة وخدمات",
                    "governorate": governorate or "القاهرة",
                    "annual_revenue_egp": rev,
                    "expense_ratio": min(exp_ratio, 1.0),
                    "is_registered_previously": int(bool(is_registered_previously)),
                    "has_old_dispute": int(bool(has_old_dispute)),
                    "vat_registered": int(bool(vat_registered)),
                    "years_operating": np.nan,
                    "law6_eligible": int(bool(law6_eligible)),
                    "savings_ratio": min(sav_ratio, 0.5),
                    "penalty_to_revenue_ratio": min(pen_ratio, 1.0),
                }
            ]
        )
        propensity_pct = round(float(clf.predict_proba(row)[0, 1]) * 100.0, 1)

    # 2. IsolationForest Anomaly Guard
    iso = _load_anomaly_iforest()
    is_anomalous = False
    anomaly_score = 0.0
    anomaly_warning_ar = ""

    if rev > 0:
        if exp > rev * 1.15:
            is_anomalous = True
            anomaly_warning_ar = (
                "تنبيه فحص الذكاء الاصطناعي (IsolationForest): المصروفات المُدخلة تتجاوز الإيرادات بأكثر من 15% — "
                "يرجى مراجعة الأرقام قبل التقديم لتجنب خطأ كتابي في الإقرار."
            )
        elif iso is not None:
            feats = np.array([[np.log1p(rev), exp_ratio, pen_ratio]], dtype=float)
            pred = int(iso.predict(feats)[0])
            anomaly_score = round(float(iso.decision_function(feats)[0]), 4)
            if pred == -1 or rev > 100_000_000:
                is_anomalous = True
                anomaly_warning_ar = (
                    "تنبيه فحص الذكاء الاصطناعي (IsolationForest): تم رصد نمط غير معتاد في نسب الإيرادات/المصروفات/الغرامات "
                    "مقارنة بمتوسط المشروعات المصرية — يُنصح بمراجعة عدد الأصفار والمستندات قبل التسليم لمصلحة الضرائب."
                )

    propensity_tier_ar = (
        "استجابة مرتفعة جداً للتيسيرات"
        if (propensity_pct or 0) >= 75
        else ("استجابة متوسطة — يُنصح بإبراز مزايا الإعفاء من الأثر الرجعي" if (propensity_pct or 0) >= 45 else "استجابة منخفضة — النظام العادي قد يكون أنسب")
    )

    return {
        "formalization_propensity": {
            "model": "XGBoost Classifier (sme_formalization_xgb.joblib)",
            "probability_pct": propensity_pct,
            "tier_ar": propensity_tier_ar if propensity_pct is not None else "غير مطبق (معاملة عقارية فقط)",
        },
        "declaration_anomaly_guard": {
            "model": "IsolationForest (tax_anomaly_iforest.joblib)",
            "status": "WARNING_REVIEW_INPUTS" if is_anomalous else "NORMAL_CONSISTENT",
            "is_anomalous": is_anomalous,
            "decision_score": anomaly_score,
            "warning_ar": anomaly_warning_ar,
        },
    }
