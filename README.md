# مُيسّر الضرائب الذكي — Moyasser AI v2.5 🇪🇬
### Sovereign Bilingual Neuro-Symbolic & Formally Verified Egyptian Tax Facilitation Assistant
**Official Submission for Egyptian Ministry of Finance & MCIT / ITIDA (`SECC`) Impact Hackathon — Challenge `TAX-02` (`مساعد ذكى للتيسيرات الضريبية`)**

[![CI — Lean 4 & 14-Test Suite](https://github.com/mohamedneqody/moyasser-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamedneqody/moyasser-ai/actions/workflows/ci.yml)

---

## Key Highlights
- **100% `TAX-02` Specification Coverage (External Taxpayer/Foreign-Investor Track + Internal ETA Specialist Track)**:
  - **Bilingual (`AR` Colloquial -> Simplified MSA + `EN` Foreign Investors)** across **Text, Live Voice (`Web Speech` + `/api/voice-query`), and Uploaded PDF/Scanned Notices (`pypdfium2`)**.
  - **Dialect Grounding X-Ray (`PyTorch BiGRU Slot Tagger`)**: Compiles spoken Egyptian phrases (`ميتين ألف في الشهر`, `مليون ونص`, `ومعليش غرامات`) into annualized EGP figures and visual X-Ray proof pills before calculation.
  - **100% On-Premise Sovereign Local LLM Mode (`Ollama: Qwen-Hybrid & Liquid LFM-Q4`)**: Optional zero-cloud advisory mode governed by a strict **Lean 4 Post-Generation Numeric Hallucination Firewall** and **Law No. 151/2020 Anonymous-by-Design PII Shield** (auto-redacting multi-turn history and uploaded documents).
  - **Statutory Effective & Expiry Date Guard** + **Mandatory Non-Binding Disclaimer** + **3 Contextual Follow-Up Suggestions** + **Out-of-Scope Polite Refusal & Escalation (`ETA-ESC-...`)**.
  - **Authorized Zero-Rebuild Hot-Reloadable Decree Ingestion** (`POST /api/knowledge/ingest-decree` protected by constant-time `X-ETA-Officer-Key` verification) + **Top Frequent Questions Dashboard**.
- **Zero Numeric Hallucination (`SciPy HiGHS MILP` + Native `Lean 4.36.0` Kernel)**:
  - **10 Machine-Checked Formal Theorems (`0 sorry`)** compiled to `moyasser_verifier.exe`, issuing `SHA-256` Proof Certificates (`LEAN4-PROOF-...`) in `< 40 ms`.
- **100% Real Official MoF & ETA Data + 7 Trained AI/ML Models**:
  - **18 Official PDFs (`437` verbatim clauses)** downloaded from `eta.gov.eg` & `media.mof.gov.eg` (Laws 5–7/2025, Laws 148–153/2026, Oct 2026 Real Estate FAQ, Oct 2026 MoF Monthly Financial Report) + 22-year fiscal panel (`2002–2027`).
- **CI/CD & SECC Security Verified**: Passing **GitHub Actions** (`Lean 4.36.0` compile + `14/14 pytest` suite), audited with `cyberguard-verify v3.3` (**`0` vulnerabilities**).

---

## One-Click Launch

```powershell
# Option A: Double-click or run start_moyasser.bat
.\start_moyasser.bat

# Option B: Manual launch
cd D:\CyberGuardAI\moyasser-ai
C:\Users\zbook\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn moyasser.app:app --host 127.0.0.1 --port 8080
```
- **Main Dual-Track Platform:** `http://127.0.0.1:8080`
- **Executive Pitch Deck (Printable to PDF):** `http://127.0.0.1:8080/pitch-deck`
