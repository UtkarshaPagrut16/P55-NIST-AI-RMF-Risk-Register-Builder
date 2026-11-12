# NIST AI RMF Risk Register Builder

Generative AI mini project (P55) by **Utkarsha Pagrut**, Roll 55, Enrollment ADT23SOCB1243, MIT Art, Design and Technology University.

A Streamlit web app that takes an AI project description and builds a structured risk register using the NIST AI Risk Management Framework.

| Task | Feature | Module |
|---|---|---|
| U1 | Lifecycle risk mapping (8 stages) | `modules/lifecycle.py` |
| U2 | LLM risk generation (Gemini or Groq) + labelled local fallback | `modules/llm_client.py`, `modules/fallback.py` |
| U3 | Sentence-BERT (`all-MiniLM-L6-v2`) retrieval over a NIST knowledge base | `modules/nist_retriever.py` |
| U4 | Score = Likelihood x Impact, 5x5 heatmap, statistics, CSV export | `modules/risk_scoring.py`, `charts.py`, `export.py` |
| U5 | GOVERN / MAP / MEASURE / MANAGE classification | `modules/rmf_classifier.py` |

## Quick start (Windows 11 PowerShell)

```powershell
cd application
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first time you open **NIST Guidance (U3)** the Sentence-BERT model (about 90 MB) downloads automatically; internet is needed once.

## API key (optional)

Without a key the app uses a **clearly labelled local fallback** ("Local template fallback (not LLM)").
For real LLM output set one of these (free tiers): `GEMINI_API_KEY` (Google AI Studio) or `GROQ_API_KEY`.

```powershell
$env:GEMINI_API_KEY = "your-key"     # this PowerShell window only
streamlit run app.py
```

Or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in. Never commit that file (it is in `.gitignore`).

## Samples

Use **Load sample** in the sidebar: Project A (healthcare diagnosis) and Project B (recruitment screening). Their registers are hand-authored and labelled "Curated sample", not LLM output.

## Tests

```powershell
python -m pytest -v
```

Three tests load the real Sentence-BERT model and are skipped automatically if it cannot be downloaded.

## Honesty notes

* Fallback and sample risks are always labelled as such; they are never shown as LLM output.
* If Sentence-BERT cannot load, the U3 page says so. Keyword matching is never substituted.
* The NIST knowledge base (`data/nist_ai_rmf_knowledge_base.json`) is a student paraphrase of NIST AI 100-1; check wording against the official documents.

## Tools for the report

```powershell
pip install playwright reportlab matplotlib pillow
playwright install chromium
python tools/collect_results.py ../documentation/results.json
python tools/capture_screenshots.py ../screenshots      # app must be running
python tools/build_report.py                            # writes ../55_Utkarsha_Pagrut_P55.pdf
```

Running these on a computer with internet access executes the real Sentence-BERT model, so the screenshots and report then contain genuine U3 results.
