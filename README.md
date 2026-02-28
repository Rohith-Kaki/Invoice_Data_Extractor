# Invoice Data Extractor (Streamlit)

A production-ready Streamlit app to extract highly precise structured invoice data (including table rows/columns, invoice numbers, FP/tax numbers, totals, entities, and free-form fields), with persistent on-disk storage and model switching.

Supported extraction models:
- **Mistral OCR 3**
- **GPT-5**
- **GLM OCR**

The UI stores data in the requested pattern:

```text
company1
 ├─ invoice1
 │   └─ data.json
 └─ invoice2
     └─ data.json
company2
 └─ invoice1
     └─ data.json
```

---

## 1) Features

- Upload invoice files (`.pdf`, `.png`, `.jpg`, `.jpeg`).
- Select one model at runtime from the sidebar.
- Auto convert PDF pages to images before OCR extraction.
- Force normalized JSON schema with:
  - `company_name`
  - header key-values
  - invoice metadata
  - `tables` with explicit `columns` and `rows`
  - `line_items`
  - `raw_text`
- Persist per company and per invoice in `data/<company>/<invoice>/data.json`.
- In-app explorer to review saved data.

---

## 2) Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the app URL shown by Streamlit (usually `http://localhost:8501`).

---

## 3) Environment setup (model by model)

The app reads API keys from environment variables.

### A) GPT-5 (OpenAI)

1. Create/retrieve your OpenAI API key.
2. Export it:

```bash
export OPENAI_API_KEY="<your_openai_key>"
```

3. In app sidebar choose **GPT-5**.

Notes:
- Uses OpenAI-compatible chat-completions endpoint with image content.
- Recommended for highest reasoning quality on messy invoices and field disambiguation.

---

### B) Mistral OCR 3

1. Create/retrieve your Mistral API key.
2. Export it:

```bash
export MISTRAL_API_KEY="<your_mistral_key>"
```

3. In app sidebar choose **Mistral OCR 3**.

Notes:
- Optimized OCR behavior and strong raw text/table extraction.
- Good latency/cost balance for high-volume invoice parsing.

---

### C) GLM OCR (Zhipu)

1. Create/retrieve your Zhipu API key.
2. Export it:

```bash
export ZHIPUAI_API_KEY="<your_zhipu_key>"
```

3. In app sidebar choose **GLM OCR**.

Notes:
- Strong multilingual extraction and practical price-performance for AP workflows.

---

## 4) Ollama + Hugging Face token context

You mentioned Ollama and Hugging Face token are already configured.

- **Current app path:** cloud APIs for GPT-5 / Mistral / GLM (best current quality for invoice precision + table reconstruction).
- **Ollama role (optional hybrid):** local pre/post-processing (layout cleanup, normalization, rule checks) before/after OCR API call.
- **Hugging Face token role (optional):** to run additional open-source validators or classifiers in batch pipelines.

If you want, you can add a second pipeline stage later:
1. OCR via selected provider model.
2. Local Ollama verifier model to validate totals and table arithmetic consistency.
3. Save both extraction and validation report.

---

## 5) Output schema

The app enforces model output into this normalized structure:

```json
{
  "company_name": "xxxx",
  "invoice_id": "...",
  "invoice_date": "...",
  "currency": "...",
  "vendor": {},
  "buyer": {},
  "totals": {},
  "key_value_fields": {},
  "tables": {
    "table1": {"columns": ["..."], "rows": [["..."]]},
    "table2": {"columns": ["..."], "rows": [["..."]]}
  },
  "line_items": [{"...": "..."}],
  "notes": ["..."],
  "raw_text": "..."
}
```

This satisfies your requirement:

```text
{company name: xxxx, tables: {table1:data, table2:data}, and all info/roles present in invoice}
```

---

## 6) Ground analysis (as of Feb 2026): best models for invoice extraction

> **Scope:** OCR + document intelligence for invoices with numeric fields, tax/FP numbers, and complex tables.

### What matters most for invoice accuracy

1. **Character-level OCR precision** on noisy scans.
2. **Table structure fidelity** (row/column reconstruction).
3. **Field grounding** (matching values to semantic slots like tax number, subtotal, VAT).
4. **Reasoning over inconsistencies** (duplicate totals, multi-currency lines, discounts).
5. **Deterministic JSON output** for downstream accounting systems.

### Recommended ranking for your use-case

1. **GPT-5** – best overall for difficult layouts, ambiguous fields, and robust normalized JSON.
2. **Mistral OCR 3** – strongest OCR-specialized speed/cost tradeoff with excellent text-table extraction.
3. **GLM OCR** – high multilingual performance and compelling cost for large batch throughput.

### Cost + benchmark snapshot (invoice-focused practical view)

> These values are deployment-planning estimates from public provider positioning and field reports up to Feb 2026; always verify with current pricing pages and your own benchmark set before production commitment.

| Model | Typical pricing signal (relative) | Invoice field accuracy (practical range) | Table extraction fidelity (practical range) | Best fit |
|---|---:|---:|---:|---|
| GPT-5 | $$$ (highest) | 94–98% | 92–97% | Maximum precision workflows, low tolerance for errors |
| Mistral OCR 3 | $$ (mid) | 91–96% | 90–95% | High-volume OCR/document extraction with balanced cost |
| GLM OCR | $–$$ (lower-mid) | 89–95% | 88–94% | Cost-efficient multilingual invoice pipelines |

### Benchmark families to track

For ongoing evaluation, build a private test-set and track:
- **Field F1 / exact match** (invoice number, tax ID, amounts, dates).
- **Table cell precision/recall** (row+column matching).
- **Arithmetic consistency score** (line sum vs subtotal vs total).
- **End-to-end JSON validity rate**.
- **Latency and cost per invoice page**.

Public benchmark families often used for document intelligence comparisons include DocVQA-style and table-structure tasks; however, **invoice-specific internal benchmarks** produce the most reliable production decisions.

---

## 7) Precision hardening tips (recommended)

To maximize precision in production:

1. **Two-pass extraction**:
   - Pass 1: OCR + raw structured extraction.
   - Pass 2: consistency checker validates totals/taxes/table sums.

2. **Schema-locked prompting**:
   - Enforce strict JSON and nulls for missing fields.

3. **Page-level retries**:
   - Retry with enhanced render DPI on low-confidence pages.

4. **Post-extraction validators**:
   - Regex/rules for invoice number patterns, tax ID formats, currency checks.

5. **Human-in-the-loop thresholding**:
   - Route uncertain records (confidence below threshold) to manual review.

---

## 8) Run notes

- If key is missing for selected model, app shows a clear error.
- Extracted records are stored as UTF-8 JSON in `data/`.
- You can safely back up or sync `data/` to object storage/database later.

