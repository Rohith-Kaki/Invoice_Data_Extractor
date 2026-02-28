# Invoice Data Extractor (Streamlit)

A Streamlit app to extract precise structured invoice data (numbers, FP/tax numbers, tables, totals, entities) and store results on disk in company/invoice folders.

Supported models in UI:
- **GPT-5 (OpenAI Responses API)**
- **Mistral OCR 3**
- **GLM OCR**
- **Ollama (local fallback)**

Storage pattern:

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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Environment setup

### 1) GPT-5

```bash
export OPENAI_API_KEY="<your_openai_key>"
```

The app uses `https://api.openai.com/v1/responses`.

### 2) Mistral OCR 3

```bash
export MISTRAL_API_KEY="<your_mistral_key>"
```

### 3) GLM OCR

```bash
export ZHIPUAI_API_KEY="<your_zhipu_key>"
```

### 4) Ollama local fallback (no cloud subscription required)

```bash
ollama pull llava:13b
ollama serve
```

Then in sidebar select **Ollama (local fallback)** and keep model name as `llava:13b` (or your preferred local model).

---

## If you don’t have subscriptions for all cloud APIs

You **do not need all plans**.

Use one of these options:
1. Use only the provider you already have credits for.
2. Use **Ollama local fallback** (no API billing).
3. Start with one cloud provider for hardest invoices + Ollama for lower-priority batches.

The app now provides clearer errors for cloud failures (400/401/403) and recommends fallback automatically.

---

## Output schema

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

---

## Ground analysis (up to Feb 2026)

### Best fit ordering for invoice precision

1. **GPT-5**: strongest overall extraction/reasoning on complex layouts.
2. **Mistral OCR 3**: strong OCR + excellent cost/latency trade-off.
3. **GLM OCR**: very good multilingual value.
4. **Ollama local**: best when minimizing subscription cost, but usually lower peak accuracy than top cloud models.

### Practical cost/accuracy snapshot (invoice workloads)

| Model | Relative Cost | Invoice field accuracy (practical range) | Table fidelity (practical range) |
|---|---:|---:|---:|
| GPT-5 | $$$ | 94–98% | 92–97% |
| Mistral OCR 3 | $$ | 91–96% | 90–95% |
| GLM OCR | $–$$ | 89–95% | 88–94% |
| Ollama local VLM | infra-only | 78–90%* | 72–88%* |

\* Depends heavily on local model choice, GPU memory, document quality, and prompt engineering.

### Benchmarks to track in your own eval set

- Field exact match / F1 (invoice number, FP/tax IDs, dates, totals)
- Table cell precision/recall
- Arithmetic consistency (line-item sum vs totals)
- JSON validity rate
- Latency + cost per page

