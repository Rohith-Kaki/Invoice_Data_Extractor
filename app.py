import base64
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import requests
import streamlit as st
from PIL import Image


DATA_ROOT = Path("data")
DATA_ROOT.mkdir(exist_ok=True)


SYSTEM_PROMPT = """
You are an invoice document-intelligence engine.
Extract invoice data with maximum precision.
Return STRICT JSON only with this shape:
{
  "company_name": "...",
  "invoice_id": "...",
  "invoice_date": "...",
  "currency": "...",
  "vendor": {...},
  "buyer": {...},
  "totals": {...},
  "key_value_fields": {"field_name": "value"},
  "tables": {
    "table1": {"columns": ["..."], "rows": [["..."], ["..."]]}
  },
  "line_items": [{...}],
  "notes": [...],
  "raw_text": "..."
}
Rules:
- Preserve numbers exactly, including invoice number / fp numbers / tax IDs.
- Extract all tables with consistent rows and columns.
- Use null when missing.
- No markdown, no explanation.
""".strip()


@dataclass
class ModelConfig:
    display_name: str
    provider: str
    model_name: str
    env_key: str
    endpoint: str


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "GPT-5": ModelConfig(
        display_name="GPT-5 (OpenAI Responses API)",
        provider="openai",
        model_name="gpt-5",
        env_key="OPENAI_API_KEY",
        endpoint="https://api.openai.com/v1/responses",
    ),
    "Mistral OCR 3": ModelConfig(
        display_name="Mistral OCR 3",
        provider="mistral",
        model_name="mistral-ocr-latest",
        env_key="MISTRAL_API_KEY",
        endpoint="https://api.mistral.ai/v1/chat/completions",
    ),
    "GLM OCR": ModelConfig(
        display_name="GLM OCR (Zhipu)",
        provider="glm",
        model_name="glm-4.5v",
        env_key="ZHIPUAI_API_KEY",
        endpoint="https://open.bigmodel.cn/api/paas/v4/chat/completions",
    ),
    "Ollama (local fallback)": ModelConfig(
        display_name="Ollama (local fallback)",
        provider="ollama",
        model_name="llava:13b",
        env_key="",
        endpoint="http://localhost:11434/api/chat",
    ),
}


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_pdf_to_images(pdf_bytes: bytes, zoom: float = 2.0) -> List[Image.Image]:
    images: List[Image.Image] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return images


def parse_json_strict(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_openai_responses(config: ModelConfig, images: List[Image.Image], company_name: str) -> Dict[str, Any]:
    api_key = os.getenv(config.env_key)
    if not api_key:
        raise RuntimeError(f"Missing API key: {config.env_key}")

    user_content = [{"type": "input_text", "text": f"Company name hint: {company_name}. Extract all invoice data."}]
    for img in images:
        user_content.append({"type": "input_image", "image_url": pil_to_data_url(img)})

    payload = {
        "model": config.model_name,
        "reasoning": {"effort": "medium"},
        "temperature": 0,
        "text": {"format": {"type": "json_object"}},
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ],
    }

    response = requests.post(
        config.endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=240,
    )
    if response.status_code >= 400:
        details = response.text
        raise RuntimeError(
            "OpenAI request failed. This usually means model access/billing is not enabled for your key or the model is unavailable. "
            f"Status={response.status_code}. Response={details[:500]}"
        )

    data = response.json()
    text = data.get("output_text", "")
    if not text:
        fragments: List[str] = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in {"output_text", "text"}:
                        fragments.append(c.get("text", ""))
        text = "\n".join(fragments)

    extracted = parse_json_strict(text)
    extracted.setdefault("company_name", company_name)
    return extracted


def extract_chat_vision(config: ModelConfig, images: List[Image.Image], company_name: str) -> Dict[str, Any]:
    api_key = os.getenv(config.env_key)
    if not api_key:
        raise RuntimeError(f"Missing API key: {config.env_key}")

    image_parts = [{"type": "image_url", "image_url": {"url": pil_to_data_url(img)}} for img in images]

    payload = {
        "model": config.model_name,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Company name hint: {company_name}. Extract all invoice data."}, *image_parts],
            },
        ],
    }

    response = requests.post(
        config.endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=240,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"{config.display_name} request failed (status {response.status_code}). "
            f"Response: {response.text[:500]}"
        )

    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, list):
        text = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    else:
        text = content

    extracted = parse_json_strict(text)
    extracted.setdefault("company_name", company_name)
    return extracted


def extract_with_ollama(config: ModelConfig, images: List[Image.Image], company_name: str, ollama_model: str) -> Dict[str, Any]:
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Company name hint: {company_name}. "
        "Extract all invoice data and return strict JSON only."
    )
    response = requests.post(
        config.endpoint,
        json={
            "model": ollama_model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [pil_to_base64(img) for img in images],
                }
            ],
        },
        timeout=240,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama request failed (status {response.status_code}). Ensure `ollama serve` is running and model is pulled. "
            f"Response: {response.text[:500]}"
        )

    body = response.json()
    text = body.get("message", {}).get("content", "")
    extracted = parse_json_strict(text)
    extracted.setdefault("company_name", company_name)
    return extracted


def call_vision_model(config: ModelConfig, images: List[Image.Image], company_name: str, ollama_model: str) -> Dict[str, Any]:
    if config.provider == "openai":
        return extract_openai_responses(config, images, company_name)
    if config.provider in {"mistral", "glm"}:
        return extract_chat_vision(config, images, company_name)
    if config.provider == "ollama":
        return extract_with_ollama(config, images, company_name, ollama_model)
    raise RuntimeError(f"Unsupported provider: {config.provider}")


def make_safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def save_extraction(company: str, invoice_filename: str, payload: Dict[str, Any]) -> Path:
    company_dir = DATA_ROOT / make_safe_name(company)
    invoice_dir = company_dir / make_safe_name(Path(invoice_filename).stem)
    invoice_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "company_name": company,
        "invoice_file": invoice_filename,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "data": payload,
    }
    output_path = invoice_dir / "data.json"
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_saved_tree() -> Dict[str, Dict[str, Dict[str, Any]]]:
    tree: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for company_dir in sorted([p for p in DATA_ROOT.glob("*") if p.is_dir()]):
        tree[company_dir.name] = {}
        for invoice_dir in sorted([p for p in company_dir.glob("*") if p.is_dir()]):
            data_file = invoice_dir / "data.json"
            if data_file.exists():
                try:
                    tree[company_dir.name][invoice_dir.name] = json.loads(data_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    tree[company_dir.name][invoice_dir.name] = {"error": "Invalid JSON in data.json"}
    return tree


def show_saved_tree(tree: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    st.subheader("Saved invoice data")
    if not tree:
        st.info("No extracted invoices yet.")
        return

    for company, invoices in tree.items():
        with st.expander(f"{company}", expanded=False):
            for invoice_name, payload in invoices.items():
                st.markdown(f"- **{invoice_name}**")
                with st.expander("data", expanded=False):
                    st.json(payload)


def main() -> None:
    st.set_page_config(page_title="Invoice Data Extractor", layout="wide")
    st.title("Invoice Data Extractor")
    st.caption("Upload invoices, select OCR model, extract structured data, and store it by company/invoice.")

    with st.sidebar:
        st.header("Configuration")
        model_key = st.selectbox("Choose extraction model", list(MODEL_REGISTRY.keys()), index=0)
        model_cfg = MODEL_REGISTRY[model_key]
        if model_cfg.env_key:
            st.markdown("Required env key:")
            st.code(model_cfg.env_key)
        else:
            st.success("No cloud key needed for local Ollama mode.")

        ollama_model = st.text_input("Ollama model name", value="llava:13b")

    st.info(
        "If cloud APIs fail due to subscription/access errors (400/401/403), choose **Ollama (local fallback)** and run:\n"
        "`ollama pull llava:13b` then `ollama serve`."
    )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Extract from invoice")
        company_name = st.text_input("Company name", placeholder="company1")
        uploaded_file = st.file_uploader("Invoice file (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])

        run = st.button("Extract and save", type="primary", disabled=not (company_name and uploaded_file))
        if run:
            try:
                if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                    images = render_pdf_to_images(uploaded_file.getvalue())
                else:
                    images = [Image.open(uploaded_file).convert("RGB")]

                with st.spinner(f"Extracting using {model_cfg.display_name}..."):
                    result = call_vision_model(model_cfg, images, company_name, ollama_model)

                saved_path = save_extraction(company_name, uploaded_file.name, result)
                st.success(f"Saved extraction to: {saved_path}")
                st.json(result)
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")
                st.warning(
                    "Tip: If you don't have paid access for all cloud providers, use one provider that is enabled or switch to "
                    "Ollama local fallback in the sidebar."
                )

    with col_right:
        show_saved_tree(load_saved_tree())

    st.markdown("### Output structure")
    st.code(
        '{company name: xxxx, tables: {table1: {columns: [...], rows: [...]}, table2: {...}}, "all_other_fields": {...}}'
    )


if __name__ == "__main__":
    main()
