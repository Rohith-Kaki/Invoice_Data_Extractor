import streamlit as st
import os
import json
from datetime import datetime
from pdf2image import convert_from_path

from gpt_extractor import extract_with_gpt5

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(layout="wide")
st.title("Invoice Intelligence System")

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Model",
    ["GPT-5"]
)

company_name = st.sidebar.text_input("Company Name")

uploaded_file = st.file_uploader(
    "Upload Invoice",
    type=["png", "jpg", "jpeg", "pdf"]
)


# -------- Helper Functions --------

def save_invoice(company, invoice_number, data):
    folder = os.path.join(DATA_DIR, company)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{invoice_number}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return path


def pdf_to_images(pdf_path):
    """Convert PDF pages to images (300 DPI for accuracy)"""
    pages = convert_from_path(pdf_path, dpi=300)

    image_paths = []
    base = os.path.splitext(pdf_path)[0]

    for i, page in enumerate(pages):
        img_path = f"{base}_page_{i+1}.jpg"
        page.save(img_path, "JPEG")
        image_paths.append(img_path)

    return image_paths


# -------- Main Flow --------

if uploaded_file:

    temp_path = f"temp_{uploaded_file.name}"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Preview
    if uploaded_file.type == "application/pdf":
        st.info("PDF uploaded — will convert to images")
    else:
        st.image(temp_path, width=400)

    if st.button("Extract Data"):

        if not company_name:
            st.error("Enter Company Name in sidebar")
            st.stop()

        with st.spinner("Processing invoice..."):

            # PDF Handling
            if temp_path.endswith(".pdf"):
                image_paths = pdf_to_images(temp_path)

                all_pages = []
                for img in image_paths:
                    page_data = extract_with_gpt5(img)
                    all_pages.append(page_data)

                # Merge items from all pages
                data = all_pages[0]
                merged_items = []

                for page in all_pages:
                    if "items" in page:
                        merged_items.extend(page["items"])

                data["items"] = merged_items

            else:
                data = extract_with_gpt5(temp_path)

            # Invoice number fallback
            invoice_number = data.get(
                "invoice_number",
                str(datetime.now().timestamp())
            )

            save_path = save_invoice(company_name, invoice_number, data)

        st.success(f"Saved to: {save_path}")

        # -------- Display --------

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Invoice Summary")

            st.write(
                "Company:",
                data.get("bill_to", {}).get("name", "Not found")
            )

            st.write(
                "Invoice No:",
                data.get("invoice_number", "Not found")
            )

            st.write(
                "Date:",
                data.get("dates", {}).get("issue_date", "Not found")
            )

            st.write(
                "Currency:",
                data.get("currency", {}).get("code", "Not found")
            )

            st.write(
                "Total:",
                data.get("total", "Not found")
            )

            if "line_items" in data:
                st.subheader("line_items")
                st.table(data["line_items"])

        with col2:
            st.subheader("Full JSON")
            st.json(data)
