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

mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Invoice", "View Invoices"]
)

def cleanup_files(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)

def get_companies():
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]


def get_invoices(company):
    path = os.path.join(DATA_DIR, company)
    return [f for f in os.listdir(path) if f.endswith(".json")]

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

if mode == "View Invoices":

    st.header("Invoice Repository")

    companies = get_companies()

    if not companies:
        st.info("No invoices found yet")
        st.stop()

    company = st.selectbox("Select Company", companies)

    invoices = get_invoices(company)

    if not invoices:
        st.warning("No invoices for this company")
        st.stop()

    invoice_file = st.selectbox("Select Invoice", invoices)

    file_path = os.path.join(DATA_DIR, company, invoice_file)

    with open(file_path, "r") as f:
        data = json.load(f)

    # Display nicely
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Invoice Summary")
        st.write("Company:", data.get("seller", {}).get("name"))
        st.write("Invoice No:", data.get("invoice_number"))
        st.write("Date:", data.get("issue_date"))
        st.write("Currency:", data.get("currency", {}).get("code"))
        st.write("Total:", data.get("total"))

        if "items" in data:
            st.subheader("Line Items")
            st.table(data["items"])

    with col2:
        st.subheader("Full JSON")
        st.json(data)



if mode == "Upload Invoice":

    st.header("Upload New Invoice")

    company_name = st.sidebar.text_input("Company Name", key="upload_company")

    uploaded_file = st.file_uploader(
    "Upload Invoice",
    type=["png", "jpg", "jpeg", "pdf"]
    )

    if uploaded_file is not None:
        temp_files = []
        data = None
        save_path = None

        # Unique temp name
        temp_path = f"temp_{datetime.now().timestamp()}_{uploaded_file.name}"
        temp_files.append(temp_path)

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

            try:
                with st.spinner("Processing invoice..."):

                    # PDF Handling
                    if temp_path.endswith(".pdf"):
                        image_paths = pdf_to_images(temp_path)
                        temp_files.extend(image_paths)

                        all_pages = []
                        for img in image_paths:
                            page_data = extract_with_gpt5(img)
                            all_pages.append(page_data)

                        data = all_pages[0]
                        merged_items = []

                        for page in all_pages:
                            if "items" in page:
                                merged_items.extend(page["items"])

                        data["items"] = merged_items

                    else:
                        data = extract_with_gpt5(temp_path)

                    invoice_number = data.get(
                        "invoice_number",
                        str(datetime.now().timestamp())
                    )

                    save_path = save_invoice(company_name, invoice_number, data)

            except Exception as e:
                st.error(f"Extraction failed: {e}")

            finally:
                cleanup_files(temp_files)

            # -------- Display ONLY if success --------
            if data:
                st.success(f"Saved to: {save_path}")

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
                        data.get("issue_date", "Not found")
                    )
                    st.write(
                        "Currency:",
                        data.get("currency", {}).get("code", "Not found")
                    )
                    st.write(
                        "Total:",
                        data.get("total", "Not found")
                    )

                    if "items" in data:
                        st.subheader("Items")
                        st.table(data["items"])

                with col2:
                    st.subheader("Full JSON")
                    st.json(data)