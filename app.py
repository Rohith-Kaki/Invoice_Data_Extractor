import json
import os
from datetime import datetime
from urllib.parse import urlencode

import streamlit as st

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(layout="wide", page_title="Invoice Data Extractor")


CASLY_FAQ = {
    "communities": "Casly Homes builds boutique residential communities across the Pacific Northwest, with a focus on thoughtful layouts, enduring materials, and small-scale developments where details receive personal attention.",
    "locations": "Casly Homes is rooted in the Pacific Northwest and highlights desirable Washington communities, including the Bothell area. For a specific current availability check, the bot can route visitors to the Casly Homes team.",
    "warranty": "Casly Homes positions each purchase with a structured QBW-backed warranty program designed to protect the home well beyond move-in.",
    "features": "Casly homes emphasize high ceilings, open floorplans, generous storage, abundant natural light, quality materials, and everyday livability for modern families.",
    "smart home": "Smart-home features are presented as optional and may vary by property, so the safest next step is to ask the team which homes currently include them.",
    "land": "Casly Homes also supports land-purchase conversations for property owners and agents who want guidance on unlocking a property's value.",
    "contact": "Visitors can contact Casly Homes through caslyhomes.com/contact or email info@caslyhomes.com for availability, tours, land inquiries, and sales questions.",
}

CASLY_KEYWORDS = {
    "casly", "home", "homes", "townhome", "townhomes", "property", "properties",
    "real estate", "builder", "build", "construction", "community", "communities",
    "bothell", "washington", "pacific northwest", "pnw", "warranty", "qbw",
    "smart", "tour", "availability", "available", "land", "purchase", "buyer",
    "floorplan", "floorplans", "storage", "ceiling", "light", "materials", "contact",
}


# -------- Shared UI Helpers --------

def inject_casly_styles():
    st.markdown(
        """
        <style>
        :root {
            --casly-ink: #17211b;
            --casly-forest: #234337;
            --casly-sage: #80936d;
            --casly-cream: #f7f1e7;
            --casly-stone: #d8cdbd;
            --casly-gold: #c8a96a;
        }
        .casly-shell {
            min-height: 82vh;
            padding: 34px;
            border-radius: 30px;
            background:
                radial-gradient(circle at 15% 10%, rgba(200,169,106,.26), transparent 28%),
                linear-gradient(135deg, #14231d 0%, #263f35 44%, #f7f1e7 44.2%, #efe4d4 100%);
            box-shadow: 0 28px 70px rgba(23,33,27,.20);
            color: var(--casly-ink);
        }
        .casly-hero {
            display: grid;
            grid-template-columns: 1.02fr .98fr;
            gap: 32px;
            align-items: stretch;
        }
        .casly-copy {
            color: #fffdf7;
            padding: 22px 12px 22px 6px;
        }
        .casly-kicker {
            display: inline-flex;
            gap: 10px;
            align-items: center;
            padding: 8px 13px;
            border: 1px solid rgba(255,255,255,.24);
            border-radius: 999px;
            background: rgba(255,255,255,.09);
            font-size: 13px;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .casly-copy h1 {
            margin: 22px 0 16px;
            font-size: clamp(36px, 6vw, 68px);
            line-height: .95;
            font-family: Georgia, 'Times New Roman', serif;
            letter-spacing: -.04em;
        }
        .casly-copy p {
            max-width: 620px;
            color: rgba(255,253,247,.82);
            font-size: 18px;
            line-height: 1.7;
        }
        .casly-proof-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 28px;
        }
        .casly-proof {
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 18px;
            padding: 15px;
            background: rgba(255,255,255,.08);
            backdrop-filter: blur(16px);
        }
        .casly-proof strong { display: block; font-size: 22px; color: #fff; }
        .casly-proof span { color: rgba(255,253,247,.72); font-size: 13px; }
        .chat-card {
            border-radius: 30px;
            overflow: hidden;
            background: rgba(255,253,247,.94);
            border: 1px solid rgba(255,255,255,.55);
            box-shadow: 0 24px 80px rgba(16,28,22,.30);
        }
        .chat-topbar {
            padding: 18px 20px;
            background: linear-gradient(135deg, #fffaf1, #efe2d0);
            border-bottom: 1px solid rgba(35,67,55,.13);
            display: flex;
            justify-content: space-between;
            gap: 16px;
            align-items: center;
        }
        .avatar-lockup { display: flex; align-items: center; gap: 12px; }
        .casly-avatar {
            width: 46px; height: 46px;
            border-radius: 16px;
            display: grid; place-items: center;
            background: linear-gradient(135deg, #234337, #80936d);
            color: #fff7e8;
            font-family: Georgia, 'Times New Roman', serif;
            font-weight: 700;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.20);
        }
        .avatar-lockup h3 { margin: 0; color: #17211b; font-size: 18px; }
        .avatar-lockup small { color: #657065; }
        .status-pill {
            padding: 8px 11px;
            border-radius: 999px;
            background: rgba(128,147,109,.16);
            color: #234337;
            font-weight: 650;
            font-size: 12px;
        }
        .chat-window {
            min-height: 470px;
            max-height: 540px;
            overflow-y: auto;
            padding: 22px;
            background:
                linear-gradient(rgba(255,253,247,.90), rgba(255,253,247,.90)),
                repeating-linear-gradient(45deg, rgba(35,67,55,.04) 0 1px, transparent 1px 18px);
        }
        .bubble-row { display: flex; margin: 13px 0; }
        .bubble-row.user { justify-content: flex-end; }
        .bubble {
            max-width: 78%;
            padding: 13px 15px;
            border-radius: 20px;
            line-height: 1.48;
            font-size: 15px;
        }
        .bubble.bot {
            color: #17211b;
            background: #fffaf1;
            border: 1px solid rgba(35,67,55,.10);
            border-bottom-left-radius: 7px;
            box-shadow: 0 10px 24px rgba(23,33,27,.06);
        }
        .bubble.user {
            color: #fffdf7;
            background: linear-gradient(135deg, #234337, #3f6655);
            border-bottom-right-radius: 7px;
        }
        .bubble-time { display: block; margin-top: 7px; opacity: .62; font-size: 11px; }
        .suggestion-wrap { padding: 0 22px 20px; background: rgba(255,253,247,.94); }
        .share-card, .guardrail-card {
            border-radius: 22px;
            padding: 18px;
            border: 1px solid rgba(35,67,55,.12);
            background: rgba(255,250,241,.82);
            box-shadow: 0 14px 34px rgba(23,33,27,.07);
        }
        .share-code {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px;
            padding: 13px;
            border-radius: 14px;
            background: #17211b;
            color: #f7f1e7;
            overflow-wrap: anywhere;
        }
        .stButton>button {
            border-radius: 999px;
            border: 1px solid rgba(35,67,55,.22);
            background: #fffaf1;
            color: #234337;
            font-weight: 650;
        }
        .stButton>button:hover {
            border-color: #c8a96a;
            color: #17211b;
        }
        @media (max-width: 960px) {
            .casly-shell { padding: 18px; }
            .casly-hero { grid-template-columns: 1fr; }
            .casly-proof-row { grid-template-columns: 1fr; }
            .bubble { max-width: 92%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
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


# -------- Invoice Helper Functions --------

def save_invoice(company, invoice_number, data):
    folder = os.path.join(DATA_DIR, company)
    os.makedirs(folder, exist_ok=True)

    path = os.path.join(folder, f"{invoice_number}.json")

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    return path


def pdf_to_images(pdf_path):
    """Convert PDF pages to images (300 DPI for accuracy)"""
    from pdf2image import convert_from_path

    pages = convert_from_path(pdf_path, dpi=300)

    image_paths = []
    base = os.path.splitext(pdf_path)[0]

    for i, page in enumerate(pages):
        img_path = f"{base}_page_{i+1}.jpg"
        page.save(img_path, "JPEG")
        image_paths.append(img_path)

    return image_paths


def extract_invoice_data(file_path):
    """Lazy-load the OpenAI extractor so the app can run without API keys until extraction."""
    from gpt_extractor import extract_with_gpt5

    return extract_with_gpt5(file_path)


# -------- CaslyHomes Chatbot Helpers --------

def is_casly_related(message):
    normalized = message.lower()
    return any(keyword in normalized for keyword in CASLY_KEYWORDS)


def get_casly_response(message):
    normalized = message.lower()

    if not is_casly_related(message):
        return (
            "I can only help with Casly Homes topics—communities, availability, tours, "
            "home features, warranty, smart-home options, land purchasing, and contact routing. "
            "Try asking: ‘What makes Casly Homes different?’"
        )

    if any(term in normalized for term in ["available", "availability", "tour", "schedule", "visit"]):
        return (
            "I can help visitors move toward a private tour. The best flow is: collect preferred community, "
            "budget range, move-in timing, and contact details, then route the lead to info@caslyhomes.com."
        )
    if any(term in normalized for term in ["warranty", "qbw", "protect"]):
        return CASLY_FAQ["warranty"]
    if any(term in normalized for term in ["smart", "technology", "automation"]):
        return CASLY_FAQ["smart home"]
    if any(term in normalized for term in ["land", "sell", "property owner", "agent"]):
        return CASLY_FAQ["land"]
    if any(term in normalized for term in ["where", "location", "bothell", "washington", "pnw", "pacific"]):
        return CASLY_FAQ["locations"]
    if any(term in normalized for term in ["feature", "ceiling", "floor", "storage", "light", "design", "material"]):
        return CASLY_FAQ["features"]
    if any(term in normalized for term in ["contact", "email", "phone", "reach"]):
        return CASLY_FAQ["contact"]

    return (
        f"{CASLY_FAQ['communities']} I can also answer about tours, warranty, smart-home options, "
        "land purchasing, and how to contact the Casly Homes team."
    )


def add_chat_message(role, content):
    st.session_state.casly_messages.append(
        {
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%I:%M %p"),
        }
    )


def render_chat_messages():
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)
    for message in st.session_state.casly_messages:
        role = message["role"]
        css_role = "user" if role == "user" else "bot"
        safe_content = message["content"].replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f"""
            <div class="bubble-row {css_role}">
                <div class="bubble {css_role}">
                    {safe_content}
                    <span class="bubble-time">{message['time']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_casly_chatbot():
    inject_casly_styles()

    if "casly_messages" not in st.session_state:
        st.session_state.casly_messages = [
            {
                "role": "assistant",
                "content": "Welcome to Casly Homes. Ask me about boutique communities, home features, tours, warranty, smart-home options, or land purchasing.",
                "time": datetime.now().strftime("%I:%M %p"),
            }
        ]

    st.markdown(
        """
        <div class="casly-shell">
            <div class="casly-hero">
                <section class="casly-copy">
                    <div class="casly-kicker">✦ CaslyHomes conversational prototype</div>
                    <h1>A polished home-finding assistant for boutique real estate.</h1>
                    <p>
                        This prototype shows the website widget experience: elegant, brand-aligned,
                        lead-friendly, and intentionally scoped to Casly Homes topics only.
                    </p>
                    <div class="casly-proof-row">
                        <div class="casly-proof"><strong>PNW</strong><span>Local builder positioning</span></div>
                        <div class="casly-proof"><strong>24/7</strong><span>Guided visitor answers</span></div>
                        <div class="casly-proof"><strong>Scoped</strong><span>Casly-only guardrails</span></div>
                    </div>
                </section>
                <section class="chat-card">
                    <div class="chat-topbar">
                        <div class="avatar-lockup">
                            <div class="casly-avatar">C</div>
                            <div><h3>Casly Concierge</h3><small>Boutique homes assistant</small></div>
                        </div>
                        <div class="status-pill">Online • Casly-only</div>
                    </div>
        """,
        unsafe_allow_html=True,
    )

    render_chat_messages()

    st.markdown('<div class="suggestion-wrap">', unsafe_allow_html=True)
    quick_prompts = [
        "What makes Casly Homes different?",
        "Can I schedule a tour?",
        "Tell me about the warranty.",
        "Do homes include smart-home features?",
    ]
    cols = st.columns(4)
    for col, prompt in zip(cols, quick_prompts):
        with col:
            if st.button(prompt, key=f"casly_prompt_{prompt}"):
                add_chat_message("user", prompt)
                add_chat_message("assistant", get_casly_response(prompt))
                st.rerun()
    st.markdown('</div></section></div></div>', unsafe_allow_html=True)

    user_prompt = st.chat_input("Ask Casly Concierge about homes, tours, warranty, or land purchasing...")
    if user_prompt:
        add_chat_message("user", user_prompt)
        add_chat_message("assistant", get_casly_response(user_prompt))
        st.rerun()

    st.divider()
    share_url = "https://your-streamlit-domain.example/casly-chatbot?" + urlencode({"mode": "chatbot", "brand": "caslyhomes"})
    embed_code = (
        '<iframe src="https://your-streamlit-domain.example/casly-chatbot?mode=chatbot&brand=caslyhomes" '
        'title="Casly Homes Chatbot" width="420" height="680" style="border:0;border-radius:24px;box-shadow:0 24px 70px rgba(0,0,0,.18);"></iframe>'
    )

    col1, col2 = st.columns([1.1, .9])
    with col1:
        st.markdown(
            f"""
            <div class="share-card">
                <h3>Shareable prototype link</h3>
                <p>Use this as the handoff URL when the Streamlit app is deployed to Streamlit Community Cloud, Render, or any internal demo site.</p>
                <div class="share-code">{share_url}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="guardrail-card">
                <h3>Website embed snippet</h3>
                <p>Drop this iframe into a staging page to test the component inside the existing CaslyHomes website.</p>
                <div class="share-code">{embed_code.replace('<', '&lt;').replace('>', '&gt;')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Prototype behavior and guardrails"):
        st.markdown(
            """
            - **On-brand scope:** Answers only Casly Homes, boutique communities, Washington/PNW location context, warranty, tours, smart-home options, land purchasing, and contact routing.
            - **Off-topic handling:** Politely refuses unrelated questions and suggests a CaslyHomes-focused prompt.
            - **Lead path:** Can be extended to collect name, email, timeline, preferred community, budget, and tour request.
            - **Share path:** Deploy this app and replace the placeholder domain in the link/iframe with the production demo URL.
            """
        )


# -------- Main Flow --------

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Model",
    ["GPT-5"]
)

mode = st.sidebar.radio(
    "Select Mode",
    ["CaslyHomes Chatbot Prototype", "Upload Invoice", "View Invoices"]
)

if mode == "CaslyHomes Chatbot Prototype":
    st.title("CaslyHomes Chatbot Prototype")
    render_casly_chatbot()

if mode == "View Invoices":
    st.title("Invoice Intelligence System")

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
    st.title("Invoice Intelligence System")

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
                            page_data = extract_invoice_data(img)
                            all_pages.append(page_data)

                        data = all_pages[0]
                        merged_items = []

                        for page in all_pages:
                            if "items" in page:
                                merged_items.extend(page["items"])

                        data["items"] = merged_items

                    else:
                        data = extract_invoice_data(temp_path)

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
