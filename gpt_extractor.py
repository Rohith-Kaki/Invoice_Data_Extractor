import os
import json
import base64
import mimetypes
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT = """
Extract all data from this invoice and return STRICT JSON only.

Schema:
{
"document_type":"invoice",
"invoice_title":string,
"invoice_number":string,
"issue_date":string,
"due_date":string,
"payment_terms":string,
"purchase_order_number":string,
"currency":{"code":string,"symbol":string},
"seller":{"name":string,"contact_name":string},
"bill_to":{"name":string},
"ship_to":{"address":string},
"balance_due":number,
"items":[
{"description":string,"quantity":number,"unit_price":number,"amount":number}
],
"subtotal":number,
"discount":{"amount":number,"rate":number},
"tax":{"amount":number,"rate":number},
"shipping":number,
"total":number,
"notes":string,
"terms":string,
"calculations_verified":boolean
}

Rules:
- Return valid JSON only (no text, no markdown)
- Use null if missing
- Keep numbers exact
- Items must be an array
- Ensure totals match line items

"""


def encode_file(path):
    """Convert file to base64 data URL"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"

    return f"data:{mime};base64,{b64}"


def extract_with_gpt5(file_path):
    """
    Extract invoice data from image file using GPT-5
    Returns parsed JSON dict
    """

    data_url = encode_file(file_path)

    response = client.responses.create(
        model="gpt-5",
        max_output_tokens=3000,
        input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": data_url},
                ]
            }]
    )

    # Safe text extraction (SDK version independent)
    text_output = None

    if hasattr(response, "output_text") and response.output_text:
        text_output = response.output_text
    elif response.output:
        for item in response.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if hasattr(c, "text"):
                        text_output = c.text
                        break

    if text_output is None:
        raise Exception("No text returned from model")

    try:
        data = json.loads(text_output)
    except:
        print("Invalid JSON returned:")
        print(text_output)
        raise

    # Clean empty line items
    if "items" in data:
        data["items"] = [
            item for item in data["items"]
            if item.get("description") or (item.get("amount") or 0) > 0
        ]

    return data