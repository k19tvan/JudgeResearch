from google import genai
from dotenv import load_dotenv
import os
import re
import json

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

def clean_json_response(raw_text: str) -> str:
    if not raw_text:
        raise ValueError("Empty AI response")

    text = raw_text.strip()

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        return m.group(1).strip()

    return text

def validate_ai_response(ai_raw_response: str) -> bool:
    try:
        json_string = clean_json_response(ai_raw_response)
        cleaned_json = json.loads(json_string)
        return True
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return False

prompt = """
You are a strict JSON Validator and Senior Deep Learning Engineer. Your task is to ingest a raw, potentially malformed or technically inaccurate JSON string representing a roadmap of programming tasks, repair it, and output a standardized, production-ready JSON array.

### Your Objectives:

1. **Syntax & Formatting Repair**:
- Detect and fix any syntax errors (e.g., missing commas, unescaped double quotes inside descriptions, trailing commas, mismatched brackets, or unescaped newline characters).
- Ensure the output is a perfectly valid JSON array of objects.

2. **Technical Correction**:
- Check the technical accuracy of the deep learning concepts, PyTorch references, and module structures inside each object.
- If there is an inaccurate technical term, mathematical error, or confusing description, refine it to be scientifically accurate while keeping the original context and intention.

3. **Schema Enforcement**:
- Each object in the array MUST contain exactly these three keys:
- "title": (String) Concise, professional technical title.
- "description": (String) Clearly defined task requirements, parameter lists, and expected behavior.
- "target_module": (String) Reference to the PyTorch module, class, or script path.
- Do not allow any extra keys or nested structures.

4. **Response Format Constraint**:
- Output ONLY the clean, verified, and parsed JSON array.
- Do not write any conversational introduction, notes, markdown explanation, or post-text. The response must start with `[` and end with `]`.

---
### RAW INPUT JSON TO REPAIR AND NORMALIZE:

{last_ai_response}

"""

def main():
    with open("latest_ai_response.txt", "r", encoding="utf-8") as f:
        ai_raw_response = f.read()
    print(validate_ai_response(ai_raw_response))


    text = prompt.format(last_ai_response=ai_raw_response)

    ai_response_fixed = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        config={
            "system_instruction": "You are a strict JSON Validator and Senior Deep Learning Engineer. Your task is to ingest a raw, potentially malformed or technically inaccurate JSON string representing a roadmap of programming tasks, repair it, and output a standardized, production-ready JSON array. Follow the objectives and constraints outlined in the prompt meticulously."
        },
        contents=f"text:{text}"
    ).text

    print(validate_ai_response(ai_response_fixed))

main()