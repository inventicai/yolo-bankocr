import json
import re
import requests
import base64
import os 
from dotenv import load_dotenv

load_dotenv()

FOLDER_PATH = os.getenv("FOLDER_PATH")
def extract_json_from_content(content):
    """
    Extract JSON from content that may have extra text or markdown formatting.
    Handles cases where content includes ```json markers or other text.
    """
    # Remove markdown JSON code blocks if present
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*$', '', content)
    
    # Try to find JSON object pattern
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, return empty predictions
    return {"predictions": []}

# Modified function to use the extraction
def process_ocr_output(content, basename):
    """
    Process OCR output content and extract account numbers.
    Handles cases where content may not be pure JSON.
    """
    # Extract JSON from content
    parsed = extract_json_from_content(content)
    
    # Safely get predictions
    predictions = parsed.get("predictions", [])
    
    # Extract account numbers from predictions
    account_numbers = []
    for pred in predictions:
        if isinstance(pred, dict) and "account_number" in pred:
            account_numbers.append(str(pred["account_number"]))
    
    # Ensure we have at least 3 predictions (pad with empty strings if needed)
    while len(account_numbers) < 3:
        account_numbers.append("")
    
    return {f'{basename}': account_numbers[:3]}


def run_inference(model_name, image_base64,basename):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            """Extract the Account Number from the image. Return STRICT JSON ONLY:
                            {
                            "predictions": [
                                {"account_number": "<string>", "confidence": 0.00},
                                {"account_number": "<string>", "confidence": 0.00},
                                {"account_number": "<string>", "confidence": 0.00}
                            ]
                            }
                            Rules: Return exactly 3 predictions in strict json, confidence between 0 and 1, sort by confidence DESC. No explanations."""
                            )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }

    response = requests.post(
        url=os.getenv("OPEN_ROUTER_API_URL"),
        headers={
            "Authorization": f"Bearer {os.getenv('OPEN_ROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        json=payload
    )

    resp_json = response.json()
    # -----------------------------
    # Extract predictions
    # -----------------------------
    content = resp_json["choices"][0]["message"]["content"]

    return process_ocr_output(content, basename)


final_dict={}  
model_name=os.getenv("MODEL_REPO")
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(FOLDER_PATH, filename)
        key_1=os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")    
        try:
            run_inference_result = run_inference(
                model_name=model_name,
                image_base64=image_base64,
                basename=key_1)
            final_dict.update(run_inference_result)
        except:
            pass
        
os.makedirs(model_name, exist_ok=True)
with open(f"{model_name}.json",'w') as f:
    json.dump(final_dict,f,indent=4)    