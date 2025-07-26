import json
import re

def extract_json_block(text):
    """
    Extracts the first valid JSON object from a text string.
    """
    try:
        match = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON block found in the input text.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

# Example usage
response_text = """
The hypothesis states the primary objective is to push the large orange/blue block...
{
  "reason": "The objective is to push the large orange/blue block into the black goal.",
  "action": "S"
}
"""

parsed = extract_json_block(response_text)
print(parsed)
