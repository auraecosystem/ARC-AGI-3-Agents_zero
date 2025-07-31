import re

def extract_first_json_block(markdown_text):
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, markdown_text, re.DOTALL)
    return match.group(1).strip() if match else ""

markdown = """
Some text here.

```json
{
  "name": "Alice",
  "age": 30
}
````

Another block:

```json
{
  "city": "Wonderland"
}
```

"""

result = extract_first_json_block(markdown)
print(result)
