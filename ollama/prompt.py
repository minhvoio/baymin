# MODEL = "qwen3:1.7b"
# MODEL = "gpt-oss-bn-json"

import requests
import json
from IPython.display import display, Markdown, clear_output

def answer_this_prompt(prompt, stream=False, model="qwen3:1.7b", temperature=0, format=None):
    payload = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "max_new_tokens": 50, # only when stream = False work
        "format": format
    }
    headers = {
        'Content-Type': 'application/json'
    }
    endpoint = "http://localhost:11434/api/generate"

    # Send the POST request with streaming enabled
    with requests.post(endpoint, headers=headers, json=payload, stream=True) as response:
        if response.status_code == 200:
            try:
                # Process the response incrementally
                full_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():  # Skip empty lines
                        response_json = json.loads(line)
                        chunk = response_json.get("response", "")
                        full_response += chunk
                        
                        # Render the response as Markdown
                        if stream:
                            clear_output(wait=True)
                            display(Markdown(full_response))
                        
                return full_response
            except json.JSONDecodeError as e:
                return "Failed to parse JSON: " + str(e)
        else:
            return "Failed to retrieve response: " + str(response.status_code)