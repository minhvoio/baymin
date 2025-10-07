import requests, json
from IPython.display import display, Markdown, clear_output

def answer_this_prompt(prompt, stream=False, model="gpt-oss:latest", temperature=0, format=None, max_tokens=200):
    payload = {
        "prompt": 'You always return a JSON answer, for example {answer: "your answer here"}, now answer this prompt: ' + prompt,
        "model": model,
        "temperature": temperature,
        "max_new_tokens": max_tokens, # only when stream = False work
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

def generate_chat(prompt, stream=False, model="gpt-oss:latest", temperature=0.0, json_format=None, max_tokens=200):
    endpoint = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        # stream defaults to True server-side; we control it client-side by iterating lines
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens)
        }
    }
    # Optional: request structured JSON output
    if json_format is not None:
        # either "json" or a JSON schema dict
        payload["format"] = json_format

    with requests.post(endpoint, json=payload, stream=True) as resp:
        if resp.status_code != 200:
            return f"Failed: {resp.status_code} {resp.text}"

        full_text = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                # sometimes servers add a trailing line—ignore
                continue

            # /api/chat streaming chunks
            if "message" in chunk and "content" in chunk["message"]:
                content_piece = chunk["message"]["content"]
                full_text += content_piece
                if stream:
                    clear_output(wait=True)
                    display(Markdown(full_text))

            # handle tool_calls or other fields if present
            if chunk.get("done", False):
                break

        return full_text

# print(generate_chat("Hello, how are you?"))