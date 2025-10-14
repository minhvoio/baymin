import requests, json
from IPython.display import display, Markdown, clear_output

def answer_this_prompt(
    prompt,
    stream=False,
    model="llama3.1:70b",
    temperature=0.0,
    format=None,                # e.g., "json" to enable JSON mode; otherwise None
    num_ctx=8000,              # <= 131072 
    max_tokens=1800,            
    num_keep=64                 # keep first N tokens (system/header) when truncating
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,       
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "num_keep": num_keep
        }
    }
    if format:
        payload["format"] = format

    headers = {"Content-Type": "application/json"}
    endpoint = "http://localhost:11434/api/generate"

    with requests.post(endpoint, headers=headers, json=payload, stream=stream) as r:
        if r.status_code != 200:
            return f"Failed to retrieve response: {r.status_code}"
        full = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                return "Failed to parse JSON: " + str(e)
            chunk = obj.get("response", "")
            full += chunk
            if stream:
                clear_output(wait=True)
                display(Markdown(full))
        return full