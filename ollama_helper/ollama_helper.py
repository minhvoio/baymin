import requests, json
from IPython.display import display, Markdown, clear_output
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from bn_helpers.constants import MODEL, OLLAMA_URL
from ollama_helper.structure_output import AnswerStructure, QuizAnswer
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

def answer_this_prompt(
    prompt,
    stream=False,
    model=MODEL,
    temperature=0.0,
    top_p=None,
    seed=None,
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
            **({"top_p": top_p} if top_p is not None else {}),
            **({"seed": seed} if seed is not None else {}),
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

async def get_answer_from_ollama(prompt, model=MODEL, max_tokens=1000, temperature=0.3, stream=False, show_thinking=False, top_p=None, seed=None):
    result = await _run_ollama_agent(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_type=AnswerStructure,
        stream=stream,
        top_p=top_p,
        seed=seed,
    )
    if show_thinking:
        return result.output.answer, getattr(result.output, 'thinking', None)
    return result.output.answer

async def get_quiz_answer_from_thinking_model(prompt, model=MODEL, max_tokens=1000, temperature=0, stream=False, top_p=None, seed=None):
    result = await _run_ollama_agent(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        output_type=QuizAnswer,
        stream=stream,
        top_p=top_p,
        seed=seed,
    )
    return result.output.one_letter_answer

def get_quiz_answer_from_thinking_model_sync(prompt, model=MODEL, max_tokens=1000, temperature=0, stream=False, top_p=None, seed=None):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(
                get_quiz_answer_from_thinking_model(
                    prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    top_p=top_p,
                    seed=seed,
                )
            )
        else:
            return loop.run_until_complete(
                get_quiz_answer_from_thinking_model(
                    prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    top_p=top_p,
                    seed=seed,
                )
            )
    except RuntimeError:
        return asyncio.run(
            get_quiz_answer_from_thinking_model(
                prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                top_p=top_p,
                seed=seed,
            )
        )

async def _run_ollama_agent(prompt, model, max_tokens, temperature, output_type, stream=False, top_p=None, seed=None):
    ollama_model = OpenAIChatModel(
        model_name=model,
        provider=OllamaProvider(base_url=OLLAMA_URL + 'v1'),  
    )
    agent = Agent(ollama_model, output_type=output_type)
    return await agent.run(
        prompt,
        model_settings={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **({"top_p": top_p} if top_p is not None else {}),
            **({"seed": seed} if seed is not None else {}),
        },
    )