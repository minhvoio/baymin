import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MODEL = os.getenv("LLM_MODEL", "gpt-oss:latest")
MODEL_QUIZ = os.getenv("LLM_MODEL_QUIZ", "llama3.1:70b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/")
OLLAMA_CHAT_URL = OLLAMA_URL + "api/chat"
