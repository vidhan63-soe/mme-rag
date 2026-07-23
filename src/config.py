"""
Configuration and LLM provider routing.

Priority: GROQ_API_KEY → Groq (free, fast) | OPENAI_API_KEY → OpenAI
"""
from dotenv import load_dotenv
load_dotenv()
import os

# --------------- Embedding ---------------
EMBED_MODEL = "all-MiniLM-L6-v2"  # local, free, 384-dim
EMBED_DIM = 384

# --------------- LLM Provider ---------------
def get_provider():
    """Return (provider_name, api_key) based on env vars."""
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if groq_key:
        return "groq", groq_key
    if openai_key:
        return "openai", openai_key
    raise EnvironmentError(
        "Set GROQ_API_KEY or OPENAI_API_KEY. "
        "Groq is free at console.groq.com — recommended for this prototype."
    )

# Model configs per provider
MODELS = {
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "cost_per_1m_input": 0.59,   # USD
        "cost_per_1m_output": 0.79,  # USD
    },
    "openai": {
        "model": "gpt-4o-mini",
        "cost_per_1m_input": 0.15,   # USD
        "cost_per_1m_output": 0.60,  # USD
    },
}

# --------------- Paths ---------------
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_colleges.csv")
