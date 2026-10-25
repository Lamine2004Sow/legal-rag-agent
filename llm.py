"""LLM DeepSeek (API compatible OpenAI) avec cache local des appels.

Usage :
    from llm import get_llm
    llm = get_llm()            # temperature 0 par défaut

La clé se met dans `.env` : DEEPSEEK_API_KEY=...  (modèle : DEEPSEEK_MODEL, optionnel).
Le cache SQLite (.cache/llm.sqlite) rend les rejeux d'évaluation gratuits et déterministes.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
CACHE_DIR = Path(".cache")
CACHE_PATH = CACHE_DIR / "llm.sqlite"

_cache_enabled = False


class DeepSeekChat(ChatOpenAI):
    """DeepSeek refuse `response_format=json_schema` (défaut de LangChain) : on passe par le tool calling."""

    def with_structured_output(self, schema, *, method: str = "function_calling", **kwargs):
        return super().with_structured_output(schema, method=method, **kwargs)


def enable_cache() -> None:
    """Active le cache SQLite des appels LLM (une seule fois par processus)."""
    global _cache_enabled
    if _cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(CACHE_PATH)))
    _cache_enabled = True


def get_llm(temperature: float = 0.0) -> DeepSeekChat:
    """Client DeepSeek. Lève une RuntimeError si la clé d'API est absente."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY est absente : ajoute-la dans le fichier .env à la racine du projet."
        )
    enable_cache()
    return DeepSeekChat(
        model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
        timeout=60,
    )
