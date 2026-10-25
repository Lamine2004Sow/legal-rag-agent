"""Fixtures partagées des tests.

- sys.path : racine du projet et eval/ (pour importer graph, retriever, run_eval, baseline...).
- retriever : HybridRetriever réel, portée session. Même config que l'agent (lue dans .env).
- fake_llm : classe FakeLLM, faux LLM déterministe, sans appel réseau.
- --run-eval : la suite de régression relance eval/run_eval.py si aucun run ne correspond.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Même configuration que l'agent (RETRIEVER_DATA_DIR, RETRIEVER_MODEL, DEEPSEEK_MODEL...).
load_dotenv(ROOT / ".env")

from ingest import DEFAULT_MODEL  # noqa: E402
from retriever import HybridRetriever  # noqa: E402


class FakeLLM:
    """Faux chat model. `with_structured_output(schema)` renvoie une chaîne qui appelle
    `responses[schema.__name__](messages)` et enregistre les messages dans `calls`."""

    def __init__(self, responses: dict[str, Callable] | None = None) -> None:
        self.responses = responses or {}
        self.calls: dict[str, list] = {}

    def with_structured_output(self, schema, **kwargs):
        name = schema.__name__

        def invoke(prompt_value):
            messages = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value
            self.calls.setdefault(name, []).append(messages)
            return self.responses[name](messages)

        return RunnableLambda(invoke)


@pytest.fixture
def fake_llm():
    """La classe FakeLLM : `fake_llm({"NomDuSchema": fonction})` construit un faux LLM."""
    return FakeLLM


@pytest.fixture(scope="session")
def retriever() -> HybridRetriever:
    """Retriever réel, chargé une seule fois. Skip si l'index n'a pas été construit."""
    data_dir = Path(os.environ.get("RETRIEVER_DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if not (data_dir / "faiss_index" / "index.faiss").exists() or not (data_dir / "articles.parquet").exists():
        pytest.skip("index absent : lancer python ingest.py")
    return HybridRetriever(data_dir, os.environ.get("RETRIEVER_MODEL", DEFAULT_MODEL))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "regression: compare le run courant de l'évaluation au baseline (voir tests/test_regression.py)"
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="lancer eval/run_eval.py --label regression si aucun run ne correspond à la config courante",
    )
