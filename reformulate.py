"""Reformulation de la question en vocabulaire juridique, puis recherche multi-requêtes.

Usage :
    python reformulate.py "ma question"

Le LLM transforme la question en 2-3 requêtes rédigées comme des articles de code ;
chaque requête (et la question d'origine) est cherchée avec HybridRetriever, puis les
résultats sont fusionnés par RRF.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm import get_llm
from retriever import RRF_K, HybridRetriever

logger = logging.getLogger(__name__)



class Reformulation(BaseModel):
    queries: list[str] = Field(
        description="2 à 3 requêtes de recherche courtes, en vocabulaire juridique (termes exacts des codes)."
    )
    code_hint: Literal["Code civil", "Code de la consommation", "inconnu"]


REFORMULATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Tu es juriste en droit français. Tu transformes la question d'un citoyen en 2 à 3 requêtes \
de recherche destinées au Code civil et au Code de la consommation.

Chaque requête est rédigée comme le texte d'un article de code, avec les termes juridiques exacts \
(« droit de rétractation », « vices rédhibitoires », « prescription », « responsabilité du fait des animaux »…). \
Fais une requête par notion juridique distincte, courte (une phrase ou un groupe de termes).

Règles :
- Ne réponds PAS à la question.
- N'invente pas de numéro d'article. Si la question cite un article, garde ce numéro dans une des requêtes.
- code_hint : « Code civil » ou « Code de la consommation » si la question relève clairement de l'un des deux, sinon « inconnu ».

Exemples (illustratifs) :
Question : Mon fils de 15 ans peut-il ouvrir un compte bancaire seul ?
Requêtes : capacité du mineur non émancipé à accomplir des actes de la vie civile ; actes accomplis par un mineur sans représentant légal
code_hint : Code civil

Question : Le vendeur refuse de reprendre un produit défectueux acheté il y a 6 mois.
Requêtes : garantie légale de conformité, défaut de conformité présumé ; obligation du vendeur de réparer ou remplacer le bien
code_hint : Code de la consommation""",
        ),
        ("human", "{question}"),
    ]
)


def reformulate(question: str, llm: BaseChatModel) -> Reformulation:
    """Appelle le LLM ; en cas d'échec ou de réponse vide, renvoie une liste de requêtes vide."""
    empty = Reformulation(queries=[], code_hint="inconnu")
    try:
        chain = REFORMULATE_PROMPT | llm.with_structured_output(Reformulation)
        result = chain.invoke({"question": question})
    except Exception:
        logger.exception("Reformulation échouée, recherche sur la question seule.")
        return empty
    queries = [q.strip() for q in result.queries if q.strip()]
    if not queries:
        logger.warning("Reformulation vide pour : %s", question)
        return empty
    return Reformulation(queries=queries, code_hint=result.code_hint)


def _key(doc: Document) -> tuple:
    return (doc.metadata["article_id"], doc.metadata["chunk"])


def multi_query_search(
    retriever: HybridRetriever,
    question: str,
    queries: list[str],
    k: int = 5,
    n_candidates: int = 30,
) -> list[Document]:
    """Recherche la question et chaque reformulation, puis fusionne par RRF."""
    rankings = [
        retriever.search(q, k=n_candidates, n_candidates=n_candidates) for q in [question, *queries]
    ]

    score: dict[tuple, float] = defaultdict(float)
    count: dict[tuple, int] = defaultdict(int)
    exact: set[tuple] = set()
    first: dict[tuple, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            key = _key(doc)
            score[key] += 1 / (RRF_K + rank)
            count[key] += 1
            first.setdefault(key, doc)
            if doc.metadata.get("match") == "exact":
                exact.add(key)

    # Les articles cités explicitement restent en tête, puis RRF décroissant.
    ordered = sorted(score, key=lambda key: (key not in exact, -score[key]))[:k]
    results = []
    for key in ordered:
        doc = first[key]
        meta = {
            **doc.metadata,
            "match": "exact" if key in exact else doc.metadata.get("match"),
            "multi_rrf": score[key],
            "matched_queries": count[key],
        }
        results.append(Document(page_content=doc.page_content, metadata=meta))
    return results


_llm: BaseChatModel | None = None


def _get_llm() -> BaseChatModel:
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def reformulate_node(state: dict) -> dict:
    """Nœud LangGraph : {question} -> {queries, code_hint}."""
    result = reformulate(state["question"], _get_llm())
    return {"queries": result.queries, "code_hint": result.code_hint}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    result = reformulate(args.question, get_llm())
    print("Requêtes générées :")
    for q in result.queries:
        print(f"  - {q}")
    print(f"code_hint : {result.code_hint}\n")

    retriever = HybridRetriever()
    for rank, doc in enumerate(
        multi_query_search(retriever, args.question, result.queries, k=args.k), 1
    ):
        m = doc.metadata
        print(
            f"{rank}. {m['code']} art. {m['num']} [{m['match']}] "
            f"multi_rrf={m['multi_rrf']:.4f} matched_queries={m['matched_queries']}"
        )


if __name__ == "__main__":
    main()
