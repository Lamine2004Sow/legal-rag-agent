"""Fusion multi-requêtes (reformulate.multi_query_search), hors-ligne : retriever réel, aucun LLM."""

from __future__ import annotations

import pytest

import reformulate as rf
from reformulate import _key

QUESTION = "Mon voisin m'a prêté une voiture et elle a été endommagée. Qui doit réparer ?"
QUERIES = [
    "réparation du dommage causé à autrui, obligation d'indemniser",
    "responsabilité du fait des choses que l'on a sous sa garde",
]


def test_resultats_dedupliques_et_au_plus_k(retriever) -> None:
    docs = rf.multi_query_search(retriever, QUESTION, QUERIES, k=8)
    keys = [_key(d) for d in docs]
    assert len(keys) == len(set(keys))
    assert 0 < len(docs) <= 8


def test_articles_exacts_en_tete(retriever) -> None:
    docs = rf.multi_query_search(retriever, "article 1240 du code civil", QUERIES, k=8)
    matches = [d.metadata["match"] for d in docs]
    assert matches[0] == "exact"
    assert matches == sorted(matches, key=lambda m: m != "exact")


def test_meilleur_resultat_de_chaque_reformulation_present(retriever) -> None:
    docs = rf.multi_query_search(retriever, QUESTION, QUERIES, k=8, per_query=1)
    keys = {_key(d) for d in docs}
    for query in QUERIES:
        top1 = retriever.search(query, k=1, n_candidates=30)[0]
        assert _key(top1) in keys, f"meilleur résultat de « {query} » absent du top 8"


def test_per_query_zero_ne_garantit_rien_mais_respecte_k(retriever) -> None:
    docs = rf.multi_query_search(retriever, QUESTION, QUERIES, k=5, per_query=0)
    assert len(docs) <= 5


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_poids_question_d_origine_accepte(retriever, weight: float) -> None:
    docs = rf.multi_query_search(retriever, QUESTION, QUERIES, k=8, original_weight=weight)
    assert len(docs) <= 8
    assert all("multi_rrf" in d.metadata for d in docs)


def test_signature_d_appel_inchangee(retriever) -> None:
    docs = rf.multi_query_search(retriever, QUESTION, QUERIES, k=8)
    assert len(docs) <= 8
    assert rf.DEFAULT_PER_QUERY == 1
    assert rf.DEFAULT_ORIGINAL_WEIGHT == 1.0
