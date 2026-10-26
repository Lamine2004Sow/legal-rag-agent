"""Tests de l'interface Streamlit, hors-ligne : fonctions pures et AppTest avec un faux graphe.

Le graphe réel n'est jamais construit : `graph.build_graph` est remplacé par monkeypatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import app
import graph as agent_graph
from graph import DISCLAIMER

ROOT = Path(__file__).resolve().parent.parent
LEGIFRANCE_URL = "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000032226842"


def test_examples_found_for_every_id():
    examples = app.load_examples()
    assert [e["id"] for e in examples] == app.EXAMPLE_IDS
    assert len(examples) == 8
    assert all(e["question"] for e in examples)
    assert set(app.EXAMPLE_LABELS) == set(app.EXAMPLE_IDS)


def test_check_question_length_and_quota(monkeypatch):
    monkeypatch.setattr(app, "MAX_QUESTIONS_PER_SESSION", 3)
    assert app.check_question("Une question ?", asked=0) is None
    assert app.check_question("a" * app.MAX_CHARS, asked=0) is None
    assert "dépasse" in app.check_question("a" * (app.MAX_CHARS + 1), asked=0)
    assert app.check_question("   ", asked=0) is not None
    assert "limite" in app.check_question("Une question ?", asked=3)


def test_every_graph_status_has_labels():
    statuses = [
        v
        for k, v in vars(agent_graph).items()
        if k == "ANSWERED" or k.startswith("REFUSED_")
    ]
    assert len(statuses) == 6
    for status in statuses:
        assert app.STATUS_LABELS.get(status), status
        assert app.STATUS_EXPLANATIONS.get(status), status
        assert app.STATUS_COLORS.get(status), status


def test_format_metrics_and_missing_baseline(tmp_path):
    baseline = {"metrics": {"answer_rate": 0.975, "retrieval_hit_rate": 0.9}}
    assert app.format_metrics(baseline) == [
        ("Recherche : bon article trouvé", "90,0 %"),
        ("Questions auxquelles l'agent répond", "97,5 %"),
    ]
    assert app.load_baseline(tmp_path / "absent.json") is None


def test_unique_retrieved_keeps_order():
    retrieved = [
        {"code": "Code civil", "num": "1240"},
        {"code": "Code civil", "num": "1241"},
        {"code": "Code civil", "num": "1240"},
    ]
    assert app.unique_retrieved(retrieved) == ["Code civil, art. 1240", "Code civil, art. 1241"]


class FakeGraph:
    """Faux graphe : une réponse « answered » avec une citation, sans appel réseau."""

    def invoke(self, state: dict) -> dict:
        return {
            **state,
            "status": agent_graph.ANSWERED,
            "answer": "Quatorze jours pour exercer votre droit de rétractation "
            "[Code de la consommation, art. L221-18].\n\n" + DISCLAIMER,
            "citations": [
                {"code": "Code de la consommation", "num": "L221-18", "url": LEGIFRANCE_URL}
            ],
            "docs": [],
            "queries": ["délai de rétractation"],
        }


@pytest.fixture
def fake_graph(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(agent_graph, "build_graph", lambda *args, **kwargs: FakeGraph())
    yield
    st.cache_resource.clear()


def test_app_example_click_shows_verified_answer(fake_graph):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception

    button = next(b for b in at.button if b.key == "ex-con-01")
    button.click().run()
    assert not at.exception

    # Le badge est rendu en markdown : `:green-badge[...]`.
    texts = " ".join(m.value for m in at.markdown)
    assert ":green-badge[Réponse vérifiée]" in texts
    assert f"]({LEGIFRANCE_URL})" in texts
