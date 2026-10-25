"""Tests unitaires et logique de non-régression, hors-ligne : aucun appel réseau ni LLM réel.

Le retriever réel est chargé une fois (fixture de session) ; les LLM sont des FakeLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

import baseline as baseline_mod
import graph as g
import reformulate as rf
import run_eval
import test_regression as reg
from graph import Citation, GeneratedAnswer, ScopeDecision
from ingest import clean_titre, normalize_num, split_long
from llm import get_llm
from reformulate import Reformulation
from retriever import find_article_refs

QUESTIONS = run_eval.load_questions()
QUESTION_VESTE = "J'ai commandé une veste sur internet et elle ne me plaît pas. Combien de temps ai-je pour changer d'avis ?"
MANUAL_QUERY = "droit de rétractation contrat conclu à distance délai quatorze jours"


# --------------------------------------------------------------------------- #
# Normalisation et découpage (ingest)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("L. 221-18", "L221-18"),
        ("R*3231-4", "R3231-4"),
        ("  L1251-33-1 ", "L1251-33-1"),
        ("liminaire", "Article liminaire"),
        ("1240", "1240"),
    ],
)
def test_normalize_num(raw: str, expected: str) -> None:
    assert normalize_num(raw) == expected


def test_clean_titre_retire_les_niveaux_repetes() -> None:
    assert clean_titre("Titre III : A Titre III : B") == "Titre III : A"


def test_split_long_respecte_max_words() -> None:
    texte = "Le débiteur doit réparer le dommage causé à autrui par sa faute. " * 40
    chunks = split_long(texte, max_words=50)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)


def test_split_long_texte_court_inchange() -> None:
    assert split_long("Un court texte.", max_words=50) == ["Un court texte."]


def test_split_long_ne_coupe_pas_une_reference() -> None:
    texte = "Voir l'art. L. 121-1 pour les pratiques trompeuses. Autre phrase utile ici. " * 20
    for chunk in split_long(texte, max_words=25):
        assert not chunk.rstrip().endswith(("art.", "L.")), chunk
        assert "121-1" not in chunk or "L. 121-1" in chunk, chunk


# --------------------------------------------------------------------------- #
# Détection des références explicites (retriever)
# --------------------------------------------------------------------------- #


def test_find_article_refs_vrais_positifs() -> None:
    assert ("Code civil", "1240") in find_article_refs("article 1240 du code civil")
    assert (None, "L221-18") in find_article_refs("Que dit l'article L. 221-18 ?")
    assert (None, "L221-18") in find_article_refs("Le délai de L221-18 ?")
    assert {n for _, n in find_article_refs("articles 1240 et 1241")} == {"1240", "1241"}


@pytest.mark.parametrize(
    "question",
    ["Mon contrat de 30 jours en 2024 est-il valable ?", "Combien font 17 fois 23 ?"],
)
def test_find_article_refs_vrais_negatifs(question: str) -> None:
    assert find_article_refs(question) == []


# --------------------------------------------------------------------------- #
# Catalogue et recherche (retriever réel)
# --------------------------------------------------------------------------- #


def test_get_article(retriever) -> None:
    art = retriever.get_article("Code civil", "1240")
    assert art is not None and art["num"] == "1240"
    assert retriever.get_article(None, "L. 221-18")["code"] == "Code de la consommation"


EXPECTED_REFS = [(ref["code"], ref["num"]) for q in QUESTIONS for ref in q.get("expected", [])]


@pytest.mark.parametrize("code, num", EXPECTED_REFS)
def test_expected_existe_dans_le_catalogue(retriever, code: str, num: str) -> None:
    assert retriever.get_article(code, num) is not None, f"[{code}, art. {num}] absent du catalogue"


@pytest.mark.parametrize("num", ["4521-99", "L999-1"])
def test_articles_pieges_hors_catalogue(retriever, num: str) -> None:
    assert retriever.get_article(None, num) is None


def test_questions_ids_uniques() -> None:
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_questions_trois_categories() -> None:
    assert {q["category"] for q in QUESTIONS} == {"answerable", "out_of_corpus", "off_topic"}


def test_recherche_article_cite_en_tete(retriever) -> None:
    docs = retriever.search("Que dit l'article 1240 du code civil ?", k=5)
    assert docs[0].metadata["num"] == "1240"
    assert docs[0].metadata["code"] == "Code civil"
    assert docs[0].metadata["match"] == "exact"


# --------------------------------------------------------------------------- #
# Vérification des citations (graph)
# --------------------------------------------------------------------------- #


def _doc(code: str, num: str) -> Document:
    return Document(page_content="texte", metadata={"code": code, "num": num})


def test_verify_citations_valide(retriever) -> None:
    errors = g.verify_citations(
        "Le délai est de quatorze jours [Code de la consommation, art. L. 221-18].",
        [{"code": "Code de la consommation", "num": "L221-18"}],
        [_doc("Code de la consommation", "L221-18")],
        retriever,
    )
    assert errors == []


def test_verify_citations_article_inexistant(retriever) -> None:
    errors = g.verify_citations(
        "Le délai est de sept jours [Code civil, art. 99999].",
        [{"code": "Code civil", "num": "99999"}],
        [_doc("Code civil", "99999")],
        retriever,
    )
    assert any("inexistant" in e for e in errors)


def test_verify_citations_article_non_recupere(retriever) -> None:
    errors = g.verify_citations(
        "Le dommage doit être réparé [Code civil, art. 1240].",
        [{"code": "Code civil", "num": "1240"}],
        [_doc("Code civil", "1241")],
        retriever,
    )
    assert any("non récupéré" in e for e in errors)


def test_verify_citations_sans_reference_inline(retriever) -> None:
    errors = g.verify_citations(
        "Réponse sans source.",
        [],
        [_doc("Code civil", "1240")],
        retriever,
    )
    assert any("aucune citation" in e for e in errors)
    assert any("aucune référence inline" in e for e in errors)


# --------------------------------------------------------------------------- #
# Graphe avec FakeLLM (scénarios de test_graph.py)
# --------------------------------------------------------------------------- #


def scope_of(value: str):
    return lambda messages: ScopeDecision(scope=value, reason="test")


def reform(messages) -> Reformulation:
    return Reformulation(queries=[MANUAL_QUERY], code_hint="Code de la consommation")


def first_extract(messages) -> tuple[str, str]:
    """Premier [code, art. num] présent dans le message humain (les extraits)."""
    import re

    m = re.search(r"\[(Code civil|Code de la consommation), art\. ([^\]]+)\]", messages[-1].content)
    assert m, "aucun extrait trouvé dans le message humain"
    return m.group(1), m.group(2)


def gen_valid(messages) -> GeneratedAnswer:
    code, num = first_extract(messages)
    return GeneratedAnswer(
        sufficient=True,
        answer=f"Vous disposez de quatorze jours pour changer d'avis [{code}, art. {num}].",
        citations=[Citation(code=code, num=num)],
    )


def gen_hallucinated(messages) -> GeneratedAnswer:
    return GeneratedAnswer(
        sufficient=True,
        answer="Le délai est de sept jours [Code civil, art. 99999].",
        citations=[Citation(code="Code civil", num="99999")],
    )


def gen_insufficient(messages) -> GeneratedAnswer:
    return GeneratedAnswer(sufficient=False, answer="", citations=[])


def base_responses(**overrides) -> dict:
    responses = {"ScopeDecision": scope_of("civil_conso"), "Reformulation": reform, "GeneratedAnswer": gen_valid}
    responses.update(overrides)
    return responses


def run_agent(llm, retriever, question: str) -> dict:
    return g.answer_question(question, graph=g.build_graph(llm=llm, retriever=retriever))


def spy_search(monkeypatch, retriever) -> dict:
    """Compte les appels à retriever.search (restauré en fin de test)."""
    counter = {"n": 0}
    original = retriever.search

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(retriever, "search", wrapped)
    return counter


def test_graphe_hors_sujet(retriever, fake_llm, monkeypatch) -> None:
    llm = fake_llm(base_responses(ScopeDecision=scope_of("hors_sujet")))
    counter = spy_search(monkeypatch, retriever)
    res = run_agent(llm, retriever, "Quelle est la recette du gâteau au chocolat ?")
    assert res["status"] == g.REFUSED_OFF_TOPIC
    assert "Reformulation" not in llm.calls and "GeneratedAnswer" not in llm.calls
    assert counter["n"] == 0


def test_graphe_autre_droit(retriever, fake_llm, monkeypatch) -> None:
    llm = fake_llm(base_responses(ScopeDecision=scope_of("autre_droit")))
    counter = spy_search(monkeypatch, retriever)
    res = run_agent(llm, retriever, "Mon employeur peut-il me licencier pendant un arrêt maladie ?")
    assert res["status"] == g.REFUSED_OUT_OF_SCOPE
    assert counter["n"] == 0 and "GeneratedAnswer" not in llm.calls


def test_graphe_article_inexistant(retriever, fake_llm) -> None:
    llm = fake_llm(base_responses())
    res = run_agent(llm, retriever, "Que prévoit l'article 4521-99 du Code civil ?")
    assert res["status"] == g.REFUSED_UNKNOWN_ARTICLE
    assert "Reformulation" not in llm.calls


def test_graphe_nominal(retriever, fake_llm) -> None:
    llm = fake_llm(base_responses())
    res = run_agent(llm, retriever, QUESTION_VESTE)
    assert res["status"] == g.ANSWERED
    assert res["citations"] and all(c["url"].startswith("https://www.legifrance.gouv.fr") for c in res["citations"])
    assert g.DISCLAIMER in res["answer"]
    assert all({"code": c["code"], "num": c["num"]} in res["retrieved"] for c in res["citations"])


def test_graphe_hallucination_deux_fois_refusee(retriever, fake_llm) -> None:
    llm = fake_llm(base_responses(GeneratedAnswer=gen_hallucinated))
    res = run_agent(llm, retriever, QUESTION_VESTE)
    assert res["status"] == g.REFUSED_UNVERIFIED
    assert len(llm.calls["GeneratedAnswer"]) == 2


def test_graphe_hallucination_puis_correcte(retriever, fake_llm) -> None:
    attempts = {"n": 0}

    def gen_then_valid(messages):
        attempts["n"] += 1
        return gen_hallucinated(messages) if attempts["n"] == 1 else gen_valid(messages)

    llm = fake_llm(base_responses(GeneratedAnswer=gen_then_valid))
    res = run_agent(llm, retriever, QUESTION_VESTE)
    assert res["status"] == g.ANSWERED
    second_call = llm.calls["GeneratedAnswer"][1][-1].content
    assert "Ta réponse précédente a été rejetée" in second_call and "99999" in second_call


def test_graphe_sufficient_faux(retriever, fake_llm) -> None:
    llm = fake_llm(base_responses(GeneratedAnswer=gen_insufficient))
    res = run_agent(llm, retriever, QUESTION_VESTE)
    assert res["status"] == g.REFUSED_NO_INFO


def test_graphe_exception_du_garde_fou_continue(retriever, fake_llm) -> None:
    def guard_raises(messages):
        raise RuntimeError("panne réseau simulée")

    llm = fake_llm(base_responses(ScopeDecision=guard_raises))
    res = run_agent(llm, retriever, QUESTION_VESTE)
    assert res["status"] == g.ANSWERED


def test_graphe_mermaid_sans_reseau(fake_llm) -> None:
    diagram = g.build_graph(llm=fake_llm({}), retriever=object()).get_graph().draw_mermaid()
    assert "guard" in diagram and "refuse" in diagram


# --------------------------------------------------------------------------- #
# Reformulation (reformulate.py)
# --------------------------------------------------------------------------- #


def test_reformulate_nominal(fake_llm) -> None:
    good = Reformulation(queries=[MANUAL_QUERY], code_hint="Code de la consommation")
    r = rf.reformulate("question", fake_llm({"Reformulation": lambda m: good}))
    assert r.queries == good.queries and r.code_hint == good.code_hint


def test_reformulate_exception_donne_liste_vide(fake_llm) -> None:
    def boom(messages):
        raise ValueError("boom")

    r = rf.reformulate("question", fake_llm({"Reformulation": boom}))
    assert r.queries == [] and r.code_hint == "inconnu"


def test_reformulate_requetes_vides(fake_llm) -> None:
    empty = Reformulation(queries=["  ", ""], code_hint="inconnu")
    r = rf.reformulate("question", fake_llm({"Reformulation": lambda m: empty}))
    assert r.queries == []


def test_reformulate_node_utilise_le_llm_injecte(fake_llm, monkeypatch) -> None:
    good = Reformulation(queries=[MANUAL_QUERY], code_hint="Code de la consommation")
    monkeypatch.setattr(rf, "_llm", fake_llm({"Reformulation": lambda m: good}))
    out = rf.reformulate_node({"question": QUESTION_VESTE})
    assert out == {"queries": good.queries, "code_hint": good.code_hint}


def test_multi_query_articles_cites_en_tete(retriever) -> None:
    docs = rf.multi_query_search(retriever, "article 1240 du code civil", [MANUAL_QUERY], k=10)
    matches = [d.metadata["match"] for d in docs]
    assert matches[0] == "exact"
    assert matches == sorted(matches, key=lambda m: m != "exact")


def test_get_llm_sans_cle_leve_erreur(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        get_llm()


# --------------------------------------------------------------------------- #
# Métriques d'évaluation (eval/run_eval.py)
# --------------------------------------------------------------------------- #


def _rec(rid, cat, status, *, expected=(), retrieved=(), citations=(), latency=None, error=None) -> dict:
    return {
        "id": rid,
        "category": cat,
        "question": f"q {rid}",
        "expected": [{"code": "Code civil", "num": n} for n in expected],
        "reference": "ref",
        "status": status,
        "answer": "a" if status == "answered" else None,
        "citations": [{"code": "Code civil", "num": n, "url": "u"} for n in citations],
        "retrieved": [{"code": "Code civil", "num": n} for n in retrieved],
        "queries": [],
        "latency_s": latency,
        "error": error,
    }


def synthetic_results() -> list[dict]:
    return [
        _rec("civ-01", "answerable", "answered", expected=["1240", "1241"], retrieved=["1240"], citations=["1240"], latency=1),
        _rec("civ-02", "answerable", "answered", expected=["1243"], retrieved=["1242"], citations=["1382"], latency=2),
        _rec("civ-03", "answerable", "refused_no_info", expected=["414"], retrieved=["414"], latency=3),
        _rec("civ-04", "answerable", "refused_unknown_article", expected=["144"], retrieved=[], latency=4),
        _rec("oos-01", "out_of_corpus", "refused_no_info", latency=5),
        _rec("oos-02", "out_of_corpus", "answered", citations=["1240"], latency=6),
        _rec("off-01", "off_topic", "refused_off_topic", latency=7),
        _rec("off-02", "off_topic", "refused_out_of_scope", latency=8),
        _rec("civ-05", "answerable", "error", expected=["1104"], error="RuntimeError: boom", latency=10),
    ]


def test_compute_metrics_taux() -> None:
    m = run_eval.compute_metrics(synthetic_results())
    assert m["retrieval_hit_rate"] == 2 / 4
    assert m["answer_rate"] == 2 / 4
    assert m["false_refusal_rate"] == 1 - 2 / 4
    assert m["citation_accuracy"] == 1 / 2
    assert m["correct_refusal_rate"] == 3 / 4
    assert m["false_answer_rate"] == 1 - 3 / 4
    assert m["correct_refusal_rate_out_of_corpus"] == 1 / 2
    assert m["correct_refusal_rate_off_topic"] == 2 / 2
    assert m["off_topic_detection"] == 1 / 2
    assert m["error_count"] == 1
    assert m["n_total"] == 9


def test_compute_metrics_statuts_et_latence() -> None:
    m = run_eval.compute_metrics(synthetic_results())
    assert m["status_counts"]["answerable"] == {
        "answered": 2,
        "refused_no_info": 1,
        "refused_unknown_article": 1,
        "error": 1,
    }
    assert m["latency_p50_s"] == 5
    assert math.isclose(m["latency_p95_s"], 9.2)


def test_compute_metrics_echecs() -> None:
    failures = {f["id"]: f["reason"] for f in run_eval.compute_metrics(synthetic_results())["failures"]}
    assert set(failures) == {"civ-02", "civ-03", "civ-04", "oos-02", "civ-05"}
    assert failures["civ-02"] == "answerable cité faux"
    assert failures["civ-03"] == "answerable refusé (refused_no_info)"
    assert failures["oos-02"] == "non-answerable répondu"
    assert failures["civ-05"].startswith("erreur")


def test_compute_metrics_vide() -> None:
    m = run_eval.compute_metrics([])
    assert m["retrieval_hit_rate"] is None
    assert m["latency_p50_s"] is None
    assert m["failures"] == []


def test_compute_metrics_uniquement_erreurs() -> None:
    m = run_eval.compute_metrics([_rec("civ-09", "answerable", "error", error="x")])
    assert m["answer_rate"] is None
    assert m["error_count"] == 1


def test_citation_match_exige_le_meme_code() -> None:
    expected = [{"code": "Code civil", "num": "1240"}]
    assert run_eval._matches([{"code": "Code civil", "num": "1240"}], expected)
    assert not run_eval._matches([{"code": "Code de la consommation", "num": "1240"}], expected)


def test_load_et_latest_run(tmp_path: Path) -> None:
    def make(name: str, summary: dict | None) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "results.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in synthetic_results()) + "\n", encoding="utf-8"
        )
        if summary is not None:
            (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return d

    make("20260101-100000_base", {"answer_rate": 0.1})
    make("20260102-100000_other", {"answer_rate": 0.2})
    make("20260103-100000_base", {"answer_rate": 0.3, "config": {"label": "base"}})
    make("20260104-100000_base", None)  # run incomplet : ignoré
    make("20260105-100000_wip", None)
    (tmp_path / "fichier.txt").write_text("pas un run", encoding="utf-8")

    assert run_eval.latest_run(runs_dir=tmp_path) == tmp_path / "20260103-100000_base"
    assert run_eval.latest_run("other", runs_dir=tmp_path) == tmp_path / "20260102-100000_other"
    assert run_eval.latest_run("wip", runs_dir=tmp_path) is None
    assert run_eval.latest_run(runs_dir=tmp_path / "inexistant") is None

    summary, results = run_eval.load_run(tmp_path / "20260103-100000_base")
    assert summary == {"answer_rate": 0.3, "config": {"label": "base"}}
    assert len(results) == 9 and results[0]["id"] == "civ-01"


def test_judge_summary_normalisation() -> None:
    verdicts = [
        {"id": "a", "correctness": 2, "groundedness": 1, "explanation": "", "error": None},
        {"id": "b", "correctness": 0, "groundedness": None, "explanation": None, "error": "x"},
    ]
    s = run_eval.judge_summary(verdicts)
    assert s["judge_correctness_mean"] == 1 / 2
    assert s["judge_groundedness_mean"] == 1 / 2
    assert s["judge_errors"] == 1


def test_citation_texts_format() -> None:
    texts = {("Code civil", "1240"): "Tout fait quelconque..."}
    assert run_eval.citation_texts([{"code": "Code civil", "num": "1240"}], texts) == [
        "Code civil, art. 1240 : Tout fait quelconque..."
    ]


def test_run_eval_help() -> None:
    out = subprocess.run(
        [sys.executable, str(run_eval.__file__), "--help"],
        cwd=run_eval.ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    for flag in ("--label", "--judge", "--ragas", "--limit", "--ids"):
        assert flag in out.stdout


# --------------------------------------------------------------------------- #
# Baseline et logique de régression, sur de faux runs (jamais eval/runs/)
# --------------------------------------------------------------------------- #

FP = {
    "prompt_hashes": {"graph.GUARD_PROMPT": "a1", "graph.GENERATE_PROMPT": "b2", "reformulate.REFORMULATE_PROMPT": "c3"},
    "DEEPSEEK_MODEL": "deepseek-chat",
    "RETRIEVER_DATA_DIR": "data",
    "RETRIEVER_MODEL": "intfloat/multilingual-e5-small",
}
NO_OPTIONS = {"judge": False, "ragas": False, "limit": None, "ids": None}


def write_run(runs_dir: Path, name: str, fingerprint: dict, *, options: dict | None = None, summary: bool = True) -> Path:
    d = runs_dir / name
    d.mkdir(parents=True)
    config = {**fingerprint, "options": options or NO_OPTIONS}
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if summary:
        (d / "summary.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    return d


def test_run_correspondant_retenu(tmp_path: Path) -> None:
    run = write_run(tmp_path, "20260101-100000_x", FP)
    assert reg.find_matching_run(FP, runs_dir=tmp_path) == run


def test_prompt_modifie_aucun_run_correspondant(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101-100000_x", FP)
    modified = {**FP, "prompt_hashes": {**FP["prompt_hashes"], "graph.GENERATE_PROMPT": "zz"}}
    assert reg.find_matching_run(modified, runs_dir=tmp_path) is None


def test_modele_modifie_aucun_run_correspondant(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101-100000_x", FP)
    assert reg.find_matching_run({**FP, "RETRIEVER_MODEL": "autre"}, runs_dir=tmp_path) is None


def test_plus_recent_correspondant_gagne(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101-100000_a", FP)
    recent = write_run(tmp_path, "20260102-100000_b", FP)
    write_run(tmp_path, "20260103-100000_c", {**FP, "RETRIEVER_MODEL": "autre"})  # plus récent, autre config
    assert reg.find_matching_run(FP, runs_dir=tmp_path) == recent


def test_run_sans_summary_ignore(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101-100000_x", FP, summary=False)
    assert reg.find_matching_run(FP, runs_dir=tmp_path) is None


def test_run_partiel_ignore(tmp_path: Path) -> None:
    write_run(tmp_path, "20260101-100000_x", FP, options={**NO_OPTIONS, "limit": 5})
    assert reg.find_matching_run(FP, runs_dir=tmp_path) is None


def test_repertoire_des_runs_monkeypatchable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(reg, "RUNS_DIR", tmp_path)
    run = write_run(tmp_path, "20260101-100000_x", FP)
    assert reg.find_matching_run(FP) == run


def test_empreinte_courante_alignee_sur_build_config() -> None:
    import graph
    import reformulate

    args = argparse.Namespace(judge=False, ragas=False, limit=None, ids=None)
    cfg = run_eval.build_config("test", args, 1, graph, reformulate)
    fingerprint = reg.current_fingerprint()
    assert {k: cfg[k] for k in fingerprint} == fingerprint


def test_comparaison_identique_sans_baisse() -> None:
    values = {k: 0.8 for k in reg.HIGHER_IS_BETTER}
    assert all(reg.metric_drop(k, v, v) is None for k, v in values.items())


def test_baisse_au_dela_de_la_tolerance_detectee() -> None:
    message = reg.metric_drop("answer_rate", 0.70, 0.80)
    assert message is not None and "0.700" in message and "0.800" in message


def test_baisse_dans_la_tolerance_acceptee() -> None:
    assert reg.metric_drop("answer_rate", 0.77, 0.80) is None


def test_metrique_none_ignoree_si_baseline_none() -> None:
    assert reg.metric_drop("x", None, None) is None
    assert reg.metric_drop("x", 0.5, None) is None


def test_metrique_disparue_detectee() -> None:
    assert reg.metric_drop("x", None, 0.5) is not None


def test_questions_regressees_detectees() -> None:
    failures = [{"id": "civ-02", "reason": "r"}, {"id": "civ-03", "reason": "r"}, {"id": "civ-09", "reason": "r"}]
    passing = ["civ-01", "civ-02", "civ-03", "civ-04"]
    assert reg.regressed_ids(passing, failures) == ["civ-02", "civ-03"]
    assert len(reg.regressed_ids(passing, failures[:1])) <= reg.MAX_NEW_FAILURES


def test_make_baseline_sur_un_faux_run(tmp_path: Path) -> None:
    results = synthetic_results()
    run_dir = tmp_path / "20260101-100000_x"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    summary = run_eval.compute_metrics(results)
    summary["config"] = {**FP, "options": NO_OPTIONS}
    summary["judge_correctness_mean"] = 0.9
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    data = baseline_mod.make_baseline(run_dir)
    failing = {f["id"] for f in summary["failures"]}
    assert data["source_run"] == run_dir.name
    assert data["passing_ids"] == [r["id"] for r in results if r["id"] not in failing]
    assert data["metrics"]["retrieval_hit_rate"] == summary["retrieval_hit_rate"]
    assert "latency_p50_s" not in data["metrics"] and "failures" not in data["metrics"]
    assert data["llm_metrics"] == {"judge_correctness_mean": 0.9}
    assert data["config"] == summary["config"]
