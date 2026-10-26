"""Interface Streamlit de l'agent juridique (Code civil, Code de la consommation).

Usage :
    streamlit run app.py          # depuis la racine du projet

Nécessite DEEPSEEK_API_KEY dans `.env` et l'index dans RETRIEVER_DATA_DIR (voir README).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import streamlit as st

import graph as agent
from graph import (
    ANSWERED,
    DISCLAIMER,
    REFUSED_NO_INFO,
    REFUSED_OFF_TOPIC,
    REFUSED_OUT_OF_SCOPE,
    REFUSED_UNKNOWN_ARTICLE,
    REFUSED_UNVERIFIED,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
BASELINE_PATH = ROOT / "eval" / "baseline.json"
REPO_URL = "https://github.com/Lamine2004Sow/legal-rag-agent"

# Exemples cliquables : 4 réponses sourcées, puis 4 refus.
EXAMPLE_IDS = ["con-01", "civ-03", "civ-21", "con-05", "ooc-01", "ooc-07", "off-01", "off-08"]
EXAMPLE_LABELS = {
    "con-01": "Délai de rétractation",
    "civ-03": "Âge de la majorité",
    "civ-21": "Article 1240 (cité)",
    "con-05": "Démarchage téléphonique",
    "ooc-01": "Droit du travail",
    "ooc-07": "Article inexistant",
    "off-01": "Recette de cuisine",
    "off-08": "Injection de consigne",
}

MAX_CHARS = 500
MAX_QUESTIONS_PER_SESSION = int(os.environ.get("APP_MAX_QUESTIONS", "10"))

STATUS_LABELS = {
    ANSWERED: "Réponse vérifiée",
    REFUSED_OFF_TOPIC: "Hors sujet",
    REFUSED_OUT_OF_SCOPE: "Autre domaine du droit",
    REFUSED_UNKNOWN_ARTICLE: "Article inexistant",
    REFUSED_NO_INFO: "Information absente des deux codes",
    REFUSED_UNVERIFIED: "Réponse non vérifiable",
}
STATUS_COLORS = {
    ANSWERED: "green",
    REFUSED_OFF_TOPIC: "gray",
    REFUSED_OUT_OF_SCOPE: "orange",
    REFUSED_UNKNOWN_ARTICLE: "orange",
    REFUSED_NO_INFO: "orange",
    REFUSED_UNVERIFIED: "red",
}
# Pourquoi l'agent répond ou refuse : c'est voulu, l'explication le rend visible.
STATUS_EXPLANATIONS = {
    ANSWERED: "Chaque article cité existe et fait partie des extraits trouvés dans les codes.",
    REFUSED_OFF_TOPIC: "Ce n'est pas une question de droit : l'agent ne répond qu'aux questions juridiques.",
    REFUSED_OUT_OF_SCOPE: "La question relève d'un autre code (travail, pénal, route...) que cet agent ne couvre pas.",
    REFUSED_UNKNOWN_ARTICLE: "La question cite un article qui n'existe pas : l'agent ne l'invente pas.",
    REFUSED_NO_INFO: "Aucun article trouvé ne permet de répondre : l'agent préfère refuser plutôt que deviner.",
    REFUSED_UNVERIFIED: "La réponse proposée citait des articles qui n'ont pas pu être vérifiés : elle a été écartée.",
}


# --- Fonctions pures (testables sans Streamlit ni réseau) ---------------------------------


def load_examples(ids: list[str] = EXAMPLE_IDS, path: Path = QUESTIONS_PATH) -> list[dict]:
    """Questions d'exemple dans l'ordre de `ids`. Lève KeyError si un identifiant manque."""
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_id[row["id"]] = row
    return [by_id[i] for i in ids]


def check_question(text: str, asked: int) -> str | None:
    """Message d'erreur si la question ne peut pas être posée, None sinon."""
    text = text.strip()
    if not text:
        return "Saisissez une question."
    if len(text) > MAX_CHARS:
        return f"La question dépasse {MAX_CHARS} caractères ({len(text)} saisis)."
    if asked >= MAX_QUESTIONS_PER_SESSION:
        return "Vous avez atteint la limite de questions pour cette session."
    return None


def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    """Métriques de référence, ou None si le fichier est absent ou illisible."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fmt_pct(value: float) -> str:
    """0.975 -> « 97,5 % »."""
    return f"{value * 100:.1f} %".replace(".", ",")


BASELINE_LABELS = [
    ("retrieval_hit_rate", "Recherche : bon article trouvé"),
    ("answer_rate", "Questions auxquelles l'agent répond"),
    ("citation_accuracy", "Réponses citant un article attendu"),
    ("correct_refusal_rate", "Refus corrects (hors corpus, hors sujet)"),
    ("false_answer_rate", "Réponses à tort (hors corpus, hors sujet)"),
]


def format_metrics(baseline: dict) -> list[tuple[str, str]]:
    """Couples (libellé, valeur formatée) pour les métriques présentes dans la baseline."""
    metrics = baseline.get("metrics", {})
    return [(label, fmt_pct(metrics[key])) for key, label in BASELINE_LABELS if key in metrics]


def unique_retrieved(retrieved: list[dict]) -> list[str]:
    """« Code civil, art. 1240 » sans doublons, dans l'ordre de récupération."""
    return list(dict.fromkeys(f"{r['code']}, art. {r['num']}" for r in retrieved))


# --- Rendu Streamlit -------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_graph():
    """Agent compilé, construit une seule fois par processus (LLM et index)."""
    return agent.build_graph()


def render_result(result: dict, seconds: float) -> None:
    """Affiche le statut, la réponse, les sources et le détail du raisonnement."""
    status = result["status"]
    st.badge(STATUS_LABELS.get(status, status), color=STATUS_COLORS.get(status, "gray"))
    st.caption(STATUS_EXPLANATIONS.get(status, ""))
    st.markdown(result["answer"])

    if result["citations"]:
        st.markdown("**Sources**")
        for c in result["citations"]:
            st.markdown(f"- [{c['code']}, art. {c['num']}]({c['url']})")

    with st.expander("Comment l'agent a travaillé"):
        if result["queries"]:
            st.markdown("**Requêtes reformulées**")
            for q in result["queries"]:
                st.markdown(f"- {q}")
        articles = unique_retrieved(result["retrieved"])
        if articles:
            st.markdown("**Articles récupérés**")
            for a in articles:
                st.markdown(f"- {a}")
        if status == ANSWERED:
            st.success(
                "Toutes les citations ont été vérifiées : chaque article cité existe "
                "et fait partie des extraits récupérés."
            )

    st.caption(f"Temps de réponse : {seconds:.1f} s")


def render_sidebar(remaining: int) -> None:
    """Fiabilité mesurée, lien vers le dépôt et compteur de questions."""
    st.sidebar.header("Fiabilité mesurée")
    baseline = load_baseline()
    if baseline:
        for label, value in format_metrics(baseline):
            st.sidebar.metric(label, value)
        st.sidebar.caption(
            "Mesurées sur les 60 questions du jeu de test du projet ; elles ne garantissent "
            "pas le comportement sur des questions réelles."
        )
    else:
        st.sidebar.caption("Mesures indisponibles (eval/baseline.json absent).")

    st.sidebar.markdown(f"[Code source sur GitHub]({REPO_URL})")
    st.sidebar.metric("Questions restantes", f"{remaining} / {MAX_QUESTIONS_PER_SESSION}")


def example_buttons(examples: list[dict]) -> str | None:
    """Boutons d'exemple. Renvoie la question cliquée, ou None."""
    clicked = None
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(EXAMPLE_LABELS[ex["id"]], key=f"ex-{ex['id']}", help=ex["question"]):
            clicked = ex["question"]
    return clicked


def main() -> None:
    st.set_page_config(page_title="Agent juridique — Code civil & consommation", layout="wide")
    st.title("Agent juridique — Code civil & consommation")
    st.write(
        "Posez une question sur un contrat, un achat, un dommage ou la famille. L'agent répond "
        "à partir des articles du Code civil et du Code de la consommation, cite chaque article "
        "utilisé, et refuse quand ces textes ne permettent pas de répondre."
    )
    st.warning(
        f"{DISCLAIMER} Le corpus est limité au Code civil et au Code de la consommation, figé à "
        "avril 2025. Les questions sont envoyées au fournisseur d'IA tiers DeepSeek : "
        "ne saisissez pas de données personnelles."
    )

    st.session_state.setdefault("history", [])
    history: list[dict] = st.session_state.history
    asked = sum(1 for h in history if h["free"])
    remaining = max(0, MAX_QUESTIONS_PER_SESSION - asked)
    render_sidebar(remaining)

    try:
        with st.spinner("Chargement du modèle et de l'index…"):
            graph = get_graph()
    except Exception as exc:
        st.error(f"L'agent n'a pas pu démarrer : {exc}")
        st.markdown(
            "Pour corriger :\n"
            "- clé absente : ajoutez `DEEPSEEK_API_KEY=...` dans le fichier `.env` à la racine ;\n"
            "- index absent : `python ingest.py --model intfloat/multilingual-e5-base --out data/e5-base`."
        )
        st.stop()

    examples = load_examples()
    st.subheader("Exemples")
    st.caption("Les exemples ne comptent pas dans votre quota de questions.")
    st.markdown("**Réponses sourcées**")
    clicked = example_buttons(examples[:4])
    st.markdown("**Refus motivés**")
    clicked = example_buttons(examples[4:]) or clicked

    for h in history:
        with st.chat_message("user"):
            st.write(h["question"])
        with st.chat_message("assistant"):
            render_result(h["result"], h["seconds"])

    typed = None
    if remaining > 0:
        typed = st.chat_input("Votre question (500 caractères maximum)")
    else:
        st.info("Vous avez atteint la limite de questions pour cette session.")

    question, free = (clicked, False) if clicked else (typed, True)
    if question:
        error = None if not free else check_question(question, asked)
        if error:
            st.warning(error)
        else:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    start = time.perf_counter()
                    with st.spinner("L'agent cherche dans les codes…"):
                        result = agent.answer_question(question, graph)
                    seconds = time.perf_counter() - start
                except Exception:
                    logger.exception("Échec de l'agent.")
                    st.error("Une erreur est survenue pendant la réponse. Réessayez dans un instant.")
                else:
                    render_result(result, seconds)
                    history.append({"question": question, "free": free, "result": result, "seconds": seconds})
                    st.rerun()


if __name__ == "__main__":
    main()
