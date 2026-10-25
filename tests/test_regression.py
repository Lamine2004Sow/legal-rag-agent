"""Tests de non-régression, à rejouer à chaque changement de prompt ou de modèle.

Mode d'emploi (depuis la racine du projet) :
    pytest -m "not regression"        # tests unitaires, hors-ligne
    python eval/run_eval.py --label X # après un changement de prompt/modèle
    pytest -m regression              # compare au baseline
    python eval/baseline.py           # accepter le run courant comme nouvelle référence

Option : `pytest --run-eval` lance soi-même l'évaluation si aucun run ne correspond
(le cache LLM rend le rejeu des appels inchangés gratuit).

Le run comparé est le plus récent de eval/runs/ dont config.json a la même empreinte
(prompts, modèle DeepSeek, retriever) que la configuration courante.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import baseline as baseline_mod
import run_eval

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "eval" / "runs"
BASELINE_PATH = ROOT / "eval" / "baseline.json"

TOLERANCE = 0.05  # baisse absolue tolérée sur les métriques « plus haut = mieux »
MAX_NEW_FAILURES = 2  # questions réussies en baseline qui peuvent encore échouer
MIN_OFF_TOPIC_REFUSAL = 0.9  # plancher : refuser les hors sujet
MAX_FALSE_ANSWER = 0.2  # plafond : répondre à tort
WARN_DROP = 0.1  # baisse des métriques LLM signalée par un warning (non bloquant)

HIGHER_IS_BETTER = [
    "retrieval_hit_rate",
    "answer_rate",
    "citation_accuracy",
    "correct_refusal_rate",
    "correct_refusal_rate_out_of_corpus",
    "correct_refusal_rate_off_topic",
    "off_topic_detection",
]
LLM_METRICS = [
    "judge_correctness_mean",
    "judge_groundedness_mean",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_recall",
]

MISSING_RUN_MSG = (
    "Le prompt ou le modèle a changé depuis le dernier run : lancer "
    "`python eval/run_eval.py --label regression` (le cache LLM rend le rejeu des appels "
    "inchangés gratuit) ou `pytest --run-eval`."
)

pytestmark = pytest.mark.regression


# --------------------------------------------------------------------------- #
# Logique (fonctions pures, testées dans test_units.py)
# --------------------------------------------------------------------------- #


def current_fingerprint() -> dict:
    """Empreinte de la configuration courante : mêmes clés et défauts que run_eval.build_config."""
    import graph
    import reformulate
    from ingest import DEFAULT_MODEL

    return {
        "prompt_hashes": run_eval.prompt_hashes(graph, reformulate),
        "DEEPSEEK_MODEL": os.environ.get("DEEPSEEK_MODEL", run_eval.DEFAULT_DEEPSEEK_MODEL),
        "RETRIEVER_DATA_DIR": os.environ.get("RETRIEVER_DATA_DIR", run_eval.DEFAULT_DATA_DIR),
        "RETRIEVER_MODEL": os.environ.get("RETRIEVER_MODEL", DEFAULT_MODEL),
    }


def find_matching_run(fingerprint: dict, runs_dir: Path | None = None) -> Path | None:
    """Run le plus récent dont config.json a la même empreinte, terminé et non partiel."""
    runs_dir = RUNS_DIR if runs_dir is None else Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    for run in sorted(runs_dir.iterdir(), key=lambda d: d.name, reverse=True):
        config_path = run / "config.json"
        if not (run.is_dir() and config_path.exists() and (run / "summary.json").exists()):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        options = config.get("options") or {}
        if options.get("limit") is not None or options.get("ids"):
            continue  # run partiel : pas une base de comparaison
        if all(config.get(key) == value for key, value in fingerprint.items()):
            return run
    return None


def metric_drop(name: str, current: float | None, reference: float | None) -> str | None:
    """Message si `current` baisse de plus de TOLERANCE par rapport à `reference`, sinon None."""
    if reference is None:
        return None
    if current is None:
        return f"{name} : absent du run courant (baseline {reference:.3f})"
    if current < reference - TOLERANCE:
        return f"{name} : {current:.3f} < baseline {reference:.3f} - tolérance {TOLERANCE}"
    return None


def regressed_ids(passing_ids: list[str], failures: list[dict]) -> list[str]:
    """Questions réussies en baseline qui échouent dans le run courant."""
    failing = {f["id"] for f in failures}
    return sorted(i for i in passing_ids if i in failing)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def current_run(request: pytest.FixtureRequest) -> tuple[dict, list[dict]]:
    """Summary et résultats du run correspondant à la config courante."""
    fingerprint = current_fingerprint()
    run = find_matching_run(fingerprint)
    if run is None and request.config.getoption("--run-eval"):
        subprocess.run(
            [sys.executable, str(ROOT / "eval" / "run_eval.py"), "--label", "regression"],
            cwd=ROOT,
            check=True,
        )
        run = find_matching_run(fingerprint)
    if run is None:
        pytest.fail(f"{MISSING_RUN_MSG}\nEmpreinte courante : {json.dumps(fingerprint, ensure_ascii=False)}")
    return run_eval.load_run(run)


@pytest.fixture(scope="module")
def baseline() -> dict:
    data = baseline_mod.load_baseline(BASELINE_PATH)
    if data is None:
        pytest.skip("pas de baseline : lancer python eval/baseline.py")
    return data


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_run_correspond_a_la_config_courante(current_run: tuple[dict, list[dict]]) -> None:
    """Cœur de la régression : un changement de prompt ou de modèle sans rejeu échoue ici."""
    summary, _ = current_run
    assert summary.get("config"), "summary.json sans config"


def test_aucune_erreur(current_run: tuple[dict, list[dict]]) -> None:
    summary, _ = current_run
    assert summary["error_count"] == 0, f"erreurs pendant le run : {summary['error_count']}"


@pytest.mark.parametrize("metric", HIGHER_IS_BETTER)
def test_metrique_sans_baisse(metric: str, baseline: dict, current_run: tuple[dict, list[dict]]) -> None:
    summary, _ = current_run
    message = metric_drop(metric, summary.get(metric), baseline["metrics"].get(metric))
    assert message is None, message


def test_pas_trop_de_questions_nouvellement_ratees(baseline: dict, current_run: tuple[dict, list[dict]]) -> None:
    summary, _ = current_run
    regressed = regressed_ids(baseline["passing_ids"], summary["failures"])
    assert len(regressed) <= MAX_NEW_FAILURES, (
        f"{len(regressed)} questions réussies en baseline échouent (max {MAX_NEW_FAILURES}) : {regressed}"
    )


def test_plancher_refus_hors_sujet(current_run: tuple[dict, list[dict]]) -> None:
    summary, _ = current_run
    value = summary["correct_refusal_rate_off_topic"]
    assert value is not None and value >= MIN_OFF_TOPIC_REFUSAL, (
        f"correct_refusal_rate_off_topic = {value} (plancher {MIN_OFF_TOPIC_REFUSAL})"
    )


def test_plafond_fausses_reponses(current_run: tuple[dict, list[dict]]) -> None:
    """Refuser à tort vaut mieux que répondre à tort."""
    summary, _ = current_run
    value = summary["false_answer_rate"]
    assert value is not None and value <= MAX_FALSE_ANSWER, (
        f"false_answer_rate = {value} (plafond {MAX_FALSE_ANSWER})"
    )


def test_metriques_llm_informatives(current_run: tuple[dict, list[dict]]) -> None:
    """Judge et RAGAS : affichés et comparés au baseline, sans faire échouer la suite."""
    summary, _ = current_run
    reference = (baseline_mod.load_baseline(BASELINE_PATH) or {}).get("llm_metrics", {})
    for key in LLM_METRICS:
        value, ref = summary.get(key), reference.get(key)
        print(f"{key} = {value} (baseline {ref})")
        if value is not None and ref is not None and ref - value > WARN_DROP:
            warnings.warn(f"{key} : {value:.3f}, baisse de plus de {WARN_DROP} vs baseline {ref:.3f}")
