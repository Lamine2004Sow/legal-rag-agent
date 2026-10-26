"""Évaluation du pipeline RAG sur eval/questions.jsonl.

Usage (depuis la racine du projet) :
    python eval/run_eval.py                                  # métriques déterministes seules
    python eval/run_eval.py --label v2 --judge --ragas       # + LLM-as-a-judge et RAGAS (appels DeepSeek)
    python eval/run_eval.py --limit 5 --ids civ-01,con-01    # sous-ensemble
    python eval/run_eval.py --questions autre.jsonl          # autre jeu de questions (hors non-régression)
    REFORMULATE=0 python eval/run_eval.py --label sans-reformulation
    python eval/run_eval.py --from-run eval/runs/<dir> --ragas  # rescore un run existant, sans relancer l'agent

Chaque run écrit eval/runs/<YYYYmmdd-HHMMSS>_<label>/ :
    results.jsonl  une ligne par question (réponse, citations, latence, erreur)
    config.json    label, date, commit git, modèles, hash des prompts
    summary.json   métriques (+ judge / RAGAS si demandés) et config
    judge.jsonl    verdicts LLM-as-a-judge (--judge)
    ragas.csv      scores RAGAS par question (--ragas)

Sans --judge ni --ragas, seul le pipeline (graph.answer_question) appelle le LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RUNS_DIR = ROOT / "eval" / "runs"
DEFAULT_LABEL = "run"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"  # même défaut que llm.py
DEFAULT_DATA_DIR = "data"  # même défaut que retriever.py (relatif à la racine)

ANSWERABLE = "answerable"
OUT_OF_CORPUS = "out_of_corpus"
OFF_TOPIC = "off_topic"
STATUS_ANSWERED = "answered"
STATUS_OFF_TOPIC = "refused_off_topic"
STATUS_ERROR = "error"

RAGAS_KEYS = {
    "faithfulness": "ragas_faithfulness",
    "answer_relevancy": "ragas_answer_relevancy",
    "context_recall": "ragas_context_recall",
}

log = logging.getLogger("run_eval")


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #


def _key(item: dict) -> tuple[str, str]:
    """Identifiant d'article : (code, num)."""
    return (str(item.get("code", "")).strip(), str(item.get("num", "")).strip())


def _matches(items: list[dict] | None, expected: list[dict] | None) -> bool:
    """Vrai si au moins un élément de `items` figure dans `expected` (code + num)."""
    wanted = {_key(e) for e in expected or []}
    return any(_key(item) in wanted for item in items or [])


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _complement(rate: float | None) -> float | None:
    return None if rate is None else 1 - rate


def _not_answered(record: dict) -> bool:
    return record.get("status") != STATUS_ANSWERED


def _percentile(values: list[float], p: float) -> float | None:
    """Percentile par interpolation linéaire."""
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _hash(obj: Any) -> str:
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()[:12]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _data_dir() -> Path:
    path = Path(os.environ.get("RETRIEVER_DATA_DIR", DEFAULT_DATA_DIR))
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_questions(questions: list[dict], ids: str | None, limit: int | None) -> list[dict]:
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
        known = {q["id"] for q in questions}
        unknown = [i for i in wanted if i not in known]
        if unknown:
            raise SystemExit(f"Identifiants inconnus : {', '.join(unknown)}")
        questions = [q for q in questions if q["id"] in set(wanted)]
    if limit is not None:
        questions = questions[:limit]
    return questions


# --------------------------------------------------------------------------- #
# Métriques déterministes
# --------------------------------------------------------------------------- #


def compute_metrics(results: list[dict]) -> dict:
    """Métriques sans LLM. Les questions en erreur sont exclues des taux et comptées à part."""
    errors = [r for r in results if r.get("status") == STATUS_ERROR]
    scored = [r for r in results if r.get("status") != STATUS_ERROR]
    answerable = [r for r in scored if r.get("category") == ANSWERABLE]
    answered = [r for r in answerable if r.get("status") == STATUS_ANSWERED]
    out_corpus = [r for r in scored if r.get("category") == OUT_OF_CORPUS]
    off_topic = [r for r in scored if r.get("category") == OFF_TOPIC]
    non_answerable = out_corpus + off_topic

    retrieval_hits = sum(_matches(r.get("retrieved"), r.get("expected")) for r in answerable)
    citation_hits = sum(_matches(r.get("citations"), r.get("expected")) for r in answered)
    refused_out_corpus = sum(_not_answered(r) for r in out_corpus)
    refused_off_topic = sum(_not_answered(r) for r in off_topic)
    detected_off_topic = sum(r.get("status") == STATUS_OFF_TOPIC for r in off_topic)

    answer_rate = _rate(len(answered), len(answerable))
    correct_refusal = _rate(refused_out_corpus + refused_off_topic, len(non_answerable))
    correct_refusal_oc = _rate(refused_out_corpus, len(out_corpus))
    correct_refusal_ot = _rate(refused_off_topic, len(off_topic))

    status_counts: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        status_counts[str(r.get("category"))][str(r.get("status"))] += 1

    failures = []
    for r in results:
        status, category = r.get("status"), r.get("category")
        reason = None
        if status == STATUS_ERROR:
            reason = f"erreur : {r.get('error')}"
        elif category == ANSWERABLE:
            if status != STATUS_ANSWERED:
                reason = f"answerable refusé ({status})"
            elif not _matches(r.get("citations"), r.get("expected")):
                reason = "answerable cité faux"
        elif category in (OUT_OF_CORPUS, OFF_TOPIC) and status == STATUS_ANSWERED:
            reason = "non-answerable répondu"
        if reason:
            failures.append({"id": r.get("id"), "reason": reason})

    latencies = [r["latency_s"] for r in results if isinstance(r.get("latency_s"), (int, float))]

    return {
        "n_total": len(results),
        "n_by_category": {
            ANSWERABLE: len(answerable),
            OUT_OF_CORPUS: len(out_corpus),
            OFF_TOPIC: len(off_topic),
        },
        "retrieval_hit_rate": _rate(retrieval_hits, len(answerable)),
        "answer_rate": answer_rate,
        "false_refusal_rate": _complement(answer_rate),
        "citation_accuracy": _rate(citation_hits, len(answered)),
        "correct_refusal_rate": correct_refusal,
        "false_answer_rate": _complement(correct_refusal),
        "correct_refusal_rate_out_of_corpus": correct_refusal_oc,
        "correct_refusal_rate_off_topic": correct_refusal_ot,
        "off_topic_detection": _rate(detected_off_topic, len(off_topic)),
        "status_counts": {cat: dict(counter) for cat, counter in status_counts.items()},
        "error_count": len(errors),
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "failures": failures,
    }


# --------------------------------------------------------------------------- #
# Textes des articles (judge et RAGAS)
# --------------------------------------------------------------------------- #


def load_article_texts(data_dir: Path) -> dict[tuple[str, str], str]:
    import pandas as pd

    df = pd.read_parquet(data_dir / "articles.parquet", columns=["code", "num", "contenu"])
    return {_key(row): row["contenu"] for row in df.to_dict("records")}


def citation_texts(citations: list[dict], texts: dict[tuple[str, str], str]) -> list[str]:
    """« {code}, art. {num} : {contenu} » pour chaque citation."""
    out = []
    for citation in citations or []:
        code, num = _key(citation)
        body = texts.get((code, num), "(texte introuvable dans le catalogue)")
        out.append(f"{code}, art. {num} : {body}")
    return out


# --------------------------------------------------------------------------- #
# LLM-as-a-judge
# --------------------------------------------------------------------------- #


class JudgeVerdict(BaseModel):
    correctness: int = Field(
        ge=0, le=2, description="Exactitude par rapport à la référence : 0, 1 ou 2."
    )
    groundedness: int = Field(
        ge=0, le=2, description="Ancrage dans le texte des articles cités : 0, 1 ou 2."
    )
    explanation: str = Field(description="Justification courte, en français.")


JUDGE_SYSTEM = """Tu es correcteur expert en droit français. Tu évalues la réponse d'un assistant juridique.

Deux notes, chacune entière de 0 à 2 :

correctness (exactitude par rapport à la référence) :
- 2 : la réponse répond à la question et concorde avec la référence sur l'essentiel.
- 1 : réponse partiellement correcte (élément clé manquant, imprécision, ou détail faux).
- 0 : réponse fausse, hors sujet, ou qui contredit la référence.

groundedness (ancrage dans les textes) :
- 2 : chaque affirmation de la réponse est soutenue par le TEXTE des articles cités fournis.
- 1 : au moins une affirmation n'est soutenue que partiellement, ou ne figure pas dans les articles.
- 0 : des affirmations importantes ne sont pas soutenues, ou les articles cités ne correspondent pas à la réponse.

Ne tiens pas compte de la longueur ni du style. Justifie brièvement."""

JUDGE_HUMAN = """Question :
{question}

Référence :
{reference}

Articles cités par la réponse :
{articles}

Réponse à évaluer :
{answer}"""


def judge_answers(results: list[dict], texts: dict) -> tuple[list[dict], int]:
    """Juge chaque question answerable répondue. Renvoie (verdicts, nombre d'erreurs)."""
    from langchain_core.prompts import ChatPromptTemplate

    from llm import get_llm

    prompt = ChatPromptTemplate.from_messages([("system", JUDGE_SYSTEM), ("human", JUDGE_HUMAN)])
    chain = prompt | get_llm(temperature=0.0).with_structured_output(JudgeVerdict)

    verdicts, errors = [], 0
    targets = [
        r for r in results if r.get("category") == ANSWERABLE and r.get("status") == STATUS_ANSWERED
    ]
    for r in targets:
        verdict = {"id": r["id"], "correctness": None, "groundedness": None, "explanation": None, "error": None}
        try:
            out = chain.invoke(
                {
                    "question": r["question"],
                    "reference": r.get("reference") or "",
                    "articles": "\n\n".join(citation_texts(r.get("citations"), texts))
                    or "(aucun article cité)",
                    "answer": r.get("answer") or "",
                }
            )
            verdict.update(
                correctness=out.correctness, groundedness=out.groundedness, explanation=out.explanation
            )
        except Exception as exc:  # une question ratée ne doit pas arrêter le juge
            errors += 1
            verdict["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("Juge en échec pour %s : %s", r["id"], verdict["error"])
        verdicts.append(verdict)
    return verdicts, errors


def judge_summary(verdicts: list[dict]) -> dict:
    """Moyennes normalisées sur 0-1."""
    correctness = [v["correctness"] for v in verdicts if v["correctness"] is not None]
    groundedness = [v["groundedness"] for v in verdicts if v["groundedness"] is not None]
    return {
        "judge_correctness_mean": sum(correctness) / len(correctness) / 2 if correctness else None,
        "judge_groundedness_mean": sum(groundedness) / len(groundedness) / 2 if groundedness else None,
        "judge_errors": sum(1 for v in verdicts if v["error"]),
    }


# --------------------------------------------------------------------------- #
# RAGAS
# --------------------------------------------------------------------------- #


def run_ragas(results: list[dict], texts: dict, model: str) -> tuple[dict, Any]:
    """Faithfulness, ResponseRelevancy, LLMContextRecall sur les answerable répondues.

    Renvoie (moyennes par clé summary, DataFrame par question ou None si aucune cible).
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

    from ingest import load_embeddings
    from llm import get_llm

    targets = [
        r for r in results if r.get("category") == ANSWERABLE and r.get("status") == STATUS_ANSWERED
    ]
    means: dict[str, float | None] = {key: None for key in RAGAS_KEYS.values()}
    if not targets:
        return means, None

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r.get("answer") or "",
                "retrieved_contexts": citation_texts(r.get("citations"), texts),
                "reference": r.get("reference") or "",
            }
            for r in targets
        ]
    )
    # bypass_n : DeepSeek n'accepte que n=1 ; RAGAS enchaîne alors les appels au lieu de passer n.
    llm = LangchainLLMWrapper(get_llm(), bypass_n=True)
    embeddings = LangchainEmbeddingsWrapper(load_embeddings(model))
    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextRecall(llm=llm),
    ]
    # Cache SQLite coupé pendant RAGAS : avec bypass_n, les prompts identiques partent en parallèle
    # et le cache lève IntegrityError (UNIQUE prompt/llm/idx).
    from langchain_core.globals import get_llm_cache, set_llm_cache

    previous_cache = get_llm_cache()
    set_llm_cache(None)
    try:
        result = evaluate(dataset, metrics=metrics, raise_exceptions=False, show_progress=False)
    finally:
        set_llm_cache(previous_cache)
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in targets])

    for name, key in RAGAS_KEYS.items():
        if name in df.columns:
            value = float(df[name].mean(skipna=True))
            means[key] = None if math.isnan(value) else value
    return means, df


# --------------------------------------------------------------------------- #
# Exécution du pipeline
# --------------------------------------------------------------------------- #


def _load_pipeline_modules() -> tuple[Any, Any]:
    """Import paresseux : graph.py peut ne pas exister pendant le développement."""
    import graph as graph_mod
    import reformulate as reformulate_mod

    return graph_mod, reformulate_mod


def prompt_hashes(graph_mod: Any, reformulate_mod: Any) -> dict[str, str]:
    hashes = {}
    for name in ("GUARD_PROMPT", "GENERATE_PROMPT"):
        value = getattr(graph_mod, name, None)
        if value is not None:
            hashes[f"graph.{name}"] = _hash(value)
    value = getattr(reformulate_mod, "REFORMULATE_PROMPT", None)
    if value is not None:
        hashes["reformulate.REFORMULATE_PROMPT"] = _hash(value)
    return hashes


def build_config(label: str, args: argparse.Namespace, n_questions: int, graph_mod: Any, reformulate_mod: Any) -> dict:
    from ingest import DEFAULT_MODEL

    return {
        "label": label,
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "DEEPSEEK_MODEL": os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "RETRIEVER_DATA_DIR": os.environ.get("RETRIEVER_DATA_DIR", DEFAULT_DATA_DIR),
        "RETRIEVER_MODEL": os.environ.get("RETRIEVER_MODEL", DEFAULT_MODEL),
        "REFORMULATE": os.environ.get("REFORMULATE", "1"),
        "prompt_hashes": prompt_hashes(graph_mod, reformulate_mod),
        "options": {
            "judge": args.judge,
            "ragas": args.ragas,
            "limit": args.limit,
            "ids": args.ids,
            "questions": args.questions,
        },
        "n_questions": n_questions,
    }


def run_questions(questions: list[dict], graph_mod: Any) -> list[dict]:
    graph = graph_mod.build_graph()  # construit une seule fois
    results = []
    for i, q in enumerate(questions, 1):
        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q.get("expected", []),
            "reference": q.get("reference", ""),
            "status": None,
            "answer": None,
            "citations": [],
            "retrieved": [],
            "queries": [],
            "latency_s": None,
            "error": None,
        }
        start = time.perf_counter()
        try:
            out = graph_mod.answer_question(q["question"], graph=graph)
            record.update(
                status=out.get("status"),
                answer=out.get("answer"),
                citations=list(out.get("citations") or []),
                retrieved=list(out.get("retrieved") or []),
                queries=list(out.get("queries") or []),
            )
        except Exception as exc:  # une question ratée ne doit pas arrêter le run
            record.update(status=STATUS_ERROR, error=f"{type(exc).__name__}: {exc}")
            log.error("Erreur sur %s : %s", q["id"], record["error"])
        record["latency_s"] = round(time.perf_counter() - start, 3)
        results.append(record)
        print(f"[{i}/{len(questions)}] {q['id']:<8} {str(record['status']):<22} {record['latency_s']:.1f}s", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Résumé et affichage
# --------------------------------------------------------------------------- #


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f} %"
    if kind == "sec":
        return f"{value:.2f} s"
    return str(value)


def print_summary(summary: dict) -> None:
    rows = [
        ("Retrieval hit rate (answerable)", summary["retrieval_hit_rate"], "pct"),
        ("Answer rate (answerable)", summary["answer_rate"], "pct"),
        ("False refusal rate", summary["false_refusal_rate"], "pct"),
        ("Citation accuracy (answerable répondues)", summary["citation_accuracy"], "pct"),
        ("Correct refusal rate (hors corpus + hors sujet)", summary["correct_refusal_rate"], "pct"),
        ("False answer rate", summary["false_answer_rate"], "pct"),
        ("  dont hors corpus", summary["correct_refusal_rate_out_of_corpus"], "pct"),
        ("  dont hors sujet", summary["correct_refusal_rate_off_topic"], "pct"),
        ("Off-topic detection (refused_off_topic)", summary["off_topic_detection"], "pct"),
        ("Erreurs", summary["error_count"], "num"),
        ("Latence p50", summary["latency_p50_s"], "sec"),
        ("Latence p95", summary["latency_p95_s"], "sec"),
        ("Judge correctness (moyenne)", summary.get("judge_correctness_mean"), "pct"),
        ("Judge groundedness (moyenne)", summary.get("judge_groundedness_mean"), "pct"),
        ("Judge erreurs", summary.get("judge_errors"), "num"),
        ("RAGAS faithfulness", summary.get("ragas_faithfulness"), "pct"),
        ("RAGAS answer relevancy", summary.get("ragas_answer_relevancy"), "pct"),
        ("RAGAS context recall", summary.get("ragas_context_recall"), "pct"),
    ]
    width = max(len(label) for label, _, _ in rows)
    print()
    for label, value, kind in rows:
        print(f"{label:<{width}}  {_fmt(value, kind):>10}")
    if summary.get("ragas_error"):
        print(f"RAGAS en erreur : {summary['ragas_error']}")


def print_failures(failures: list[dict]) -> None:
    print(f"\nFailures ({len(failures)}) :")
    for f in failures:
        print(f"  - {f['id']} : {f['reason']}")


# --------------------------------------------------------------------------- #
# Lecture des runs (tests de non-régression)
# --------------------------------------------------------------------------- #


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return summary, [json.loads(line) for line in lines if line.strip()]


def rescore_run(run_dir: Path, judge: bool = False, ragas: bool = False) -> dict:
    """Recalcule judge et/ou RAGAS sur un run existant, sans relancer l'agent.

    Lit results.jsonl et config.json, écrit judge.jsonl / ragas.csv dans run_dir et
    fusionne les nouvelles clés dans summary.json (les autres clés sont conservées).
    """
    run_dir = Path(run_dir)
    summary, results = load_run(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if judge or ragas:
        _check_api_key()

    update: dict[str, Any] = {"rescored_at": datetime.now().isoformat(timespec="seconds")}
    texts = None
    if judge or ragas:
        data_dir = Path(config.get("RETRIEVER_DATA_DIR", DEFAULT_DATA_DIR))
        texts = load_article_texts(data_dir if data_dir.is_absolute() else ROOT / data_dir)

    if judge:
        verdicts, _ = judge_answers(results, texts)
        _write_jsonl(run_dir / "judge.jsonl", verdicts)
        update.update(judge_summary(verdicts))
    if ragas:
        update.update({key: None for key in RAGAS_KEYS.values()})
        update["ragas_error"] = None
        try:
            means, df = run_ragas(results, texts, config["RETRIEVER_MODEL"])
            update.update(means)
            if df is not None:
                df.to_csv(run_dir / "ragas.csv", index=False)
        except Exception as exc:  # RAGAS ne doit pas faire perdre le reste
            log.exception("RAGAS en échec")
            update["ragas_error"] = f"{type(exc).__name__}: {exc}"

    summary.update(update)
    options = summary.setdefault("config", config).setdefault("options", {})
    options["judge"] = bool(options.get("judge")) or judge
    options["ragas"] = bool(options.get("ragas")) or ragas
    _write_json(run_dir / "summary.json", summary)
    return summary


def latest_run(label: str | None = None, runs_dir: Path = RUNS_DIR) -> Path | None:
    """Dernier run terminé (summary.json présent), filtré par label si fourni."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir() or "_" not in d.name or not (d / "summary.json").exists():
            continue
        _, run_label = d.name.split("_", 1)
        if label is None or run_label == label:
            candidates.append(d)
    return max(candidates, key=lambda d: d.name) if candidates else None


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Nom du run (dossier eval/runs/<date>_<label>).")
    parser.add_argument("--judge", action="store_true", help="LLM-as-a-judge sur les réponses answerable.")
    parser.add_argument("--ragas", action="store_true", help="Métriques RAGAS sur les réponses answerable.")
    parser.add_argument("--limit", type=int, default=None, help="Ne garder que les N premières questions.")
    parser.add_argument("--ids", default=None, help="Liste d'identifiants séparés par des virgules.")
    parser.add_argument(
        "--questions",
        default=None,
        help="Autre jeu de questions (JSONL, même format) ; le run n'est alors pas comparé au baseline.",
    )
    parser.add_argument(
        "--from-run",
        metavar="RUN_DIR",
        default=None,
        help="Recalcule --judge / --ragas sur un run existant (sans relancer l'agent).",
    )
    return parser.parse_args(argv)


def _check_api_key() -> None:
    import llm  # noqa: F401  (charge .env)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY absente : --judge et --ragas nécessitent le LLM.")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    if args.from_run:
        summary = rescore_run(Path(args.from_run), judge=args.judge, ragas=args.ragas)
        print_summary(summary)
        print(f"\nRun recalculé dans {args.from_run}")
        return 0

    if args.judge or args.ragas:
        _check_api_key()

    questions_path = Path(args.questions) if args.questions else QUESTIONS_PATH
    questions = select_questions(load_questions(questions_path), args.ids, args.limit)
    if not questions:
        print("Aucune question à évaluer.", file=sys.stderr)
        return 1

    import llm  # noqa: F401  (charge .env avant de lire la config)

    graph_mod, reformulate_mod = _load_pipeline_modules()
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config = build_config(args.label, args, len(questions), graph_mod, reformulate_mod)
    _write_json(run_dir / "config.json", config)

    results = run_questions(questions, graph_mod)
    _write_jsonl(run_dir / "results.jsonl", results)
    summary = compute_metrics(results)

    texts = None
    if args.judge or args.ragas:
        texts = load_article_texts(_data_dir())

    if args.judge:
        verdicts, _ = judge_answers(results, texts)
        _write_jsonl(run_dir / "judge.jsonl", verdicts)
        summary.update(judge_summary(verdicts))
    else:
        summary.update(judge_correctness_mean=None, judge_groundedness_mean=None, judge_errors=None)

    summary.update({key: None for key in RAGAS_KEYS.values()})
    summary["ragas_error"] = None
    if args.ragas:
        try:
            means, df = run_ragas(results, texts, config["RETRIEVER_MODEL"])
            summary.update(means)
            if df is not None:
                df.to_csv(run_dir / "ragas.csv", index=False)
        except Exception as exc:  # RAGAS ne doit pas faire perdre le reste du run
            log.exception("RAGAS en échec")
            summary["ragas_error"] = f"{type(exc).__name__}: {exc}"

    summary["config"] = config
    _write_json(run_dir / "summary.json", summary)

    print_summary(summary)
    print_failures(summary["failures"])
    print(f"\nRun écrit dans {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
