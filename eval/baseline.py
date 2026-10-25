"""Baseline de non-régression : métriques de référence d'un run accepté.

Usage (depuis la racine du projet) :
    python eval/baseline.py              # dernier run terminé de eval/runs/
    python eval/baseline.py <run_dir>    # un run précis

Écrit eval/baseline.json :
    source_run   nom du dossier du run
    config       config du run (prompts, modèles, retriever)
    metrics      métriques déterministes scalaires de summary.json
    llm_metrics  scores judge / RAGAS du run, s'ils existent (informatifs, non bloquants)
    passing_ids  ids des questions sans failure dans ce run

À relancer quand une baisse est voulue et validée.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_eval import latest_run, load_run

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "eval" / "baseline.json"

# Métriques non déterministes (LLM, latence) : exclues de `metrics`.
NON_DETERMINISTIC_PREFIXES = ("judge_", "ragas_", "latency_")
LLM_METRIC_KEYS = (
    "judge_correctness_mean",
    "judge_groundedness_mean",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_recall",
)


def deterministic_metrics(summary: dict) -> dict[str, Any]:
    """Valeurs scalaires (nombre ou None) de summary, hors métriques LLM et latence."""
    out = {}
    for key, value in summary.items():
        if key.startswith(NON_DETERMINISTIC_PREFIXES):
            continue
        if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            out[key] = value
    return out


def make_baseline(run_dir: Path) -> dict:
    """Construit le contenu de baseline.json à partir d'un run terminé."""
    run_dir = Path(run_dir)
    summary, results = load_run(run_dir)
    failing = {f["id"] for f in summary.get("failures", [])}
    return {
        "source_run": run_dir.name,
        "config": summary.get("config", {}),
        "metrics": deterministic_metrics(summary),
        "llm_metrics": {k: summary[k] for k in LLM_METRIC_KEYS if summary.get(k) is not None},
        "passing_ids": [r["id"] for r in results if r["id"] not in failing],
    }


def load_baseline(path: Path | None = None) -> dict | None:
    """Baseline JSON, ou None si le fichier n'existe pas."""
    path = BASELINE_PATH if path is None else Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", type=Path, help="dossier du run (défaut : dernier run)")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or latest_run()
    if run_dir is None:
        print("Aucun run terminé dans eval/runs/.", file=sys.stderr)
        return 1
    data = make_baseline(run_dir)
    BASELINE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Baseline écrite : {BASELINE_PATH} (source : {data['source_run']}, "
          f"{len(data['passing_ids'])} questions réussies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
