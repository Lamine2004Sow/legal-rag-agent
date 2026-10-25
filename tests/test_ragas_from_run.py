"""--from-run sans --judge ni --ragas : pas d'appel réseau, summary.json fusionné.

Un faux run est écrit dans tmp_path. judge_answers et run_ragas sont remplacés par des
fonctions qui échouent : si l'un est appelé, le test casse.
"""

from __future__ import annotations

import json

import pytest

import run_eval


def _fake_run(run_dir):
    run_dir.mkdir(parents=True)
    results = [
        {
            "id": "civ-01",
            "category": "answerable",
            "question": "Q ?",
            "expected": [{"code": "Code civil", "num": "1240"}],
            "reference": "Réf.",
            "status": "answered",
            "answer": "Réponse [Code civil, art. 1240].",
            "citations": [{"code": "Code civil", "num": "1240"}],
            "retrieved": [{"code": "Code civil", "num": "1240"}],
            "queries": [],
            "latency_s": 1.0,
            "error": None,
        },
        {
            "id": "off-01",
            "category": "off_topic",
            "question": "Recette ?",
            "expected": [],
            "reference": "",
            "status": "refused_off_topic",
            "answer": "Hors sujet.",
            "citations": [],
            "retrieved": [],
            "queries": [],
            "latency_s": 0.5,
            "error": None,
        },
    ]
    config = {
        "label": "fake",
        "RETRIEVER_DATA_DIR": "data",
        "RETRIEVER_MODEL": "fake-model",
        "options": {"judge": False, "ragas": False, "limit": None, "ids": None},
        "n_questions": 2,
    }
    summary = run_eval.compute_metrics(results)
    summary.update(run_eval.judge_summary([]))
    summary.update({key: None for key in run_eval.RAGAS_KEYS.values()})
    summary["ragas_error"] = None
    summary["custom_key"] = "gardé"
    summary["config"] = config

    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _boom(*args, **kwargs):
    raise AssertionError("appel réseau interdit dans ce test")


def test_from_run_without_judge_or_ragas_merges_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "judge_answers", _boom)
    monkeypatch.setattr(run_eval, "run_ragas", _boom)
    run_dir = tmp_path / "20260101-000000_fake"
    before = _fake_run(run_dir)
    results_before = (run_dir / "results.jsonl").read_text(encoding="utf-8")

    assert run_eval.main(["--from-run", str(run_dir)]) == 0

    after = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert [k for k in before if k not in after] == []
    assert after["custom_key"] == "gardé"
    assert after["n_total"] == before["n_total"]
    assert "rescored_at" in after
    assert (run_dir / "results.jsonl").read_text(encoding="utf-8") == results_before
    assert not (run_dir / "judge.jsonl").exists()
    assert not (run_dir / "ragas.csv").exists()
    assert list((tmp_path).iterdir()) == [run_dir]  # aucun nouveau dossier de run


def test_from_run_does_not_need_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(run_eval, "judge_answers", _boom)
    monkeypatch.setattr(run_eval, "run_ragas", _boom)
    run_dir = tmp_path / "20260101-000000_fake"
    _fake_run(run_dir)

    assert run_eval.main(["--from-run", str(run_dir)]) == 0


@pytest.mark.parametrize("argv", [["--from-run", "x", "--judge"], ["--from-run", "x", "--ragas"]])
def test_parse_args_accepts_from_run_with_scoring_flags(argv):
    args = run_eval.parse_args(argv)
    assert args.from_run == "x"
    assert args.judge or args.ragas
