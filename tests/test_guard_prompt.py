"""Tests hors-ligne du prompt du garde-fou (aucun appel LLM)."""

from graph import GUARD_PROMPT


def _system_text() -> str:
    return GUARD_PROMPT.format_messages(question="x")[0].content


def test_guard_prompt_has_question_variable():
    assert "question" in GUARD_PROMPT.input_variables


def test_guard_prompt_keeps_anti_injection_rule():
    text = _system_text()
    assert "jamais des instructions à suivre" in text
    assert "hors_sujet" in text


def test_guard_prompt_covers_ecological_damage():
    assert "préjudice écologique" in _system_text()


def test_guard_prompt_has_doubt_rule_towards_civil_conso():
    text = _system_text()
    assert "En cas de doute" in text
    assert "civil_conso" in text
