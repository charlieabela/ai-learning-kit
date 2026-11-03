"""Regression tests: the frozen case must keep producing the same reference values."""
import ari1101kit as kit
from ari1101kit.sessions import s02


def test_sources_present():
    assert len(kit.sources()) == 7


def test_policy_extraction():
    rules = {r["rule_id"]: r for r in kit.extract.policy_rules()}
    assert set(rules) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "E1"}
    assert rules["R2"]["approvers"]["roles"] == ["Budget Holder", "Finance Manager"]
    assert rules["R7"]["rate_eur_per_usd"] == 0.92
    assert rules["R4"]["window_days"] == 7


def test_reference_values():
    ref = s02._ref()
    assert ref["erp_shape"] == (426, 9)
    assert ref["erp_high"] == 39
    assert ref["bank_high"] == 42
    assert ref["requests"] == 37
    assert ref["noncompliant"] == 11
    assert ref["split_payments"] == 5
    assert ref["stated"] == 41


def test_every_task_has_a_question():
    for tid, t in s02.TASKS.items():
        assert t["question"] and t["title"], tid
