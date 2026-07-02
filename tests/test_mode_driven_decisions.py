"""Tests for mode-driven decision functions in scientist_agent.py.

These were previously hardcoded constants (3, 5) in solve_level()'s failure
handling, completely decoupled from `current_reasoning_mode` — the mode was
computed, logged, and printed but never actually changed agent behavior.
Extracted as pure functions (mode in, threshold/message out) precisely so
they're testable without a live arc_agi runtime (ScientistAgent itself can't
be instantiated in this environment — it requires a live game session).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.scientist_agent import (
    ReasoningMode,
    phase_attempts_threshold,
    phase_escalation_threshold,
    reconcile_perception_with_phase,
)


# ── phase_attempts_threshold ──────────────────────────────────────────────────
def test_default_threshold_is_base():
    assert phase_attempts_threshold(ReasoningMode.EXPLORATION) == 3
    assert phase_attempts_threshold(ReasoningMode.GOAL_INFERENCE) == 3


def test_doubt_modes_escalate_sooner():
    assert phase_attempts_threshold(ReasoningMode.SOCRATIC) == 2
    assert phase_attempts_threshold(ReasoningMode.COUNTERFACTUAL) == 2


def test_commit_modes_tolerate_more_retries():
    assert phase_attempts_threshold(ReasoningMode.PATHFINDING) == 4
    assert phase_attempts_threshold(ReasoningMode.ROTATION) == 4


def test_doubt_threshold_never_below_one():
    assert phase_attempts_threshold(ReasoningMode.SOCRATIC, base=1) == 1


# ── phase_escalation_threshold ────────────────────────────────────────────────
def test_default_escalation_is_base():
    assert phase_escalation_threshold(ReasoningMode.EXPLORATION) == 5


def test_doubt_modes_force_skip_sooner():
    assert phase_escalation_threshold(ReasoningMode.SOCRATIC) == 3
    assert phase_escalation_threshold(ReasoningMode.COUNTERFACTUAL) == 3


def test_escalation_threshold_never_below_two():
    assert phase_escalation_threshold(ReasoningMode.SOCRATIC, base=2) == 2


# ── reconcile_perception_with_phase ───────────────────────────────────────────
def test_no_perception_result_returns_none():
    assert reconcile_perception_with_phase("navigate-to-target", None, ReasoningMode.EXPLORATION) is None


def test_no_recommendation_returns_none():
    assert reconcile_perception_with_phase("navigate-to-target", {}, ReasoningMode.EXPLORATION) is None


def test_agreement_returns_none():
    perception = {"recommended_skills": ["navigate-to-target"]}
    assert reconcile_perception_with_phase("navigate-to-target", perception, ReasoningMode.EXPLORATION) is None


def test_disagreement_returns_message():
    perception = {"recommended_skills": ["rotate-to-goal"]}
    msg = reconcile_perception_with_phase("navigate-to-target", perception, ReasoningMode.SOCRATIC)
    assert msg is not None
    assert "rotate-to-goal" in msg
    assert "navigate-to-target" in msg
    assert "socratic" in msg


def test_non_list_recommendation_is_handled():
    perception = {"recommended_skills": "interact-with-object"}
    msg = reconcile_perception_with_phase("navigate-to-target", perception, ReasoningMode.EXPLORATION)
    assert "interact-with-object" in msg
