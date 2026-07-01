"""Tests for active experimentation — the disambiguate-then-observe loop.

All pure/synthetic (no live arc_agi runtime): competing hypotheses are plain
callables, so the information-gain scoring, experiment selection, belief
update, and the wall/floor worked example are all directly exercisable.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.active_experiment import (
    PredictiveHypothesis,
    outcome_distribution,
    discrimination_score,
    select_experiment,
    update_beliefs,
    surviving_hypotheses,
    build_wall_floor_experiment,
)


def _wall():
    return PredictiveHypothesis("wall", lambda a: "blocked" if a == 1 else "moved")


def _floor():
    return PredictiveHypothesis("floor", lambda a: "moved")


# ── outcome_distribution ──────────────────────────────────────────────────────
def test_distribution_splits_on_disagreement():
    dist = outcome_distribution([_wall(), _floor()], action=1)
    assert dist == {"blocked": 0.5, "moved": 0.5}


def test_distribution_collapses_on_agreement():
    dist = outcome_distribution([_wall(), _floor()], action=2)
    assert dist == {"moved": 1.0}


def test_distribution_respects_weight():
    h1 = PredictiveHypothesis("a", lambda a: "x", weight=3.0)
    h2 = PredictiveHypothesis("b", lambda a: "y", weight=1.0)
    dist = outcome_distribution([h1, h2], action=0)
    assert dist == {"x": 0.75, "y": 0.25}


def test_distribution_ignores_zero_weight():
    h1 = PredictiveHypothesis("a", lambda a: "x", weight=0.0)
    h2 = PredictiveHypothesis("b", lambda a: "y", weight=2.0)
    assert outcome_distribution([h1, h2], 0) == {"y": 1.0}


# ── discrimination_score (entropy in bits) ────────────────────────────────────
def test_disagreement_scores_one_bit():
    assert discrimination_score([_wall(), _floor()], action=1) == 1.0


def test_agreement_scores_zero_not_negative_zero():
    score = discrimination_score([_wall(), _floor()], action=2)
    assert score == 0.0
    assert math.copysign(1, score) == 1.0  # +0.0, not -0.0


def test_three_way_split_scores_above_one_bit():
    hs = [
        PredictiveHypothesis("a", lambda a: "x"),
        PredictiveHypothesis("b", lambda a: "y"),
        PredictiveHypothesis("c", lambda a: "z"),
    ]
    assert discrimination_score(hs, 0) == math.log2(3)


# ── select_experiment ─────────────────────────────────────────────────────────
def test_selects_most_discriminating_action():
    action, score = select_experiment([_wall(), _floor()], [2, 1])
    assert action == 1
    assert score == 1.0


def test_ties_break_to_lowest_action():
    # both actions equally (un)informative -> pick lowest number deterministically
    hs = [PredictiveHypothesis("a", lambda a: "same")]
    action, score = select_experiment(hs, [4, 2, 3])
    assert action == 2
    assert score == 0.0


def test_no_candidates_returns_none():
    assert select_experiment([_wall(), _floor()], []) is None


# ── update_beliefs ────────────────────────────────────────────────────────────
def test_observed_outcome_refutes_mispredictor():
    hs = [_wall(), _floor()]
    updated = update_beliefs(hs, action=1, observed_outcome="blocked")
    survivors = [h.name for h in surviving_hypotheses(updated)]
    assert survivors == ["wall"]  # floor predicted "moved", refuted


def test_update_does_not_mutate_input():
    hs = [_wall(), _floor()]
    update_beliefs(hs, 1, "blocked")
    assert all(h.weight == 1.0 for h in hs)  # originals untouched


def test_soft_refute_halves_instead_of_zeroing():
    hs = [_wall(), _floor()]
    updated = update_beliefs(hs, 1, "blocked", refute=False)
    floor = next(h for h in updated if h.name == "floor")
    assert floor.weight == 0.5


def test_matching_prediction_keeps_weight():
    hs = [_wall(), _floor()]
    updated = update_beliefs(hs, 1, "blocked")
    wall = next(h for h in updated if h.name == "wall")
    assert wall.weight == 1.0


# ── build_wall_floor_experiment (worked example) ──────────────────────────────
def test_wall_floor_only_tests_action_stepping_onto_color():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 3] = 7  # colour 7 immediately right of player at (2,2)
    dirs = {1: (0, 1), 3: (0, -1)}  # learned: 1=right, 3=left
    hyps, candidates = build_wall_floor_experiment(7, dirs, (2, 2), grid)
    assert candidates == [1]  # only "move right" steps onto colour 7
    action, score = select_experiment(hyps, candidates)
    assert action == 1
    assert score == 1.0


def test_wall_floor_no_adjacent_color_gives_no_candidates():
    grid = np.zeros((5, 5), dtype=int)  # no colour-7 cell anywhere
    dirs = {1: (0, 1), 3: (0, -1)}
    _, candidates = build_wall_floor_experiment(7, dirs, (2, 2), grid)
    assert candidates == []


def test_wall_floor_ignores_zero_direction_actions():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 3] = 7
    dirs = {1: (0, 1), 6: (0, 0)}  # action 6 = rotate (no displacement)
    _, candidates = build_wall_floor_experiment(7, dirs, (2, 2), grid)
    assert 6 not in candidates
    assert candidates == [1]


def test_wall_floor_full_loop_resolves_ambiguity():
    """End-to-end: pick the experiment, 'observe' a block, wall hypothesis wins."""
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 3] = 7
    dirs = {1: (0, 1)}
    hyps, candidates = build_wall_floor_experiment(7, dirs, (2, 2), grid)
    action, _ = select_experiment(hyps, candidates)
    updated = update_beliefs(hyps, action, observed_outcome="blocked")
    survivors = [h.name for h in surviving_hypotheses(updated)]
    assert survivors == ["wall"]
