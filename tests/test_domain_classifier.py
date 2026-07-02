"""Tests for domain_classifier — game-type detection from grid-change patterns.

All tests use pure synthetic data — no arc_agi dependency.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.domain_classifier import (
    classify_game_type,
    classify_from_grids,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _moved_result(action: int) -> dict:
    return {"moved": True, "grid_changed": True, "prop_changes": 1}


def _static_result(action: int) -> dict:
    return {"moved": False, "grid_changed": False, "prop_changes": 0}


def _paint_result(action: int) -> dict:
    return {"moved": False, "grid_changed": True, "prop_changes": 3}


# ── navigation ───────────────────────────────────────────────────────────────

def test_navigation_detected():
    """2+ actions with movement, few pixels changed, few colours."""
    results = {1: _moved_result(1), 2: _moved_result(2), 3: _static_result(3)}
    changes = [(5, 1), (8, 2), (0, 0)]
    assert classify_game_type(results, changes) == "navigation"


def test_navigation_requires_two_moving_actions():
    """Single movement action is not enough → puzzle (small change)."""
    results = {1: _moved_result(1), 2: _static_result(2)}
    changes = [(5, 1), (0, 0)]
    assert classify_game_type(results, changes) == "puzzle"


def test_navigation_rejects_high_diversity():
    """More than 3 colour pairs → painting when avg_diff >= 8."""
    results = {1: _moved_result(1), 2: _moved_result(2)}
    changes = [(10, 5), (8, 4)]
    assert classify_game_type(results, changes) == "painting"


# ── painting ─────────────────────────────────────────────────────────────────

def test_painting_detected():
    """High avg diff + high colour diversity → painting."""
    results = {2: _paint_result(2), 3: _paint_result(3)}
    changes = [(36, 4), (32, 4)]
    assert classify_game_type(results, changes) == "painting"


def test_painting_requires_large_avg_diff():
    """Small diff → not painting."""
    results = {2: _paint_result(2)}
    changes = [(5, 4)]
    assert classify_game_type(results, changes) != "painting"


def test_painting_requires_color_diversity():
    """High diff but 2 colours → neither painting nor navigation."""
    results = {2: _paint_result(2)}
    changes = [(30, 2)]
    assert classify_game_type(results, changes) not in ("painting", "navigation")


# ── puzzle ────────────────────────────────────────────────────────────────────

def test_puzzle_detected():
    """Few pixels changed, few colours → puzzle."""
    results = {6: {"moved": False, "grid_changed": True, "prop_changes": 1}}
    changes = [(3, 1)]
    assert classify_game_type(results, changes) == "puzzle"


def test_puzzle_rejects_large_changes():
    """More than MAX_PUZZLE_DIFF pixels → not puzzle."""
    results = {6: {"moved": False, "grid_changed": True, "prop_changes": 1}}
    changes = [(15, 1)]
    assert classify_game_type(results, changes) != "puzzle"


# ── edge cases ────────────────────────────────────────────────────────────────

def test_empty_input_returns_unknown():
    assert classify_game_type({}, []) == "unknown"


def test_no_changes_returns_unknown():
    results = {1: _static_result(1), 2: _static_result(2)}
    changes = [(0, 0), (0, 0)]
    assert classify_game_type(results, changes) == "unknown"


# ── classify_from_grids (alternative entry point) ────────────────────────────

def test_classify_from_grids_navigation():
    """Small moving region across frames with different actions."""
    g0 = np.zeros((5, 5), dtype=int)
    g1 = np.zeros((5, 5), dtype=int)
    g2 = np.zeros((5, 5), dtype=int)
    g0[2, 1] = 5
    g1[2, 2] = 5
    g2[2, 3] = 5

    game_type, results = classify_from_grids([1, 2], [g0, g1], [g1, g2])
    assert game_type == "navigation"
    assert 1 in results
    assert 2 in results


def test_classify_from_grids_painting():
    """Large area with 3+ colour pairs → painting."""
    g0 = np.zeros((5, 5), dtype=int)
    g1 = np.zeros((5, 5), dtype=int)
    # 9 cells change from colour 2 to 4 different colours
    g0[1:4, 1:4] = 2
    g1[1:4, 1:4] = np.array([[9, 10, 11], [10, 9, 12], [11, 12, 9]])

    game_type, results = classify_from_grids([2], [g0], [g1])
    assert game_type == "painting"


def test_classify_from_grids_no_change():
    """Identical grids → unknown."""
    g = np.zeros((3, 3), dtype=int)
    game_type, results = classify_from_grids([1], [g], [g.copy()])
    assert game_type == "unknown"
