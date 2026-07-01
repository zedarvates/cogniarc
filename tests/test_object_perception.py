"""Tests for ObjectTracker — generic player/action/wall inference from grids.

All scenarios use small synthetic grids built by hand (no live arc_agi game
needed). The point being verified is that player identity, action->direction,
and wall colors are all *learned* from grid+action observations, with zero
hardcoded tags, sprite names, or action->direction assumptions — the
generalizable counterpart to scientist_agent_discovery.py's LS20-specific
tag lookups.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.object_perception import ObjectTracker, segment_regions


def grid(h=5, w=6, cells=None):
    g = np.zeros((h, w), dtype=int)
    for (r, c), v in (cells or {}).items():
        g[r, c] = v
    return g


# ── segment_regions ───────────────────────────────────────────────────────────
def test_segment_regions_finds_single_pixel_object():
    g = grid(cells={(2, 1): 5})
    regions = segment_regions(g)
    assert len(regions) == 1
    assert regions[0].color == 5
    assert regions[0].center == (2.0, 1.0)


def test_segment_regions_empty_grid_has_no_regions():
    g = grid()
    assert segment_regions(g) == []


# ── player identification ─────────────────────────────────────────────────────
def test_player_identified_from_moving_region():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    g2 = grid(cells={(2, 3): 5})
    t.observe(g0, action=1, grid_after=g1)
    t.observe(g1, action=1, grid_after=g2)
    assert t.player_color == 5


def test_static_object_is_not_mistaken_for_player():
    """A never-moving region (e.g. a goal/target) must not outrank a mover,
    however much bigger or more frequently observed it is."""
    t = ObjectTracker()
    for _ in range(5):
        g_before = grid(cells={(2, 1): 5, (4, 4): 7})  # 7 = static target
        g_after = grid(cells={(2, 2): 5, (4, 4): 7})
        t.observe(g_before, action=1, grid_after=g_after)
        g_before, g_after = g_after, grid(cells={(2, 1): 5, (4, 4): 7})
    assert t.player_color == 5


def test_no_movement_yields_no_player_candidate():
    t = ObjectTracker()
    g = grid(cells={(2, 1): 5})
    t.observe(g, action=1, grid_after=g)  # nothing moved
    assert t.player_color is None


# ── action direction is learned, not assumed ──────────────────────────────────
def test_action_direction_learned_from_observation():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    g2 = grid(cells={(2, 3): 5})
    t.observe(g0, action=1, grid_after=g1)
    t.observe(g1, action=1, grid_after=g2)
    dr, dc = t.action_direction(1)
    assert dr == 0.0
    assert dc == 1.0  # moving right (column increases) — learned, not assumed


def test_unseen_action_has_no_direction():
    t = ObjectTracker()
    assert t.action_direction(99) is None
    assert t.is_movement_action(99) is False


def test_non_movement_action_correctly_classified():
    """An action that never displaces the player-candidate region (e.g. a
    'rotate' or 'interact' action) must not be classified as a movement
    action, without ever being told which action numbers mean what."""
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t.observe(g0, action=1, grid_after=g1)  # action 1: movement
    # action 6: player stays put, some unrelated pixel changes color
    g_before = grid(cells={(2, 2): 5, (0, 0): 3})
    g_after = grid(cells={(2, 2): 5, (0, 0): 4})
    t.observe(g_before, action=6, grid_after=g_after)
    assert t.is_movement_action(1) is True
    assert t.is_movement_action(6) is False


# ── wall detection grounded in actual blocked moves ───────────────────────────
def test_wall_color_confirmed_after_repeated_blocked_moves():
    t = ObjectTracker(min_wall_votes=2)
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    g2 = grid(cells={(2, 3): 5})
    t.observe(g0, action=1, grid_after=g1)
    t.observe(g1, action=1, grid_after=g2)

    # Player at (2,3), wall color 3 immediately to the right at (2,4).
    # Movement is blocked: grid_after == grid_before.
    g_blocked = grid(cells={(2, 3): 5, (2, 4): 3})
    t.observe(g_blocked, action=1, grid_after=g_blocked)
    assert 3 not in t.wall_colors  # single observation: below min_wall_votes
    t.observe(g_blocked, action=1, grid_after=g_blocked)
    assert t.wall_colors == {3}


def test_wall_evidence_not_recorded_before_direction_is_known():
    """No bootstrap assumption: if the action's direction has never been
    established from a successful move, a blocked-looking observation must
    not produce wall evidence (we don't know it was even a movement attempt)."""
    t = ObjectTracker(min_wall_votes=1)
    g = grid(cells={(2, 1): 5, (2, 2): 3})
    t.observe(g, action=1, grid_after=g)  # no prior evidence action=1 moves anything
    assert t.wall_colors == set()


def test_no_crash_on_completely_static_game():
    t = ObjectTracker()
    g = grid(cells={(1, 1): 2})
    for a in (1, 2, 3, 4):
        t.observe(g, action=a, grid_after=g)
    assert t.player_color is None
    assert t.wall_colors == set()
    assert "0 observations" not in t.report()


# ── last_step_player_moved / current_position: the generic replacement for
# self.player.x/.y attribute-name probing (see docs/EVALUATION.md — this
# attribute-name guesslist was found broken on two real holdout games) ───────
def test_last_step_player_moved_none_before_any_observation():
    t = ObjectTracker()
    assert t.last_step_player_moved is None


def test_last_step_player_moved_true_after_a_move():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t.observe(g0, action=1, grid_after=g1)
    assert t.last_step_player_moved is True


def test_last_step_player_moved_false_when_blocked():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t.observe(g0, action=1, grid_after=g1)  # establish player + direction
    g_blocked = grid(cells={(2, 2): 5, (2, 3): 3})
    t.observe(g_blocked, action=1, grid_after=g_blocked)
    assert t.last_step_player_moved is False


def test_last_step_player_moved_none_without_player_candidate():
    """A completely static game (nothing ever moves) has no player candidate,
    so last_step_player_moved must stay None, not silently False — that
    distinction is exactly what generic movement detection needs (unknown
    vs. known-and-blocked)."""
    t = ObjectTracker()
    g = grid(cells={(1, 1): 2})
    t.observe(g, action=1, grid_after=g)
    assert t.last_step_player_moved is None


def test_current_position_none_before_player_established():
    t = ObjectTracker()
    assert t.current_position(grid(cells={(2, 1): 5})) is None


def test_current_position_tracks_player_after_move():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t.observe(g0, action=1, grid_after=g1)
    assert t.current_position(g1) == (2, 2)


def test_current_position_none_if_player_color_absent_from_grid():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t.observe(g0, action=1, grid_after=g1)  # player_color = 5 established
    empty_grid = grid()  # no colour 5 anywhere
    assert t.current_position(empty_grid) is None
