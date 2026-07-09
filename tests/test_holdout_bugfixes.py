"""Tests for the two blocking bugs from the 2026-07-05 holdout report.

B1 — SkillTree cache crash: a legacy/corrupt *_skill_tree.json (top-level
list instead of dict) crashed ScientistAgent.__init__ with
`'list' object has no attribute 'get'` because SkillTree.load_for_game() had
no error handling. sp80 could not start at all.

B2 — no target ever found on holdout games: hypothesis formation read
`player_position`/`known_positions` from ObjectTracker.get_perception_summary()
but the method never returned those keys; and even with a target, every
navigation tier was gated on `self.player` (None on all holdout games).

All tests here are pure/synthetic (no live arc_agi runtime).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.object_perception import ObjectTracker, best_action_toward
from cogniarc.skill_tree import SkillTree


def grid(h=6, w=8, cells=None):
    g = np.zeros((h, w), dtype=int)
    for (r, c), v in (cells or {}).items():
        g[r, c] = v
    return g


# ── B1: SkillTree survives legacy/corrupt cache files ─────────────────────────
def test_skill_tree_survives_legacy_list_format(tmp_path):
    """Top-level list (the exact format that crashed sp80's startup)."""
    p = tmp_path / "legacy_skill_tree.json"
    p.write_text(json.dumps([
        {"name": "move-right", "level": 1, "confidence": 0.9},
        {"name": "interact", "level": 2, "confidence": 0.8},
    ]))
    tree = SkillTree(save_path=str(p))  # must not raise
    assert "move-right" in tree.skills
    assert tree.skills["move-right"].confidence == 0.9


def test_skill_tree_survives_skills_as_list(tmp_path):
    p = tmp_path / "variant_skill_tree.json"
    p.write_text(json.dumps({
        "current_level": 3,
        "skills": [{"name": "rotate", "level": 1}],
        "level_caps": {},
    }))
    tree = SkillTree(save_path=str(p))
    assert "rotate" in tree.skills
    assert tree.current_level == 3


def test_skill_tree_survives_garbage_json(tmp_path):
    p = tmp_path / "garbage_skill_tree.json"
    p.write_text('"just a string"')
    tree = SkillTree(save_path=str(p))  # must not raise
    assert tree.skills == {}


def test_skill_tree_survives_unparseable_file(tmp_path):
    p = tmp_path / "broken_skill_tree.json"
    p.write_text("{not json at all")
    tree = SkillTree(save_path=str(p))  # must not raise
    assert tree.skills == {}


def test_skill_tree_normal_format_still_works(tmp_path):
    """Regression guard: the current dict format must load exactly as before."""
    p = tmp_path / "normal_skill_tree.json"
    p.write_text(json.dumps({
        "skills": {"jump": {"name": "jump", "level": 2, "confidence": 0.7}},
        "level_caps": {"2": ["jump"]},
        "current_level": 2,
    }))
    tree = SkillTree(save_path=str(p))
    assert tree.skills["jump"].confidence == 0.7
    assert tree.level_caps[2] == ["jump"]


# ── B2a: get_perception_summary now returns positions when given a grid ───────
def _tracked(g0, g1):
    t = ObjectTracker()
    t.observe(g0, action=1, grid_after=g1)
    t.observe(g1, action=1, grid_after=g1)  # second obs to accumulate
    return t


def test_summary_without_grid_has_no_position_keys():
    t = ObjectTracker()
    s = t.get_perception_summary()
    assert "player_position" not in s
    assert "known_positions" not in s


def test_summary_with_grid_returns_player_position_xy():
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t = _tracked(g0, g1)
    s = t.get_perception_summary(grid=g1)
    # (x, y) = (col, row): player at row 2, col 2 -> (2, 2) here, but check
    # an asymmetric case to pin the convention:
    g2 = grid(cells={(4, 1): 5})
    s2 = t.get_perception_summary(grid=g2)
    assert s2["player_position"] == (1, 4)  # x=col=1, y=row=4


def test_summary_known_positions_excludes_background():
    g0 = grid(cells={(2, 1): 5, (4, 6): 7})
    g1 = grid(cells={(2, 2): 5, (4, 6): 7})
    t = _tracked(g0, g1)
    s = t.get_perception_summary(grid=g1)
    known = s["known_positions"]
    assert 0 not in known
    assert known[7] == (6, 4)  # x=col=6, y=row=4
    assert 5 in known


def test_summary_known_positions_picks_largest_region_per_color():
    g = grid(cells={(0, 0): 7, (4, 5): 7, (4, 6): 7, (5, 5): 7, (5, 6): 7})
    g0 = grid(cells={(2, 1): 5}); g1 = grid(cells={(2, 2): 5})
    t = _tracked(g0, g1)
    s = t.get_perception_summary(grid=g)
    x, y = s["known_positions"][7]
    # The 2x2 blob (rows 4-5, cols 5-6) outweighs the lone pixel at (0,0).
    assert (x, y) != (0, 0)
    assert 4 <= y <= 6 and 4 <= x <= 7


# ── B2b: best_action_toward greedy navigation (pure planner) ──────────────────
def test_best_action_picks_learned_direction_toward_target():
    dirs = {1: (0.0, 1.0), 3: (0.0, -1.0), 2: (1.0, 0.0)}
    assert best_action_toward(dirs, (2, 2), (5, 2)) == 1  # need +cols -> right
    assert best_action_toward(dirs, (2, 4), (1, 2)) == 3  # need -cols -> left
    assert best_action_toward(dirs, (0, 2), (2, 4)) == 2  # need +rows -> down


def test_best_action_none_when_at_target():
    dirs = {1: (0.0, 1.0)}
    assert best_action_toward(dirs, (3, 5), (5, 3)) is None  # (x=5,y=3)==(r3,c5)


def test_best_action_none_when_no_direction_helps():
    dirs = {1: (0.0, 1.0)}  # can only go right
    assert best_action_toward(dirs, (2, 5), (2, 2)) is None  # target is left


def test_best_action_none_with_no_learned_directions():
    assert best_action_toward({}, (0, 0), (5, 5)) is None


def test_best_action_respects_nonstandard_mappings():
    """LS20-style 'non-standard' mapping: whatever numbers the game uses,
    only the LEARNED directions matter — no assumption that 1=up."""
    dirs = {9: (-1.0, 0.0)}  # action 9 learned to move UP (row decreases)
    assert best_action_toward(dirs, (5, 3), (3, 1)) == 9  # target is above
