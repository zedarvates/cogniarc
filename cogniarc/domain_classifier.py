"""Classify ARC-AGI-3 game type from grid-change patterns observed during scout.

Pure functions (no arc_agi dependency, testable with synthetic grids).

Three primary game types:
- **navigation**: a single moving region (player) + static obstacles (walls)
- **painting**: many pixels change colour in clusters (brush/tool actions)
- **puzzle**: few targeted pixel changes (rotation, toggle, swap)
- **unknown**: insufficient or contradictory signal
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

GameType = str  # "navigation" | "painting" | "puzzle" | "unknown"

class DomainClassifier:
    """Thin wrapper around classify_game_type for import compatibility.

    domain_profiler.py instantiates DomainClassifier(env) — this
    constructor accepts any arguments but ignores them; call .classify()
    to use the pure function.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def classify(scout_results: dict, grid_changes: list) -> str:
        return classify_game_type(scout_results, grid_changes)


# Re-export helpers from transforms so domain_profiler can import them.
# These were originally in the old domain_classifier.py before the rewrite.
from .transforms import _hash_grid  # noqa: E402, F401


def _diff_grid(before: np.ndarray, after: np.ndarray) -> Tuple[int, np.ndarray]:
    """Count changed pixels and return a boolean mask.

    Returns (n_changed, changed_mask) where mask has True where grids differ.
    """
    mask = before != after
    return int(np.sum(mask)), mask


def _color_diversity(grid_before: np.ndarray, grid_after: np.ndarray) -> int:
    """Count of distinct (before→after) colour pairs among changed cells."""
    changed = np.argwhere(grid_before != grid_after)
    pairs = set()
    for r, c in changed:
        pairs.add((int(grid_before[r, c]), int(grid_after[r, c])))
    return len(pairs)


def classify_game_type(
    scout_results: Dict[int, dict],
    grid_changes: List[Tuple[int, int]],
) -> GameType:
    """Classify game type from scout-phase observations.

    Args:
        scout_results: {action: {moved, grid_changed, prop_changes}} — from
            PKM discovery.scout_results during the scout phase.
        grid_changes: [(n_pixels_changed, n_colors_changed)] per action, in
            the same order actions were tested.

    Returns:
        One of "navigation", "painting", "puzzle", "unknown".
    """
    if not grid_changes:
        return "unknown"

    n_movement = sum(
        1 for r in scout_results.values() if r.get("moved", False)
    )
    diffs = [c[0] for c in grid_changes]
    color_divs = [c[1] for c in grid_changes]
    max_diff = max(diffs) if diffs else 0
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    max_color_div = max(color_divs) if color_divs else 0

    # Detection thresholds
    MAX_NAVIGATION_DIFF = 20
    MIN_PAINTING_AVG_DIFF = 8
    MIN_PAINTING_COLORS = 3
    MAX_PUZZLE_DIFF = 6       # Smaller than navigation threshold
    MIN_PUZZLE_DIFF = 1       # Must have at least some change

    # ── No meaningful change → unknown ──
    if max_diff == 0:
        return "unknown"

    # ── navigation: player region moves, few pixels change, limited colours ──
    if n_movement >= 2 and max_diff <= MAX_NAVIGATION_DIFF and max_color_div <= 3:
        return "navigation"

    # ── painting: many pixels change, diverse colour pairs ──
    if avg_diff >= MIN_PAINTING_AVG_DIFF and max_color_div >= MIN_PAINTING_COLORS:
        return "painting"

    # ── puzzle: tiny, targeted changes ──
    if MIN_PUZZLE_DIFF <= max_diff <= MAX_PUZZLE_DIFF and max_color_div <= 2:
        return "puzzle"

    return "unknown"


def classify_from_grids(
    actions_tested: List[int],
    grids_before: List[np.ndarray],
    grids_after: List[np.ndarray],
) -> Tuple[GameType, Dict[int, dict]]:
    """Alternative entry point: pass grid pairs + actions directly (no PKM).

    Returns (game_type, scout_results_dict).
    """
    scout_results = {}
    grid_changes = []

    for action, gb, ga in zip(actions_tested, grids_before, grids_after):
        diff = int(np.sum(gb != ga))
        colors = _color_diversity(gb, ga)
        moved = diff > 0 and colors <= 2
        scout_results[action] = {
            "moved": moved,
            "grid_changed": diff > 0,
            "prop_changes": colors,
        }
        grid_changes.append((diff, colors))

    return classify_game_type(scout_results, grid_changes), scout_results
