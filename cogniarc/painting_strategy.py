"""Painting strategy for games where actions apply brush/tool effects to the grid.

Used when domain_classifier returns "painting" — e.g. sc25, where actions 2-4
change pixel colours in clusters rather than moving a player character.

Strategy:
  1. Discover what each action does to grid colours (action effects)
  2. Apply the most impactful action in a loop
  3. Track which colour transitions are "progress" vs "noise"
  4. Try action 6 (click) with coordinates if available
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain_classifier import classify_game_type, _color_diversity


class PaintingStrategy:
    """Solve strategy for painting/interaction games (sc25, etc.)."""

    def __init__(self, agent):
        self.agent = agent
        self._paint_actions = [2, 3, 4]  # Typical paint action numbers
        self._click_action = 6           # Click/interact action
        self._best_action = None
        self._consecutive_no_change = 0

    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Try painting actions in different patterns until level completes.

        Returns True if the level was solved.
        """
        prev_lvl = self.agent.obs.levels_completed
        print("  🎨 Paint strategy: discovering action effects...")

        # Phase 1: Discover action effects
        effects = self._discover_action_effects()
        print(f"  🎨 Action effects: {len(effects)} actions with effects")

        # Phase 2: Find the best action
        self._best_action = self._pick_best_action(effects)
        if self._best_action is None:
            print("  🎨 No effective action found — trying them all")
            self._best_action = self._paint_actions[0]

        # Phase 3: Apply best action repeatedly
        for iteration in range(50):
            if self.agent.obs.levels_completed > prev_lvl:
                print(f"  🎨 Level {self.agent.obs.levels_completed} completed in {iteration} iterations!")
                return True

            grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame else None
            self.agent.step(self._best_action)
            grid_after = self.agent.obs.frame[0] if self.agent.obs.frame else None

            if grid_before is not None and grid_after is not None:
                diff = int(np.sum(grid_before != grid_after))
                if diff == 0:
                    self._consecutive_no_change += 1
                    if self._consecutive_no_change >= 3:
                        print(f"  🎨 Action {self._best_action} stopped changing grid — switching")
                        # Try another action
                        self._best_action = self._cycle_action(effects)
                        self._consecutive_no_change = 0
                else:
                    self._consecutive_no_change = 0
                    colors = _color_diversity(grid_before, grid_after)
                    print(f"  🎨 Action {self._best_action}: {diff}px, {colors} colours")

        # Phase 4: Try click actions if paint didn't work
        if self._click_action in (self.agent.obs.available_actions or []):
            print(f"  🎨 Paint didn't solve — trying click (action {self._click_action})")
            for _ in range(20):
                if self.agent.obs.levels_completed > prev_lvl:
                    return True
                self.agent.step(self._click_action)

        return self.agent.obs.levels_completed > prev_lvl

    def _discover_action_effects(self) -> Dict[int, Dict]:
        """Discover what each painting action does to grid colours.

        Returns {action_num: {n_changed, color_pairs}} for actions
        that actually change the grid.
        """
        effects = {}
        for action in self._paint_actions:
            if action in (self.agent.obs.available_actions or []):
                grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame else None
                if grid_before is None:
                    continue
                self.agent.step(action)
                grid_after = self.agent.obs.frame[0] if self.agent.obs.frame else None
                if grid_after is None:
                    continue
                changed = np.argwhere(grid_before != grid_after)
                if len(changed) == 0:
                    continue
                color_pairs = set()
                for r, c in changed:
                    color_pairs.add((int(grid_before[r, c]), int(grid_after[r, c])))
                effects[action] = {
                    "n_changed": len(changed),
                    "color_pairs": color_pairs,
                    "n_colors": len(color_pairs),
                }
        return effects

    def _pick_best_action(self, effects: Dict[int, Dict]) -> Optional[int]:
        """Pick the action with the most impactful effect.

        Prefers actions that change many pixels with diverse colours
        (sign of a painting tool).
        """
        if not effects:
            return None
        best = max(
            effects,
            key=lambda a: effects[a]["n_changed"] * effects[a]["n_colors"]
        )
        return best

    def _cycle_action(self, effects: Dict[int, Dict]) -> int:
        """Cycle to the next effective action."""
        available = sorted(effects.keys()) if effects else self._paint_actions
        if self._best_action in available:
            idx = available.index(self._best_action)
            return available[(idx + 1) % len(available)]
        return available[0]
