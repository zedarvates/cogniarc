"""Painting strategy for games where actions apply brush/tool effects to the grid.

Used when domain_classifier returns "painting" — e.g. sc25, where actions 2-4
change pixel colours in clusters rather than moving a player character.

Strategy:
  1. Cycle through all paint actions in sequence (2→3→4→2→3→4...)
  2. After each action, check if level is complete
  3. After N no-change cycles, try click action 6
  4. Cycle actions to explore different effects on the grid
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain_classifier import _color_diversity


class PaintingStrategy:
    """Solve strategy for painting/interaction games (sc25, etc.)."""

    def __init__(self, agent):
        self.agent = agent
        self._paint_actions = [2, 3, 4]  # Typical paint action numbers
        self._click_action = 6           # Click/interact action
        self._action_idx = 0
        self._consecutive_no_change = 0

    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Cycle through paint actions until level completes.

        More effective than picking a single "best" action because many
        painting games need specific action sequences.

        Returns True if the level was solved.
        """
        prev_lvl = self.agent.obs.levels_completed
        print("  🎨 Paint strategy: cycling actions...")

        # Phase 1: Cycle through all paint actions
        for iteration in range(60):
            if self.agent.obs.levels_completed > prev_lvl:
                print(f"  🎨 Level {self.agent.obs.levels_completed} completed in {iteration} iterations!")
                return True

            action = self._paint_actions[self._action_idx]
            self._action_idx = (self._action_idx + 1) % len(self._paint_actions)

            grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame else None
            self.agent.step(action)
            grid_after = self.agent.obs.frame[0] if self.agent.obs.frame else None

            if grid_before is not None and grid_after is not None:
                diff = int(np.sum(grid_before != grid_after))
                if diff == 0:
                    self._consecutive_no_change += 1
                else:
                    self._consecutive_no_change = 0
                    if iteration < 5 or diff > 4:
                        colors = _color_diversity(grid_before, grid_after)
                        print(f"  🎨 A{action}: {diff}px, {colors}c")

            # No change after full cycle → try click
            if self._consecutive_no_change >= len(self._paint_actions):
                if self._click_action in (self.agent.obs.available_actions or []):
                    print(f"  🎨 All paint actions idle — trying click (A{self._click_action})")
                    for _ in range(10):
                        if self.agent.obs.levels_completed > prev_lvl:
                            return True
                        self.agent.step(self._click_action)
                break

        # Phase 2: If cycling didn't work, try longer paint sequence
        if self.agent.obs.levels_completed <= prev_lvl and iteration >= 55:
            print("  🎨 Long paint cycle exhausted")

        return self.agent.obs.levels_completed > prev_lvl
