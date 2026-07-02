"""Painting strategy for games where actions apply brush/tool effects to the grid.

Uses Nano-LLM tier to guide action selection — reads the grid state and proposes
the next action, falling back to blind cycling when nano-LLM is unavailable.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain_classifier import _color_diversity


def _grid_to_text(grid: np.ndarray) -> str:
    """Compact grid representation for LLM consumption."""
    h, w = grid.shape
    # Show unique colours and their rough positions
    unique = np.unique(grid)
    color_positions = {}
    for c in unique:
        ys, xs = np.where(grid == c)
        if len(ys) > 0:
            color_positions[int(c)] = {
                "count": int(len(ys)),
                "center": (int(np.mean(ys)), int(np.mean(xs))),
                "bounds": (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            }
    return f"Grid {h}×{w}, colours: {sorted(color_positions.keys())}, regions: {len(color_positions)}"


class PaintingStrategy:
    """Solve strategy for painting/interaction games (sc25, etc.)."""

    def __init__(self, agent):
        self.agent = agent
        self._paint_actions = [2, 3, 4]
        self._click_action = 6
        self._action_idx = 0
        self._consecutive_no_change = 0

    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Solve painting level using Nano-LLM proposals when available."""
        prev_lvl = self.agent.obs.levels_completed
        print("  🎨 Paint strategy + nano-LLM...")

        # Phase 1: Discover action effects via nano-LLM or blind cycling
        for iteration in range(60):
            if self.agent.obs.levels_completed > prev_lvl:
                print(f"  🎨 Level {self.agent.obs.levels_completed} in {iteration} iterations!")
                return True

            # Try nano-LLM action proposal every 5 iterations
            action = None
            if iteration % 5 == 0:
                try:
                    action = self.agent._nano_propose_action()
                    if action is not None:
                        print(f"  🤖 Nano suggests action {action}")
                except Exception:
                    pass

            if action is None:
                # Fallback: cycle through paint actions
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
                    if iteration < 8 or diff > 4:
                        colors = _color_diversity(grid_before, grid_after)
                        print(f"  🎨 A{action}: {diff}px, {colors}c")

            if self._consecutive_no_change >= 3:
                if self._click_action in (self.agent.obs.available_actions or []):
                    print(f"  🎨 Trying click A{self._click_action}")
                    for _ in range(10):
                        if self.agent.obs.levels_completed > prev_lvl:
                            return True
                        self.agent.step(self._click_action)
                break

        return self.agent.obs.levels_completed > prev_lvl
