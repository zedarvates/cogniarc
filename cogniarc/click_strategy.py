"""Click strategy for point-and-click games (su15, etc.).

Only actions >= 6 available. Tries clicking at various coordinates.
The arc_agi env.step() accepts data dict for action 6.
"""
from typing import List, Optional, Tuple

import numpy as np
from arcengine import GameAction


class ClickStrategy:
    """Solve strategy for click-based games."""

    def __init__(self, agent):
        self.agent = agent
        self._click_action = 6

    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Try clicking at various positions until level completes."""
        prev_lvl = self.agent.obs.levels_completed
        print("  👆 Click strategy: probing targets...")

        # Generate candidate click positions
        candidates = self._get_candidates()
        print(f"  👆 Testing {len(candidates)} positions...")

        for x, y in candidates:
            if self.agent.obs.levels_completed > prev_lvl:
                return True

            grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame else None
            self.agent.obs = self.agent.env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            self.agent.steps += 1
            grid_after = self.agent.obs.frame[0] if self.agent.obs.frame else None

            if grid_before is not None and grid_after is not None:
                diff = int(np.sum(grid_before != grid_after))
                if diff > 0:
                    print(f"  👆 Click ({x},{y}): {diff}px changed")
                    if self.agent.obs.levels_completed > prev_lvl:
                        print(f"  ✅ Level {self.agent.obs.levels_completed}!")
                        return True

        # Try action 7
        if 7 in (self.agent.obs.available_actions or []):
            print("  👆 Trying action 7...")
            for _ in range(20):
                if self.agent.obs.levels_completed > prev_lvl:
                    return True
                self.agent.obs = self.agent.env.step(getattr(GameAction, "ACTION7"))
                self.agent.steps += 1

        return self.agent.obs.levels_completed > prev_lvl

    def _get_candidates(self) -> List[Tuple[int, int]]:
        """Generate candidate click positions from sprites and grid."""
        grid = self.agent.obs.frame[0] if self.agent.obs.frame else None
        h = grid.shape[0] if grid is not None else 64
        w = grid.shape[1] if grid is not None else 64
        candidates = []

        # Priority 1: sprite positions (most likely targets)
        game = getattr(self.agent, 'game', None)
        if game:
            level = getattr(game, 'current_level', None)
            if level:
                sprites = getattr(level, '_sprites', [])
                for s in sprites:
                    candidates.append((s.x, s.y))

        # Priority 2: non-background pixels
        if grid is not None:
            mask = grid != 0
            # Sample at regular intervals
            step = max(3, min(w, h) // 12)
            for y in range(step, h, step * 2):
                for x in range(step, w, step * 2):
                    if y < h and x < w and mask[y, x]:
                        candidates.append((x, y))

        # Priority 3: evenly spaced grid (fallback)
        if not candidates:
            step = max(5, min(w, h) // 8)
            for y in range(step, h, step):
                for x in range(step, w, step):
                    candidates.append((x, y))

        return candidates
