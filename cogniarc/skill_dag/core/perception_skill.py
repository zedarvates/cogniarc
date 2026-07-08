"""Perception Skill — State observation, grid hashing, object detection for ARC-AGI-3."""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class GridState:
    """Observed grid state with metadata."""
    grid: np.ndarray
    hash: str
    objects: List[Dict[str, Any]]
    player_pos: Optional[Tuple[int, int]] = None
    available_actions: Optional[List[int]] = None


class PerceptionSkill:
    """
    Core perception skill for ARC-AGI-3 environments.
    
    Extracts: grid state, object list, player position, available actions.
    Provides: fast state hashing for cycle detection.
    """

    def __init__(self):
        self.last_state: Optional[GridState] = None
        self.state_history: List[str] = []

    def observe(self, env, obs) -> GridState:
        """Full observation pipeline."""
        if obs and hasattr(obs, 'frame') and obs.frame and len(obs.frame) > 0:
            grid = obs.frame[0]
        else:
            grid = np.zeros((64, 64), dtype=np.int8)
        state_hash = self._hash_grid(grid)
        objects = self._extract_objects(grid)
        player_pos = self._find_player(objects, grid)
        actions = list(getattr(obs, 'available_actions', []) or [])

        state = GridState(
            grid=grid,
            hash=state_hash,
            objects=objects,
            player_pos=player_pos,
            available_actions=actions,
        )

        self.last_state = state
        self.state_history.append(state_hash)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]

        return state

    def _hash_grid(self, grid: np.ndarray) -> str:
        """Fast SHA256 hash of grid (16-char prefix)."""
        return hashlib.sha256(grid.tobytes()).hexdigest()[:16]

    def _extract_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Extract connected components as objects."""
        objects = []
        visited = np.zeros(grid.shape, dtype=bool)

        for color in np.unique(grid):
            if color == 0:
                continue
            mask = (grid == color) & ~visited
            if not mask.any():
                continue

            from collections import deque
            while mask.any():
                start = tuple(np.argwhere(mask)[0])
                component = []
                q = deque([start])
                mask[start] = False
                visited[start] = True

                while q:
                    r, c = q.popleft()
                    component.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < 64 and 0 <= nc < 64:
                            if grid[nr, nc] == color and not visited[nr, nc]:
                                visited[nr, nc] = True
                                mask[nr, nc] = False
                                q.append((nr, nc))

                if len(component) >= 1:
                    coords = np.array(component)
                    objects.append({
                        "color": int(color),
                        "center": tuple(coords.mean(axis=0).astype(int)),
                        "size": len(component),
                        "bbox": (tuple(coords.min(axis=0)), tuple(coords.max(axis=0))),
                        "coords": coords,
                    })
        return objects

    def _find_player(self, objects: List[Dict], grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Heuristic: player is unique colored object that can move."""
        # For now return None - requires step comparison
        # Actual player detection happens in physics/execution skills
        return None

    def detect_change(self, env, obs) -> Dict[str, Any]:
        """Compare current observation with last state."""
        if not self.last_state:
            return {"changed": True, "reason": "first_observation"}

        new_state = self.observe(env, obs)
        changed_pixels = int((self.last_state.grid != new_state.grid).sum())

        return {
            "changed": changed_pixels > 0,
            "changed_pixels": changed_pixels,
            "new_hash": new_state.hash,
            "old_hash": self.last_state.hash,
        }

    def get_novelty(self, state_hash: str) -> float:
        """Novelty score: 1.0 = never seen, 0.0 = seen recently."""
        if state_hash not in self.state_history:
            return 1.0
        # Recent repetition penalty
        recent = self.state_history[-20:]
        count = recent.count(state_hash)
        return max(0.0, 1.0 - count * 0.1)

    def reset(self):
        self.last_state = None
        self.state_history.clear()


# Factory for SkillDAG registry
def create_perception_skill() -> PerceptionSkill:
    return PerceptionSkill()