#!/usr/bin/env python3
"""
Shared Transform Utilities for ARC-AGI-3.

Common transform mapping logic used by both ArcAgentV3 and TransformInference.
Eliminates duplication between arc_agent.py:map_transforms() and
transform_inference.py:_map_transforms().
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Any

try:
    from arcengine import GameAction as ArcGameAction
except ImportError:
    class _FallbackGameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0
    ArcGameAction = _FallbackGameAction


class Transform:
    """A grid transformation: region + how pixels change."""
    def __init__(self, action_id: int):
        self.action_id = action_id
        self.region: Tuple[Any, Any] | None = None  # (row_slice, col_slice)
        self.mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (r,c) -> (old_val, new_val)
        self.changed_pixels: int = 0
        self.is_identity: bool = True

    def describe(self) -> str:
        if self.is_identity:
            return f"ACTION{self.action_id}: IDENTITY (no change)"
        r = self.region
        if r:
            return (f"ACTION{self.action_id}: {self.changed_pixels}px in "
                    f"rows [{r[0].start}-{r[0].stop}], cols [{r[1].start}-{r[1].stop}]")
        return f"ACTION{self.action_id}: {self.changed_pixels}px changed"


def map_transforms(env, initial_grid: np.ndarray, actions: List[int]) -> Dict[int, Transform]:
    """Map transforms from initial state for given actions."""
    transforms = {}

    for act_num in actions[:4]:
        env.reset()
        obs = env.step(getattr(ArcGameAction, f"ACTION{act_num}"))
        grid2 = obs.frame[0]

        diff = initial_grid != grid2
        coords = np.argwhere(diff)
        t = Transform(act_num)

        if len(coords) > 0:
            t.is_identity = False
            t.changed_pixels = len(coords)
            t.region = (
                slice(int(coords[:, 0].min()), int(coords[:, 0].max()) + 1),
                slice(int(coords[:, 1].min()), int(coords[:, 1].max()) + 1),
            )
            for r, c in coords:
                t.mapping[(int(r), int(c))] = (int(initial_grid[r, c]), int(grid2[r, c]))

        transforms[act_num] = t

    return transforms


def _hash_grid(grid: np.ndarray) -> str:
    """Hash a grid for state tracking."""
    import hashlib
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]