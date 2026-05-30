#!/usr/bin/env python3
"""
Transform-Based Goal Inference for ARC-AGI-3.

Many (most?) ARC-AGI-3 games are TRANSFORM-BASED:
Actions apply transformations to regions of the grid — they don't move an agent.
The goal is to reach a target grid pattern.

This module:
  1. Maps each action to its TRANSFORM (region + operation)
  2. Identifies OUTPUT AREAS (where the "lock" is)
  3. Infers the TARGET PATTERN (what the output should look like)
  4. Plans a transform sequence to reach the target

Usage:
    from transform_inference import TransformInference
    ti = TransformInference(env)
    plan = ti.solve()
"""

from __future__ import annotations

import json
import numpy as np
import hashlib
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


class Transform:
    """A grid transformation: region + how pixels change."""
    def __init__(self, action_id: int):
        self.action_id = action_id
        self.region: Optional[Tuple[slice, slice]] = None  # (row_slice, col_slice)
        self.mapping: Dict[Tuple[int,int], Tuple[int,int]] = {}  # (r,c) → (old_val, new_val)
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


class TransformInference:
    """Infer and solve transform-based ARC-AGI-3 games."""

    def __init__(self, env, max_steps: int = 200):
        self.env = env
        self.max_steps = max_steps
        self.transforms: Dict[int, Transform] = {}
        self.initial_grid: Optional[np.ndarray] = None
        self.state_space: Dict[str, np.ndarray] = {}  # hash → grid
        self.transition_graph: Dict[str, Dict[int, str]] = defaultdict(dict)
        self.output_region: Optional[Tuple[slice, slice]] = None
        self.target_pattern: Optional[np.ndarray] = None

    def solve(self) -> Optional[List[int]]:
        """Full solve pipeline."""
        obs = self.env.reset()
        self.initial_grid = obs.frame[0].copy()

        # Phase 1: Map transforms for each action
        self._map_transforms(obs)

        # Phase 2: Explore state space via BFS
        self._explore_state_space(obs)

        # Phase 3: Identify output region and target
        self._infer_goal(obs)

        # Phase 4: Find path to goal
        return self._plan()

    def _map_transforms(self, obs):
        """For each available action, record its transform."""
        initial = self.initial_grid
        for act_num in (obs.available_actions or []):
            if act_num not in [1, 2, 3, 4]:
                continue
            self.env.reset()
            result = self.env.step(getattr(GameAction, f"ACTION{act_num}"))
            grid2 = result.frame[0]

            diff = initial != grid2
            coords = np.argwhere(diff)
            t = Transform(act_num)

            if len(coords) == 0:
                t.is_identity = True
            else:
                t.is_identity = False
                t.changed_pixels = len(coords)
                t.region = (
                    slice(int(coords[:,0].min()), int(coords[:,0].max()) + 1),
                    slice(int(coords[:,1].min()), int(coords[:,1].max()) + 1),
                )
                for r, c in coords:
                    t.mapping[(int(r), int(c))] = (int(initial[r,c]), int(grid2[r,c]))

            self.transforms[act_num] = t

    def _explore_state_space(self, obs):
        """BFS to discover reachable states."""
        self.env.reset()
        initial_hash = _hash_grid(self.initial_grid)
        self.state_space[initial_hash] = self.initial_grid.copy()

        queue = deque([(initial_hash, [])])
        visited = {initial_hash}

        while queue and len(self.state_space) < 1000 and len(visited) < 5000:
            sh, path = queue.popleft()
            if len(path) >= 20:
                continue

            # Apply each transform from this state
            grid = self.state_space.get(sh)
            if grid is None:
                continue

            # We need to navigate to this state first — skip for now
            # Just explore from initial state breadth-first
            for act_num in self.transforms:
                if act_num not in [1, 2, 3, 4]:
                    continue

                # Apply transform manually to grid
                new_grid = self._apply_transform(grid, act_num)
                new_hash = _hash_grid(new_grid)

                if new_hash not in visited:
                    visited.add(new_hash)
                    self.state_space[new_hash] = new_grid
                    self.transition_graph[sh][act_num] = new_hash
                    queue.append((new_hash, path + [act_num]))

    def _apply_transform(self, grid: np.ndarray, act_num: int) -> np.ndarray:
        """Apply a known transform to a grid copy."""
        t = self.transforms.get(act_num)
        if not t or t.is_identity:
            return grid.copy()

        new_grid = grid.copy()
        # Copy the transform's mapping to the new grid
        # But the transform was recorded from initial state — need to apply to arbitrary state
        # Simplification: just apply the deltas
        for (r, c), (old_val, new_val) in t.mapping.items():
            if 0 <= r < 64 and 0 <= c < 64:
                # Only change if the pixel matches what we expect
                if grid[r, c] == old_val:
                    new_grid[r, c] = new_val

        return new_grid

    def _infer_goal(self, obs):
        """Identify what the goal pattern should be."""
        grid = self.initial_grid

        # Heuristic 1: The output area is the most "structured" region
        # that changes when transforms are applied.

        # Heuristic 2: Look for symmetry/symmetry-breaking
        # Many ARC puzzles have a target pattern that is a mirror/rotation

        # Heuristic 3: Count how many times each pixel changes across transforms
        change_freq = np.zeros((64, 64), dtype=int)
        for t in self.transforms.values():
            if t.is_identity:
                continue
            for (r, c) in t.mapping:
                if 0 <= r < 64 and 0 <= c < 64:
                    change_freq[r, c] += 1

        # Pixels that change frequently = part of the mechanism
        # Pixels that change rarely or predictably = output area
        self.output_region = (slice(0, 64), slice(50, 64))  # rightmost columns

    def _plan(self) -> Optional[List[int]]:
        """Find shortest path from initial state to any state matching goal."""
        start_hash = _hash_grid(self.initial_grid)

        # For now: BFS to any state that hasn't been seen before
        # (pure exploration as a baseline)
        queue = deque([(start_hash, [])])
        visited = {start_hash}

        while queue:
            sh, path = queue.popleft()
            if len(path) >= 15:
                continue

            for act_num in sorted(self.transitions.get(sh, {}).keys()):
                next_hash = self.transitions[sh][act_num]
                if next_hash not in visited:
                    visited.add(next_hash)
                    new_path = path + [act_num]
                    # Check goal condition
                    grid = self.state_space.get(next_hash)
                    if grid is not None and self._is_goal(grid):
                        return new_path
                    queue.append((next_hash, new_path))

        return None

    def _is_goal(self, grid: np.ndarray) -> bool:
        """Check if this grid state satisfies the goal condition."""
        # Simplest goal condition: output area is all same color (completed)
        if self.output_region:
            region = grid[self.output_region]
            unique = len(np.unique(region))
            if unique <= 2:
                return True
        return False

    def report(self) -> str:
        lines = ["Transform Model:"]
        for act_num, t in sorted(self.transforms.items()):
            lines.append(f"  {t.describe()}")
        lines.append(f"  States explored: {len(self.state_space)}")
        return "\n".join(lines)

    def save(self):
        out = Path("/home/redgamer/arc_agi_agent/transform_model.json")
        data = {
            "transforms": {str(k): {"pixels": v.changed_pixels, "identity": v.is_identity}
                          for k, v in self.transforms.items()},
            "states": len(self.state_space),
        }
        out.write_text(json.dumps(data, indent=2))
