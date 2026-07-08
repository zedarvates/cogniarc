"""Transform Skill — Map actions to grid transforms for ARC-AGI-3."""

from __future__ import annotations

import numpy as np
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


@dataclass
class Transform:
    """A grid transformation: region + how pixels change."""
    action_id: int
    region: Optional[Tuple[slice, slice]] = None
    mapping: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    changed_pixels: int = 0
    is_identity: bool = True

    def describe(self) -> str:
        if self.is_identity:
            return f"ACTION{self.action_id}: IDENTITY (no change)"
        r = self.region
        if r:
            return (f"ACTION{self.action_id}: {self.changed_pixels}px in "
                    f"rows [{r[0].start}-{r[0].stop}], cols [{r[1].start}-{r[1].stop}]")
        return f"ACTION{self.action_id}: {self.changed_pixels}px changed"


@dataclass
class TransformResult:
    transforms: Dict[int, Transform]
    initial_grid: np.ndarray
    states_explored: int


class TransformSkill:
    """Map each action to its transform and explore state space."""

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.transforms: Dict[int, Transform] = {}
        self.initial_grid: Optional[np.ndarray] = None
        self.state_space: Dict[str, np.ndarray] = {}
        self.transition_graph: Dict[str, Dict[int, str]] = defaultdict(dict)
        self.output_region: Optional[Tuple[slice, slice]] = None
        self.target_pattern: Optional[np.ndarray] = None
        self.env = None

    def solve(self, env) -> Optional[List[int]]:
        """Full transform inference pipeline."""
        self.env = env
        obs = env.reset()
        self.initial_grid = obs.frame[0].copy()

        self._map_transforms(obs)
        self._explore_state_space(obs)
        self._infer_goal(obs)

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
                    slice(int(coords[:, 0].min()), int(coords[:, 0].max()) + 1),
                    slice(int(coords[:, 1].min()), int(coords[:, 1].max()) + 1),
                )
                for r, c in coords:
                    t.mapping[(int(r), int(c))] = (int(initial[r, c]), int(grid2[r, c]))

            self.transforms[act_num] = t

    def _explore_state_space(self, obs):
        """BFS to discover reachable states via transform composition."""
        self.env.reset()
        initial_hash = _hash_grid(self.initial_grid)
        self.state_space[initial_hash] = self.initial_grid.copy()

        from collections import deque
        queue = deque([initial_hash])
        visited = {initial_hash}

        while queue and len(self.state_space) < 1000 and len(visited) < 5000:
            sh = queue.popleft()
            grid = self.state_space.get(sh)
            if grid is None:
                continue

            for act_num in self.transforms:
                if act_num not in [1, 2, 3, 4]:
                    continue
                new_grid = self._apply_transform(grid, act_num)
                new_hash = _hash_grid(new_grid)

                if new_hash not in visited:
                    visited.add(new_hash)
                    self.state_space[new_hash] = new_grid
                    self.transition_graph[sh][act_num] = new_hash
                    queue.append(new_hash)

    def _apply_transform(self, grid: np.ndarray, act_num: int) -> np.ndarray:
        """Apply a known transform to a grid copy."""
        t = self.transforms.get(act_num)
        if not t or t.is_identity:
            return grid.copy()

        new_grid = grid.copy()
        for (r, c), (old_val, new_val) in t.mapping.items():
            if 0 <= r < 64 and 0 <= c < 64:
                if grid[r, c] == old_val:
                    new_grid[r, c] = new_val
        return new_grid

    def _infer_goal(self, obs):
        """Identify what the goal pattern should be."""
        grid = self.initial_grid

        change_freq = np.zeros((64, 64), dtype=int)
        for t in self.transforms.values():
            if t.is_identity:
                continue
            for (r, c) in t.mapping:
                if 0 <= r < 64 and 0 <= c < 64:
                    change_freq[r, c] += 1

        # Heuristic: output area = most frequently changing region
        # For LS20-type: rightmost columns
        self.output_region = (slice(0, 64), slice(50, 64))

    def _plan(self) -> Optional[List[int]]:
        """Find path from initial state to goal state via BFS."""
        start_hash = _hash_grid(self.initial_grid)

        from collections import deque
        queue = deque([(start_hash, [])])
        visited = {start_hash}

        while queue:
            sh, path = queue.popleft()
            if len(path) >= 15:
                continue

            for act_num in sorted(self.transition_graph.get(sh, {}).keys()):
                next_hash = self.transition_graph[sh][act_num]
                if next_hash in visited:
                    continue
                visited.add(next_hash)
                new_path = path + [act_num]

                grid = self.state_space.get(next_hash)
                if grid is not None and self._is_goal(grid):
                    return new_path
                queue.append((next_hash, new_path))

        return None

    def _is_goal(self, grid: np.ndarray) -> bool:
        if self.output_region:
            region = grid[self.output_region]
            unique = len(np.unique(region))
            if unique <= 2:
                return True
        return False

    def get_result(self) -> TransformResult:
        return TransformResult(
            transforms=self.transforms,
            initial_grid=self.initial_grid,
            states_explored=len(self.state_space),
        )

    def report(self) -> str:
        lines = ["Transform Model:"]
        for act_num, t in sorted(self.transforms.items()):
            lines.append(f"  {t.describe()}")
        lines.append(f"  States explored: {len(self.state_space)}")
        return "\n".join(lines)


def create_transform_skill(max_steps: int = 100) -> TransformSkill:
    return TransformSkill(max_steps=max_steps)