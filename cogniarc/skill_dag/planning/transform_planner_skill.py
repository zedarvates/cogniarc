"""Transform Planner Skill — Plan via transform composition (like transform_inference.py)."""

from __future__ import annotations

import numpy as np
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque


def state_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


@dataclass
class Transform:
    """A composable grid transform."""
    action_id: int
    region: Optional[Tuple[slice, slice]] = None
    mapping: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)
    changed_pixels: int = 0
    is_identity: bool = True
    preconditions: Dict[str, Any] = field(default_factory=dict)  # e.g., "player_adjacent": True

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply transform to a grid."""
        if self.is_identity:
            return grid.copy()

        new_grid = grid.copy()
        for (r, c), (old_val, new_val) in self.mapping.items():
            if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
                if new_grid[r, c] == old_val:
                    new_grid[r, c] = new_val
        return new_grid

    def can_apply(self, grid: np.ndarray, context: Dict[str, Any] = None) -> bool:
        """Check if transform can be applied in current state."""
        if self.is_identity:
            return True

        # Check preconditions
        if context:
            for cond, required in self.preconditions.items():
                if context.get(cond) != required:
                    return False

        # Check if mapping matches current grid
        for (r, c), (old_val, _) in self.mapping.items():
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                if grid[r, c] != old_val:
                    return False
        return True

    def compose(self, other: 'Transform') -> 'Transform':
        """Compose this transform after another (self ∘ other)."""
        if self.is_identity:
            return other
        if other.is_identity:
            return self

        # Apply other first, then self
        intermediate = other.apply(np.zeros((1, 1)))  # placeholder
        # For now, just chain mappings
        composed = Transform(0)
        composed.mapping = {**other.mapping, **self.mapping}
        composed.changed_pixels = other.changed_pixels + self.changed_pixels
        composed.is_identity = False
        return composed


@dataclass
class TransformPlan:
    sequence: List[int]
    total_cost: float
    expected_final_hash: str
    confidence: float


class TransformPlannerSkill:
    """Planner that reasons in transform space (faster than pixel space)."""

    def __init__(self):
        self.transforms: Dict[int, Transform] = {}
        self.initial_grid: Optional[np.ndarray] = None
        self.target_pattern: Optional[np.ndarray] = None
        self.output_region: Optional[Tuple[slice, slice]] = None
        self.state_cache: Dict[str, np.ndarray] = {}

    def load_transforms(self, transforms: Dict[int, Transform],
                        initial_grid: np.ndarray,
                        target_pattern: Optional[np.ndarray] = None,
                        output_region: Optional[Tuple[slice, slice]] = None):
        self.transforms = {k: v for k, v in transforms.items() if not v.is_identity}
        self.initial_grid = initial_grid
        self.target_pattern = target_pattern
        self.output_region = output_region
        self.state_cache = {state_hash(initial_grid): initial_grid.copy()}

    def plan(self, goal_condition: Callable[[np.ndarray], bool],
             max_depth: int = 15,
             max_states: int = 1000) -> Optional[TransformPlan]:
        """BFS in transform space."""
        start_hash = state_hash(self.initial_grid)

        queue = deque([(start_hash, [], 0.0)])
        visited = {start_hash}

        while queue and len(visited) < max_states:
            sh, seq, cost = queue.popleft()

            if len(seq) >= max_depth:
                continue

            grid = self.state_cache.get(sh)
            if grid is None:
                continue

            if goal_condition(grid) and seq:
                return TransformPlan(seq, cost, sh, 1.0)

            for action_id, transform in self.transforms.items():
                if not transform.can_apply(grid):
                    continue

                new_grid = transform.apply(grid)
                new_hash = state_hash(new_grid)

                if new_hash in visited:
                    continue

                visited.add(new_hash)
                self.state_cache[new_hash] = new_grid
                new_seq = seq + [action_id]
                new_cost = cost + transform.changed_pixels * 0.01
                queue.append((new_hash, new_seq, new_cost))

        return None

    def plan_to_target(self, target_grid: np.ndarray,
                       max_depth: int = 15) -> Optional[TransformPlan]:
        """Plan to reach a specific target grid pattern."""
        target_hash = state_hash(target_grid)

        def goal_cond(grid):
            return state_hash(grid) == target_hash

        return self.plan(goal_cond, max_depth)

    def plan_to_region_match(self, region: Tuple[slice, slice],
                             target_pattern: np.ndarray,
                             max_depth: int = 15) -> Optional[TransformPlan]:
        """Plan so that a specific region matches target pattern."""
        def goal_cond(grid):
            reg = grid[region]
            return reg.shape == target_pattern.shape and np.array_equal(reg, target_pattern)

        return self.plan(goal_cond, max_depth)


class MacroPlannerSkill:
    """Macro-action planner: sequences of primitives as single actions."""

    def __init__(self):
        self.macros: Dict[str, List[int]] = {}
        self.macro_preconditions: Dict[str, Callable] = {}
        self.macro_effects: Dict[str, Callable] = {}

    def register_macro(self, name: str, sequence: List[int],
                       preconditions: Callable = None,
                       effects: Callable = None):
        self.macros[name] = sequence
        self.macro_preconditions[name] = preconditions or (lambda ctx: True)
        self.macro_effects[name] = effects or (lambda ctx: None)

    def get_available_macros(self, context: Dict[str, Any]) -> List[str]:
        return [name for name, pre in self.macro_preconditions.items() if pre(context)]

    def expand_plan(self, macro_plan: List[str]) -> List[int]:
        """Expand macro plan to primitive actions."""
        actions = []
        for macro_name in macro_plan:
            if macro_name in self.macros:
                actions.extend(self.macros[macro_name])
            else:
                # Single action
                try:
                    actions.append(int(macro_name))
                except ValueError:
                    pass
        return actions


# Predefined macros for LS20-type games
LS20_MACROS = {
    "navigate_to_changer": [4, 4, 4, 3, 3, 3, 3, 3, 3, 1, 1, 1],  # R×3, L×6, U×3
    "cycle_rotation": [6, 3],  # enter changer + cycle
    "navigate_to_lock": [2, 2, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1],  # D×3, R×3, U×7
    "interact_lock": [3, 3, 3, 3, 3],  # L×5 to reach lock
}


def create_transform_planner_skill() -> TransformPlannerSkill:
    return TransformPlannerSkill()


def create_macro_planner_skill() -> MacroPlannerSkill:
    mp = MacroPlannerSkill()
    for name, seq in LS20_MACROS.items():
        mp.register_macro(name, seq)
    return mp