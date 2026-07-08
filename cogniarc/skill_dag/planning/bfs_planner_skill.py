"""BFS Planner Skill — Real-step BFS / A* on physics transitions."""

from __future__ import annotations

import hashlib
import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field
from collections import deque


def state_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


@dataclass
class PlanResult:
    success: bool
    action_sequence: List[int]
    states_explored: int
    goal_state_hash: Optional[str] = None
    path_cost: int = 0


class BFSPlannerSkill:
    """BFS/A* planner using known physics transitions."""

    def __init__(self, max_depth: int = 30, max_states: int = 10000):
        self.max_depth = max_depth
        self.max_states = max_states
        self.transitions: Dict[Tuple[str, int], str] = {}
        self.reverse_transitions: Dict[str, List[Tuple[str, int]]] = {}
        self.states_seen: Set[str] = set()
        self.action_costs: Dict[int, int] = {}

    def load_physics(self, transitions: Dict[Tuple[str, int], str],
                     states_seen: Set[str],
                     action_costs: Optional[Dict[int, int]] = None):
        """Load transitions from physics skill."""
        self.transitions = transitions
        self.states_seen = states_seen
        self.action_costs = action_costs or {}

        # Build reverse index for backward search
        self.reverse_transitions = {}
        for (sh, action), next_sh in transitions.items():
            if next_sh not in self.reverse_transitions:
                self.reverse_transitions[next_sh] = []
            self.reverse_transitions[next_sh].append((sh, action))

    def set_action_cost(self, action: int, cost: int):
        self.action_costs[action] = cost

    def bfs_plan(self, start_grid: np.ndarray,
                 goal_condition: Callable[[np.ndarray], bool],
                 state_cache: Dict[str, np.ndarray]) -> PlanResult:
        """Breadth-first search for shortest action sequence."""
        start_hash = state_hash(start_grid)

        if start_hash not in self.states_seen:
            return PlanResult(False, [], 0, path_cost=0)

        queue = deque([(start_hash, [])])
        visited = {start_hash}
        path_cost = 0

        while queue and len(visited) < self.max_states:
            sh, path = queue.popleft()

            if len(path) >= self.max_depth:
                continue

            if sh in state_cache:
                grid = state_cache[sh]
                if goal_condition(grid):
                    return PlanResult(True, path, len(visited), sh, len(path))

            for action in sorted(self.action_costs.keys()):
                key = (sh, action)
                next_hash = self.transitions.get(key)
                if next_hash is None or next_hash in visited:
                    continue

                visited.add(next_hash)
                new_path = path + [action]
                path_cost = sum(self.action_costs.get(a, 1) for a in new_path)
                queue.append((next_hash, new_path))

        return PlanResult(False, [], len(visited), path_cost=0)

    def astar_plan(self, start_grid: np.ndarray,
                   goal_condition: Callable[[np.ndarray], bool],
                   state_cache: Dict[str, np.ndarray],
                   heuristic: Optional[Callable[[str, np.ndarray], float]] = None) -> PlanResult:
        """A* search with heuristic."""
        start_hash = state_hash(start_grid)

        if start_hash not in self.states_seen:
            return PlanResult(False, [], 0)

        # Default heuristic: 0 (Dijkstra-like)
        if heuristic is None:
            heuristic = lambda sh, grid: 0.0

        open_set = [(heuristic(start_hash, start_grid), 0, start_hash, [])]
        g_score = {start_hash: 0}
        visited = set()

        while open_set and len(visited) < self.max_states:
            f, g, sh, path = heapq.heappop(open_set)

            if sh in visited:
                continue
            visited.add(sh)

            if len(path) >= self.max_depth:
                continue

            if sh in state_cache:
                grid = state_cache[sh]
                if goal_condition(grid):
                    return PlanResult(True, path, len(visited), sh, g)

            for action in sorted(self.action_costs.keys()):
                key = (sh, action)
                next_hash = self.transitions.get(key)
                if next_hash is None or next_hash in visited:
                    continue

                tentative_g = g + self.action_costs.get(action, 1)
                if tentative_g < g_score.get(next_hash, float('inf')):
                    g_score[next_hash] = tentative_g
                    new_path = path + [action]
                    h = heuristic(next_hash, state_cache.get(next_hash, np.zeros((64, 64))))
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, next_hash, new_path))

        return PlanResult(False, [], len(visited))

    def backward_plan(self, start_grid: np.ndarray,
                      goal_hash: str,
                      state_cache: Dict[str, np.ndarray]) -> PlanResult:
        """Plan backward from known goal state."""
        start_hash = state_hash(start_grid)

        if goal_hash not in self.reverse_transitions:
            return PlanResult(False, [], 0)

        # BFS from goal backward
        queue = deque([(goal_hash, [])])
        visited = {goal_hash}

        while queue and len(visited) < self.max_states:
            sh, path = queue.popleft()

            if sh == start_hash:
                return PlanResult(True, list(reversed(path)), len(visited), goal_hash, len(path))

            if len(path) >= self.max_depth:
                continue

            for prev_hash, action in self.reverse_transitions.get(sh, []):
                if prev_hash in visited:
                    continue
                visited.add(prev_hash)
                queue.append((prev_hash, [action] + path))

        return PlanResult(False, [], len(visited))


class RealtimeBFSSkill:
    """Real-step BFS that executes actions in the environment (like arc_agent.py)."""

    def __init__(self, max_steps: int = 500, max_depth: int = 15):
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.steps_taken = 0
        self.env = None

    def solve(self, env, actions: List[int],
              goal_condition: Callable) -> Optional[Tuple[List[int], Any]]:
        """Real-step BFS: reset + replay path for each branch."""
        self.env = env
        self.steps_taken = 0

        obs = env.reset()
        start_grid = obs.frame[0].copy()
        start_hash = state_hash(start_grid)

        stack = [(start_hash, [], obs)]
        visited = {start_hash}
        best = None
        best_levels = 0

        while stack and self.steps_taken < self.max_steps:
            sh, path, obs = stack.pop()

            if len(path) >= self.max_depth:
                continue

            for act_num in actions:
                # Replay path from start
                env.reset()
                obs2 = env.reset()
                for a in path:
                    obs2 = env.step(getattr(__import__('arcengine', fromlist=['GameAction']).GameAction, f"ACTION{a}"))
                    self.steps_taken += 1

                # Apply candidate action
                action_enum = getattr(__import__('arcengine', fromlist=['GameAction']).GameAction, f"ACTION{act_num}")
                obs2 = env.step(action_enum)
                self.steps_taken += 1

                new_hash = state_hash(obs2.frame[0])
                new_path = path + [act_num]
                levels = getattr(obs2, 'levels_completed', 0)

                if levels > best_levels:
                    best_levels = levels
                    best = (new_path, obs2)
                    if goal_condition(obs2):
                        return best

                if new_hash not in visited:
                    visited.add(new_hash)
                    stack.append((new_hash, new_path, obs2))

        return best


def create_bfs_planner_skill(max_depth: int = 30, max_states: int = 10000) -> BFSPlannerSkill:
    return BFSPlannerSkill(max_depth=max_depth, max_states=max_states)


def create_realtime_bfs_skill(max_steps: int = 500, max_depth: int = 15) -> RealtimeBFSSkill:
    return RealtimeBFSSkill(max_steps=max_steps, max_depth=max_depth)