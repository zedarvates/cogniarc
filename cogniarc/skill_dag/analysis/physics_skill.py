"""Physics Skill — Deterministic world model for ARC-AGI-3."""

from __future__ import annotations

import hashlib
import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path

try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


def state_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(grid.tobytes()).hexdigest()[:20]


def grid_diff(a: np.ndarray, b: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    mask = a != b
    coords = np.argwhere(mask)
    return int(mask.sum()), mask, coords


@dataclass
class ActionProfile:
    id: int
    tests: int = 0
    effects: List[int] = field(default_factory=list)
    is_noop: bool = True
    is_movement: bool = False
    is_interaction: bool = False
    changes_agent: bool = False
    changes_other: bool = False
    blocks: Set[str] = field(default_factory=set)
    works: Set[str] = field(default_factory=set)


@dataclass
class ObjectTracker:
    objects: List[Dict[str, Any]] = field(default_factory=list)
    agent_id: Optional[int] = None

    def extract(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        objs = []
        for color in np.unique(grid):
            if color == 0:
                continue
            mask = grid == color
            coords = np.argwhere(mask)
            if len(coords) < 2:
                continue
            objs.append({
                "color": int(color),
                "center": tuple(coords.mean(axis=0).astype(int)),
                "size": len(coords),
                "bbox": (tuple(coords.min(axis=0)), tuple(coords.max(axis=0))),
            })
        return objs

    def find_agent(self, before: List[Dict], after: List[Dict]) -> Optional[int]:
        for i, (b, a) in enumerate(zip(before, after)):
            if b.get("center") != a.get("center"):
                return i
        return None


@dataclass
class PhysicsModel:
    states_discovered: int = 0
    transitions_mapped: int = 0
    actions_profiled: int = 0
    movement_actions: List[int] = field(default_factory=list)
    interaction_actions: List[int] = field(default_factory=list)
    noop_actions: List[int] = field(default_factory=list)
    confidence: float = 0.0
    steps_taken: int = 0


class PhysicsSkill:
    """Deterministic world model: state × action → next_state."""

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.transitions: Dict[Tuple[str, int], str] = {}
        self.states_seen: Set[str] = set()
        self.actions: Dict[int, ActionProfile] = {}
        self.tracker = ObjectTracker()
        self.current_state: Optional[str] = None
        self.confidence: float = 0.0
        self.steps_taken: int = 0
        self.domain: str = "unknown"
        self.env = None

    def discover(self, env) -> PhysicsModel:
        """Run full discovery: test actions, build model."""
        self.env = env
        obs = env.reset()
        self.current_state = state_hash(obs.frame[0])
        self.states_seen.add(self.current_state)

        self._phase_test_actions(obs)

        if self._has_movement():
            self._phase_explore_positions()

        return self._summarize()

    def _phase_test_actions(self, obs):
        available = list(obs.available_actions or [])
        for act_num in available[:4]:
            if self.steps_taken >= self.max_steps:
                return
            action = getattr(GameAction, f"ACTION{act_num}", None)
            if action is None:
                continue

            before_grid = obs.frame[0].copy()
            before_hash = state_hash(before_grid)
            before_objects = self.tracker.extract(before_grid)

            obs = self.env.step(action)
            self.steps_taken += 1

            after_grid = obs.frame[0]
            after_hash = state_hash(after_grid)
            after_objects = self.tracker.extract(after_grid)

            changed, mask, coords = grid_diff(before_grid, after_grid)

            self.transitions[(before_hash, act_num)] = after_hash
            self.states_seen.add(after_hash)

            profile = self.actions.setdefault(act_num, ActionProfile(act_num))
            profile.tests += 1
            profile.effects.append(changed)

            if changed == 0:
                profile.is_noop = True
                profile.blocks.add(before_hash)
            else:
                profile.is_noop = False
                profile.works.add(before_hash)

                agent_id = self.tracker.find_agent(before_objects, after_objects)
                if agent_id is not None:
                    profile.changes_agent = True
                    profile.is_movement = True
                if changed > 0 and not profile.changes_agent:
                    profile.changes_other = True
                    profile.is_interaction = True

            self.current_state = after_hash

    def _has_movement(self) -> bool:
        return any(p.is_movement for p in self.actions.values())

    def _phase_explore_positions(self):
        movement_actions = [aid for aid, p in self.actions.items() if p.is_movement]
        if not movement_actions:
            return

        for _ in range(5):
            if self.steps_taken >= self.max_steps:
                return
            for aid in movement_actions:
                self.env.step(getattr(GameAction, f"ACTION{aid}"))
                self.steps_taken += 1

        obs = self.env.step(getattr(GameAction, "ACTION1"))
        for act_num in self.actions:
            if self.steps_taken >= self.max_steps:
                return
            if act_num in movement_actions:
                continue
            before_hash = state_hash(obs.frame[0])
            action = getattr(GameAction, f"ACTION{act_num}")
            obs = self.env.step(action)
            self.steps_taken += 1
            after_hash = state_hash(obs.frame[0])
            self.transitions[(before_hash, act_num)] = after_hash

    def _summarize(self) -> PhysicsModel:
        n_states = len(self.states_seen)
        n_transitions = len(self.transitions)
        movement = [aid for aid, p in self.actions.items() if p.is_movement]
        interaction = [aid for aid, p in self.actions.items() if p.is_interaction]
        noop = [aid for aid, p in self.actions.items() if p.is_noop]

        total_possible = n_states * len(self.actions)
        self.confidence = min(n_transitions / max(total_possible, 1), 1.0)

        model = PhysicsModel(
            states_discovered=n_states,
            transitions_mapped=n_transitions,
            actions_profiled=len(self.actions),
            movement_actions=movement,
            interaction_actions=interaction,
            noop_actions=noop,
            confidence=round(self.confidence, 3),
            steps_taken=self.steps_taken,
        )
        self._save(model)
        return model

    def predict(self, grid: np.ndarray, action: int) -> Optional[str]:
        sh = state_hash(grid)
        return self.transitions.get((sh, action))

    def plan(self, start_grid: np.ndarray,
             goal_condition: Callable,
             max_depth: int = 30) -> Optional[List[int]]:
        from collections import deque

        start_hash = state_hash(start_grid)
        if start_hash not in self.states_seen:
            return None

        queue = deque([(start_hash, [])])
        visited = {start_hash}

        while queue:
            sh, path = queue.popleft()
            if len(path) >= max_depth:
                continue

            for aid in sorted(self.actions.keys()):
                next_hash = self.transitions.get((sh, aid))
                if next_hash is None or next_hash in visited:
                    continue
                visited.add(next_hash)
                new_path = path + [aid]
                queue.append((next_hash, new_path))

        return None

    def _save(self, model: PhysicsModel):
        out = Path("/home/redgamer/arc_agi_agent/physics_model.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": model.__dict__,
            "n_transitions": len(self.transitions),
            "n_states": len(self.states_seen),
            "actions": {str(k): {"is_movement": v.is_movement,
                                  "is_interaction": v.is_interaction,
                                  "is_noop": v.is_noop,
                                  "tests": v.tests}
                        for k, v in self.actions.items()},
        }
        out.write_text(json.dumps(data, indent=2, default=str))

    def report(self) -> str:
        return (
            f"Physics Model: {model.states_discovered} states, "
            f"{model.transitions_mapped} transitions\n"
            f"Confidence: {model.confidence:.2%}\n"
            f"Movement: {model.movement_actions}\n"
            f"Interaction: {model.interaction_actions}\n"
            f"No-op: {model.noop_actions}\n"
            f"Steps: {model.steps_taken}"
        )


def create_physics_skill(max_steps: int = 100) -> PhysicsSkill:
    return PhysicsSkill(max_steps=max_steps)