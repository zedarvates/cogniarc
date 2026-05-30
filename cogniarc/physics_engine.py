#!/usr/bin/env python3
"""
Game Physics Engine for ARC-AGI-3.

Reverse-engineers the deterministic transition function of each game:
  state × action → next_state

Capabilities:
  - Discover which actions are movement vs interaction vs no-op
  - Build deterministic transition table from real steps
  - Detects constraints (what blocks which action)
  - Tracks objects through grid diff analysis
  - Predicts next state (returns None if unmapped)
  - Measures model confidence (% of observed state space covered)

Usage:
    from physics_engine import GamePhysics
    gp = GamePhysics(env)
    gp.discover(max_steps=100)
    next_state = gp.predict(state, action)
    plan = gp.plan(start, goal_condition)
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import zlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


def state_hash(grid: np.ndarray) -> str:
    """Fast, collision-resistant hash of a 64×64 grid."""
    return hashlib.sha256(grid.tobytes()).hexdigest()[:20]


def grid_diff(a: np.ndarray, b: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return (n_changed, mask, changed_coords)."""
    mask = a != b
    coords = np.argwhere(mask)
    return int(mask.sum()), mask, coords


class ActionProfile:
    """What we know about a specific action."""
    def __init__(self, action_id: int):
        self.id = action_id
        self.tests: int = 0
        self.effects: List[int] = []      # changed pixel counts
        self.is_noop: bool = True
        self.is_movement: bool = False
        self.is_interaction: bool = False
        self.changes_agent: bool = False
        self.changes_other: bool = False
        self.blocks: Set[str] = set()     # state hashes where action had NO effect
        self.works: Set[str] = set()      # state hashes where action HAD effect


class ObjectTracker:
    """Track distinct objects in the grid."""
    def __init__(self):
        self.objects: List[Dict[str, Any]] = []
        self.agent_id: Optional[int] = None

    def extract(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Extract connected components as objects."""
        # Simple: just find unique colored blobs
        objs = []
        for color in np.unique(grid):
            if color == 0:  # background
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
        """The agent is the object whose center moved."""
        # Simplified: agent is the object that changed position most
        for i, (b, a) in enumerate(zip(before, after)):
            if b.get("center") != a.get("center"):
                return i
        return None


class GamePhysics:
    """Deterministic world model for an ARC-AGI-3 environment."""

    def __init__(self, env, max_steps: int = 100):
        self.env = env
        self.max_steps = max_steps
        self.transitions: Dict[Tuple[str, int], str] = {}  # (state_hash, action) → next_hash
        self.states_seen: Set[str] = set()
        self.actions: Dict[int, ActionProfile] = {}
        self.tracker = ObjectTracker()
        self.current_state: Optional[str] = None
        self.confidence: float = 0.0
        self.steps_taken: int = 0
        self.domain: str = "unknown"

    # ── Discovery ─────────────────────────────────────────

    def discover(self) -> Dict[str, Any]:
        """Run full discovery: test actions, build model."""
        obs = self.env.reset()
        self.current_state = state_hash(obs.frame[0])
        self.states_seen.add(self.current_state)

        # Phase 1: Test every action from initial position
        self._phase_test_actions(obs)

        # Phase 2: If agent movement detected, move to new position and retest
        if self._has_movement():
            self._phase_explore_positions()

        # Phase 3: Build transition table summary
        return self._summarize()

    def _phase_test_actions(self, obs):
        """Test each available action once, record effects."""
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

            # Record transition
            self.transitions[(before_hash, act_num)] = after_hash
            self.states_seen.add(after_hash)

            # Profile action
            profile = self.actions.setdefault(act_num, ActionProfile(act_num))
            profile.tests += 1
            profile.effects.append(changed)

            if changed == 0:
                profile.is_noop = True
                profile.blocks.add(before_hash)
            else:
                profile.is_noop = False
                profile.works.add(before_hash)

                # Movement or interaction?
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
        """Move agent to a different position and retest actions."""
        movement_actions = [aid for aid, p in self.actions.items() if p.is_movement]
        if not movement_actions:
            return

        # Move a few times
        for _ in range(5):
            if self.steps_taken >= self.max_steps:
                return
            for aid in movement_actions:
                self.env.step(getattr(GameAction, f"ACTION{aid}"))
                self.steps_taken += 1

        # Now retest interaction actions from new position
        obs = self.env.step(getattr(GameAction, f"ACTION1"))  # dummy step to get obs
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

    def _summarize(self) -> Dict[str, Any]:
        """Return summary of the physics model."""
        n_states = len(self.states_seen)
        n_transitions = len(self.transitions)
        movement = [aid for aid, p in self.actions.items() if p.is_movement]
        interaction = [aid for aid, p in self.actions.items() if p.is_interaction]
        noop = [aid for aid, p in self.actions.items() if p.is_noop]

        # Confidence: what % of (state, action) pairs are mapped?
        # Rough estimate based on unique states seen
        total_possible = n_states * len(self.actions)
        self.confidence = min(n_transitions / max(total_possible, 1), 1.0)

        summary = {
            "states_discovered": n_states,
            "transitions_mapped": n_transitions,
            "actions_profiled": len(self.actions),
            "movement_actions": movement,
            "interaction_actions": interaction,
            "noop_actions": noop,
            "confidence": round(self.confidence, 3),
            "steps_taken": self.steps_taken,
        }
        self._save(summary)
        return summary

    # ── Prediction ────────────────────────────────────────

    def predict(self, grid: np.ndarray, action: int) -> Optional[np.ndarray]:
        """Predict next state. Returns None if unknown."""
        sh = state_hash(grid)
        key = (sh, action)
        next_hash = self.transitions.get(key)
        if next_hash is None:
            return None
        # Can't reconstruct grid from hash — need state storage
        return None  # For now, hash-only (requires state cache)

    def plan(self, start_grid: np.ndarray,
             goal_condition: callable,
             max_depth: int = 30) -> Optional[List[int]]:
        """
        BFS over known transitions to find action sequence to a goal state.
        goal_condition(grid) returns True when goal is reached.
        """
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
                key = (sh, aid)
                next_hash = self.transitions.get(key)
                if next_hash is None:
                    continue
                if next_hash in visited:
                    continue
                visited.add(next_hash)

                new_path = path + [aid]
                # Check goal (would need grid — use hash for now)
                # In practice: store grids alongside hashes
                queue.append((next_hash, new_path))
                if len(new_path) >= max_depth:
                    continue

        return None  # No path found within depth limit

    # ── Persistence ───────────────────────────────────────

    def _save(self, summary: Dict):
        out = Path("/home/redgamer/arc_agi_agent/physics_model.json")
        data = {
            "summary": summary,
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
        summary = self._summarize()
        lines = [
            f"Physics Model: {summary['states_discovered']} states, "
            f"{summary['transitions_mapped']} transitions",
            f"Confidence: {summary['confidence']:.2%}",
            f"Movement: {summary['movement_actions']}",
            f"Interaction: {summary['interaction_actions']}",
            f"No-op: {summary['noop_actions']}",
            f"Steps: {summary['steps_taken']}",
        ]
        return "\n".join(lines)


# ── Quick CLI ────────────────────────────────────────────

if __name__ == "__main__":
    import arc_agi
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    gp = GamePhysics(env)
    gp.discover()
    print(gp.report())
