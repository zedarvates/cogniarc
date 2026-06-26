#!/usr/bin/env python3
"""
ARC-AGI-3 Agent v3 — Real-step BFS with transform caching.

Strategy:
  1. Map action transforms from initial state
  2. BFS the real environment (steps are fast, ~1000 FPS)
  3. Use transform model to SKIP already-explored states
  4. Cache state→sequence mappings
  5. Execute winning sequence, continue to next sub-level
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent))

from transforms import Transform, map_transforms, _hash_grid
from skill_tree import SkillTree
from temporal_inference import TemporalReasoner

import arc_agi
from arcengine import GameAction


def real_bfs(env, actions: List[int], max_depth: int = 10,
             max_states: int = 10000) -> Optional[Tuple[List[int], Any]]:
    """
    BFS the REAL environment to find a winning sequence.
    Uses env.reset() between branches.
    Returns (sequence, winning_observation) or None.
    """
    # For true BFS, we'd need to reset and replay for each branch.
    # More efficient: DFS with rollback via reset+replay, or
    # just do BFS from a single position and keep going.

    # Practical approach: run the environment step by step,
    # tracking states. If we see a new state, explore from there.
    # This is DFS, not BFS — but uses minimal resets.

    env.reset()
    obs = env.reset()
    start_hash = _hash_grid(obs.frame[0])
    steps = 0

    # Stack: (state_hash, path_to_here, observation)
    stack = [(start_hash, [], obs)]
    visited = {start_hash}
    best = None
    best_levels = 0

    while stack and steps < max_states:
        sh, path, obs = stack.pop()

        if len(path) >= max_depth:
            continue

        for act_num in actions:
            # Replay path to get to this state
            env.reset()
            obs2 = env.reset()
            for a in path:
                obs2 = env.step(getattr(GameAction, f"ACTION{a}"))
            # Now apply the new action
            obs2 = env.step(getattr(GameAction, f"ACTION{act_num}"))
            steps += 1

            new_hash = _hash_grid(obs2.frame[0])
            new_path = path + [act_num]
            levels = getattr(obs2, 'levels_completed', 0)

            if levels > best_levels:
                best_levels = levels
                best = (new_path, obs2)
                print(f"  [BFS] Found level {levels} at depth {len(new_path)}: {new_path}")
                if str(getattr(obs2, 'state', '')) in ('WIN', 'FINISHED', 'GameState.WIN'):
                    return best

            if new_hash not in visited:
                visited.add(new_hash)
                stack.append((new_hash, new_path, obs2))

    print(f"  [BFS] Explored {steps} states, {len(visited)} unique, best_level={best_levels}")
    return best


# ── Agent ────────────────────────────────────────────────

class ArcAgentV3:
    def __init__(self, game_name: str, max_steps: int = 500):
        self.game_name = game_name
        self.max_steps = max_steps
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(game_name)
        self.skills = SkillTree()
        self.transforms: Dict[int, Transform] = {}

    def run(self) -> dict:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f" ARC AGENT V3 — {self.game_name}")
        print(f"{'='*60}")

        obs = self.env.reset()
        initial_grid = obs.frame[0].copy()
        actions = list(obs.available_actions or [])
        win_levels = getattr(obs, 'win_levels', 1)

        print(f"  Grid: {initial_grid.shape}, Actions: {actions}, Levels: {win_levels}")

        # Map transforms
        print("\n[Phase 1] Mapping transforms...")
        self.transforms = map_transforms(self.env, initial_grid, actions)
        for act_num, t in sorted(self.transforms.items()):
            print(f"  {t.describe()}")

        # Temporal analysis
        print("\n[Phase 1.5] Temporal pattern analysis...")
        temporal = TemporalReasoner()
        temporal.add_frame(initial_grid)
        grid_history = [initial_grid]
        # Take 3 random steps to observe temporal dynamics
        obs_temp = self.env.reset()
        for _ in range(min(3, len(actions) - 1)):
            action = np.random.choice(actions)
            obs_temp = self.env.step(getattr(GameAction, f"ACTION{action}"))
            grid = obs_temp.frame[0].copy()
            grid_history.append(grid)
            temporal.add_frame(grid)

        temporal_pattern = temporal.analyze()
        print(f"  Temporal pattern: {temporal_pattern.type.value} "
              f"(confidence: {temporal_pattern.confidence:.0%})")
        print(f"  Deltas: {len(temporal.deltas)} frames analyzed")

        # If temporal pattern indicates a clear transformation, predict next state
        if temporal_pattern.confidence > 0.7:
            predicted = temporal.predict()
            if predicted is not None:
                print(f"  Predicted next state shape: {predicted.shape}")

        # Reset environment for BFS
        self.env.reset()

        # Real-step BFS
        print(f"\n[Phase 2] Real-step BFS (max_depth=10)...")
        result = real_bfs(self.env, actions, max_depth=10, max_states=5000)

        total_steps = 0
        if result:
            plan, winning_obs = result
            print(f"\n[Phase 3] Found solution: {plan}")
            # Execute one more time to verify
            self.env.reset()
            obs = self.env.reset()
            for act_num in plan:
                obs = self.env.step(getattr(GameAction, f"ACTION{act_num}"))
                total_steps += 1
            levels = getattr(obs, 'levels_completed', 0)
            state = getattr(obs, 'state', '?')
            print(f"  Levels: {levels}/{win_levels}, State: {state}")
        else:
            print(f"\n[Phase 3] No solution found within limits")
            total_steps = 0

        elapsed = time.time() - t0
        result_dict = {
            "game": self.game_name,
            "steps": total_steps,
            "plan_found": result is not None,
            "elapsed": round(elapsed, 1),
        }

        print(f"\n{'='*60}")
        print(f" RESULT: {self.game_name} — plan={'YES' if result else 'NO'}, "
              f"{elapsed:.1f}s")
        print(f"{'='*60}")

        return result_dict


if __name__ == "__main__":
    games = sys.argv[1:] if len(sys.argv) > 1 else ["ls20"]
    for game in games:
        agent = ArcAgentV3(game, max_steps=300)
        agent.run()
