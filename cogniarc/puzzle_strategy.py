"""Puzzle strategy for grid-transformation games.

Uses the ProgramSynthesis DSL to discover grid transforms from action effects,
then applies found programs to complete levels. For games where each action
applies a known transform (rotation, flip, recolor, etc.) to the grid.

Three phases:
1. **Record** — try each action, record before/after grid pairs
2. **Search** — find shortest program matching observed transitions
3. **Apply** — chain found programs to reach goal state
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .domain_classifier import _color_diversity
from .program_synthesis import DEFAULT_PRIMITIVES, apply_program, synthesize


def _grid_to_signature(grid: np.ndarray) -> str:
    """Compact hash of grid state for change detection."""
    return str(hash(grid.tobytes()))


def _count_changed_cells(before: np.ndarray, after: np.ndarray) -> int:
    """Count how many cells differ between two grids."""
    return int(np.sum(before != after))


class PuzzleStrategy:
    """Solve strategy for puzzle/transform games (vc33, ft09, re86, etc.).

    Works by discovering what each action does to the grid (via before/after
    grid-state comparison), then chaining transforms to produce the target state.
    Falls back to systematic action cycling when transform search fails.
    """

    def __init__(self, agent):
        self.agent = agent
        self._action_effects: Dict[int, float] = {}  # action -> avg pixels changed
        self._action_programs: Dict[int, List[str]] = {}  # action -> found program
        self._grid_history: List[np.ndarray] = []
        self._consecutive_no_change = 0
        self._all_actions_tried = False

    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Solve a puzzle level by discovering and applying grid transforms."""
        prev_lvl = self.agent.obs.levels_completed
        print("  🧩 Puzzle strategy: discovering grid transforms...")

        available = list(self.agent.obs.available_actions or [])
        all_actions = [a for a in available if a >= 1] or [1, 2, 3, 4, 5, 6]

        # Phase 1: Record action effects
        print(f"  🧩 Recording effects for {len(all_actions)} actions...")
        for act in all_actions:
            if self.agent.obs.levels_completed > prev_lvl:
                return True
            if self.agent.steps > 400:
                break

            grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
            prev_state = str(self.agent.obs.state) if hasattr(self.agent.obs, 'state') else ""

            self.agent.step(act)

            grid_after = self.agent.obs.frame[0] if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
            if self.agent.obs.levels_completed > prev_lvl:
                print(f"  🧩 Action {act} completed level!")
                return True

            if grid_before is not None and grid_after is not None:
                diff = _count_changed_cells(grid_before, grid_after)
                colors = _color_diversity(grid_before, grid_after)
                print(f"  🧩 A{act}: {diff} cells changed, {colors} color pairs")

                if diff > 0:
                    self._action_effects[act] = diff
                    # Try to find a program that maps before→after
                    program = synthesize(
                        [(grid_before, grid_after)],
                        primitives=DEFAULT_PRIMITIVES,
                        max_depth=2,
                    )
                    if program is not None:
                        self._action_programs[act] = program
                        name = "→".join(program) if program else "identity"
                        print(f"     → Program: [{name}]")

        self._all_actions_tried = True

        # Phase 2: If we found programs, apply them in a cycle
        if self._action_programs:
            print(f"  🧩 Found {len(self._action_programs)} action programs, applying...")
            return self._apply_programs(prev_lvl, all_actions)

        # Phase 3: Fallback — systematic cycling with grid-state monitoring
        print("  🧩 No programs found, systematic cycling...")
        return self._systematic_cycling(prev_lvl, all_actions)

    def _apply_programs(self, prev_lvl: int, all_actions: List[int]) -> bool:
        """Apply discovered action programs in a cycle toward goal state.

        Tries each action that has a program, checks if applying its program
        to the current grid produces a meaningful change, and chains programs
        when a single action isn't enough.
        """
        for iteration in range(100):
            if self.agent.obs.levels_completed > prev_lvl:
                print(f"  🧩 Level {self.agent.obs.levels_completed} completed in {iteration} iterations!")
                return True
            if self.agent.steps > 400:
                break

            grid_current = self.agent.obs.frame[0].copy() if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
            if grid_current is None:
                break

            # Try each program action — pick the one that produces the most
            # grid change (indicating it's having an effect at this state)
            best_action = None
            best_diff = 0

            for act in sorted(self._action_programs.keys()):
                program = self._action_programs[act]
                predicted = apply_program(grid_current, program)
                diff = _count_changed_cells(grid_current, predicted)

                # If the predicted grid matches the current grid, this
                # action would be a no-op — skip it
                if diff > 0 and diff > best_diff:
                    best_diff = diff
                    best_action = act

            if best_action is not None and best_diff > 0:
                grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
                self.agent.step(best_action)
                grid_after = self.agent.obs.frame[0] if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
                if grid_before is not None and grid_after is not None:
                    actual_diff = _count_changed_cells(grid_before, grid_after)
                    if actual_diff > 0:
                        print(f"  🧩 +A{best_action}: {actual_diff} cells ({self._action_programs[best_action]})")
                        continue

            # If no program action works, try raw actions
            if best_action is None:
                action_taken = False
                for act in all_actions:
                    grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
                    self.agent.step(act)
                    grid_after = self.agent.obs.frame[0] if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
                    if grid_before is not None and grid_after is not None:
                        actual_diff = _count_changed_cells(grid_before, grid_after)
                        if actual_diff > 0:
                            print(f"  🧩 Raw A{act}: {actual_diff} cells")
                            action_taken = True
                            break
                        elif self.agent.obs.levels_completed > prev_lvl:
                            return True

                if not action_taken:
                    # Try interaction actions
                    for act in [5, 6]:
                        if act in (self.agent.obs.available_actions or []):
                            self.agent.step(act)
                            if self.agent.obs.levels_completed > prev_lvl:
                                return True
                            break
                    print(f"  🧩 No action has effect — trying next level")
                    return False

        return self.agent.obs.levels_completed > prev_lvl

    def _systematic_cycling(self, prev_lvl: int, all_actions: List[int]) -> bool:
        """Fallback: cycle through all actions systematically.

        Records grid state before/after each action to detect which
        actions actually change the grid at current state.
        """
        action_idx = 0
        consecutive_no_change = 0
        last_grid_hash = None

        for iteration in range(150):
            if self.agent.obs.levels_completed > prev_lvl:
                print(f"  🧩 Level completed in {iteration} iterations!")
                return True
            if self.agent.steps > 400:
                break

            grid_before = self.agent.obs.frame[0].copy() if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
            if grid_before is None:
                break

            current_hash = _grid_to_signature(grid_before)
            if current_hash == last_grid_hash:
                consecutive_no_change += 1
            else:
                consecutive_no_change = 0
                last_grid_hash = current_hash

            # If grid hasn't changed after trying all actions, try interact
            if consecutive_no_change >= len(all_actions):
                for act in [5, 6]:
                    if act in (self.agent.obs.available_actions or []):
                        prev_lvl_check = self.agent.obs.levels_completed
                        self.agent.step(act)
                        if self.agent.obs.levels_completed > prev_lvl_check:
                            return True
                        grid_after = self.agent.obs.frame[0] if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None
                        if grid_after is not None:
                            diff = _count_changed_cells(grid_before, grid_after)
                            if diff > 0:
                                print(f"  🧩 +A{act}: interact works ({diff} cells)")
                                consecutive_no_change = 0
                                break
                else:
                    # Really stuck — try any action
                    self.agent.step(all_actions[iteration % len(all_actions)])
                continue

            # Normal cycling
            action = all_actions[action_idx % len(all_actions)]
            self.agent.step(action)
            grid_after = self.agent.obs.frame[0] if self.agent.obs.frame and len(self.agent.obs.frame) > 0 else None

            if grid_before is not None and grid_after is not None:
                diff = _count_changed_cells(grid_before, grid_after)
                if diff > 0 and iteration < 20:
                    print(f"  🧩 +A{action}: {diff} cells")

            action_idx += 1

        return self.agent.obs.levels_completed > prev_lvl
