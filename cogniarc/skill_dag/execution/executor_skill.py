"""Executor Skill — Step environment, record, apply cognitive drives."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


@dataclass
class StepResult:
    action: int
    state_hash: str
    levels_completed: int
    reward: float
    game_state: str
    grid: np.ndarray
    success: bool


@dataclass
class ExecutionConfig:
    max_steps: int = 500
    record_history: bool = True
    apply_drives: bool = True
    save_path: Optional[str] = None


class ExecutorSkill:
    """Executes actions in environment, tracks state, applies cognitive drives."""

    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.env = None
        self.obs = None
        self.step_count = 0
        self.history: List[StepResult] = []
        self.total_reward = 0.0
        self.levels_completed = 0

    def setup(self, env) -> any:
        """Initialize with environment."""
        self.env = env
        self.obs = env.reset()
        self.step_count = 0
        self.history.clear()
        self.total_reward = 0.0
        self.levels_completed = 0
        return self.obs

    def execute(self, action: int,
                cognition_skill: Optional[Callable] = None,
                perception_skill: Optional[Callable] = None) -> StepResult:
        """Execute single action with optional cognitive evaluation."""
        if self.obs is None or self.env is None:
            raise RuntimeError("Executor not setup - call setup(env) first")

        if self.step_count >= self.config.max_steps:
            return StepResult(
                action=-1, state_hash="", levels_completed=self.levels_completed,
                reward=0.0, game_state="MAX_STEPS", grid=np.array([]), success=False
            )

        # Execute action
        action_enum = getattr(GameAction, f"ACTION{action}")
        self.obs = self.env.step(action_enum)
        self.step_count += 1

        # Extract results
        if not self.obs or not self.obs.frame or len(self.obs.frame) == 0:
            grid = np.array([])
        else:
            grid = self.obs.frame[0].copy()
        state_hash = self._hash_grid(grid)
        prev_levels = self.levels_completed
        self.levels_completed = getattr(self.obs, 'levels_completed', 0)
        game_state = str(getattr(self.obs, 'state', 'RUNNING'))

        reward = 1.0 if self.levels_completed > prev_levels else 0.0
        self.total_reward += reward

        # Apply cognitive drives if provided
        drive_score = 0.0
        if self.config.apply_drives and cognition_skill:
            drive_score = cognition_skill(action, state_hash, grid=grid)

        # Update perception if provided
        if perception_skill:
            perception_skill(self.env, self.obs)

        result = StepResult(
            action=action,
            state_hash=state_hash,
            levels_completed=self.levels_completed,
            reward=reward,
            game_state=game_state,
            grid=grid,
            success=self.levels_completed > prev_levels,
        )

        if self.config.record_history:
            self.history.append(result)

        return result

    def execute_sequence(self, actions: List[int],
                         stop_on_level: bool = True,
                         cognition_skill: Optional[Callable] = None,
                         perception_skill: Optional[Callable] = None) -> List[StepResult]:
        """Execute a sequence of actions."""
        results = []
        prev_level = self.levels_completed

        for action in actions:
            result = self.execute(action, cognition_skill, perception_skill)
            results.append(result)

            if stop_on_level and result.levels_completed > prev_level:
                break

            if not result.success and result.game_state in ('WIN', 'FINISHED', 'GameState.WIN'):
                break

        return results

    def execute_until(self, condition: Callable[[StepResult], bool],
                      max_actions: int = 100,
                      action_selector: Optional[Callable] = None) -> List[StepResult]:
        """Execute actions until condition met or max_actions reached."""
        results = []

        for _ in range(max_actions):
            if self.step_count >= self.config.max_steps:
                break

            # Select action
            if action_selector:
                action = action_selector(self)
            else:
                # Default: try actions 1-4
                available = getattr(self.obs, 'available_actions', [1, 2, 3, 4])
                action = available[0] if available else 1

            result = self.execute(action)
            results.append(result)

            if condition(result):
                break

        return results

    def _hash_grid(self, grid: np.ndarray) -> str:
        import hashlib
        return hashlib.sha256(grid.tobytes()).hexdigest()[:16]

    def get_state(self) -> Dict[str, Any]:
        current_grid = None
        if self.obs and self.obs.frame and len(self.obs.frame) > 0:
            current_grid = self.obs.frame[0]
        return {
            "step_count": self.step_count,
            "levels_completed": self.levels_completed,
            "total_reward": self.total_reward,
            "current_state": current_grid,
            "history_length": len(self.history),
        }

    def save_history(self, path: str = None):
        if not self.config.record_history:
            return
        out = Path(path or self.config.save_path or "/home/redgamer/arc_agi_agent/execution_history.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "steps": self.step_count,
            "levels": self.levels_completed,
            "reward": self.total_reward,
            "history": [
                {
                    "action": r.action,
                    "state_hash": r.state_hash,
                    "levels": r.levels_completed,
                    "reward": r.reward,
                    "state": r.game_state,
                }
                for r in self.history
            ],
        }
        out.write_text(json.dumps(data, indent=2))

    def reset(self):
        self.step_count = 0
        self.history.clear()
        self.total_reward = 0.0
        self.levels_completed = 0
        self.obs = None


def create_executor_skill(max_steps: int = 500, record_history: bool = True) -> ExecutorSkill:
    return ExecutorSkill(ExecutionConfig(max_steps=max_steps, record_history=record_history))