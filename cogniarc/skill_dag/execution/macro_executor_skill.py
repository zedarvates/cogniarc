"""Macro Executor Skill — Execute skill_tree macros / composite actions."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


@dataclass
class Macro:
    name: str
    actions: List[int]
    preconditions: Optional[Callable] = None
    expect_level_complete: bool = False
    max_duration: int = 100


@dataclass
class MacroResult:
    macro_name: str
    success: bool
    steps_executed: int
    levels_before: int
    levels_after: int
    actions_completed: List[int]
    final_state: str


class MacroExecutorSkill:
    """Executes macro actions (sequences) with verification."""

    def __init__(self):
        self.macros: Dict[str, Macro] = {}
        self.env = None
        self.obs = None
        self.execution_log: List[MacroResult] = []

    def register_macro(self, name: str, actions: List[int],
                       preconditions: Callable = None,
                       expect_level_complete: bool = False,
                       max_duration: int = 100):
        self.macros[name] = Macro(
            name=name,
            actions=actions,
            preconditions=preconditions,
            expect_level_complete=expect_level_complete,
            max_duration=max_duration,
        )

    def setup(self, env):
        self.env = env
        self.obs = env.reset()

    def execute_macro(self, macro_name: str,
                      verify_fn: Optional[Callable] = None) -> MacroResult:
        """Execute a registered macro with optional verification."""
        if macro_name not in self.macros:
            return MacroResult(
                macro_name=macro_name, success=False, steps_executed=0,
                levels_before=0, levels_after=0, actions_completed=[],
                final_state="MACRO_NOT_FOUND"
            )

        macro = self.macros[macro_name]

        # Check preconditions
        if macro.preconditions and not macro.preconditions(self):
            return MacroResult(
                macro_name=macro_name, success=False, steps_executed=0,
                levels_before=getattr(self.obs, 'levels_completed', 0),
                levels_after=getattr(self.obs, 'levels_completed', 0),
                actions_completed=[],
                final_state="PRECONDITION_FAILED"
            )

        levels_before = getattr(self.obs, 'levels_completed', 0)
        actions_completed = []
        start_time = time.time()

        for i, action in enumerate(macro.actions):
            if time.time() - start_time > macro.max_duration:
                break

            action_enum = getattr(GameAction, f"ACTION{action}")
            self.obs = self.env.step(action_enum)
            actions_completed.append(action)

            # Early exit if level completed
            if macro.expect_level_complete:
                if getattr(self.obs, 'levels_completed', 0) > levels_before:
                    break

        levels_after = getattr(self.obs, 'levels_completed', 0)
        final_state = str(getattr(self.obs, 'state', 'RUNNING'))

        # Verify with custom function
        success = True
        if verify_fn:
            success = verify_fn(self)
        elif macro.expect_level_complete:
            success = levels_after > levels_before

        result = MacroResult(
            macro_name=macro_name,
            success=success,
            steps_executed=len(actions_completed),
            levels_before=levels_before,
            levels_after=levels_after,
            actions_completed=actions_completed,
            final_state=final_state,
        )

        self.execution_log.append(result)
        return result

    def execute_chain(self, macro_names: List[str],
                      stop_on_failure: bool = True) -> List[MacroResult]:
        """Execute a chain of macros in sequence."""
        results = []
        for name in macro_names:
            result = self.execute_macro(name)
            results.append(result)
            if stop_on_failure and not result.success:
                break
        return results

    def register_ls20_macros(self):
        """Register standard LS20 macros."""
        self.register_macro(
            "level1_to_changer",
            [4, 4, 4, 3, 3, 3, 3, 3, 3, 1, 1, 1],  # R×3, L×6, U×3
            expect_level_complete=False,
        )
        self.register_macro(
            "cycle_rotation_0",
            [6, 3],
            expect_level_complete=False,
        )
        self.register_macro(
            "level1_to_lock",
            [2, 2, 2, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1],  # D×3, R×3, U×7
            expect_level_complete=True,
        )
        self.register_macro(
            "interact_lock",
            [3, 3, 3, 3, 3],  # L×5
            expect_level_complete=True,
        )

    def get_available_macros(self, context: Dict[str, Any]) -> List[str]:
        available = []
        for name, macro in self.macros.items():
            if macro.preconditions and not macro.preconditions(self, context):
                continue
            available.append(name)
        return available


def create_macro_executor_skill() -> MacroExecutorSkill:
    me = MacroExecutorSkill()
    me.register_ls20_macros()
    return me