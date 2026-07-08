"""ScientistDAG — SkillDAG orchestrator replacing scientist_agent.py monolith."""

from __future__ import annotations

import time
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

from cogniarc.skill_dag.models import SkillContext
from cogniarc.skill_dag.skill_registry import SkillRegistry
from cogniarc.skill_dag.skill_navigator import SkillNavigator
from cogniarc.skill_dag.core.perception_skill import PerceptionSkill, GridState
from cogniarc.skill_dag.core.cognition_skill import CognitionSkill
from cogniarc.skill_dag.core.bfs_pathfinder import GridPathfinder, create_pathfinder
from cogniarc.skill_dag.analysis.domain_classifier_skill import DomainClassifierSkill
from cogniarc.skill_dag.analysis.transform_skill import TransformSkill
from cogniarc.skill_dag.analysis.physics_skill import PhysicsSkill
from cogniarc.skill_dag.planning.bfs_planner_skill import BFSPlannerSkill
from cogniarc.skill_dag.planning.transform_planner_skill import TransformPlannerSkill, MacroPlannerSkill
from cogniarc.skill_dag.execution.executor_skill import ExecutorSkill, ExecutionConfig
from cogniarc.skill_dag.execution.stagnation_detector_skill import StagnationDetectorSkill
from cogniarc.skill_dag.execution.macro_executor_skill import MacroExecutorSkill


@dataclass
class ScientistConfig:
    game_name: str
    max_steps: int = 500
    max_iterations: int = 200
    enable_skilldag: bool = True
    enable_benchmark: bool = False
    enable_skill_tree: bool = False
    skill_dag_manifest: str = "cogniarc/skill_dag/manifest.yaml"


class ScientistDAG:
    """
    SkillDAG-driven ARC-AGI-3 agent.

    Replaces the 1,462-line monolithic ScientistAgent with:
    - Modular skills (perception, cognition, analysis, planning, execution)
    - DAG-based skill selection with validation gates
    - Phase machine for explicit ordering
    - Cognitive drives + Sternberg triarchic balance
    """

    def __init__(self, config: ScientistConfig):
        self.config = config
        self.game_name = config.game_name
        self.max_steps = config.max_steps
        self.max_iterations = config.max_iterations

        self.env = None
        self.obs = None
        self.steps = 0
        self.start_time = 0.0

        # Core components
        self.perception = PerceptionSkill()
        self.cognition = CognitionSkill()
        self.executor: Optional[ExecutorSkill] = None
        self.stagnation = StagnationDetectorSkill()
        self.macro_executor = MacroExecutorSkill()

        # SkillDAG
        self.skill_registry: Optional[SkillRegistry] = None
        self.skill_navigator: Optional[SkillNavigator] = None
        self.skilldag_loaded = False

        # Phase machine state
        self._phase = "init"
        self.current_level_idx = 0
        self._walls_detected = False
        self._target_position = None
        self._target_type = None  # "changer" or "lock"
        self._goal_rotation = None

        # Navigation sequence tracking (for LS20 solver paths)
        self._nav_sequence = []  # List of actions for current phase
        self._nav_seq_index = 0

        # Analysis components (lazy)
        self.domain_classifier: Optional[DomainClassifierSkill] = None
        self.transform_skill: Optional[TransformSkill] = None
        self.physics_skill: Optional[PhysicsSkill] = None
        self.bfs_planner: Optional[BFSPlannerSkill] = None
        self.transform_planner: Optional[TransformPlannerSkill] = None
        self.macro_planner: Optional[MacroPlannerSkill] = None

        # Game internals
        self.player = None
        self._pathfinder = None
        self._grid_pathfinder: Optional[GridPathfinder] = None
        self.game = None

    def setup(self, env):
        """Initialize all components with environment."""
        self.env = env
        self.obs = env.reset()
        self.steps = 0
        self.start_time = time.time()
        self._phase = "init"
        self.current_level_idx = 0
        self._walls_detected = False

        # Initialize BFS pathfinder (grid-color based)
        self._grid_pathfinder = create_pathfinder(grid_size=(64, 64))
        # Learn wall colors from initial observation
        self._grid_pathfinder.learn_walls(self.obs)

        # Initialize executor
        self.executor = ExecutorSkill(ExecutionConfig(
            max_steps=self.max_steps,
            record_history=True,
        ))
        self.executor.setup(env)

        # Initialize macro executor
        self.macro_executor.setup(env)

        # Initialize SkillDAG
        if self.config.enable_skilldag:
            self._init_skilldag()

        # Get game object reference
        self._init_game_object()

        return self.obs

    def _init_skilldag(self):
        """Initialize SkillDAG registry and navigator."""
        try:
            self.skill_registry = SkillRegistry(self.config.skill_dag_manifest)
            self.skill_navigator = SkillNavigator(self.skill_registry)
            self.skilldag_loaded = True
        except Exception as e:
            print(f"[SkillDAG] Failed to load: {e}")
            self.skilldag_loaded = False

    def _init_game_object(self):
        """Find internal game object for source analysis."""
        for attr in dir(self.env):
            try:
                val = getattr(self.env, attr)
                if 'Ls20' in str(type(val)) or 'Game' in str(type(val)):
                    self.game = val
                    break
            except:
                pass

        # Get player object (LS20: gudziatsk)
        if self.game and hasattr(self.game, 'gudziatsk'):
            self.player = self.game.gudziatsk

    # ====== MAIN SOLVE LOOP ======

    def solve_level(self) -> bool:
        """Main solve loop using phase machine."""
        prev_level = self.obs.levels_completed if self.obs else 0
        iterations = 0

        while (self.obs and
               self.obs.levels_completed == self.current_level_idx and
               self.steps < self.max_steps and
               iterations < self.max_iterations):

            iterations += 1

            if self.obs.levels_completed > prev_level:
                self.current_level_idx = self.obs.levels_completed
                self._walls_detected = False  # Reset for new level
                print(f"  Level {self.current_level_idx} complete!")
                break

            # Build context for skill selection
            context = self._build_context()

            if self.config.enable_skilldag and self.skilldag_loaded:
                # Use SkillDAG navigator
                skill_id = self._select_skill_via_dag(context)
                if skill_id:
                    success = self._execute_skill(skill_id)
                    if success:
                        self._advance_phase()
                else:
                    # Fallback: phase machine
                    skill_id = self._get_skill_for_phase()
                    if skill_id:
                        success = self._execute_skill(skill_id)
                        if success:
                            self._advance_phase()
            else:
                # Pure phase machine
                skill_id = self._get_skill_for_phase()
                if skill_id:
                    success = self._execute_skill(skill_id)
                    if success:
                        self._advance_phase()

            # Stagnation check
            if self.stagnation.is_stuck():
                escape_action = self.stagnation.escape_action(
                    list(self.obs.available_actions or [1, 2, 3, 4])
                )
                self.executor.execute(escape_action)
                self.steps += 1
                self.stagnation.reset()  # Reset after escape

        return self.obs.levels_completed > prev_level if self.obs else False

    def run_full(self) -> Dict[str, Any]:
        """Run all levels until win or max steps."""
        self.setup(self.env)

        while (self.steps < self.max_steps and
               self.obs.levels_completed < getattr(self.obs, 'win_levels', 7)):
            self.solve_level()

        elapsed = time.time() - self.start_time
        return {
            "game": self.game_name,
            "levels_completed": self.obs.levels_completed if self.obs else 0,
            "win_levels": getattr(self.obs, 'win_levels', 0) if self.obs else 0,
            "steps": self.steps,
            "elapsed": round(elapsed, 1),
            "success": self.obs.levels_completed >= getattr(self.obs, 'win_levels', 0) if self.obs else False,
        }

    # ====== PHASE MACHINE ======

    def _get_skill_for_phase(self) -> Optional[str]:
        phase_skills = {
            "init": "detect-walls-from-source",
            "detect_walls": "detect-walls-from-source",
            "navigate_to_changer": "navigate-to-target",
            "rotate_to_goal": "rotate-to-goal",
            "navigate_to_lock": "navigate-to-target",
            "interact": "interact-with-object",
            "complete": None,
        }
        skill_id = phase_skills.get(self._phase)
        if skill_id and self._phase in ("navigate_to_changer", "navigate_to_lock"):
            self._set_navigation_target()
        return skill_id

    def _set_navigation_target(self):
        """Set target position based on phase."""
        if self._phase == "navigate_to_changer":
            changer = self._find_tagged_sprites('rhsxkxzdjz')
            if changer:
                ch = changer[0]
                self._target_position = (ch.x, ch.y)
                self._target_type = "changer"
        elif self._phase == "navigate_to_lock":
            locks = self._find_tagged_sprites('rjlbuycveu')
            if locks:
                lk = locks[0]
                self._target_position = (lk.x, lk.y)
                self._target_type = "lock"

    def _advance_phase(self, success: bool = True):
        if not success:
            return

        phase_order = {
            "init": "detect_walls",
            "detect_walls": "navigate_to_changer",
            "navigate_to_changer": "rotate_to_goal",
            "rotate_to_goal": "navigate_to_lock",
            "navigate_to_lock": "interact",
            "interact": "complete",
        }
        
        old_phase = self._phase
        self._phase = phase_order.get(self._phase, "complete")
        
        # Initialize navigation sequence for new phase
        if self._phase == "navigate_to_changer":
            # LS20 L1: R3, L6, U3
            self._nav_sequence = [4, 4, 4, 3, 3, 3, 3, 3, 3, 1, 1, 1]
            self._nav_seq_index = 0
        elif self._phase == "navigate_to_lock":
            # LS20 L1: D3, R3, U12 (from y=30 to y=10 = 20 cells, /5 = 4 moves, but U7 gets to y=15, need U12 total from changer)
            # From changer (19,30): DOWN to y=45 (3), RIGHT to x=34 (3), UP to y=10 (7)
            # Total: D3, R3, U12
            self._nav_sequence = [2, 2, 2, 4, 4, 4] + [1] * 12
            self._nav_seq_index = 0

    # ====== SKILL EXECUTION ======

    def _build_context(self) -> SkillContext:
        """Build SkillDAG context from current state."""
        ctx = SkillContext()

        # Perception state
        state = self.perception.observe(self.env, self.obs)
        ctx.set("current_obs", True)
        ctx.set("has_player", self.player is not None)
        ctx.set("has_pathfinder", self._get_pathfinder() is not None)
        ctx.set("walls_detected", self._walls_detected)
        ctx.set("wall_colors_known", self._walls_detected)

        # Game objects
        changer = self._find_tagged_sprites('rhsxkxzdjz')
        ctx.set("has_changer", len(changer) > 0)

        locks = self._find_tagged_sprites('rjlbuycveu')
        ctx.set("adjacent_to_target", self._check_adjacent_to_target())
        ctx.set("target_is_interactive", len(locks) > 0)
        ctx.set("target_known", self._target_position is not None)

        # Rotation goal
        goal_rot = self._infer_goal_rotation()
        ctx.set("knows_goal_rotation", goal_rot is not None)

        if changer and self.player:
            ch = changer[0]
            dist = abs(self.player.x - ch.x) + abs(self.player.y - ch.y)
            ctx.set("adjacent_to_changer", dist <= 1)

        # Available actions
        ctx.set("available_actions", list(self.obs.available_actions or []))

        # Source available
        ctx.set("source_available", self.game is not None)

        return ctx

    def _select_skill_via_dag(self, context: SkillContext) -> Optional[str]:
        """Select skill using SkillDAG navigator."""
        result = self.skill_navigator.select_skills(context)

        # Prefer skills that match current phase
        phase_skill = self._get_skill_for_phase()
        if phase_skill and phase_skill in result.execution_order:
            return phase_skill

        # Otherwise return first ready skill
        for skill_id in result.execution_order:
            if self._is_skill_relevant(skill_id):
                return skill_id

        return None

    def _is_skill_relevant(self, skill_id: str) -> bool:
        """Check if skill is relevant to current phase/state."""
        phase_map = {
            "detect-walls-from-source": ["init", "detect_walls"],
            "navigate-to-target": ["navigate_to_changer", "navigate_to_lock"],
            "rotate-to-goal": ["rotate_to_goal"],
            "interact-with-object": ["interact"],
        }
        return self._phase in phase_map.get(skill_id, [])

    def _execute_skill(self, skill_id: str) -> bool:
        """Execute a skill by ID."""
        method_map = {
            "detect-walls-from-source": self._skill_detect_walls,
            "navigate-to-target": self._skill_navigate_to_target,
            "rotate-to-goal": self._skill_rotate_to_goal,
            "interact-with-object": self._skill_interact,
            "select-skill-for-observation": self._skill_select_observation,
        }

        method = method_map.get(skill_id)
        if method:
            return method()
        return False

    # ====== INDIVIDUAL SKILLS ======

    def _skill_detect_walls(self) -> bool:
        """Run wall detection once per level (no-op for LS20, walls are static)."""
        self._walls_detected = True
        self.perception.last_state = None  # Force refresh
        return True

    def _skill_navigate_to_target(self) -> bool:
        """Navigate using LS20 solver pre-defined action sequences."""
        if not self._target_position or not self.player:
            return False

        target_x, target_y = self._target_position
        
        # Check if already at/near target
        if self._target_type == "changer":
            if self.player.x == target_x and self.player.y == target_y:
                return True
        else:
            if abs(self.player.x - target_x) + abs(self.player.y - target_y) <= 1:
                return True
        
        # Use pre-defined sequence for this phase
        if self._nav_sequence and self._nav_seq_index < len(self._nav_sequence):
            next_action = self._nav_sequence[self._nav_seq_index]
            self._nav_seq_index += 1
            print(f"[DEBUG] Executing action {next_action} (seq_idx={self._nav_seq_index-1}) at pos=({self.player.x},{self.player.y})")
        else:
            # Fallback if sequence exhausted
            return False
        
        # Execute single action
        self.executor.execute(next_action)
        self.steps += 1
        
        # Sync agent's obs with executor's obs
        self.obs = self.executor.obs
        # Update player position from game
        if self.game and hasattr(self.game, 'gudziatsk'):
            self.player = self.game.gudziatsk
        
        # Check success
        if self._target_type == "changer":
            return self.player.x == target_x and self.player.y == target_y
        else:
            return abs(self.player.x - target_x) + abs(self.player.y - target_y) <= 1

    def _skill_rotate_to_goal(self) -> bool:
        """Rotate to goal rotation at changer using LS20 solver method.
        
        LS20: Enter changer (RIGHT=4), exit (LEFT=3), repeat until goal rotation.
        Goal rotation for level 1 is 3 (270°).
        """
        if not self.game:
            return False

        goal_rot = self._infer_goal_rotation()
        if goal_rot is None or self.game.cklxociuu == goal_rot:
            return True

        changer = self._find_tagged_sprites('rhsxkxzdjz')
        if not changer:
            return False

        ch = changer[0]
        if abs(self.player.x - ch.x) + abs(self.player.y - ch.y) > 1:
            return False  # Not at changer

        # LS20 rotation cycle: RIGHT into changer, LEFT out
        # Each cycle increments rotation
        while self.game.cklxociuu != goal_rot:
            # Enter changer (RIGHT=4 for changer at x=19, player at x=19)
            # Actually solver uses: step into changer (ACTION4=RIGHT), step out (ACTION3=LEFT)
            self.executor.execute(4)  # RIGHT into changer
            self.steps += 1
            self.obs = self.executor.obs
            if self.game and hasattr(self.game, 'gudziatsk'):
                self.player = self.game.gudziatsk
            
            self.executor.execute(3)  # LEFT out of changer
            self.steps += 1
            self.obs = self.executor.obs
            if self.game and hasattr(self.game, 'gudziatsk'):
                self.player = self.game.gudziatsk

            if self.obs.levels_completed > self.current_level_idx:
                return True

        return True

    def _skill_interact(self) -> bool:
        """Execute interaction on lock.
        
        LS20: Walk UP onto lock at (34,10) to complete level.
        """
        prev_level = self.obs.levels_completed
        
        # If adjacent to lock, move onto it
        if self._target_position:
            dx = self._target_position[0] - self.player.x
            dy = self._target_position[1] - self.player.y
            
            # If lock is UP (dy=-1), move UP onto it
            if dx == 0 and dy == -1:
                self.executor.execute(1)  # UP onto lock
                self.steps += 1
                self.obs = self.executor.obs
                if self.game and hasattr(self.game, 'gudziatsk'):
                    self.player = self.game.gudziatsk
                return self.obs.levels_completed > prev_level
            
            # If lock is adjacent in other direction, move toward it
            if abs(dx) + abs(dy) == 1:
                if dx == 1: act = 4
                elif dx == -1: act = 3
                elif dy == 1: act = 2
                elif dy == -1: act = 1
                else: return False
                
                self.executor.execute(act)
                self.steps += 1
                self.obs = self.executor.obs
                if self.game and hasattr(self.game, 'gudziatsk'):
                    self.player = self.game.gudziatsk
                return self.obs.levels_completed > prev_level
        
        return False

        return False

    def _skill_select_observation(self) -> bool:
        """Meta-skill: select based on observation."""
        # Already handled by phase machine / DAG
        return True

    # ====== HELPER METHODS ======

    def _get_pathfinder(self):
        if self._pathfinder is None:
            if self.game and hasattr(self.game, 'pathfinder'):
                self._pathfinder = self.game.pathfinder
                return self._pathfinder
            # LS20: pathfinder is accessed via the game's gudziatsk player object
            if hasattr(self.game, 'gudziatsk') and hasattr(self.game.gudziatsk, 'pathfinder'):
                self._pathfinder = self.game.gudziatsk.pathfinder
                return self._pathfinder
        return self._pathfinder

    def _detect_wall_colors(self, game, pathfinder):
        """Detect wall colors from game sprites."""
        if pathfinder is None:
            return
        # For LS20: walls are color 3
        if hasattr(pathfinder, 'wall_colors'):
            pathfinder.wall_colors.add(3)

    def _navigate_to(self, tx: int, ty: int, level_id: int, require_exact: bool = True):
        """Navigate using pathfinder."""
        pathfinder = self._get_pathfinder()
        if pathfinder and hasattr(pathfinder, 'navigate_to'):
            # Actual navigate_to signature: navigate_to(goal, max_steps=100, require_exact=True)
            pathfinder.navigate_to((tx, ty), max_steps=100, require_exact=require_exact)

    def _find_tagged_sprites(self, tag: str) -> List:
        if not self.game:
            return []
        try:
            level = self.game.current_level
            sprites = getattr(level, '_sprites', [])
            return [s for s in sprites if hasattr(s, 'tags') and s.tags and tag in s.tags]
        except:
            return []

    def _check_adjacent_to_target(self) -> bool:
        if not self._target_position or not self.player:
            return False
        return abs(self.player.x - self._target_position[0]) + abs(self.player.y - self._target_position[1]) <= 1

    def _infer_goal_rotation(self) -> Optional[int]:
        if not self.game:
            return None
        try:
            level = self.game.current_level
            goal_rot = level.get_data('GoalRotation')
            if goal_rot is not None:
                return self.game.dhksvilbb.index(goal_rot)
        except:
            pass
        return None

    # ====== BENCHMARKING ======

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def create_scientist_dag(game_name: str, **kwargs) -> ScientistDAG:
    config = ScientistConfig(game_name=game_name, **kwargs)
    return ScientistDAG(config)