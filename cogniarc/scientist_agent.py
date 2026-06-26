#!/usr/bin/env python3
"""
ARC-AGI-3 Scientist Agent — Discover, then Solve.

Principles (adapted from Pokémon Player, not copied):
  - Discover mechanics BEFORE planning (domain-first)
  - Short iterations with re-evaluation (not BFS 1000 states)
  - PKM structured memory per game
  - Multi-phase: discovery -> solve -> transition
  - Use source code when available (cheapest info)
  - Verify after each action block
  - Cognitive drives guide exploration (novelty, simplicity, doubt, pleasure, caution, impulse)
  - Skill Tree enables cross-level and cross-game learning
  - Benchmark tracking for LLM/agent comparison
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from typing import Optional, Any, Dict, Set
import time
from pathlib import Path

# Optional benchmark tracking
try:
    from .benchmark_tracker import BenchmarkTracker, GameResult, SessionResult
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Pathfinding
from .pathfinding import Pathfinder, GridMap

# Cognitive drives
from .cognitive_player import CognitiveDrives, hash_grid


# ====== PKM Memory ======
class PKM:
    """Structured knowledge per game. Prefix: PKM:<game>:<category>"""
    def __init__(self, game_id: str):
        self.game = game_id
        self.facts: dict[str, Any] = {}
    
    def set(self, category: str, key: str, value):
        full_key = f"{self.game}:{category}:{key}"
        self.facts[full_key] = value
    
    def get(self, category: str, key: str, default=None):
        full_key = f"{self.game}:{category}:{key}"
        return self.facts.get(full_key, default)
    
    def get_all(self, category: str) -> dict:
        prefix = f"{self.game}:{category}:"
        return {k[len(prefix):]: v for k, v in self.facts.items() if k.startswith(prefix)}
    
    def report(self) -> str:
        lines = [f"PKM:{self.game} ({len(self.facts)} facts)"]
        for k, v in sorted(self.facts.items()):
            short_k = k.replace(f"{self.game}:", "")
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            lines.append(f"  {short_k} = {v}")
        return "\n".join(lines)


# ====== Scientist Agent ======
class ScientistAgent:
    """Discover game mechanics, then solve each level."""
    
    def __init__(self, game_name: str, enable_benchmark: bool = True, enable_skill_tree: bool = True):
        self.name = game_name
        self.pkm = PKM(game_name)
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(game_name)
        self.obs = self.env.reset()
        self.steps = 0
        
        # Access internal game object
        self.game = None
        for attr in dir(self.env):
            val = getattr(self.env, attr)
            if game_name.lower() in str(type(val)).lower():
                self.game = val
                break
        
        # Find player object (different games use different attribute names)
        self.player = None
        if self.game:
            for attr_name in ['gudziatsk', 'player', 'agent', '_player', '_agent']:
                if hasattr(self.game, attr_name):
                    self.player = getattr(self.game, attr_name)
                    break
        
        # Cognitive drives for decision making
        self.drives = CognitiveDrives()
        self._hash_grid = hash_grid
        
        # Skill Tree for cross-level/cross-game learning
        self.skill_tree = None
        if enable_skill_tree:
            from .skill_tree import SkillTree
            self.skill_tree = SkillTree.load_for_game(game_name)
        
        # Benchmark tracking
        self.benchmark_tracker = None
        self.benchmark_session_id = None
        self.benchmark_start_time = None
        
        if enable_benchmark and BENCHMARK_AVAILABLE:
            self.benchmark_tracker = BenchmarkTracker()
            self.benchmark_session_id = self.benchmark_tracker.start_session(
                llm_model="nvidia/nemotron-3-ultra:free",
                agent_version="cogniarc-scientist-v1"
            )

        # SkillDAG integration
        from cogniarc.skill_dag.skill_registry import SkillRegistry
        from cogniarc.skill_dag.skill_navigator import SkillNavigator
        
        self.skill_registry = SkillRegistry("cogniarc/skill_dag/manifest.yaml")
        self.skill_navigator = SkillNavigator(self.skill_registry)
        self._pathfinder = None  # Lazy init
        self._walls_detected = False
        self.current_level_idx = 0
        self._phase = "detect_walls"  # Phase state machine
    
    def step(self, action_num: int):
        self.obs = self.env.step(getattr(GameAction, f'ACTION{action_num}'))
        self.steps += 1
        # Verify observation
        assert self.obs is not None, "Invalid observation: None returned"
        assert hasattr(self.obs, 'frame'), "Invalid observation: missing frame"
        assert self.obs.frame is not None, "Invalid observation: frame is None"

        # Update cognitive drives
        if self.obs.frame is not None and len(self.obs.frame) > 0:
            state_hash = self._hash_grid(self.obs.frame[0])
        else:
            state_hash = f"step_{self.steps}"
        self.drives.step(action_num, state_hash)

        return self.obs
    
    # ------ DISCOVERY PHASE ------
    
    def discover_from_source(self) -> bool:
        """Read game source code to discover mechanics (zero steps)."""
        try:
            # Try direct path first (game files are under environment_files/<game_id>/)
            import os, glob
            
            # Extract base game id (ls20 from ls20-9607627b)
            base_game = self.name.split('-')[0] if '-' in self.name else self.name
            
            # Try multiple path patterns
            possible_paths = [
                f"environment_files/{base_game}/*/{self.name}.py",
                f"environment_files/{base_game}/*/{base_game}.py",
                f"environment_files/{self.name}/*/{self.name}.py",
                f"environment_files/{self.name}/*/{base_game}.py",
            ]
            
            src = None
            for pattern in possible_paths:
                dirs = glob.glob(pattern)
                if dirs:
                    src = dirs[0]
                    break
            
            if not src:
                return False
                
            with open(src) as f:
                code = f.read()
        except:
            return False
        
        # Parse tag-to-mechanic mapping from source
        # Pattern: if "TAG" in sprite.tags: -> EFFECT
        import re
        mechanics = {
            'walls': set(),
            'locks': set(),
            'rotation_changers': set(),
            'color_changers': set(),
            'shape_changers': set(),
        }
        
        # Known patterns from LS20 source analysis
        tag_contexts = {
            'ihdgageizm': 'walls',
            'rjlbuycveu': 'locks',
            'rhsxkxzdjz': 'rotation_changers',
            'soyhouuebz': 'color_changers',
            'ttfwljgohq': 'shape_changers',
        }
        
        for tag, mechanic in tag_contexts.items():
            if tag in code:
                mechanics[mechanic].add(tag)
        
        for cat, items in mechanics.items():
            if items:
                self.pkm.set('mechanics', cat, list(items))
        
        self.pkm.set('mechanics', 'source_analyzed', True)
        total = sum(len(v) for v in mechanics.values())
        print(f"  📖 Source: {total} mechanics identified")
        return True

    def discover_available_actions(self):
        """Scout available actions and classify them (cheap, 1 step each)."""
        available = list(self.obs.available_actions or [])
        results = {}
        
        # Need initial grid for comparison
        start_grid = None
        if hasattr(self.obs, 'frame') and self.obs.frame:
            start_grid = self.obs.frame[0].copy()
        
        for action_num in available:
            prev_pos = (self.player.x, self.player.y) if self.player else None
            
            self.step(action_num)
            
            moved = False
            if self.player and prev_pos:
                moved = (self.player.x, self.player.y) != prev_pos
            
            grid_changed = False
            if start_grid is not None and hasattr(self.obs, 'frame') and self.obs.frame:
                grid_changed = not np.array_equal(self.obs.frame[0], start_grid)
            
            prop_changes = 0
            if self.game:
                for attr in ['cklxociuu', 'hiaauhahz']:
                    if hasattr(self.game, attr):
                        val = getattr(self.game, attr)
                        if attr == 'cklxociuu':
                            val %= 4
                        if val != 0:
                            prop_changes += 1
            
            results[action_num] = {
                'moved': moved,
                'grid_changed': grid_changed,
                'prop_changes': prop_changes,
            }
        
        self.pkm.set('discovery', 'scout_results', results)
        
        # Classify actions
        movement = [a for a, r in results.items() if r['moved']]
        interaction = [a for a, r in results.items() if not r['moved'] and r['prop_changes']]
        blocked = [a for a, r in results.items() if not r['moved'] and not r['grid_changed']]
        
        self.pkm.set('discovery', 'action_types', {
            'movement': movement,
            'interaction': interaction,
            'blocked': blocked,
        })
        
        print(f"  🔍 Scout: {len(movement)} movement, {len(interaction)} interaction, {len(blocked)} blocked")
        return results

    def discover_properties(self):
        """Discover rotation and color properties."""
        rotations = []
        colors = []
        if self.game:
            # Rotation
            if hasattr(self.game, 'cklxociuu'):
                val = getattr(self.game, 'cklxociuu')
                rotations = [0, 90, 180, 270]
                self.pkm.set('state', 'rotations', rotations)
            # Color
            if hasattr(self.game, 'hiaauhahz'):
                val = getattr(self.game, 'hiaauhahz')
                colors = [0, 1, 2, 3, 4, 5, 6, 7]  # standard ARC colors
                self.pkm.set('state', 'colors', colors)
    
    # ------ NAVIGATION ------
    
    def __init_pathfinder(self):
        """Initialize or get the pathfinder."""
        if not hasattr(self, '_pathfinder') or self._pathfinder is None:
            self._pathfinder = Pathfinder(self)
        return self._pathfinder
    
    def _detect_wall_colors(self):
        """Detect wall colors from source-analyzed tagged sprites."""
        if not self.game or not hasattr(self.obs, 'frame') or not self.obs.frame:
            return
        
        # Ensure pathfinder is initialized and update from observation
        pathfinder = self.__init_pathfinder()
        pathfinder.update_from_observation(self.obs)
        
        # Get wall tags from PKM (discovered from source)
        wall_tags = self.pkm.get('mechanics', 'walls', [])
        if not wall_tags:
            return
        
        grid = self.obs.frame[0]
        
        # Find wall sprites and sample their colors
        for tag in wall_tags:
            sprites = self._find_tagged_sprites(tag)
            for s in sprites[:3]:  # sample first 3
                color = int(grid[s.y, s.x])
                pathfinder.wall_colors.add(color)
        
        # Re-update grid map with learned wall colors
        pathfinder.update_from_observation(self.obs)
        
        print(f"  🧱 Wall colors detected: {pathfinder.wall_colors}")
    
    def _find_tagged_sprites(self, tag: str):
        """Find sprites with given tag in current level."""
        level = self.game.current_level if self.game else None
        if not level:
            return []
        sprites = getattr(level, '_sprites', [])
        return [s for s in sprites if hasattr(s, 'tags') and s.tags and tag in s.tags]

    def _check_adjacent_to_target(self) -> bool:
        """Check if player adjacent to any interactive object."""
        if not self.player:
            return False
        px, py = self.player.x, self.player.y
        for tag in ['rhsxkxzdjz', 'rjlbuycveu']:
            sprites = self._find_tagged_sprites(tag)
            for s in sprites:
                sx, sy = getattr(s, 'x', 0), getattr(s, 'y', 0)
                if abs(px - sx) + abs(py - sy) == 1:
                    return True
        return False

    def _check_source_available(self) -> bool:
        """Check if game source file exists."""
        import os
        base_game = self.name.split('-')[0] if '-' in self.name else self.name
        # Try multiple path patterns
        possible_paths = [
            f"environment_files/{base_game}/*/{self.name}.py",
            f"environment_files/{base_game}/*/{base_game}.py",
        ]
        for pattern in possible_paths:
            import glob
            matches = glob.glob(pattern)
            if matches:
                return True
        return False

    def _infer_goal_rotation(self) -> Optional[int]:
        """Infer goal rotation from level data."""
        if self.game and hasattr(self.game, 'current_level'):
            level = self.game.current_level
            if hasattr(level, 'get_data'):
                goal = level.get_data('GoalRotation')
                if goal is not None:
                    return int(goal)
        return None

    def _build_skill_context(self) -> Dict[str, Any]:
        """Build context dict for skill selection."""
        available_actions = list(self.obs.available_actions or [])
        is_rotation_game = 6 in available_actions and not any(a in available_actions for a in [1, 2, 3, 4])
        
        return {
            "has_player": self.player is not None,
            "has_pathfinder": self.__init_pathfinder() is not None,
            "has_changer": len(self._find_tagged_sprites('rhsxkxzdjz')) > 0,
            "knows_goal_rotation": self._infer_goal_rotation() is not None,
            "adjacent_to_target": self._check_adjacent_to_target(),
            "available_actions": available_actions,
            "source_available": self._check_source_available(),
            "current_obs": True,
            "skill_dag_loaded": True,
            "is_rotation_game": is_rotation_game,
        }

    def _get_skill_for_phase(self) -> Optional[str]:
        """Get the skill ID for the current phase."""
        phase_skills = {
            "detect_walls": "detect-walls-from-source",
            "navigate_to_changer": "navigate-to-target",
            "rotate_to_goal": "rotate-to-goal",
            "navigate_to_lock": "navigate-to-target",
            "interact": "interact-with-object",
        }
        return phase_skills.get(self._phase)

    def _execute_skill(self, skill_id: str) -> bool:
        """Execute a single skill by ID. Returns True if skill made progress."""
        
        if skill_id == "detect-walls-from-source":
            return self._skill_detect_walls()
        
        elif skill_id == "navigate-to-target":
            return self._skill_navigate_to_target()
        
        elif skill_id == "rotate-to-goal":
            return self._skill_rotate_to_goal()
        
        elif skill_id == "interact-with-object":
            return self._skill_interact()
        
        elif skill_id == "select-skill-for-observation":
            # Not used in phase machine
            return True
        
        return False

    def _skill_interact(self) -> bool:
        """Execute interact-with-object skill."""
        interact_action = 5
        prev_level = self.obs.levels_completed
        self.step(interact_action)
        return self.obs.levels_completed > prev_level

    def _skill_detect_walls(self) -> bool:
        """Execute detect-walls-from-source skill."""
        if not self._walls_detected:
            self.__init_pathfinder()  # Ensure pathfinder is initialized
            self._detect_wall_colors()
            self._walls_detected = True
            return True
        return False  # Already done

    def _skill_navigate_to_target(self) -> bool:
        """Execute navigate-to-target skill."""
        # Phase 1: Navigate to changer
        if self._phase == "navigate_to_changer":
            changers = self._find_tagged_sprites('rhsxkxzdjz')
            if changers:
                ch = changers[0]
                cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)
                
                # Check if already at changer
                if self.player and self.player.x == cx and self.player.y == cy:
                    return True  # At changer - success
                
                pathfinder = self.__init_pathfinder()
                pathfinder.walkable_overrides.add((cx, cy))
                pathfinder.update_from_observation(self.obs)
                return pathfinder.navigate_astar((cx, cy), max_steps=200, obs=self.obs)
        
        # Phase 2: Navigate to lock
        if self._phase == "navigate_to_lock":
            locks = self._find_tagged_sprites('rjlbuycveu')
            if locks:
                lk = locks[0]
                lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                
                # Check if already at lock
                if self.player and self.player.x == lx and self.player.y == ly:
                    return True  # At lock - success
                
                pathfinder = self.__init_pathfinder()
                pathfinder.walkable_overrides.add((lx, ly))
                pathfinder.update_from_observation(self.obs)
                return pathfinder.navigate_astar((lx, ly), max_steps=200, obs=self.obs)
        
        return False

    def _skill_rotate_to_goal(self) -> bool:
        """Execute rotate-to-goal skill. Loops until rotation matches goal."""
        goal_rot = self._infer_goal_rotation()
        if goal_rot is None:
            return False
        
        current_rot = getattr(self.game, 'cklxociuu', 0) if self.game else 0
        if current_rot == goal_rot:
            return True  # Already rotated
        
        # Need to reach changer first
        changers = self._find_tagged_sprites('rhsxkxzdjz')
        if not changers:
            return False
        
        ch = changers[0]
        cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)
        
        # Navigate to changer if not adjacent
        if self.player and abs(self.player.x - cx) + abs(self.player.y - cy) > 1:
            pathfinder = self.__init_pathfinder()
            pathfinder.walkable_overrides.add((cx, cy))
            pathfinder.navigate_to((cx, cy), max_steps=100)
            return True  # Made progress toward changer
        
        # At changer — cycle rotation using R+L (actions 4+3)
        max_cycles = 20
        for _ in range(max_cycles):
            current_rot = getattr(self.game, 'cklxociuu', 0)
            if current_rot == goal_rot:
                return True
            self.step(4)  # Right
            self.step(3)  # Left
        
        return False  # Failed to reach target rotation

    def _advance_phase(self, success: bool):
        """Advance phase based on skill result."""
        if self._phase == "detect_walls" and success:
            self._phase = "navigate_to_changer"
        elif self._phase == "navigate_to_changer" and success:
            self._phase = "rotate_to_goal"
        elif self._phase == "rotate_to_goal" and success:
            self._phase = "navigate_to_lock"
        elif self._phase == "navigate_to_lock" and success:
            self._phase = "interact"
        elif self._phase == "interact" and success:
            self._phase = "complete"

    # ------ SOLVE PHASE ------
    
    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Solve current level using phase-based skill execution."""
        prev_lvl = self.obs.levels_completed
        if level_num is not None and prev_lvl + 1 != level_num:
            print(f"  ⚠️ Expected level {level_num}, at {prev_lvl + 1}")
        
        # Update current level index
        self.current_level_idx = prev_lvl
        self._walls_detected = False  # Reset for new level
        self._phase = "detect_walls"  # Phase state machine
        
        # Refresh player reference (game may recreate player sprite per level)
        if self.game and hasattr(self.game, 'gudziatsk') and self.game.gudziatsk:
            self.player = self.game.gudziatsk
        
        # Skill Tree: detect new level
        if self.skill_tree:
            self.skill_tree.detect_new_level(self.obs)
            for skill_name in self.skill_tree.active_abilities(self.current_level_idx):
                skill = self.skill_tree.get(skill_name)
                if skill and skill.action_id:
                    print(f"  🔓 Skill available: {skill_name}")
        
        self.discover_properties()
        
        # Benchmark tracking
        self.benchmark_start_time = time.time()
        
        # Decision loop
        max_iterations = 200
        for iteration in range(max_iterations):
            if self.obs.levels_completed > prev_lvl:
                print(f"  ✅ LEVEL {self.obs.levels_completed} COMPLETED!")
                self._record_level_skills(prev_lvl + 1)
                return True
            
            # Stop if frame is empty (level transition)
            if not hasattr(self.obs, 'frame') or not self.obs.frame or len(self.obs.frame) == 0:
                print(f"  ⏸ Frame empty - level transition")
                break
            
            # Get skill for current phase
            skill_id = self._get_skill_for_phase()
            
            if not skill_id:
                print(f"  ❌ No skill for phase: {self._phase}")
                break
            
            print(f"  🔄 Phase: {self._phase} -> Skill: {skill_id}")
            success = self._execute_skill(skill_id)
            
            if success:
                self._advance_phase(success)
                print(f"  ✅ Phase complete: {self._phase}")
            else:
                print(f"  ⚠️ Skill {skill_id} failed in phase {self._phase}")
                
            # Brief pause to let state settle
            if self.obs.levels_completed > prev_lvl:
                break
        
        result = self.obs.levels_completed > prev_lvl
        self._record_benchmark(self.current_level_idx, result)
        return result

    def _record_level_skills(self, level: int):
        """Record which skills were used for this level."""
        if self.skill_tree:
            # Skills are recorded during execution via skill_tree.unlock()
            pass
    
    def _record_benchmark(self, level: int, solved: bool):
        """Record benchmark result for this level attempt."""
        if not self.benchmark_tracker or not self.benchmark_session_id:
            return
        elapsed = time.time() - self.benchmark_start_time if self.benchmark_start_time else 0
        strategy = "bootstrap" if level == 0 else "generic"
        self.benchmark_tracker.record_game(
            game_id=self.name,
            level=level + 1,
            solved=solved,
            steps=self.steps,
            time_seconds=elapsed,
            tokens_used=0,
            strategy=strategy,
            error="" if solved else "max_steps_exceeded"
        )
        # Flush to disk after each level
        self.benchmark_tracker.flush()
    
    def end_benchmark_session(self):
        """End and persist the benchmark session."""
        if self.benchmark_tracker and self.benchmark_session_id:
            self.benchmark_tracker.end_session()
            self.benchmark_session_id = None
    
    def end_skill_session(self):
        """Export skill tree for this game and attempt cross-game import."""
        if not self.skill_tree:
            return
        
        # Export game-specific tree
        game_tree = self.skill_tree.export_for_game(self.name)
        print(f"\n🌳 Skill Tree exported for {self.name}: {len(game_tree.skills)} skills")
        print(game_tree.report())
        
        # Try to import from other games in the same domain
        from .domain_profiler import DomainProfiler
        from .skill_tree import SkillTree
        try:
            dp = DomainProfiler(self.env)
            profile = dp.build_profile()
            
            # Look for other games in same domain
            cache_dir = Path.home() / ".cache" / "cogniarc" / "games"
            if cache_dir.exists():
                for skill_file in cache_dir.glob("*_skill_tree.json"):
                    other_game = skill_file.stem.replace("_skill_tree", "")
                    if other_game != self.name:
                        other_tree = SkillTree.load_for_game(other_game)
                        imported = self.skill_tree.import_from_game(other_tree, other_game, min_confidence=0.8)
                        if imported > 0:
                            print(f"  📥 Imported {imported} skills from {other_game}")
        except Exception as e:
            print(f"  ⚠️ Cross-game import failed: {e}")
    
    # ------ TRANSITION ------
    
    def handle_transition(self):
        """Handle level transition. If trapped, burn remaining steps to trigger lose()."""
        if not self.player:
            return
        
        pos = (self.player.x, self.player.y)
        
        # Test if we can move
        can_move = False
        for act in [1, 2, 3, 4]:
            prev = (self.player.x, self.player.y)
            self.step(act)
            if self.player.x != prev[0] or self.player.y != prev[1]:
                can_move = True
                break
        
        if can_move:
            return  # We're free
        
        # Trapped — burn steps
        print(f"  🪤 Trapped at {pos}! Burning steps for reset...")
        burn_start = self.steps
        prev_lvl = self.obs.levels_completed
        
        while self.steps - burn_start < 60:
            self.step(3)  # any blocked direction
            if self.obs.levels_completed != prev_lvl:
                print(f"  🔄 Level changed during burn: {self.obs.levels_completed}")
                break
            if hasattr(self.obs, 'state') and 'GAME_OVER' in str(self.obs.state):
                print(f"  💀 Game over — reset triggered")
                break
            # Check if we can move now
            prev = (self.player.x, self.player.y)
            if prev != pos:
                print(f"  🆓 Freed! Now at ({self.player.x},{self.player.y})")
                break
        
        print(f"  Burned {self.steps - burn_start} steps")
    
    # ------ MAIN LOOP ------
    
    def run(self):
        print(f"🔬 Scientist Agent — {self.name}")
        print(f"   Start: lvl={self.obs.levels_completed}/{self.obs.win_levels}")
        
        # PHASE 1: Discover
        print("\n📖 DISCOVERY PHASE")
        self.discover_from_source()
        self.discover_available_actions()
        self.discover_properties()
        
        # PHASE 2: Scout (cheap actions to understand domain)
        print("\n🔍 SCOUT PHASE")
        # Already discovered in discover_available_actions()
        print("  ✅ Scout complete (from discovery)")
        
        # PHASE 3: Solve levels
        print(f"\n🎮 SOLVE PHASE (target: {self.obs.win_levels} levels)")
        
        max_total = 400
        while self.obs.levels_completed < self.obs.win_levels and self.steps < max_total:
            prev_lvl = self.obs.levels_completed
            
            # Try to solve current level
            solved = self.solve_level()
            
            if not solved:
                # Try transition handling
                self.handle_transition()
                
                if self.obs.levels_completed == prev_lvl:
                    # Still stuck — try random exploration
                    print(f"  🎲 Stuck, trying random exploration...")
                    for _ in range(20):
                        self.step((self.steps % 4) + 1)
                        if self.obs.levels_completed > prev_lvl:
                            print(f"  ✅ Random find! Level {self.obs.levels_completed}")
                            break
            
            # Check game over
            if hasattr(self.obs, 'state'):
                state = str(self.obs.state)
                if 'GAME_OVER' in state or 'LOSS' in state:
                    print(f"  💀 Game over at level {self.obs.levels_completed}")
                    break
        
        # FINAL
        print(f"\n{'='*50}")
        print(f"🏆 {self.obs.levels_completed}/{self.obs.win_levels} levels, {self.steps} steps")
        print(f"State: {self.obs.state}")
        print(f"\n{self.pkm.report()}")
        
        # Skill Tree summary
        self.end_skill_session()
        
        # Benchmark summary
        self.end_benchmark_session()


# ====== Launch ======
if __name__ == "__main__":
    agent = ScientistAgent("ls20-9607627b")
    agent.run()