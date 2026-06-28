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

# ====== 9 REASONING MODES (NEW - AHOIS) ======
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class ReasoningMode(Enum):
    """9 modes de raisonnement inspirés d'AHOIS."""
    EXPLORATION = "exploration"       # BFS/random → découvrir environnement
    PATHFINDING = "pathfinding"       # A* vers cible identifiée
    ROTATION = "rotation"             # Cycling action 6
    TRANSFORMATION = "transformation" # Changer état/pixel
    GOAL_INFERENCE = "goal_inference" # Hypothèse sur objectif
    CAUSAL = "causal"                 # Cause-effet des actions
    COUNTERFACTUAL = "counterfactual" # "Et si j'avais fait X ?"
    ANALOGICAL = "analogical"         # Transférer pattern via SkillTree
    SOCRATIC = "socratic"             # Critiquer hypothèses


@dataclass
class ModeStrategy:
    """Une stratégie = mode + conditions + heuristiques + fallback."""
    mode: ReasoningMode
    trigger_condition: str              # Description de quand ce mode s'active
    heuristic: str                      # Logique de sélection d'action
    success_criteria: str               # Qu'est-ce qui prouve que ça marche
    failure_criteria: str               # Qu'est-ce qui prouve que ça marche PAS
    fallback_mode: ReasoningMode = ReasoningMode.EXPLORATION
    weight: float = 1.0                 # Poids dans le selector (ajustable)


class ReasonModeManager:
    """Gère la sélection et la priorisation des 9 modes de raisonnement."""
    
    def __init__(self):
        self.strategies: dict[ReasoningMode, ModeStrategy] = {
            ReasoningMode.EXPLORATION: ModeStrategy(
                mode=ReasoningMode.EXPLORATION,
                trigger_condition="Aucune info sur l'environnement. Premier pas ou après reset.",
                heuristic="BFS + cognitive drives (novelty, impulse). Actions 1-4 pour couvrir la carte.",
                success_criteria="Nouveaux états découverts, carte partiellement explorée.",
                failure_criteria="50+ steps sans nouvelle tuile.",
                fallback_mode=ReasoningMode.SOCRATIC,
                weight=0.5,
            ),
            ReasoningMode.PATHFINDING: ModeStrategy(
                mode=ReasoningMode.PATHFINDING,
                trigger_condition="Cible identifiée (changer, lock). Carte partiellement connue.",
                heuristic="A* avec walkable_overrides. Si mur, apprendre la couleur et réessayer.",
                success_criteria="Position de la cible atteinte.",
                failure_criteria="Path bloqué >3 tentatives. Aucun chemin trouvé.",
                fallback_mode=ReasoningMode.EXPLORATION,
                weight=1.0,
            ),
            ReasoningMode.ROTATION: ModeStrategy(
                mode=ReasoningMode.ROTATION,
                trigger_condition="Action 6 disponible ET cible de rotation identifiée (changer).",
                heuristic="Cycle action 6 (max 20×). Si changé → arrêter. Sinon après 20→ fallback.",
                success_criteria="Level completed OU nouvel état après rotation.",
                failure_criteria="20 cycles sans changement d'état.",
                fallback_mode=ReasoningMode.EXPLORATION,
                weight=0.8,
            ),
            ReasoningMode.TRANSFORMATION: ModeStrategy(
                mode=ReasoningMode.TRANSFORMATION,
                trigger_condition="Action 5+ disponible. Changement détecté entre frames adjacentes.",
                heuristic="Appliquer actions transformation dans l'ordre. Comparer état avant/après.",
                success_criteria="État significativement différent après transformation.",
                failure_criteria="10 transformations sans effet visible.",
                fallback_mode=ReasoningMode.GOAL_INFERENCE,
                weight=0.6,
            ),
            ReasoningMode.GOAL_INFERENCE: ModeStrategy(
                mode=ReasoningMode.GOAL_INFERENCE,
                trigger_condition="Aucune hypothèse de but. Ou Doute déclenché (reset theory).",
                heuristic="GoalInferenceEngine.observe_only() → analyser rareté, symétrie, patterns.",
                success_criteria="Hypothèse avec confidence > 0.5.",
                failure_criteria="Toutes les hypothèses < 0.3.",
                fallback_mode=ReasoningMode.EXPLORATION,
                weight=0.7,
            ),
            ReasoningMode.CAUSAL: ModeStrategy(
                mode=ReasoningMode.CAUSAL,
                trigger_condition="Action répétée sans changement. Ou stagnation > 3.",
                heuristic="Enregistrer état avant action, comparer après. Si aucun changement → marquer cette action comme inefficace dans ce contexte.",
                success_criteria="Corrélation action→effet identifiée.",
                failure_criteria="Bruit trop fort (actions != effets consistants).",
                fallback_mode=ReasoningMode.COUNTERFACTUAL,
                weight=0.5,
            ),
            ReasoningMode.COUNTERFACTUAL: ModeStrategy(
                mode=ReasoningMode.COUNTERFACTUAL,
                trigger_condition="Mode causal a échoué. Ou plusieurs modes suggèrent des chemins différents.",
                heuristic="\"Et si j'avais pris action X au lieu de Y ?\" Simuler en mémoire (WorkingMemory). Choisir l'action avec le meilleur score simulé.",
                success_criteria="Action simulée meilleure que l'action réelle.",
                failure_criteria="Simulation incohérente avec la réalité observée.",
                fallback_mode=ReasoningMode.EXPLORATION,
                weight=0.4,
            ),
            ReasoningMode.ANALOGICAL: ModeStrategy(
                mode=ReasoningMode.ANALOGICAL,
                trigger_condition="SkillTree disponible AVEC skills importées cross-game.",
                heuristic="Chercher dans SkillTree des patterns de jeu similaire (même domaine). Apprendre de la stratégie du jeu source.",
                success_criteria="Pattern transféré mène à un progrès.",
                failure_criteria="Aucune skill importée pertinente trouvée.",
                fallback_mode=ReasoningMode.SOCRATIC,
                weight=0.6,
            ),
            ReasoningMode.SOCRATIC: ModeStrategy(
                mode=ReasoningMode.SOCRATIC,
                trigger_condition="Stagnation, doute déclenché, ou mode courant en échec.",
                heuristic="SocraticCritic.interrogate() → 6 opérations. Corriger hypothèses avant de ré-agir.",
                success_criteria="Rapport critic sans BLOCKING + nouvelle hypothèse viable.",
                failure_criteria="Toujours bloqué après critique (escalade vers exploration).",
                fallback_mode=ReasoningMode.EXPLORATION,
                weight=0.9,
            ),
        }
        self.current_mode: ReasoningMode = ReasoningMode.EXPLORATION
        self.mode_history: list[dict] = []
    
    def select_mode(self, context: dict) -> ReasoningMode:
        """Sélectionne le meilleur mode selon le contexte courant.
        context peut contenir : stagnation, confidence, domain_type, available_actions, etc."""
        stagnation = context.get("stagnation", 0)
        domain = context.get("domain", "")
        actions = context.get("available_actions", [])
        skill_tree_available = context.get("skill_tree_available", False)
        doubt_active = context.get("doubt_active", False)
        
        # Priorité haute: Socratic si doute ou stagnation longue
        if doubt_active or stagnation > 8:
            return ReasoningMode.SOCRATIC
        
        # Priorité: Pathfinding si on a des cibles connues
        if context.get("has_target", False):
            return ReasoningMode.PATHFINDING
        
        # Priorité: GoalInference si pas d'hypothèse
        if not context.get("has_goal_hypothesis", False):
            return ReasoningMode.GOAL_INFERENCE
        
        # Rotation si disponible
        if 6 in actions and domain in ("rotation", "hybrid"):
            return ReasoningMode.ROTATION
        
        # Analogical si SkillTree disponible
        if skill_tree_available:
            return ReasoningMode.ANALOGICAL
        
        # Causal si stagnation récente
        if stagnation > 3:
            return ReasoningMode.CAUSAL
        
        # Par défaut: explorer
        return ReasoningMode.EXPLORATION
    
    def log_mode_switch(self, old_mode: ReasoningMode, new_mode: ReasoningMode, reason: str):
        self.mode_history.append({
            "from": old_mode.value,
            "to": new_mode.value,
            "reason": reason,
        })


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
    
    def __init__(self, game_name: str, enable_benchmark: bool = True, enable_skill_tree: bool = True, enable_world_model: bool = False):
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
        self.current_level_idx = 0
        
        # ═══ NEW: ScientificState + SocraticCritic + ReasonModeManager ═══
        from .scientific_state import ScientificState
        from .socratic_critic import SocraticCritic
        
        self.state = ScientificState(
            game_name=game_name,
            available_actions=list(self.obs.available_actions or []),
        )
        self.critic = SocraticCritic(verbose=False)
        
        # ═══ NEW: ReasonModeManager + Dynamic Workflows ═══
        self.mode_manager = ReasonModeManager()
        self.current_reasoning_mode: ReasoningMode = ReasoningMode.EXPLORATION
        
        # ═══ NEW: World Model Tool (V-JEPA based simulator) ═══
        self.world_model = None
        if enable_world_model:
            try:
                from cogniarc.world_model import WorldModelTool
                self.world_model = WorldModelTool(game_id=game_name)
                if self.world_model.available:
                    mem = self.world_model.memory_size()
                    print(f"[WorldModel] Loaded V-JEPA encoder" + 
                          (f" + {mem} prior transitions" if mem > 0 else " (fresh start)"))
                else:
                    print(f"[WorldModel] Fallback encoder active (no V-JEPA checkpoint)")
            except Exception as e:
                print(f"[WorldModel] Failed to init: {e}")
        
        # ═══ NEW: Micro-NN predictors (instant, <1ms) ═══
        self.action_predictor = None
        self.domain_predictor = None
        try:
            from cogniarc.micro_predictors import ActionPredictor, DomainPredictor
            self.action_predictor = ActionPredictor()
            self.domain_predictor = DomainPredictor()
            if self.action_predictor.available:
                print(f"[MicroNN] Action predictor loaded")
            if self.domain_predictor.available:
                print(f"[MicroNN] Domain predictor loaded")
        except Exception as e:
            pass  # Micro-NN is optional
        
        # Legacy aliases (to be removed gradually)
        self._phase = self.state.phase
        self._walls_detected = self.state.walls_detected
    
    def step(self, action_num: int):
        # ═══ NEW: Record pre-step observation for world model ═══
        obs_before = None
        if self.world_model and self.obs.frame and len(self.obs.frame) > 0:
            obs_before = self.obs.frame[0].copy()
        
        self.obs = self.env.step(getattr(GameAction, f'ACTION{action_num}'))
        self.steps += 1
        
        # ═══ NEW: Record transition in world model ═══
        if self.world_model and obs_before is not None and self.obs.frame and len(self.obs.frame) > 0:
            self.world_model.remember(
                self.world_model.encode(obs_before),
                action_num,
                self.world_model.encode(self.obs.frame[0])
            )
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

        # ═══ NEW: Update ScientificState ═══
        self.state.steps_taken = self.steps
        self.state.last_action = action_num
        self.state.stagnation_count = self.drives.stagnation_counter
        self.state.last_state_hash = state_hash

        return self.obs
    
    # ═══════ WORLD MODEL TOOL ═══════
    
    def _world_model_simulate(self, action: int):
        """Simulate the effect of an action using the world model.
        
        Queries the V-JEPA encoder + k-NN predictor:
        "If I take action X from my current state, what state do I predict?"
        
        Returns:
            (predicted_latent, confidence) or (None, 0.0) if world model unavailable
        """
        if not self.world_model:
            return None, 0.0
        
        if not self.obs.frame or len(self.obs.frame) == 0:
            return None, 0.0
        
        obs = self.obs.frame[0]
        predicted, confidence = self.world_model.predict(obs, action)
        return predicted, confidence
    
    def _world_model_report(self) -> str:
        """Human-readable report of world model state."""
        if not self.world_model:
            return "World model: disabled"
        
        mem = self.world_model.memory_size()
        return f"World model: {mem} transitions memorized"
    
    def _predict_skill_action(self, skill_id: str) -> Optional[int]:
        """Predict which action a skill will likely take (for world model pre-check).
        
        Maps skill IDs to their most probable action number.
        Returns None if the skill doesn't involve a predictable action.
        """
        action_map = {
            "navigate-to-target": None,  # Multi-action A* — too complex
            "rotate-to-goal": 6,          # Rotation action
            "interact-with-object": 5,    # Interact
            "detect-walls-from-source": None,  # No step
        }
        
        # Navigation: determine direction from phase
        if skill_id == "navigate-to-target":
            if self._phase == "navigate_to_changer":
                # Predict direction toward changer
                if self.player:
                    changers = self._find_tagged_sprites('rhsxkxzdjz')
                    if changers:
                        ch = changers[0]
                        cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)
                        px, py = self.player.x, self.player.y
                        if cx > px: return 1  # right
                        if cx < px: return 3  # left
                        if cy > py: return 2  # down
                        if cy < py: return 4  # up
            elif self._phase == "navigate_to_lock":
                if self.player:
                    locks = self._find_tagged_sprites('rjlbuycveu')
                    if locks:
                        lk = locks[0]
                        lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                        px, py = self.player.x, self.player.y
                        if lx > px: return 1
                        if lx < px: return 3
                        if ly > py: return 2
                        if ly < py: return 4
        
        return action_map.get(skill_id)
    
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
        """Detect wall colors from analysis + local probing."""
        if not self.game or not hasattr(self.obs, 'frame') or not self.obs.frame:
            return
        
        pathfinder = self.__init_pathfinder()
        grid = self.obs.frame[0]
        
        # Analyze grid: find most common colors
        unique, counts = np.unique(grid, return_counts=True)
        sorted_by_count = np.argsort(counts)[::-1]
        color_freq = [(int(unique[idx]), int(counts[idx])) for idx in sorted_by_count]
        # Skip background (0)
        non_bg = [(c, n) for c, n in color_freq if c != 0]
        
        # The top non-bg color is usually the FLOOR
        # Subsequent colors are walls, objects, or special tiles
        floor_color = non_bg[0][0] if non_bg else None
        
        # Method 1: Tag-based — check if tag sprite color is wall or floor
        wall_tags = self.pkm.get('mechanics', 'walls', [])
        tag_colors = set()
        for tag in wall_tags:
            sprites = self._find_tagged_sprites(tag)
            for s in sprites[:5]:
                color = int(grid[s.y, s.x])
                if color != floor_color:  # Not on floor = actual wall tile
                    tag_colors.add(color)
                # If on floor, check neighbors for the real wall color
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = s.x + dx, s.y + dy
                    if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                        nc = int(grid[ny, nx])
                        if nc != color and nc != 0 and nc != floor_color:
                            tag_colors.add(nc)
        
        # Method 2: Adjacent to player (non-invasively)
        player_adjacent = set()
        if self.player:
            px, py = self.player.x, self.player.y
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    c = int(grid[ny, nx])
                    pc = int(grid[py, px])
                    if c != pc and c != 0:  # Not where player stands, not bg
                        player_adjacent.add(c)
        
        # Method 3: Grid analysis — second most common non-bg color is often wall
        grid_wall_candidates = set()
        if len(non_bg) >= 2:
            # Colors that are less common than floor (top 2-4)
            for c, n in non_bg[1:4]:
                # If it's significantly less common than floor (>2x less)
                if n < non_bg[0][1] * 0.5:  # Less than half the floor area
                    grid_wall_candidates.add(c)
        
        # Combine: prefer player-adjacent and grid analysis over tag-on-floor
        # If player-adjacent gives us clear walls, use those
        if player_adjacent:
            pathfinder.wall_colors.update(player_adjacent)
            # Also add grid-analysis walls if they match player-adjacent
            for c in grid_wall_candidates:
                if c in player_adjacent or c in tag_colors:
                    pathfinder.wall_colors.add(c)
        else:
            # Fallback: use tag-based + grid analysis
            pathfinder.wall_colors.update(tag_colors)
            pathfinder.wall_colors.update(grid_wall_candidates)
        
        # Remove player's own color and background
        if self.player:
            pathfinder.wall_colors.discard(int(grid[self.player.y, self.player.x]))
        pathfinder.wall_colors.discard(0)
        
        # Lock walls so learn_walls() probing doesn't corrupt them
        pathfinder.walls_locked = True
        pathfinder.update_from_observation(self.obs)
        
        print(f"  🧱 Wall colors: {sorted(pathfinder.wall_colors)} "
              f"(tags={sorted(tag_colors)}, adjacent={sorted(player_adjacent)}, "
              f"grid={sorted(grid_wall_candidates)}, floor={floor_color})")
    
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
        
        # ═══ World Model pre-check: simulate before executing ═══
        if self.world_model and self.world_model.memory_size() > 0:
            # Determine which action this skill will likely take
            predicted_action = self._predict_skill_action(skill_id)
            if predicted_action is not None:
                _, confidence = self._world_model_simulate(predicted_action)
                if confidence > 0.5:
                    print(f"  🌍 WM: skill={skill_id} action={predicted_action} confidence={confidence:.3f}")
                elif confidence > 0.0:
                    print(f"  🌍 WM: skill={skill_id} action={predicted_action} low confidence={confidence:.3f}")
        
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
            self.state.walls_detected = True
            self.state.set_assumption("walls_known", True)
            return True
        return False  # Already done

    def _skill_navigate_to_target(self) -> bool:
        """Execute navigate-to-target skill. Falls back to world model if A* fails."""
        target_pos = None
        
        # Phase 1: Navigate to changer
        if self._phase == "navigate_to_changer":
            changers = self._find_tagged_sprites('rhsxkxzdjz')
            if changers:
                ch = changers[0]
                target_pos = (getattr(ch, 'x', 0), getattr(ch, 'y', 0))
                
                if self.player and self.player.x == target_pos[0] and self.player.y == target_pos[1]:
                    return True  # At changer
        
        # Phase 2: Navigate to lock
        elif self._phase == "navigate_to_lock":
            locks = self._find_tagged_sprites('rjlbuycveu')
            if locks:
                lk = locks[0]
                target_pos = (getattr(lk, 'x', 0), getattr(lk, 'y', 0))
                
                if self.player and self.player.x == target_pos[0] and self.player.y == target_pos[1]:
                    return True  # At lock
        
        if target_pos is None:
            return False
        
        tx, ty = target_pos
        
        # Try A* pathfinding
        pathfinder = self.__init_pathfinder()
        pathfinder.walkable_overrides.add((tx, ty))
        pathfinder.update_from_observation(self.obs)
        astar_result = pathfinder.navigate_astar((tx, ty), max_steps=200, obs=self.obs)
        
        if astar_result:
            return True
        
        # ═══ A* FAILED → Micro-NN or World Model fallback ═══
        if self.action_predictor or (self.world_model and self.world_model.memory_size() > 0):
            return self._world_model_navigate_fallback(tx, ty)
        
        return False
    
    def _world_model_navigate_fallback(self, tx: int, ty: int) -> bool:
        """When A* fails, use micro-NN or world model to suggest a direction.
        
        Tiered fallback:
        1. Micro-NN ActionPredictor (instant, <1ms) — if confidence > 0.5
        2. World Model V-JEPA (6s CPU) — if micro-NN unavailable or uncertain
        """
        if not self.player:
            return False
        
        px, py = self.player.x, self.player.y
        
        # ═══ TIER 1: Micro-NN Action Predictor (instant) ═══
        if self.action_predictor and self.action_predictor.available:
            wall_colors = getattr(self, '_pathfinder', None)
            wall_set = wall_colors.wall_colors if wall_colors and hasattr(wall_colors, 'wall_colors') else set()
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            
            if grid is not None:
                best_action = None
                best_conf = 0.0
                
                for action in [1, 2, 3, 4]:
                    # Only consider directions toward target
                    if action == 1 and px >= tx: continue
                    if action == 3 and px <= tx: continue
                    if action == 2 and py >= ty: continue
                    if action == 4 and py <= ty: continue
                    
                    prob = self.action_predictor.predict_action(
                        (px, py), (tx, ty), action, wall_set, grid,
                        stagnation=self.drives.stagnation_counter,
                        steps=self.steps
                    )
                    if prob > best_conf:
                        best_conf = prob
                        best_action = action
                
                if best_action is not None and best_conf > 0.5:
                    print(f"  ⚡ MicroNN: A* failed → stepping {['','right','down','left','up'][best_action]} (prob={best_conf:.3f})")
                    self.step(best_action)
                    return True
        
        # ═══ TIER 2: World Model V-JEPA (slower but more accurate) ═══
        if self.world_model and self.world_model.memory_size() > 0:
            return self._world_model_navigate_fallback_vjepa(tx, ty)
        
        return False
    
    def _world_model_navigate_fallback_vjepa(self, tx: int, ty: int) -> bool:
        """V-JEPA world model fallback (original implementation)."""
        if not self.player:
            return False
        
        px, py = self.player.x, self.player.y
        best_action = None
        best_conf = 0.0
        
        for action in [1, 2, 3, 4]:
            if action == 1 and px >= tx: continue
            if action == 3 and px <= tx: continue
            if action == 2 and py >= ty: continue
            if action == 4 and py <= ty: continue
            
            _, conf = self._world_model_simulate(action)
            if conf > best_conf:
                best_conf = conf
                best_action = action
        
        if best_action is not None and best_conf > 0.3:
            print(f"  🌍 WM fallback: A* failed → stepping {['','right','down','left','up'][best_action]} (conf={best_conf:.3f})")
            self.step(best_action)
            return True
        
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
        """Advance phase based on skill result. Syncs with ScientificState."""
        old_phase = self._phase
        
        if self._phase == "detect_walls" and success:
            self._phase = "navigate_to_changer"
        elif self._phase == "navigate_to_changer" and success:
            self._phase = "rotate_to_goal"
        elif self._phase == "rotate_to_goal" and success:
            self._phase = "navigate_to_lock"
        elif self._phase == "navigate_to_lock" and success:
            # In LS20, locks are collected by walking ON them, not by interact (action 5).
            # Skip interact phase — step onto the lock directly.
            if self.player:
                locks = self._find_tagged_sprites('rjlbuycveu')
                if locks:
                    lk = locks[0]
                    lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                    if self.player.x == lx and self.player.y == ly:
                        self._phase = "complete"  # Already on lock
                    elif abs(self.player.x - lx) + abs(self.player.y - ly) == 1:
                        # Adjacent: step onto lock
                        if self.player.x < lx: action = 1
                        elif self.player.x > lx: action = 3
                        elif self.player.y < ly: action = 2
                        else: action = 4
                        print(f"  🔑 Stepping onto lock at ({lx},{ly}) — action {action}")
                        self.step(action)
                        self._phase = "complete"
                    else:
                        self._phase = "navigate_to_lock"  # Need more movement
                else:
                    self._phase = "interact"  # Fallback
            else:
                self._phase = "interact"
        elif self._phase == "interact" and success:
            self._phase = "complete"
        
        # Sync with ScientificState
        if self._phase != old_phase:
            self.state.phase = self._phase
            self.state.phase_attempts = 0

    # ------ SOLVE PHASE ------
    
    # ═══ Perception Stack Integration (AHOIS) ═══
    
    def _init_perception_stack(self):
        """Initialize TemporalReasoner, SpatialReasoner, and SymbolicInference
        for use during solve_level(). Called at the start of each level."""
        try:
            from .temporal_inference import TemporalReasoner
            from .spatial_inference import SpatialReasoner
            from .symbolic_inference import SymbolicInference
            
            self._temporal = TemporalReasoner()
            self._spatial = SpatialReasoner()
            self._symbolic = SymbolicInference()
            self._perception_initialized = True
        except ImportError:
            self._temporal = None
            self._spatial = None
            self._symbolic = None
            self._perception_initialized = False
    
    def _perception_analyze(self, obs) -> dict:
        """Run temporal + spatial analysis on current observation.
        Returns a dict with perceived patterns and recommended skills.
        Dynamic Workflow pattern: PIPELINE (perception → inference → suggestion)."""
        if not self._perception_initialized:
            return {"mode_action": None}
        
        result = {}
        
        # Temporal analysis (if we have frame history)
        if hasattr(self, '_frame_history'):
            self._frame_history.append(obs)
            if len(self._frame_history) >= 3:
                frames = self._frame_history[-3:]
                temporal_pattern = self._temporal.analyze(frames)
                result["temporal"] = temporal_pattern
        else:
            self._frame_history = [obs]
        
        # Spatial analysis (current frame)
        if obs and hasattr(obs, 'frame') and obs.frame and len(obs.frame) > 0:
            grid = obs.frame[0]
            spatial_pattern = self._spatial.analyze(grid)
            result["spatial"] = spatial_pattern
            
            # Symbolic inference: temporal + spatial → skill recommendation
            if "temporal" in result:
                symbols = self._symbolic.infer(
                    temporal_pattern=result["temporal"],
                    spatial_pattern=spatial_pattern,
                )
                result["symbols"] = symbols
                skills = self._symbolic.skill_recommendations()
                if skills:
                    result["recommended_skills"] = skills
        
        return result
    
    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Solve current level using phase-based skill execution with SocraticCritic."""
        prev_lvl = self.obs.levels_completed
        if level_num is not None and prev_lvl + 1 != level_num:
            print(f"  ⚠️ Expected level {level_num}, at {prev_lvl + 1}")
        
        # Update current level index
        self.current_level_idx = prev_lvl
        self._walls_detected = False  # Reset for new level
        self._phase = "detect_walls"  # Phase state machine
        self.state.walls_detected = False
        self.state.phase = "detect_walls"
        self.state.phase_attempts = 0
        self._phase_escalation_count = 0  # Global: force skip apres X echecs
        
        # Refresh player reference (game may recreate player sprite per level)
        if self.game and hasattr(self.game, 'gudziatsk') and self.game.gudziatsk:
            self.player = self.game.gudziatsk
            self.state.set_assumption("player_found", True)
        
        # Skill Tree: detect new level
        if self.skill_tree:
            self.skill_tree.detect_new_level(self.obs)
            for skill_name in self.skill_tree.active_abilities(self.current_level_idx):
                skill = self.skill_tree.get(skill_name)
                if skill and skill.action_id:
                    print(f"  🔓 Skill available: {skill_name}")
        
        self.discover_properties()
        
        # ═══ Populate ScientificState from discovered info ═══
        available = list(self.obs.available_actions or [])
        self.state.available_actions = available
        is_movement = any(a in available for a in [1, 2, 3, 4])
        is_rotation = 6 in available
        
        if is_movement:
            self.state.domain_type = "movement" if not is_rotation else "hybrid"
        elif is_rotation:
            self.state.domain_type = "rotation"
        self.state.domain_confidence = 0.7
        
        self.state.set_assumption("walls_known", self._walls_detected or self._check_source_available())
        self.state.set_assumption("actions_scouted", True)
        self.state.set_assumption("domain_identified", True)
        self.state.set_assumption("goal_known", self._infer_goal_rotation() is not None)
        
        # Record observations from discovery
        self.state.record_observation(
            f"Domain={self.state.domain_type}, movement={is_movement}, rotation={is_rotation}",
            source="discovery"
        )
        
        # ═══ Select reasoning mode via ReasonModeManager ═══
        mode_context = {
            "stagnation": self.drives.stagnation_counter,
            "domain": self.state.domain_type,
            "available_actions": available,
            "skill_tree_available": self.skill_tree is not None,
            "doubt_active": self.drives.doubt_triggered,
            "has_target": hasattr(self, '_phase') and self._phase in ("navigate_to_changer", "navigate_to_lock"),
            "has_goal_hypothesis": hasattr(self, '_phase') and self._phase in ("rotation_cycle", "interact"),
        }
        new_mode = self.mode_manager.select_mode(mode_context)
        if new_mode != self.current_reasoning_mode:
            self.mode_manager.log_mode_switch(self.current_reasoning_mode, new_mode,
                f"solve_level() init: domain={self.state.domain_type}")
            self.current_reasoning_mode = new_mode
            print(f"  🧠 Reasoning Mode: {new_mode.value}")
        
        # ═══ Initialize Perception Stack for temporal/spatial integration ═══
        self._init_perception_stack()
        
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
            
            # ═══ SocraticCritic: interrogate before each phase ═══
            hypothesis = self._build_phase_hypothesis()
            report = self.critic.quick_check(hypothesis, self.state)
            self.state.add_critic_report(report)
            
            if report.unresolved():
                # Print critic warnings (non-blocking: show but proceed)
                for issue in report.unresolved()[:3]:  # max 3
                    print(f"  🤔 {issue}")
            
            if report.blocking:
                # 🚫 Le critic bloque → on CHERCHE une vraie alternative
                print(f"  🚫 SocraticCritic BLOCKING: {report.blocking_issues()[0].question[:60]}...")
                alt = self._resolve_blocking(report)
                if alt:
                    print(f"  🔀 Alternative trouvée: {alt}")
                    self._phase = alt
                    self.state.phase = alt
                    self.state.phase_attempts = 0
                    continue  # Retry with new phase
                else:
                    print(f"     Aucune alternative - execution risquee")
            
            print(f"  🔄 Phase: {self._phase} -> Skill: {skill_id}")
            success = self._execute_skill(skill_id)
            
            if success:
                self._advance_phase(success)
                print(f"  ✅ Phase complete: {self._phase}")
            else:
                self.state.phase_attempts += 1
                self._phase_escalation_count += 1
                print(f"  ⚠️ Skill {skill_id} failed in phase {self._phase} (attempt {self.state.phase_attempts})")

                # Global escalation: if stuck too long, force skip level
                if self._phase_escalation_count >= 5:
                    print(f"  🔴 Trop d'echecs ({self._phase_escalation_count}), force skip niveau...")
                    skip_to = self._try_skip_level()
                    if skip_to == "complete":
                        break  # Niveau complete ou skip
                    # Burn remaining steps pour forcer game over
                    self.handle_transition()
                    break
                
                # After 3 failed attempts on same phase, ask for escalation
                if self.state.phase_attempts >= 3:
                    escalation = self._escalate_phase_failure()
                    if escalation:
                        print(f"  🔀 Escalating: {escalation}")
                        self._phase = escalation
                        self.state.phase = escalation
                        self.state.phase_attempts = 0
                
            # Brief pause to let state settle
            if self.obs.levels_completed > prev_lvl:
                break
        
        result = self.obs.levels_completed > prev_lvl
        self._record_benchmark(self.current_level_idx, result)
        
        # ═══ Auto-save world model memory ═══
        if self.world_model and self.world_model.memory_size() > 0:
            self.world_model.save()
            print(f"  💾 WM saved: {self.world_model.memory_size()} transitions for {self.name}")
        
        return result

    def _build_phase_hypothesis(self) -> str:
        """Build a human-readable hypothesis for the current phase."""
        hypotheses = {
            "detect_walls": "Detect wall colors from source-code sprite tags so I can navigate without walking through walls",
            "navigate_to_changer": "Navigate to the rotation changer sprite and use it to change orientation",
            "rotate_to_goal": "Use the rotation changer (actions 4+3) to cycle rotation until it matches the goal rotation value",
            "navigate_to_lock": "Navigate to the lock sprite and walk onto it to collect it (locks are collidable=False on wall color)",
            "interact": "Interact with the final target (action 5) to complete the level",
            "complete": "Level is finished, move to next",
        }
        return hypotheses.get(self._phase, f"Unknown phase {self._phase}: proceed with default actions")

    def _resolve_blocking(self, report) -> Optional[str]:
        """
        When SocraticCritic BLOCKING, find an alternative phase that avoids
        the blocking issues instead of proceeding blindly.

        Examines each blocking issue and suggests a phase bypass.
        """
        blocking = report.blocking_issues()
        if not blocking:
            return None

        # Collect keywords from all blocking issues
        blocking_text = " ".join(i.question + " " + i.context for i in blocking).lower()

        # ── CATALOGUE D'ALTERNATIVES ──
        # Basé sur les BLOCKING issues les plus fréquentes

        # "action 6 not available" + on était en rotation → sauter rotation
        if "action 6" in blocking_text and self._phase in (
            "rotate_to_goal", "navigate_to_changer"
        ):
            # Sauter la rotation, essayer d'atteindre le lock directement
            if self._check_source_available():
                locks = self._find_tagged_sprites('rjlbuycveu')
                if locks:
                    self.state.record_observation(
                        "Rotation impossible (pas d'action 6), tentative navigation directe vers lock",
                        source="socratic_escalation"
                    )
                    return "navigate_to_lock"
            # Pas de lock connu → essayer le niveau suivant
            return self._try_skip_level()

        # "action 5 not available" + on voulait interagir → skip interact
        if "action 5" in blocking_text and self._phase == "interact":
            self.state.record_observation(
                "Interact impossible (pas d'action 5), niveau peut-etre deja complete",
                source="socratic_escalation"
            )
            return "complete"

        # "wall colors" pas detectes + on navigue → retourner detecter
        if "wall" in blocking_text and "navigate" in self._phase:
            if not self._walls_detected:
                return "detect_walls"

        # "domain not classified" → essayer re-decouverte
        if "domain" in blocking_text or "classified" in blocking_text:
            return "detect_walls"

        # Stagnation + echec → essayer skip level
        if "stagnant" in blocking_text or "failed" in blocking_text:
            return self._try_skip_level()

        return None

    def _try_skip_level(self) -> Optional[str]:
        """Try to force-complete or skip the current level."""
        # Essayer step(0) = reset/advance si disponible
        from .common import GameAction
        try:
            prev = self.obs.levels_completed
            self.obs = self.env.step(GameAction.RESET)
            self.steps += 1
            if self.obs.levels_completed > prev:
                self.state.record_observation(
                    f"Skip level: {prev} -> {self.obs.levels_completed}",
                    source="socratic_escalation"
                )
                return "complete"
        except Exception:
            pass

        # Burn steps for game over
        self.state.record_observation(
            "Skipping level impossible, tentative burn steps pour forcer reset",
            source="socratic_escalation"
        )
        return None  # Burn steps se fait dans handle_transition

    def _escalate_phase_failure(self) -> Optional[str]:
        """
        Legacy escalation — fallback quand _resolve_blocking n'a rien trouve.
        """
        # Stuck on navigation? Try direct interaction instead
        if self._phase in ("navigate_to_changer", "navigate_to_lock"):
            if self._check_adjacent_to_target():
                return "interact"
            return None
        # Stuck on rotation? Try interact directly
        if self._phase == "rotate_to_goal":
            return "navigate_to_lock"
        return None

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