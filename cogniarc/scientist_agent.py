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

This module holds orchestration only (init, the step loop, solve_level, phase
resolution, benchmark/skill-session lifecycle, run). The previously
monolithic ScientistAgent (~1600 lines, ~50 methods) was split by concern
into mixins to make each part independently readable/testable:
  - MLTiersMixin     (scientist_agent_ml_tiers.py)  — world model, nano-LLM
  - DiscoveryMixin   (scientist_agent_discovery.py) — mechanics discovery
  - SkillsMixin      (scientist_agent_skills.py)    — skill execution
No behavior changed during the split — methods were relocated verbatim.
"""

# arc_agi / arcengine are only needed to actually drive a live game. Keep the
# import optional so the rest of the package (TemporalReasoner, SpatialReasoner,
# SocraticCritic, ...) stays importable without the ARC-AGI runtime installed.
try:
    import arc_agi
    from arcengine import GameAction
    ARC_RUNTIME_AVAILABLE = True
except ImportError as _arc_import_error:
    arc_agi = None
    GameAction = None
    ARC_RUNTIME_AVAILABLE = False
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

# Cognitive drives
from .cognitive_player import CognitiveDrives, hash_grid

# Optional physics world-model tools (Mode 10 / SIMULATION_PHYSIQUE). Guarded
# like benchmark_tracker above: a physics-tree issue must degrade gracefully,
# not take down the whole package (this exact block did exactly that until
# fixed — see world_model_physics/ rename + relative-import fixes).
try:
    from .world_model_physics.physics.tools.mass_gravity import MassProperties
    from .world_model_physics.physics.tools.momentum_inertia import MomentumAnalyzer
    from .world_model_physics.physics.tools.spatial_zoning import SpatialAnalyzer
    from .world_model_physics.physics.tools.scene_graph import SceneGraph
    from .world_model_physics.physics.tools.torque_experts import ExpertRegistry, build_default_registry
    from .world_model_physics.physics.tools.discrete_classifier import classify_per_body
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False

# Mixins (see module docstring)
from .scientist_agent_ml_tiers import MLTiersMixin
from .scientist_agent_discovery import DiscoveryMixin
from .scientist_agent_skills import SkillsMixin

# ====== 9 REASONING MODES (NEW - AHOIS) ======
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class ReasoningMode(Enum):
    """10 modes de raisonnement — AHOIS + simulation cognitive."""
    EXPLORATION = "exploration"       # BFS/random → découvrir environnement
    PATHFINDING = "pathfinding"       # A* → naviguer vers cible connue
    ROTATION = "rotation"             # Cycle rotation → atteindre orientation
    TRANSFORMATION = "transformation" # Appliquer transform connue
    GOAL_INFERENCE = "goal_inference" # Inférer objectif depuis observations
    CAUSAL = "causal"                 # Raisonnement cause→effet
    COUNTERFACTUAL = "counterfactual" # "Et si j'avais fait X au lieu de Y?"
    ANALOGICAL = "analogical"         # Transférer skill d'un autre jeu/niveau
    SOCRATIC = "socratic"             # Questionner hypothèses via SocraticCritic
    SIMULATION = "simulation"         # Mode 10: projeter état futur via physique


# ═══ Mode-driven decision functions ═══
# These were previously hardcoded constants (3, 5) in solve_level()'s failure
# handling, completely independent of `current_reasoning_mode` — the mode was
# selected, logged, and printed but never actually changed agent behavior.
# Pure functions (mode in, threshold out) so they're unit-testable without
# instantiating ScientistAgent (which requires a live arc_agi runtime).

# Modes where the agent has reason to actively doubt the current plan: escalate
# sooner instead of repeating a failing deterministic skill.
DOUBT_MODES = (ReasoningMode.COUNTERFACTUAL, ReasoningMode.SOCRATIC, ReasoningMode.SIMULATION)
# Modes pursuing a known deterministic target: tolerate more retries before
# giving up, since the skill is likely to eventually succeed (e.g. A* retry).
COMMIT_MODES = (ReasoningMode.PATHFINDING, ReasoningMode.ROTATION)


def phase_attempts_threshold(mode: ReasoningMode, base: int = 3) -> int:
    """How many failed attempts on a phase before triggering escalation."""
    if mode in DOUBT_MODES:
        return max(1, base - 1)
    if mode in COMMIT_MODES:
        return base + 1
    return base


def phase_escalation_threshold(mode: ReasoningMode, base: int = 5) -> int:
    """Global hard-skip threshold (force-skip the level after this many
    consecutive skill failures, regardless of which phase)."""
    if mode in DOUBT_MODES:
        return max(2, base - 2)
    return base


def reconcile_perception_with_phase(
    phase_skill: Optional[str], perception_result: Optional[dict], mode: ReasoningMode
) -> Optional[str]:
    """Compare the perception stack's recommended skill against the phase
    machine's chosen skill. Returns a human-readable disagreement message to
    record via state.record_observation(), or None if they agree / there is
    no recommendation.

    Deliberately non-blocking (observes, does not override): the perception
    stack (TemporalReasoner/SpatialReasoner/SymbolicInference) was previously
    dead code — _perception_analyze() computed recommendations no caller ever
    read. Wiring it as an override without held-out-game validation would risk
    a working game-specific machine on unverified general perception. This
    surfaces disagreement as data first; see docs/EVALUATION.md for the plan
    to use that data to decide when perception should start steering instead
    of just observing.
    """
    if not perception_result:
        return None
    recommended = perception_result.get("recommended_skills")
    if not recommended:
        return None
    top = recommended[0] if isinstance(recommended, (list, tuple)) else recommended
    if top and top != phase_skill:
        return (f"Perception suggests '{top}' but phase machine chose "
                f"'{phase_skill}' (mode={mode.value})")
    return None


@dataclass
class ModeStrategy:
    """Stratégie associée à un mode de raisonnement."""
    mode: ReasoningMode
    description: str
    trigger_condition: Callable[[dict], bool]
    priority: int = 0  # Plus haut = prioritaire en cas de conflit


class ReasonModeManager:
    """Sélectionne et gère le mode de raisonnement actif.

    9 modes inspirés d'AHOIS, sélection automatique basée sur le contexte
    (stagnation, domaine détecté, disponibilité skill tree, etc.)
    """

    def __init__(self):
        self.current_mode = ReasoningMode.EXPLORATION
        self.mode_history: list = []
        self.strategies = self._build_strategies()

    def _build_strategies(self) -> list:
        return [
            ModeStrategy(
                ReasoningMode.SOCRATIC,
                "Questionner les hypothèses bloquantes",
                lambda ctx: ctx.get("doubt_active", False),
                priority=10,
            ),
            ModeStrategy(
                ReasoningMode.COUNTERFACTUAL,
                "Explorer alternative après échec répété",
                lambda ctx: ctx.get("stagnation", 0) >= 5,
                priority=9,
            ),
            ModeStrategy(
                ReasoningMode.SIMULATION,
                "Projeter état futur via simulation physique (Box3D)",
                lambda ctx: (
                    ctx.get("causal_ambiguity", False) or
                    (ctx.get("drive_caution", 0.0) > 0.7 and ctx.get("stagnation", 0) >= 2) or
                    ctx.get("needs_physical_verification", False)
                ),
                priority=9,
            ),
            ModeStrategy(
                ReasoningMode.ANALOGICAL,
                "Transférer skill depuis jeu/niveau similaire",
                lambda ctx: ctx.get("skill_tree_available", False) and ctx.get("stagnation", 0) >= 3,
                priority=8,
            ),
            ModeStrategy(
                ReasoningMode.CAUSAL,
                "Identifier cause d'un échec de skill",
                lambda ctx: ctx.get("last_skill_failed", False),
                priority=7,
            ),
            ModeStrategy(
                ReasoningMode.ROTATION,
                "Cycler rotation vers objectif connu",
                lambda ctx: ctx.get("domain") == "rotation" and ctx.get("has_goal_hypothesis", False),
                priority=6,
            ),
            ModeStrategy(
                ReasoningMode.PATHFINDING,
                "Naviguer vers cible identifiée",
                lambda ctx: ctx.get("has_target", False),
                priority=5,
            ),
            ModeStrategy(
                ReasoningMode.TRANSFORMATION,
                "Appliquer transformation de couleur/forme connue",
                lambda ctx: ctx.get("domain") == "transform",
                priority=4,
            ),
            ModeStrategy(
                ReasoningMode.GOAL_INFERENCE,
                "Inférer objectif depuis observations",
                lambda ctx: not ctx.get("has_goal_hypothesis", False) and ctx.get("domain_identified", True),
                priority=3,
            ),
            ModeStrategy(
                ReasoningMode.EXPLORATION,
                "Explorer environnement (mode par défaut)",
                lambda ctx: True,  # Toujours applicable (fallback)
                priority=0,
            ),
        ]

    def select_mode(self, context: dict) -> ReasoningMode:
        """Sélectionne le mode de raisonnement le plus prioritaire applicable."""
        applicable = [s for s in self.strategies if s.trigger_condition(context)]
        applicable.sort(key=lambda s: -s.priority)
        selected = applicable[0].mode if applicable else ReasoningMode.EXPLORATION
        return selected

    def log_mode_switch(self, old_mode: ReasoningMode, new_mode: ReasoningMode, reason: str):
        """Enregistre un changement de mode pour analyse post-mortem."""
        self.mode_history.append({
            "from": old_mode.value,
            "to": new_mode.value,
            "reason": reason,
        })


class PKM:
    """Persistent Knowledge Memory — structured per-game memory."""

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.data: Dict[str, Dict[str, Any]] = {}

    def set(self, category: str, key: str, value):
        self.data.setdefault(category, {})[key] = value

    def get(self, category: str, key: str, default=None):
        return self.data.get(category, {}).get(key, default)

    def get_all(self, category: str) -> dict:
        return self.data.get(category, {})

    def report(self) -> str:
        lines = [f"PKM[{self.game_id}]"]
        for category, items in self.data.items():
            lines.append(f"  {category}: {items}")
        return "\n".join(lines)


class ScientistAgent(MLTiersMixin, DiscoveryMixin, SkillsMixin):
    """Discover game mechanics, then solve each level."""

    def __init__(self, game_name: str, enable_benchmark: bool = True, enable_skill_tree: bool = True, enable_world_model: bool = False, enable_nano_llm: bool = True):
        if not ARC_RUNTIME_AVAILABLE:
            raise RuntimeError(
                "ScientistAgent requires the ARC-AGI runtime. Install it with "
                "`pip install 'arc-agi>=0.9.9,<1.0'` (provides arc_agi + arcengine)."
            )
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

        # ═══ World Model Tool v2 (Multi-Modal JEPA: noise + bottleneck + IDM) ═══
        self.world_model = None
        self._wm_latent_actions = None  # IDM-discovered actions
        if enable_world_model:
            try:
                from cogniarc.world_model_v2 import MultiModalWorldModel, WorldModelConfigV2
                cfg = WorldModelConfigV2(
                    noise_sigma=0.05,
                    bottleneck_dim=128,
                    idm_num_actions=8,
                    rollout_steps=5,
                )
                self.world_model = MultiModalWorldModel(config=cfg, game_id=game_name)
                if self.world_model.available:
                    mem = self.world_model.memory_size()
                    print(f"[WorldModel v2] V-JEPA + noise={cfg.noise_sigma} + bn {cfg.latent_dim}->{cfg.bottleneck_dim}" +
                          (f" + {mem} prior transitions" if mem > 0 else " (fresh start)"))
                else:
                    print(f"[WorldModel v2] Fallback encoder (no V-JEPA) - noise + bottleneck active")
            except Exception as e:
                print(f"[WorldModel v2] Failed to init: {e}")

        # ═══ NEW: Micro-NN predictors (instant, <1ms) ═══
        self.action_predictor = None
        self.domain_predictor = None
        self.pathfinder_nn = None  # Micro-NN pathfinder
        try:
            from cogniarc.micro_predictors import ActionPredictor, DomainPredictor, PathfinderPredictor
            self.action_predictor = ActionPredictor()
            self.domain_predictor = DomainPredictor()
            self.pathfinder_nn = PathfinderPredictor()
            if self.pathfinder_nn.available:
                print(f"[MicroNN] Pathfinder loaded (53→32→16→4)")
        except Exception as e:
            pass

        # ═══ NEW: Nano-LLM HF tier (Qwen2.5-0.5B via Ollama, opt-in) ═══
        # Sits between micro-NN and V-JEPA in the escalation chain. Wrapped in a
        # safety harness that validates proposals against walls / known failures.
        self.nano_llm = None
        self.nano_harness = None
        if enable_nano_llm:
            try:
                from cogniarc.nano_llm import NanoLLM, NanoLLMHarness
                self.nano_llm = NanoLLM()
                self.nano_harness = NanoLLMHarness(self.nano_llm)
                if self.nano_llm.available:
                    print(f"[NanoLLM] {self.nano_llm.model} online (Ollama)")
                else:
                    print(f"[NanoLLM] offline — {self.nano_llm._last_error}")
            except Exception as e:
                print(f"[NanoLLM] Failed to init: {e}")

        # ═══ NEW: ObjectTracker (generic player/action/wall inference) ═══
        # Always on (cheap: pure numpy segmentation, no model to load). Feeds
        # _detect_wall_colors() as reinforcing evidence — see
        # scientist_agent_discovery.py and object_perception.py for why this
        # is the generalizable counterpart to LS20's hardcoded sprite tags.
        from cogniarc.object_perception import ObjectTracker
        self.object_tracker = ObjectTracker()

        # GoalSanityChecker — wrong-goal loop detection (Tufa Labs interview insight)
        from cogniarc.goal_sanity import GoalSanityChecker
        self.goal_sanity = GoalSanityChecker(self)

        # Legacy aliases (to be removed gradually)
        self._phase = self.state.phase
        self._walls_detected = self.state.walls_detected

    def step(self, action_num: int):
        # ═══ NEW: Record pre-step observation for world model + object tracker + goal sanity ═══
        obs_before = None
        if (self.world_model or self.object_tracker or hasattr(self, 'goal_sanity')) and self.obs.frame and len(self.obs.frame) > 0:
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

        # ═══ NEW: Record transition in object tracker (generic player/wall evidence) ═══
        if self.object_tracker and obs_before is not None and self.obs.frame and len(self.obs.frame) > 0:
            self.object_tracker.observe(obs_before, action_num, self.obs.frame[0])
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

        # ═══ GoalSanityChecker: record action for wrong-goal loop detection ═══
        if hasattr(self, 'goal_sanity') and self.player and obs_before is not None:
            px_before = (self.player.x, self.player.y)
            # Position after step (player may have moved)
            px_after = (self.player.x, self.player.y) if self.player else px_before
            self.goal_sanity.record_action(action_num, px_before, px_after)

        # ═══ NEW: Update ScientificState ═══
        self.state.steps_taken = self.steps
        self.state.last_action = action_num
        self.state.stagnation_count = self.drives.stagnation_counter
        self.state.last_state_hash = state_hash

        return self.obs

    def _invalidate_current_goal(self):
        """Reset the current goal hypothesis — called by GoalSanityChecker
        when a wrong-goal loop is detected. Forces the agent to abandon
        its current theory and re-explore from scratch."""
        if hasattr(self, 'state') and self.state:
            if self.state.current_hypothesis:
                # Remember failed hypothesis before erasing it
                if not hasattr(self, '_failed_hypotheses'):
                    self._failed_hypotheses = set()
                self._failed_hypotheses.add(str(self.state.current_hypothesis.description)
                    if hasattr(self.state.current_hypothesis, 'description')
                    else str(self.state.current_hypothesis))
                self.state.refute_current_hypothesis("GoalSanityChecker: invalid goal")
            self.state.uncertainty = 1.0
        # Reset pathfinding caches and wall detection
        if hasattr(self, '_pathfinder'):
            self._pathfinder = None
        self._walls_detected = False
        self.state.walls_detected = False
        if hasattr(self, 'goal_sanity'):
            self.goal_sanity.reset()

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
        """Run temporal + spatial analysis on the current observation.

        Returns {temporal, spatial, symbols, recommended_skills} where present.
        Advisory only (its output is surfaced, not forced) and fully defensive:
        a perception failure must never crash the solve loop.

        Correctness note (found by the first holdout run, sc25): this method
        was dead code until wired in feat/wire-cognitive-architecture, and the
        dead version was broken three ways — it called analyze() with an
        argument (both reasoners' analyze() take none), it stored raw obs
        objects instead of grids in the frame history, and it passed pattern
        objects where infer() wants (type, confidence) tuples. All fixed here;
        the whole thing is also wrapped so any residual API drift degrades to
        an empty result rather than aborting the run.
        """
        if not self._perception_initialized:
            return {"mode_action": None}

        # Extract the current grid; nothing to analyze without it.
        if not (obs and hasattr(obs, 'frame') and obs.frame and len(obs.frame) > 0):
            return {}
        grid = obs.frame[0]

        result: dict = {}
        try:
            # Frame history holds GRIDS (not obs objects).
            history = getattr(self, '_frame_history', None)
            if history is None:
                history = self._frame_history = []
            history.append(grid)
            if len(history) > 8:  # bounded
                del history[:-8]

            # Temporal: analyze the recent grid sequence (needs >= 2 frames).
            temporal_pattern = None
            if len(history) >= 2:
                from .temporal_inference import TemporalReasoner
                temporal_pattern = TemporalReasoner(frames=list(history[-3:])).analyze()
                result["temporal"] = temporal_pattern

            # Spatial: analyze the current grid (SpatialReasoner(grid) segments
            # in its constructor; analyze() then reads self.grid).
            from .spatial_inference import SpatialReasoner
            spatial_reasoner = SpatialReasoner(grid)
            spatial_pattern = spatial_reasoner.analyze()
            result["spatial"] = spatial_pattern

            # Symbolic: infer() wants (type, confidence) tuples, not the objects.
            symbols = self._symbolic.infer(
                temporal_pattern=(
                    (temporal_pattern.type, temporal_pattern.confidence)
                    if temporal_pattern is not None else None
                ),
                spatial_pattern=(spatial_pattern.type, spatial_pattern.confidence),
                regions=spatial_reasoner.regions,
            )
            result["symbols"] = symbols
            skills = self._symbolic.skill_recommendations()
            if skills:
                # skill_recommendations() returns (name, confidence, reason)
                # tuples; expose plain skill names for reconcile_*.
                result["recommended_skills"] = [
                    s[0] if isinstance(s, (list, tuple)) else s for s in skills
                ]
        except Exception as e:
            # Advisory: never let perception abort the solve loop.
            return {"perception_error": str(e)}

        return result

    def solve_level(self, level_num: Optional[int] = None) -> bool:
        """Solve current level using phase-based skill execution with SocraticCritic."""

        # ═══ Route to domain-specific strategy based on game type ═══
        game_type = getattr(self, '_game_type', None)
        if game_type == "painting":
            from cogniarc.painting_strategy import PaintingStrategy
            strategy = PaintingStrategy(self)
            return strategy.solve_level(level_num)
        if game_type == "click":
            from cogniarc.click_strategy import ClickStrategy
            strategy = ClickStrategy(self)
            return strategy.solve_level(level_num)
        if game_type == "puzzle":
            from cogniarc.puzzle_strategy import PuzzleStrategy
            strategy = PuzzleStrategy(self)
            return strategy.solve_level(level_num)

        # ── Navigation / puzzle / unknown: use the phase machine ──

        prev_lvl = self.obs.levels_completed
        if level_num is not None and prev_lvl + 1 != level_num:
            print(f"  ⚠️ Expected level {level_num}, at {prev_lvl + 1}")

        # Update current level index
        self.current_level_idx = prev_lvl
        self._walls_detected = False  # Reset for new level
        self._phase = "observe"  # GENERIC phase flow (was: detect_walls)
        self.state.walls_detected = False
        self.state.phase = "observe"
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
        goal_rotation_known = self._infer_goal_rotation() is not None
        self.state.set_assumption("goal_known", goal_rotation_known)
        # Tell SocraticCritic a rotation mechanism exists even without action 6
        # (e.g. LS20's changer, cycled via two other actions) — built from
        # already-discovered generic signals (rotation-changer mechanic tag,
        # known goal rotation), not a new hardcoded assumption.
        self.state.set_assumption(
            "has_rotation_mechanism",
            is_rotation
            or bool(self.pkm.get('mechanics', 'rotation_changers', []))
            or goal_rotation_known,
        )

        # Record observations from discovery
        self.state.record_observation(
            f"Domain={self.state.domain_type}, movement={is_movement}, rotation={is_rotation}",
            source="discovery"
        )

        # ═══ Select reasoning mode via ReasonModeManager ═══
        # Compute causal ambiguity: do we have evidence that contradicts
        # our current understanding? (e.g., ObjectTracker confused,
        # unexpected grid changes, hypothesis refuted)
        ot = getattr(self, 'object_tracker', None)
        causal_ambiguity = False
        if ot is not None and ot.has_enough_observations():
            pc = ot.player_color
            if pc is None:
                # Player identification still ambiguous after enough steps
                causal_ambiguity = self.steps > 15
            # Check if last step had unexpected results
            if ot.last_step_player_moved is False:
                # Expected movement but didn't move — wall ambiguity
                causal_ambiguity = causal_ambiguity or (self.steps > 10)
                
        needs_physical_verification = (
            # Hypothesis involves movement prediction that can be verified
            hasattr(self, '_phase') and self._phase in ("navigate_to_target", "interact")
            and ot is not None and ot.last_step_player_moved is not None
        )

        mode_context = {
            "stagnation": self.drives.stagnation_counter,
            "domain": self.state.domain_type,
            "available_actions": available,
            "skill_tree_available": self.skill_tree is not None,
            "doubt_active": self.drives.doubt_triggered,
            "has_target": hasattr(self, '_phase') and self._phase in ("navigate_to_changer", "navigate_to_lock", "navigate_to_target"),
            "has_goal_hypothesis": hasattr(self, '_phase') and self._phase in ("rotation_cycle", "interact"),
            "causal_ambiguity": causal_ambiguity,
            "drive_caution": self.drives.drive_values.get("caution", 0.0),
            "needs_physical_verification": needs_physical_verification,
        }
        new_mode = self.mode_manager.select_mode(mode_context)
        if new_mode != self.current_reasoning_mode:
            self.mode_manager.log_mode_switch(self.current_reasoning_mode, new_mode,
                f"solve_level() init: domain={self.state.domain_type}")
            self.current_reasoning_mode = new_mode
            print(f"  🧠 Reasoning Mode: {new_mode.value}")

        # ═══ Initialize Perception Stack for temporal/spatial integration ═══
        self._init_perception_stack()

        # ═══ Active experimentation: if there's an ambiguous wall/floor colour
        # with high info gain, execute the discriminating action instead of
        # continuing the phase machine. This is the "steering" upgrade over the
        # previous advisory-only version. ═══
        try:
            experiment_action = self.suggest_wall_experiment(min_info_bits=1.0)
            if experiment_action is not None:
                print(f"  🔬 Executing active experiment: action {experiment_action}")
                self.step(experiment_action)
                if self.obs.levels_completed > prev_lvl:
                    print(f"  ✅ LEVEL {self.obs.levels_completed} COMPLETED via experiment!")
                    self._record_level_skills(prev_lvl + 1)
                    return True
                # Continue to phase machine loop — the experiment was just one step
        except Exception as e:
            print(f"  🔬 Active experiment skipped: {e}")

        # ═══ IDM: Latent Action Discovery (WorldModel v2) ═══
        # After collecting some transitions, discover latent actions without labels.
        # This helps the agent understand game mechanics even when actions are unknown.
        if self.world_model and hasattr(self.world_model, 'discover_actions'):
            wm = self.world_model
            if wm.memory_size() >= 8 and self._wm_latent_actions is None:
                try:
                    actions, clusters = wm.discover_actions(num_actions=4, min_transitions=8)
                    self._wm_latent_actions = actions
                    if len(actions) > 0:
                        print(f"  🎯 IDM: discovered {len(actions)} latent actions from {wm.memory_size()} transitions")
                        for i, cnt in enumerate(np.bincount(clusters, minlength=len(actions))):
                            print(f"     Action {i}: {cnt} transitions")
                except Exception as e:
                    print(f"  🎯 IDM skipped: {e}")

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

            # ═══ Perception Stack: observe (non-blocking) disagreement with
            # the phase machine's skill choice. See reconcile_perception_with_phase()
            # docstring for why this informs rather than overrides for now. ═══
            perception = self._perception_analyze(self.obs)
            disagreement = reconcile_perception_with_phase(
                skill_id, perception, self.current_reasoning_mode
            )
            if disagreement:
                self.state.record_observation(disagreement, source="perception_reconcile")
                print(f"  👁️ {disagreement}")

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

            # ═══ Mode 10 — SIMULATION_PHYSIQUE: project future state ═══
            if self.current_reasoning_mode == ReasoningMode.SIMULATION and PHYSICS_AVAILABLE:
                try:
                    self._run_physical_simulation(skill_id)
                except Exception as e:
                    print(f"  ⚙️ Physics sim skipped: {e}")
                    self.state.phase_attempts = 0
                    continue  # Retry with new phase
                else:
                    print(f"     Aucune alternative - execution risquee")

            # ═══ COGNITIVE DRIVES: adaptive dwell before skill execution ═══
            # If cognitive fatigue is high, skip deep planning and try random actions
            if self.drives.fatigue.intuition_mode and self.state.phase_attempts >= 1:
                print(f"  🧠 Fatigue mode: trying random action (intuition)")
                import random
                available = list(self.obs.available_actions or [1, 2, 3, 4])
                action = random.choice(available)
                prev_lvl = self.obs.levels_completed
                self.step(action)
                if self.obs.levels_completed > prev_lvl:
                    print(f"  ✅ LEVEL {self.obs.levels_completed} COMPLETED via intuition!")
                    self._record_level_skills(prev_lvl + 1)
                    return True
                self.drives.fatigue.spend(2)
                continue

            # ═══ If doubt is high and we keep failing same phase, switch to exploration ═══
            if (self.drives.doubt_score() > 0.3 and self.state.phase_attempts >= 2
                    and self._phase not in ("observe", "discovery")):
                print(f"  🧠 Doubt high ({self.drives.doubt_score():.2f}), re-observing...")
                self._phase = "observe"
                self.state.phase = "observe"
                self.state.phase_attempts = 0
                self.drives.stagnation_counter = 0
                continue

            print(f"  🔄 Phase: {self._phase} -> Skill: {skill_id}")
            success = self._execute_skill(skill_id)

            if success:
                self._advance_phase(success)
                print(f"  ✅ Phase complete: {self._phase}")
            else:
                self.state.phase_attempts += 1
                self._phase_escalation_count += 1
                print(f"  ⚠️ Skill {skill_id} failed in phase {self._phase} (attempt {self.state.phase_attempts})")
                # Advance phase even on failure (e.g., plan→refine, execute→refine)
                self._advance_phase(success)

            # ═══ GoalSanityChecker: only check after N failed attempts (let refine work first) ═══
            if hasattr(self, 'goal_sanity') and hasattr(self, 'critic'):
                try:
                    recent_report = getattr(self.state, '_last_critic_report', None)
                    if recent_report is None:
                        recent_report = self.critic.quick_check(
                            self._build_phase_hypothesis(), self.state
                        )
                    unresolved = recent_report.unresolved() if recent_report else []
                    self.goal_sanity.record_critic_issues(unresolved)

                    verdict = self.goal_sanity.check(
                        phase_failed=(self.state.phase_attempts >= 3)
                    )
                    if not verdict.sane:
                        print(f"  🚫 GOAL INVALIDÉ: {verdict.reason}")
                        print(f"     → {verdict.suggested_action}")
                        self._invalidate_current_goal()
                        self.state.current_hypothesis = None
                        if hasattr(self.state, 'uncertainty'):
                            self.state.uncertainty = 1.0
                        self._phase = "observe"
                        self.state.phase = "observe"
                        self.state.phase_attempts = 0
                        self.drives.doubt_triggered = True
                        self.drives.stagnation_counter = 10
                        self.current_reasoning_mode = ReasoningMode.EXPLORATION
                        self.goal_sanity.reset()
                        continue
                    elif verdict.failed_checks:
                        print(f"  ⚠️ GoalSanity warn: {verdict.reason}")
                except Exception as e:
                    print(f"  ⚠️ GoalSanity check failed: {e}")

            if not success:

                # ═══ Nano-LLM tier: if phase stuck, ask the nano-LLM for a safe
                # action proposal before escalating. Only fires after 2+ failures
                # and when nano-LLM is available (opt-in via enable_nano_llm=True).
                if self.state.phase_attempts >= 2 and self.nano_harness is not None:
                    try:
                        print("  🤖 Nano-LLM proposing action...")
                        proposal = self._nano_propose_action()
                        if proposal is not None:
                            print(f"  🤖 Nano-LLM suggests action {proposal}")
                            self.step(proposal)
                            if self.obs.levels_completed > prev_lvl:
                                print(f"  ✅ LEVEL {self.obs.levels_completed} COMPLETED via Nano-LLM!")
                                self._record_level_skills(prev_lvl + 1)
                                return True
                            # Continue — the Nano-LLM step may have changed state
                    except Exception as e:
                        print(f"  🤖 Nano-LLM proposal failed: {e}")

                # Global escalation: if stuck too long, force skip level.
                # Threshold is mode-driven: COUNTERFACTUAL/SOCRATIC modes signal
                # active doubt about the current plan, so give up sooner.
                if self._phase_escalation_count >= phase_escalation_threshold(self.current_reasoning_mode):
                    print(f"  🔴 Trop d'echecs ({self._phase_escalation_count}), force skip niveau...")
                    skip_to = self._try_skip_level()
                    if skip_to == "complete":
                        break  # Niveau complete ou skip
                    # Burn remaining steps pour forcer game over
                    self.handle_transition()
                    break

                # After N failed attempts on same phase, ask for escalation.
                # N is mode-driven: PATHFINDING/ROTATION trust the deterministic
                # skill longer (it's likely to eventually succeed); doubt modes
                # escalate sooner.
                if self.state.phase_attempts >= phase_attempts_threshold(self.current_reasoning_mode):
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

    def _run_physical_simulation(self, skill_id: str) -> None:
        """Mode 10 handler: project future state using physics engine.

        Called when SIMULATION_PHYSIQUE mode is active. Uses the
        existing physics tools to simulate the current hypothesis
        and logs any contradictions found.

        Current simulation: lightweight 2D physics (simulator/physics.py)
        for movement prediction. Box3D 3D sim available when lib is built.
        """
        if not PHYSICS_AVAILABLE:
            return

        print(f"  ⚙️ [Mode 10] Simulating '{skill_id}'...")
        hypothesis = self._build_phase_hypothesis()

        if "navigate" in skill_id or "navigate" in hypothesis.lower():
            # Navigation simulation: predict player displacement
            ot = getattr(self, 'object_tracker', None)
            if ot is not None and ot.has_enough_observations():
                summary = ot.get_perception_summary(grid=self.obs.frame[0] if hasattr(self.obs, 'frame') else None)
                directions = summary.get("action_directions", {})
                pos = summary.get("player_position")
                if pos and directions:
                    print(f"  ⚙️   Player at {pos}, {len(directions)} action directions known")
                    # Log predicted next positions for each action
                    for action, (dr, dc) in list(directions.items())[:3]:
                        nx = pos[0] + dc  # col displacement
                        ny = pos[1] + dr  # row displacement
                        # Check if within grid bounds
                        grid = self.obs.frame[0] if hasattr(self.obs, 'frame') and self.obs.frame else None
                        if grid is not None and 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            target_color = int(grid[ny, nx])
                            wall = target_color in ot.wall_colors
                            print(f"  ⚙️   Action {action}: predict ({nx:.0f}, {ny:.0f}) "
                                  f"→ color {target_color} {'🧱' if wall else '✅'}")
                        else:
                            print(f"  ⚙️   Action {action}: predict ({nx:.0f}, {ny:.0f}) → out of bounds")

        elif "rotate" in skill_id or "rotation" in hypothesis.lower():
            # Rotation simulation: use discrete_classifier to predict movement class
            print(f"  ⚙️   Rotation hypothesis — no physics simulation available yet")
        else:
            print(f"  ⚙️   No physics simulation for '{skill_id}' yet")

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

        # PHASE 1: Discover — generic ObjectTracker first, then fallback
        print("\n📖 DISCOVERY PHASE")
        print("  Generic perception: scouting via ObjectTracker...")

        # Adaptive exploration: continue until ObjectTracker has enough observations
        # OR we've cycled through all actions enough times, OR we hit max scout steps.
        scout_actions = list(self.obs.available_actions or [1, 2, 3, 4, 5, 6])
        scout_grid_changes = []  # (n_pixels, n_colors) per action
        max_scout_steps = 30  # Upper bound to prevent infinite loops
        min_scout_steps = 6   # Always do at least this many

        ot = self.object_tracker

        for scout_step in range(max_scout_steps):
            action = scout_actions[self.steps % len(scout_actions)]
            grid_before = self.obs.frame[0].copy() if self.obs.frame and len(self.obs.frame) > 0 else None
            self.step(action)
            grid_after = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            if grid_before is not None and grid_after is not None:
                from cogniarc.domain_classifier import _color_diversity
                diff = int(np.sum(grid_before != grid_after))
                colors = _color_diversity(grid_before, grid_after)
                scout_grid_changes.append((diff, colors))

            # Early exit if ObjectTracker has enough data (≥3 movement directions)
            if scout_step >= min_scout_steps and ot and ot.has_enough_observations():
                # Check how many movement directions we've learned
                summary = ot.get_perception_summary()
                movement_count = sum(
                    1 for d in summary.get('action_directions', {}).values()
                    if d is not None and (d[0]**2 + d[1]**2)**0.5 >= 0.5
                )
                if movement_count >= 3:
                    print(f"  ✅ Adaptive scout: {movement_count} movement directions learned in {scout_step+1} steps")
                    break
                # Also exit if we have wall colors and player color (enough to navigate)
                if summary.get('player_color') is not None and summary.get('wall_colors'):
                    print(f"  ✅ Adaptive scout: player + walls known in {scout_step+1} steps")
                    break
                # Also exit if we've cycled through all actions at least 3 times
                if scout_step >= len(scout_actions) * 3:
                    print(f"  ✅ Adaptive scout: all actions tried {scout_step+1 // len(scout_actions)} times in {scout_step+1} steps")
                    break

        if ot and ot.has_enough_observations():
            summary = ot.get_perception_summary()
            print(f"  🎯 Player color: {summary['player_color']}")
            dirs_str = ', '.join(f"{a}→{d}" for a, d in sorted(summary['action_directions'].items()))
            print(f"  🧭 Learned directions: {dirs_str}")
            walls = summary['wall_colors']
            if walls:
                print(f"  🧱 Wall evidence: {sorted(walls)}")
            self.state.set_assumption("walls_known", bool(walls))
            self.state.set_assumption("player_found", summary['player_color'] is not None)
            self.state.set_assumption("actions_scouted", True)
            self._walls_detected = bool(walls)
            self.state.walls_detected = bool(walls)
            self.state.record_observation(
                f"ObjectTracker: player={summary['player_color']}, "
                f"directions={dirs_str}, walls={sorted(walls)}",
                source="discovery"
            )
        else:
            # Fallback: source-code tag detection (LS20-specific)
            print("  ⚠️ ObjectTracker insufficient, fallback to source discovery...")
            self.discover_from_source()

        self.discover_available_actions()
        self.discover_properties()

        # PHASE 2: Classify game type from scout observations
        print("\n🎮 GAME TYPE CLASSIFICATION")
        from cogniarc.domain_classifier import classify_game_type

        # Build scout_results from discovery PKM, use ObjectTracker as override
        scout_results = self.pkm.get('discovery', 'scout_results', {}) or {}
        ot_summary = ot.get_perception_summary() if ot and ot.has_enough_observations() else None
        available = list(self.obs.available_actions or [])
        game_type = classify_game_type(scout_results, scout_grid_changes, ot_summary, available)
        self._game_type = game_type
        print(f"  🎮 Detected: {game_type} (actions: {available})")

        # PHASE 3: Solve levels
        print(f"\n🎮 SOLVE PHASE (target: {self.obs.win_levels} levels)")

        # Total step budget. A holdout smoke run can lower this via
        # _holdout_max_steps (set by scripts/run_holdout.py --max-steps) to keep
        # remote-API usage bounded.
        max_total = min(400, getattr(self, '_holdout_max_steps', 400) or 400)
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
