#!/usr/bin/env python3
"""
DomainProfiler — Extended DomainClassifier with action semantics, safe actions,
and strategy recommendations for ARC-AGI-3 solving.

Usage:
    from domain_classifier import DomainProfiler
    dp = DomainProfiler(env)
    profile = dp.profile()
    print(profile.action_semantics)   # {1: "up", 2: "down", 6: "rotate_cw"}
    print(profile.safe_actions)       # [1, 2, 3, 4] (excludes buggy 6/7)
    print(profile.recommended_strategy)  # "cognitive_movement"
"""

from __future__ import annotations

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

from .domain_classifier import DomainClassifier, _hash_grid, _diff_grid
from .common import GameAction


@dataclass
class DomainProfile:
    """Complete profile of a game domain for solver strategy selection."""
    domain_type: str
    confidence: float
    action_semantics: Dict[int, str] = field(default_factory=dict)      # action_id -> "up", "rotate", etc.
    safe_actions: List[int] = field(default_factory=list)               # actions that don't crash
    buggy_actions: List[int] = field(default_factory=list)              # actions that raise exceptions
    suspected_state_vars: List[str] = field(default_factory=list)       # hidden vars like "cklxociuu"
    complexity_estimate: int = 50                                       # steps per level
    recommended_strategy: str = "explore"                               # solver strategy name
    evidence: Dict[str, Any] = field(default_factory=dict)


class DomainProfiler:
    """Extended domain classifier with solver-oriented profiling."""

    # Cross-game action semantics learned from 25 games
    DEFAULT_ACTION_SEMANTICS = {
        1: "up",
        2: "down",
        3: "left",
        4: "right",
        5: "interact",
        6: "rotate",      # often rotation/global parameter
        7: "special",     # often secondary parameter/time
    }

    # Known buggy action patterns across games
    KNOWN_BUGGY_PATTERNS = {
        # (game_id_pattern, action) -> error_type
        ("ft09", 6): "KeyError 'x'",
        ("bp35", 6): "KeyError 'x'",
        ("tn36", 6): "KeyError 'x'",
    }

    def __init__(self, env, max_steps: int = 30):
        # Store original env and extract game_id BEFORE any classification
        self.original_env = env
        self.max_steps = max_steps
        # Extract game_id from the original env
        self.game_id = self._extract_game_id(env)
        self.base_classifier = DomainClassifier(env, max_steps)
        # Get available actions from original env (before classify)
        self.actions_available = self._get_available_actions(env)
        self.profile: Optional[DomainProfile] = None

    def _extract_game_id(self, env) -> str:
        """Extract game_id from environment robustly."""
        # Try environment_info first (arc_agi wrapper)
        for attr in ['environment_info', 'info']:
            try:
                info = getattr(env, attr, None)
                if info and hasattr(info, 'game_id') and info.game_id:
                    return str(info.game_id)
            except Exception:
                pass
        
        # Try _game attribute (the actual game object)
        try:
            game = getattr(env, '_game', None)
            if game:
                # Game class name like "ls20-9607627b.Ls20" 
                module = getattr(game, '__module__', '')
                if module and '.' in module:
                    return module.split('.')[0]  # e.g., "ls20-9607627b"
        except Exception:
            pass
        
        # Try observation_space or _last_response
        for attr in ['observation_space', '_last_response']:
            try:
                val = getattr(env, attr, None)
                if val and hasattr(val, 'get'):
                    game_id = val.get('game_id')
                    if game_id:
                        return str(game_id)
            except Exception:
                pass
        
        # Fallback attributes
        for attr in ['_game_id', '_env_id', 'game_id', 'spec.id']:
            try:
                if attr == 'spec.id':
                    val = getattr(getattr(env, 'spec', None), 'id', None)
                else:
                    val = getattr(env, attr, None)
                if val and str(val) != 'unknown':
                    return str(val)
            except Exception:
                pass
        
        # Try unwrapped
        if hasattr(env, 'unwrapped'):
            try:
                val = getattr(env.unwrapped, '_game_id', None)
                if val and str(val) != 'unknown':
                    return str(val)
            except Exception:
                pass
        
        return 'unknown'

    def _get_available_actions(self, env) -> List[int]:
        """Get available actions from environment."""
        try:
            if hasattr(env, 'reset'):
                obs = env.reset()
                if hasattr(obs, 'available_actions') and obs.available_actions:
                    return list(obs.available_actions)
        except Exception:
            pass
        # Fallback: assume standard 1-7 for ARC-AGI-3
        return list(range(1, 8))

    def build_profile(self) -> DomainProfile:
        """Run full profiling: classify + test actions + read source + recommend strategy."""
        # Step 1: Base classification (uses its own env internally)
        domain = self.base_classifier.classify()

        # Step 1.5: Ensure we have a fresh env for subsequent steps
        self.env = self._make_fresh_env()

        # Step 2: Test all actions for safety and semantics
        safe_actions, buggy_actions = self._test_action_safety()

        # Step 3: Infer action semantics from effects
        action_semantics = self._infer_action_semantics(safe_actions)

        # Step 4: Read source for hidden state variables
        suspected_vars = self._extract_state_variables()

        # Step 5: Estimate complexity
        complexity = self._estimate_complexity()

        # Step 6: Recommend strategy
        strategy = self._recommend_strategy(domain, action_semantics, suspected_vars)

        # Step 7: Final fresh env for any cleanup
        self.env = self._make_fresh_env()

        self.profile = DomainProfile(
            domain_type=domain,
            confidence=self.base_classifier.confidence,
            action_semantics=action_semantics,
            safe_actions=safe_actions,
            buggy_actions=buggy_actions,
            suspected_state_vars=suspected_vars,
            complexity_estimate=complexity,
            recommended_strategy=strategy,
            evidence=self.base_classifier.evidence,
        )

        return self.profile

    def _test_action_safety(self) -> Tuple[List[int], List[int]]:
        """Test each action: safe if it doesn't raise, buggy if it crashes."""
        safe = []
        buggy = []
        actions = list(self.actions_available or [])

        for act_num in actions:
            fresh_env = self._make_fresh_env()
            try:
                action = getattr(GameAction, f"ACTION{act_num}")
                obs = fresh_env.step(action)
                if obs is not None and hasattr(obs, 'frame') and obs.frame is not None:
                    safe.append(act_num)
                else:
                    buggy.append(act_num)
                    print(f"  ⚠️ ACTION{act_num} buggy: returned invalid observation")
            except Exception as e:
                buggy.append(act_num)
                print(f"  ⚠️ ACTION{act_num} buggy: {type(e).__name__}: {e}")

        return safe, buggy

    def _make_fresh_env(self):
        """Create a fresh environment instance."""
        import arc_agi
        arcade = arc_agi.Arcade()
        return arcade.make(self.game_id)

    def _infer_action_semantics(self, safe_actions: List[int]) -> Dict[int, str]:
        """Infer semantic meaning of each safe action from grid effects."""
        semantics = {}
        
        for act_num in safe_actions:
            fresh_env = self._make_fresh_env()
            try:
                # Need to reset twice: once to get initial state, once for before
                fresh_env.reset()
                before = fresh_env.reset().frame[0].copy()
                action = getattr(GameAction, f"ACTION{act_num}")
                fresh_env.step(action)
                after = fresh_env.step(action).frame[0]
                changed, mask = _diff_grid(before, after)

                # Movement: small localized change (agent moves)
                if 1 <= changed <= 20:
                    semantics[act_num] = self.DEFAULT_ACTION_SEMANTICS.get(act_num, "move")
                # Rotation/global: large uniform change
                elif changed > 100:
                    semantics[act_num] = self.DEFAULT_ACTION_SEMANTICS.get(act_num, "global")
                # Interaction: medium change, specific region
                else:
                    semantics[act_num] = self.DEFAULT_ACTION_SEMANTICS.get(act_num, "interact")

            except Exception:
                semantics[act_num] = "unknown"

        return semantics

    def _extract_state_variables(self) -> List[str]:
        """Scan source code for hidden game state variables (like LS20's cklxociuu)."""
        suspected = []
        try:
            import glob, os, re
            game_id = getattr(self.env, '_game_id', 'unknown')
            env_dir = f"environment_files/{game_id}"
            dirs = glob.glob(f"{env_dir}/*/")
            if dirs:
                src = os.path.join(dirs[0], f"{game_id}.py")
                with open(src) as f:
                    code = f.read()

                # Pattern: self.variable_name = value (game state)
                pattern = r'self\.([a-z_]{4,20})\s*='
                for match in re.finditer(pattern, code):
                    var = match.group(1)
                    if var not in ['x', 'y', 'width', 'height', 'frame', 'level', 'score']:
                        suspected.append(var)

                # Known important vars from cross-game analysis
                known_keys = ['cklxociuu', 'hiaauhahz', 'fwckfzsyc', 'gudziatsk', 'dhksvilbb']
                for k in known_keys:
                    if k in code:
                        suspected.append(k)

        except Exception:
            pass

        return list(set(suspected))  # dedupe

    def _estimate_complexity(self) -> int:
        """Estimate steps needed per level based on domain and grid."""
        domain = self.base_classifier.result or "unknown"
        evidence = self.base_classifier.evidence

        base_estimates = {
            "movement": 40,
            "rotation": 20,
            "temporal": 60,
            "drawing": 80,
            "symbolic": 100,
            "selection": 50,
            "physics_chain": 60,
            "growth": 50,
        }

        base = base_estimates.get(domain, 50)

        # Adjust by grid density
        density = evidence.get("grid_density", 0.5)
        if density > 0.5:
            base = int(base * 1.5)

        # Adjust by action count
        num_actions = len(self.base_classifier.actions_available or [])
        if num_actions > 4:
            base = int(base * 1.2)

        return base

    def _recommend_strategy(self, domain: str, action_semantics: Dict[int, str], suspected_vars: List[str]) -> str:
        """Recommend solver strategy based on profile."""
        has_rotation = "rotate" in action_semantics.values()
        has_hidden_vars = len(suspected_vars) > 0
        has_movement = any(s in ["up", "down", "left", "right"] for s in action_semantics.values())

        if has_hidden_vars and has_movement and has_rotation:
            return "cognitive_hybrid"  # LS20 style - use ScientistAgent + CognitiveDrives
        elif has_hidden_vars and has_rotation:
            return "goal_inference_rotation"  # VC33/CD82 style - infer goal vars
        elif has_movement:
            return "cognitive_movement"  # TR87/TU93 - CognitiveDrives + BFS
        elif has_rotation:
            return "goal_inference_rotation"
        elif domain == "temporal":
            return "transform_inference"
        else:
            return "cognitive_explore"

    def save_profile(self, path: str = "") -> None:
        """Save profile to JSON for cross-game learning."""
        if not self.profile:
            return
        if not path:
            game_id = getattr(self.env, '_game_id', 'unknown')
            path = f"~/.cache/cogniarc/profiles/{game_id}_profile.json"
        path_obj = Path(path).expanduser()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        import dataclasses
        path_obj.write_text(json.dumps(dataclasses.asdict(self.profile), indent=2, default=str))

    def report(self) -> str:
        """Human-readable profile report."""
        if not self.profile:
            return "Not profiled yet. Call profile() first."

        p = self.profile
        lines = [
            f"=== DOMAIN PROFILE ===",
            f"Domain: {p.domain_type} (confidence: {p.confidence:.2f})",
            f"Recommended Strategy: {p.recommended_strategy}",
            f"Complexity: ~{p.complexity_estimate} steps/level",
            f"",
            f"Action Semantics:",
        ]
        for act, sem in sorted(p.action_semantics.items()):
            status = "✅" if act in p.safe_actions else "❌"
            lines.append(f"  {status} ACTION{act}: {sem}")
        if p.buggy_actions:
            lines.append(f"  Buggy Actions: {p.buggy_actions}")
        if p.suspected_state_vars:
            lines.append(f"Suspected State Vars: {p.suspected_state_vars}")
        return "\n".join(lines)


# ── Quick CLI ────────────────────────────────────────────
if __name__ == "__main__":
    import arc_agi
    arc = arc_agi.Arcade()
    game = "ls20-9607627b"
    env = arc.make(game)
    dp = DomainProfiler(env)
    profile = dp.build_profile()
    print(dp.report())
    dp.save_profile()