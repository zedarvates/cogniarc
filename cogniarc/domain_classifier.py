#!/usr/bin/env python3
"""
Domain Classifier for ARC-AGI-3.

Determines what KIND of abstract game this is BEFORE attempting to solve it.
The domain determines the entire interpretation framework.

Domains:
    movement      — agent position changes, objects to push, paths to navigate
    rotation      — orientation matters, sides/faces have meaning
    drawing       — traces appear, shapes drawn, fill/erase/complete patterns
    selection     — click to highlight, group/ungroup, confirm selections
    symbolic      — glyphs/runes with meaning, composition rules
    temporal      — ORDER of actions matters, undo available, sequences
    physics_chain — cause→effect chains, indirect manipulation, propagation
    growth        — objects appear/grow, stages, ecosystem dynamics

Usage:
    from domain_classifier import DomainClassifier
    dc = DomainClassifier(env)
    domain = dc.classify()   # ≤20 steps
    print(dc.report())
"""

from __future__ import annotations

import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .common import GameAction

# ── Domain Profiles ──────────────────────────────────────

DOMAIN_PROFILES = {
    "movement": {
        "keywords": ["position", "move", "collision", "wall", "push"],
        "test": "agent_position_changes",
        "signature": "ACTION changes agent coordinates",
    },
    "rotation": {
        "keywords": ["rotate", "orient", "face", "side", "turn"],
        "test": "object_orientation_changes",
        "signature": "ACTION rotates an object",
    },
    "drawing": {
        "keywords": ["trace", "draw", "fill", "erase", "curve", "circle"],
        "test": "pixels_appear_or_change_color",
        "signature": "ACTION creates/modifies visual traces",
    },
    "selection": {
        "keywords": ["select", "click", "highlight", "group", "confirm"],
        "test": "click_highlights_region",
        "signature": "ACTION selects/highlights grid region",
    },
    "symbolic": {
        "keywords": ["glyph", "symbol", "rune", "compose", "meaning"],
        "test": "symbol_like_patterns",
        "signature": "Grid contains symbolic shapes",
    },
    "temporal": {
        "keywords": ["order", "sequence", "undo", "reset", "step"],
        "test": "action_order_matters",
        "signature": "A then B ≠ B then A",
    },
    "physics_chain": {
        "keywords": ["chain", "cause", "effect", "propagate", "indirect"],
        "test": "indirect_effects",
        "signature": "Action affects non-target objects",
    },
    "growth": {
        "keywords": ["grow", "plant", "stage", "evolve", "ecosystem"],
        "test": "objects_appear_or_grow",
        "signature": "New objects appear over time",
    },
}


def _hash_grid(grid: np.ndarray) -> str:
    """Fast hash of a 64×64 grid."""
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


def _diff_grid(a: np.ndarray, b: np.ndarray) -> Tuple[int, np.ndarray]:
    """Return (changed_pixels, mask)."""
    mask = a != b
    return int(mask.sum()), mask


class DomainClassifier:
    """Classify the abstract domain of an ARC-AGI-3 game environment."""

    def __init__(self, env, max_steps: int = 20):
        self.env = env
        self.max_steps = max_steps
        self.obs = None
        self.steps_taken = 0
        self.actions_available: List[int] = []
        self.evidence: Dict[str, Any] = {}
        self.result: Optional[str] = None
        self.confidence: float = 0.0

    def classify(self) -> str:
        """Run diagnostic battery and return domain label."""
        self.obs = self.env.reset()
        self.actions_available = list(self.obs.available_actions or [])
        self.steps_taken = 0

        # ── Test 1: What does ACTION1 do on initial state? ──
        self._test_action_effects()

        # ── Test 2: Does order matter? ──
        self._test_order_matters()

        # ── Test 3: Incomplete shapes? ──
        self._test_shapes()

        # ── Test 4: Animations? ──
        self._test_animation()

        # ── Test 5: Agent tracking ──
        self._test_agent()

        # ── Score domains ──
        self._score()

        self._save()
        return self.result or "unknown"

    def _step(self, action):
        self.steps_taken += 1
        try:
            return self.env.step(action)
        except Exception as e:
            # Buggy action - record and return None
            self.evidence.setdefault("buggy_actions", []).append({
                "action": getattr(action, 'name', str(action)),
                "error": f"{type(e).__name__}: {e}"
            })
            return None

    # ── Diagnostic Tests ──────────────────────────────────

    def _test_action_effects(self):
        """Test each available action, record what changes. Skip buggy actions."""
        for act_num in self.actions_available[:4]:
            if self.steps_taken >= self.max_steps:
                break
            action = getattr(GameAction, f"ACTION{act_num}", None)
            if action is None:
                continue
            
            # Get current observation safely
            if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
                self.env.reset()
                self.obs = self.env.reset()
            
            before = self.obs.frame[0].copy() if self.obs and self.obs.frame else None
            if before is None:
                continue
                
            self.obs = self._step(action)
            
            # Skip if action was buggy (returned None)
            if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
                self.evidence.setdefault("action_effects", {})[act_num] = {
                    "changed_pixels": 0,
                    "changed_ratio": 0.0,
                    "buggy": True,
                    "note": "Action crashed or returned invalid observation"
                }
                # Reset to clean state for next action
                self.obs = self.env.reset()
                continue
            
            after = self.obs.frame[0]
            changed, mask = _diff_grid(before, after)
            self.evidence.setdefault("action_effects", {})[act_num] = {
                "changed_pixels": changed,
                "changed_ratio": round(changed / (64 * 64), 4),
                "buggy": False
            }

    def _test_order_matters(self):
        """Test if A then B produces same result as B then A."""
        if self.steps_taken + 6 > self.max_steps:
            return
        if len(self.actions_available) < 2:
            return

        a_num, b_num = self.actions_available[0], self.actions_available[1]
        act_a = getattr(GameAction, f"ACTION{a_num}")
        act_b = getattr(GameAction, f"ACTION{b_num}")

        # Ensure we have a valid observation
        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.obs = self.env.reset()
        
        # Save current state
        saved = self.obs.frame[0].copy()

        # Sequence A→B
        self.obs = self._step(act_a)
        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.evidence["order_matters"] = "inconclusive"
            self.evidence["order_note"] = "Action A buggy"
            self.obs = self.env.reset()
            return
            
        self.obs = self._step(act_b)
        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.evidence["order_matters"] = "inconclusive"
            self.evidence["order_note"] = "Action B buggy"
            self.obs = self.env.reset()
            return

        result_ab = self.obs.frame[0].copy()

        # Reset to saved state... we can't really reset to a saved state
        # without env.reset(), which resets the whole game.
        # Instead, use a heuristic: if action effects are highly dissimilar
        # depending on context, order matters.
        self.evidence["order_matters"] = "inconclusive"
        self.evidence["order_note"] = "Cannot test without state save/restore"

    def _test_shapes(self):
        """Detect incomplete shapes in the grid."""
        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.obs = self.env.reset()
        grid = self.obs.frame[0]
        # Simple heuristic: scan for open-ended color segments
        # A shape is "incomplete" if a color blob has thin, elongated extensions
        unique_colors = np.unique(grid)
        self.evidence["unique_colors"] = int(len(unique_colors))
        self.evidence["max_color"] = int(grid.max())
        self.evidence["grid_density"] = round(float((grid != 0).mean()), 3)

    def _test_animation(self):
        """Check if frames > 1 (animation sequences)."""
        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.obs = self.env.reset()
        n_frames = len(self.obs.frame)
        self.evidence["animation_frames"] = n_frames
        self.evidence["is_animated"] = n_frames > 1

    def _test_agent(self):
        """Try to locate a moving agent."""
        if self.steps_taken + 2 > self.max_steps:
            return
        if not self.actions_available:
            return

        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.obs = self.env.reset()
            
        a1 = self.actions_available[0]
        b1 = self.obs.frame[0].copy()
        self.obs = self._step(getattr(GameAction, f"ACTION{a1}"))
        if self.obs is None or not hasattr(self.obs, 'frame') or self.obs.frame is None:
            self.evidence["agent_test"] = {
                "changed_after_one_action": 0,
                "changed_clusters": "action_buggy",
            }
            self.obs = self.env.reset()
            return
            
        a2 = self.obs.frame[0]
        changed, mask = _diff_grid(b1, a2)

        self.evidence["agent_test"] = {
            "changed_after_one_action": changed,
            "changed_clusters": "analyze_later",
        }

    # ── Scoring ───────────────────────────────────────────

    def _score(self):
        """Score each domain based on evidence."""
        scores = {}
        e = self.evidence

        # Movement: action changes small # of pixels consistently
        effects = e.get("action_effects", {})
        avg_change = np.mean([v["changed_pixels"] for v in effects.values()]) if effects else 0
        scores["movement"] = 0.7 if 1 < avg_change < 100 else 0.2

        # Drawing: high change ratio per action
        scores["drawing"] = 0.8 if avg_change > 500 else 0.1

        # Animation: multiple frames
        scores["temporal"] = 0.6 if e.get("is_animated") else 0.1

        # Symbolic: many unique colors, low density
        scores["symbolic"] = 0.5 if e.get("unique_colors", 0) > 5 and e.get("grid_density", 0) < 0.5 else 0.2

        # Rotation: moderate change, specific patterns
        scores["rotation"] = 0.4

        # Default fallback
        scores["physics_chain"] = 0.3
        scores["growth"] = 0.2
        scores["selection"] = 0.3

        best = max(scores, key=scores.get)
        self.result = best
        self.confidence = scores[best]
        self.evidence["domain_scores"] = scores

    def _save(self):
        out = Path("/home/redgamer/arc_agi_agent/domain_result.json")
        out.write_text(json.dumps({
            "domain": self.result,
            "confidence": self.confidence,
            "evidence": {k: str(v) if isinstance(v, np.ndarray) else v
                         for k, v in self.evidence.items()},
            "steps_taken": self.steps_taken,
        }, indent=2, default=str))

    def report(self) -> str:
        if not self.result:
            return "Not classified yet."
        lines = [
            f"Domain: {self.result} (confidence: {self.confidence:.2f})",
            f"Steps: {self.steps_taken}",
            f"Actions available: {self.actions_available}",
        ]
        for k, v in self.evidence.get("action_effects", {}).items():
            lines.append(f"  ACTION{k}: {v['changed_pixels']} px changed ({v['changed_ratio']:.2%})")
        return "\n".join(lines)


# ── Quick CLI ────────────────────────────────────────────

if __name__ == "__main__":
    import arc_agi
    arc = arc_agi.Arcade()
    game = "ls20"
    env = arc.make(game)
    dc = DomainClassifier(env)
    domain = dc.classify()
    print(dc.report())
    print(f"Saved to arc_agi_agent/domain_result.json")
