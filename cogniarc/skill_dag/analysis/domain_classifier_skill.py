"""Domain Classifier Skill — Classify ARC-AGI-3 game domain (8 types)."""

from __future__ import annotations

import numpy as np
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from arcengine import GameAction
except ImportError:
    class GameAction:
        ACTION1 = 1; ACTION2 = 2; ACTION3 = 3; ACTION4 = 4
        ACTION5 = 5; ACTION6 = 6; ACTION7 = 7; RESET = 0


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
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


def _diff_grid(a: np.ndarray, b: np.ndarray) -> Tuple[int, np.ndarray]:
    mask = a != b
    return int(mask.sum()), mask


@dataclass
class DomainResult:
    domain: str
    confidence: float
    evidence: Dict[str, Any]
    steps_taken: int


class DomainClassifierSkill:
    """Classify the abstract domain of an ARC-AGI-3 game environment."""

    def __init__(self, max_steps: int = 20):
        self.max_steps = max_steps
        self.obs = None
        self.steps_taken = 0
        self.actions_available: List[int] = []
        self.evidence: Dict[str, Any] = {}
        self.result: Optional[DomainResult] = None

    def classify(self, env) -> DomainResult:
        """Run diagnostic battery and return domain label."""
        self.obs = env.reset()
        self.actions_available = list(self.obs.available_actions or [])
        self.steps_taken = 0
        self.evidence = {}

        self._test_action_effects()
        self._test_order_matters()
        self._test_shapes()
        self._test_animation()
        self._test_agent()

        self._score()
        return self.result

    def _step(self, action) -> Any:
        self.steps_taken += 1
        return self.env.step(action)

    def _test_action_effects(self):
        for act_num in self.actions_available[:4]:
            if self.steps_taken >= self.max_steps:
                break
            before = self.obs.frame[0].copy()
            action = getattr(GameAction, f"ACTION{act_num}", None)
            if action is None:
                continue
            self.obs = self._step(action)
            after = self.obs.frame[0]
            changed, mask = _diff_grid(before, after)
            self.evidence.setdefault("action_effects", {})[act_num] = {
                "changed_pixels": changed,
                "changed_ratio": round(changed / (64 * 64), 4),
            }

    def _test_order_matters(self):
        if self.steps_taken + 6 > self.max_steps:
            return
        if len(self.actions_available) < 2:
            return
        self.evidence["order_matters"] = "inconclusive"
        self.evidence["order_note"] = "Cannot test without state save/restore"

    def _test_shapes(self):
        grid = self.obs.frame[0]
        unique_colors = np.unique(grid)
        self.evidence["unique_colors"] = int(len(unique_colors))
        self.evidence["max_color"] = int(grid.max())
        self.evidence["grid_density"] = round(float((grid != 0).mean()), 3)

    def _test_animation(self):
        n_frames = len(self.obs.frame)
        self.evidence["animation_frames"] = n_frames
        self.evidence["is_animated"] = n_frames > 1

    def _test_agent(self):
        if self.steps_taken + 2 > self.max_steps:
            return
        if not self.actions_available:
            return
        a1 = self.actions_available[0]
        b1 = self.obs.frame[0].copy()
        self.obs = self._step(getattr(GameAction, f"ACTION{a1}"))
        a2 = self.obs.frame[0]
        changed, mask = _diff_grid(b1, a2)
        self.evidence["agent_test"] = {
            "changed_after_one_action": changed,
            "changed_clusters": "analyze_later",
        }

    def _score(self):
        scores = {}
        e = self.evidence
        effects = e.get("action_effects", {})
        avg_change = np.mean([v["changed_pixels"] for v in effects.values()]) if effects else 0

        scores["movement"] = 0.7 if 1 < avg_change < 100 else 0.2
        scores["drawing"] = 0.8 if avg_change > 500 else 0.1
        scores["temporal"] = 0.6 if e.get("is_animated") else 0.1
        scores["symbolic"] = 0.5 if e.get("unique_colors", 0) > 5 and e.get("grid_density", 0) < 0.5 else 0.2
        scores["rotation"] = 0.4
        scores["physics_chain"] = 0.3
        scores["growth"] = 0.2
        scores["selection"] = 0.3

        best = max(scores, key=scores.get)
        self.result = DomainResult(
            domain=best,
            confidence=scores[best],
            evidence={**e, "domain_scores": scores},
            steps_taken=self.steps_taken,
        )

    def report(self) -> str:
        if not self.result:
            return "Not classified yet."
        lines = [
            f"Domain: {self.result.domain} (confidence: {self.result.confidence:.2f})",
            f"Steps: {self.result.steps_taken}",
            f"Actions available: {self.actions_available}",
        ]
        for k, v in self.result.evidence.get("action_effects", {}).items():
            lines.append(f"  ACTION{k}: {v['changed_pixels']} px changed ({v['changed_ratio']:.2%})")
        return "\n".join(lines)


def create_domain_classifier_skill(max_steps: int = 20) -> DomainClassifierSkill:
    return DomainClassifierSkill(max_steps=max_steps)