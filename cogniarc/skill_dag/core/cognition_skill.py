"""Cognition Skill — 6 cognitive drives + Sternberg triarchic balance."""

from __future__ import annotations

import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import hashlib


@dataclass
class WorkingMemory:
    """Limited working memory — Miller's Law: 7±2 items."""
    capacity: int = 7
    _buffer: OrderedDict = field(default_factory=OrderedDict)

    def remember(self, key: str, item: Any):
        if key in self._buffer:
            del self._buffer[key]
        elif len(self._buffer) >= self.capacity:
            self._buffer.popitem(last=False)
        self._buffer[key] = item

    def recall(self, key: str) -> Any:
        return self._buffer.get(key)

    def contains(self, key: str) -> bool:
        return key in self._buffer

    def snapshot(self) -> List:
        return list(self._buffer.values())

    def __len__(self) -> int:
        return len(self._buffer)


@dataclass
class CognitiveFatigue:
    """Cognitive budget — limited thinking then intuition takes over."""
    initial_budget: int = 50
    remaining: int = 50
    fatigue_level: float = 0.0

    def spend(self, cost: int = 1):
        self.remaining -= cost
        self.fatigue_level = 1.0 - (max(0, self.remaining) / self.initial_budget)

    def reset(self):
        self.remaining = self.initial_budget
        self.fatigue_level = 0.0

    @property
    def intuition_mode(self) -> bool:
        return self.fatigue_level > 0.7


@dataclass
class TriarchicState:
    """Sternberg's three intelligences balance."""
    analytical: float = 0.33
    creative: float = 0.33
    practical: float = 0.33

    @property
    def balance_score(self) -> float:
        vals = [self.analytical, self.creative, self.practical]
        if max(vals) == 0:
            return 0.0
        return min(vals) / max(vals)

    @property
    def dominant_mode(self) -> str:
        modes = {'analytical': self.analytical, 'creative': self.creative, 'practical': self.practical}
        return max(modes, key=lambda k: modes[k])

    @property
    def weakest_mode(self) -> str:
        modes = {'analytical': self.analytical, 'creative': self.creative, 'practical': self.practical}
        return min(modes.keys(), key=lambda k: modes[k])

    def recommendation(self) -> str:
        """What should the agent do to rebalance?"""
        if self.balance_score > 0.7:
            return "✅ Triarchic balance OK — all three modes active"

        recs = {
            'analytical': "🔍 Boost ANALYTICAL: pause and critique your plan. Ask 'What am I missing?'",
            'creative': "🎨 Boost CREATIVE: reframe the problem entirely. Ask 'Is there a completely different approach?'",
            'practical': "⚡ Boost PRACTICAL: stop theorizing and act. Ask 'What's the smallest step I can take RIGHT NOW?'",
        }
        return recs[self.weakest_mode]


class CognitiveDrives:
    """Six human-like cognitive drives modulating decisions."""

    def __init__(self):
        self.weights = {
            'novelty': 0.15,      # curiosity
            'simplicity': 0.20,   # laziness / Occam's razor
            'doubt': 0.25,        # skepticism
            'pleasure': 0.10,     # aesthetic satisfaction
            'caution': 0.15,      # don't waste steps
            'impulse': 0.15,      # just try things
        }

        self.seen_states: set = set()
        self.action_history: List[int] = []
        self.stagnation_counter: int = 0
        self.last_state_hash: str = ""
        self.total_steps: int = 0

        self.world_model_confidence: float = 0.5
        self.goal_hypothesis_confidence: float = 0.3

        self.memory = WorkingMemory(capacity=7)
        self.fatigue = CognitiveFatigue(initial_budget=50)

        self.doubt_triggered: bool = False
        self.doubt_count: int = 0

        # Drive-to-triarchic mapping
        self.drive_to_mode = {
            'novelty': 'creative',
            'simplicity': 'practical',
            'doubt': 'analytical',
            'pleasure': 'creative',
            'caution': 'analytical',
            'impulse': 'creative',
        }

    # DRIVE 1: NOVELTY (Curiosity)
    def novelty_score(self, state_hash: str) -> float:
        if state_hash in self.seen_states:
            return 0.0
        novelty = 1.0
        for seen in list(self.seen_states)[-20:]:
            if seen[:8] == state_hash[:8]:
                novelty -= 0.3
        return max(0.0, novelty)

    def register_state(self, state_hash: str):
        self.seen_states.add(state_hash)
        self.memory.remember(state_hash, {'novel': True, 'hash': state_hash})

    # DRIVE 2: SIMPLICITY (Laziness)
    def simplicity_score(self, plan_length: int) -> float:
        if plan_length <= 0:
            return 1.0
        return max(0.0, 1.0 - (plan_length * 0.05))

    # DRIVE 3: DOUBT (Skepticism)
    def doubt_check(self, stagnation: int, confidence: float) -> bool:
        if stagnation > 5 and confidence > 0.7:
            self.doubt_triggered = True
            self.doubt_count += 1
            self.world_model_confidence = 0.2
            self.goal_hypothesis_confidence = 0.1
            return True
        return False

    def doubt_score(self) -> float:
        if not self.doubt_triggered:
            return 0.0
        decay = min(1.0, self.doubt_count * 0.15)
        return max(0.0, 1.0 - decay * self.stagnation_counter)

    # DRIVE 4: PLEASURE (Aesthetics)
    def pleasure_score(self, grid: Optional[np.ndarray]) -> float:
        if grid is None:
            return 0.5
        grid_np = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        h, w = grid_np.shape

        # Symmetry
        left = grid_np[:, :w//2]
        right = np.fliplr(grid_np[:, w//2 + w%2:])
        min_w = min(left.shape[1], right.shape[1])
        symmetry = np.mean(left[:, :min_w] == right[:, :min_w]) if min_w > 0 else 0.0

        # Density
        non_bg = np.sum(grid_np != 0)
        density = non_bg / grid_np.size

        # Order (inverse entropy)
        unique, counts = np.unique(grid_np, return_counts=True)
        probs = counts / grid_np.size
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        max_entropy = np.log2(len(unique) + 1)
        order = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        return 0.4 * symmetry + 0.3 * density + 0.3 * order

    # DRIVE 5: CAUTION
    def caution_score(self, action_id: int) -> float:
        recent = self.action_history[-10:]
        if len(recent) >= 3 and recent[-3:] == [action_id] * 3:
            return 0.0
        if recent.count(action_id) > len(recent) * 0.5:
            return 0.2
        return 1.0

    # DRIVE 6: IMPULSE
    def impulse_score(self) -> float:
        base = 0.1
        fatigue_bonus = self.fatigue.fatigue_level * 0.4
        doubt_bonus = self.doubt_score() * 0.3
        return min(1.0, base + fatigue_bonus + doubt_bonus)

    # COMPOSITE SCORING
    def score_action(self, action_id: int, state_hash: str,
                     plan_length: int = 1, grid: Optional[np.ndarray] = None) -> float:
        scores = {
            'novelty': self.novelty_score(state_hash),
            'simplicity': self.simplicity_score(plan_length),
            'doubt': self.doubt_score(),
            'pleasure': self.pleasure_score(grid),
            'caution': self.caution_score(action_id),
            'impulse': self.impulse_score(),
        }

        total = sum(scores[drive] * self.weights[drive] for drive in self.weights)

        if self.fatigue.intuition_mode:
            total = 0.6 * scores['impulse'] + 0.4 * total

        return total

    # STATE MANAGEMENT
    def step(self, action_id: int, state_hash: str, grid: Optional[np.ndarray] = None):
        self.total_steps += 1
        self.action_history.append(action_id)
        self.fatigue.spend(1)

        if state_hash == self.last_state_hash:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
            self.world_model_confidence = min(1.0, self.world_model_confidence + 0.02)

        self.last_state_hash = state_hash
        self.doubt_check(self.stagnation_counter, self.world_model_confidence)
        self.register_state(state_hash)

    def reset(self):
        self.seen_states.clear()
        self.action_history.clear()
        self.stagnation_counter = 0
        self.last_state_hash = ""
        self.total_steps = 0
        self.world_model_confidence = 0.5
        self.goal_hypothesis_confidence = 0.3
        self.memory = WorkingMemory(capacity=7)
        self.fatigue.reset()
        self.doubt_triggered = False
        self.doubt_count = 0

    def get_drive_scores(self, state_hash: str, grid: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get all 6 drive scores for triarchic update."""
        return {
            'novelty': self.novelty_score(state_hash),
            'simplicity': self.simplicity_score(1),
            'doubt': self.doubt_score(),
            'pleasure': self.pleasure_score(grid),
            'caution': self.caution_score(1),
            'impulse': self.impulse_score(),
        }

    def status_report(self) -> str:
        return (
            f"Cognitive Status (step {self.total_steps}):\n"
            f"  Novelty: seen {len(self.seen_states)} states, memory: {len(self.memory)}/7\n"
            f"  Fatigue: {self.fatigue.fatigue_level:.1%} "
            f"({'INTUITION' if self.fatigue.intuition_mode else 'planning'})\n"
            f"  Confidence: world={self.world_model_confidence:.1%} "
            f"goal={self.goal_hypothesis_confidence:.1%}\n"
            f"  Stagnation: {self.stagnation_counter} steps "
            f"({'DOUBT!' if self.doubt_triggered else 'ok'})\n"
            f"  Action history: last 5 = {self.action_history[-5:]}"
        )


class TriarchicEngine:
    """Enforces Sternberg's triarchic balance on cognitive drives."""

    def __init__(self):
        self.state = TriarchicState()
        self.history: List[Dict] = []
        self.stagnation_counter: int = 0
        self.last_recommendation: str = ""

        self.drive_to_mode = {
            'novelty': 'creative',
            'simplicity': 'practical',
            'doubt': 'analytical',
            'pleasure': 'creative',
            'caution': 'analytical',
            'impulse': 'creative',
        }

    def update(self, drive_scores: Dict[str, float]) -> TriarchicState:
        analytical_drives = []
        creative_drives = []
        practical_drives = []

        for drive, score in drive_scores.items():
            mode = self.drive_to_mode.get(drive, 'analytical')
            if mode == 'analytical':
                analytical_drives.append(score)
            elif mode == 'creative':
                creative_drives.append(score)
            else:
                practical_drives.append(score)

        self.state.analytical = float(np.mean(analytical_drives)) if analytical_drives else 0.33
        self.state.creative = float(np.mean(creative_drives)) if creative_drives else 0.33
        self.state.practical = float(np.mean(practical_drives)) if practical_drives else 0.33

        self.last_recommendation = self._recommendation()
        self.history.append({
            'analytical': round(self.state.analytical, 3),
            'creative': round(self.state.creative, 3),
            'practical': round(self.state.practical, 3),
            'balance': round(self.state.balance_score, 3),
        })

        if len(self.history) >= 3:
            recent = self.history[-3:]
            if all(abs(h['balance'] - recent[0]['balance']) < 0.05 for h in recent):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

        return self.state

    def _recommendation(self) -> str:
        if self.state.balance_score > 0.7:
            return "✅ Triarchic balance OK — all three modes active"

        recs = {
            'analytical': "🔍 Boost ANALYTICAL: pause and critique your plan. Ask 'What am I missing?'",
            'creative': "🎨 Boost CREATIVE: reframe the problem entirely. Ask 'Is there a completely different approach?'",
            'practical': "⚡ Boost PRACTICAL: stop theorizing and act. Ask 'What's the smallest step RIGHT NOW?'",
        }
        return recs[self.state.weakest_mode]

    def needs_reframe(self) -> bool:
        return self.stagnation_counter > 8 and self.state.balance_score < 0.4

    def reframe_prompt(self) -> str:
        prompts = [
            "What if the problem isn't what we think it is?",
            "What frame am I using that might be wrong?",
            "If I had to solve this with the OPPOSITE approach, what would it be?",
            "What would Faraday/Douglass see that I'm missing?",
        ]
        return prompts[self.stagnation_counter % len(prompts)]

    def status_report(self) -> str:
        te = self.state
        return (
            f"🧠 Triarchic Balance (Sternberg WICS):\n"
            f"  Analytical: {'█' * int(te.analytical * 20)}{'░' * (20 - int(te.analytical * 20))} {te.analytical:.1%}\n"
            f"  Creative:   {'█' * int(te.creative * 20)}{'░' * (20 - int(te.creative * 20))} {te.creative:.1%}\n"
            f"  Practical:  {'█' * int(te.practical * 20)}{'░' * (20 - int(te.practical * 20))} {te.practical:.1%}\n"
            f"  Balance: {te.balance_score:.1%} | Dominant: {te.dominant_mode} | Need: {te.weakest_mode}\n"
            f"  {self.last_recommendation}"
        )


class CognitionSkill:
    """Unified cognition skill combining drives + triarchic balance."""

    def __init__(self):
        self.drives = CognitiveDrives()
        self.triarchic = TriarchicEngine()

    def evaluate_action(self, action_id: int, state_hash: str,
                        plan_length: int = 1, grid: Optional[np.ndarray] = None) -> float:
        """Score an action through all cognitive drives."""
        return self.drives.score_action(action_id, state_hash, plan_length, grid)

    def step(self, action_id: int, state_hash: str, grid: Optional[np.ndarray] = None):
        """Update internal state after action."""
        self.drives.step(action_id, state_hash, grid)

        # Update triarchic balance
        drive_scores = self.drives.get_drive_scores(state_hash, grid)
        self.triarchic.update(drive_scores)

    def get_balance(self) -> TriarchicState:
        return self.triarchic.state

    def get_recommendation(self) -> str:
        return self.triarchic.last_recommendation or self.triarchic.state.recommendation()

    def needs_reframe(self) -> bool:
        return self.triarchic.needs_reframe()

    def get_reframe_prompt(self) -> str:
        return self.triarchic.reframe_prompt()

    def reset(self):
        self.drives.reset()
        self.triarchic = TriarchicEngine()

    def status_report(self) -> str:
        return f"{self.drives.status_report()}\n\n{self.triarchic.status_report()}"


def create_cognition_skill() -> CognitionSkill:
    return CognitionSkill()