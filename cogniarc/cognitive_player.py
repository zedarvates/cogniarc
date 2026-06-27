#!/usr/bin/env python3
"""
Cognitive Player — Human-like drives for ARC-AGI-3 agents.

Six cognitive drives that transform a cold solver into a PLAYER:
  1. NOUVEAUTÉ (Curiosity) — explore states never seen before
  2. SIMPLICITÉ (Laziness) — shortest path is the best path
  3. DOUTE (Skepticism) — if too confident but stuck, scrap everything
  4. PLAISIR (Aesthetics) — attracted to symmetry, completion, order
  5. MÉMOIRE LIMITÉE (Miller's Law) — 7±2 states max, forces abstraction
  6. FATIGUE COGNITIVE (Budget) — limited thinking per level, then intuition

The genius: these LOOK like weaknesses but are actually SUPERIOR to unlimited
computation because they force the agent to FIND PATTERNS instead of memorizing
instances. This is exactly what JEPA/V-JEPA do — learn the world model, not
the data points.

Implementation: each drive returns a SCORE (0..1) that modulates action selection.
The agent picks the action that maximizes: goal_progress + Σ(drive_score × weight)
"""

import numpy as np
from collections import deque, OrderedDict
from typing import Optional
import hashlib
import json

from .common import get_working_memory_capacity


class WorkingMemory:
    """Limited working memory — Miller's Law: 7±2 items."""
    DEF_CAPACITY = get_working_memory_capacity()
    
    def __init__(self, capacity: int | None = None):
        cap = capacity if capacity is not None else self.DEF_CAPACITY
        if cap < 1:
            raise ValueError(f"WorkingMemory capacity must be >= 1, got {cap}")
        self.capacity = cap
        self._buffer: OrderedDict = OrderedDict()

    def remember(self, key: str, item):
        """Add item. If full, forget the oldest (LRU eviction)."""
        if key in self._buffer:
            del self._buffer[key]
        elif len(self._buffer) >= self.capacity:
            self._buffer.popitem(last=False)  # forget oldest
        self._buffer[key] = item
    
    def recall(self, key: str):
        """Recall item by key. Returns None if forgotten."""
        return self._buffer.get(key)
    
    def contains(self, key: str) -> bool:
        return key in self._buffer
    
    def snapshot(self) -> list:
        return list(self._buffer.values())
    
    def __len__(self):
        return len(self._buffer)


import os

class CognitiveFatigue:
    """Cognitive budget — limited thinking per level, then intuition takes over."""
    DEFAULT_BUDGET = 50

    def __init__(self, budget: int | None = None):
        if budget is None:
            budget = int(os.environ.get("COGNIARC_TOKEN_BUDGET", self.DEFAULT_BUDGET))
        self.initial_budget = budget
        self.remaining = budget
        self.fatigue_level = 0.0  # 0 = fresh, 1 = exhausted

    def spend(self, cost: int = 1):
        """Spend cognitive resource. Heavy planning costs more."""
        self.remaining -= cost
        self.fatigue_level = 1.0 - (max(0, self.remaining) / self.initial_budget)

    def reset(self):
        self.remaining = self.initial_budget
        self.fatigue_level = 0.0

    def is_exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def intuition_mode(self) -> bool:
        """After budget expires, act on intuition (greedy/random/heuristic)."""
        return self.fatigue_level > 0.7


class CognitiveDrives:
    """Six human-like cognitive drives that modulate decision-making."""
    
    def __init__(self):
        # Drive weights (tunable per game type)
        self.weights = {
            'novelty':    0.15,  # curiosity
            'simplicity': 0.20,  # laziness / Occam's razor
            'doubt':      0.25,  # skepticism (important for escaping loops!)
            'pleasure':   0.10,  # aesthetic satisfaction
            'caution':    0.15,  # don't waste steps
            'impulse':    0.15,  # sometimes just TRY things
        }
        
        # State tracking
        self.seen_states: set = set()
        self.visited_positions: set = set()
        self.action_history: list = []
        self.stagnation_counter: int = 0
        self.last_state_hash: str = ""
        self.total_steps: int = 0
        
        # ═══ NEW: Drive metrics + history ═══
        self.drive_history: list[dict] = []  # Snapshot périodique
        self.drive_values: dict[str, list[float]] = {
            drive: [] for drive in self.weights
        }  # Historique complet des valeurs
        
        # Confidence tracking
        self.world_model_confidence: float = 0.5
        self.goal_hypothesis_confidence: float = 0.3
        
        # Limited memory (7±2)
        self.memory = WorkingMemory(capacity=7)
        
        # Fatigue
        self.fatigue = CognitiveFatigue()
        
        # Doute déclencheur
        self.doubt_triggered: bool = False
        self.doubt_count: int = 0
    
    # ====== DRIVE 1: NOUVEAUTÉ (Curiosity) ======
    
    def novelty_score(self, state_hash: str) -> float:
        """How NEW is this state? 1.0 = completely novel, 0.0 = seen before."""
        if state_hash in self.seen_states:
            return 0.0
        # Partial novelty: hash parts to detect partial similarities
        novelty = 1.0
        for seen in list(self.seen_states)[-20:]:  # compare with recent
            if seen[:8] == state_hash[:8]:  # similar prefix
                novelty -= 0.3
        return max(0.0, novelty)
    
    def register_state(self, state_hash: str):
        self.seen_states.add(state_hash)
        self.memory.remember(state_hash, {'novel': True, 'hash': state_hash})
    
    # ====== DRIVE 2: SIMPLICITÉ (Laziness) ======
    
    def simplicity_score(self, plan_length: int) -> float:
        """Shorter plans are better. 1.0 = 1 step, decays with length."""
        if plan_length <= 0:
            return 1.0
        return max(0.0, 1.0 - (plan_length * 0.05))
    
    # ====== DRIVE 3: DOUTE (Skepticism) ======
    
    def doubt_check(self, stagnation: int, confidence: float) -> bool:
        """
        If I'm confident but STUCK, my confidence is FALSE.
        Trigger DOUBT: scrap current theory, restart exploration.
        
        This is the #1 fix for GPT-5.5's error #3 (stuck in wrong theory).
        """
        if stagnation > 5 and confidence > 0.7:
            self.doubt_triggered = True
            self.doubt_count += 1
            self.world_model_confidence = 0.2  # reset confidence
            self.goal_hypothesis_confidence = 0.1
            return True
        return False
    
    def doubt_score(self) -> float:
        """How much doubt is currently active? 0 = certain, 1 = everything is suspect."""
        if not self.doubt_triggered:
            return 0.0
        # Doubt decays over successful steps
        decay = min(1.0, self.doubt_count * 0.15)
        return max(0.0, 1.0 - decay * self.stagnation_counter)
    
    # ====== DRIVE 4: PLAISIR (Aesthetics) ======
    
    def pleasure_score(self, grid, player_pos=None) -> float:
        """
        How aesthetically PLEASING is this state?
        - Symmetry: mirrored patterns are satisfying
        - Completion: closed shapes feel "finished"
        - Order: low entropy = more organized
        """
        if grid is None:
            return 0.5  # neutral
        
        grid_np = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        
        # Symmetry: compare left/right halves
        h, w = grid_np.shape
        left = grid_np[:, :w//2]
        right = np.fliplr(grid_np[:, w//2 + w%2:])
        min_w = min(left.shape[1], right.shape[1])
        symmetry = np.mean(left[:, :min_w] == right[:, :min_w]) if min_w > 0 else 0.0
        
        # Completion: ratio of non-background pixels (more = more complete?)
        non_bg = np.sum(grid_np != 0)
        total = grid_np.size
        density = non_bg / total
        
        # Order: entropy of color distribution
        unique, counts = np.unique(grid_np, return_counts=True)
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        max_entropy = np.log2(len(unique) + 1)
        order = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        # Combined: symmetry is most important for "beauty"
        return 0.4 * symmetry + 0.3 * density + 0.3 * order
    
    # ====== DRIVE 5: CAUTION (Don't waste steps) ======
    
    def caution_score(self, action_id: int) -> float:
        """Penalize actions that were already tried recently with no effect."""
        recent = self.action_history[-10:]
        if len(recent) >= 3 and recent[-3:] == [action_id] * 3:
            return 0.0  # same action 3x in a row = stuck
        if recent.count(action_id) > len(recent) * 0.5:
            return 0.2  # too repetitive
        return 1.0
    
    # ====== DRIVE 6: IMPULSE (Just try it!) ======
    
    def impulse_score(self) -> float:
        """Random impulse to try something new. Increases with fatigue."""
        base_impulse = 0.1
        # Fatigue increases impulsivity (like a tired human)
        fatigue_bonus = self.fatigue.fatigue_level * 0.4
        # Doubt also increases impulsivity (try anything!)
        doubt_bonus = self.doubt_score() * 0.3
        return min(1.0, base_impulse + fatigue_bonus + doubt_bonus)
    
    # ====== COMPOSITE SCORING ======
    
    def score_action(self, action_id: int, state_hash: str, 
                     plan_length: int = 1, grid=None) -> float:
        """
        Compute the total cognitive score for an action.
        The agent picks the action with the HIGHEST score.
        
        This is NOT just "does it reach the goal?" — it's "does this action
        FEEL RIGHT to a human player?"
        """
        scores = {
            'novelty':    self.novelty_score(state_hash),
            'simplicity': self.simplicity_score(plan_length),
            'doubt':      self.doubt_score(),
            'pleasure':   self.pleasure_score(grid),
            'caution':    self.caution_score(action_id),
            'impulse':    self.impulse_score(),
        }
        
        # Weighted sum
        total = sum(
            scores[drive] * self.weights[drive]
            for drive in self.weights
        )
        
        # If fatigue is high, impulse dominates (intuition mode)
        if self.fatigue.intuition_mode:
            total = 0.6 * scores['impulse'] + 0.4 * total
        
        # ═══ NEW: Log drive values to history ═══
        for drive, val in scores.items():
            self.drive_values[drive].append(val)
        
        return total
    
    def snapshot_drives(self) -> dict:
        """Snapshot des valeurs courantes des drives (pour logging/benchmark)."""
        return {
            'novelty': self.novelty_score(self.last_state_hash) if self.last_state_hash else 0.5,
            'simplicity': self.simplicity_score(len(self.action_history[-5:]) if self.action_history else 1),
            'doubt': self.doubt_score(),
            'pleasure': self.pleasure_score(None),
            'caution': self.caution_score(self.action_history[-1] if self.action_history else 1),
            'impulse': self.impulse_score(),
            'stagnation': self.stagnation_counter,
            'fatigue': self.fatigue.fatigue_level,
            'confidence': self.world_model_confidence,
            'step': self.total_steps,
        }
    
    def log_snapshot(self):
        """Enregistre un snapshot périodique dans drive_history."""
        snap = self.snapshot_drives()
        self.drive_history.append(snap)
        
    def status_report(self) -> str:
        """Rapport formaté pour debug/affichage."""
        snap = self.snapshot_drives()
        lines = ["📊 Cognitive Drives Status:"]
        for drive in ['novelty', 'simplicity', 'doubt', 'pleasure', 'caution', 'impulse']:
            val = snap[drive]
            bar = '█' * int(val * 10) + '░' * (10 - int(val * 10))
            lines.append(f"  {drive:12s} [{bar}] {val:.2f}")
        lines.append(f"  {'stagnation':12s}  {snap['stagnation']}")
        lines.append(f"  {'fatigue':12s}  {snap['fatigue']:.2f}")
        lines.append(f"  {'confidence':12s}  {snap['confidence']:.2f}")
        return '\n'.join(lines)
    
    # ====== STATE MANAGEMENT ======
    
    def step(self, action_id: int, state_hash: str, grid=None):
        """Called after each action to update internal state."""
        self.total_steps += 1
        self.action_history.append(action_id)
        self.visited_positions.add(state_hash[:16])
        self.fatigue.spend(1)
        
        # Track stagnation
        if state_hash == self.last_state_hash:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
            # Gradually rebuild confidence after a change
            self.world_model_confidence = min(1.0, self.world_model_confidence + 0.02)
        
        self.last_state_hash = state_hash
        
        # Doubt check
        self.doubt_check(self.stagnation_counter, self.world_model_confidence)
        
        # Register novelty
        self.register_state(state_hash)
    
    def reset(self):
        """Reset all drives (for new game or after doubt-triggered restart)."""
        self.seen_states.clear()
        self.visited_positions.clear()
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
    
    def status_report(self) -> str:
        """Human-readable status of all cognitive drives."""
        return (
            f"Cognitive Status (step {self.total_steps}):\n"
            f"  Novelty: seen {len(self.seen_states)} states, "
            f"memory: {len(self.memory)}/7\n"
            f"  Fatigue: {self.fatigue.fatigue_level:.1%} "
            f"({'INTUITION' if self.fatigue.intuition_mode else 'planning'})\n"
            f"  Confidence: world={self.world_model_confidence:.1%} "
            f"goal={self.goal_hypothesis_confidence:.1%}\n"
            f"  Stagnation: {self.stagnation_counter} steps "
            f"({'DOUBT!' if self.doubt_triggered else 'ok'})\n"
            f"  Action history: last 5 = {self.action_history[-5:]}"
        )


def hash_grid(grid) -> str:
    """Fast hash of a grid state for comparison."""
    if grid is None:
        return "no_grid"
    np_grid = np.array(grid) if not isinstance(grid, np.ndarray) else grid
    return hashlib.md5(np_grid.tobytes()).hexdigest()


# ====== Quick Test ======
if __name__ == "__main__":
    print("=== Cognitive Drives Test ===\n")
    
    drives = CognitiveDrives()
    
    # Simulate exploration
    test_grids = [
        np.zeros((5, 5)),           # empty
        np.eye(5),                  # diagonal (some structure)
        np.ones((5, 5)),           # full
        np.array([[0,0,1,0,0],     # symmetric
                  [0,1,0,1,0],
                  [1,0,0,0,1],
                  [0,1,0,1,0],
                  [0,0,1,0,0]]),
    ]
    
    for i, grid in enumerate(test_grids):
        h = hash_grid(grid)
        novelty = drives.novelty_score(h)
        pleasure = drives.pleasure_score(grid)
        
        # Score 4 actions
        action_scores = {}
        for act in range(1, 5):
            action_scores[act] = drives.score_action(act, h, grid=grid)
        
        print(f"State {i}: novelty={novelty:.2f} pleasure={pleasure:.2f}")
        print(f"  Actions: { {a: f'{s:.3f}' for a, s in action_scores.items()} }")
        
        drives.step(i + 1, h, grid)
    
    print(f"\n{drives.status_report()}")
    
    # Test doubt trigger
    print("\n--- Testing doubt ---")
    for _ in range(10):
        drives.step(1, "same_hash_repeated", np.zeros((5,5)))
    print(drives.status_report())
    print(f"Doubt triggered: {drives.doubt_triggered}")
