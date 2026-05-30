#!/usr/bin/env python3
"""
Stagnation Detector for ARC-AGI-3.

Prevents infinite loops. Detects when the agent is:
  - Repeating the same actions with no progress
  - Stuck in a state with no novel transitions
  - Making no progress toward level completion

When stuck, triggers REFLECTION:
  - Try a DIFFERENT action (explore unexplored branches)
  - Seek NOVEL states (states not seen before = high value)
  - Balance EXPLORE vs EXPLOIT dynamically

Usage:
    from stagnation_detector import StagnationDetector
    sd = StagnationDetector()
    sd.record(action, state_hash, levels_completed)
    if sd.is_stuck():
        action = sd.escape_action(available_actions)
"""

from __future__ import annotations

import json
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Any
from pathlib import Path


class StagnationDetector:
    """Detect and escape from agent stagnation."""

    def __init__(self, window: int = 20, repeat_threshold: int = 3):
        self.window = window
        self.repeat_threshold = repeat_threshold

        # History
        self.action_history: List[Dict[str, Any]] = []
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.action_counts: Dict[int, int] = defaultdict(int)
        self.levels_progress: List[int] = [0]

        # Stagnation signals
        self.last_novelty: float = 1.0
        self.stuck_count: int = 0
        self.escape_attempts: int = 0

        # Unexplored branches
        self.tried: Set[tuple] = set()  # (state_hash, action)
        self.untried_actions: List[int] = []

    def record(self, action_id: int, state_hash: str,
               levels_completed: int, available_actions: List[int]):
        """Record one agent step."""
        self.action_history.append({
            "action": action_id,
            "state": state_hash[:8],
            "levels": levels_completed,
            "time": time.time(),
        })

        # Trim history
        if len(self.action_history) > self.window * 3:
            self.action_history = self.action_history[-self.window * 2:]

        # Track state visits
        self.state_visits[state_hash] += 1
        self.action_counts[action_id] += 1

        # Track progress
        if levels_completed != self.levels_progress[-1]:
            self.levels_progress.append(levels_completed)

        # Track tried combinations
        self.tried.add((state_hash[:8], action_id))

        # Available actions
        self.untried_actions = [a for a in available_actions
                                if (state_hash[:8], a) not in self.tried]

        # Compute novelty (entropy of recent states)
        self._update_novelty()

    # ── Detection ─────────────────────────────────────────

    def is_stuck(self) -> bool:
        """Detect if agent is stagnating."""
        reasons = []

        # Check 1: Recent actions are identical?
        if self._repeated_actions():
            reasons.append("repeated_actions")

        # Check 2: No progress in N steps?
        if self._no_progress(steps=15):
            reasons.append("no_progress")

        # Check 3: State revisited too many times?
        if self._state_loop():
            reasons.append("state_loop")

        # Check 4: Novelty collapsed?
        if self.last_novelty < 0.2:
            reasons.append("low_novelty")

        if reasons:
            self.stuck_count += 1
            return True
        return False

    def _repeated_actions(self) -> bool:
        """Last N actions are all the same?"""
        recent = [h["action"] for h in self.action_history[-self.repeat_threshold:]]
        return len(recent) >= self.repeat_threshold and len(set(recent)) == 1

    def _no_progress(self, steps: int = 15) -> bool:
        """Zero level progress in last N steps?"""
        if len(self.levels_progress) < 2:
            return False
        recent = self.action_history[-steps:]
        levels = [h["levels"] for h in recent]
        return len(set(levels)) == 1 and len(recent) >= steps

    def _state_loop(self) -> bool:
        """Visiting the same state repeatedly?"""
        recent_states = [h["state"] for h in self.action_history[-10:]]
        if len(recent_states) < 5:
            return False
        most_common = max(set(recent_states), key=recent_states.count)
        return recent_states.count(most_common) >= 4

    def _update_novelty(self):
        """Compute entropy of recent state visits."""
        recent = [h["state"] for h in self.action_history[-self.window:]]
        if len(recent) < 3:
            self.last_novelty = 1.0
            return
        counts = defaultdict(int)
        for s in recent:
            counts[s] += 1
        total = len(recent)
        # Simple: ratio of unique states
        self.last_novelty = len(counts) / total

    # ── Escape ────────────────────────────────────────────

    def escape_action(self, available_actions: List[int]) -> int:
        """
        Choose an action to break out of stagnation.
        Prioritize: untried > low-count > random.
        """
        self.escape_attempts += 1

        # Priority 1: Try an action we haven't tried in this state
        if self.untried_actions:
            return self.untried_actions[0]

        # Priority 2: Try the least-used action
        if available_actions:
            counts = {a: self.action_counts.get(a, 0) for a in available_actions}
            return min(counts, key=counts.get)

        # Priority 3: Random
        import random
        return random.choice(available_actions) if available_actions else 1

    def should_explore(self) -> bool:
        """Should we explore (try new things) or exploit (use known path)?"""
        if self.stuck_count > 0:
            return True
        if self.last_novelty < 0.3:
            return True
        if self.untried_actions:
            return True
        return False

    # ── Report ────────────────────────────────────────────

    def report(self) -> str:
        lines = [
            f"Stagnation: {'STUCK' if self.is_stuck() else 'OK'}",
            f"  Stuck count: {self.stuck_count}",
            f"  Escape attempts: {self.escape_attempts}",
            f"  Novelty: {self.last_novelty:.2f}",
            f"  States seen: {len(self.state_visits)}",
            f"  Actions taken: {len(self.action_history)}",
            f"  Levels progress: {self.levels_progress}",
            f"  Untried in current state: {len(self.untried_actions)}",
        ]
        return "\n".join(lines)

    def save(self, path: str = "/home/redgamer/arc_agi_agent/stagnation_state.json"):
        Path(path).write_text(json.dumps({
            "stuck_count": self.stuck_count,
            "escape_attempts": self.escape_attempts,
            "novelty": self.last_novelty,
            "states_seen": len(self.state_visits),
            "levels_progress": self.levels_progress,
        }, indent=2))

    def reset(self):
        """Reset for a new game."""
        self.action_history.clear()
        self.state_visits.clear()
        self.action_counts.clear()
        self.levels_progress = [0]
        self.stuck_count = 0
        self.escape_attempts = 0
        self.tried.clear()
        self.untried_actions.clear()
