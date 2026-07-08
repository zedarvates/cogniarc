"""Stagnation Detector Skill — Detect/escapes agent stagnation."""

from __future__ import annotations

import time
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field


@dataclass
class StagnationConfig:
    window: int = 20
    repeat_threshold: int = 3
    no_progress_steps: int = 15
    state_loop_threshold: int = 4
    novelty_threshold: float = 0.2


@dataclass
class StagnationReport:
    is_stuck: bool
    reasons: List[str]
    stuck_count: int
    escape_attempts: int
    novelty: float
    states_seen: int
    action_counts: Dict[int, int]
    level_progress: List[int]
    untried_actions: List[int]


class StagnationDetectorSkill:
    """Detect and escape from agent stagnation."""

    def __init__(self, config: StagnationConfig = None):
        self.config = config or StagnationConfig()
        self.action_history: List[Dict[str, Any]] = []
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.action_counts: Dict[int, int] = defaultdict(int)
        self.levels_progress: List[int] = [0]

        self.last_novelty: float = 1.0
        self.stuck_count: int = 0
        self.escape_attempts: int = 0

        self.tried: Set[Tuple[str, int]] = set()
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

        if len(self.action_history) > self.config.window * 3:
            self.action_history = self.action_history[-self.config.window * 2:]

        self.state_visits[state_hash] += 1
        self.action_counts[action_id] += 1

        if levels_completed != self.levels_progress[-1]:
            self.levels_progress.append(levels_completed)

        self.tried.add((state_hash[:8], action_id))
        self.untried_actions = [a for a in available_actions
                                if (state_hash[:8], a) not in self.tried]

        self._update_novelty()

    def is_stuck(self) -> bool:
        """Detect if agent is stagnating."""
        reasons = []

        if self._repeated_actions():
            reasons.append("repeated_actions")

        if self._no_progress():
            reasons.append("no_progress")

        if self._state_loop():
            reasons.append("state_loop")

        if self.last_novelty < self.config.novelty_threshold:
            reasons.append("low_novelty")

        if reasons:
            self.stuck_count += 1
            return True
        return False

    def _repeated_actions(self) -> bool:
        recent = [h["action"] for h in self.action_history[-self.config.repeat_threshold:]]
        return len(recent) >= self.config.repeat_threshold and len(set(recent)) == 1

    def _no_progress(self) -> bool:
        if len(self.levels_progress) < 2:
            return False
        recent = self.action_history[-self.config.no_progress_steps:]
        levels = [h["levels"] for h in recent]
        return len(set(levels)) == 1 and len(recent) >= self.config.no_progress_steps

    def _state_loop(self) -> bool:
        recent_states = [h["state"] for h in self.action_history[-10:]]
        if len(recent_states) < 5:
            return False
        most_common = max(set(recent_states), key=recent_states.count)
        return recent_states.count(most_common) >= self.config.state_loop_threshold

    def _update_novelty(self):
        recent = [h["state"] for h in self.action_history[-self.config.window:]]
        if len(recent) < 3:
            self.last_novelty = 1.0
            return
        counts = defaultdict(int)
        for s in recent:
            counts[s] += 1
        total = len(recent)
        self.last_novelty = len(counts) / total

    def escape_action(self, available_actions: List[int]) -> int:
        """Choose action to break stagnation."""
        self.escape_attempts += 1

        if self.untried_actions:
            return self.untried_actions[0]

        if available_actions:
            counts = {a: self.action_counts.get(a, 0) for a in available_actions}
            return min(counts, key=counts.get)

        return random.choice(available_actions) if available_actions else 1

    def should_explore(self) -> bool:
        """Should we explore (try new) or exploit (use known path)?"""
        return (self.stuck_count > 0 or
                self.last_novelty < 0.3 or
                len(self.untried_actions) > 0)

    def report(self) -> StagnationReport:
        return StagnationReport(
            is_stuck=self.is_stuck(),
            reasons=self._get_reasons(),
            stuck_count=self.stuck_count,
            escape_attempts=self.escape_attempts,
            novelty=self.last_novelty,
            states_seen=len(self.state_visits),
            action_counts=dict(self.action_counts),
            level_progress=self.levels_progress,
            untried_actions=self.untried_actions,
        )

    def _get_reasons(self) -> List[str]:
        reasons = []
        if self._repeated_actions():
            reasons.append("repeated_actions")
        if self._no_progress():
            reasons.append("no_progress")
        if self._state_loop():
            reasons.append("state_loop")
        if self.last_novelty < self.config.novelty_threshold:
            reasons.append("low_novelty")
        return reasons

    def reset(self):
        self.action_history.clear()
        self.state_visits.clear()
        self.action_counts.clear()
        self.levels_progress = [0]
        self.stuck_count = 0
        self.escape_attempts = 0
        self.tried.clear()
        self.untried_actions.clear()
        self.last_novelty = 1.0


def create_stagnation_detector_skill(window: int = 20,
                                      repeat_threshold: int = 3) -> StagnationDetectorSkill:
    return StagnationDetectorSkill(StagnationConfig(
        window=window,
        repeat_threshold=repeat_threshold
    ))