#!/usr/bin/env python3
"""
ScientificState — Shared scientific state for ARC-AGI-3 agents.

Inspired by AHOIS (arXiv:2606.26722) "shared scientific state":
hypothesis, assumptions, action_plan, expected_observation, uncertainty, evidence.

This replaces ad-hoc variables scattered across ScientistAgent with a
structured, inspectable, transferable state object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum, auto
import time
import typing


class EvidenceReliability(Enum):
    """How reliable is a piece of evidence?"""
    CONFIRMED = auto()       # Verified multiple times
    SINGLE_OBSERVATION = auto()  # Seen once
    INFERRED = auto()        # Deduced, not directly observed
    HYPOTHETICAL = auto()    # Guessed / assumed


@dataclass
class Observation:
    """A single observed fact about the environment."""
    description: str
    reliability: EvidenceReliability = EvidenceReliability.CONFIRMED
    source: str = ""           # "step_42", "source_code", "domain_profiler"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Current theory about how the game works."""
    description: str
    confidence: float = 0.5     # 0.0 (wild guess) to 1.0 (certain)
    supporting_evidence: List[str] = field(default_factory=list)
    refuted_by: List[str] = field(default_factory=list)
    falsification_criteria: List[str] = field(default_factory=list)
    created_at: float = 0.0
    version: int = 1


@dataclass
class ActionPlan:
    """Planned sequence of actions and their expected outcome."""
    actions: List[int] = field(default_factory=list)
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    success: Optional[bool] = None


# Lazy import for SocraticReport to avoid circular dependency
def _get_socratic_report_class():
    """Import SocraticReport lazily to avoid circular imports."""
    try:
        from .socratic_critic import SocraticReport
        return SocraticReport
    except (ImportError, RuntimeError):
        return None


@dataclass
class ScientificState:
    """
    Shared scientific state — the agent's current understanding.

    Pattern: AHOIS shared scientific state
    (hypothesis, assumptions, action, expected_observation, uncertainty, evidence)
    """

    # ── Core hypothesis ──
    current_hypothesis: Optional[Hypothesis] = None
    refuted_hypotheses: List[Hypothesis] = field(default_factory=list)

    # ── Assumptions (unverified beliefs) ──
    assumptions: Dict[str, bool] = field(default_factory=dict)
    # E.g. {"player_can_walk_through_locks": False, "action_6_rotates": True}

    # ── Current plan ──
    current_plan: Optional[ActionPlan] = None
    plan_history: List[ActionPlan] = field(default_factory=list)

    # ── Evidence ──
    evidence: List[Observation] = field(default_factory=list)
    expected_outcome: Optional[str] = None

    # ── Uncertainty ──
    uncertainty: float = 1.0       # 0.0 = certain, 1.0 = blind
    domain_confidence: float = 0.5  # How sure are we of the domain type?
    strategy_confidence: float = 0.3  # How sure are we of the chosen strategy?

    # ── Execution tracking ──
    steps_taken: int = 0
    stagnation_count: int = 0
    last_action: Optional[int] = None
    last_state_hash: str = ""

    # ── Meta ──
    phase: str = "discovery"        # discovery → planning → execution → verification
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ── Socratic critic reports ──
    critic_reports: List[typing.Any] = field(default_factory=list)

    # ── Phase tracking ──
    phase_attempts: int = 0

    # ── Wall detection ──
    walls_detected: bool = False

    # ── Game metadata ──
    game_name: str = ""
    domain_type: str = ""           # "movement", "rotation", "hybrid", "temporal"
    available_actions: List[int] = field(default_factory=list)

    def record_observation(self, description: str,
                           reliability: EvidenceReliability = EvidenceReliability.CONFIRMED,
                           source: str = "",
                           details: Optional[Dict[str, Any]] = None) -> Observation:
        """Record a new observation."""
        obs = Observation(
            description=description,
            reliability=reliability,
            source=source,
            details=details or {},
        )
        self.evidence.append(obs)
        self.updated_at = time.time()
        return obs

    def update_hypothesis(self, description: str, confidence: float = 0.5,
                          falsification_criteria: Optional[List[str]] = None) -> Hypothesis:
        """Set or update the current hypothesis. Old one is archived."""
        if self.current_hypothesis is not None:
            self.refuted_hypotheses.append(self.current_hypothesis)

        h = Hypothesis(
            description=description,
            confidence=confidence,
            falsification_criteria=falsification_criteria or [],
            created_at=time.time(),
            version=(self.current_hypothesis.version + 1 if self.current_hypothesis else 1),
        )
        self.current_hypothesis = h
        self.updated_at = time.time()
        return h

    def refute_current_hypothesis(self, reason: str):
        """Mark current hypothesis as refuted and archive it."""
        if self.current_hypothesis is not None:
            self.current_hypothesis.refuted_by.append(reason)
            self.refuted_hypotheses.append(self.current_hypothesis)
            self.current_hypothesis = None
        self.uncertainty = min(1.0, self.uncertainty + 0.2)
        self.updated_at = time.time()

    def set_assumption(self, name: str, value: bool):
        """Record an assumption."""
        self.assumptions[name] = value
        self.updated_at = time.time()

    def get_assumption(self, name: str, default: bool = False) -> bool:
        """Get an assumption value."""
        return self.assumptions.get(name, default)

    def record_plan(self, actions: List[int], expected_outcome: Optional[str] = None) -> ActionPlan:
        """Record a new action plan."""
        plan = ActionPlan(
            actions=actions,
            expected_outcome=expected_outcome,
        )
        if self.current_plan is not None:
            self.plan_history.append(self.current_plan)
        self.current_plan = plan
        self.updated_at = time.time()
        return plan

    def complete_plan(self, outcome: str, success: bool):
        """Mark current plan as complete with actual outcome."""
        if self.current_plan is not None:
            self.current_plan.actual_outcome = outcome
            self.current_plan.success = success
        self.updated_at = time.time()

    def add_critic_report(self, report):
        """Record a Socratic critic report."""
        self.critic_reports.append(report)
        self.updated_at = time.time()

    def report(self) -> str:
        """Human-readable summary of current state."""
        lines = [
            f"── ScientificState ──",
            f"  Phase: {self.phase}",
            f"  Uncertainty: {self.uncertainty:.2f}",
            f"  Domain: {self.domain_type} (confidence: {self.domain_confidence:.2f})",
            f"  Steps: {self.steps_taken} | Stagnation: {self.stagnation_count}",
            f"  Evidence: {len(self.evidence)} observations",
            f"  Assumptions: {len(self.assumptions)}",
            f"  Plans executed: {len(self.plan_history)}",
            f"  Refuted hypotheses: {len(self.refuted_hypotheses)}",
            f"  Critic reports: {len(self.critic_reports)}",
        ]

        h = self.current_hypothesis
        if h:
            lines.append(f"  Current hypothesis: '{h.description[:60]}' (conf={h.confidence:.2f})")

        return "\n".join(lines)
