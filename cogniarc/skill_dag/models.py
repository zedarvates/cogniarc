"""SkillDAG Models — Pydantic models for typed skill graphs with cognitive drive integration."""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
from pydantic import BaseModel, Field


# ─── Cognitive Drive Types ────────────────────────────────────────────

class DriveType(str, Enum):
    """The 6 human-like cognitive drives from CogniARC."""
    CURIOSITY = "curiosity"
    PATTERN_MATCH = "pattern_match"
    CAUSAL = "causal"
    EFFICIENCY = "efficiency"
    MEMORY = "memory"
    VERIFY = "verify"

    @classmethod
    def all_drives(cls) -> list:
        return [d for d in cls]


class CognitiveDriveModel(BaseModel):
    """Current activation state of cognitive drives (0.0–1.0)."""
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0, description="Explore the unknown")
    pattern_match: float = Field(default=0.5, ge=0.0, le=1.0, description="Recognize visual/structural patterns")
    causal: float = Field(default=0.5, ge=0.0, le=1.0, description="Understand cause-effect")
    efficiency: float = Field(default=0.5, ge=0.0, le=1.0, description="Seek simplest solution")
    memory: float = Field(default=0.5, ge=0.0, le=1.0, description="Retain/recall past patterns")
    verify: float = Field(default=0.5, ge=0.0, le=1.0, description="Validate and self-correct")

    def get(self, drive: DriveType) -> float:
        return getattr(self, drive.value)

    def set(self, drive: DriveType, value: float):
        setattr(self, drive, max(0.0, min(1.0, value)))

    def dominant(self) -> DriveType:
        """Return the highest-activated drive."""
        return max(DriveType, key=lambda d: self.get(d))

    def vector(self) -> Tuple[float, ...]:
        """6D drive vector for distance computations."""
        return tuple(self.get(d) for d in DriveType)


# ─── SkillDAG Core Models ─────────────────────────────────────────────


class SkillType(str):
    """Standard skill types for ARC-AGI-3."""
    NAVIGATION = "navigation"
    ROTATION = "rotation"
    INTERACTION = "interaction"
    PERCEPTION = "perception"
    META = "meta"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    COGNITION = "cognition"
    MEMORY = "memory"


class SkillManifest(BaseModel):
    """A single skill in the SkillDAG."""
    id: str = Field(..., description="Unique skill identifier (kebab-case)")
    type: str = Field(..., description="Skill type (navigation, rotation, etc.)")
    file: str = Field(..., description="Path to skill markdown body (relative to skill_dag/)")
    preconditions: List[str] = Field(default_factory=list, description="Context keys that must be true")
    effects: List[str] = Field(default_factory=list, description="Context keys set to true after execution")
    depends_on: List[str] = Field(default_factory=list, description="Skill IDs this skill depends on")
    validation_levels: List[int] = Field(default_factory=list, description="Validation gate levels required")
    description: str = Field(default="", description="Human-readable description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Skill-specific parameters")
    version: str = Field(default="1.0.0")
    enabled: bool = Field(default=True)

    def get_preconditions_set(self) -> set:
        return set(self.preconditions)

    def get_effects_set(self) -> set:
        return set(self.effects)


class SkillDAGManifest(BaseModel):
    """Root manifest for a SkillDAG."""
    version: str = Field(default="1.0")
    game: str = Field(default="universal", description="Game ID or 'universal'")
    skills: List[SkillManifest] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        for s in self.skills:
            if s.id == skill_id:
                return s
        return None

    def get_skills_by_type(self, skill_type: str) -> List[SkillManifest]:
        return [s for s in self.skills if s.type == skill_type and s.enabled]

    def get_all_preconditions(self) -> set:
        all_pre = set()
        for s in self.skills:
            if s.enabled:
                all_pre.update(s.preconditions)
        return all_pre


class SkillContext(BaseModel):
    """Runtime context for skill selection."""
    game_state: Dict[str, Any] = Field(default_factory=dict)
    ctx_flags: Dict[str, bool] = Field(default_factory=dict)
    consumed_keys: set = Field(default_factory=set)

    def has_precondition(self, key: str) -> bool:
        return self.ctx_flags.get(key, False)

    def set(self, key: str, value: bool = True):
        self.ctx_flags[key] = value

    def consume(self, key: str):
        self.consumed_keys.add(key)


class SkillSelectionResult(BaseModel):
    """Result of skill selection."""
    selected_skills: List[str] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
    context_summary: str = ""
    missing_preconditions: Dict[str, List[str]] = Field(default_factory=dict)
    drive_state: Optional[CognitiveDriveModel] = Field(default=None, description="Drive state at selection time")
    selection_log: List[str] = Field(default_factory=list, description="Why each skill was selected/rejected")


# ─── Drive-Annotated Skills ───────────────────────────────────────────

class DriveAnnotation(BaseModel):
    """Per-drive relevance for a skill (0.0–1.0, sum not constrained)."""
    curiosity: float = Field(default=0.0, ge=0.0, le=1.0)
    pattern_match: float = Field(default=0.0, ge=0.0, le=1.0)
    causal: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    memory: float = Field(default=0.0, ge=0.0, le=1.0)
    verify: float = Field(default=0.0, ge=0.0, le=1.0)

    def get(self, drive: DriveType) -> float:
        return getattr(self, drive.value)

    def score(self, drives: CognitiveDriveModel) -> float:
        """Cosine-like drive alignment score between skill annotation and current drives."""
        a = self.vector()
        d = drives.vector()
        # Weighted dot product: high-value drives × high-relevance skills win
        dot = sum(a[i] * d[i] for i in range(6))
        # Normalize by sum of drive values (not L2 — we care about magnitude)
        norm = sum(d) if sum(d) > 0 else 1.0
        return dot / norm

    def vector(self) -> Tuple[float, ...]:
        return (self.curiosity, self.pattern_match, self.causal,
                self.efficiency, self.memory, self.verify)


class DriveAnnotatedSkill(SkillManifest):
    """A skill with cognitive drive relevance annotations."""
    drives: DriveAnnotation = Field(default_factory=DriveAnnotation)
    min_drive_threshold: float = Field(default=0.0, ge=0.0, le=1.0,
                                        description="Minimum drive alignment to consider this skill")


# ─── Drive Weighter ───────────────────────────────────────────────────

class DriveWeighter:
    """Compute drive activations from context and adjust skill selection."""

    def __init__(self):
        self.history: List[Tuple[CognitiveDriveModel, str]] = []  # (drives, reason)

    def compute(self,
                context: SkillContext,
                stagnation_count: int = 0,
                iteration: int = 0) -> CognitiveDriveModel:
        """Compute drive activations based on context state."""
        drives = CognitiveDriveModel()

        # Stagnation boosts curiosity + verify
        if stagnation_count > 5:
            drives.set(DriveType.CURIOSITY, 0.9)
            drives.set(DriveType.VERIFY, 0.8)
        elif stagnation_count > 2:
            drives.set(DriveType.CURIOSITY, 0.7)

        # Known patterns boost efficiency
        known_count = sum(1 for k in context.ctx_flags if context.ctx_flags[k])
        if known_count > 3:
            drives.set(DriveType.EFFICIENCY, 0.8)
            drives.set(DriveType.MEMORY, 0.7)

        # Early iterations boost pattern_match
        if iteration < 10:
            drives.set(DriveType.PATTERN_MATCH, 0.8)

        # Late iterations boost verify + causal
        if iteration > 100:
            drives.set(DriveType.VERIFY, 0.9)
            drives.set(DriveType.CAUSAL, 0.7)

        self.history.append((drives, f"stagn={stagnation_count} iter={iteration} known={known_count}"))
        return drives