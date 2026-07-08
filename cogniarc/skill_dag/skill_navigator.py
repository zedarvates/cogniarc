"""Skill Navigator v2 — Drive-weighted skill selection.

Extends the original SkillNavigator with cognitive drive activation
that dynamically weights skill relevance based on context state.
"""

from __future__ import annotations

from typing import List, Dict, Set, Optional
from collections import Counter

from .models import (
    SkillContext,
    SkillSelectionResult,
    SkillManifest,
    CognitiveDriveModel,
    DriveType,
    DriveWeighter,
    DriveAnnotation,
)


class SkillNavigator:
    """Selects skills based on context preconditions AND cognitive drive alignment."""

    def __init__(self, registry):
        self.registry = registry
        self.drive_weighter = DriveWeighter()
        self.current_drives = CognitiveDriveModel()

        # Drive profiles for auto-annotation when YAML lacks drives section
        self._default_drive_profiles: Dict[str, DriveAnnotation] = {
            "perception": DriveAnnotation(
                curiosity=0.7, pattern_match=0.9, verify=0.5
            ),
            "navigation": DriveAnnotation(
                efficiency=0.8, memory=0.6, pattern_match=0.5
            ),
            "rotation": DriveAnnotation(
                causal=0.6, efficiency=0.7, pattern_match=0.4
            ),
            "interaction": DriveAnnotation(
                causal=0.8, verify=0.7, curiosity=0.3
            ),
            "meta": DriveAnnotation(
                verify=0.9, memory=0.7, curiosity=0.6
            ),
            "analysis": DriveAnnotation(
                pattern_match=0.9, causal=0.7, curiosity=0.5
            ),
            "planning": DriveAnnotation(
                efficiency=0.9, memory=0.6, causal=0.5
            ),
            "execution": DriveAnnotation(
                efficiency=0.9, verify=0.5, pattern_match=0.3
            ),
            "cognition": DriveAnnotation(
                memory=0.9, pattern_match=0.8, causal=0.7, verify=0.5
            ),
            "memory": DriveAnnotation(
                memory=1.0, pattern_match=0.7, efficiency=0.4
            ),
        }

    # ─── Main Selection ────────────────────────────────────────────────

    def select_skills(
        self,
        context: SkillContext,
        stagnation_count: int = 0,
        iteration: int = 0,
    ) -> SkillSelectionResult:
        """
        Select skills whose preconditions are met, weighted by cognitive drives.

        1. Filter by preconditions (unchanged)
        2. Score by drive alignment (NEW)
        3. Add transitive dependencies
        4. Topological sort
        """
        # Update drives from context
        self.current_drives = self.drive_weighter.compute(
            context, stagnation_count, iteration
        )

        # Phase 1: Filter skills where ALL preconditions met
        candidates: List[SkillManifest] = []
        missing_pre: Dict[str, List[str]] = {}
        selection_log: List[str] = []

        for skill in self.registry.list_skills():
            met = [p for p in skill.preconditions if context.has_precondition(p)]
            unmet = [p for p in skill.preconditions if not context.has_precondition(p)]

            if unmet:
                missing_pre[skill.id] = unmet
                selection_log.append(f"SKIP {skill.id}: missing {unmet}")
            else:
                candidates.append(skill)

        # Phase 2: Score by drive alignment
        scored: List[tuple[float, str]] = []
        for skill in candidates:
            annotation = self._get_annotation(skill)
            drive_score = annotation.score(self.current_drives)
            threshold = getattr(skill, 'min_drive_threshold', 0.0)
            if drive_score >= threshold:
                scored.append((drive_score, skill.id))
                selection_log.append(
                    f"SCORE {skill.id}: {drive_score:.3f} "
                    f"(drives={self.current_drives.dominant().value}:{self.current_drives.get(self.current_drives.dominant()):.2f})"
                )
            else:
                selection_log.append(
                    f"BELOW_THRESHOLD {skill.id}: {drive_score:.3f} < {threshold}"
                )

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Phase 3: Add transitive dependencies (ordered before their dependents)
        selected_ids = [sid for _, sid in scored]
        all_selected: Set[str] = self.registry.get_transitive_dependencies(selected_ids)

        # Phase 4: Topological sort within the filtered set
        topo_order = [
            sid for sid in self.registry.topological_order()
            if sid in all_selected
        ]

        # Build summary
        met_count = len(scored)
        dominant = self.current_drives.dominant()
        summary = (
            f"Drives: {dominant.value}={self.current_drives.get(dominant):.2f} | "
            f"Candidates: {len(candidates)} → "
            f"Scored: {met_count} → "
            f"Ordered: {len(topo_order)} | "
            f"Top: {scored[0][1] if scored else 'none'}"
        )

        return SkillSelectionResult(
            selected_skills=list(all_selected),
            execution_order=topo_order,
            context_summary=summary,
            missing_preconditions=missing_pre,
            drive_state=self.current_drives,
            selection_log=selection_log,
        )

    # ─── Helpers ───────────────────────────────────────────────────────

    def _get_annotation(self, skill: SkillManifest) -> DriveAnnotation:
        """Get drive annotation for a skill, falling back to type-based default."""
        # Check if skill has drive annotations (DriveAnnotatedSkill)
        drives_attr = getattr(skill, 'drives', None)  # type: ignore[union-attr]
        if isinstance(drives_attr, DriveAnnotation):
            return drives_attr

        # Fallback: type-based default profile
        skill_type = getattr(skill, 'type', 'unknown')
        return self._default_drive_profiles.get(
            skill_type,
            DriveAnnotation()  # all zeros for unknown
        )

    def get_current_drives(self) -> CognitiveDriveModel:
        return self.current_drives

    def explain_selection(self, result: SkillSelectionResult) -> str:
        """Return a human-readable explanation of the selection."""
        return "\n".join([
            f"[{self.current_drives.dominant().value}:{self.current_drives.get(self.current_drives.dominant()):.2f}] "
            + f"→ {result.context_summary}",
            *result.selection_log,
        ])
