from typing import Dict, List, Set, Optional
from .skill_registry import SkillRegistry
from .models import SkillManifest


class SkillNavigator:
    """Selects relevant skill subtree given current context."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self._selected: Set[str] = set()

    def select_skills(self, context: Dict[str, bool]) -> List[str]:
        """Return skill IDs that are relevant and have met preconditions."""
        self._selected.clear()

        # 1. Find all skills whose preconditions are met
        candidates = []
        for skill in self.registry.skills.values():
            if self.registry.check_preconditions(skill.id, context):
                candidates.append(skill.id)

        # 2. Include all dependencies (transitive closure)
        for skill_id in candidates:
            self._include_with_deps(skill_id)

        # 3. Return in topological order
        order = self.registry.topological_order()
        return [sid for sid in order if sid in self._selected]

    def _include_with_deps(self, skill_id: str):
        """Add skill and all its dependencies."""
        if skill_id in self._selected:
            return
        self._selected.add(skill_id)
        skill = self.registry.skills.get(skill_id)
        if skill:
            for dep in skill.depends_on:
                self._include_with_deps(dep)

    def get_skill_body(self, skill_id: str) -> Optional[str]:
        """Lazy-load skill markdown body."""
        skill = self.registry.get_skill(skill_id)
        return skill.body if skill else None

    def build_context_prompt(self, context: Dict[str, bool]) -> str:
        """Build compact prompt with selected skills for LLM."""
        selected = self.select_skills(context)
        lines = ["=== AVAILABLE SKILLS ==="]
        for sid in selected:
            skill = self.registry.get_skill(sid)
            if skill and skill.body:
                # Extract first paragraph as summary
                summary = skill.body.split('\n\n')[0][:200]
                lines.append(f"- {sid} ({skill.type.value}): {summary}")
        return "\n".join(lines)