import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque

from .models import SkillManifest, SkillDAGManifest, SkillType


class SkillRegistry:
    """Loads, indexes, and validates SkillDAG manifests."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.raw_manifest: Optional[SkillDAGManifest] = None
        self.skills: Dict[str, SkillManifest] = {}
        self._load()

    def _load(self):
        with open(self.manifest_path) as f:
            data = yaml.safe_load(f)
        self.raw_manifest = SkillDAGManifest(**data)
        # Load markdown bodies
        for skill in self.raw_manifest.skills:
            skill_file = self.base_dir / skill.file
            if skill_file.exists():
                skill.body = skill_file.read_text()
            self.skills[skill.id] = skill
        self._validate()

    def _validate(self):
        """Validate DAG: no cycles, all deps exist."""
        skill_ids = set(self.skills.keys())
        # Check all dependencies exist
        for skill in self.skills.values():
            for dep in skill.depends_on:
                if dep not in skill_ids:
                    raise ValueError(f"Skill {skill.id} depends on missing skill: {dep}")
        # Check for cycles (Kahn's algorithm)
        self.topological_order()  # Will raise if cycle

    def topological_order(self) -> List[str]:
        """Return skills in dependency order."""
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        for skill in self.skills.values():
            for dep in skill.depends_on:
                graph[dep].append(skill.id)
                in_degree[skill.id] += 1
        # Add nodes with no deps
        for sid in self.skills:
            if sid not in in_degree:
                in_degree[sid] = 0

        queue = deque([sid for sid, deg in in_degree.items() if deg == 0])
        order = []
        while queue:
            sid = queue.popleft()
            order.append(sid)
            for neighbor in graph[sid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.skills):
            raise ValueError("Cycle detected in skill DAG")
        return order

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        return self.skills.get(skill_id)

    def get_skills_by_type(self, skill_type: SkillType) -> List[SkillManifest]:
        return [s for s in self.skills.values() if s.type == skill_type]

    def check_preconditions(self, skill_id: str, context: Dict[str, bool]) -> bool:
        """Check if all preconditions are met in context."""
        skill = self.skills.get(skill_id)
        if not skill:
            return False
        return all(context.get(pc, False) for pc in skill.preconditions)