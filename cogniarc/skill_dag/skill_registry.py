"""Skill Registry — Loads, validates, and manages SkillDAG skills."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict

from .models import SkillManifest, SkillDAGManifest, SkillContext


class SkillRegistry:
    """Loads and validates SkillDAG manifest + skill bodies."""

    def __init__(self, manifest_path: str = None, manifest: SkillDAGManifest = None):
        if manifest is not None:
            self.manifest_path = None
            self.manifest = manifest
        elif manifest_path is not None:
            self.manifest_path = Path(manifest_path)
            self.manifest: SkillDAGManifest = self._load_manifest()
        else:
            raise ValueError("Either manifest_path or manifest must be provided")
        self.skill_bodies: Dict[str, str] = {}
        self._validate()

    def _load_manifest(self) -> SkillDAGManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path) as f:
            data = yaml.safe_load(f)

        return SkillDAGManifest(**data)

    def _validate(self):
        """Validate DAG: all deps exist, no cycles (Kahn's algorithm)."""
        skill_ids = {s.id for s in self.manifest.skills if s.enabled}

        # Check all dependencies exist
        for skill in self.manifest.skills:
            if not skill.enabled:
                continue
            for dep in skill.depends_on:
                if dep not in skill_ids:
                    raise ValueError(f"Skill '{skill.id}' depends on missing skill '{dep}'")

        # Kahn's algorithm for cycle detection
        in_degree = {sid: 0 for sid in skill_ids}
        adj = defaultdict(list)

        for skill in self.manifest.skills:
            if not skill.enabled:
                continue
            for dep in skill.depends_on:
                adj[dep].append(skill.id)
                in_degree[skill.id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        processed = []

        while queue:
            sid = queue.pop()
            processed.append(sid)
            for neighbor in adj[sid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(processed) != len(skill_ids):
            # Find cycle
            remaining = skill_ids - set(processed)
            raise ValueError(f"Cycle detected in skills: {remaining}")

    def get_skill(self, skill_id: str) -> Optional[SkillManifest]:
        return self.manifest.get_skill(skill_id)

    def get_skill_body(self, skill_id: str) -> str:
        """Lazy load skill markdown body."""
        if skill_id in self.skill_bodies:
            return self.skill_bodies[skill_id]

        skill = self.get_skill(skill_id)
        if not skill:
            return ""

        body_path = self.manifest_path.parent / skill.file
        if body_path.exists():
            with open(body_path) as f:
                body = f.read()
            self.skill_bodies[skill_id] = body
            return body
        return ""

    def topological_order(self) -> List[str]:
        """Return skills in dependency order (Kahn's)."""
        skill_ids = [s.id for s in self.manifest.skills if s.enabled]
        in_degree = {sid: 0 for sid in skill_ids}
        adj = defaultdict(list)

        for skill in self.manifest.skills:
            if not skill.enabled:
                continue
            for dep in skill.depends_on:
                adj[dep].append(skill.id)
                in_degree[skill.id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            sid = queue.pop()
            result.append(sid)
            for neighbor in adj[sid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def check_preconditions(self, skill_id: str, context: SkillContext) -> bool:
        """Check if all preconditions are met."""
        skill = self.get_skill(skill_id)
        if not skill:
            return False
        return all(context.has_precondition(k) for k in skill.preconditions)

    def get_transitive_dependencies(self, skill_ids: List[str]) -> Set[str]:
        """Get all transitive dependencies of selected skills."""
        result = set(skill_ids)
        changed = True
        while changed:
            changed = False
            for skill in self.manifest.skills:
                if skill.id in result and skill.enabled:
                    for dep in skill.depends_on:
                        if dep not in result:
                            result.add(dep)
                            changed = True
        return result

    def list_skills(self) -> List[SkillManifest]:
        return [s for s in self.manifest.skills if s.enabled]