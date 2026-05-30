#!/usr/bin/env python3
"""
RPG-style Skill Tree for ARC-AGI-3.

Tracks discovered mechanics across levels like an RPG ability tree.
Skills compose: MoveRight + PushBlock = push_right.

Key insight: Each level UNLOCKS new skills. Level N reuses skills from
Levels 1..N-1 and adds new ones. Like a mage getting new spells each level.

Mechanics that were "new" in Level 2 become "basic" in Level 3 —
the tree grows cumulatively.

Usage:
    from skill_tree import SkillTree
    st = SkillTree()
    st.unlock("MoveRight", level=1, evidence={"action": 1, "effect": "agent x+1"})
    st.detect_new_level(obs)  # scans for novel objects/mechanics
    combo = st.compose("MoveRight", "PushBlock")
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Skill:
    """A discovered game mechanic."""
    name: str
    level_discovered: int
    action_id: Optional[int] = None
    description: str = ""
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    composed_from: List[str] = field(default_factory=list)
    confidence: float = 1.0
    tests: int = 0
    successes: int = 0


class SkillTree:
    """Cumulative ability tree across levels."""

    def __init__(self, save_path: str = "/home/redgamer/arc_agi_agent/skill_tree.json"):
        self.skills: Dict[str, Skill] = {}
        self.compositions: Dict[tuple, Skill] = {}
        self.level_caps: Dict[int, List[str]] = {}
        self.current_level: int = 0
        self.save_path = Path(save_path)
        self._load()

    # ── Skill Management ──────────────────────────────────

    def unlock(self, name: str, level: int, **kwargs) -> Skill:
        """Record a newly discovered mechanic."""
        if name in self.skills:
            skill = self.skills[name]
            skill.confidence = min(skill.confidence + 0.1, 1.0)
            skill.tests += 1
            skill.successes += 1
            self._save()
            return skill

        skill = Skill(name=name, level_discovered=level, **kwargs)
        self.skills[name] = skill
        self.level_caps.setdefault(level, []).append(name)
        self._save()
        return skill

    def get(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def active_abilities(self, level: int = None) -> List[str]:
        """All skills available up to given level (cumulative)."""
        if level is None:
            level = self.current_level
        abilities = []
        for lvl in sorted(self.level_caps.keys()):
            if lvl <= level:
                abilities.extend(self.level_caps[lvl])
        return abilities

    # ── Level Transition ──────────────────────────────────

    def detect_new_level(self, obs) -> Dict[str, Any]:
        """
        Entering a new level — what's DIFFERENT?
        Like entering a new dungeon: new enemies, new items, new mechanics.
        """
        self.current_level += 1
        discoveries: Dict[str, Any] = {
            "level": self.current_level,
            "new_objects": [],
            "new_colors": [],
            "new_actions_available": [],
            "continues_from": self.active_abilities(self.current_level - 1),
        }

        # Detect new colors
        grid = obs.frame[0]
        colors = set(int(c) for c in grid.flatten())
        known_colors = set()
        discoveries["all_colors"] = sorted(colors)

        # Detect new available actions
        available = list(obs.available_actions or [])
        discoveries["actions_available"] = available

        # Count known vs unknown
        known = self.active_abilities(self.current_level - 1)
        discoveries["known_skills"] = len(known)
        discoveries["level"] = self.current_level

        return discoveries

    # ── Composition ───────────────────────────────────────

    def compose(self, skill_a: str, skill_b: str) -> Optional[Skill]:
        """
        Combine two known skills into a new capability.
        Like spellcrafting: Fireball + Telekinesis = Guided Fireball.
        """
        key = tuple(sorted([skill_a, skill_b]))
        if key in self.compositions:
            return self.compositions[key]

        s_a = self.skills.get(skill_a)
        s_b = self.skills.get(skill_b)
        if not s_a or not s_b:
            return None

        composed_name = f"{skill_a}+{skill_b}"
        composed = Skill(
            name=composed_name,
            level_discovered=max(s_a.level_discovered, s_b.level_discovered),
            description=f"Composed: {s_a.description} + {s_b.description}",
            composed_from=[skill_a, skill_b],
            preconditions=s_a.preconditions + s_b.preconditions,
            effects=s_a.effects + s_b.effects,
        )
        self.compositions[key] = composed
        self.skills[composed_name] = composed
        self._save()
        return composed

    # ── Persistence ───────────────────────────────────────

    def _save(self):
        data = {
            "skills": {k: {"name": v.name, "level": v.level_discovered,
                           "action_id": v.action_id, "description": v.description,
                           "composed_from": v.composed_from,
                           "confidence": v.confidence}
                       for k, v in self.skills.items()},
            "level_caps": {str(k): v for k, v in self.level_caps.items()},
            "current_level": self.current_level,
        }
        self.save_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self.save_path.exists():
            return
        data = json.loads(self.save_path.read_text())
        self.current_level = data.get("current_level", 0)
        for name, d in data.get("skills", {}).items():
            self.skills[name] = Skill(
                name=d["name"],
                level_discovered=d["level"],
                action_id=d.get("action_id"),
                description=d.get("description", ""),
                composed_from=d.get("composed_from", []),
                confidence=d.get("confidence", 1.0),
            )
        for lvl, names in data.get("level_caps", {}).items():
            self.level_caps[int(lvl)] = names

    def report(self) -> str:
        lines = [f"Skill Tree — Level {self.current_level}"]
        for lvl in sorted(self.level_caps.keys()):
            names = self.level_caps[lvl]
            lines.append(f"  Lv.{lvl}: {', '.join(names)}")
        comps = [s.name for s in self.compositions.values()]
        if comps:
            lines.append(f"  Compositions: {', '.join(comps)}")
        return "\n".join(lines)
