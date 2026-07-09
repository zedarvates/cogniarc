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
    game_id: str = ""  # Track which game this skill came from
    action_id: Optional[int] = None
    description: str = ""
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    composed_from: List[str] = field(default_factory=list)
    confidence: float = 1.0
    tests: int = 0
    successes: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.tests, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level_discovered,
            "game_id": self.game_id,
            "action_id": self.action_id,
            "description": self.description,
            "preconditions": self.preconditions,
            "effects": self.effects,
            "composed_from": self.composed_from,
            "confidence": self.confidence,
            "tests": self.tests,
            "successes": self.successes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            name=d["name"],
            level_discovered=d["level"],
            game_id=d.get("game_id", ""),
            action_id=d.get("action_id"),
            description=d.get("description", ""),
            preconditions=d.get("preconditions", []),
            effects=d.get("effects", []),
            composed_from=d.get("composed_from", []),
            confidence=d.get("confidence", 1.0),
            tests=d.get("tests", 0),
            successes=d.get("successes", 0),
        )


import os
from pathlib import Path

def _default_skill_tree_path() -> Path:
    """Get default skill tree path in user cache directory."""
    cache_dir = Path.home() / ".cache" / "cogniarc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "skill_tree.json"


class SkillTree:
    """Cumulative ability tree across levels."""

    def __init__(self, save_path: str | None = None):
        self.skills: Dict[str, Skill] = {}
        self.compositions: Dict[tuple, Skill] = {}
        self.level_caps: Dict[int, List[str]] = {}
        self.current_level: int = 0
        if save_path is None:
            save_path = os.environ.get("COGNIARC_SKILL_TREE_PATH", str(_default_skill_tree_path()))
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
        if hasattr(obs, 'frame') and obs.frame and len(obs.frame) > 0:
            grid = obs.frame[0]
            colors = set(int(c) for c in grid.flatten())
        else:
            colors = set()
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
            "skills": {k: v.to_dict() for k, v in self.skills.items()},
            "level_caps": {str(k): v for k, v in self.level_caps.items()},
            "current_level": self.current_level,
        }
        self.save_path.write_text(json.dumps(data, indent=2))

    def _load(self):
        """Load the tree from disk, tolerating legacy/corrupt cache files.

        Called from SkillTree.load_for_game() during ScientistAgent __init__
        with NO try/except above it — before this hardening, a single cache
        file in an old format crashed the whole agent at startup with
        `'list' object has no attribute 'get'` (2026-07-05 holdout report:
        sp80 could not start at all because of this). A cache file must never
        be able to prevent an agent run: on any parse problem we warn and
        start with an empty tree instead.
        """
        if not self.save_path.exists():
            return
        try:
            data = json.loads(self.save_path.read_text())

            # Legacy format: top-level list of skill dicts (pre-dict schema).
            if isinstance(data, list):
                data = {"skills": {d.get("name", f"skill_{i}"): d
                                   for i, d in enumerate(data) if isinstance(d, dict)}}

            if not isinstance(data, dict):
                raise ValueError(f"unsupported skill-tree JSON root: {type(data).__name__}")

            self.current_level = data.get("current_level", 0)

            skills = data.get("skills", {})
            # Legacy variant: "skills" stored as a list instead of a dict.
            if isinstance(skills, list):
                skills = {d.get("name", f"skill_{i}"): d
                          for i, d in enumerate(skills) if isinstance(d, dict)}
            for name, d in skills.items():
                self.skills[name] = Skill.from_dict(d)

            for lvl, names in data.get("level_caps", {}).items():
                self.level_caps[int(lvl)] = names
        except Exception as e:
            print(f"[SkillTree] Corrupt/legacy cache ignored ({self.save_path.name}): {e}")
            self.skills = {}
            self.level_caps = {}
            self.current_level = 0

    # ── Cross-Game Transfer ─────────────────────────────────

    @staticmethod
    def get_game_path(game_id: str) -> Path:
        """Get the skill tree path for a specific game."""
        cache_dir = Path.home() / ".cache" / "cogniarc" / "games"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{game_id}_skill_tree.json"

    def export_for_game(self, game_id: str) -> "SkillTree":
        """Create a new SkillTree containing only skills from this game."""
        game_tree = SkillTree(save_path=str(self.get_game_path(game_id)))
        for name, skill in self.skills.items():
            if skill.game_id == game_id or skill.game_id == "":
                game_tree.skills[name] = skill
        for lvl, names in self.level_caps.items():
            game_tree.level_caps[lvl] = [n for n in names if self.skills[n].game_id == game_id or self.skills[n].game_id == ""]
        game_tree.current_level = self.current_level
        game_tree._save()
        return game_tree

    @classmethod
    def load_for_game(cls, game_id: str) -> "SkillTree":
        """Load a game-specific skill tree."""
        return cls(save_path=str(cls.get_game_path(game_id)))

    def import_from_game(self, other: "SkillTree", source_game: str, min_confidence: float = 0.7):
        """Import skills from another game's tree (cross-game transfer).
        
        Only imports skills above confidence threshold.
        Marks imported skills with source_game for traceability.
        """
        imported = 0
        for name, skill in other.skills.items():
            if skill.confidence >= min_confidence:
                # Rename to avoid conflicts
                new_name = f"{source_game}:{name}" if not name.startswith(f"{source_game}:") else name
                if new_name not in self.skills:
                    imported_skill = Skill(
                        name=new_name,
                        level_discovered=skill.level_discovered,
                        game_id=source_game,
                        action_id=skill.action_id,
                        description=f"[imported from {source_game}] {skill.description}",
                        preconditions=skill.preconditions.copy(),
                        effects=skill.effects.copy(),
                        composed_from=skill.composed_from.copy(),
                        confidence=skill.confidence * 0.9,  # Slight penalty for transfer
                        tests=skill.tests,
                        successes=skill.successes,
                    )
                    self.skills[new_name] = imported_skill
                    imported += 1
        if imported > 0:
            self._save()
        return imported

    def report(self) -> str:
        lines = [f"Skill Tree — Level {self.current_level}"]
        for lvl in sorted(self.level_caps.keys()):
            names = self.level_caps[lvl]
            lines.append(f"  Lv.{lvl}: {', '.join(names)}")
        comps = [s.name for s in self.compositions.values()]
        if comps:
            lines.append(f"  Compositions: {', '.join(comps)}")
        return "\n".join(lines)
