from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class SkillType(str, Enum):
    NAVIGATION = "navigation"
    ROTATION = "rotation"
    INTERACTION = "interaction"
    PERCEPTION = "perception"
    META = "meta"


class SkillManifest(BaseModel):
    id: str
    type: SkillType
    file: str
    preconditions: List[str] = Field(default_factory=list)
    effects: List[str] = Field(default_factory=list)
    validation_levels: List[int] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    # Runtime fields (not in YAML)
    body: Optional[str] = None  # Markdown content loaded lazily
    compiled: Optional[Any] = None  # Compiled callable if skill is executable


class SkillDAGManifest(BaseModel):
    version: str
    game: str
    skills: List[SkillManifest]