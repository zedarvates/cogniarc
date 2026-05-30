# CogniArc — Cognitive ARC-AGI-3 Agent Framework

from .scientist_agent import ScientistAgent
from .cognitive_player import CognitivePlayer
from .domain_classifier import DomainClassifier
from .physics_engine import PhysicsEngine
from .skill_tree import SkillTree
from .goal_inference import GoalInference
from .stagnation_detector import StagnationDetector
from .transform_inference import TransformInference

__version__ = "0.1.0"
__all__ = [
    "ScientistAgent",
    "CognitivePlayer",
    "DomainClassifier",
    "PhysicsEngine",
    "SkillTree",
    "GoalInference",
    "StagnationDetector",
    "TransformInference",
]
