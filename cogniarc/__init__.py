# CogniArc — Cognitive ARC-AGI-3 Agent Framework

from .scientist_agent import ScientistAgent
from .cognitive_player import CognitivePlayer
from .domain_classifier import DomainClassifier
from .physics_engine import PhysicsEngine
from .skill_tree import SkillTree
from .goal_inference import GoalInference
from .stagnation_detector import StagnationDetector
from .transform_inference import TransformInference
from .cli import main as cli_main
from .vision_sensor import cmd_analyze as vision_analyze, cmd_watch as vision_watch

__version__ = "0.2.0"
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
