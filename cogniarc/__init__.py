# CogniArc — Cognitive ARC-AGI-3 Agent Framework

from .scientist_agent import ScientistAgent
from .cognitive_player import CognitiveDrives as CognitivePlayer
from .domain_classifier import DomainClassifier
from .domain_profiler import DomainProfiler
from .physics_engine import GamePhysics as PhysicsEngine
from .skill_tree import SkillTree
from .goal_inference import GoalInferenceEngine as GoalInference
from .stagnation_detector import StagnationDetector
from .transform_inference import TransformInference
from .temporal_inference import TemporalReasoner, Delta, DeltaPattern, PatternType
from .benchmark_tracker import BenchmarkTracker, GameResult, SessionResult
from .cli import main as cli_main
from .vision_sensor import cmd_analyze as vision_analyze, cmd_watch as vision_watch

__version__ = "0.2.0"
__all__ = [
    "ScientistAgent",
    "CognitivePlayer",
    "DomainClassifier",
    "DomainProfiler",
    "PhysicsEngine",
    "SkillTree",
    "GoalInference",
    "StagnationDetector",
    "TransformInference",
    "TemporalReasoner", "Delta", "DeltaPattern", "PatternType",
    "BenchmarkTracker", "GameResult", "SessionResult",
    "cli_main", "vision_analyze", "vision_watch",
]
