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
from .spatial_inference import SpatialReasoner, Region, Relation, RelationType, SpatialPattern, SpatialPatternType
from .benchmark_tracker import BenchmarkTracker, GameResult, SessionResult
from .cli import main as cli_main
from .vision_sensor import cmd_analyze as vision_analyze, cmd_watch as vision_watch
from .scientific_state import ScientificState, Hypothesis, Observation, EvidenceReliability, ActionPlan
from .socratic_critic import SocraticCritic, SocraticReport, SocraticIssue, SocraticIssueType, SophismType
from .goal_sanity import GoalSanityChecker, SanityVerdict

__version__ = "0.3.0"
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
    "SpatialReasoner", "Region", "Relation", "RelationType", "SpatialPattern", "SpatialPatternType",
    "BenchmarkTracker", "GameResult", "SessionResult",
    "cli_main", "vision_analyze", "vision_watch",
    "ScientificState", "Hypothesis", "Observation", "EvidenceReliability", "ActionPlan",
    "SocraticCritic", "SocraticReport", "SocraticIssue", "SocraticIssueType", "SophismType",
    "GoalSanityChecker", "SanityVerdict",
]

# Physics World Model
from .world_model.physics.simulator.physics_v3 import PhysicsWorldV3, V3_SCENARIOS
from .world_model.physics.tools.spatial_zoning import SpatialAnalyzer
from .world_model.physics.tools.mass_gravity import MassProperties
from .world_model.physics.tools.kinematic_engine import MobilityAnalyzer
