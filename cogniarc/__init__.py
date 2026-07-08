# CogniARC — Cognitive ARC-AGI-3 Agent Framework

from .scientist_agent import ScientistAgent
from .cognitive_player import CognitiveDrives, WorkingMemory, CognitiveFatigue
from .domain_classifier import DomainClassifier
from .physics_engine import GamePhysics
from .skill_tree import SkillTree, Skill
from .goal_inference import GoalInferenceEngine, GoalHypothesis
from .stagnation_detector import StagnationDetector
from .transform_inference import TransformInference, Transform
from .triarchic_engine import TriarchicEngine, TriarchicState
from .representation_engine import RepresentationEngine, Representation
from .arc_agent import ArcAgentV3
from .cli import main as cli_main
from .vision_sensor import cmd_analyze as vision_analyze, cmd_watch as vision_watch

# SkillDAG modules
from .skill_dag.models import (
    SkillManifest,
    SkillDAGManifest,
    SkillContext,
    SkillSelectionResult,
    SkillType,
    # New: cognitive drive models
    DriveType,
    CognitiveDriveModel,
    DriveAnnotation,
    DriveAnnotatedSkill,
    DriveWeighter,
)
from .skill_dag.skill_registry import SkillRegistry
from .skill_dag.skill_navigator import SkillNavigator
from .skill_dag.core.perception_skill import PerceptionSkill, GridState
from .skill_dag.core.cognition_skill import CognitionSkill, CognitiveDrives as SkillCognitiveDrives
from .skill_dag.core.bfs_pathfinder import GridPathfinder, create_pathfinder
from .skill_dag.analysis.domain_classifier_skill import DomainClassifierSkill, DomainResult
from .skill_dag.analysis.transform_skill import TransformSkill, Transform as SkillTransform, TransformResult
from .skill_dag.analysis.physics_skill import PhysicsSkill, PhysicsModel
from .skill_dag.planning.bfs_planner_skill import BFSPlannerSkill, RealtimeBFSSkill, PlanResult
from .skill_dag.planning.transform_planner_skill import TransformPlannerSkill, MacroPlannerSkill, TransformPlan
from .skill_dag.execution.executor_skill import ExecutorSkill, ExecutionConfig, StepResult
from .skill_dag.execution.stagnation_detector_skill import StagnationDetectorSkill, StagnationConfig, StagnationReport
from .skill_dag.execution.macro_executor_skill import MacroExecutorSkill, Macro, MacroResult
from .skill_dag.integration.scientist_dag import ScientistDAG, ScientistConfig

__version__ = "0.2.0"
__all__ = [
    "ScientistAgent",
    "CognitiveDrives",
    "WorkingMemory",
    "CognitiveFatigue",
    "DomainClassifier",
    "GamePhysics",
    "SkillTree",
    "Skill",
    "GoalInferenceEngine",
    "GoalHypothesis",
    "StagnationDetector",
    "TransformInference",
    "Transform",
    "TriarchicEngine",
    "TriarchicState",
    "RepresentationEngine",
    "Representation",
    "ArcAgentV3",
    # SkillDAG
    "SkillManifest",
    "SkillDAGManifest",
    "SkillContext",
    "SkillSelectionResult",
    "SkillType",
    "SkillRegistry",
    "SkillNavigator",
    "PerceptionSkill",
    "GridState",
    "CognitionSkill",
    "SkillCognitiveDrives",
    "GridPathfinder",
    "create_pathfinder",
    "DomainClassifierSkill",
    "DomainResult",
    "TransformSkill",
    "SkillTransform",
    "TransformResult",
    "PhysicsSkill",
    "PhysicsModel",
    "BFSPlannerSkill",
    "RealtimeBFSSkill",
    "PlanResult",
    "TransformPlannerSkill",
    "MacroPlannerSkill",
    "TransformPlan",
    "ExecutorSkill",
    "ExecutionConfig",
    "StepResult",
    "StagnationDetectorSkill",
    "StagnationConfig",
    "StagnationReport",
    "MacroExecutorSkill",
    "Macro",
    "MacroResult",
    "ScientistDAG",
    "ScientistConfig",
]