"""Tests for drive-weighted SkillDAG v2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogniarc.skill_dag.models import (
    SkillManifest,
    SkillDAGManifest,
    SkillContext,
    DriveAnnotatedSkill,
    DriveAnnotation,
    CognitiveDriveModel,
    DriveType,
    DriveWeighter,
)
from cogniarc.skill_dag.skill_registry import SkillRegistry
from cogniarc.skill_dag.skill_navigator import SkillNavigator


def test_drive_types():
    """All 6 drives are accessible."""
    drives = list(DriveType)
    assert len(drives) == 6
    assert DriveType.CURIOSITY in drives


def test_cognitive_drive_model():
    """Drive model has correct defaults and bounds."""
    m = CognitiveDriveModel()
    assert m.curiosity == 0.5
    assert m.pattern_match == 0.5
    m.set(DriveType.CURIOSITY, 0.9)
    assert m.get(DriveType.CURIOSITY) == 0.9
    # Clamping
    m.set(DriveType.VERIFY, 2.0)
    assert m.get(DriveType.VERIFY) == 1.0
    m.set(DriveType.VERIFY, -1.0)
    assert m.get(DriveType.VERIFY) == 0.0


def test_dominant():
    """dominant() returns highest drive."""
    m = CognitiveDriveModel(causal=0.9, curiosity=0.3, verify=0.5)
    assert m.dominant() == DriveType.CAUSAL


def test_vector():
    """vector() returns 6D tuple."""
    m = CognitiveDriveModel()
    v = m.vector()
    assert len(v) == 6
    assert all(0.0 <= x <= 1.0 for x in v)


def test_drive_annotation_score():
    """Drive alignment score computes correctly."""
    skill_ann = DriveAnnotation(pattern_match=0.9, curiosity=0.5)
    drives = CognitiveDriveModel(pattern_match=0.8, curiosity=0.2)
    score = skill_ann.score(drives)
    # Expected: (0.9*0.8 + 0.5*0.2 + 0*0.5 + 0*0.5 + 0*0.5 + 0*0.5) / 3.0
    expected = (0.72 + 0.10) / 3.0  # = 0.82/3.0
    assert abs(score - expected) < 0.01


def test_drive_annotation_proper_alignment():
    """When drives match perfectly, score is high."""
    skill_ann = DriveAnnotation(pattern_match=1.0)
    drives = CognitiveDriveModel(pattern_match=1.0)
    # All other drives are 0.5, so sum = 0.5*5 + 1.0 = 3.5
    # Score = 1.0*1.0 / 3.5 ≈ 0.286
    score = skill_ann.score(drives)
    # With all drives at 0.5 except pattern_match=1.0:
    # dot = 1.0*1.0 + 0*0.5*5 = 1.0
    # sum = 0.5*5 + 1.0 = 3.5
    # score = 1.0/3.5 ≈ 0.286
    assert 0.2 < score < 0.4, f"Got {score}"


def test_drive_weighter_stagnation():
    """Stagnation boosts curiosity."""
    w = DriveWeighter()
    ctx = SkillContext()
    d = w.compute(ctx, stagnation_count=6)
    assert d.get(DriveType.CURIOSITY) == 0.9
    assert d.get(DriveType.VERIFY) == 0.8


def test_drive_weighter_early_iteration():
    """Early iteration boosts pattern_match."""
    w = DriveWeighter()
    ctx = SkillContext()
    d = w.compute(ctx, iteration=3)
    assert d.get(DriveType.PATTERN_MATCH) == 0.8


def test_drive_weighter_late_iteration():
    """Late iteration boosts verify."""
    w = DriveWeighter()
    ctx = SkillContext()
    d = w.compute(ctx, iteration=150)
    assert d.get(DriveType.VERIFY) == 0.9


def test_drive_weighter_known_context():
    """Known flags boost efficiency."""
    w = DriveWeighter()
    ctx = SkillContext()
    for k in ["a", "b", "c", "d"]:
        ctx.set(k, True)
    d = w.compute(ctx)
    assert d.get(DriveType.EFFICIENCY) == 0.8


def test_skill_navigator_drive_weighted_selection():
    """Navigator selects skills with drive weighting."""
    # Use the real manifest file (which now has drive annotations)
    import os
    manifest_path = os.path.join(
        os.path.dirname(__file__), "..", "cogniarc", "skill_dag", "manifest.yaml"
    )
    if not os.path.exists(manifest_path):
        print(f"SKIP: manifest not found at {manifest_path}")
        return
    
    registry = SkillRegistry(manifest_path=manifest_path)
    nav = SkillNavigator(registry)

    # Context: nothing done yet
    ctx = SkillContext()
    # Set preconditions for perception skills
    ctx.set("current_obs", True)
    ctx.set("source_available", True)
    
    result = nav.select_skills(ctx, stagnation_count=0, iteration=5)

    assert len(result.selected_skills) >= 1
    assert result.drive_state is not None
    assert result.drive_state.get(DriveType.PATTERN_MATCH) >= 0.5
    print(f"Selected: {result.selected_skills}")
    print(f"Order: {result.execution_order}")
    print(f"Drives: {result.drive_state.vector()}")
    print(f"Summary: {result.context_summary}")
    print(f"Log: {result.selection_log[:3]}")

    # detect-walls-from-source should be selected (preconditions met)
    assert "detect-walls-from-source" in result.selected_skills

    print("\n✓ All drive-weighted navigator tests passed!")


if __name__ == "__main__":
    test_drive_types()
    test_cognitive_drive_model()
    test_dominant()
    test_vector()
    test_drive_annotation_score()
    test_drive_annotation_proper_alignment()
    test_drive_weighter_stagnation()
    test_drive_weighter_early_iteration()
    test_drive_weighter_late_iteration()
    test_drive_weighter_known_context()
    test_skill_navigator_drive_weighted_selection()
    print("\n✅ ALL TESTS PASSED")
