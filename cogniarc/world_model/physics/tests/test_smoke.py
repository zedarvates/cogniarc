"""Smoke tests — verify all modules import and basic functions work."""
import sys
sys.path.insert(0, '/home/redgamer/projects/world-model-tool')


def test_imports():
    """All core modules import cleanly"""
    import constants
    from simulator import physics, physics_v3
    from models import trainer
    from tools import discrete_classifier, scene_graph, llm_tool, predictor
    from tools import relation_engine, kinematic_engine, mass_gravity
    from tools import momentum_inertia, torque_experts, advanced_physics
    from tools import spatial_zoning, spatial_reasoning
    return True


def test_physics_v3():
    """Physics V3 scenarios create and run"""
    from simulator.physics_v3 import V3_SCENARIOS
    for name in ["ramp", "vehicle", "jenga"]:
        world = V3_SCENARIOS[name]()
        for _ in range(10):
            world.step(1/60)
        assert len(world.bodies) > 0, f"{name}: no bodies"
    return True


def test_discrete_classifier():
    """8-state classifier runs"""
    from tools.discrete_classifier import train_discrete_model, predict_object_fate
    result = train_discrete_model("ramp", steps=100)
    assert result["trained"], "Training failed"
    pred = predict_object_fate("ramp")
    assert "predictions" in pred, "Prediction failed"
    return True


def test_scene_graph():
    """Scene graph builds without crash"""
    from tools.scene_graph import SceneGraph
    from simulator.physics import create_ramp_scenario
    world = create_ramp_scenario()
    for _ in range(20):
        world.step(1/60)
    graph = SceneGraph("test")
    graph.build_from_physics(world.get_state())
    ascii_art = graph.to_ascii_art()
    assert len(ascii_art) > 100, "ASCII art too short"
    return True


def test_spatial():
    """Spatial zoning + perception work"""
    from tools.spatial_zoning import analyze_spatial_scene
    from tools.spatial_reasoning import OcclusionEngine, Observer, Pathfinder
    # Zoning
    result = analyze_spatial_scene("test", [
        {"id": "table", "pos": [0, 0], "radius": 2.0, "tags": ["reference"]},
        {"id": "chaise", "pos": [0, -1.5], "radius": 0.5, "tags": ["meuble"]},
    ], "table")
    assert "ANALYSE" in result, "Spatial analysis failed"
    # Pathfinding
    pf = Pathfinder(20, 20)
    pf.add_obstacle((10, 10), 2.0)
    pf.compute_clearance()
    result = pf.find_path((2, 2), (18, 18))
    assert result["found"], "Path not found"
    return True


def test_kinematic():
    """Kinematic engine runs"""
    from tools.kinematic_engine import create_four_bar_linkage, analyze_mechanism
    joints, links = create_four_bar_linkage()
    result = analyze_mechanism("test", joints, links)
    assert "MOBILITÉ" in result, "Kinematic analysis failed"
    return True


def test_advanced_physics():
    """Advanced physics engines run"""
    from tools.advanced_physics import (
        BeamAnalysis, Oscillator, ChaosAnalyzer
    )
    beam = BeamAnalysis("bois_chene", 3.0, 0.1, 0.2, 2000)
    assert beam.safety_factor() > 0, "Beam analysis failed"
    
    osc = Oscillator(100, 1000, 10)
    assert osc.natural_frequency_hz > 0, "Oscillator failed"
    
    chaos = ChaosAnalyzer.sensitivity_analysis(0.5, 3.9, 0.0001)
    assert "is_chaotic" in chaos, "Chaos analysis failed"
    return True


def test_relation_engine():
    """Relation engine runs"""
    from tools.relation_engine import RelationNetwork, RelationEdge, RelationCategory
    net = RelationNetwork("test")
    net.add_node("A", (0, 0))
    net.add_node("B", (1, 0))
    net.add_edge(RelationEdge("A", "B", RelationCategory.MECHANICAL, label="test"))
    result = net.to_ascii_network()
    assert len(result) > 100, "Network too small"
    return True


if __name__ == "__main__":
    tests = [
        ("imports", test_imports),
        ("physics_v3", test_physics_v3),
        ("discrete_classifier", test_discrete_classifier),
        ("scene_graph", test_scene_graph),
        ("spatial+pathfinding", test_spatial),
        ("kinematic", test_kinematic),
        ("advanced_physics", test_advanced_physics),
        ("relation_engine", test_relation_engine),
    ]
    
    passed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passés")
