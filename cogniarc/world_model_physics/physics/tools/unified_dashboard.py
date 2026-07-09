"""
World Model Unified Dashboard — All V3 systems integrated.
Launch: python3 tools/unified_dashboard.py
"""

from ..simulator.physics_v3 import (
    PhysicsWorldV3, V3_SCENARIOS,
    CausalEvent, EventType, CompoundBody, CompoundType,
    EnergySnapshot, AgentGoal, GoalType, ThermalProperties
)
from ..simulator.physics import Vec2, Shape, ShapeType, PhysicsBody, MATERIALS


def run_full_analysis(scenario: str = "vehicle", steps: int = 180):
    """Run simulation and collect all system outputs for LLM consumption"""
    
    if scenario not in V3_SCENARIOS:
        print(f"Unknown scenario '{scenario}'. Available: {list(V3_SCENARIOS.keys())}")
        return
    
    world = V3_SCENARIOS[scenario]()
    
    print(f"\n{'='*70}")
    print(f"  WORLD MODEL V3 — {scenario.upper()}")
    print(f"{'='*70}")
    
    # Run
    for i in range(steps):
        world.step(1/60)
    
    state = world.get_full_state()
    
    # === 1. CAUSAL GRAPH ===
    print(f"\n{'─'*50}")
    print("  1. GRAPHE CAUSAL")
    print(f"{'─'*50}")
    print(f"  Événements enregistrés: {state['causal']['event_count']}")
    for obj, supported in state['causal']['support_graph'].items():
        print(f"  {obj} soutient: {', '.join(supported)}")
    if state['causal']['last_event']:
        print(f"  Dernier: {state['causal']['last_event']}")
    
    # Counterfactual
    dynamic = [b for b in world.bodies if b.body_type == "dynamic"]
    if dynamic:
        obj = dynamic[0].id
        print(f"\n  ⚡ Contrefactuel: {world.causal.what_if_removed(obj)}")
    
    # === 2. COMPOUND BODIES ===
    print(f"\n{'─'*50}")
    print("  2. CORPS COMPOSÉS")
    print(f"{'─'*50}")
    for cid, c in world.compounds.compounds.items():
        body = next((b for b in world.bodies if b.id == cid), None)
        vel = body.velocity if body else Vec2(0, 0)
        print(f"  {cid}: {c.compound_type.value} (parent={c.parent_id}) "
              f"v={vel.length():.1f}m/s tags={c.tags}")
    
    # === 3. ENERGY BUDGET ===
    print(f"\n{'─'*50}")
    print("  3. BUDGET ÉNERGÉTIQUE")
    print(f"{'─'*50}")
    print(world.energy.get_llm_summary())
    
    # === 4. TIME REVERSAL ===
    print(f"\n{'─'*50}")
    print("  4. HISTORIQUE TEMPOREL")
    print(f"{'─'*50}")
    print(f"  Snaphots: {len(world.time_reversal.state_history)}")
    if dynamic:
        print(f"  {world.time_reversal.trace_origin(dynamic[0].id)}")
    
    # === 5. AGENTS ===
    print(f"\n{'─'*50}")
    print("  5. AGENTS")
    print(f"{'─'*50}")
    if world.agents.goals:
        print(world.agents.get_status())
    else:
        print("  Aucun agent actif dans ce scénario")
    
    # === 6. THERMAL ===
    print(f"\n{'─'*50}")
    print("  6. SYSTÈME THERMIQUE")
    print(f"{'─'*50}")
    print(world.thermal.get_llm_summary() if world.thermal.thermal_props else "  Aucun suivi thermique")
    
    # === SUMMARY ===
    print(f"\n{'='*70}")
    print(f"  RÉSUMÉ POUR LLM")
    print(f"{'='*70}")
    
    bodies_dynamic = [b for b in world.bodies if b.body_type == "dynamic"]
    bodies_static = [b for b in world.bodies if b.body_type == "static"]
    
    summary = f"""
Le monde '{scenario}' contient {len(bodies_dynamic)} objets dynamiques et {len(bodies_static)} statiques.

Objets en mouvement:
{chr(10).join(f"  • {b.id} ({b.material.name}) → v={b.velocity.length():.1f}m/s" for b in bodies_dynamic)}

Énergie totale: {state['energy']['total']:.0f}J (dérive: {state['energy']['drift_pct']}%)
Événements causaux: {state['causal']['event_count']}

Ce que le LLM peut en déduire:
  → L'énergie est {'conservée' if abs(state['energy']['drift_pct']) < 1 else 'partiellement dissipée'}.
  → {len(state['causal']['support_graph'])} relations de support structurel existent.
  → {'Présence de corps composés (véhicules/conteneurs).' if world.compounds.compounds else 'Pas de corps composés.'}
"""
    print(summary)
    
    return world, state


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('scenario', nargs='?', default='vehicle', choices=list(V3_SCENARIOS.keys()))
    p.add_argument('--steps', type=int, default=180)
    args = p.parse_args()
    run_full_analysis(args.scenario, args.steps)
