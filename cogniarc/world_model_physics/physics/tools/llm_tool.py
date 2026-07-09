"""
World Model LLM Tool — Final interface.
The LLM calls these functions to simulate and predict physics.
Discrete classification: 8 states, 4 human-readable levels.
"""

import json
from .discrete_classifier import (
    MoveState, classify_per_body, predict_object_fate,
    train_discrete_model, DiscreteWorldModel, STATE_DESCRIPTIONS
)
from ..simulator.physics import SCENARIOS, create_ramp_scenario


def simulate(spec: str) -> str:
    """
    Simulate a physics scenario and describe what happens.
    
    Usage: simulate("ramp (ball on 45° wooden plank)")
    Returns: JSON with state classification per object
    """
    # Parse spec
    spec_lower = spec.lower()
    scenario = "ramp"
    for key in SCENARIOS:
        if key in spec_lower:
            scenario = key
            break
    
    # Run simulation
    world = SCENARIOS.get(scenario, create_ramp_scenario)()
    
    # Simulate a few seconds
    states_at = {}
    for step in range(180):
        world.step(1/60)
        t = world.time
        if abs(t - 1.0) < 0.05 or abs(t - 2.0) < 0.05 or step == 179:
            state = world.get_state()
            classification = classify_per_body(state)
            states_at[f"{t:.1f}s"] = {
                bid: {"state": info["state_name"], "speed": info["speed"], "desc": info["description"]}
                for bid, info in classification.items()
            }
    
    # Final prediction
    prediction = predict_object_fate(scenario)
    
    return json.dumps({
        "scenario": scenario,
        "objects": [b.id for b in world.bodies],
        "timeline": states_at,
        "prediction": prediction
    }, indent=2, ensure_ascii=False)


def predict(spec: str) -> str:
    """
    Predict what will happen next without running simulation.
    Uses trained discrete model.
    
    Usage: predict("ramp ball on wooden ramp 45°")
    """
    spec_lower = spec.lower()
    scenario = "ramp"
    for key in SCENARIOS:
        if key in spec_lower:
            scenario = key
            break
    
    result = predict_object_fate(scenario)
    
    # Format as readable summary
    lines = []
    for bid, r in result["predictions"].items():
        lines.append(f"• {bid}: {r['fate']}")
    
    return "\n".join(lines) + "\n\n--- Raw ---\n" + json.dumps(result, indent=2, ensure_ascii=False)


def classify(scenario: str = "ramp", object_id: str = None) -> str:
    """
    Classify the current state of objects in a scenario.
    
    Usage: classify("ramp") or classify("ramp", "ball")
    """
    world = SCENARIOS.get(scenario, create_ramp_scenario)()
    for _ in range(30):
        world.step(1/60)
    
    state = world.get_state()
    classification = classify_per_body(state)
    
    if object_id:
        info = classification.get(object_id, {})
        return json.dumps({object_id: info}, indent=2, ensure_ascii=False)
    
    return json.dumps(classification, indent=2, ensure_ascii=False)


def ask_llm_question(prompt: str) -> str:
    """
    Natural language interface. The LLM can ask:
    - "If I drop a metal ball from 8 meters onto a 45° wooden ramp, what happens?"
    - "Will the ball bounce or roll?"
    - "How do rubber and steel balls compare?"
    
    This function interprets the question and runs the appropriate simulation.
    """
    prompt_lower = prompt.lower()
    
    # Detect parameters
    height = 8.0
    angle = 45.0
    material_ball = "steel"
    material_ramp = "wood"
    
    if "rubber" in prompt_lower:
        material_ball = "rubber"
    elif "ice" in prompt_lower:
        material_ball = "ice"
    elif "stone" in prompt_lower:
        material_ball = "stone"
    
    if "60°" in prompt or "60 deg" in prompt_lower or "pente raide" in prompt_lower:
        angle = 60.0
    elif "30°" in prompt or "30 deg" in prompt_lower:
        angle = 30.0
    elif "20°" in prompt or "20 deg" in prompt_lower:
        angle = 20.0
    
    # Extract height
    import re
    heights = re.findall(r'(\d+)\s*m', prompt_lower)
    if heights:
        height = float(heights[0])
    
    # Build variant
    from simulator.physics import create_ramp_scenario
    world = create_ramp_scenario(
        ball_material=material_ball,
        ramp_material=material_ramp,
        drop_height=height,
        ramp_angle_deg=angle
    )
    
    # Simulate
    events = []
    prev_state = None
    contact_detected = False
    bounce_count = 0
    
    for step in range(300):
        world.step(1/60)
        state = world.get_state()
        
        # Detect contact
        ball = [b for b in world.bodies if b.id == "ball"][0]
        speed = (ball.velocity.x**2 + ball.velocity.y**2)**0.5
        
        if not contact_detected and state["contacts"] > 0:
            contact_detected = True
            events.append(f"t={(step/60):.2f}s: Balle touche la rampe — vitesse={speed:.1f}m/s")
        
        # Detect bounce (velocity reversal in Y)
        if prev_state:
            prev_ball = [b for b in prev_state["bodies"] if b["id"] == "ball"][0]
            if prev_ball["vel"][1] < -3 and ball.velocity.y > 1:
                bounce_count += 1
                events.append(f"t={(step/60):.2f}s: Rebond #{bounce_count} — vy={ball.velocity.y:.1f}m/s")
        
        prev_state = state
        if step == 299:
            classification = classify_per_body(state)
            ball_state = classification.get("ball", {})
            events.append(f"t={(step/60):.2f}s: Final — état={ball_state.get('state_name','?')} position=({ball.position.x:.1f},{ball.position.y:.1f})")
    
    prediction = predict_object_fate("ramp")
    ball_pred = prediction["predictions"].get("ball", {})
    
    answer = f"""
Scénario: Balle en {material_ball} lâchée de {height:.0f}m sur rampe {material_ramp} à {angle:.0f}°

Résultat:
{chr(10).join(f'  {e}' for e in events)}

Prédiction: {ball_pred.get('fate', '?')}
  Niveau: {ball_pred.get('movement_level', '?')}
  Confiance: {ball_pred.get('confidence', 0):.0%}
  État actuel: {ball_pred.get('current', '?')}
  État prédit: {ball_pred.get('predicted', '?')}

Raisonnement: La balle en {material_ball} (restitution={1.0 if material_ball=='rubber' else 0.6 if material_ball=='steel' else 0.5}) 
sur {material_ramp} ({angle:.0f}°) rebondit {bounce_count} fois avant de se stabiliser.
"""
    return answer


if __name__ == "__main__":
    print("=== World Model LLM Tool ===\n")
    
    # Demo: simulate
    print("1. SIMULATE:")
    result = simulate("ramp")
    print(result[:500])
    
    print("\n2. PREDICT:")
    result = predict("ramp ball on wood")
    print(result[:500])
    
    print("\n3. ASK LLM:")
    result = ask_llm_question("If I drop a rubber ball from 5 meters onto a 45° wooden ramp, what happens?")
    print(result)
