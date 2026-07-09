"""
World Model Prediction Tool — LLM-callable interface.
The LLM can invoke this tool to:
1. Simulate physics scenarios
2. Predict future states
3. Query the world model

Designed for Hermes Agent tool integration.
"""

import json
import numpy as np
from ..simulator.physics import (
    PhysicsWorld, PhysicsBody, Vec2, SCENARIOS,
    create_ramp_scenario, generate_training_data
)
from ..models.trainer import SimpleWorldModel, WorldModelConfig

# Globals for tool state persistence
_current_world = None
_current_model = None


def simulate_scenario(scenario_type: str = "gravity", steps: int = 120, 
                      record_interval: int = 10) -> dict:
    """
    Simulate a physics scenario and return the trajectory.
    LLM can call this to get ground truth data.
    
    Args:
        scenario_type: "gravity", "pendulum", "fluid", "adhesion"
        steps: Number of simulation steps (at 60fps)
        record_interval: Record state every N steps
    
    Returns:
        dict with states, world description
    """
    global _current_world
    
    scenarios = dict(SCENARIOS)
    creator = scenarios.get(scenario_type, create_ramp_scenario)
    world = creator()
    _current_world = world
    
    states = []
    for i in range(steps):
        world.step(1/60)
        if i % record_interval == 0 or i == steps - 1:
            states.append(world.get_state())
    
    return {
        "scenario": scenario_type,
        "total_steps": steps,
        "bodies": world.get_state()["bodies"],
        "trajectory": states,
        "state_vector": world.get_state_vector().tolist(),
        "state_size": world.state_size(),
        "description": _describe_scenario(scenario_type, world)
    }


def predict_next_state(model_path: str = None, current_state: list = None) -> dict:
    """
    Predict the next physics state using the trained world model.
    Loads model from disk or uses in-memory model.
    
    Args:
        model_path: Path to trained model .pkl file
        current_state: Current state vector as list [x1,y1,vx1,vy1,...]
    
    Returns:
        dict with predicted next state
    """
    global _current_model
    
    if model_path and _current_model is None:
        _current_model = SimpleWorldModel.load(model_path)
    
    if _current_model is None:
        return {"error": "No model loaded. Train or load a model first."}
    
    if current_state is None:
        # Use current world state if available
        global _current_world
        if _current_world:
            current_state = _current_world.get_state_vector().tolist()
        else:
            return {"error": "No state provided and no current simulation running."}
    
    state_array = np.array(current_state, dtype=np.float32).reshape(1, -1)
    prediction = _current_model.predict(state_array).reshape(-1)
    
    # Format as per-body predictions
    n_bodies = len(current_state) // 4
    bodies = []
    for i in range(n_bodies):
        idx = i * 4
        bodies.append({
            "body_index": i,
            "pos_predicted": [round(float(prediction[idx]), 3), round(float(prediction[idx+1]), 3)],
            "vel_predicted": [round(float(prediction[idx+2]), 3), round(float(prediction[idx+3]), 3)],
            "pos_current": [round(current_state[idx], 3), round(current_state[idx+1], 3)],
            "vel_current": [round(current_state[idx+2], 3), round(current_state[idx+3], 3)]
        })
    
    return {
        "state_size": len(current_state),
        "num_bodies": n_bodies,
        "predicted_state": prediction.tolist(),
        "bodies": bodies
    }


def predict_multiple_steps(model_path: str = None, steps: int = 30, 
                           current_state: list = None) -> dict:
    """
    Predict multiple steps ahead (autoregressive rollout).
    Shows the model's long-term prediction capability.
    """
    global _current_model
    
    if model_path and _current_model is None:
        _current_model = SimpleWorldModel.load(model_path)
    
    if _current_model is None:
        return {"error": "No model loaded."}
    
    if current_state is None:
        global _current_world
        if _current_world:
            current_state = _current_world.get_state_vector().tolist()
        else:
            return {"error": "No state provided."}
    
    state_array = np.array(current_state, dtype=np.float32)
    predictions = _current_model.predict_multi(state_array, steps=steps)
    
    trajectory = []
    for p in predictions:
        trajectory.append(p.tolist())
    
    return {
        "initial_state": current_state,
        "steps": steps,
        "trajectory": trajectory,
        "drift_analysis": _analyze_drift(current_state, predictions)
    }


def compare_prediction_vs_ground_truth(scenario_type: str = "gravity", 
                                        model_path: str = None) -> dict:
    """
    Run simulation and compare model predictions vs actual physics.
    This is the core validation: how well does the small model approximate real physics?
    """
    global _current_world, _current_model
    
    # Setup
    scenarios = {
        "gravity": create_gravity_scene,
        "pendulum": create_pendulum_scene,
        "fluid": create_fluid_scene,
        "adhesion": create_adhesion_scene
    }
    world = scenarios.get(scenario_type, create_gravity_scene)()
    _current_world = world
    
    if model_path:
        _current_model = SimpleWorldModel.load(model_path)
    
    if _current_model is None:
        # Auto-load matching model
        try:
            _current_model = SimpleWorldModel.load(
                f"/home/redgamer/projects/world-model-tool/models/{scenario_type}_model.pkl")
        except:
            return {"error": "No model available. Train first."}
    
    initial_state = world.get_state_vector()
    state_array = initial_state.copy()
    
    results = []
    mse_per_step = []
    
    for step in range(60):
        # Get ground truth
        world.step(1/60)
        ground_truth = world.get_state_vector().tolist()
        
        # Get prediction
        prediction = _current_model.predict(state_array.reshape(1, -1)).reshape(-1).tolist()
        
        # MSE
        mse = np.mean((np.array(prediction) - np.array(ground_truth)) ** 2)
        mse_per_step.append(float(mse))
        
        if step % 10 == 0 or step == 59:
            results.append({
                "step": step,
                "mse": round(float(mse), 6),
                "prediction": [round(x, 3) for x in prediction[:8]],
                "ground_truth": [round(x, 3) for x in ground_truth[:8]]
            })
        
        # Feed prediction back as next input (autoregressive)
        state_array = np.array(prediction, dtype=np.float32)
    
    return {
        "scenario": scenario_type,
        "steps": 60,
        "mean_mse": round(float(np.mean(mse_per_step)), 6),
        "max_mse": round(float(np.max(mse_per_step)), 6),
        "min_mse": round(float(np.min(mse_per_step)), 6),
        "results": results,
        "assessment": _assess_model_quality(np.mean(mse_per_step))
    }


def create_custom_scenario(bodies_config: list, gravity_y: float = -9.81,
                           bounds: list = [-10, -10, 10, 10], steps: int = 60) -> dict:
    """
    Create a custom physics scenario from a JSON description.
    The LLM can design its own scenarios to test predictions.
    
    Args:
        bodies_config: List of body dicts with id, pos, vel, mass, radius, type
        gravity_y: Gravity strength (negative = downward)
        bounds: World bounds [min_x, min_y, max_x, max_y]
    """
    global _current_world
    
    world = PhysicsWorld(
        gravity=Vec2(0, gravity_y),
        world_bounds=(bounds[0], bounds[1], bounds[2], bounds[3])
    )
    
    for bc in bodies_config:
        world.add_body(PhysicsBody(
            id=bc.get("id", f"body_{len(world.bodies)}"),
            position=Vec2(bc.get("pos", [0, 0])[0], bc.get("pos", [0, 0])[1]),
            velocity=Vec2(bc.get("vel", [0, 0])[0], bc.get("vel", [0, 0])[1]),
            mass=bc.get("mass", 1.0),
            radius=bc.get("radius", 0.5),
            body_type=bc.get("type", "dynamic"),
            constraints=bc.get("constraints", []),
            color=bc.get("color", "#888888")
        ))
    
    _current_world = world
    
    states = []
    for i in range(steps):
        world.step(1/60)
        if i % 5 == 0 or i == steps - 1:
            states.append(world.get_state())
    
    return {
        "scenario": "custom",
        "total_steps": steps,
        "num_bodies": len(world.bodies),
        "trajectory": states,
        "state_vector": world.get_state_vector().tolist()
    }


def train_model(scenario_type: str = "gravity", steps: int = 3000, epochs: int = 150) -> dict:
    """
    Train a world model on a specific scenario.
    Call this before using predict functions.
    """
    from models.trainer import train_on_scenario
    
    scenarios = dict(SCENARIOS)
    creator = scenarios.get(scenario_type, create_ramp_scenario)
    model = train_on_scenario(scenario_type, creator, steps=steps, epochs=epochs)
    path = f"/home/redgamer/projects/world-model-tool/models/{scenario_type}_model.pkl"
    model.save(path)
    
    return {
        "scenario": scenario_type,
        "trained": True,
        "model_path": path,
        "final_loss": model.training_history[-1]["train_loss"],
        "config": {
            "hidden_size": model.config.hidden_size,
            "epochs": epochs,
            "training_steps": steps
        }
    }


# === Helpers ===

def _describe_scenario(scenario_type: str, world: PhysicsWorld) -> str:
    descriptions = {
        "gravity": "Multiple balls falling under gravity with collisions and bouncing off walls.",
        "pendulum": "Chain of connected bodies swinging like a pendulum from a fixed anchor.",
        "fluid": "Particles falling into simulated fluid zone with buoyancy and drag forces.",
        "adhesion": "Bodies attracting each other at close range, forming clusters."
    }
    return descriptions.get(scenario_type, "Custom physics scenario.")


def _analyze_drift(initial_state: list, predictions: list) -> dict:
    """Analyze how much predictions drift from physics laws"""
    initial = np.array(initial_state)
    
    total_energy_initial = 0
    for i in range(0, len(initial), 4):
        vx, vy = initial[i+2], initial[i+3]
        total_energy_initial += vx**2 + vy**2
    
    energies = []
    for pred in predictions:
        p = np.array(pred)
        energy = 0
        for i in range(0, len(p), 4):
            vx, vy = p[i+2], p[i+3]
            energy += vx**2 + vy**2
        energies.append(float(energy))
    
    return {
        "initial_energy": round(total_energy_initial, 3),
        "final_energy": round(energies[-1], 3),
        "energy_drift_pct": round(100 * (energies[-1] - total_energy_initial) / max(total_energy_initial, 1e-10), 1),
        "stable": abs(energies[-1] - total_energy_initial) / max(total_energy_initial, 1e-10) < 0.5
    }


def _assess_model_quality(mse: float) -> str:
    """Assess model quality based on MSE (adjusted for approximate reasoning)"""
    if mse < 0.01:
        return "Excellent — near-perfect physics approximation"
    elif mse < 0.5:
        return "Good — predictions close to ground truth, usable for planning"
    elif mse < 2.0:
        return "Fair — approximate prediction with some drift, usable for short horizons"
    elif mse < 10.0:
        return "Approximate — captures general trends, not exact positions. OK for small LLM reasoning."
    else:
        return "Drifts — works for 1-step, needs more training for multi-step"


if __name__ == "__main__":
    # Demo: train model and predict
    print("=== World Model Prediction Tool ===")
    
    # Train
    result = train_model("gravity", steps=2000, epochs=100)
    print(json.dumps(result, indent=2))
    
    # Simulate
    sim = simulate_scenario("gravity", steps=120)
    print(f"\nSimulation: {sim['scenario']} - {sim['total_steps']} steps")
    
    # Predict
    pred = predict_next_state(
        model_path="/home/redgamer/projects/world-model-tool/models/gravity_model.pkl",
        current_state=sim['state_vector']
    )
    print(f"\nPrediction: {pred['num_bodies']} bodies")
    for b in pred['bodies']:
        print(f"  Body {b['body_index']}: pos=({b['pos_current'][0]:.2f},{b['pos_current'][1]:.2f}) → ({b['pos_predicted'][0]:.2f},{b['pos_predicted'][1]:.2f})")
    
    # Compare
    comp = compare_prediction_vs_ground_truth("gravity")
    print(f"\nComparison: mean_mse={comp['mean_mse']}")
    print(f"Assessment: {comp['assessment']}")
