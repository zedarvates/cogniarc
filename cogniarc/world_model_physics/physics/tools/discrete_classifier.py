"""
Discrete State Classifier for World Model.
Predicts 8 qualitative states instead of continuous positions.
Designed for small LLM approximate reasoning.
"""

import numpy as np
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import IntEnum


# === 8 Discrete Movement States ===

class MoveState(IntEnum):
    """8 qualitative movement states for approximate prediction"""
    IMMOBILE = 0       # Object stays completely still
    SLIGHT_MOVE = 1    # Barely moves (sub-pixel drift)
    SLIDE_SLOW = 2     # Slides slowly on surface
    SLIDE_FAST = 3     # Slides/rolls quickly on surface
    FALLING = 4        # In free-fall (vertical motion dominant)
    BOUNCING = 5       # Active bouncing (rebound in progress)
    PROJECTILE = 6     # Airborne with significant horizontal motion
    COLLISION_CHAIN = 7  # About to trigger/participate in collision chain


class EffectType(IntEnum):
    """What physical effect dominates the object's behavior"""
    NONE = 0
    GRAVITY_DOMINANT = 1
    CONTACT_SLIDING = 2
    CONTACT_BOUNCING = 3
    MAGNETIC = 4
    BUOYANCY = 5
    CONSTRAINT_SPRING = 6
    MULTIPLE_COLLISIONS = 7


# === State Classification Logic ===

def classify_object_movement(velocity: np.ndarray, acceleration: np.ndarray,
                              on_ground: bool, in_fluid: bool, 
                              has_contacts: bool, magnetic_influence: float) -> int:
    """
    Classify object into one of 8 discrete movement states.
    
    Args:
        velocity: [vx, vy] current velocity
        acceleration: [ax, ay] current acceleration (from last frame)
        on_ground: True if object is in contact with a surface below
        in_fluid: True if object is submerged in fluid
        has_contacts: True if object is touching another object
        magnetic_influence: 0-1 how much magnetic force affects this object
    
    Returns:
        MoveState index (0-7)
    """
    speed = float(np.sqrt(velocity[0]**2 + velocity[1]**2))
    accel_mag = float(np.sqrt(acceleration[0]**2 + acceleration[1]**2))
    is_falling = velocity[1] < -1.0  # Significant downward velocity
    is_rising = velocity[1] > 1.0
    has_horizontal = abs(velocity[0]) > 0.5
    
    # Static conditions
    if speed < 0.05:
        if accel_mag < 0.01:
            return MoveState.IMMOBILE
        return MoveState.SLIGHT_MOVE
    
    # Magnetic dominance
    if magnetic_influence > 0.5 and speed > 0.5:
        return MoveState.SLIDE_FAST  # Magnetic pull overrides other states
    
    # Bouncing (recent velocity reversal in Y)
    if has_contacts and (is_rising or (accel_mag > 50 and speed > 1.0)):
        return MoveState.BOUNCING
    
    # Falling (free air, no ground contact, downward velocity)
    if not on_ground and not in_fluid and is_falling:
        if has_horizontal and speed > 5:
            return MoveState.PROJECTILE
        return MoveState.FALLING
    
    # Sliding on surface
    if on_ground and speed > 0.05:
        if speed > 3.0:
            return MoveState.SLIDE_FAST
        return MoveState.SLIDE_SLOW
    
    # Fluid effects
    if in_fluid:
        return MoveState.SLIDE_SLOW  # Buoyancy slows everything
    
    # Collision chain detection
    if has_contacts and speed > 1.0 and accel_mag > 20:
        return MoveState.COLLISION_CHAIN
    
    # Default: classify by speed
    if speed < 0.3:
        return MoveState.SLIGHT_MOVE
    elif speed < 2.0:
        return MoveState.SLIDE_SLOW
    elif speed < 6.0:
        return MoveState.SLIDE_FAST
    else:
        return MoveState.PROJECTILE


def classify_per_body(world_state: dict, prev_state: dict = None) -> dict:
    """
    Classify all bodies in a world state.
    Returns discrete states for each body.
    """
    dt = 1/60
    results = {}
    
    bodies = world_state.get("bodies", [])
    prev_bodies = {b["id"]: b for b in prev_state.get("bodies", [])} if prev_state else {}
    
    for b in bodies:
        bid = b["id"]
        if b.get("type") == "static":
            results[bid] = {
                "state": MoveState.IMMOBILE,
                "state_name": "IMMOBILE",
                "speed": 0.0,
                "description": "Static object, never moves"
            }
            continue
        
        vel = b.get("vel", [0, 0])
        speed = float(np.sqrt(vel[0]**2 + vel[1]**2))
        
        # Compute acceleration
        prev = prev_bodies.get(bid, {})
        prev_vel = prev.get("vel", vel)
        accel = [(vel[0] - prev_vel[0]) / dt, (vel[1] - prev_vel[1]) / dt]
        
        # Detect ground contact (is there a static body below?)
        on_ground = False
        pos = b.get("pos", [0, 0])
        for other in bodies:
            if other.get("type") == "static" and other["id"] != bid:
                opos = other.get("pos", [0, 0])
                # Is static body below this object and close enough?
                if opos[1] < pos[1] and abs(pos[0] - opos[0]) < 1.0:
                    dist = pos[1] - opos[1]
                    if dist < b.get("radius", 0.5) * 3:
                        on_ground = True
                        break
        
        in_fluid = b.get("pos", [0, 0])[1] < -5.0
        
        # Check contacts from world state
        has_contacts = world_state.get("contacts", 0) > 0
        
        # Magnetic influence (simplified)
        magnetic = 1.0 if b.get("material") == "Steel" else 0.0
        
        state = classify_object_movement(
            np.array(vel), np.array(accel),
            on_ground, in_fluid, has_contacts, magnetic
        )
        
        results[bid] = {
            "state": int(state),
            "state_name": MoveState(state).name,
            "speed": round(speed, 2),
            "velocity": vel,
            "on_ground": on_ground,
            "in_fluid": in_fluid,
            "description": STATE_DESCRIPTIONS.get(int(state), "Unknown")
        }
    
    return results


STATE_DESCRIPTIONS = {
    0: "Immobile — aucune force nette, objet au repos",
    1: "Mouvement léger — dérive très lente, quasi-statique",
    2: "Glissement lent — frottement dominant, vitesse réduite",
    3: "Glissement rapide — surface lisse ou forte poussée",
    4: "Chute libre — gravité pure, sans contact sol",
    5: "Rebond actif — collision élastique en cours",
    6: "Projectile — mouvement balistique avec composante horizontale",
    7: "Chaîne de collision — transfert d'impulsion entre objets",
}


# === Discrete Transition Model ===

@dataclass
class DiscreteWorldModel:
    """
    Learns P(next_state | current_state) transition probabilities.
    Tiny model: 8×8 × n_bodies transition matrices.
    """
    n_bodies: int = 0
    n_states: int = 8
    transition_counts: np.ndarray = None  # [n_bodies, 8, 8]
    transition_probs: np.ndarray = None   # [n_bodies, 8, 8]
    trained: bool = False
    
    def init(self, n_bodies: int):
        self.n_bodies = n_bodies
        # Initialize with prior: state tends to stay same
        self.transition_counts = np.ones((n_bodies, self.n_states, self.n_states))
        # Add identity prior (state_k → state_k+1 for falling, state_k for others)
        for i in range(self.n_states):
            self.transition_counts[:, i, i] += 10  # Strong self-transition prior
    
    def observe_transition(self, body_idx: int, from_state: int, to_state: int):
        self.transition_counts[body_idx, from_state, to_state] += 1
    
    def finalize(self):
        """Normalize counts to probabilities"""
        sums = self.transition_counts.sum(axis=2, keepdims=True)
        self.transition_probs = self.transition_counts / sums
        self.trained = True
    
    def predict_next_state(self, body_idx: int, current_state: int) -> Tuple[int, float, List[Tuple[int, float]]]:
        """Predict most likely next state with confidence"""
        if not self.trained:
            return current_state, 0.0, []
        
        probs = self.transition_probs[body_idx, current_state]
        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        
        # Top 3 predictions
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(int(i), float(probs[i])) for i in top3_idx]
        
        return top_idx, confidence, top3
    
    def predict_next_states(self, states: List[int], steps: int = 5) -> List[List[Tuple[int, float]]]:
        """Predict future states for all bodies over multiple steps"""
        n = len(states)
        trajectory = []
        current = list(states)
        
        for _ in range(steps):
            next_states = []
            for i, s in enumerate(current):
                pred, conf, top3 = self.predict_next_state(i, s)
                next_states.append((pred, conf))
            trajectory.append(next_states)
            current = [ns[0] for ns in next_states]
        
        return trajectory
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                "n_bodies": self.n_bodies,
                "n_states": self.n_states,
                "counts": self.transition_counts,
                "probs": self.transition_probs,
                "trained": self.trained
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "DiscreteWorldModel":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.n_bodies = data["n_bodies"]
        model.n_states = data["n_states"]
        model.transition_counts = data["counts"]
        model.transition_probs = data["probs"]
        model.trained = data["trained"]
        return model


# === Training Data Generator ===

def generate_discrete_training_data(world, steps: int = 2000) -> DiscreteWorldModel:
    """Run simulation and collect discrete state transitions"""
    n_bodies = sum(1 for b in world.bodies if b.body_type == "dynamic")
    model = DiscreteWorldModel()
    model.init(n_bodies)
    
    prev_state = world.get_state()
    prev_class = classify_per_body(prev_state)
    
    for i in range(steps):
        world.step(1/60)
        cur_state = world.get_state()
        cur_class = classify_per_body(cur_state, prev_state)
        
        # Record transitions
        for idx, (bid, info) in enumerate(cur_class.items()):
            if info["state_name"] == "IMMOBILE" and bid not in [b.id for b in world.bodies if b.body_type == "dynamic"]:
                continue
            prev_info = prev_class.get(bid, {})
            prev_s = prev_info.get("state", 0)
            cur_s = info["state"]
            model.observe_transition(idx, prev_s, cur_s)
        
        prev_state = cur_state
        prev_class = cur_class
    
    model.finalize()
    return model


# === LLM-Tool Interface ===

def predict_object_fate(scenario_name: str = "ramp", model_path: str = None) -> dict:
    """
    Predict what will happen to each object: stay still, move a little, moderately, or a lot.
    This is the main LLM-callable function for approximate reasoning.
    """
    from simulator.physics import SCENARIOS, create_ramp_scenario
    
    # Load or train model
    if model_path:
        model = DiscreteWorldModel.load(model_path)
    else:
        world = (SCENARIOS.get(scenario_name, create_ramp_scenario))()
        model = generate_discrete_training_data(world, steps=1500)
    
    # Run test simulation
    world = (SCENARIOS.get(scenario_name, create_ramp_scenario))()
    
    # Run a few steps for initial state
    for _ in range(10):
        world.step(1/60)
    
    state = world.get_state()
    prev_state = None
    classification = classify_per_body(state)
    
    # Predict next state for each body
    results = {}
    for i, (bid, info) in enumerate(classification.items()):
        if info["state_name"] == "IMMOBILE":
            results[bid] = {
                "current": info["state_name"],
                "predicted": "IMMOBILE",
                "confidence": 1.0,
                "fate": "fixe — ne bougera pas",
                "movement_level": "none"
            }
            continue
        
        pred, conf, top3 = model.predict_next_state(i % model.n_bodies, info["state"])
        
        # Quantize prediction to 4 human-readable levels
        if pred in [MoveState.IMMOBILE, MoveState.SLIGHT_MOVE]:
            level = "none"
            fate = "fixe — restera immobile"
        elif pred in [MoveState.SLIDE_SLOW]:
            level = "light"
            fate = "bouge un peu — glissement lent"
        elif pred in [MoveState.SLIDE_FAST, MoveState.FALLING]:
            level = "moderate"
            fate = "bouge moyennement — chute ou glisse"
        elif pred in [MoveState.BOUNCING, MoveState.PROJECTILE, MoveState.COLLISION_CHAIN]:
            level = "heavy"
            fate = "bouge beaucoup — rebondit, vole, ou percute"
        else:
            level = "moderate"
            fate = "bouge moyennement"
        
        results[bid] = {
            "current": info["state_name"],
            "current_speed": info["speed"],
            "predicted": MoveState(pred).name,
            "confidence": round(conf, 3),
            "fate": fate,
            "movement_level": level,
            "alternatives": [(MoveState(s).name, round(p, 3)) for s, p in top3]
        }
    
    return {
        "scenario": scenario_name,
        "num_bodies": len(results),
        "predictions": results,
        "summary": _summarize_fates(results)
    }


def _summarize_fates(results: dict) -> str:
    """Human-readable summary of what will happen"""
    lines = []
    for bid, r in results.items():
        lines.append(f"{bid}: {r['fate']} (confiance: {r['confidence']:.0%})")
    return "\n".join(lines)


def train_discrete_model(scenario: str = "ramp", steps: int = 2000) -> dict:
    """Train and save a discrete state transition model"""
    from simulator.physics import SCENARIOS, create_ramp_scenario
    
    world = (SCENARIOS.get(scenario, create_ramp_scenario))()
    print(f"Training discrete model on '{scenario}' scenario ({steps} frames)...")
    
    model = generate_discrete_training_data(world, steps)
    path = f"/home/redgamer/projects/world-model-tool/models/{scenario}_discrete.pkl"
    model.save(path)
    
    # Show learned transition matrix for body 0
    print(f"\nTransition matrix (body 0):")
    print("    " + " ".join(f"{MoveState(i).name[:5]:6s}" for i in range(8)))
    for i in range(8):
        row = " ".join(f"{model.transition_probs[0, i, j]:.3f}" for j in range(8))
        print(f"{MoveState(i).name[:5]:4s} {row}")
    
    return {"scenario": scenario, "model_path": path, "trained": True}


if __name__ == "__main__":
    # Train and test
    result = train_discrete_model("ramp", steps=2000)
    print(f"\nTrained: {result['model_path']}")
    
    # Test prediction
    pred = predict_object_fate("ramp")
    print(f"\n=== Prediction ===")
    print(json.dumps(pred, indent=2, ensure_ascii=False))
