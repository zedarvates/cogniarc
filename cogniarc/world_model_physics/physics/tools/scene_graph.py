"""
Symbolic Scene Graph for LLM Mental Visualization.
Converts physics scenes into tagged graphs the LLM can reason about.
Like a human's mental napkin sketch — primitives, labels, relationships.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum


# === Graph Node Types ===

class NodeType(Enum):
    OBJECT = "object"
    SURFACE = "surface"
    CONTAINER = "container"
    VEHICLE = "vehicle"
    PERSON = "person"
    OBSTACLE = "obstacle"
    FORCE_FIELD = "force_field"

class RelationType(Enum):
    ON_TOP_OF = "on_top_of"
    INSIDE = "inside"
    CONNECTED_TO = "connected_to"
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    MOVING_TOWARD = "moving_toward"
    COLLIDING_WITH = "colliding_with"
    SUPPORTS = "supports"
    CONTAINS = "contains"
    ATTRACTED_TO = "attracted_to"


# === Symbolic Primitives ===

@dataclass
class SymbolicObject:
    """Mental representation of an object — like a labeled sketch"""
    id: str
    shape: str          # "circle", "box", "triangle", "line"
    size: str           # "tiny", "small", "medium", "large", "huge"
    material: str       # "steel", "wood", "rubber", etc.
    color: str          # "#ff4444"
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    tags: List[str] = field(default_factory=list)  # ["heavy", "bouncy", "magnetic", "fragile"]
    role: str = ""      # "projectile", "target", "obstacle", "ramp", "ground"
    movement_state: str = "unknown"  # discrete state
    neighbors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "shape": self.shape,
            "size": self.size,
            "material": self.material,
            "color": self.color,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "speed": round(float(np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)), 2),
            "tags": self.tags,
            "role": self.role,
            "state": self.movement_state
        }


# === Scene Graph ===

class SceneGraph:
    """
    Symbolic scene representation for LLM reasoning.
    Nodes = objects, Edges = spatial relationships.
    Can be rendered as 2D primitives or ASCII art.
    """
    
    def __init__(self, name: str = "scene"):
        self.name = name
        self.objects: Dict[str, SymbolicObject] = {}
        self.relations: List[Tuple[str, str, RelationType]] = []
        self.time: float = 0.0
        self.scenario_type: str = ""
    
    def add_object(self, obj: SymbolicObject):
        self.objects[obj.id] = obj
    
    def add_relation(self, source: str, target: str, relation: RelationType):
        self.relations.append((source, target, relation))
    
    def build_from_physics(self, world_state: dict):
        """Convert physics world state to symbolic graph"""
        bodies = world_state.get("bodies", [])
        
        # Create nodes
        for b in bodies:
            shape = b.get("shape", "circle")
            radius = b.get("radius", 0.5)
            
            # Determine symbolic size
            if radius < 0.3:
                size = "tiny"
            elif radius < 0.6:
                size = "small"
            elif radius < 1.0:
                size = "medium"
            elif radius < 2.0:
                size = "large"
            else:
                size = "huge"
            
            # Determine role
            role = "object"
            if b.get("type") == "static":
                if "ground" in b["id"] or "floor" in b["id"]:
                    role = "ground"
                elif "wall" in b["id"]:
                    role = "obstacle"
                elif "ramp" in b["id"]:
                    role = "ramp"
                else:
                    role = "obstacle"
            elif b.get("material") == "Steel" and "magnet" in b["id"]:
                role = "magnet"
            elif abs(b.get("vel", [0, 0])[1]) > 3:
                role = "projectile"
            
            # Auto-tagging
            tags = []
            mat = b.get("material", "default")
            if mat == "Steel":
                tags.extend(["heavy", "magnetic", "conductive"])
            elif mat == "Rubber":
                tags.extend(["light", "bouncy", "elastic"])
            elif mat == "Wood":
                tags.extend(["medium", "breakable", "flammable"])
            elif mat == "Ice":
                tags.extend(["light", "slippery", "fragile"])
            elif mat == "Stone":
                tags.extend(["heavy", "hard", "immovable"])
            
            vel = b.get("vel", [0, 0])
            speed = (vel[0]**2 + vel[1]**2)**0.5
            if speed > 10:
                tags.append("fast")
            elif speed > 3:
                tags.append("moving")
            elif speed < 0.1:
                tags.append("still")
            
            obj = SymbolicObject(
                id=b["id"],
                shape=shape,
                size=size,
                material=mat,
                color=b.get("color", "#888"),
                position=tuple(b.get("pos", [0, 0])),
                velocity=tuple(vel),
                tags=tags,
                role=role
            )
            self.add_object(obj)
        
        # Build spatial relationships
        self._build_spatial_relations()
        self._build_contact_relations(world_state)
    
    def _build_spatial_relations(self):
        """Build LEFT_OF, RIGHT_OF, ABOVE, BELOW based on positions"""
        for id1, obj1 in self.objects.items():
            for id2, obj2 in self.objects.items():
                if id1 >= id2:
                    continue
                
                dx = obj2.position[0] - obj1.position[0]
                dy = obj2.position[1] - obj1.position[1]
                
                if abs(dx) > 0.5:
                    if dx > 0:
                        self.add_relation(id2, id1, RelationType.RIGHT_OF)
                        self.add_relation(id1, id2, RelationType.LEFT_OF)
                    else:
                        self.add_relation(id1, id2, RelationType.RIGHT_OF)
                        self.add_relation(id2, id1, RelationType.LEFT_OF)
                
                if abs(dy) > 0.5:
                    if dy > 0:
                        self.add_relation(id2, id1, RelationType.ABOVE)
                        self.add_relation(id1, id2, RelationType.BELOW)
                    else:
                        self.add_relation(id1, id2, RelationType.ABOVE)
                        self.add_relation(id2, id1, RelationType.BELOW)
    
    def _build_contact_relations(self, world_state: dict):
        """Detect contacts, containments, collisions"""
        contacts = world_state.get("contacts", 0)
        dynamic = [b for b in world_state.get("bodies", []) if b.get("type") == "dynamic"]
        static = [b for b in world_state.get("bodies", []) if b.get("type") == "static"]
        
        # Check if objects are on top of static surfaces
        for d in dynamic:
            dpos = d.get("pos", [0, 0])
            dvy = d.get("vel", [0, 0])[1]
            
            for s in static:
                spos = s.get("pos", [0, 0])
                
                # Is dynamic object ON the static one?
                vertical_dist = dpos[1] - spos[1]
                min_dist = d.get("radius", 0.5) + s.get("radius", 1.0)
                
                if 0 < vertical_dist < min_dist + 0.5 and abs(dvy) < 2.0:
                    self.add_relation(d["id"], s["id"], RelationType.ON_TOP_OF)
                    self.add_relation(s["id"], d["id"], RelationType.SUPPORTS)
                    self.objects[d["id"]].tags.append("grounded")
        
        # Movement toward detection
        for i, d1 in enumerate(dynamic):
            for d2 in dynamic[i+1:]:
                p1, v1 = np.array(d1.get("pos", [0, 0])), np.array(d1.get("vel", [0, 0]))
                p2, v2 = np.array(d2.get("pos", [0, 0])), np.array(d2.get("vel", [0, 0]))
                
                # Relative velocity toward each other
                rel_pos = p2 - p1
                rel_vel = v2 - v1
                closing = -np.dot(rel_pos, rel_vel) if np.linalg.norm(rel_pos) > 1e-10 else 0
                
                if closing > 10:  # Fast approach
                    self.add_relation(d1["id"], d2["id"], RelationType.MOVING_TOWARD)
                    self.add_relation(d2["id"], d1["id"], RelationType.MOVING_TOWARD)
                    
                    # Check if they'll collide soon
                    dist = np.linalg.norm(rel_pos)
                    r_sum = d1.get("radius", 0.5) + d2.get("radius", 0.5)
                    if dist < r_sum * 3 and closing > 0:
                        self.add_relation(d1["id"], d2["id"], RelationType.COLLIDING_WITH)
    
    def to_graphviz(self) -> str:
        """Export as DOT graph (for visualization)"""
        lines = ["digraph SceneGraph {"]
        lines.append(f'  label="{self.name} (t={self.time:.1f}s)";')
        lines.append('  node [shape=box, style=filled];')
        
        for obj_id, obj in self.objects.items():
            label = f"{obj.id}\\n{obj.shape} {obj.size}\\n{obj.material} {obj.role}\\n[{','.join(obj.tags[:3])}]"
            color = obj.color.replace("#", "")
            lines.append(f'  "{obj_id}" [label="{label}", fillcolor="#{color}22", fontsize=10];')
        
        for src, tgt, rel in self.relations:
            rel_color = {
                RelationType.ON_TOP_OF: "green",
                RelationType.COLLIDING_WITH: "red",
                RelationType.MOVING_TOWARD: "orange",
                RelationType.SUPPORTS: "blue",
            }.get(rel, "gray")
            lines.append(f'  "{src}" -> "{tgt}" [label="{rel.value}", color={rel_color}, fontsize=8];')
        
        lines.append("}")
        return "\n".join(lines)
    
    def to_ascii_art(self, width: int = 60, height: int = 30) -> str:
        """Render as 2D ASCII art — like a napkin sketch"""
        # Find bounds
        all_pos = [obj.position for obj in self.objects.values()]
        if not all_pos:
            return "(empty)"
        
        xs = [p[0] for p in all_pos]
        ys = [p[1] for p in all_pos]
        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2
        
        # Create canvas
        canvas = [[' ' for _ in range(width)] for _ in range(height)]
        
        def world_to_screen(x, y):
            if max_x == min_x:
                sx = width // 2
            else:
                sx = int((x - min_x) / (max_x - min_x) * (width - 1))
            if max_y == min_y:
                sy = height // 2
            else:
                sy = height - 1 - int((y - min_y) / (max_y - min_y) * (height - 1))
            return max(0, min(width-1, sx)), max(0, min(height-1, sy))
        
        # Draw objects
        for obj in self.objects.values():
            cx, cy = world_to_screen(obj.position[0], obj.position[1])
            r = max(1, int(obj.size.replace("tiny","1").replace("small","2").replace("medium","3").replace("large","4").replace("huge","5")))
            if obj.shape == "circle":
                symbol = "O"
            elif obj.shape == "box":
                symbol = "#"
            elif obj.shape == "plane":
                symbol = "="
            else:
                symbol = "?"
            
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < width and 0 <= py < height:
                            canvas[py][px] = symbol
            
            # Label
            label = obj.id[:3]
            for i, ch in enumerate(label):
                px, py = cx + i - 1, cy
                if 0 <= px < width and 0 <= py < height:
                    canvas[py][px] = ch
        
        result = []
        result.append(f"+{'-'*width}+")
        result.append(f"| {'Scene: ' + self.name:<{width-2}} |")
        result.append(f"+{'-'*width}+")
        for row in canvas:
            result.append("|" + "".join(row) + "|")
        result.append(f"+{'-'*width}+")
        
        # Legend
        result.append("\nLegend:")
        for obj in self.objects.values():
            tags = ','.join(obj.tags[:3])
            result.append(f"  {obj.id[:3]:<3} = {obj.shape} {obj.material} ({obj.role}) [{tags}]")
        
        for rel in self.relations[:10]:
            src, tgt, rtype = rel
            result.append(f"  {src} → {rtype.value} → {tgt}")
        
        return "\n".join(result)
    
    def to_llm_context(self) -> str:
        """Format as structured text for LLM consumption"""
        lines = [f"=== SCENE GRAPH: {self.name} ==="]
        lines.append(f"Time: {self.time:.2f}s | {len(self.objects)} objects | {len(self.relations)} relations\n")
        
        lines.append("OBJECTS:")
        for obj in sorted(self.objects.values(), key=lambda o: o.role):
            tags = ', '.join(obj.tags[:5])
            speed = round(float(np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)), 1)
            lines.append(
                f"  [{obj.role:>10}] {obj.id:<15} {obj.shape:<6} {obj.size:<6} "
                f"{obj.material:<8} pos=({obj.position[0]:5.1f},{obj.position[1]:5.1f}) "
                f"v={speed:.1f}m/s [{tags}]"
            )
        
        if self.relations:
            lines.append("\nRELATIONS:")
            seen = set()
            for src, tgt, rtype in sorted(self.relations, key=lambda r: r[2].value):
                key = (src, tgt, rtype)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  {src} --{rtype.value}--> {tgt}")
        
        lines.append(f"\nSCENARIO TYPE: {self.scenario_type}")
        return "\n".join(lines)


# === Scenario Predictor (Graph-based reasoning) ===

class ScenarioPredictor:
    """
    Uses the scene graph + discrete state model to predict:
    - What will stay still, move a little, moderately, or a lot
    - Which objects will collide
    - Chain reactions
    """
    
    def __init__(self):
        self.fall_threshold = -2.0  # m/s downward = falling
        self.impact_threshold = 8.0   # m/s = likely to cause bounce/chain
    
    def analyze_current_state(self, graph: SceneGraph) -> dict:
        """Analyze the scene graph to predict outcomes without simulation"""
        analysis = {
            "stable_objects": [],
            "moving_objects": [],
            "potential_collisions": [],
            "falling_objects": [],
            "chain_risk": "none"
        }
        
        for obj_id, obj in graph.objects.items():
            speed = round(float(np.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)), 1)
            
            if speed < 0.1:
                analysis["stable_objects"].append({
                    "id": obj_id,
                    "reason": "virtually still",
                    "supported_by": [tgt for src, tgt, rel in graph.relations 
                                    if src == obj_id and rel == RelationType.ON_TOP_OF]
                })
            elif speed < 2.0:
                analysis["moving_objects"].append({
                    "id": obj_id,
                    "level": "light",
                    "speed": speed,
                    "fate": "should settle soon if friction is present"
                })
            elif speed < 6.0:
                level = "moderate"
                fate = "will continue moving"
                if obj.velocity[1] < self.fall_threshold:
                    level = "moderate"
                    fate = "falling, will accelerate"
                    analysis["falling_objects"].append({"id": obj_id, "speed": speed})
                analysis["moving_objects"].append({
                    "id": obj_id, "level": level, "speed": speed, "fate": fate
                })
            else:
                analysis["moving_objects"].append({
                    "id": obj_id,
                    "level": "heavy",
                    "speed": speed,
                    "fate": "fast motion, likely to cause impact on collision"
                })
            
            # Collision prediction
            for src, tgt, rel in graph.relations:
                if src == obj_id and rel == RelationType.COLLIDING_WITH:
                    analysis["potential_collisions"].append({
                        "objects": [src, tgt],
                        "severity": "high" if speed > self.impact_threshold else "moderate"
                    })
        
        # Chain reaction risk
        if len(analysis["potential_collisions"]) > 1:
            analysis["chain_risk"] = "high — multiple simultaneous collisions"
        elif len(analysis["potential_collisions"]) == 1:
            analysis["chain_risk"] = "moderate — single collision likely"
        else:
            analysis["chain_risk"] = "none — no imminent collisions"
        
        return analysis
    
    def predict_final_state(self, graph: SceneGraph, discrete_model=None) -> str:
        """
        Human-readable prediction of the final state.
        Uses graph reasoning + discrete transition model.
        """
        analysis = self.analyze_current_state(graph)
        lines = []
        
        # Stable objects
        stable = len(analysis["stable_objects"])
        if stable > 0:
            names = [o["id"] for o in analysis["stable_objects"][:3]]
            lines.append(f"→ {', '.join(names)} resteront immobiles.")
        
        # Moving
        for obj in analysis["moving_objects"]:
            if obj["level"] == "light":
                lines.append(f"→ {obj['id']} bouge un peu ({obj['fate']}).")
            elif obj["level"] == "moderate":
                lines.append(f"→ {obj['id']} bouge modérément ({obj['fate']}).")
            else:
                lines.append(f"→ {obj['id']} bouge beaucoup ({obj['fate']}).")
        
        # Collisions
        for col in analysis["potential_collisions"]:
            a, b = col["objects"]
            lines.append(f"→ COLLISION entre {a} et {b} (sévérité: {col['severity']}).")
        
        # Chain
        lines.append(f"→ Risque de réaction en chaîne: {analysis['chain_risk']}.")
        
        # Final settlement
        lines.append(f"\nScénario final probable:")
        n_dynamic = len([o for o in graph.objects.values() if o.role not in ("ground", "obstacle", "ramp")])
        lines.append(f"  Sur {n_dynamic} objets dynamiques, {stable} se stabilisent.")
        
        n_moving = len(analysis["moving_objects"])
        lines.append(f"  {n_moving} continuent de bouger {'(peu)' if all(o['level'] != 'heavy' for o in analysis['moving_objects']) else '(certains beaucoup)'}.")
        
        return "\n".join(lines)


# === Demo ===

def demo_scene_graph():
    from ..simulator.physics import create_ramp_scenario, SCENARIOS
    
    # Simulate ramp scene
    world = create_ramp_scenario(drop_height=8.0, ramp_angle_deg=45)
    
    # Run a few steps
    for _ in range(50):
        world.step(1/60)
    
    state = world.get_state()
    
    # Build graph
    graph = SceneGraph("Ramp Scenario")
    graph.scenario_type = "ramp"
    graph.time = round(world.time, 2)
    graph.build_from_physics(state)
    
    print("=== MENTAL SKETCH (ASCII) ===\n")
    print(graph.to_ascii_art())
    
    print("\n=== LLM CONTEXT ===\n")
    print(graph.to_llm_context())
    
    print("\n=== GRAPHVIZ DOT ===\n")
    print(graph.to_graphviz()[:500])
    
    print("\n=== SCENARIO PREDICTION ===\n")
    predictor = ScenarioPredictor()
    prediction = predictor.predict_final_state(graph)
    print(prediction)


if __name__ == "__main__":
    demo_scene_graph()
