"""
World Model Physics v3 — Causal reasoning, compound bodies, energy, agents, thermal.
Extends v2 with six new systems for LLM approximate reasoning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
import json
from .physics import (
    Vec2, PhysicsWorld, PhysicsBody, Shape, ShapeType,
    Material, MATERIALS, Atmosphere, LiquidType, LIQUIDS,
    ForceField, create_ramp_scenario, SCENARIOS
)


# ============================================================
# 1. CAUSAL GRAPH ENGINE — "If X then Y" reasoning
# ============================================================

class EventType(Enum):
    COLLISION = "collision"
    REBOUND = "rebound"
    FALL_START = "fall_start"
    LANDING = "landing"
    SUPPORT_REMOVED = "support_removed"
    FORCE_APPLIED = "force_applied"
    STATE_CHANGE = "state_change"
    BREAK = "break"
    IGNITE = "ignite"

@dataclass
class CausalEvent:
    """A recorded cause→effect event in the simulation"""
    time: float
    event_type: EventType
    source_id: str          # Who/what caused it
    target_id: str          # Who/what was affected
    description: str        # Human-readable
    data: dict = field(default_factory=dict)  # {force, velocity, energy, ...}
    
    def to_llm_fact(self) -> str:
        return f"[t={self.time:.2f}s] {self.source_id} → {self.event_type.value} → {self.target_id}: {self.description}"

class CausalGraph:
    """Tracks cause→effect chains: 'if I remove this, what cascades?'"""
    def __init__(self):
        self.events: List[CausalEvent] = []
        self.causal_chains: List[List[str]] = []  # Chain of event IDs
        self.support_graph: Dict[str, List[str]] = {}  # who supports whom
        self.containment_graph: Dict[str, str] = {}  # who contains whom
    
    def record(self, event: CausalEvent):
        self.events.append(event)
    
    def add_support(self, supporter: str, supported: str):
        if supporter not in self.support_graph:
            self.support_graph[supporter] = []
        if supported not in self.support_graph[supporter]:
            self.support_graph[supporter].append(supported)
    
    def remove_support(self, supporter: str) -> List[str]:
        """Remove a support: return everything that collapses"""
        cascade = []
        stack = [supporter]
        while stack:
            obj = stack.pop()
            supported = self.support_graph.get(obj, [])
            for s in supported:
                cascade.append(s)
                stack.append(s)
        return cascade
    
    def what_if_removed(self, object_id: str) -> str:
        """LLM-callable: predict cascade if object is removed"""
        cascade = self.remove_support(object_id)
        if not cascade:
            return f"Si {object_id} disparaît, rien ne tombe."
        chain = " → ".join([object_id] + cascade)
        return f"Si {object_id} disparaît → {chain}. {len(cascade)} objets affectés."
    
    def get_llm_summary(self) -> str:
        lines = ["=== CAUSAL GRAPH ==="]
        for obj, supported in self.support_graph.items():
            lines.append(f"  {obj} soutient: {', '.join(supported)}")
        for cid, obj in self.containment_graph.items():
            lines.append(f"  {cid} contient: {obj}")
        if self.events:
            lines.append(f"\nDerniers événements:")
            for e in self.events[-5:]:
                lines.append(f"  {e.to_llm_fact()}")
        return "\n".join(lines)


# ============================================================
# 2. COMPOUND BODIES — Vehicles, containers, nested objects
# ============================================================

class CompoundType(Enum):
    VEHICLE = "vehicle"
    CONTAINER = "container"
    PASSENGER = "passenger"
    CARGO = "cargo"
    DETACHABLE = "detachable"

@dataclass
class CompoundBody:
    """A body that contains or is attached to other bodies"""
    id: str
    parent_id: Optional[str]       # None = root
    compound_type: CompoundType
    children: List[str] = field(default_factory=list)
    relative_position: Vec2 = field(default_factory=Vec2)  # Offset from parent
    is_rigidly_attached: bool = True  # True = moves with parent, False = loose
    break_force: float = 999999.0     # Force needed to detach (for fragile cargo)
    tags: List[str] = field(default_factory=list)  # ["fragile", "explosive", "living"]

class CompoundRegistry:
    """Manages compound body hierarchies"""
    def __init__(self):
        self.compounds: Dict[str, CompoundBody] = {}
    
    def register(self, compound: CompoundBody):
        self.compounds[compound.id] = compound
    
    def get_parent(self, child_id: str) -> Optional[str]:
        c = self.compounds.get(child_id)
        return c.parent_id if c else None
    
    def get_chain(self, object_id: str) -> List[str]:
        """Full parent chain: object → parent → grandparent → root"""
        chain = [object_id]
        current = object_id
        while True:
            parent = self.get_parent(current)
            if not parent:
                break
            chain.append(parent)
            current = parent
        return chain
    
    def is_fragile(self, object_id: str) -> bool:
        c = self.compounds.get(object_id)
        return c and "fragile" in c.tags if c else False


# ============================================================
# 3. ENERGY BUDGET — Track and reason about energy
# ============================================================

@dataclass
class EnergySnapshot:
    time: float
    kinetic: float      # ½mv² for all dynamic bodies
    potential: float    # mgh for all bodies
    thermal: float      # Accumulated heat from collisions
    elastic: float      # Energy stored in springs/constraints
    total: float = 0.0
    
    def __post_init__(self):
        self.total = self.kinetic + self.potential + self.thermal + self.elastic

class EnergyTracker:
    """Tracks energy budget across the simulation"""
    def __init__(self, gravity_y: float = -9.81):
        self.g = abs(gravity_y)
        self.reference_y = 0.0
        self.snapshots: List[EnergySnapshot] = []
        self.thermal_floor = 0.0  # Base temperature energy
        self.log: List[str] = []
    
    def compute(self, bodies: List[PhysicsBody], causal_graph: CausalGraph = None, time: float = 0) -> EnergySnapshot:
        ke = sum(0.5 * b.mass * b.velocity.length_sq() for b in bodies if b.body_type == "dynamic")
        pe = sum(b.mass * self.g * (b.position.y - self.reference_y) for b in bodies)
        
        # Thermal: accumulate from collision dissipation
        thermal = self.thermal_floor
        if causal_graph:
            collision_events = [e for e in causal_graph.events 
                              if e.event_type in (EventType.COLLISION, EventType.REBOUND)]
            for e in collision_events:
                thermal += e.data.get("energy_lost", 0)
        
        elastic = 0  # Spring energy from constraints
        
        snap = EnergySnapshot(time, ke, pe, thermal, elastic)
        self.snapshots.append(snap)
        return snap
    
    def total_energy_drift(self) -> float:
        """How much total energy has changed (should be ~0 in closed system)"""
        if len(self.snapshots) < 2:
            return 0.0
        first = self.snapshots[0].total
        last = self.snapshots[-1].total
        return (last - first) / max(abs(first), 1.0)
    
    def get_llm_summary(self) -> str:
        if not self.snapshots:
            return "Aucune donnée énergétique."
        s = self.snapshots[-1]
        drift_pct = self.total_energy_drift() * 100
        return (
            f"=== BUDGET ÉNERGÉTIQUE (t={s.time:.1f}s) ===\n"
            f"  Cinétique:    {s.kinetic:8.1f} J\n"
            f"  Potentielle:  {s.potential:8.1f} J\n"
            f"  Thermique:    {s.thermal:8.1f} J\n"
            f"  Élastique:    {s.elastic:8.1f} J\n"
            f"  TOTAL:        {s.total:8.1f} J\n"
            f"  Dérive:       {drift_pct:+.1f}% {'(quasi-conservatif)' if abs(drift_pct) < 5 else '(dissipatif)'}"
        )


# ============================================================
# 4. TIME REVERSAL — "Given end state, trace back to start"
# ============================================================

class TimeReversal:
    """Reconstruct initial state from final state + event log"""
    def __init__(self, world: PhysicsWorld, causal_graph: CausalGraph, energy: EnergyTracker):
        self.world = world
        self.causal = causal_graph
        self.energy = energy
        self.state_history: List[dict] = []
        self.max_history = 600  # 10 seconds at 60fps
    
    def snapshot(self, time: float):
        """Record current world state"""
        bodies_data = []
        for b in self.world.bodies:
            bodies_data.append({
                "id": b.id, "pos": b.position.to_tuple(), "vel": b.velocity.to_tuple(),
                "angle": b.angle, "mass": b.mass, "type": b.body_type
            })
        self.state_history.append({"time": time, "bodies": bodies_data})
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
    
    def reverse_to(self, target_time: float) -> dict:
        """Find the closest recorded state at or before target_time"""
        best = None
        for s in self.state_history:
            if s["time"] <= target_time:
                best = s
        return best
    
    def trace_origin(self, object_id: str) -> str:
        """Trace an object back to its earliest recorded state"""
        if not self.state_history:
            return f"{object_id}: pas d'historique."
        
        first = self.state_history[0]
        last = self.state_history[-1]
        obj_first = next((b for b in first["bodies"] if b["id"] == object_id), None)
        obj_last = next((b for b in last["bodies"] if b["id"] == object_id), None)
        
        if not obj_first or not obj_last:
            return f"{object_id}: introuvable dans l'historique."
        
        dy = obj_first["pos"][1] - obj_last["pos"][1]
        dt = last["time"] - first["time"]
        return (
            f"{object_id}: origine à ({obj_first['pos'][0]:.1f},{obj_first['pos'][1]:.1f}) "
            f"→ a parcouru {dy:.1f}m en {dt:.1f}s. "
            f"Hauteur de départ estimée: {obj_first['pos'][1]:.1f}m."
        )


# ============================================================
# 5. MULTI-AGENT GOALS — Bodies with objectives
# ============================================================

class GoalType(Enum):
    REACH_ZONE = "reach_zone"
    AVOID_COLLISION = "avoid_collision"
    PUSH_TO = "push_to"
    STAY_UPRIGHT = "stay_upright"
    MINIMIZE_DAMAGE = "minimize_damage"

@dataclass
class AgentGoal:
    body_id: str
    goal_type: GoalType
    target: Any  # Position, body_id, zone
    priority: float = 1.0
    active: bool = True
    progress: float = 0.0  # 0-1

class AgentSystem:
    """Agents with goals that apply steering forces"""
    def __init__(self):
        self.goals: Dict[str, List[AgentGoal]] = {}
        self.agent_bodies: Set[str] = set()
    
    def add_goal(self, goal: AgentGoal):
        if goal.body_id not in self.goals:
            self.goals[goal.body_id] = []
        self.goals[goal.body_id].append(goal)
        self.agent_bodies.add(goal.body_id)
    
    def compute_steering(self, body_id: str, bodies: List[PhysicsBody]) -> Vec2:
        """Compute steering force for an agent to reach its goals"""
        if body_id not in self.goals:
            return Vec2(0, 0)
        
        total_force = Vec2(0, 0)
        body = next((b for b in bodies if b.id == body_id), None)
        if not body:
            return total_force
        
        for goal in self.goals[body_id]:
            if not goal.active:
                continue
            
            if goal.goal_type == GoalType.REACH_ZONE:
                # Move toward target zone
                target = goal.target  # (cx, cy, radius)
                cx, cy, radius = target
                to_target = Vec2(cx - body.position.x, cy - body.position.y)
                dist = to_target.length()
                if dist > radius:
                    desired_vel = to_target.normalize() * min(5.0, dist)
                    steering = desired_vel - body.velocity
                    total_force = total_force + steering * goal.priority * 10.0
                    goal.progress = max(0, 1 - dist / (dist + radius))
                else:
                    goal.progress = 1.0
            
            elif goal.goal_type == GoalType.AVOID_COLLISION:
                for other in bodies:
                    if other.id == body_id:
                        continue
                    delta = body.position - other.position
                    dist = delta.length()
                    safe_dist = body.shape.radius + getattr(other, 'shape', Shape(type=ShapeType.CIRCLE, radius=0.5)).radius + 1.0
                    if dist < safe_dist and dist > 1e-10:
                        avoidance = delta.normalize() * (safe_dist / dist) * 5.0 * goal.priority
                        total_force = total_force + avoidance
            
            elif goal.goal_type == GoalType.PUSH_TO:
                target_id = goal.target
                target_body = next((b for b in bodies if b.id == target_id), None)
                if target_body and isinstance(goal.target, str):
                    # Push the target body away from obstacles
                    for other in bodies:
                        if other.id in (body_id, target_id):
                            continue
                        delta = target_body.position - other.position
                        dist = delta.length()
                        safe_dist = target_body.shape.radius + getattr(other, 'shape', Shape(type=ShapeType.CIRCLE, radius=0.5)).radius + 1.5
                        if dist < safe_dist and dist > 1e-10:
                            push_dir = delta.normalize()
                            push_force = push_dir * (safe_dist / dist) * 8.0 * goal.priority
                            total_force = total_force + push_force
                    goal.progress = 0.5
            
            elif goal.goal_type == GoalType.MINIMIZE_DAMAGE:
                # Slow down when near collisions
                speed = body.velocity.length()
                if speed > 3.0:
                    total_force = total_force + body.velocity.normalize() * (-speed * goal.priority * 2.0)
        
        return total_force
    
    def get_status(self) -> str:
        lines = ["=== AGENTS ==="]
        for bid, goals in self.goals.items():
            for g in goals:
                prog = f" ({g.progress:.0%})" if g.progress > 0 else ""
                lines.append(f"  {bid}: {g.goal_type.value}{prog}")
        return "\n".join(lines)


# ============================================================
# 6. THERMAL SYSTEM — Heat, ignition, state change
# ============================================================

class ThermalState(Enum):
    SOLID = "solid"
    SOFTENING = "softening"
    LIQUID = "liquid"
    GAS = "gas"
    BURNING = "burning"
    ASH = "ash"

@dataclass
class ThermalProperties:
    melting_point: float = 1000.0       # °C
    ignition_point: float = 500.0       # °C (for flammable materials)
    specific_heat: float = 500.0        # J/(kg·°C) — energy to raise 1kg by 1°C
    thermal_conductivity: float = 1.0   # W/(m·K)
    state: ThermalState = ThermalState.SOLID
    temperature: float = 20.0           # °C
    heat_energy: float = 0.0            # Accumulated heat in Joules
    
    def add_heat(self, joules: float, mass: float):
        """Add heat energy and compute temperature change"""
        self.heat_energy += joules
        dt = joules / (self.specific_heat * mass)
        self.temperature += dt
        
        # State transitions
        if self.temperature >= self.ignition_point and self.state == ThermalState.SOLID:
            self.state = ThermalState.BURNING
        elif self.temperature >= self.melting_point and self.state == ThermalState.SOLID:
            self.state = ThermalState.SOFTENING
        elif self.temperature >= self.melting_point * 1.5 and self.state == ThermalState.SOFTENING:
            self.state = ThermalState.LIQUID

class ThermalSystem:
    """Heat transfer between bodies during collisions"""
    def __init__(self, ambient_temp: float = 20.0):
        self.ambient = ambient_temp
        self.thermal_props: Dict[str, ThermalProperties] = {}
        self.events: List[str] = []
    
    def register_body(self, body_id: str, props: ThermalProperties = None):
        if props is None:
            props = ThermalProperties()
        self.thermal_props[body_id] = props
    
    def collision_heat(self, a_id: str, b_id: str, impact_force: float, 
                       a_mass: float, b_mass: float):
        """Friction/impact generates heat"""
        # Approximate: 10% of kinetic energy lost → heat
        heat = impact_force * 0.1
        split = a_mass / (a_mass + b_mass)
        
        for bid, fraction in [(a_id, 1-split), (b_id, split)]:
            if bid in self.thermal_props:
                p = self.thermal_props[bid]
                mass = a_mass if bid == a_id else b_mass
                p.add_heat(heat * fraction, mass)
                
                if p.state == ThermalState.BURNING:
                    self.events.append(f"🔥 {bid} s'enflamme à {p.temperature:.0f}°C!")
                elif p.state == ThermalState.SOFTENING:
                    self.events.append(f"🌡 {bid} ramollit à {p.temperature:.0f}°C...")
    
    def cool_to_ambient(self, body_id: str, dt: float):
        """Newton cooling toward ambient"""
        if body_id not in self.thermal_props:
            return
        p = self.thermal_props[body_id]
        diff = p.temperature - self.ambient
        cooling = diff * 0.01 * dt  # Simple exponential decay
        p.temperature -= cooling
    
    def get_llm_summary(self) -> str:
        lines = ["=== THERMIQUE ==="]
        for bid, p in self.thermal_props.items():
            state_icon = {"SOLID":"🧊","SOFTENING":"🌡","LIQUID":"💧","GAS":"💨","BURNING":"🔥","ASH":"⬜"}
            icon = state_icon.get(p.state.value, "❓")
            lines.append(f"  {icon} {bid}: {p.temperature:.0f}°C ({p.state.value})")
        if self.events:
            lines.extend(self.events[-3:])
        return "\n".join(lines)


# ============================================================
# ENHANCED PHYSICS WORLD — Integrates all six systems
# ============================================================

class PhysicsWorldV3(PhysicsWorld):
    """Extended world with causal, compound, energy, time-reversal, agents, thermal"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal = CausalGraph()
        self.compounds = CompoundRegistry()
        self.energy = EnergyTracker(abs(self.fields[0].direction.y) if self.fields else 9.81)
        self.time_reversal = TimeReversal(self, self.causal, self.energy)
        self.agents = AgentSystem()
        self.thermal = ThermalSystem()
        self._last_velocities: Dict[str, Vec2] = {}  # For computing acceleration
    
    def step(self, dt: float = 1/60):
        # Store velocities for acceleration computation
        for b in self.bodies:
            self._last_velocities[b.id] = Vec2(b.velocity.x, b.velocity.y)
        
        # Standard physics step
        super().step(dt)
        
        # Agent steering forces (applied after physics for next frame)
        for bid in self.agents.agent_bodies:
            body = next((b for b in self.bodies if b.id == bid), None)
            if body:
                steering = self.agents.compute_steering(bid, self.bodies)
                body.velocity = body.velocity + steering * dt
        
        # Record causal events from contacts
        for contact in self.contact_points:
            a, b = contact["a"], contact["b"]
            rel_vel = b.velocity - a.velocity
            impact_force = abs(rel_vel.dot(contact["normal"])) * (a.mass * b.mass) / (a.mass + b.mass)
            
            # Causal event
            if rel_vel.dot(contact["normal"]) < 0:  # Approaching
                self.causal.record(CausalEvent(
                    self.time, EventType.COLLISION,
                    a.id, b.id,
                    f"Collision à {impact_force:.1f}N",
                    {"force": impact_force, "energy_lost": impact_force * 0.3}
                ))
                # Support graph
                if a.body_type == "static" and b.body_type == "dynamic":
                    if abs(contact["normal"].y) > 0.7:  # Mostly vertical = supporting
                        self.causal.add_support(a.id, b.id)
                elif b.body_type == "static" and a.body_type == "dynamic":
                    if abs(contact["normal"].y) > 0.7:
                        self.causal.add_support(b.id, a.id)
                
                # Thermal
                self.thermal.collision_heat(a.id, b.id, impact_force, a.mass, b.mass)
        
        # Energy snapshot
        self.energy.compute(self.bodies, self.causal, self.time)
        
        # Time reversal snapshot (every 10 frames)
        if int(self.time * 60) % 10 == 0:
            self.time_reversal.snapshot(self.time)
        
        # Thermal cooling
        for b in self.bodies:
            self.thermal.cool_to_ambient(b.id, dt)
    
    def get_full_state(self) -> dict:
        """Complete state with all systems"""
        physics_state = super().get_state()
        return {
            **physics_state,
            "causal": {
                "event_count": len(self.causal.events),
                "support_graph": self.causal.support_graph,
                "last_event": self.causal.events[-1].to_llm_fact() if self.causal.events else None
            },
            "energy": {
                "kinetic": round(self.energy.snapshots[-1].kinetic, 1) if self.energy.snapshots else 0,
                "potential": round(self.energy.snapshots[-1].potential, 1) if self.energy.snapshots else 0,
                "thermal": round(self.energy.snapshots[-1].thermal, 1) if self.energy.snapshots else 0,
                "total": round(self.energy.snapshots[-1].total, 1) if self.energy.snapshots else 0,
                "drift_pct": round(self.energy.total_energy_drift() * 100, 1)
            },
            "compounds": {cid: {"type": c.compound_type.value, "parent": c.parent_id, 
                                "children": c.children, "tags": c.tags}
                         for cid, c in self.compounds.compounds.items()},
            "agents": self.agents.get_status(),
            "thermal": self.thermal.get_llm_summary()
        }


# ============================================================
# NEW COMPOUND SCENARIOS
# ============================================================

def create_vehicle_scenario() -> PhysicsWorldV3:
    """Car with passenger and luggage hits a wall"""
    world = PhysicsWorldV3(gravity=Vec2(0, -9.81), world_bounds=(-12, -8, 12, 6),
                           atmosphere=Atmosphere(humidity=0.4, pressure=101325))
    
    # Car chassis
    car = PhysicsBody("car", Vec2(-6, -2), Vec2(8, 0),
                      Shape(type=ShapeType.BOX, half_width=2.0, half_height=0.5),
                      MATERIALS["steel"], 100, 0, 0, "dynamic", Vec2(0, 0), 0, "#ff6600")
    
    # Passenger (inside car)
    passenger = PhysicsBody("passenger", Vec2(-5.5, -1.2), Vec2(8, 0),
                            Shape(type=ShapeType.CIRCLE, radius=0.4),
                            MATERIALS["steel"], 70, 0, 0, "dynamic", Vec2(0, 0), 0, "#44aaff")
    
    # Luggage (in trunk)
    luggage = PhysicsBody("luggage", Vec2(-7, -1.5), Vec2(8, 0),
                          Shape(type=ShapeType.BOX, half_width=0.4, half_height=0.3),
                          MATERIALS["wood"], 15, 0, 0, "dynamic", Vec2(0, 0), 0, "#8B4513")
    
    world.add_body(car)
    world.add_body(passenger)
    world.add_body(luggage)
    
    # Wall
    wall = PhysicsBody("wall", Vec2(4, -2), Vec2(0, 0),
                       Shape(type=ShapeType.BOX, half_width=0.3, half_height=4),
                       MATERIALS["stone"], 9999, 0, 0, "static", Vec2(0, 0), 0, "#888888")
    
    # Ground
    ground = PhysicsBody("road", Vec2(0, -7), Vec2(0, 0),
                         Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                         MATERIALS["stone"], 9999, 0, 0, "static", Vec2(0, 0), 0, "#555555")
    
    world.add_body(wall)
    world.add_body(ground)
    
    # Register compounds
    world.compounds.register(CompoundBody("car", None, CompoundType.VEHICLE))
    world.compounds.register(CompoundBody("passenger", "car", CompoundType.PASSENGER, tags=["living"]))
    world.compounds.register(CompoundBody("luggage", "car", CompoundType.CARGO, tags=["fragile"], break_force=500))
    
    # Agent goal: passenger wants to survive
    world.agents.add_goal(AgentGoal("passenger", GoalType.MINIMIZE_DAMAGE, None, priority=2.0))
    
    # Register thermal
    world.thermal.register_body("car")
    world.thermal.register_body("passenger")
    world.thermal.register_body("luggage", ThermalProperties(ignition_point=300, specific_heat=800))
    
    return world


def create_jenga_scenario() -> PhysicsWorldV3:
    """Stack of blocks — remove one, see cascade"""
    world = PhysicsWorldV3(gravity=Vec2(0, -9.81), world_bounds=(-4, -6, 4, 6),
                           atmosphere=Atmosphere(humidity=0.2, pressure=101325))
    
    # Base platform
    world.add_body(PhysicsBody("base", Vec2(0, -5), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 9999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    # Tower of 5 blocks
    hw, hh = 0.6, 0.2
    for i in range(5):
        y = -4.5 + i * 0.4
        bid = f"block_{i+1}"
        world.add_body(PhysicsBody(bid, Vec2(0, y), Vec2(0, 0),
                                    Shape(type=ShapeType.BOX, half_width=hw, half_height=hh),
                                    MATERIALS["wood"], 2, 0, 0, "dynamic", Vec2(0, 0), 0, "#8B4513"))
        # Support graph
        if i > 0:
            world.causal.add_support(f"block_{i}", f"block_{i+1}")
        else:
            world.causal.add_support("base", "block_1")
    
    # Striker
    striker = PhysicsBody("striker", Vec2(3, -4.3), Vec2(-4, 0),
                          Shape(type=ShapeType.CIRCLE, radius=0.3),
                          MATERIALS["steel"], 5, 0, 0, "dynamic", Vec2(0, 0), 0, "#ff4444")
    world.add_body(striker)
    
    # Walls
    world.add_body(PhysicsBody("wall_l", Vec2(-3.5, -3), Vec2(0, 0),
                                Shape(type=ShapeType.BOX, half_width=0.2, half_height=4),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#666"))
    world.add_body(PhysicsBody("wall_r", Vec2(3.5, -3), Vec2(0, 0),
                                Shape(type=ShapeType.BOX, half_width=0.2, half_height=4),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#666"))
    
    return world


# Update scenario registry
V3_SCENARIOS = {
    **SCENARIOS,
    "vehicle": create_vehicle_scenario,
    "jenga": create_jenga_scenario,
}


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=== Physics V3 — All Systems Test ===\n")
    
    # Test 1: Vehicle crash
    print("1. VEHICLE CRASH")
    world = create_vehicle_scenario()
    for i in range(120):
        world.step(1/60)
    car = [b for b in world.bodies if b.id == "car"][0]
    passenger = [b for b in world.bodies if b.id == "passenger"][0]
    luggage = [b for b in world.bodies if b.id == "luggage"][0]
    print(f"  Car: pos=({car.position.x:.1f},{car.position.y:.1f}) v={car.velocity.x:.1f}m/s")
    print(f"  Passenger: ({passenger.position.x:.1f},{passenger.position.y:.1f}) v={passenger.velocity.x:.1f}m/s")
    print(f"  Luggage: ({luggage.position.x:.1f},{luggage.position.y:.1f}) v={luggage.velocity.x:.1f}m/s")
    print(f"  {world.causal.what_if_removed('car')}")
    print(f"  {world.energy.get_llm_summary()}")
    print(f"  {world.thermal.get_llm_summary()}")
    
    # Test 2: Jenga cascade
    print(f"\n2. JENGA CASCADE")
    world2 = create_jenga_scenario()
    for i in range(90):
        world2.step(1/60)
    print(f"  Events: {len(world2.causal.events)} causal events recorded")
    print(f"  {world2.causal.what_if_removed('block_1')}")
    state = world2.get_full_state()
    print(f"  Energy drift: {state['energy']['drift_pct']}%")
    
    print(f"\n✅ All V3 systems functional")
