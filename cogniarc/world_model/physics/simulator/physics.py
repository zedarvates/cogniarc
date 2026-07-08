"""
World Model Physics Engine v2 — Complete game-style physics simulator.
Supports: boxes, circles, ramps, friction, materials, magnetic fields, fluid, constraints.
Designed for approximate reasoning by small LLMs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from enum import Enum
import json

# === Math ===

@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0
    def __add__(self, o): return Vec2(self.x + o.x, self.y + o.y)
    def __sub__(self, o): return Vec2(self.x - o.x, self.y - o.y)
    def __mul__(self, s): return Vec2(self.x * s, self.y * s)
    def __truediv__(self, s): return Vec2(self.x / s, self.y / s)
    def __neg__(self): return Vec2(-self.x, -self.y)
    def dot(self, o): return self.x * o.x + self.y * o.y
    def cross(o): return self.x * o.y - self.y * o.x  # cross(self, o)
    def length(self): return np.sqrt(self.x**2 + self.y**2)
    def length_sq(self): return self.x**2 + self.y**2
    def normalize(self):
        l = self.length()
        return self / l if l > 1e-12 else Vec2(0, 0)
    def rotate(self, angle: float):
        c, s = np.cos(angle), np.sin(angle)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)
    def perpendicular(self): return Vec2(-self.y, self.x)
    def to_tuple(self): return (self.x, self.y)

# === Shapes ===

class ShapeType(Enum):
    CIRCLE = "circle"
    BOX = "box"
    PLANE = "plane"  # Infinite half-plane for ramps/floors

@dataclass
class Shape:
    type: ShapeType
    radius: float = 0.5       # For circles
    half_width: float = 0.5   # For boxes
    half_height: float = 0.5  # For boxes
    normal: Vec2 = field(default_factory=lambda: Vec2(0, 1))  # For planes
    offset: float = 0.0       # Plane distance from origin along normal
    
    def get_vertices(self, pos: Vec2, angle: float) -> List[Vec2]:
        """Get world-space vertices"""
        if self.type == ShapeType.BOX:
            hw, hh = self.half_width, self.half_height
            local = [Vec2(-hw, -hh), Vec2(hw, -hh), Vec2(hw, hh), Vec2(-hw, hh)]
            return [v.rotate(angle) + pos for v in local]
        elif self.type == ShapeType.CIRCLE:
            return [pos]
        elif self.type == ShapeType.PLANE:
            return [pos]
        return [pos]
    
    def get_normals(self, angle: float) -> List[Vec2]:
        """Get axis normals for SAT collision"""
        if self.type == ShapeType.BOX:
            return [Vec2(1, 0).rotate(angle), Vec2(0, 1).rotate(angle)]
        elif self.type == ShapeType.CIRCLE:
            return []
        elif self.type == ShapeType.PLANE:
            return [self.normal]
        return []

# === Materials ===

class Material:
    """Physical material properties"""
    def __init__(self, name: str, restitution: float = 0.3, 
                 static_friction: float = 0.5, dynamic_friction: float = 0.3,
                 density: float = 1.0, magnetic_susceptibility: float = 0.0):
        self.name = name
        self.restitution = restitution         # bounce coefficient (0-1)
        self.static_friction = static_friction  # μ_s
        self.dynamic_friction = dynamic_friction  # μ_k
        self.density = density                  # relative density
        self.magnetic_susceptibility = magnetic_susceptibility  # 0=non-magnetic

# Material presets
MATERIALS = {
    "steel": Material("Steel", restitution=0.6, static_friction=0.3, dynamic_friction=0.25, density=7.8, magnetic_susceptibility=1.0),
    "wood": Material("Wood", restitution=0.2, static_friction=0.6, dynamic_friction=0.4, density=0.7, magnetic_susceptibility=0.0),
    "rubber": Material("Rubber", restitution=0.8, static_friction=1.0, dynamic_friction=0.8, density=1.2, magnetic_susceptibility=0.0),
    "stone": Material("Stone", restitution=0.1, static_friction=0.8, dynamic_friction=0.5, density=2.5, magnetic_susceptibility=0.0),
    "ice": Material("Ice", restitution=0.5, static_friction=0.05, dynamic_friction=0.03, density=0.9, magnetic_susceptibility=0.0),
    "default": Material("Default", restitution=0.3, static_friction=0.5, dynamic_friction=0.3, density=1.0),
}

# === Liquids ===

class LiquidType:
    """Liquid with viscosity, density, and interaction properties"""
    def __init__(self, name: str, density: float, viscosity: float,
                 surface_tension: float = 0.0, temperature: float = 20.0,
                 color: str = "#4488ff", flammable: bool = False):
        self.name = name
        self.density = density          # kg/m³ (water = 1000)
        self.viscosity = viscosity      # Pa·s (water = 0.001, oil = 0.1-1.0)
        self.surface_tension = surface_tension  # N/m
        self.temperature = temperature  # °C
        self.color = color
        self.flammable = flammable
    
    @property
    def kinematic_viscosity(self) -> float:
        return self.viscosity / max(self.density, 1.0)
    
    def drag_coefficient(self, object_radius: float, velocity: float) -> float:
        """Reynolds-number-dependent drag coefficient"""
        if velocity < 1e-10:
            return 0.0
        re = abs(velocity) * object_radius * 2 / max(self.kinematic_viscosity, 1e-12)
        if re < 1:
            return 24.0 / max(re, 0.01)  # Stokes flow
        elif re < 1000:
            return 0.5 + 24.0 / max(re, 0.01)  # Transition
        else:
            return 0.47  # Turbulent

LIQUIDS = {
    "water": LiquidType("Water", density=1000, viscosity=0.001, surface_tension=0.072, color="#3388ff"),
    "oil": LiquidType("Oil", density=900, viscosity=0.8, surface_tension=0.025, color="#ffaa00", flammable=True),
    "honey": LiquidType("Honey", density=1400, viscosity=10.0, surface_tension=0.1, color="#cc8800"),
    "mercury": LiquidType("Mercury", density=13500, viscosity=0.0015, surface_tension=0.47, color="#aaaaaa"),
    "air": LiquidType("Air", density=1.225, viscosity=0.000018, surface_tension=0.0, color="#ffffff"),
    "lava": LiquidType("Lava", density=3100, viscosity=100.0, surface_tension=0.3, color="#ff4400", flammable=False),
}

# === Atmospheric Environment ===

class Atmosphere:
    """Global atmospheric conditions affecting all physics"""
    def __init__(self, humidity: float = 0.5, pressure: float = 101325.0,
                 temperature: float = 20.0, wind: Vec2 = None,
                 depression_zone: Tuple[float, float, float] = None):
        self.humidity = humidity              # 0-1 (0=dry, 1=saturated)
        self.pressure = pressure              # Pa (101325 = 1 atm)
        self.temperature = temperature        # °C
        self.wind = wind or Vec2(0, 0)        # m/s wind vector
        # Depression zone: (center_x, center_y, radius) — low pressure area
        self.depression = depression_zone      # creates inward flow
        self.air_density = self._compute_air_density()
    
    def _compute_air_density(self) -> float:
        """Air density from temperature, pressure, humidity (ideal gas law approx)"""
        R_specific = 287.058  # J/(kg·K) for dry air
        T_kelvin = self.temperature + 273.15
        # Humidity correction
        R_humid = R_specific * (1 + 0.61 * self.humidity)
        return self.pressure / (R_humid * T_kelvin)
    
    def friction_modifier(self) -> float:
        """Humidity affects surface friction (wet = less friction)"""
        return 1.0 - self.humidity * 0.4  # 100% humid → 60% friction
    
    def drag_multiplier(self) -> float:
        """Air density affects aerodynamic drag"""
        return self.air_density / 1.225  # Relative to standard air
    
    def depression_force_at(self, pos: Vec2) -> Vec2:
        """Wind force from depression zone (low pressure sucks air in)"""
        if not self.depression:
            return Vec2(0, 0)
        cx, cy, radius = self.depression
        center = Vec2(cx, cy)
        delta = center - pos
        dist = max(delta.length(), 0.01)
        if dist < radius:
            # Inside depression: upward force (rising air)
            strength = (1.0 - dist / radius) * (self.pressure / 101325.0)
            return Vec2(0, strength * 5.0) + delta.normalize() * strength * 2.0
        elif dist < radius * 3:
            # Near depression: inward flow toward low pressure
            strength = (1.0 - (dist - radius) / (2 * radius)) * (self.pressure / 101325.0)
            return delta.normalize() * strength * 3.0
        return Vec2(0, 0)

# === Physics World with Environment ===

@dataclass
class ForceField:
    """Global force field (gravity, wind, magnetic, etc.)"""
    type: str  # "gravity", "wind", "magnetic", "drag", "buoyancy"
    direction: Vec2 = field(default_factory=lambda: Vec2(0, -9.81))
    strength: float = 1.0
    center: Vec2 = field(default_factory=Vec2)  # For point-source fields
    falloff: float = 0.0  # 0 = uniform, >0 = inverse-square
    
    def force_at(self, pos: Vec2, body_mass: float, body_susceptibility: float = 0) -> Vec2:
        """Compute force on a body at position"""
        if self.type == "gravity":
            return self.direction * body_mass * self.strength
        elif self.type == "magnetic":
            if body_susceptibility <= 0:
                return Vec2(0, 0)
            delta = self.center - pos
            dist = max(delta.length(), 0.01)
            if self.falloff > 0:
                strength = self.strength / (1 + self.falloff * dist**2)
            else:
                strength = self.strength
            return delta.normalize() * strength * body_susceptibility * body_mass
        elif self.type == "wind":
            return self.direction * self.strength
        elif self.type == "drag":
            return Vec2(0, 0)  # Drag handled per-body
        return Vec2(0, 0)


# === Physics Body ===

@dataclass
class PhysicsBody:
    id: str
    position: Vec2
    velocity: Vec2
    shape: Shape
    material: Material = field(default_factory=lambda: MATERIALS["default"])
    mass: float = 1.0
    angle: float = 0.0          # rotation in radians
    angular_velocity: float = 0.0
    body_type: str = "dynamic"   # dynamic, static, kinematic
    forces: Vec2 = field(default_factory=Vec2)
    torque: float = 0.0
    color: str = "#4488ff"
    charge: float = 0.0          # electric charge (for EM fields)
    constraints: List[str] = field(default_factory=list)
    
    @property
    def inertia(self) -> float:
        """Moment of inertia (approximate for box)"""
        if self.shape.type == ShapeType.BOX:
            return (1/12) * self.mass * (self.shape.half_width**2 + self.shape.half_height**2)
        elif self.shape.type == ShapeType.CIRCLE:
            return 0.5 * self.mass * self.shape.radius**2
        return 1.0
    
    def to_state_vector(self) -> list:
        return [self.position.x, self.position.y,
                self.velocity.x, self.velocity.y,
                self.angle, self.angular_velocity]


# === Physics World ===

class PhysicsWorld:
    """Complete physics simulation with environment, multiple liquids, and atmosphere."""
    
    def __init__(self, gravity: Vec2 = Vec2(0, -9.81),
                 world_bounds: Tuple[float, float, float, float] = (-10, -10, 10, 10),
                 atmosphere: Atmosphere = None):
        self.bodies: List[PhysicsBody] = []
        self.fields: List[ForceField] = [
            ForceField("gravity", direction=gravity, strength=1.0)
        ]
        self.bounds_min = Vec2(world_bounds[0], world_bounds[1])
        self.bounds_max = Vec2(world_bounds[2], world_bounds[3])
        self.time = 0.0
        self.substeps = 8
        self.contact_points = []
        self.atmosphere = atmosphere or Atmosphere()
        self.liquids: Dict[str, dict] = {}  # liquid_zone_name → {liquid, y_min, y_max, x_min, x_max}
    
    def add_body(self, body: PhysicsBody):
        self.bodies.append(body)
    
    def add_field(self, field: ForceField):
        self.fields.append(field)
    
    # === Collision Detection (SAT for boxes, GJK-lite for shapes) ===
    
    def _find_contact_circle_circle(self, a: PhysicsBody, b: PhysicsBody) -> Optional[dict]:
        delta = b.position - a.position
        dist = delta.length()
        min_dist = a.shape.radius + b.shape.radius
        if dist < min_dist and dist > 1e-10:
            normal = delta.normalize()
            penetration = min_dist - dist
            return {"normal": normal, "penetration": penetration, 
                    "contact_point": a.position + normal * a.shape.radius,
                    "a": a, "b": b}
        return None
    
    def _find_contact_circle_box(self, circle: PhysicsBody, box: PhysicsBody) -> Optional[dict]:
        # Transform circle center to box local space
        local_center = (circle.position - box.position).rotate(-box.angle)
        hw, hh = box.shape.half_width, box.shape.half_height
        
        # Clamp to box extents
        closest_x = max(-hw, min(local_center.x, hw))
        closest_y = max(-hh, min(local_center.y, hh))
        closest_local = Vec2(closest_x, closest_y)
        
        delta_local = local_center - closest_local
        dist_sq = delta_local.length_sq()
        radius = circle.shape.radius
        
        if dist_sq < radius**2:
            if dist_sq < 1e-12:
                # Circle center is inside box
                # Push out along shortest axis
                dx_out = hw - abs(local_center.x)
                dy_out = hh - abs(local_center.y)
                if dx_out < dy_out:
                    normal_local = Vec2(1 if local_center.x > 0 else -1, 0)
                    penetration = hw - abs(local_center.x) + radius
                else:
                    normal_local = Vec2(0, 1 if local_center.y > 0 else -1)
                    penetration = hh - abs(local_center.y) + radius
            else:
                dist = np.sqrt(dist_sq)
                normal_local = delta_local.normalize()
                penetration = radius - dist
            
            normal_world = normal_local.rotate(box.angle)
            return {"normal": normal_world, "penetration": penetration,
                    "contact_point": circle.position - normal_world * radius,
                    "a": circle, "b": box}
        return None
    
    def _find_contact_box_box(self, a: PhysicsBody, b: PhysicsBody) -> Optional[dict]:
        """SAT (Separating Axis Theorem) for box-box collision"""
        # Get all axes to test
        axes = []
        axes.extend(a.shape.get_normals(a.angle))
        axes.extend(b.shape.get_normals(b.angle))
        
        verts_a = a.shape.get_vertices(a.position, a.angle)
        verts_b = b.shape.get_vertices(b.position, b.angle)
        
        best_normal = Vec2(0, 0)
        best_penetration = float('inf')
        
        for axis in axes:
            if axis.length_sq() < 1e-12:
                continue
            axis = axis.normalize()
            
            # Project both shapes onto axis
            min_a = min(v.dot(axis) for v in verts_a)
            max_a = max(v.dot(axis) for v in verts_a)
            min_b = min(v.dot(axis) for v in verts_b)
            max_b = max(v.dot(axis) for v in verts_b)
            
            if max_a < min_b or max_b < min_a:
                return None  # Separating axis found, no collision
            
            # Compute penetration
            penetration = min(max_a - min_b, max_b - min_a)
            if penetration < best_penetration:
                best_penetration = penetration
                # Normal points from A to B
                mid_a = (min_a + max_a) / 2
                mid_b = (min_b + max_b) / 2
                if mid_a < mid_b:
                    best_normal = axis
                else:
                    best_normal = -axis
        
        if best_penetration < float('inf'):
            return {"normal": best_normal, "penetration": best_penetration,
                    "contact_point": (a.position + b.position) * 0.5,
                    "a": a, "b": b}
        return None
    
    def _find_contact_circle_plane(self, circle: PhysicsBody, plane: PhysicsBody) -> Optional[dict]:
        dist = (circle.position - plane.position).dot(plane.shape.normal)
        penetration = circle.shape.radius - (dist - plane.shape.offset)
        if penetration > 0:
            return {"normal": -plane.shape.normal, "penetration": penetration,
                    "contact_point": circle.position - plane.shape.normal * circle.shape.radius,
                    "a": circle, "b": plane}
        return None
    
    def _find_contact(self, a: PhysicsBody, b: PhysicsBody) -> Optional[dict]:
        """Dispatch to correct collision handler"""
        if a.body_type == "static" and b.body_type == "static":
            return None
        
        types = (a.shape.type, b.shape.type)
        
        if types == (ShapeType.CIRCLE, ShapeType.CIRCLE):
            return self._find_contact_circle_circle(a, b)
        elif ShapeType.CIRCLE in types and ShapeType.BOX in types:
            if a.shape.type == ShapeType.CIRCLE:
                return self._find_contact_circle_box(a, b)
            else:
                result = self._find_contact_circle_box(b, a)
                if result:
                    result["normal"] = -result["normal"]
                    result["a"], result["b"] = a, b
                return result
        elif types == (ShapeType.BOX, ShapeType.BOX):
            return self._find_contact_box_box(a, b)
        elif ShapeType.PLANE in types:
            if a.shape.type == ShapeType.CIRCLE and b.shape.type == ShapeType.PLANE:
                return self._find_contact_circle_plane(a, b)
            elif b.shape.type == ShapeType.CIRCLE and a.shape.type == ShapeType.PLANE:
                result = self._find_contact_circle_plane(b, a)
                if result:
                    result["normal"] = -result["normal"]
                    result["a"], result["b"] = a, b
                return result
        
        return None
    
    def _resolve_contact(self, contact: dict):
        """Resolve contact with friction (Coulomb model)"""
        a, b = contact["a"], contact["b"]
        normal = contact["normal"]
        
        # Combined material
        restitution = (a.material.restitution + b.material.restitution) / 2
        static_friction = (a.material.static_friction + b.material.static_friction) / 2
        dynamic_friction = (a.material.dynamic_friction + b.material.dynamic_friction) / 2
        
        # Separate
        total_mass = (a.mass if a.body_type == "dynamic" else 99999) + (b.mass if b.body_type == "dynamic" else 99999)
        if a.body_type == "dynamic":
            a.position = a.position - normal * (contact["penetration"] * b.mass / total_mass)
        if b.body_type == "dynamic":
            b.position = b.position + normal * (contact["penetration"] * a.mass / total_mass)
        
        # Velocity resolution
        rel_vel = (b.velocity if b.body_type == "dynamic" else Vec2(0, 0)) - \
                  (a.velocity if a.body_type == "dynamic" else Vec2(0, 0))
        vel_normal = rel_vel.dot(normal)
        
        if vel_normal < 0:  # Objects approaching
            j_normal = -(1 + restitution) * vel_normal / total_mass
            
            # Friction impulse
            tangent = (rel_vel - normal * vel_normal)
            tangent_len = tangent.length()
            if tangent_len > 1e-10:
                tangent = tangent / tangent_len
                j_friction = max(-j_normal * dynamic_friction, -tangent_len / total_mass)
            else:
                j_friction = 0.0
            
            impulse = normal * j_normal + tangent * j_friction
            
            if a.body_type == "dynamic":
                a.velocity = a.velocity - impulse * (b.mass / total_mass)
            if b.body_type == "dynamic":
                b.velocity = b.velocity + impulse * (a.mass / total_mass)
    
    def _apply_field_forces(self):
        """Apply gravitational, magnetic, drag + atmospheric effects"""
        atm = self.atmosphere
        for b in self.bodies:
            if b.body_type != "dynamic":
                continue
            
            # Standard fields (gravity, magnetic, wind)
            for f in self.fields:
                force = f.force_at(b.position, b.mass, b.material.magnetic_susceptibility)
                if f.type == "drag":
                    vel = b.velocity.length()
                    if vel > 1e-10:
                        drag_dir = -b.velocity.normalize()
                        area = b.shape.radius * 2 if b.shape.type == ShapeType.CIRCLE else b.shape.half_width * 2
                        # Aerodynamic drag adjusted by atmosphere
                        drag_force = 0.5 * atm.air_density * 0.47 * area * vel * vel * f.strength * atm.drag_multiplier()
                        b.forces = b.forces + drag_dir * drag_force
                else:
                    b.forces = b.forces + force
            
            # Atmospheric depression
            dep_force = atm.depression_force_at(b.position)
            b.forces = b.forces + dep_force * b.mass
            
            # Wind force
            wind_force = atm.wind * b.mass * 0.5
            b.forces = b.forces + wind_force
            
            # Torque from off-center forces
            if b.shape.type == ShapeType.BOX:
                leverage = 0.5 * b.shape.half_width
                b.torque += b.forces.x * leverage * (1 if b.forces.y > 0 else -1)
    
    def add_liquid_zone(self, name: str, liquid: LiquidType, 
                         y_min: float, y_max: float,
                         x_min: float = -100, x_max: float = 100):
        """Add a liquid body (water, oil, etc.) to the world"""
        self.liquids[name] = {
            "liquid": liquid, "y_min": y_min, "y_max": y_max,
            "x_min": x_min, "x_max": x_max
        }
    
    def _apply_fluid(self):
        """Buoyancy + viscosity-driven drag for all liquid zones"""
        atm = self.atmosphere
        for name, zone in self.liquids.items():
            liquid = zone["liquid"]
            for b in self.bodies:
                if b.body_type != "dynamic":
                    continue
                
                # Determine object extent
                if b.shape.type == ShapeType.CIRCLE:
                    bottom = b.position.y - b.shape.radius
                    top = b.position.y + b.shape.radius
                    radius = b.shape.radius
                    volume = (4/3) * np.pi * radius**3
                elif b.shape.type == ShapeType.BOX:
                    verts = b.shape.get_vertices(b.position, b.angle)
                    bottom = min(v.y for v in verts)
                    top = max(v.y for v in verts)
                    radius = max(b.shape.half_width, b.shape.half_height)
                    volume = b.shape.half_width * b.shape.half_height * 4
                else:
                    continue
                
                # Is object in liquid zone?
                in_x = zone["x_min"] <= b.position.x <= zone["x_max"]
                in_y = bottom < zone["y_max"] and top > zone["y_min"]
                if not (in_x and in_y):
                    continue
                
                # Submerged fraction
                obj_height = top - bottom
                if obj_height < 1e-10:
                    continue
                submerged_top = min(top, zone["y_max"])
                submerged_bottom = max(bottom, zone["y_min"])
                submerged_height = max(0, submerged_top - submerged_bottom)
                submerged = min(1.0, submerged_height / obj_height)
                
                if submerged > 0:
                    # Buoyancy (Archimedes)
                    buoyancy = Vec2(0, liquid.density * volume * submerged * 9.81 / 1000.0)
                    b.forces = b.forces + buoyancy
                    
                    # Surface tension (resists entry/exit near surface)
                    if abs(b.position.y - zone["y_max"]) < radius:
                        surface_force = Vec2(0, -liquid.surface_tension * radius * 0.1)
                        b.forces = b.forces + surface_force
                    
                    # Viscous drag (Reynolds-dependent)
                    speed = b.velocity.length()
                    if speed > 1e-10:
                        drag_dir = -b.velocity.normalize()
                        Cd = liquid.drag_coefficient(radius, speed)
                        # Characteristic area
                        area = np.pi * radius**2 if b.shape.type == ShapeType.CIRCLE else b.shape.half_width * 2
                        drag_force = 0.5 * liquid.density * Cd * area * speed * speed / 1000.0
                        b.forces = b.forces + drag_dir * drag_force * submerged
                        # Angular drag from viscosity
                        if b.shape.type == ShapeType.BOX:
                            b.torque -= b.angular_velocity * liquid.viscosity * 10.0 * submerged
    
    def _apply_constraints(self, stiffness: float = 100.0, damping: float = 5.0):
        """Spring constraints between connected bodies"""
        for b in self.bodies:
            for cid in b.constraints:
                other = next((o for o in self.bodies if o.id == cid), None)
                if other:
                    delta = other.position - b.position
                    # Dynamic rest length based on shapes
                    if b.shape.type == ShapeType.CIRCLE and other.shape.type == ShapeType.CIRCLE:
                        rest = b.shape.radius + other.shape.radius
                    else:
                        rest = 1.0
                    cur_len = delta.length()
                    if cur_len > 0.001:
                        direction = delta.normalize()
                        stretch = cur_len - rest
                        b.forces = b.forces + direction * (stiffness * stretch)
                        b.forces = b.forces + (other.velocity - b.velocity) * damping
    
    def _integrate(self, dt: float):
        """Semi-implicit Euler with angular dynamics"""
        for b in self.bodies:
            if b.body_type == "dynamic":
                acc = b.forces / b.mass
                b.velocity = b.velocity + acc * dt
                b.position = b.position + b.velocity * dt
                # Rotation
                if b.shape.type == ShapeType.BOX:
                    alpha = b.torque / max(b.inertia, 0.01)
                    b.angular_velocity += alpha * dt
                    b.angle += b.angular_velocity * dt
                    b.angular_velocity *= 0.99  # Angular damping
                    b.torque = 0.0
            b.forces = Vec2(0, 0)
    
    def step(self, dt: float = 1/60):
        sub_dt = dt / self.substeps
        
        for _ in range(self.substeps):
            # Collect all contacts
            contacts = []
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    c = self._find_contact(self.bodies[i], self.bodies[j])
                    if c:
                        contacts.append(c)
            
            # Solve contacts (multiple iterations for stacking)
            for _ in range(3):
                for c in contacts:
                    self._resolve_contact(c)
            
            # Apply forces
            self._apply_field_forces()
            self._apply_fluid()
            self._apply_constraints()
            
            # Integrate
            self._integrate(sub_dt)
        
        self.time += dt
        self.contact_points = contacts
    
    def get_state(self) -> dict:
        bodies = []
        for b in self.bodies:
            bodies.append({
                "id": b.id,
                "pos": list(b.position.to_tuple()),
                "vel": list(b.velocity.to_tuple()),
                "angle": round(b.angle, 4),
                "ang_vel": round(b.angular_velocity, 4),
                "shape": b.shape.type.value,
                "radius": b.shape.radius,
                "hw": b.shape.half_width if b.shape.type == ShapeType.BOX else None,
                "hh": b.shape.half_height if b.shape.type == ShapeType.BOX else None,
                "mass": b.mass,
                "material": b.material.name,
                "type": b.body_type,
                "color": b.color
            })
        return {
            "time": round(self.time, 3),
            "bodies": bodies,
            "contacts": len(self.contact_points),
            "atmosphere": {
                "humidity": round(self.atmosphere.humidity, 2),
                "pressure_pa": round(self.atmosphere.pressure),
                "temp_c": round(self.atmosphere.temperature, 1),
                "air_density": round(self.atmosphere.air_density, 3),
                "wind": list(self.atmosphere.wind.to_tuple()),
                "depression_active": self.atmosphere.depression is not None
            },
            "liquids": {name: {"type": z["liquid"].name, "viscosity": z["liquid"].viscosity} 
                        for name, z in self.liquids.items()}
        }
    
    def get_state_vector(self) -> np.ndarray:
        vec = []
        for b in sorted(self.bodies, key=lambda b: b.id):
            vec.extend(b.to_state_vector())
        return np.array(vec, dtype=np.float32)
    
    def state_size(self) -> int:
        return sum(6 for _ in self.bodies)


# === Enhanced Scenarios ===

def create_ramp_scenario(ball_material: str = "steel", ramp_material: str = "wood",
                         drop_height: float = 8.0, ramp_angle_deg: float = 45.0,
                         ball_radius: float = 0.3) -> PhysicsWorld:
    """
    Classic physics demo: metal ball falls onto a wooden ramp at 45°.
    The ball rebounds/rolls depending on drop height and materials.
    """
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-5, -10, 5, 10))
    
    angle_rad = np.radians(ramp_angle_deg)
    ramp_len = 6.0
    
    # Ramp as an angled box
    ramp = PhysicsBody(
        id="ramp",
        position=Vec2(0, -2),
        velocity=Vec2(0, 0),
        shape=Shape(type=ShapeType.BOX, half_width=ramp_len/2, half_height=0.15),
        material=MATERIALS.get(ramp_material, MATERIALS["wood"]),
        mass=999, body_type="static", angle=angle_rad,
        color="#8B4513"
    )
    
    # Ball positioned above the ramp
    ball = PhysicsBody(
        id="ball",
        position=Vec2(0, drop_height),
        velocity=Vec2(0, 0),
        shape=Shape(type=ShapeType.CIRCLE, radius=ball_radius),
        material=MATERIALS.get(ball_material, MATERIALS["steel"]),
        mass=2.0, body_type="dynamic",
        color="#C0C0C0"
    )
    
    # Ground plane
    ground = PhysicsBody(
        id="ground",
        position=Vec2(0, -7),
        velocity=Vec2(0, 0),
        shape=Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
        material=MATERIALS["stone"],
        mass=999, body_type="static",
        color="#888888"
    )
    
    world.add_body(ball)
    world.add_body(ramp)
    world.add_body(ground)
    
    # Add walls
    wall_l = PhysicsBody("wall_left", Vec2(-4.8, -3), Vec2(0, 0),
                          Shape(type=ShapeType.BOX, half_width=0.2, half_height=6),
                          MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#666")
    wall_r = PhysicsBody("wall_right", Vec2(4.8, -3), Vec2(0, 0),
                          Shape(type=ShapeType.BOX, half_width=0.2, half_height=6),
                          MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#666")
    world.add_body(wall_l)
    world.add_body(wall_r)
    
    return world


def create_magnetic_scenario() -> PhysicsWorld:
    """Magnetic field attracting/pushing metal balls."""
    world = PhysicsWorld(gravity=Vec2(0, -2.0), world_bounds=(-8, -8, 8, 8))
    
    # Magnet at center
    magnet = PhysicsBody("magnet", Vec2(0, 0), Vec2(0, 0),
                          Shape(type=ShapeType.CIRCLE, radius=0.5),
                          MATERIALS["steel"], 999, 0, 0, "static",
                          Vec2(0, 0), 0, "#ff0000")
    world.add_body(magnet)
    
    # Add magnetic field
    world.add_field(ForceField("magnetic", strength=500.0, center=Vec2(0, 0), falloff=0.1))
    
    # Metal balls around the magnet
    for i in range(8):
        angle = i * np.pi / 4
        dist = np.random.uniform(3, 6)
        ball = PhysicsBody(f"metal_{i}", Vec2(np.cos(angle) * dist, np.sin(angle) * dist),
                           Vec2(np.random.uniform(-2, 2), np.random.uniform(-2, 2)),
                           Shape(type=ShapeType.CIRCLE, radius=0.3),
                           MATERIALS["steel"], 1.0, 0, 0, "dynamic",
                           Vec2(0, 0), 0, "#4488ff")
        world.add_body(ball)
    
    return world


def create_collision_chains() -> PhysicsWorld:
    """Newton's cradle + domino effect with different materials."""
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-10, -5, 10, 8))
    
    # Platform
    world.add_body(PhysicsBody("floor", Vec2(0, -4.5), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    # Line of balls (Newton's cradle)
    for i in range(6):
        ball = PhysicsBody(f"cradle_{i}", Vec2(-4 + i * 0.7, -3.8), Vec2(0, 0),
                           Shape(type=ShapeType.CIRCLE, radius=0.3),
                           MATERIALS["steel"], 1.0, 0, 0, "dynamic",
                           Vec2(0, 0), 0, "#C0C0C0")
        world.add_body(ball)
    
    # Striker ball
    striker = PhysicsBody("striker", Vec2(-5, -3.5), Vec2(5, 2),
                          Shape(type=ShapeType.CIRCLE, radius=0.3),
                          MATERIALS["steel"], 2.0, 0, 0, "dynamic",
                          Vec2(0, 0), 0, "#ff4444")
    world.add_body(striker)
    
    return world


def create_mixed_objects() -> PhysicsWorld:
    """Multiple objects with different shapes, materials, falling together."""
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-8, -8, 8, 6))
    
    objects = [
        ("box_wood", Vec2(-3, 5), ShapeType.BOX, "wood", 2.0, "#8B4513", (1.0, 0.5)),
        ("box_metal", Vec2(0, 5.5), ShapeType.BOX, "steel", 3.0, "#C0C0C0", (0.6, 0.8)),
        ("ball_rubber", Vec2(3, 5), ShapeType.CIRCLE, "rubber", 0.5, "#ff4444", None),
        ("ball_ice", Vec2(-1, 6), ShapeType.CIRCLE, "ice", 0.3, "#aaddff", None),
        ("box_heavy", Vec2(2, 5.2), ShapeType.BOX, "stone", 5.0, "#888888", (1.2, 0.6)),
    ]
    
    for obj_id, pos, stype, mat_name, mass, color, box_dims in objects:
        if stype == ShapeType.CIRCLE:
            shape = Shape(type=ShapeType.CIRCLE, radius=0.4)
        else:
            shape = Shape(type=ShapeType.BOX, half_width=box_dims[0], half_height=box_dims[1])
        
        world.add_body(PhysicsBody(obj_id, pos, Vec2(0, 0), shape,
                                    MATERIALS[mat_name], mass, 0, np.random.uniform(-0.5, 0.5),
                                    "dynamic", Vec2(0, 0), 0, color))
    
    # Ground
    world.add_body(PhysicsBody("ground", Vec2(0, -7.5), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    return world


def create_propulsion_demo() -> PhysicsWorld:
    """Rocket/propulsion: body with continuous force applied."""
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-10, -10, 10, 10))
    
    # Rocket
    rocket = PhysicsBody("rocket", Vec2(0, -8), Vec2(0, 0),
                          Shape(type=ShapeType.BOX, half_width=0.3, half_height=0.8),
                          MATERIALS["steel"], 1.0, 0, 0, "dynamic",
                          Vec2(0, 0), 0, "#ff6600")
    world.add_body(rocket)
    
    # Add thrust as a global upward wind field (simulates engine)
    world.add_field(ForceField("wind", direction=Vec2(0, 1), strength=25.0))
    
    # Side thruster
    world.add_field(ForceField("wind", direction=Vec2(1, 0), strength=5.0))
    
    # Ground
    world.add_body(PhysicsBody("floor", Vec2(0, -9.5), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    return world


# === Scenario Registry ===

def create_viscous_liquid_scenario(liquid_name: str = "oil") -> PhysicsWorld:
    """Balls falling into a viscous liquid (oil, honey, water, mercury)"""
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-6, -8, 6, 8),
                         atmosphere=Atmosphere(humidity=0.3, pressure=101325))
    liquid = LIQUIDS.get(liquid_name, LIQUIDS["water"])
    
    # Add liquid zone (from y=-5 to y=-3)
    world.add_liquid_zone("liquid", liquid, y_min=-5, y_max=-2, x_min=-5, x_max=5)
    
    # Balls with different densities
    balls = [
        ("steel_ball", 0, 6, "steel", 3.0, "#C0C0C0", 0.3),
        ("wood_ball", -1, 5.5, "wood", 1.0, "#8B4513", 0.4),
        ("rubber_ball", 1, 5.5, "rubber", 0.5, "#ff4444", 0.3),
        ("ice_ball", -2, 6, "ice", 0.3, "#aaddff", 0.25),
        ("heavy_ball", 2, 6.5, "steel", 5.0, "#444444", 0.5),
    ]
    
    for bid, x, y, mat, mass, color, radius in balls:
        world.add_body(PhysicsBody(bid, Vec2(x, y), Vec2(0, 0),
                                   Shape(type=ShapeType.CIRCLE, radius=radius),
                                   MATERIALS[mat], mass, 0, 0, "dynamic", Vec2(0, 0), 0, color))
    
    world.add_body(PhysicsBody("floor", Vec2(0, -7.5), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    return world


def create_humidity_scenario() -> PhysicsWorld:
    """Same ramp but with varying humidity affecting friction"""
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-6, -8, 6, 6),
                         atmosphere=Atmosphere(humidity=0.9, pressure=101325, temperature=25))
    
    # Wet ramp (humidity reduces friction to 60%)
    world.add_body(PhysicsBody("ramp", Vec2(0, -1), Vec2(0, 0),
                                Shape(type=ShapeType.BOX, half_width=3, half_height=0.15),
                                MATERIALS["wood"], 999, 0.785, 0, "static", Vec2(0, 0), 0, "#8B4513"))
    
    world.add_body(PhysicsBody("ball", Vec2(0, 5), Vec2(0, 0),
                                Shape(type=ShapeType.CIRCLE, radius=0.3),
                                MATERIALS["steel"], 2.0, 0, 0, "dynamic", Vec2(0, 0), 0, "#C0C0C0"))
    
    world.add_body(PhysicsBody("floor", Vec2(0, -6), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    return world


def create_depression_scenario() -> PhysicsWorld:
    """Low pressure zone creating convection (like a mini tornado)"""
    world = PhysicsWorld(gravity=Vec2(0, -9.81), world_bounds=(-10, -10, 10, 10),
                         atmosphere=Atmosphere(humidity=0.8, pressure=95000, temperature=28,
                                              wind=Vec2(2, 0), depression_zone=(0, 3, 5)))
    
    # Light objects that get caught in the depression flow
    for i in range(12):
        angle = i * np.pi / 6
        dist = np.random.uniform(3, 7)
        world.add_body(PhysicsBody(f"leaf_{i}", Vec2(np.cos(angle)*dist, np.random.uniform(-3, 5)),
                                   Vec2(np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5)),
                                   Shape(type=ShapeType.CIRCLE, radius=0.15),
                                   MATERIALS["ice"], 0.1, 0, 0, "dynamic", Vec2(0, 0), 0, "#88ff88"))
    
    world.add_body(PhysicsBody("floor", Vec2(0, -9.5), Vec2(0, 0),
                                Shape(type=ShapeType.PLANE, normal=Vec2(0, 1), offset=0),
                                MATERIALS["stone"], 999, 0, 0, "static", Vec2(0, 0), 0, "#888"))
    
    return world


SCENARIOS = {
    "ramp": create_ramp_scenario,
    "magnetic": create_magnetic_scenario,
    "collision_chain": create_collision_chains,
    "mixed_objects": create_mixed_objects,
    "propulsion": create_propulsion_demo,
    "viscous": create_viscous_liquid_scenario,
    "humidity": create_humidity_scenario,
    "depression": create_depression_scenario,
}

def generate_training_data(world: PhysicsWorld, steps: int = 1000, record_every: int = 1):
    data = []
    for i in range(steps):
        cur = world.get_state_vector()
        world.step(1/60)
        nxt = world.get_state_vector()
        if i % record_every == 0:
            data.append((cur, nxt))
    return data


# === JSON for LLM ===

def scenario_to_json(scenario_name: str, world: PhysicsWorld, steps: int = 120) -> dict:
    states = []
    for i in range(steps):
        world.step(1/60)
        if i % 5 == 0 or i == steps - 1:
            states.append(world.get_state())
    
    return {
        "scenario": scenario_name,
        "total_steps": steps,
        "bodies": [{
            "id": b.id, "shape": b.shape.type.value,
            "material": b.material.name, "mass": b.mass,
            "initial_pos": list(b.position.to_tuple()),
            "initial_vel": list(b.velocity.to_tuple())
        } for b in world.bodies],
        "trajectory": states,
        "fields": [{"type": f.type, "strength": f.strength} for f in world.fields]
    }


if __name__ == "__main__":
    print("=== Physics Engine v2 — Demo ===\n")
    
    # Ramp scenario
    print("1. Ramp scenario (metal ball on 45° wooden ramp)")
    world = create_ramp_scenario(drop_height=8.0)
    for i in range(180):
        world.step(1/60)
        if i % 60 == 0:
            ball = [b for b in world.bodies if b.id == "ball"][0]
            print(f"   t={world.time:.1f}s: ball pos=({ball.position.x:.2f}, {ball.position.y:.2f}) vel=({ball.velocity.x:.2f}, {ball.velocity.y:.2f}) angle={ball.angle:.2f}")
    
    print(f"\n2. Magnetic scenario")
    world2 = create_magnetic_scenario()
    for i in range(60):
        world2.step(1/60)
    metals = [b for b in world2.bodies if "metal" in b.id]
    avg_dist = np.mean([b.position.length() for b in metals])
    print(f"   Average distance from magnet: {avg_dist:.2f} (expected: close to center)")
    
    print(f"\n3. State vector: {world.state_size()} dimensions ({world.state_size()//6} bodies × 6 vars)")
    print("   Format: [x, y, vx, vy, angle, angular_vel] per body")
