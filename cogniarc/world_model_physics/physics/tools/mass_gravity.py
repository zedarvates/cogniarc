"""
Mass/Inertia/Gravity Engine — Poids, inertie, attraction gravitationnelle.
Formalizes fuzzy physics concepts for small LLM reasoning.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================
# 1. MASS PROPERTIES — Beyond just a number
# ============================================================

class MassCategory(Enum):
    """Qualitative mass categories for LLM reasoning"""
    FEATHER = "feather"       # < 0.1 kg  — negligible inertia
    LIGHT = "light"           # 0.1-1 kg  — easily moved
    MODERATE = "moderate"     # 1-10 kg   — noticeable resistance
    HEAVY = "heavy"           # 10-100 kg — hard to move
    MASSIVE = "massive"       # 100-1000 kg — very hard to move
    IMMENSE = "immense"       # > 1000 kg — effectively immovable for small forces


class GravityRegime(Enum):
    """Gravity context"""
    MICROGRAVITY = "microgravity"   # Orbit, space station (~0g)
    LOW_GRAVITY = "low_gravity"     # Moon (0.16g), Mars (0.38g)
    EARTH = "earth"                 # 1g standard
    HIGH_GRAVITY = "high_gravity"   # Jupiter (2.5g), neutron star
    ZERO_G = "zero_g"               # Deep space, free fall


@dataclass
class MassProperties:
    """Complete mass characterization of an object"""
    object_id: str
    mass: float                          # kg — invariant
    radius: float = 0.5                  # m — for volume/density
    density: float = 0.0                 # kg/m³ — computed
    is_hollow: bool = False              # Affects moment of inertia
    material_name: str = "unknown"
    
    # Derived
    category: MassCategory = MassCategory.MODERATE
    weight_on_earth: float = 0.0         # N
    inertia_resistance: float = 0.0      # How hard to accelerate (normalized 0-1)
    momentum_at_v: Dict[float, float] = field(default_factory=dict)  # v → p=mv
    
    def __post_init__(self):
        if self.density == 0 and self.radius > 0:
            volume = (4/3) * math.pi * self.radius**3
            if not self.is_hollow:
                self.density = self.mass / max(volume, 1e-10)
            else:
                shell_thickness = self.radius * 0.1
                shell_volume = 4 * math.pi * self.radius**2 * shell_thickness
                self.density = self.mass / max(shell_volume, 1e-10)
        
        self.weight_on_earth = self.mass * 9.81
        self.category = self._classify_mass()
        self.inertia_resistance = self._compute_inertia_resistance()
    
    def _classify_mass(self) -> MassCategory:
        if self.mass < 0.1: return MassCategory.FEATHER
        if self.mass < 1.0: return MassCategory.LIGHT
        if self.mass < 10.0: return MassCategory.MODERATE
        if self.mass < 100.0: return MassCategory.HEAVY
        if self.mass < 1000.0: return MassCategory.MASSIVE
        return MassCategory.IMMENSE
    
    def _compute_inertia_resistance(self) -> float:
        """How hard to accelerate: 0=trivial, 1=nearly impossible"""
        log_mass = math.log10(max(self.mass, 0.001))
        return min(1.0, (log_mass + 3) / 6)  # 0.001kg→0, 1000kg→1
    
    def weight_at_gravity(self, g: float) -> float:
        """Weight = mass × local gravity"""
        return self.mass * g
    
    def momentum_at_speed(self, velocity: float) -> float:
        """p = mv"""
        return self.mass * abs(velocity)
    
    def kinetic_energy_at_speed(self, velocity: float) -> float:
        """Ek = ½mv²"""
        return 0.5 * self.mass * velocity**2
    
    def force_to_accelerate(self, target_accel: float) -> float:
        """F = ma — force needed for desired acceleration"""
        return self.mass * target_accel
    
    def stopping_distance(self, velocity: float, friction_coeff: float = 0.3) -> float:
        """How far to stop on a surface with given friction"""
        if friction_coeff <= 0:
            return float('inf')
        return velocity**2 / (2 * friction_coeff * 9.81)
    
    def compare(self, other: "MassProperties") -> dict:
        """Compare with another object"""
        ratio = self.mass / max(other.mass, 1e-10)
        if ratio > 100:
            relation = f"est {ratio:.0f}x plus massif que {other.object_id} — domine complètement"
        elif ratio > 10:
            relation = f"est {ratio:.0f}x plus lourd que {other.object_id} — net avantage"
        elif ratio > 2:
            relation = f"est {ratio:.1f}x plus lourd que {other.object_id}"
        elif ratio > 1.1:
            relation = f"est légèrement plus lourd que {other.object_id}"
        elif ratio > 0.9:
            relation = f"a une masse similaire à {other.object_id}"
        elif ratio > 0.5:
            relation = f"est légèrement plus léger que {other.object_id}"
        elif ratio > 0.1:
            relation = f"est {1/ratio:.0f}x plus léger que {other.object_id} — désavantage"
        else:
            relation = f"est négligeable face à {other.object_id}"
        
        return {
            "ratio": round(ratio, 1),
            "relation": relation,
            "collision_dominance": "collision_inelastic" if ratio > 10 else "collision_balanced"
        }


# ============================================================
# 2. GRAVITATIONAL INTERACTIONS
# ============================================================

@dataclass
class GravityPair:
    """Gravitational interaction between two masses"""
    body_a: str
    body_b: str
    mass_a: float
    mass_b: float
    distance: float                    # m — center to center
    force: float                       # N — F = G*m1*m2 / r²
    acceleration_a: float              # m/s² — on body A
    acceleration_b: float              # m/s² — on body B
    regime: str = ""                   # "dominant", "noticeable", "negligible"
    
    def describe(self) -> str:
        return (
            f"{self.body_a}↔{self.body_b}: F={self.force:.2e}N "
            f"(a_a={self.acceleration_a:.2e}m/s², a_b={self.acceleration_b:.2e}m/s²) "
            f"[{self.regime}]"
        )


G = 6.67430e-11  # Gravitational constant


def compute_gravity(m1: float, m2: float, distance: float) -> float:
    """F = G·m₁·m₂ / r²"""
    if distance < 1e-10:
        return float('inf')
    return G * m1 * m2 / (distance * distance)


def escape_velocity(mass_central: float, distance: float) -> float:
    """v_esc = √(2GM/r) — speed needed to escape gravity"""
    if distance < 1e-10:
        return float('inf')
    return math.sqrt(2 * G * mass_central / distance)


def orbital_velocity(mass_central: float, distance: float) -> float:
    """v_orb = √(GM/r) — speed for circular orbit"""
    if distance < 1e-10:
        return float('inf')
    return math.sqrt(G * mass_central / distance)


def hill_sphere(mass: float, mass_central: float, distance: float) -> float:
    """Radius where an object's gravity dominates (for moons/satellites)"""
    return distance * (mass / (3 * mass_central)) ** (1/3)


def roche_limit(density_body: float, density_central: float, radius_central: float) -> float:
    """Distance at which tidal forces break apart the body"""
    if density_central < 1e-10:
        return float('inf')
    return radius_central * 1.26 * (density_central / max(density_body, 1e-10)) ** (1/3)


def lagrange_points(m1: float, m2: float, distance: float) -> Dict[str, Tuple[float, float]]:
    """
    Compute L1, L2, L3 Lagrange points (simplified for circular restricted 3-body).
    Returns positions along the line connecting the two masses.
    """
    mu = m2 / (m1 + m2)  # Mass ratio
    
    # L1: between m1 and m2
    # Solve quintic approximately: (1-mu)/(x²) - mu/((1-x)²) - x + (1-mu) = 0
    # Approximate: x_L1 ≈ distance * (1 - (mu/3)^(1/3))
    x_L1 = distance * (1 - (mu / 3) ** (1/3))
    
    # L2: beyond m2
    x_L2 = distance * (1 + (mu / 3) ** (1/3))
    
    # L3: opposite side of m1
    x_L3 = -distance * (1 - 5 * mu / 12)
    
    return {
        "L1": (x_L1, 0),  # Between
        "L2": (x_L2, 0),  # Behind smaller mass
        "L3": (x_L3, 0),  # Opposite side
        # L4, L5 are at ±60° — equilateral triangles
        "L4": (distance * 0.5 - distance * mu, distance * math.sqrt(3) / 2),
        "L5": (distance * 0.5 - distance * mu, -distance * math.sqrt(3) / 2),
    }


# ============================================================
# 3. CENTER OF MASS — The system's balance point
# ============================================================

@dataclass
class CenterOfMass:
    """System's center of mass and related properties"""
    position: Tuple[float, float]
    total_mass: float
    contributions: List[dict]  # Each object's contribution
    moment_of_inertia: float = 0.0
    is_stable: bool = True     # Is CoM over the support base?
    
    def describe(self) -> str:
        x, y = self.position
        return (
            f"Centre de masse: ({x:.2f}, {y:.2f}), "
            f"masse totale={self.total_mass:.1f}kg, "
            f"stabilité={'✅ stable' if self.is_stable else '⚠️ instable'}"
        )


def compute_center_of_mass(masses: List[Tuple[str, float, Tuple[float, float]]]) -> CenterOfMass:
    """
    Compute center of mass for N bodies.
    masses: [(id, mass, (x, y)), ...]
    """
    if not masses:
        return CenterOfMass((0, 0), 0, [], 0, True)
    
    total_mass = sum(m for _, m, _ in masses)
    if total_mass == 0:
        return CenterOfMass((0, 0), 0, [], 0, True)
    
    com_x = sum(m * x for _, m, (x, _) in masses) / total_mass
    com_y = sum(m * y for _, m, (_, y) in masses) / total_mass
    
    contributions = []
    for obj_id, mass, (x, y) in masses:
        dx, dy = x - com_x, y - com_y
        dist = math.sqrt(dx**2 + dy**2)
        contributions.append({
            "id": obj_id,
            "mass": mass,
            "distance": round(dist, 2),
            "moment": round(mass * dist, 2),
            "weight_pct": round(100 * mass / total_mass, 1)
        })
    
    # Moment of inertia around CoM
    I = sum(m * ((x - com_x)**2 + (y - com_y)**2) for _, m, (x, y) in masses)
    
    # Stability: is CoM within the convex hull of support points?
    # Simplified: check if there are objects below the CoM
    support_below = any(y < com_y for _, _, (_, y) in masses)
    
    return CenterOfMass(
        (round(com_x, 3), round(com_y, 3)),
        round(total_mass, 2),
        contributions,
        round(I, 2),
        support_below
    )


# ============================================================
# 4. MOTION QUALIFIERS — How does it move?
# ============================================================

class MotionType(Enum):
    STATIC = "static"               # Not moving
    UNIFORM = "uniform"             # Constant velocity (Newton's 1st law)
    ACCELERATING = "accelerating"   # Changing speed
    DECELERATING = "decelerating"   # Slowing down
    OSCILLATING = "oscillating"     # Periodic motion
    ORBITING = "orbiting"           # Circular/elliptical path
    ESCAPING = "escaping"           # Exceeding escape velocity
    IMPACTING = "impacting"         # About to collide
    FREE_FALL = "free_fall"         # Under gravity only


@dataclass 
class MotionAnalysis:
    """Complete motion state of a body"""
    object_id: str
    velocity: float                  # m/s
    acceleration: float              # m/s²
    mass: float
    momentum: float                  # kg·m/s
    kinetic_energy: float            # J
    motion_type: MotionType
    force_needed_to_stop: float      # N (over 1s)
    can_be_stopped_by_human: bool   # < 500N
    can_be_stopped_by_vehicle: bool  # < 5000N
    
    def describe(self) -> str:
        icons = {
            MotionType.STATIC: "⏸",
            MotionType.UNIFORM: "➡",
            MotionType.ACCELERATING: "🚀",
            MotionType.DECELERATING: "🛑",
            MotionType.OSCILLATING: "〰",
            MotionType.ORBITING: "🔄",
            MotionType.ESCAPING: "💨",
            MotionType.IMPACTING: "💥",
            MotionType.FREE_FALL: "⬇",
        }
        icon = icons.get(self.motion_type, "❓")
        human = "arrêtable par humain" if self.can_be_stopped_by_human else "trop massif pour humain"
        return (
            f"{icon} {self.object_id}: v={self.velocity:.1f}m/s p={self.momentum:.0f}kg·m/s "
            f"Ek={self.kinetic_energy:.0f}J | {self.motion_type.value} | {human}"
        )


# ============================================================
# 5. LLM INTERFACE
# ============================================================

def analyze_mass_gravity(scene_name: str, objects: List[Dict]) -> str:
    """
    Full mass/gravity/inertia analysis.
    objects: [{id, mass, radius?, pos?, vel?, material?, hollow?}]
    """
    lines = [f"{'='*60}"]
    lines.append(f"  ANALYSE MASSE/GRAVITÉ/INERTIE: {scene_name}")
    lines.append(f"{'='*60}")
    
    # Build mass properties
    props = []
    for obj in objects:
        p = MassProperties(
            obj["id"],
            obj.get("mass", 1.0),
            obj.get("radius", 0.5),
            material_name=obj.get("material", "unknown"),
            is_hollow=obj.get("hollow", False)
        )
        props.append(p)
    
    # Per-object
    lines.append(f"\nPROPRIÉTÉS DE MASSE:")
    for p in sorted(props, key=lambda p: -p.mass)[:10]:
        lines.append(
            f"  • {p.object_id}: {p.mass:.1f}kg ({p.category.value}) "
            f"densité={p.density:.0f}kg/m³ "
            f"poids_terre={p.weight_on_earth:.1f}N "
            f"inertie={p.inertia_resistance:.0%} "
            f"{'(creux)' if p.is_hollow else ''} "
            f"[{p.material_name}]"
        )
    
    # Mass comparisons
    if len(props) >= 2:
        lines.append(f"\nCOMPARAISONS DE MASSE:")
        heaviest = max(props, key=lambda p: p.mass)
        lightest = min(props, key=lambda p: p.mass)
        lines.append(f"  {heaviest.compare(lightest)['relation']}")
        
        # Mass ratios
        for i, p1 in enumerate(props):
            for p2 in props[i+1:]:
                if abs(p1.mass - p2.mass) > 0.01:
                    comp = p1.compare(p2)
                    if "négligeable" not in comp["relation"] and "similaire" not in comp["relation"]:
                        lines.append(f"  {comp['relation']}")
    
    # Gravitational interactions
    lines.append(f"\nINTERACTIONS GRAVITATIONNELLES:")
    any_noticeable = False
    for i, p1 in enumerate(props):
        for p2 in props[i+1:]:
            # Find positions
            pos1 = next((o.get("pos", (0, 0)) for o in objects if o["id"] == p1.object_id), (0, 0))
            pos2 = next((o.get("pos", (0, 0)) for o in objects if o["id"] == p2.object_id), (0, 0))
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < 1e-10:
                continue
            
            force = compute_gravity(p1.mass, p2.mass, dist)
            a1 = force / max(p1.mass, 1e-10)
            a2 = force / max(p2.mass, 1e-10)
            
            # Classify regime
            if a1 > 0.01 or a2 > 0.01:
                regime = "dominant"
                any_noticeable = True
            elif a1 > 1e-6:
                regime = "noticeable"
                any_noticeable = True
            else:
                regime = "négligeable"
            
            if regime != "négligeable":
                pair = GravityPair(p1.object_id, p2.object_id, p1.mass, p2.mass, dist, force, a1, a2, regime)
                lines.append(f"  {pair.describe()}")
    
    if not any_noticeable:
        max_force = 0
        max_pair = None
        for i, p1 in enumerate(props):
            for p2 in props[i+1:]:
                pos1 = next((o.get("pos", (0, 0)) for o in objects if o["id"] == p1.object_id), (0, 0))
                pos2 = next((o.get("pos", (0, 0)) for o in objects if o["id"] == p2.object_id), (0, 0))
                dist = math.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
                if dist > 0:
                    f = compute_gravity(p1.mass, p2.mass, dist)
                    if f > max_force:
                        max_force = f
                        max_pair = (p1, p2, dist)
        if max_pair:
            lines.append(f"  Toutes les interactions sont négligeables (max: {max_pair[0].object_id}↔{max_pair[1].object_id} = {max_force:.2e}N)")
    
    # Center of mass
    masses_data = []
    for p in props:
        pos = next((o.get("pos", (0, 0)) for o in objects if o["id"] == p.object_id), (0, 0))
        masses_data.append((p.object_id, p.mass, pos))
    
    com = compute_center_of_mass(masses_data)
    lines.append(f"\nCENTRE DE MASSE:")
    lines.append(f"  {com.describe()}")
    lines.append(f"  Contributions:")
    for c in sorted(com.contributions, key=lambda c: -c["weight_pct"]):
        lines.append(f"    {c['id']}: {c['weight_pct']:.0f}% de la masse, moment={c['moment']:.1f}")
    
    # Motion analysis (if velocities provided)
    motions = []
    for obj in objects:
        vel = obj.get("vel", (0, 0))
        speed = math.sqrt(vel[0]**2 + vel[1]**2) if isinstance(vel, (list, tuple)) else float(vel)
        p = next((pr for pr in props if pr.object_id == obj["id"]), None)
        if p and speed > 0.01:
            acc = obj.get("acc", 0)
            motion_type = MotionType.UNIFORM
            if acc > 0.1:
                motion_type = MotionType.ACCELERATING
            elif acc < -0.1:
                motion_type = MotionType.DECELERATING
            
            ma = MotionAnalysis(
                obj["id"], speed, acc, p.mass,
                p.momentum_at_speed(speed),
                p.kinetic_energy_at_speed(speed),
                motion_type,
                p.momentum_at_speed(speed),  # Force to stop in 1s
                p.momentum_at_speed(speed) < 500,
                p.momentum_at_speed(speed) < 5000
            )
            motions.append(ma)
    
    if motions:
        lines.append(f"\nANALYSE DE MOUVEMENT:")
        for m in sorted(motions, key=lambda m: -m.kinetic_energy):
            lines.append(f"  {m.describe()}")
    
    # Orbital mechanics (if applicable)
    if len(props) >= 2:
        central = max(props, key=lambda p: p.mass)
        if central.mass > 10:
            lines.append(f"\nMÉCANIQUE ORBITALE (autour de {central.object_id}):")
            for p in props:
                if p.object_id == central.object_id:
                    continue
                pos = next((o.get("pos", (0, 0)) for o in objects if o["id"] == p.object_id), (0, 0))
                cpos = next((o.get("pos", (0, 0)) for o in objects if o["id"] == central.object_id), (0, 0))
                dist = math.sqrt((pos[0]-cpos[0])**2 + (pos[1]-cpos[1])**2)
                
                v_esc = escape_velocity(central.mass, dist)
                v_orb = orbital_velocity(central.mass, dist)
                r_hill = hill_sphere(p.mass, central.mass, dist)
                
                vel = next((o.get("vel", (0, 0)) for o in objects if o["id"] == p.object_id), (0, 0))
                speed = math.sqrt(vel[0]**2 + vel[1]**2) if isinstance(vel, (list, tuple)) else 0
                
                status = "en orbite" if abs(speed - v_orb) < v_orb * 0.1 else \
                         "s'échappe" if speed > v_esc else \
                         "en chute" if speed < v_orb else "transition"
                
                lines.append(f"  {p.object_id} @ {dist:.1f}m: v_orb={v_orb:.1f}m/s v_esc={v_esc:.1f}m/s "
                           f"v_actuel={speed:.1f}m/s → {status}")
    
    # LLM summary
    lines.append(f"\n{'─'*60}")
    lines.append("RÉSUMÉ POUR LLM:")
    total_mass = sum(p.mass for p in props)
    heaviest = max(props, key=lambda p: p.mass)
    lines.append(f"  Masse totale: {total_mass:.1f}kg ({len(props)} objets)")
    lines.append(f"  Objet dominant: {heaviest.object_id} ({heaviest.mass:.1f}kg, {heaviest.category.value})")
    
    if motions:
        max_ek = max(m.kinetic_energy for m in motions)
        most_energetic = [m for m in motions if m.kinetic_energy == max_ek][0]
        lines.append(f"  Plus énergétique: {most_energetic.object_id} (Ek={max_ek:.0f}J)")
    
    lines.append(f"  Stabilité: {'✅' if com.is_stable else '⚠️'} centre de masse {'soutenu' if com.is_stable else 'non soutenu'}")
    
    # Quick reasoning helpers
    lines.append(f"\n  → Si collision: {heaviest.object_id} domine (masse {heaviest.mass:.0f}kg vs moyenne {total_mass/len(props):.1f}kg)")
    lines.append(f"  → Sur Terre: le poids total est de {total_mass * 9.81:.0f}N ({total_mass * 9.81 / 1000:.1f}kN)")
    lines.append(f"  → Sur la Lune: le poids serait {total_mass * 1.62:.0f}N ({(1.62/9.81*100):.0f}% du poids terrestre)")
    
    return "\n".join(lines)


# ============================================================
# DEMO
# ============================================================

def demo():
    # Scene 1: Solar system (simplified)
    print("1. SYSTÈME SOLAIRE SIMPLIFIÉ")
    solar = [
        {"id": "Soleil", "mass": 1.989e30, "radius": 696.34e6, "pos": (0, 0), "material": "plasma"},
        {"id": "Terre", "mass": 5.972e24, "radius": 6371e3, "pos": (149.6e9, 0), "vel": (0, 29783), "material": "roche"},
        {"id": "Lune", "mass": 7.342e22, "radius": 1737e3, "pos": (149.6e9 + 384.4e6, 0), "vel": (0, 29783 + 1022), "material": "roche"},
        {"id": "ISS", "mass": 420e3, "radius": 50, "pos": (149.6e9 + 6.771e6, 0), "vel": (0, 29783 + 7660), "material": "metal", "hollow": True},
    ]
    print(analyze_mass_gravity("Système Terre-Lune-ISS", solar))
    
    # Scene 2: Everyday objects
    print("\n\n2. OBJETS DU QUOTIDIEN — Chariot + caisses")
    everyday = [
        {"id": "chariot", "mass": 50, "pos": (0, 0), "material": "acier"},
        {"id": "caisse_A", "mass": 25, "pos": (0.5, 0.2), "material": "bois"},
        {"id": "caisse_B", "mass": 15, "pos": (-0.3, -0.2), "material": "bois"},
        {"id": "bouteille", "mass": 1, "pos": (0.8, 0), "material": "verre"},
        {"id": "personne", "mass": 70, "pos": (-1, 0.5), "vel": (0.5, 0), "material": "humain"},
    ]
    print(analyze_mass_gravity("Chariot de livraison", everyday))
    
    # Scene 3: Vehicle crash scenario
    print("\n\n3. COLLISION — Camion vs vélo")
    crash = [
        {"id": "camion", "mass": 12000, "pos": (-20, 0), "vel": (15, 0), "material": "acier"},
        {"id": "velo", "mass": 15, "pos": (5, 0), "vel": (-5, 0), "material": "alu"},
        {"id": "cycliste", "mass": 75, "pos": (5, 0.5), "vel": (-5, 0), "material": "humain"},
    ]
    print(analyze_mass_gravity("Collision camion-vélo", crash))


if __name__ == "__main__":
    demo()
