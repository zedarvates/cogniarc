"""
Momentum & Inertia Engine — Élan, impulsion, collisions, rotation.
Deep analysis of momentum conservation and rotational inertia.
Designed to clarify these fuzzy concepts for small LLMs.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================
# 1. MOMENTUM (ÉLAN) — The conserved quantity
# ============================================================

class CollisionType(Enum):
    ELASTIC = "elastic"           # Ek conserved (ideal, no deformation)
    INELASTIC = "inelastic"       # Some Ek lost to heat
    PERFECTLY_INELASTIC = "perfectly_inelastic"  # Objects stick together
    EXPLOSIVE = "explosive"       # Objects separate with added energy


@dataclass
class MomentumState:
    """Complete momentum state of a body"""
    body_id: str
    mass: float
    velocity: Tuple[float, float]
    
    @property
    def speed(self) -> float:
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    @property
    def momentum(self) -> Tuple[float, float]:
        return (self.mass * self.velocity[0], self.mass * self.velocity[1])
    
    @property
    def momentum_magnitude(self) -> float:
        return self.mass * self.speed
    
    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.speed**2
    
    def impulse_needed_to(self, target_velocity: Tuple[float, float]) -> Tuple[float, float]:
        """Impulse Δp needed to reach target velocity"""
        return (
            self.mass * (target_velocity[0] - self.velocity[0]),
            self.mass * (target_velocity[1] - self.velocity[1])
        )
    
    def describe(self) -> str:
        p = self.momentum
        return (
            f"{self.body_id}: m={self.mass}kg v=({self.velocity[0]:.1f},{self.velocity[1]:.1f})m/s "
            f"p=({p[0]:.0f},{p[1]:.0f})kg·m/s |p|={self.momentum_magnitude:.0f} Ek={self.kinetic_energy:.0f}J"
        )


class MomentumAnalyzer:
    """
    Analyze momentum conservation in systems.
    p_total = Σ mᵢvᵢ = constant (no external forces)
    """
    
    def __init__(self):
        self.bodies: Dict[str, MomentumState] = {}
        self.external_forces: List[Tuple[str, Tuple[float, float]]] = []
        self.history: List[Tuple[float, Dict[str, Tuple[float, float]]]] = []
    
    def add_body(self, body: MomentumState):
        self.bodies[body.body_id] = body
    
    def total_momentum(self) -> Tuple[float, float]:
        px = sum(b.momentum[0] for b in self.bodies.values())
        py = sum(b.momentum[1] for b in self.bodies.values())
        return (px, py)
    
    def total_kinetic_energy(self) -> float:
        return sum(b.kinetic_energy for b in self.bodies.values())
    
    def center_of_mass_velocity(self) -> Tuple[float, float]:
        total_mass = sum(b.mass for b in self.bodies.values())
        if total_mass == 0:
            return (0, 0)
        px, py = self.total_momentum()
        return (px / total_mass, py / total_mass)
    
    def check_conservation(self, prev_px: float, prev_py: float, 
                           tolerance: float = 0.01) -> dict:
        """Check if momentum was conserved (no external impulses)"""
        px, py = self.total_momentum()
        dp = math.sqrt((px - prev_px)**2 + (py - prev_py)**2)
        conserved = dp < tolerance * max(abs(prev_px) + abs(prev_py), 1.0)
        return {
            "conserved": conserved,
            "delta_momentum": (round(px - prev_px, 3), round(py - prev_py, 3)),
            "delta_magnitude": round(dp, 3),
            "reason": "conservé" if conserved else f"variation de {dp:.1f} kg·m/s (forces externes)"
        }
    
    def predict_elastic_collision_1d(self, a: str, b: str) -> dict:
        """
        Predict 1D elastic collision outcome.
        v1_final = ((m1-m2)v1 + 2m2·v2) / (m1+m2)
        v2_final = ((m2-m1)v2 + 2m1·v1) / (m1+m2)
        """
        body_a = self.bodies[a]
        body_b = self.bodies[b]
        m1, m2 = body_a.mass, body_b.mass
        v1, v2 = body_a.speed, body_b.speed
        
        if abs(m1 + m2) < 1e-10:
            return {"error": "Both masses are zero"}
        
        v1f = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2f = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
        
        Ek_before = body_a.kinetic_energy + body_b.kinetic_energy
        Ek_after = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        
        return {
            "type": "elastic_1d",
            "v1_before": round(v1, 1), "v2_before": round(v2, 1),
            "v1_after": round(v1f, 1), "v2_after": round(v2f, 1),
            "Ek_before": round(Ek_before, 0), "Ek_after": round(Ek_after, 0),
            "Ek_conserved": abs(Ek_before - Ek_after) < 0.01,
            "momentum_transfer": round(abs(m1 * (v1f - v1)), 0),
            "case": self._elastic_case(m1, m2, v1, v2, v1f, v2f)
        }
    
    def _elastic_case(self, m1, m2, v1, v2, v1f, v2f) -> str:
        """Describe the collision outcome in human terms"""
        if m1 == m2:
            return "masses égales → échange complet des vitesses"
        if m2 > m1 * 100:
            return f"objet massif quasi-immobile → rebond élastique à {abs(v1f):.0f}m/s"
        if m1 > m2 * 100:
            return "objet massif continue presque inchangé, objet léger propulsé"
        if abs(v1f - v1) < 0.1:
            return "collision molle — vitesses presque inchangées"
        return "collision élastique standard"
    
    def predict_perfectly_inelastic_1d(self, a: str, b: str) -> dict:
        """
        Both objects stick together: v_final = (m1v1 + m2v2) / (m1+m2)
        """
        body_a = self.bodies[a]
        body_b = self.bodies[b]
        m1, m2 = body_a.mass, body_b.mass
        v1, v2 = body_a.speed, body_b.speed
        
        vf = (m1 * v1 + m2 * v2) / (m1 + m2)
        Ek_before = body_a.kinetic_energy + body_b.kinetic_energy
        Ek_after = 0.5 * (m1 + m2) * vf**2
        energy_lost = Ek_before - Ek_after
        
        return {
            "type": "perfectly_inelastic",
            "v_final": round(vf, 1),
            "Ek_before": round(Ek_before, 0), "Ek_after": round(Ek_after, 0),
            "energy_dissipated": round(energy_lost, 0),
            "dissipation_pct": round(100 * energy_lost / max(Ek_before, 1.0), 1),
            "case": "fusion complète — dissipation maximale d'énergie"
        }
    
    def impulse_from_force(self, force: Tuple[float, float], duration: float) -> Tuple[float, float]:
        """J = ∫F dt ≈ F · Δt"""
        return (force[0] * duration, force[1] * duration)
    
    def apply_impulse(self, body_id: str, impulse: Tuple[float, float]):
        """Apply impulse to change velocity: Δv = J/m"""
        if body_id not in self.bodies:
            return
        b = self.bodies[body_id]
        dvx = impulse[0] / b.mass
        dvy = impulse[1] / b.mass
        b.velocity = (b.velocity[0] + dvx, b.velocity[1] + dvy)
    
    def recoil_analysis(self, projectile_id: str, gun_id: str) -> dict:
        """Recoil: m_projectile × v_projectile = -m_gun × v_gun"""
        proj = self.bodies.get(projectile_id)
        gun = self.bodies.get(gun_id)
        if not proj or not gun:
            return {"error": "Missing bodies"}
        
        p_proj = proj.momentum_magnitude
        recoil_speed = p_proj / max(gun.mass, 1e-10)
        
        impact = "négligeable" if recoil_speed < 0.1 else \
                 "léger recul" if recoil_speed < 1.0 else \
                 "recul notable" if recoil_speed < 5.0 else \
                 "fort recul — besoin d'amortisseur"
        
        return {
            "projectile_momentum": round(p_proj, 0),
            "recoil_speed": round(recoil_speed, 2),
            "recoil_energy": round(0.5 * gun.mass * recoil_speed**2, 0),
            "impact": impact
        }


# ============================================================
# 2. ROTATIONAL INERTIA — Moment of inertia, angular momentum
# ============================================================

class Shape3D(Enum):
    SPHERE = "sphere"
    SOLID_CYLINDER = "solid_cylinder"
    HOLLOW_CYLINDER = "hollow_cylinder"
    ROD_CENTER = "rod_center"       # Rotating around center
    ROD_END = "rod_end"             # Rotating around end
    DISK = "disk"                    # Thin disk
    RING = "ring"                    # Thin ring/hoop
    RECTANGULAR_PLATE = "rectangular_plate"
    POINT_MASS = "point_mass"        # Mass at distance r


@dataclass
class RotationalBody:
    """A body with rotational properties"""
    body_id: str
    mass: float
    shape: Shape3D
    dimensions: Dict[str, float] = field(default_factory=dict)  # radius, length, width, height
    angular_velocity: float = 0.0    # rad/s
    angular_acceleration: float = 0.0
    torque: float = 0.0              # N·m
    position: Tuple[float, float] = (0, 0)
    
    def moment_of_inertia(self) -> float:
        """I — resistance to angular acceleration"""
        m = self.mass
        r = self.dimensions.get("radius", 1.0)
        l = self.dimensions.get("length", 1.0)
        w = self.dimensions.get("width", 1.0)
        h = self.dimensions.get("height", 1.0)
        d = self.dimensions.get("distance_from_axis", 1.0)
        
        formulas = {
            Shape3D.SPHERE: (2/5) * m * r**2,
            Shape3D.SOLID_CYLINDER: (1/2) * m * r**2,
            Shape3D.HOLLOW_CYLINDER: m * r**2,
            Shape3D.ROD_CENTER: (1/12) * m * l**2,
            Shape3D.ROD_END: (1/3) * m * l**2,
            Shape3D.DISK: (1/2) * m * r**2,
            Shape3D.RING: m * r**2,
            Shape3D.RECTANGULAR_PLATE: (1/12) * m * (w**2 + h**2),
            Shape3D.POINT_MASS: m * d**2,
        }
        return formulas.get(self.shape, m * r**2)
    
    @property
    def angular_momentum(self) -> float:
        """L = I·ω"""
        return self.moment_of_inertia() * self.angular_velocity
    
    @property
    def rotational_kinetic_energy(self) -> float:
        """E_rot = ½ I ω²"""
        return 0.5 * self.moment_of_inertia() * self.angular_velocity**2
    
    @property
    def radius_of_gyration(self) -> float:
        """k = √(I/m) — effective radius of mass distribution"""
        I = self.moment_of_inertia()
        return math.sqrt(I / max(self.mass, 1e-10))
    
    def torque_from_force(self, force: float, lever_arm: float, angle_deg: float = 90) -> float:
        """τ = F · r · sin(θ)"""
        return force * lever_arm * math.sin(math.radians(angle_deg))
    
    def angular_accel_from_torque(self, torque: float) -> float:
        """α = τ / I"""
        return torque / max(self.moment_of_inertia(), 1e-10)
    
    def time_to_stop(self, friction_torque: float) -> float:
        """How long to stop under constant friction torque"""
        if abs(friction_torque) < 1e-10:
            return float('inf')
        return abs(self.angular_momentum) / abs(friction_torque)
    
    def compare_rolling(self, other: "RotationalBody", incline_angle: float = 30) -> dict:
        """
        Which object rolls faster down an incline?
        a = g·sin(θ) / (1 + I/(mr²))
        Objects with lower I/(mr²) accelerate faster.
        """
        g = 9.81
        sin_theta = math.sin(math.radians(incline_angle))
        
        # I/(mr²) ratio determines rolling acceleration
        r_self = self.dimensions.get("radius", 1.0)
        r_other = other.dimensions.get("radius", 1.0)
        
        ratio_self = self.moment_of_inertia() / (self.mass * r_self**2 + 1e-10)
        ratio_other = other.moment_of_inertia() / (other.mass * r_other**2 + 1e-10)
        
        a_self = g * sin_theta / (1 + ratio_self)
        a_other = g * sin_theta / (1 + ratio_other)
        
        if a_self > a_other:
            winner = self.body_id
            reason = f"{self.shape.value} accélère plus vite que {other.shape.value} (I/mr²={ratio_self:.2f} vs {ratio_other:.2f})"
        elif a_other > a_self:
            winner = other.body_id
            reason = f"{other.shape.value} accélère plus vite que {self.shape.value} (I/mr²={ratio_other:.2f} vs {ratio_self:.2f})"
        else:
            winner = "égalité"
            reason = "même rapport I/mr² — arrivent ensemble"
        
        return {
            "winner": winner,
            "reason": reason,
            "a_self": round(a_self, 2), "a_other": round(a_other, 2),
            "ratio_self": round(ratio_self, 2), "ratio_other": round(ratio_other, 2)
        }
    
    def describe(self) -> str:
        I = self.moment_of_inertia()
        L = self.angular_momentum
        Ek = self.rotational_kinetic_energy
        return (
            f"{self.body_id}: {self.shape.value} m={self.mass}kg I={I:.2f}kg·m² "
            f"ω={self.angular_velocity:.1f}rad/s L={L:.1f} E_rot={Ek:.1f}J "
            f"k={self.radius_of_gyration:.2f}m"
        )


class RotationalAnalyzer:
    """Analyze rotational systems: gyroscopes, rolling, angular momentum"""
    
    def __init__(self):
        self.bodies: Dict[str, RotationalBody] = []
    
    def add_body(self, body: RotationalBody):
        self.bodies.append(body)
    
    def total_angular_momentum(self) -> float:
        return sum(b.angular_momentum for b in self.bodies)
    
    def gyroscopic_precession(self, body_id: str, external_torque: float) -> dict:
        """
        Gyroscopic precession: ω_p = τ / (I·ω)
        A spinning object resists changes to its axis.
        """
        body = next((b for b in self.bodies if b.body_id == body_id), None)
        if not body:
            return {"error": "Body not found"}
        
        I = body.moment_of_inertia()
        omega = body.angular_velocity
        
        if abs(omega) < 1e-10:
            return {"precession_rate": 0, "note": "Pas de rotation → pas de précession"}
        
        precession_rate = external_torque / (I * omega)
        
        stability = "très stable" if abs(omega) > 100 else \
                    "stable" if abs(omega) > 10 else \
                    "peu stable" if abs(omega) > 1 else \
                    "instable (faible rotation)"
        
        return {
            "precession_rate": round(precession_rate, 4),  # rad/s
            "precession_period": round(2 * math.pi / abs(precession_rate), 2) if abs(precession_rate) > 1e-10 else float('inf'),
            "stability": stability,
            "resists_tilting": abs(omega) > 5  # Strong gyroscopic effect
        }
    
    def angular_momentum_conservation(self, I_initial: float, omega_initial: float,
                                       I_final: float) -> dict:
        """
        If moment of inertia changes (e.g., ice skater pulls arms in),
        angular velocity changes to conserve L = I·ω.
        ω_final = (I_initial · ω_initial) / I_final
        """
        L = I_initial * omega_initial
        omega_final = L / max(I_final, 1e-10)
        
        Ek_initial = 0.5 * I_initial * omega_initial**2
        Ek_final = 0.5 * I_final * omega_final**2
        delta_Ek = Ek_final - Ek_initial
        
        return {
            "L_conserved": round(L, 2),
            "omega_initial": round(omega_initial, 2),
            "omega_final": round(omega_final, 2),
            "omega_ratio": round(omega_final / max(omega_initial, 1e-10), 1),
            "Ek_change_pct": round(100 * delta_Ek / max(abs(Ek_initial), 1.0), 1),
            "explanation": f"En réduisant I de {I_initial:.1f} à {I_final:.1f}, ω passe de {omega_initial:.1f} à {omega_final:.1f} rad/s ({omega_final/max(omega_initial,1e-10):.1f}x plus vite)"
        }
    
    def rolling_vs_sliding(self, body_id: str, incline_angle: float, 
                           friction_coeff: float = 0.3) -> dict:
        """Will it roll or slide?"""
        body = next((b for b in self.bodies if b.body_id == body_id), None)
        if not body:
            return {"error": "Body not found"}
        
        g = 9.81
        sin_theta = math.sin(math.radians(incline_angle))
        cos_theta = math.cos(math.radians(incline_angle))
        
        # Rolling condition: friction must provide enough torque
        # μ ≥ (I/(mr²)) · tan(θ) / (1 + I/(mr²))
        r = body.dimensions.get("radius", 1.0)
        I_ratio = body.moment_of_inertia() / (body.mass * r**2 + 1e-10)
        min_friction = I_ratio * math.tan(math.radians(incline_angle)) / (1 + I_ratio)
        
        will_roll = friction_coeff >= min_friction
        
        return {
            "will_roll": will_roll,
            "min_friction_needed": round(min_friction, 3),
            "available_friction": round(friction_coeff, 3),
            "behavior": "roule sans glisser" if will_roll else "glisse (friction insuffisante pour le roulement)",
            "rolling_accel": round(g * sin_theta / (1 + I_ratio), 2) if will_roll else round(g * sin_theta * (1 - friction_coeff * cos_theta / sin_theta), 2)
        }


# ============================================================
# 3. LLM INTERFACE
# ============================================================

def analyze_momentum_inertia(scene_name: str, 
                              bodies: List[dict],
                              collision_pairs: List[Tuple[str, str]] = None,
                              incline_angle: float = 30.0) -> str:
    """
    Full momentum + inertia analysis.
    bodies: [{id, mass, velocity?, shape?, dimensions?, angular_velocity?}]
    """
    lines = [f"{'='*60}"]
    lines.append(f"  ANALYSE ÉLAN & INERTIE: {scene_name}")
    lines.append(f"{'='*60}")
    
    # === MOMENTUM ANALYSIS ===
    mom = MomentumAnalyzer()
    rot = RotationalAnalyzer()
    
    for b in bodies:
        vel = b.get("velocity", (0, 0))
        if isinstance(vel, (int, float)):
            vel = (vel, 0)
        mom.add_body(MomentumState(b["id"], b.get("mass", 1.0), vel))
    
    px, py = mom.total_momentum()
    Ek = mom.total_kinetic_energy()
    cmv = mom.center_of_mass_velocity()
    
    lines.append(f"\n=== ÉLAN (MOMENTUM) ===")
    lines.append(f"Momentum total: ({px:.0f}, {py:.0f}) kg·m/s  |p|={math.sqrt(px**2+py**2):.0f}")
    lines.append(f"Énergie cinétique totale: {Ek:.0f} J")
    lines.append(f"Vitesse du centre de masse: ({cmv[0]:.2f}, {cmv[1]:.2f}) m/s")
    
    lines.append(f"\nÉTATS INDIVIDUELS:")
    for b in sorted(mom.bodies.values(), key=lambda b: -b.momentum_magnitude):
        lines.append(f"  • {b.describe()}")
    
    # Collision predictions
    if collision_pairs:
        lines.append(f"\nPRÉDICTIONS DE COLLISION:")
        for a, b in collision_pairs:
            # Elastic
            result = mom.predict_elastic_collision_1d(a, b)
            lines.append(f"\n  Élastique {a}↔{b}:")
            lines.append(f"    Avant: v₁={result['v1_before']}m/s  v₂={result['v2_before']}m/s")
            lines.append(f"    Après: v₁={result['v1_after']}m/s  v₂={result['v2_after']}m/s")
            lines.append(f"    Transfert: {result['momentum_transfer']} kg·m/s  |  {result['case']}")
            
            # Perfectly inelastic
            result2 = mom.predict_perfectly_inelastic_1d(a, b)
            lines.append(f"\n  Fusion {a}+{b}:")
            lines.append(f"    v_final={result2['v_final']}m/s  {result2['dissipation_pct']}% énergie dissipée")
    
    # Recoil (if projectile pattern detected)
    heavy = sorted(mom.bodies.values(), key=lambda b: -b.mass)
    light = sorted(mom.bodies.values(), key=lambda b: b.mass)
    if heavy and light and heavy[0].mass > light[0].mass * 10:
        lines.append(f"\nANALYSE DE RECUL:")
        recoil = mom.recoil_analysis(light[0].body_id, heavy[0].body_id)
        lines.append(f"  Si {light[0].body_id} est propulsé → recul sur {heavy[0].body_id}: {recoil['recoil_speed']:.1f}m/s ({recoil['impact']})")
    
    # === ROTATIONAL ANALYSIS ===
    rot_bodies = [b for b in bodies if "shape" in b]
    if rot_bodies:
        lines.append(f"\n\n=== INERTIE ROTATIONNELLE ===")
        
        for b in rot_bodies:
            shape = Shape3D(b["shape"])
            rb = RotationalBody(
                b["id"], b.get("mass", 1.0), shape,
                b.get("dimensions", {"radius": 0.5}),
                b.get("angular_velocity", 0),
                position=b.get("position", (0, 0))
            )
            rot.add_body(rb)
            lines.append(f"  • {rb.describe()}")
        
        # Rolling race
        if len(rot.bodies) >= 2:
            lines.append(f"\nCOURSE SUR PLAN INCLINÉ ({incline_angle}°):")
            for i, b1 in enumerate(rot.bodies):
                for b2 in rot.bodies[i+1:]:
                    race = b1.compare_rolling(b2, incline_angle)
                    lines.append(f"  {b1.body_id} vs {b2.body_id}: {race['reason']}")
                    lines.append(f"    a_{b1.body_id}={race['a_self']}m/s²  a_{b2.body_id}={race['a_other']}m/s²  → {race['winner']} gagne")
        
        # Gyroscopic analysis (fastest spinner)
        fastest = max(rot.bodies, key=lambda b: abs(b.angular_velocity))
        if abs(fastest.angular_velocity) > 1:
            gyro = rot.gyroscopic_precession(fastest.body_id, 10.0)  # 10 Nm external torque
            lines.append(f"\nEFFET GYROSCOPIQUE ({fastest.body_id}):")
            lines.append(f"  Précession: {gyro['precession_rate']:.4f} rad/s")
            lines.append(f"  Période: {gyro['precession_period']}s")
            lines.append(f"  Stabilité: {gyro['stability']}")
        
        # Ice skater effect
        if len(rot.bodies) >= 1:
            first = rot.bodies[0]
            I1 = first.moment_of_inertia()
            I2 = I1 * 0.3  # Pull arms in → reduce I
            cons = rot.angular_momentum_conservation(I1, first.angular_velocity or 10.0, I2)
            lines.append(f"\nCONSERVATION DU MOMENT CINÉTIQUE:")
            lines.append(f"  {cons['explanation']}")
            lines.append(f"  ΔEk = {cons['Ek_change_pct']}% (le travail musculaire ajoute de l'énergie)")
    
    # LLM summary
    lines.append(f"\n{'─'*60}")
    lines.append("RÉSUMÉ POUR LLM:")
    lines.append(f"  L'élan total est {'conservé' if abs(px)+abs(py) < 1e-6 else f'de ({px:.0f},{py:.0f}) kg·m/s'}.")
    lines.append(f"  Le centre de masse se déplace à ({cmv[0]:.2f},{cmv[1]:.2f}) m/s.")
    
    if rot_bodies:
        lines.append(f"  L'inertie rotationnelle dépend de la forme: ")
        examples = {
            Shape3D.SPHERE: "une sphère roule plus vite qu'un cylindre (I/mr²=0.4 vs 0.5)",
            Shape3D.RING: "un anneau est le plus lent (I/mr²=1.0, toute la masse au bord)",
            Shape3D.DISK: "un disque plein accélère modérément (I/mr²=0.5)",
        }
        shapes_seen = set(b["shape"] for b in rot_bodies)
        for s in shapes_seen:
            if s in examples:
                lines.append(f"    {examples[s]}")
    
    if collision_pairs:
        lines.append(f"  Les collisions élastiques conservent l'énergie cinétique. Les inélastiques dissipent.")
    
    return "\n".join(lines)


# ============================================================
# DEMO
# ============================================================

def demo():
    # Scene 1: Billiard balls
    print("1. BILLARD — Collision élastique")
    billiard = [
        {"id": "blanche", "mass": 0.17, "velocity": (3.0, 0), "shape": "sphere", "dimensions": {"radius": 0.028}, "angular_velocity": 50},
        {"id": "rouge", "mass": 0.17, "velocity": (0, 0), "shape": "sphere", "dimensions": {"radius": 0.028}, "angular_velocity": 0},
    ]
    print(analyze_momentum_inertia("Boules de billard", billiard, [("blanche", "rouge")]))
    
    # Scene 2: Rolling race
    print("\n\n2. COURSE — Qui arrive en bas en premier ?")
    race = [
        {"id": "bille_pleine", "mass": 1.0, "velocity": (0, 0), "shape": "sphere", "dimensions": {"radius": 0.5}, "angular_velocity": 0},
        {"id": "cylindre_plein", "mass": 1.0, "velocity": (0, 0), "shape": "solid_cylinder", "dimensions": {"radius": 0.5}, "angular_velocity": 0},
        {"id": "anneau", "mass": 1.0, "velocity": (0, 0), "shape": "ring", "dimensions": {"radius": 0.5}, "angular_velocity": 0},
    ]
    print(analyze_momentum_inertia("Course sur plan incliné 30°", race, incline_angle=30))
    
    # Scene 3: Car crash
    print("\n\n3. ACCIDENT — Collision inélastique")
    crash = [
        {"id": "voiture_A", "mass": 1200, "velocity": (20, 0)},
        {"id": "voiture_B", "mass": 800, "velocity": (-10, 0)},
    ]
    print(analyze_momentum_inertia("Collision frontale", crash, [("voiture_A", "voiture_B")]))


if __name__ == "__main__":
    demo()
