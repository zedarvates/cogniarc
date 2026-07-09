"""
Torque & Couple Engine — Moments, force couples, rotational equilibrium.
Plus: Micro-NN Expert System — tiny domain-specific networks activated on demand.
"""

import numpy as np
import math
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum


# ============================================================
# 1. TORQUE & COUPLE — Force moments, rotational equilibrium
# ============================================================

@dataclass
class ForceApplication:
    """A force applied at a specific point"""
    force_id: str
    magnitude: float           # N
    direction: Tuple[float, float]  # Unit vector
    application_point: Tuple[float, float]  # Where force is applied
    body_id: str = ""          # Which body it acts on
    
    def to_vector(self) -> Tuple[float, float]:
        return (self.magnitude * self.direction[0], 
                self.magnitude * self.direction[1])
    
    def torque_about(self, pivot: Tuple[float, float]) -> float:
        """τ = r × F (scalar in 2D, positive = CCW)"""
        rx = self.application_point[0] - pivot[0]
        ry = self.application_point[1] - pivot[1]
        Fx = self.magnitude * self.direction[0]
        Fy = self.magnitude * self.direction[1]
        return rx * Fy - ry * Fx  # Cross product in 2D
    
    def moment_arm(self, pivot: Tuple[float, float]) -> float:
        """Perpendicular distance from pivot to line of force"""
        rx = self.application_point[0] - pivot[0]
        ry = self.application_point[1] - pivot[1]
        # Distance from point to line
        Fx = self.direction[0]
        Fy = self.direction[1]
        # Line: point = app_point + t * direction
        # Distance = |(p-pivot) × direction| / |direction|
        cross = rx * Fy - ry * Fx
        return abs(cross)


@dataclass
class ForceCouple:
    """Two equal, opposite, parallel forces = pure torque"""
    force_magnitude: float
    separation: float          # Perpendicular distance between lines of action
    direction: str = "CCW"     # "CW" or "CCW"
    
    @property
    def moment(self) -> float:
        """M = F · d"""
        return self.force_magnitude * self.separation
    
    def describe(self) -> str:
        return f"Couple: {self.force_magnitude}N × {self.separation}m = {self.moment}N·m ({self.direction})"


class TorqueAnalyzer:
    """Analyze rotational effects of forces on rigid bodies"""
    
    def __init__(self):
        self.forces: List[ForceApplication] = []
        self.couples: List[ForceCouple] = []
        self.pivots: Dict[str, Tuple[float, float]] = {}
    
    def add_force(self, force: ForceApplication):
        self.forces.append(force)
    
    def add_couple(self, couple: ForceCouple):
        self.couples.append(couple)
    
    def add_pivot(self, name: str, position: Tuple[float, float]):
        self.pivots[name] = position
    
    def net_torque_about(self, pivot: Tuple[float, float]) -> float:
        """Sum of all torques around a pivot"""
        total = sum(f.torque_about(pivot) for f in self.forces)
        for c in self.couples:
            sign = 1 if c.direction == "CCW" else -1
            total += sign * c.moment
        return total
    
    def is_in_rotational_equilibrium(self, pivot: Tuple[float, float], 
                                      tolerance: float = 0.01) -> bool:
        return abs(self.net_torque_about(pivot)) < tolerance
    
    def find_equilibrium_pivot(self, search_bounds: Tuple[float, float, float, float] = (-10, -10, 10, 10),
                                resolution: int = 20) -> Optional[Tuple[float, float]]:
        """Find a pivot where net torque is zero (if one exists)"""
        best = None
        best_torque = float('inf')
        
        for x in np.linspace(search_bounds[0], search_bounds[2], resolution):
            for y in np.linspace(search_bounds[1], search_bounds[3], resolution):
                torque = self.net_torque_about((x, y))
                if abs(torque) < best_torque:
                    best_torque = abs(torque)
                    best = (x, y)
        
        return best if best_torque < 0.1 else None
    
    def wrench_equivalent(self, reduction_point: Tuple[float, float]) -> dict:
        """
        Reduce all forces + couples to a single force + single couple
        at the reduction point (equivalent wrench).
        """
        # Sum all forces
        Fx = sum(f.magnitude * f.direction[0] for f in self.forces)
        Fy = sum(f.magnitude * f.direction[1] for f in self.forces)
        F_mag = math.sqrt(Fx**2 + Fy**2)
        F_dir = (Fx / F_mag, Fy / F_mag) if F_mag > 1e-10 else (0, 0)
        
        # Net torque at reduction point
        M = self.net_torque_about(reduction_point)
        
        return {
            "resultant_force": (round(Fx, 1), round(Fy, 1)),
            "force_magnitude": round(F_mag, 1),
            "force_direction": (round(F_dir[0], 3), round(F_dir[1], 3)),
            "resultant_moment": round(M, 1),
            "type": "wrench" if abs(M) > 0.01 else "force_pure" if F_mag > 0.01 else "équilibre"
        }
    
    def stability_analysis(self, pivot: Tuple[float, float]) -> dict:
        """
        Is the system stable around this pivot?
        Stable: restoring torque opposes displacement.
        """
        torque_current = self.net_torque_about(pivot)
        
        # Test small displacements
        eps = 0.01
        torque_right = self.net_torque_about((pivot[0] + eps, pivot[1]))
        torque_up = self.net_torque_about((pivot[0], pivot[1] + eps))
        
        # Restoring check: if displaced right, does torque push back?
        # For rotational: if displaced CCW, does torque push CW?
        restoring_x = torque_right * torque_current < 0 if abs(torque_current) > 1e-10 else True
        restoring_y = torque_up * torque_current < 0 if abs(torque_current) > 1e-10 else True
        
        return {
            "current_torque": round(torque_current, 2),
            "is_equilibrium": abs(torque_current) < 0.01,
            "stable_x": restoring_x,
            "stable_y": restoring_y,
            "overall": "stable" if (restoring_x and restoring_y) else 
                       "neutre" if abs(torque_current) < 0.01 else 
                       "instable"
        }
    
    def mechanical_advantage(self, input_force_id: str, output_point: Tuple[float, float],
                             pivot: Tuple[float, float]) -> dict:
        """
        For levers: MA = input_arm / output_arm
        """
        inp = next((f for f in self.forces if f.force_id == input_force_id), None)
        if not inp:
            return {"error": "Input force not found"}
        
        input_arm = inp.moment_arm(pivot)
        
        # Output arm: distance from pivot to output point measured 
        # perpendicular to the direction of useful motion
        # (simplified: assume useful motion is vertical at output point)
        output_arm = abs(output_point[0] - pivot[0])  # Horizontal distance
        
        if abs(output_arm) < 1e-10:
            return {"mechanical_advantage": float('inf'), "type": "levier impossible"}
        
        MA = input_arm / output_arm
        
        return {
            "input_arm": round(input_arm, 2),
            "output_arm": round(output_arm, 2),
            "mechanical_advantage": round(MA, 2),
            "type": "amplification" if MA > 1 else "réduction" if MA < 1 else "transmission 1:1",
            "output_force": round(inp.magnitude * MA, 1)
        }
    
    def lever_classification(self, input_id: str, output_point: Tuple[float, float],
                             pivot: Tuple[float, float]) -> str:
        """
        Class 1: pivot between input and output (seesaw)
        Class 2: output between pivot and input (wheelbarrow)
        Class 3: input between pivot and output (tweezers)
        """
        inp = next((f for f in self.forces if f.force_id == input_id), None)
        if not inp:
            return "inconnu"
        
        ix = inp.application_point[0]
        px = pivot[0]
        ox = output_point[0]
        
        if (ix - px) * (ox - px) < 0:
            return "Classe 1 — pivot entre force et charge (levier inter-appui)"
        elif abs(ox - px) < abs(ix - px):
            return "Classe 2 — charge entre pivot et force (brouette)"
        else:
            return "Classe 3 — force entre pivot et charge (pince)"


# ============================================================
# 2. MICRO-NN EXPERT SYSTEM — Tiny domain networks
# ============================================================

class ExpertDomain(Enum):
    """Each expert specializes in one domain"""
    MOMENTUM = "momentum"           # Collisions, elastic/inelastic
    INERTIA = "inertia"             # Rotational dynamics, I, L
    TORQUE = "torque"               # Moments, levers, equilibrium
    SPATIAL = "spatial"             # Inside/outside, near/far
    THERMAL = "thermal"             # Heat transfer, phase changes
    GRAVITY = "gravity"             # Orbits, weight, attraction
    KINEMATIC = "kinematic"         # Mobility, workspace
    ENERGY = "energy"               # Budget, conservation
    FLUID = "fluid"                 # Buoyancy, viscosity, drag
    CAUSAL = "causal"               # Cause→effect chains


@dataclass
class MicroExpert:
    """
    Tiny neural network (< 500 params) specialized in one domain.
    Activated by LLM based on context keywords.
    """
    domain: ExpertDomain
    name: str
    activation_keywords: List[str]  # What triggers this expert
    
    # Tiny MLP parameters
    input_size: int = 4
    hidden_size: int = 8          # Very small
    output_size: int = 2
    
    # Learned parameters (initialized randomly)
    w1: np.ndarray = None
    b1: np.ndarray = None
    w2: np.ndarray = None
    b2: np.ndarray = None
    
    # Expert knowledge (hardcoded physics rules)
    rules: List[str] = field(default_factory=list)  # Domain-specific heuristics
    
    def __post_init__(self):
        if self.w1 is None:
            rng = np.random.RandomState(hash(self.domain.value) % 10000)
            self.w1 = rng.randn(self.input_size, self.hidden_size) * 0.1
            self.b1 = np.zeros(self.hidden_size)
            self.w2 = rng.randn(self.hidden_size, self.output_size) * 0.1
            self.b2 = np.zeros(self.output_size)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Tiny forward pass"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        return h @ self.w2 + self.b2
    
    def param_count(self) -> int:
        return (self.w1.size + self.b1.size + self.w2.size + self.b2.size)
    
    def matches(self, query: str) -> float:
        """How relevant is this expert to the query? 0-1"""
        query_lower = query.lower()
        # Count keyword matches
        matched = sum(1 for kw in self.activation_keywords if kw in query_lower)
        if matched > 0:
            # At least 1 keyword match → activate
            return min(1.0, 0.3 + matched * 0.2)
        return 0.0
    
    def get_rules(self) -> str:
        return "\n".join(f"  • {r}" for r in self.rules)


class ExpertRegistry:
    """
    Registry of micro-experts. The LLM selects which experts to activate
    based on the query context. Only activated experts consume resources.
    """
    
    def __init__(self):
        self.experts: Dict[ExpertDomain, MicroExpert] = {}
        self.active: List[MicroExpert] = []
        self.total_params_activated: int = 0
        self.total_params_available: int = 0
    
    def register(self, expert: MicroExpert):
        self.experts[expert.domain] = expert
        self.total_params_available += expert.param_count()
    
    def activate_by_query(self, query: str, threshold: float = 0.3):
        """Activate experts whose keywords match the query"""
        self.active = []
        self.total_params_activated = 0
        
        for domain, expert in self.experts.items():
            score = expert.matches(query)
            if score >= threshold:
                self.active.append(expert)
                self.total_params_activated += expert.param_count()
    
    def activate_domains(self, domains: List[ExpertDomain]):
        """Directly activate specific domains"""
        self.active = []
        self.total_params_activated = 0
        for d in domains:
            if d in self.experts:
                self.active.append(self.experts[d])
                self.total_params_activated += self.experts[d].param_count()
    
    def activate_all(self):
        self.active = list(self.experts.values())
        self.total_params_activated = self.total_params_available
    
    def forward_all(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Run all active experts on the same input, return combined results"""
        results = {}
        for expert in self.active:
            results[expert.domain.value] = expert.forward(x)
        return results
    
    def get_active_rules(self) -> str:
        """Get all rules from active experts"""
        if not self.active:
            return "Aucun expert activé."
        lines = []
        for e in self.active:
            lines.append(f"\n[{e.domain.value.upper()}] {e.name} ({e.param_count()} params):")
            lines.append(e.get_rules())
        return "\n".join(lines)
    
    def describe(self) -> str:
        lines = [f"REGISTRE D'EXPERTS: {len(self.experts)} experts, {self.total_params_available} params totaux"]
        lines.append(f"  Actifs: {len(self.active)} experts, {self.total_params_activated} params")
        lines.append(f"  Économie: {100 * (1 - self.total_params_activated / max(self.total_params_available, 1)):.0f}% de params inactifs")
        return "\n".join(lines)


def build_default_registry() -> ExpertRegistry:
    """Build all default micro-experts with domain physics rules"""
    registry = ExpertRegistry()
    
    registry.register(MicroExpert(
        ExpertDomain.MOMENTUM, "Expert Élan/Collisions",
        ["collision", "choc", "élan", "momentum", "impulsion", "recul", "percussion", "impact",
         "percute", "heurte", "cogne", "frappe", "tape", "rentre dans", "carambolage"],
        rules=[
            "p = mv — l'élan se conserve en système fermé",
            "Collision élastique: Ek conservée, v1' = ((m1-m2)v1+2m2v2)/(m1+m2)",
            "Collision inélastique: objets fusionnent, v_final = (m1v1+m2v2)/(m1+m2)",
            "Impulsion J = Δp = ∫Fdt ≈ F·Δt",
            "Masses égales + élastique → échange complet des vitesses",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.INERTIA, "Expert Inertie Rotationnelle",
        ["inertie", "rotation", "gyroscopique", "moment inertie", "angulaire", "toupie", "roule", "I=", "tourne", "tournoie", "pivote", "vrille"],
        rules=[
            "I = ∫r²dm — résistance à l'accélération angulaire",
            "Sphère: I=2/5mr² (0.40), Cylindre: I=1/2mr² (0.50), Anneau: I=mr² (1.00)",
            "L = Iω est conservé sans couple externe",
            "Patineur: I↓ → ω↑ (L constant)",
            "Sur plan incliné: a = gsinθ/(1+I/mr²) — plus I/mr² est petit, plus ça accélère",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.TORQUE, "Expert Couple/Moment",
        ["couple", "torque", "levier", "moment", "équilibre rotatif", "pivot", "τ", "bras de levier"],
        rules=[
            "τ = r × F = rF sin(θ) — couple = force × bras de levier",
            "Équilibre: Στ = 0 autour de tout pivot",
            "Couple de forces: deux forces égales, opposées, parallèles → M = Fd",
            "Levier classe 1: pivot au milieu (balançoire) — MA peut être >1 ou <1",
            "Levier classe 2: charge au milieu (brouette) — MA toujours >1",
            "Levier classe 3: force au milieu (pince) — MA toujours <1, gain de vitesse",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.SPATIAL, "Expert Spatial/Zonage",
        ["dedans", "dehors", "proche", "loin", "entre", "zone", "distance", "gauche", "droite", "près de"],
        rules=[
            "Dedans/dehors: test de point dans polygone ou distance < rayon",
            "Proche: dist < rayon_near, Loin: dist > rayon_far",
            "Entre: projection du point sur le segment AB, 0 < t < 1",
            "Clusters: groupe d'objets dont les distances mutuelles < seuil",
            "Approche/éloignement: signe de d(dist)/dt",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.THERMAL, "Expert Thermique",
        ["chaleur", "température", "fusion", "brûle", "refroidir", "thermique", "degré", "feu",
         "chaud", "froid", "huile chaude", "ébullition", "congélation", "incendie", "flamme"],
        rules=[
            "Q = mcΔT — chaleur = masse × capacité × variation température",
            "Fusion: solide → liquide à T_fusion (absorbe chaleur latente)",
            "Conduction: Q/t = kAΔT/d (Fourier)",
            "Rayonnement: P = εσAT⁴ (Stefan-Boltzmann)",
            "Refroidissement: dT/dt ∝ (T - T_ambiante) (Newton)",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.GRAVITY, "Expert Gravitation",
        ["gravité", "poids", "orbite", "pesanteur", "attraction", "chute", "g=", "newton",
         "lâche", "tombe", "lancer", "jeter", "étage", "hauteur", "sol", "atterrit"],
        rules=[
            "F = Gm₁m₂/r² — tout attire tout",
            "Poids = mg — le poids change avec g, pas la masse",
            "g_terre=9.81, g_lune=1.62, g_mars=3.72 m/s²",
            "v_orb = √(GM/r) — vitesse orbitale circulaire",
            "v_esc = √(2GM/r) — vitesse de libération",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.KINEMATIC, "Expert Cinématique",
        ["mécanisme", "bielle", "manivelle", "piston", "dof", "mobilité", "workspace"],
        rules=[
            "Grübler: M = 3(N-1-J) + Σfᵢ en 2D",
            "M=1: mouvement déterminé, M<0: hyperstatique, M>0: sous-contraint",
            "Espace de travail: lieu des points atteignables par l'effecteur",
            "Singularité: perte de degré de liberté (blocage)",
            "Transmission: avantage mécanique = produit des rapports de bras de levier",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.ENERGY, "Expert Énergie",
        ["énergie", "cinétique", "potentielle", "conservation", "joule", "travail", "puissance"],
        rules=[
            "E_méca = E_cinétique + E_potentielle (conservée sans frottement)",
            "E_cinétique = ½mv², E_potentielle = mgh",
            "Travail W = F·d·cos(θ) — transfert d'énergie par une force",
            "Puissance P = W/t = F·v",
            "Rendement η = E_utile / E_fournie < 1 (toujours)",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.FLUID, "Expert Fluides",
        ["fluide", "liquide", "viscosité", "pression", "archimède", "débit", "écoulement", "huile", "eau",
         "immergé", "flotte", "coule", "noyade", "submergé", "bulle", "vague"],
        rules=[
            "Archimède: poussée = ρ_fluide × V_immergé × g",
            "Viscosité: résistance à l'écoulement (eau=0.001, huile=0.8 Pa·s)",
            "Reynolds: Re = ρvD/μ — laminaire < 2300, turbulent > 4000",
            "Bernoulli: P + ½ρv² + ρgh = constant le long d'une ligne de courant",
            "Débit volumique Q = A·v (conservé en incompressible)",
        ]
    ))
    
    registry.register(MicroExpert(
        ExpertDomain.CAUSAL, "Expert Causalité",
        ["cause", "effet", "cascade", "dépendance", "si", "alors", "conséquence", "chaîne",
         "supporter", "planche", "poutre", "solide", "résiste", "cède", "casse", "soutient"],
        rules=[
            "Graphe causal: nœuds = événements, arêtes = cause→effet",
            "Support: si A soutient B, supprimer A → B tombe",
            "Contrefactuel: 'Que se passerait-il si X n'existait pas ?'",
            "Cascade: propager les conséquences en largeur d'abord",
            "Point faible: nœud avec le plus de dépendants sortants",
        ]
    ))
    
    return registry


# ============================================================
# 3. DEMO — Torque + Experts
# ============================================================

def demo_torque():
    """Demonstrate torque analysis"""
    print("=" * 60)
    print("  ANALYSE DE COUPLE/TORQUE")
    print("=" * 60)
    
    # Wrench example
    print("\n1. CLÉ À MOLETTE — Couple sur un écrou")
    ta = TorqueAnalyzer()
    # Two fingers at 3cm from center pushing opposite directions
    ta.add_force(ForceApplication("pouce_haut", 50, (0, 1), (0.03, 0)))     # Thumb up at 3cm right
    ta.add_force(ForceApplication("index_bas", 50, (0, -1), (-0.03, 0)))    # Index down at 3cm left
    
    pivot = (0, 0)
    torque = ta.net_torque_about(pivot)
    print(f"  Force de 50N à 2cm du centre → couple = {torque:.1f} N·m")
    print(f"  Équivalent à une force de {(torque/0.3):.0f}N avec bras de levier de 30cm")
    
    # Seesaw
    print("\n2. BALANÇOIRE — Équilibre des moments")
    ta2 = TorqueAnalyzer()
    ta2.add_force(ForceApplication("enfant_A", 300, (0, -1), (-2, 0)))  # 30kg at -2m
    ta2.add_force(ForceApplication("enfant_B", 200, (0, -1), (3, 0)))   # 20kg at 3m
    # Where to put enfant_B to balance? τ_A + τ_B = 0 → 300*2 = 200*x → x=3
    
    net = ta2.net_torque_about(pivot)
    print(f"  Enfant A (300N @ -2m): τ = {300*2} N·m")
    print(f"  Enfant B (200N @ 3m): τ = {200*3} N·m")
    print(f"  Net: {net:.0f} N·m → {'équilibré' if abs(net) < 1 else 'déséquilibré vers la ' + ('droite' if net > 0 else 'gauche')}")
    
    # Lever classification
    print("\n3. CLASSIFICATION DES LEVIERS")
    # Seesaw: pivot in middle
    cls1 = ta2.lever_classification("enfant_A", (2, 0), (0, 0))
    print(f"  Balançoire: {cls1}")
    
    # Wheelbarrow: load in middle
    ta3 = TorqueAnalyzer()
    ta3.add_force(ForceApplication("mains", 200, (0, 1), (1.5, 0)))
    cls2 = ta3.lever_classification("mains", (0.3, 0), (0, 0))
    print(f"  Brouette: {cls2}")
    
    # Tweezers: force in middle
    ta4 = TorqueAnalyzer()
    ta4.add_force(ForceApplication("doigts", 5, (0, -1), (0.04, 0)))
    cls3 = ta4.lever_classification("doigts", (0.1, 0), (0, 0))
    print(f"  Pince à épiler: {cls3}")


def demo_experts():
    """Demonstrate micro-NN expert activation"""
    print("\n\n" + "=" * 60)
    print("  SYSTÈME MICRO-NN EXPERTS")
    print("=" * 60)
    
    registry = build_default_registry()
    
    # Test queries
    queries = [
        "une bille en acier percute une bille en verre",
        "je lâche une balle du 5e étage, combien de temps pour toucher le sol ?",
        "est-ce que cette planche est assez solide pour supporter 200kg ?",
        "de l'huile à 200°C touche de l'eau froide",
    ]
    
    for query in queries:
        registry.activate_by_query(query)
        print(f"\n🔍 Query: \"{query}\"")
        print(f"   → {len(registry.active)} experts activés ({registry.total_params_activated} params):")
        for e in registry.active:
            print(f"      [{e.domain.value}] {e.name}")
    
    # Resource comparison
    print(f"\n{'─'*60}")
    print(registry.describe())
    print(f"   Pour comparaison: un petit LLM fait ~100M params.")
    print(f"   Ce système: {registry.total_params_available} params, ")
    print(f"   seuls {50}% en moyenne sont actifs → ~{registry.total_params_available // (len(registry.experts) * 2)} params en pratique.")


if __name__ == "__main__":
    demo_torque()
    demo_experts()
