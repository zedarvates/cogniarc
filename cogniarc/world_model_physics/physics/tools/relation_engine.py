"""
Relational Visualization Engine — Tensions, contraintes, moteurs, états.
Unified framework for LLM understanding of complex relational systems.
Applicable to physics, mechanics, and molecular dynamics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import json


# ============================================================
# 1. RELATION TYPES — Complete taxonomy of interactions
# ============================================================

class RelationCategory(Enum):
    """Categories of relationships between entities"""
    MECHANICAL = "mechanical"       # Forces, collisions, supports
    CONSTRAINT = "constraint"       # Joints, hinges, fixed connections
    THERMAL = "thermal"             # Heat transfer, conduction
    ELECTROMAGNETIC = "electromagnetic"  # Charges, magnets
    FLUID = "fluid"                 # Buoyancy, drag, pressure
    INFORMATIONAL = "informational" # Agent goals, signals
    CONTAINMENT = "containment"     # Inside/outside
    DEPENDENCY = "dependency"       # Causal chains


class ConstraintType(Enum):
    """Types of mechanical constraints"""
    FIXED = "fixed"              # No movement (welded)
    HINGE = "hinge"              # Rotate around point
    SLIDER = "slider"            # Linear movement only
    SPRING = "spring"            # Elastic connection
    ROPE = "rope"                # Distance limited (tension only)
    CONTACT = "contact"          # Surface contact
    GEAR = "gear"                # Rotational coupling
    PULLEY = "pulley"            # Direction reversal


class DOF(Enum):
    """Degrees of freedom"""
    FREE = "free"                    # Can move anywhere
    FIXED_ALL = "fixed_all"          # Completely immobilized
    TRANSLATE_X = "translate_x"      # Move in X only
    TRANSLATE_Y = "translate_y"      # Move in Y only
    ROTATE = "rotate"                # Rotate only
    TRANSLATE_XY = "translate_xy"    # Move in plane
    PLANAR = "planar"                # Move + rotate in plane


# ============================================================
# 2. TENSIONS & CAPACITIES — Motor and force capabilities
# ============================================================

@dataclass
class MotorCapacity:
    """What an object/agent can do"""
    body_id: str
    max_force: float = 0.0          # N (0 = passive)
    max_torque: float = 0.0         # N·m
    max_speed: float = 0.0          # m/s
    gripper: bool = False           # Can grab other objects
    sensor_range: float = 0.0       # How far it can "see"
    autonomous: bool = False        # Has own decision-making
    energy_source: str = "none"     # "battery", "fuel", "human", "gravity"
    state: str = "idle"             # Current activity state
    
    def can_act(self) -> bool:
        return self.max_force > 0 or self.max_torque > 0
    
    def can_perceive(self) -> bool:
        return self.sensor_range > 0


@dataclass
class Tension:
    """Force/stress on a connection"""
    source_id: str
    target_id: str
    force_magnitude: float          # N
    direction: Tuple[float, float]  # Unit vector
    stress_ratio: float             # 0-1, 1 = at breaking point
    type: str = "normal"            # "normal", "shear", "torsion"
    
    def is_critical(self) -> bool:
        return self.stress_ratio > 0.8
    
    def to_visual(self) -> dict:
        return {
            "source": self.source_id, "target": self.target_id,
            "force": round(self.force_magnitude, 1),
            "direction": [round(d, 2) for d in self.direction],
            "stress": round(self.stress_ratio * 100),
            "critical": self.is_critical()
        }


# ============================================================
# 3. RELATION NETWORK — The core visualization structure
# ============================================================

@dataclass
class RelationEdge:
    """A single relationship between two entities"""
    source: str
    target: str
    category: RelationCategory
    constraint_type: Optional[ConstraintType] = None
    tension: Optional[Tension] = None
    strength: float = 1.0          # 0-1 normalized intensity
    directed: bool = True           # Is it directional?
    label: str = ""                 # Human-readable label
    
    def to_llm_triple(self) -> str:
        """Subject-Predicate-Object triple for LLM"""
        pred = self.label or self.category.value
        if self.constraint_type:
            pred += f"({self.constraint_type.value})"
        if self.tension and self.tension.is_critical():
            pred += " [CRITICAL]"
        return f"({self.source}) --[{pred}]--> ({self.target})"
    
    def to_primitive(self) -> dict:
        """Convert to visual primitive"""
        color_map = {
            RelationCategory.MECHANICAL: "#ff4444",
            RelationCategory.CONSTRAINT: "#44aaff",
            RelationCategory.THERMAL: "#ff8800",
            RelationCategory.ELECTROMAGNETIC: "#aa44ff",
            RelationCategory.FLUID: "#44aaff",
            RelationCategory.INFORMATIONAL: "#44ff44",
            RelationCategory.CONTAINMENT: "#888888",
            RelationCategory.DEPENDENCY: "#ffaa00",
        }
        return {
            "source": self.source, "target": self.target,
            "category": self.category.value,
            "strength": round(self.strength, 2),
            "color": color_map.get(self.category, "#888"),
            "label": self.label,
            "tension_pct": round(self.tension.stress_ratio * 100) if self.tension else 0
        }


class RelationNetwork:
    """
    Complete relational graph:
    - Nodes = objects with states, motors, DOFs
    - Edges = typed relationships with tensions and constraints
    """
    
    def __init__(self, name: str = "system"):
        self.name = name
        self.nodes: Dict[str, dict] = {}          # id → {dof, motor, state, pos, ...}
        self.edges: List[RelationEdge] = []
        self.tensions: List[Tension] = []
        self.clusters: List[Set[str]] = []         # Groups of closely related objects
    
    def add_node(self, node_id: str, position: Tuple[float, float] = (0, 0),
                 dof: DOF = DOF.FREE, motor: MotorCapacity = None,
                 state: str = "idle", tags: List[str] = None):
        self.nodes[node_id] = {
            "pos": position,
            "dof": dof.value,
            "motor": motor or MotorCapacity(node_id),
            "state": state,
            "tags": tags or [],
            "can_act": (motor and motor.can_act()) or False,
            "can_perceive": (motor and motor.can_perceive()) or False,
        }
    
    def add_edge(self, edge: RelationEdge):
        self.edges.append(edge)
    
    def add_tension(self, tension: Tension):
        self.tensions.append(tension)
    
    def build_from_physics(self, world_state: dict, causal_graph=None):
        """Convert physics world to relation network"""
        bodies = world_state.get("bodies", [])
        
        # Nodes with DOF analysis
        for b in bodies:
            pos = tuple(b.get("pos", [0, 0]))
            btype = b.get("type", "dynamic")
            
            # Determine DOF
            if btype == "static":
                dof = DOF.FIXED_ALL
            elif "constraints" in b.get("id", ""):
                dof = DOF.ROTATE
            else:
                dof = DOF.FREE
            
            # Motor capacity
            motor = MotorCapacity(
                body_id=b["id"],
                max_force=10.0 if "agent" in b.get("id", "") else 0.0,
                autonomous="agent" in b.get("id", "") or btype == "dynamic",
                state="moving" if b.get("vel", [0, 0]) != [0, 0] else "idle"
            )
            
            self.add_node(b["id"], pos, dof, motor, motor.state, 
                         tags=[b.get("material", ""), btype])
        
        # Edges from physics relations
        contacts = world_state.get("contacts", 0)
        atmosphere = world_state.get("atmosphere", {})
        
        for i, b1 in enumerate(bodies):
            for b2 in bodies[i+1:]:
                pos1 = np.array(b1.get("pos", [0, 0]))
                pos2 = np.array(b2.get("pos", [0, 0]))
                dist = np.linalg.norm(pos2 - pos1)
                threshold = b1.get("radius", 0.5) + b2.get("radius", 0.5) + 1.0
                
                if dist < threshold:
                    # Mechanical contact — check per-pair distance
                    r1 = b1.get("radius", 0.5)
                    r2 = b2.get("radius", 0.5)
                    if dist < r1 + r2 + 0.02:  # Per-pair contact detection
                        delta = pos2 - pos1
                        direction = tuple(delta / max(np.linalg.norm(delta), 1e-10))
                        tension = Tension(b1["id"], b2["id"], dist * 10, direction, min(1.0, dist * 2))
                        self.add_edge(RelationEdge(b1["id"], b2["id"], RelationCategory.MECHANICAL,
                                                    ConstraintType.CONTACT, tension, dist / threshold,
                                                    label="contact"))
                    
                    # Containment (if one is inside another)
                    r1, r2 = b1.get("radius", 0.5), b2.get("radius", 0.5)
                    if r1 > r2 * 2 and dist < r1 - r2:
                        self.add_edge(RelationEdge(b1["id"], b2["id"], RelationCategory.CONTAINMENT,
                                                    strength=1.0, label="contient"))
                    elif r2 > r1 * 2 and dist < r2 - r1:
                        self.add_edge(RelationEdge(b2["id"], b1["id"], RelationCategory.CONTAINMENT,
                                                    strength=1.0, label="contient"))
        
        # Thermal edges
        if atmosphere:
            for b1 in bodies:
                for b2 in bodies:
                    if b1["id"] >= b2["id"]:
                        continue
                    pos1 = np.array(b1.get("pos", [0, 0]))
                    pos2 = np.array(b2.get("pos", [0, 0]))
                    dist = np.linalg.norm(pos2 - pos1)
                    if dist < 2.0:
                        self.add_edge(RelationEdge(b1["id"], b2["id"], RelationCategory.THERMAL,
                                                    strength=max(0, 1 - dist/2), label="conduction"))
        
        # Dependency edges from causal graph
        if causal_graph and hasattr(causal_graph, 'support_graph'):
            for supporter, supported_list in causal_graph.support_graph.items():
                for supported in supported_list:
                    if supporter in self.nodes and supported in self.nodes:
                        self.add_edge(RelationEdge(supporter, supported, RelationCategory.DEPENDENCY,
                                                    strength=1.0, label="soutient"))
        
        # Cluster detection (connected components)
        self._detect_clusters()
    
    def _detect_clusters(self):
        """Group nodes that are tightly connected"""
        visited = set()
        self.clusters = []
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            cluster = set()
            stack = [node_id]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                cluster.add(n)
                for edge in self.edges:
                    if edge.source == n and edge.target not in visited:
                        stack.append(edge.target)
                    elif edge.target == n and edge.source not in visited:
                        stack.append(edge.source)
            if len(cluster) > 1:
                self.clusters.append(cluster)
    
    def to_force_graph(self) -> dict:
        """Export as force-directed graph data (D3.js compatible)"""
        node_data = []
        for nid, n in self.nodes.items():
            node_data.append({
                "id": nid,
                "x": n["pos"][0],
                "y": n["pos"][1],
                "dof": n["dof"],
                "state": n["state"],
                "can_act": n["can_act"],
                "tags": n["tags"]
            })
        
        edge_data = [e.to_primitive() for e in self.edges]
        
        return {"nodes": node_data, "edges": edge_data, "clusters": [list(c) for c in self.clusters]}
    
    def to_ascii_network(self, width: int = 70) -> str:
        """ASCII visualization of the relation network"""
        lines = [f"{'='*width}"]
        lines.append(f"  RELATION NETWORK: {self.name}".center(width))
        lines.append(f"{'='*width}")
        
        # Group nodes by cluster
        for ci, cluster in enumerate(self.clusters):
            lines.append(f"\n  Cluster {ci+1}: {', '.join(sorted(cluster)[:5])}")
        
        # Tensions
        if self.tensions:
            lines.append(f"\n  TENSIONS:")
            for t in self.tensions:
                bar = "█" * int(t.stress_ratio * 20) + "░" * (20 - int(t.stress_ratio * 20))
                critical = "⚠️ " if t.is_critical() else "  "
                lines.append(f"    {critical}{t.source_id}→{t.target_id}: {t.force_magnitude:.1f}N [{bar}] {t.stress_ratio:.0%}")
        
        # DOF summary
        lines.append(f"\n  DEGREES OF FREEDOM:")
        dof_groups = {}
        for nid, n in self.nodes.items():
            dof = n["dof"]
            if dof not in dof_groups:
                dof_groups[dof] = []
            dof_groups[dof].append(nid)
        for dof, nids in dof_groups.items():
            lines.append(f"    {dof}: {', '.join(nids[:5])}")
        
        # Motor capacities
        motors = {nid: n for nid, n in self.nodes.items() if n["can_act"]}
        if motors:
            lines.append(f"\n  CAPACITÉS MOTRICES:")
            for nid, n in motors.items():
                m = n["motor"]
                details = []
                if m.max_force: details.append(f"force={m.max_force}N")
                if m.gripper: details.append("préhenseur")
                if m.autonomous: details.append("autonome")
                lines.append(f"    {nid}: {', '.join(details)} [{n['state']}]")
        
        # Edge summary
        lines.append(f"\n  RELATIONS ({len(self.edges)}):")
        categories = {}
        for e in self.edges:
            cat = e.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(e.to_llm_triple())
        
        for cat, triples in sorted(categories.items()):
            lines.append(f"    [{cat}] {len(triples)} connexions:")
            for t in triples[:3]:
                lines.append(f"      {t}")
            if len(triples) > 3:
                lines.append(f"      ... +{len(triples)-3} autres")
        
        lines.append(f"\n{'='*width}")
        return "\n".join(lines)
    
    def to_llm_context(self) -> str:
        """LLM-comprehensible structured context"""
        lines = [f"SYSTÈME: {self.name} — {len(self.nodes)} entités, {len(self.edges)} relations, {len(self.clusters)} clusters"]
        
        # What moves what
        constraining = [e for e in self.edges if e.category == RelationCategory.CONSTRAINT]
        if constraining:
            lines.append("\nContrôle mécanique:")
            for e in constraining:
                lines.append(f"  {e.source} --{e.constraint_type.value if e.constraint_type else '?'}--> {e.target}")
        
        # What depends on what
        deps = [e for e in self.edges if e.category == RelationCategory.DEPENDENCY]
        if deps:
            lines.append("\nDépendances structurelles:")
            for e in deps:
                lines.append(f"  Si {e.source} disparaît → {e.target} s'effondre")
        
        # Critical tensions
        critical = [t for t in self.tensions if t.is_critical()]
        if critical:
            lines.append("\n⚠️ Points de rupture imminents:")
            for t in critical:
                lines.append(f"  {t.source_id}↔{t.target_id}: {t.stress_ratio:.0%} de la charge max")
        
        return "\n".join(lines)


# ============================================================
# 4. MOLECULAR MODEL — Extending to atom-like systems
# ============================================================

class MolecularSystem(RelationNetwork):
    """
    Extends relation network to molecular dynamics.
    Atoms with bonds, angles, electrostatic interactions.
    """
    
    def __init__(self, name: str = "molecule"):
        super().__init__(name)
        self.bonds: List[Tuple[str, str, float]] = []  # (atom1, atom2, bond_order)
        self.angles: List[Tuple[str, str, str, float]] = []  # (a1, a2, a3, angle_deg)
        self.partial_charges: Dict[str, float] = {}
    
    def add_atom(self, atom_id: str, element: str, position: Tuple[float, float],
                 charge: float = 0.0, radius: float = 0.3):
        """Add an atom with element properties"""
        dof = DOF.FREE  # Atoms can move (in simulation)
        motor = MotorCapacity(atom_id, max_force=0, autonomous=False, 
                             energy_source="thermal", state="vibrating")
        self.add_node(atom_id, position, dof, motor, "vibrating",
                     tags=[element, f"charge={charge:+}"])

        self.partial_charges[atom_id] = charge
        
        # Electromagnetic relations between charged atoms
        for other_id, other_charge in self.partial_charges.items():
            if other_id == atom_id:
                continue
            if abs(charge) > 0.01 and abs(other_charge) > 0.01:
                is_attraction = charge * other_charge < 0
                self.add_edge(RelationEdge(
                    atom_id, other_id,
                    RelationCategory.ELECTROMAGNETIC,
                    strength=abs(charge * other_charge),
                    label="attraction" if is_attraction else "répulsion"
                ))
    
    def add_bond(self, a1: str, a2: str, bond_order: float = 1.0, 
                 bond_length: float = 1.0):
        """Add a chemical bond"""
        self.bonds.append((a1, a2, bond_order))
        # Bond = mechanical constraint
        self.add_edge(RelationEdge(a1, a2, RelationCategory.CONSTRAINT,
                                    ConstraintType.SPRING, strength=bond_order,
                                    label=f"liaison ({bond_order:.1f})"))
    
    def add_bond_angle(self, a1: str, a2: str, a3: str, angle_deg: float):
        """Add a bond angle constraint (e.g., H-O-H at 104.5°)"""
        self.angles.append((a1, a2, a3, angle_deg))
        self.add_edge(RelationEdge(a1, a3, RelationCategory.CONSTRAINT,
                                    ConstraintType.HINGE, 
                                    strength=1.0, label=f"angle {angle_deg:.0f}°"))
    
    def get_formula(self) -> str:
        """Molecular formula"""
        elements = {}
        for nid, n in self.nodes.items():
            elem = n["tags"][0] if n["tags"] else "?"
            elements[elem] = elements.get(elem, 0) + 1
        return "".join(f"{e}{c if c > 1 else ''}" for e, c in sorted(elements.items()))
    
    def predict_properties(self) -> dict:
        """Approximate molecular properties from structure"""
        n_atoms = len(self.nodes)
        n_bonds = len(self.bonds)
        total_charge = sum(self.partial_charges.values())
        
        # Very simple heuristics (intentionally approximate)
        is_polar = abs(total_charge) > 0.1
        has_charged = any(abs(c) > 0.5 for c in self.partial_charges.values())
        
        # Water-like: 3 atoms, 2 bonds, H-O-H geometry
        is_water_like = n_atoms == 3 and n_bonds == 2
        
        return {
            "formula": self.get_formula(),
            "atoms": n_atoms,
            "bonds": n_bonds,
            "total_charge": round(total_charge, 3),
            "is_polar": is_polar,
            "water_like": is_water_like,
            "predicted_phase": "liquid" if is_water_like else 
                              "gas" if n_atoms <= 2 else
                              "solid" if n_atoms > 10 else "unknown",
            "h_bond_donors": sum(1 for n in self.nodes.values() if "H" in n["tags"]),
            "h_bond_acceptors": sum(1 for n in self.nodes.values() 
                                   if "O" in n["tags"] or "N" in n["tags"]),
        }


# ============================================================
# 5. LLM REASONING LAYER — Synthesize everything
# ============================================================

class ReasoningEngine:
    """
    Takes the relation network and generates LLM-friendly insights:
    - What controls what?
    - Where are the weak points?
    - What would happen if...?
    - What are the emergent properties?
    """
    
    def __init__(self, network: RelationNetwork):
        self.network = network
    
    def what_controls_what(self) -> str:
        """Chain of control: who moves whom?"""
        constrainers = {}
        constrained = {}
        
        for e in self.network.edges:
            if e.category == RelationCategory.CONSTRAINT:
                constrainers[e.source] = constrainers.get(e.source, 0) + 1
                constrained[e.target] = constrained.get(e.target, 0) + 1
        
        lines = ["=== CONTRÔLE ==="]
        
        # Who has most influence?
        for nid, count in sorted(constrainers.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  {nid} contrôle {count} autre(s) objet(s)")
        
        # Who is most constrained?
        for nid, count in sorted(constrained.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  {nid} est contraint par {count} objet(s)")
        
        # Autonomous agents
        agents = [nid for nid, n in self.network.nodes.items() if n["can_act"]]
        if agents:
            lines.append(f"\n  AGENTS AUTONOMES: {', '.join(agents)}")
        
        return "\n".join(lines)
    
    def where_are_weak_points(self) -> str:
        """Identify structural weak points"""
        critical = [t for t in self.network.tensions if t.is_critical()]
        
        if not critical:
            return "Aucun point faible critique détecté."
        
        lines = [f"⚠️ {len(critical)} POINT(S) FAIBLE(S):"]
        for t in critical:
            lines.append(f"  {t.source_id}↔{t.target_id}: charge à {t.stress_ratio:.0%}")
        
        # What breaks if this fails?
        for t in critical[:2]:
            affected = set()
            for e in self.network.edges:
                if e.category == RelationCategory.DEPENDENCY and e.source in (t.source_id, t.target_id):
                    affected.add(e.target)
            if affected:
                lines.append(f"    → Si rupture: {', '.join(sorted(affected))} s'effondrent")
        
        return "\n".join(lines)
    
    def emergent_behavior(self) -> str:
        """What emerges from the interactions?"""
        lines = ["=== COMPORTEMENT ÉMERGENT ==="]
        
        n_nodes = len(self.network.nodes)
        n_edges = len(self.network.edges)
        n_clusters = len(self.network.clusters)
        n_agents = sum(1 for n in self.network.nodes.values() if n["can_act"])
        
        # Complexity assessment
        if n_clusters > 1:
            lines.append(f"  Système fragmenté en {n_clusters} groupes isolés.")
        else:
            lines.append(f"  Système connecté: tous les {n_nodes} éléments interagissent.")
        
        # Motion patterns
        moving = [nid for nid, n in self.network.nodes.items() if n["state"] == "moving"]
        if moving:
            lines.append(f"  {len(moving)}/{n_nodes} éléments en mouvement → comportement dynamique.")
        
        # Autonomy
        if n_agents > 0:
            lines.append(f"  {n_agents} agent(s) autonome(s) → prise de décision décentralisée.")
        
        # Phase prediction (for molecular)
        if isinstance(self.network, MolecularSystem):
            props = self.network.predict_properties()
            lines.append(f"  Molécule: {props['formula']}")
            lines.append(f"  Phase probable: {props['predicted_phase']}")
            lines.append(f"  Polaire: {props['is_polar']}")
        
        return "\n".join(lines)
    
    def full_analysis(self) -> str:
        """Complete analysis for LLM consumption"""
        sections = [
            self.network.to_llm_context(),
            self.what_controls_what(),
            self.where_are_weak_points(),
            self.emergent_behavior(),
        ]
        return "\n\n".join(sections)


# ============================================================
# DEMO — Build and test everything
# ============================================================

def demo_relation_network():
    """Build example systems and analyze them"""
    print("=" * 70)
    print("  RELATIONAL VISUALIZATION ENGINE")
    print("=" * 70)
    
    # === System 1: Mechanical (car + wall) ===
    print("\n1. SYSTÈME MÉCANIQUE: Voiture → mur")
    net = RelationNetwork("Crash Test")
    net.add_node("car", (0, 0), DOF.TRANSLATE_X, 
                MotorCapacity("car", max_force=500, autonomous=True, energy_source="fuel", state="moving"),
                "moving", ["vehicle", "steel"])
    net.add_node("wall", (5, 0), DOF.FIXED_ALL, None, "static", ["obstacle", "stone"])
    net.add_node("passenger", (0.5, 0.5), DOF.FREE,
                MotorCapacity("passenger", max_force=0, autonomous=False, state="passive"),
                "passive", ["human", "fragile"])
    
    net.add_edge(RelationEdge("car", "wall", RelationCategory.MECHANICAL,
                               ConstraintType.CONTACT,
                               Tension("car", "wall", 5000, (1, 0), 0.85),
                               0.9, label="impact"))
    net.add_edge(RelationEdge("car", "passenger", RelationCategory.CONTAINMENT,
                               strength=1.0, label="contient"))
    net.add_edge(RelationEdge("car", "passenger", RelationCategory.MECHANICAL,
                               strength=0.7, label="inertie_transférée"))
    net.add_edge(RelationEdge("wall", "car", RelationCategory.DEPENDENCY,
                               strength=1.0, label="arrête"))
    
    engine = ReasoningEngine(net)
    print(net.to_ascii_network())
    print(engine.full_analysis())
    
    # === System 2: Molecular (water H2O) ===
    print("\n\n2. SYSTÈME MOLÉCULAIRE: H₂O")
    mol = MolecularSystem("Eau")
    mol.add_atom("O", "O", (0, 0), charge=-0.66, radius=0.4)
    mol.add_atom("H1", "H", (0.76, 0.58), charge=+0.33, radius=0.2)
    mol.add_atom("H2", "H", (-0.76, 0.58), charge=+0.33, radius=0.2)
    mol.add_bond("O", "H1", 1.0, 0.96)
    mol.add_bond("O", "H2", 1.0, 0.96)
    mol.add_bond_angle("H1", "O", "H2", 104.5)
    
    mol_engine = ReasoningEngine(mol)
    print(mol.to_ascii_network())
    print(mol_engine.full_analysis())
    
    # === System 3: Multi-agent (swarm) ===
    print("\n\n3. SYSTÈME MULTI-AGENT: Essaim de drones")
    swarm = RelationNetwork("Essaim")
    for i in range(4):
        angle = i * np.pi / 2
        swarm.add_node(f"drone_{i+1}", (np.cos(angle)*2, np.sin(angle)*2),
                      DOF.FREE,
                      MotorCapacity(f"drone_{i+1}", max_force=20, max_speed=5, 
                                   autonomous=True, sensor_range=3, gripper=False,
                                   energy_source="battery", state="hovering"),
                      "hovering", ["drone", "autonomous"])
        # Drones see each other
        for j in range(i):
            swarm.add_edge(RelationEdge(f"drone_{i+1}", f"drone_{j+1}", 
                                       RelationCategory.INFORMATIONAL,
                                       strength=0.5, label="détection_mutuelle"))
            swarm.add_edge(RelationEdge(f"drone_{i+1}", f"drone_{j+1}",
                                       RelationCategory.ELECTROMAGNETIC,
                                       strength=0.2, label="évitement_collision"))
    
    swarm_engine = ReasoningEngine(swarm)
    print(swarm.to_ascii_network())
    print(swarm_engine.full_analysis())
    
    return net, mol, swarm


if __name__ == "__main__":
    demo_relation_network()
