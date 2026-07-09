"""
Kinematic Analysis Engine — Mechanical relationships → possible movements.
Computes: mobility, workspaces, motion transmission, singularities.
Designed for LLM approximate reasoning about mechanical systems.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
import json
from ..simulator.physics import Vec2

# ============================================================
# 1. KINEMATIC ELEMENTS
# ============================================================

class JointType(Enum):
    REVOLUTE = "revolute"     # Rotation only (hinge)
    PRISMATIC = "prismatic"   # Translation only (slider)
    FIXED = "fixed"           # No movement (weld)
    PIN_SLOT = "pin_slot"     # Pin in slot (1 DOF, follows curve)
    UNIVERSAL = "universal"   # 2 rotational DOF
    BALL = "ball"             # 3 rotational DOF


@dataclass
class KinematicJoint:
    """A joint connecting two links with specific constraints"""
    id: str
    joint_type: JointType
    link_a: str                # First connected link
    link_b: str                # Second connected link
    position: Tuple[float, float]  # Joint position (world or relative)
    axis: Tuple[float, float] = (0, 0)  # Axis of rotation/translation
    limits: Tuple[float, float] = (-180, 180)  # Range of motion
    current_value: float = 0.0  # Current angle/displacement
    stiffness: float = 0.0      # 0 = free, >0 = spring
    damping: float = 0.0
    
    @property
    def dof_count(self) -> int:
        return {
            JointType.REVOLUTE: 1,
            JointType.PRISMATIC: 1,
            JointType.FIXED: 0,
            JointType.PIN_SLOT: 1,
            JointType.UNIVERSAL: 2,
            JointType.BALL: 3,
        }.get(self.joint_type, 0)
    
    @property
    def range_deg(self) -> float:
        return self.limits[1] - self.limits[0]


@dataclass 
class KinematicLink:
    """A rigid body in a kinematic chain"""
    id: str
    length: float = 1.0            # m
    joints: List[str] = field(default_factory=list)  # Joint IDs at ends
    mass: float = 1.0
    is_ground: bool = False        # Fixed to world frame
    is_input: bool = False          # Driven by motor/actuator
    is_output: bool = False         # End effector / tool point
    color: str = "#888888"


# ============================================================
# 2. MOBILITY & WORKSPACE
# ============================================================

class MobilityAnalyzer:
    """Grübler's formula and workspace computation for planar mechanisms"""
    
    def __init__(self):
        self.joints: List[KinematicJoint] = []
        self.links: List[KinematicLink] = []
        self.reachable_points: Dict[str, List[Tuple[float, float]]] = {}
    
    def add_joint(self, joint: KinematicJoint):
        self.joints.append(joint)
    
    def add_link(self, link: KinematicLink):
        self.links.append(link)
    
    def gruebler_mobility(self) -> int:
        """
        Grübler's formula for planar mechanisms:
        M = 3(N - 1 - J) + Σ f_i
        N = links, J = joints, f_i = DOF per joint
        """
        N = len(self.links)
        J = len(self.joints)
        total_dof = sum(j.dof_count for j in self.joints)
        M = 3 * (N - 1 - J) + total_dof
        return max(0, M)
    
    def classify_mechanism(self) -> str:
        """Classify mechanism type based on mobility"""
        M = self.gruebler_mobility()
        N = len(self.links)
        J = len(self.joints)
        
        if M == 1:
            return "Mécanisme à 1 degré de liberté — mouvement déterminé (chaque entrée → une sortie unique)"
        elif M == 0:
            return "Structure statique — aucun mouvement possible, système isostatique"
        elif M < 0:
            return f"Structure hyperstatique — {abs(M)} contraintes en trop, système surcontraint"
        elif M == 2:
            return "Mécanisme à 2 degrés de liberté — besoin de 2 actionneurs pour contrôle complet"
        else:
            return f"Mécanisme à {M} degrés de liberté — nécessite {M} actionneurs indépendants"
    
    def compute_workspace(self, link_id: str, resolution: int = 36) -> dict:
        """
        Brute-force workspace: sample all joint positions and collect reachable points.
        Only works for simple mechanisms (≤3 joints).
        """
        points = []
        movable_joints = [j for j in self.joints if j.joint_type != JointType.FIXED]
        
        if len(movable_joints) == 0:
            return {"points": [], "area": 0, "description": "Aucun joint mobile"}
        
        # Find the link
        link = next((l for l in self.links if l.id == link_id), None)
        if not link:
            return {"points": [], "area": 0, "description": "Link non trouvé"}
        
        # Get the chain from ground to this link
        chain = self._get_chain_to_link(link_id)
        
        if len(movable_joints) == 1:
            # Single pivot — workspace is an arc
            j = movable_joints[0]
            if j.joint_type == JointType.REVOLUTE:
                angles = np.linspace(j.limits[0], j.limits[1], resolution)
                for angle_deg in angles:
                    rad = np.radians(angle_deg)
                    points.append((
                        j.position[0] + link.length * np.cos(rad),
                        j.position[1] + link.length * np.sin(rad)
                    ))
            elif j.joint_type == JointType.PRISMATIC:
                disp = np.linspace(j.limits[0], j.limits[1], resolution)
                for d in disp:
                    points.append((
                        j.position[0] + d * j.axis[0],
                        j.position[1] + d * j.axis[1]
                    ))
        
        elif len(movable_joints) == 2:
            # 2-DOF arm — workspaces is an annulus
            j1, j2 = movable_joints[0], movable_joints[1]
            link1 = next((l for l in self.links if j1.link_b == l.id), None)
            link2 = next((l for l in self.links if j2.link_b == l.id), None)
            if link1 and link2:
                l1, l2 = link1.length, link2.length
                for a1 in np.linspace(j1.limits[0], j1.limits[1], resolution):
                    for a2 in np.linspace(j2.limits[0], j2.limits[1], max(4, resolution//3)):
                        rad1, rad2 = np.radians(a1), np.radians(a2)
                        x = j1.position[0] + l1 * np.cos(rad1) + l2 * np.cos(rad1 + rad2)
                        y = j1.position[1] + l1 * np.sin(rad1) + l2 * np.sin(rad1 + rad2)
                        points.append((x, y))
        
        # Compute convex hull area for workspace size
        area = 0
        if len(points) > 2:
            pts = np.array(points)
            hull_points = self._convex_hull(pts)
            if len(hull_points) > 2:
                area = self._polygon_area(hull_points)
        
        return {
            "points": points[:50],  # Keep manageable
            "area": round(area, 2),
            "resolution": len(points),
            "description": self._describe_workspace(area, link_id)
        }
    
    def _get_chain_to_link(self, link_id: str) -> List[str]:
        """Get joint/link chain from ground to target link"""
        chain = [link_id]
        current = link_id
        for _ in range(10):  # Max depth
            found = False
            for j in self.joints:
                if j.link_b == current:
                    chain.insert(0, j.link_a)
                    chain.insert(0, j.id)
                    current = j.link_a
                    found = True
                    break
                elif j.link_a == current and j.link_b:
                    chain.insert(0, j.link_b)
                    chain.insert(0, j.id)
                    current = j.link_b
                    found = True
                    break
            if not found:
                break
        return chain
    
    def _convex_hull(self, points: np.ndarray) -> np.ndarray:
        """Simple gift-wrapping for convex hull (small n)"""
        if len(points) < 3:
            return points
        hull = []
        leftmost = points[np.argmin(points[:, 0])]
        current = leftmost
        while True:
            hull.append(current)
            next_pt = points[0]
            for p in points[1:]:
                cross = (next_pt[0] - current[0]) * (p[1] - current[1]) - \
                        (next_pt[1] - current[1]) * (p[0] - current[0])
                if cross < 0 or (cross == 0 and 
                    ((p[0] - current[0])**2 + (p[1] - current[1])**2 > 
                     (next_pt[0] - current[0])**2 + (next_pt[1] - current[1])**2)):
                    next_pt = p
            current = next_pt
            if np.allclose(current, leftmost):
                break
        return np.array(hull)
    
    def _polygon_area(self, hull: np.ndarray) -> float:
        """Shoelace formula"""
        x, y = hull[:, 0], hull[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    def _describe_workspace(self, area: float, link_id: str) -> str:
        if area == 0:
            return f"{link_id}: aucun espace atteignable (mouvement nul ou linéaire)"
        elif area < 1:
            return f"{link_id}: petit espace de travail ({area:.1f}u²) — faible mobilité"
        elif area < 10:
            return f"{link_id}: espace moyen ({area:.0f}u²) — mobilité modérée"
        elif area < 50:
            return f"{link_id}: grand espace ({area:.0f}u²) — bonne couverture"
        else:
            return f"{link_id}: très grand espace ({area:.0f}u²) — couverture étendue"


# ============================================================
# 3. MOTION TRANSMISSION — How movement propagates
# ============================================================

class MotionTransmission:
    """Analyzes how motion flows through a kinematic chain"""
    
    def __init__(self, joints: List[KinematicJoint], links: List[KinematicLink]):
        self.joints = joints
        self.links = links
        self.transmission_paths: Dict[str, List[Tuple[str, float]]] = {}
    
    def compute_transmission(self) -> dict:
        """For each input link, trace motion transmission to all outputs"""
        results = {}
        inputs = [l for l in self.links if l.is_input]
        outputs = [l for l in self.links if l.is_output]
        
        for inp in inputs:
            paths = {}
            for out in outputs:
                chain = self._find_connection_chain(inp.id, out.id)
                if chain:
                    # Compute mechanical advantage along chain
                    advantage = self._compute_advantage(chain)
                    paths[out.id] = {
                        "chain": " → ".join(chain),
                        "mechanical_advantage": round(advantage, 2),
                        "type": self._transmission_type(advantage)
                    }
            if paths:
                results[inp.id] = paths
        
        return results
    
    def _find_connection_chain(self, from_id: str, to_id: str, 
                               visited: Set[str] = None, depth: int = 0) -> Optional[List[str]]:
        """BFS to find connection between two links"""
        if depth > 10:
            return None
        if visited is None:
            visited = set()
        
        if from_id == to_id:
            return [from_id]
        
        visited.add(from_id)
        
        for j in self.joints:
            if j.link_a == from_id and j.link_b not in visited:
                result = self._find_connection_chain(j.link_b, to_id, visited, depth + 1)
                if result:
                    return [from_id] + [j.id] + result
            elif j.link_b == from_id and j.link_a not in visited:
                result = self._find_connection_chain(j.link_a, to_id, visited, depth + 1)
                if result:
                    return [from_id] + [j.id] + result
        return None
    
    def _compute_advantage(self, chain: List[str]) -> float:
        """Simple mechanical advantage: ratio of output/input lever arms"""
        # Use link lengths ratio as proxy
        total = 1.0
        revolute_count = 0
        for item in chain:
            link = next((l for l in self.links if l.id == item), None)
            if link:
                revolute_count += 1
                if link.length > 0:
                    total *= link.length
        return total
    
    def _transmission_type(self, advantage: float) -> str:
        if advantage > 1.5:
            return "amplification"  # Force multiplication
        elif advantage < 0.67:
            return "réduction"      # Speed multiplication
        else:
            return "transmission directe"
    
    def predict_lock_positions(self) -> List[str]:
        """Find configurations where mechanism locks (singularities)"""
        locks = []
        
        # Check if any joint is at its limit
        for j in self.joints:
            if j.joint_type == JointType.REVOLUTE:
                if abs(j.current_value - j.limits[0]) < 1.0:
                    locks.append(f"{j.id}: proche de la butée basse ({j.limits[0]}°)")
                elif abs(j.current_value - j.limits[1]) < 1.0:
                    locks.append(f"{j.id}: proche de la butée haute ({j.limits[1]}°)")
        
        # Check for kinematic singularities (aligned links)
        for i, j1 in enumerate(self.joints):
            for j2 in self.joints[i+1:]:
                if j1.link_b == j2.link_a:
                    link = next((l for l in self.links if l.id == j1.link_b), None)
                    if link and not link.is_ground:
                        # Check if links are near aligned
                        pos1 = Vec2(*j1.position) if hasattr(j1, 'position') else None
                        pos2 = Vec2(*j2.position) if hasattr(j2, 'position') else None
                        if pos1 and pos2:
                            dx = pos2.x - pos1.x
                            dy = pos2.y - pos1.y
                            dist = (dx**2 + dy**2)**0.5
                            if dist > 1e-10:
                                length1 = link.length
                                length2 = next((l.length for l in self.links if l.id == j2.link_b), 1.0)
                                stretch = abs(dist - (length1 + length2)) / max(length1 + length2, 1e-10)
                                if stretch < 0.01:
                                    locks.append(f"{j1.id}↔{j2.id}: singularité — bras alignés (étirement={stretch:.3f})")
        
        return locks


# ============================================================
# 4. LLM INTERFACE
# ============================================================

def analyze_mechanism(name: str, joints: List[KinematicJoint], links: List[KinematicLink]) -> str:
    """Complete mechanical analysis for LLM consumption"""
    analyzer = MobilityAnalyzer()
    for j in joints:
        analyzer.add_joint(j)
    for l in links:
        analyzer.add_link(l)
    
    transmitter = MotionTransmission(joints, links)
    
    lines = [f"{'='*60}"]
    lines.append(f"  ANALYSE CINÉMATIQUE: {name}")
    lines.append(f"{'='*60}")
    
    # Mobility
    M = analyzer.gruebler_mobility()
    lines.append(f"\nMOBILITÉ: {M} degré(s) de liberté")
    lines.append(f"  {analyzer.classify_mechanism()}")
    
    # Joint summary
    lines.append(f"\nJOINTS ({len(joints)}):")
    for j in joints:
        limits = f" [{j.limits[0]:.0f}°..{j.limits[1]:.0f}°]" if j.joint_type == JointType.REVOLUTE else \
                 f" [{j.limits[0]:.1f}..{j.limits[1]:.1f}m]" if j.joint_type == JointType.PRISMATIC else ""
        lines.append(f"  {j.id}: {j.joint_type.value} entre {j.link_a}↔{j.link_b}{limits} (position={j.position})")
    
    # Workspace
    for l in links:
        if l.is_output:
            ws = analyzer.compute_workspace(l.id)
            lines.append(f"\nESPACE DE TRAVAIL ({l.id}):")
            lines.append(f"  {ws['description']}")
            lines.append(f"  Surface: {ws['area']:.1f}u² ({ws['resolution']} points échantillonnés)")
    
    # Motion transmission
    transmission = transmitter.compute_transmission()
    if transmission:
        lines.append(f"\nTRANSMISSION DE MOUVEMENT:")
        for inp_id, paths in transmission.items():
            lines.append(f"  Depuis {inp_id}:")
            for out_id, path in paths.items():
                icon = "🔺" if path["mechanical_advantage"] > 1.5 else "🔻" if path["mechanical_advantage"] < 0.67 else "➡"
                lines.append(f"    {icon} → {out_id}: {path['chain']} (avantage: {path['mechanical_advantage']:.1f}x, {path['type']})")
    
    # Lock positions
    locks = transmitter.predict_lock_positions()
    if locks:
        lines.append(f"\n⚠️ POSITIONS DE BLOCAGE:")
        for lock in locks:
            lines.append(f"  {lock}")
    
    # LLM summary
    lines.append(f"\n{'─'*60}")
    lines.append(f"RÉSUMÉ LLM:")
    controllable = len([l for l in links if l.is_input])
    useful = len([l for l in links if l.is_output])
    lines.append(f"  Ce mécanisme a {M} degré(s) de liberté avec {controllable} entrée(s) et {useful} sortie(s).")
    if M == controllable:
        lines.append(f"  ✅ Contrôle total: chaque degré de liberté a un actionneur.")
    elif M > controllable:
        lines.append(f"  ⚠️ Sous-actionné: {M - controllable} degré(s) non contrôlé(s).")
    else:
        lines.append(f"  ⚡ Sur-contraint: {controllable - M} actionneur(s) en trop → conflits possibles.")
    
    return "\n".join(lines)


# ============================================================
# 5. PRE-BUILT MECHANISMS
# ============================================================

def create_four_bar_linkage() -> Tuple[List[KinematicJoint], List[KinematicLink]]:
    """Classic 4-bar linkage (Grashof crank-rocker)"""
    joints = [
        KinematicJoint("A", JointType.REVOLUTE, "ground", "link1", (0, 0), (0, 0), (0, 360), 45),
        KinematicJoint("B", JointType.REVOLUTE, "link1", "link2", (2.0, 0), (0, 0), (-180, 180), 0),
        KinematicJoint("C", JointType.REVOLUTE, "link2", "link3", (4.0, 1.0), (0, 0), (-180, 180), 0),
        KinematicJoint("D", JointType.REVOLUTE, "link3", "ground", (3.0, 2.0), (0, 0), (0, 360), 120),
    ]
    links = [
        KinematicLink("ground", 0, ["A", "D"], 999, is_ground=True, color="#444"),
        KinematicLink("link1", 2.0, ["A", "B"], 1.0, is_input=True, color="#ff4444"),
        KinematicLink("link2", 3.0, ["B", "C"], 1.5, color="#44aaff"),
        KinematicLink("link3", 2.5, ["C", "D"], 1.0, is_output=True, color="#44ff44"),
    ]
    return joints, links


def create_slider_crank() -> Tuple[List[KinematicJoint], List[KinematicLink]]:
    """Piston-crank mechanism (engine)"""
    joints = [
        KinematicJoint("A", JointType.REVOLUTE, "ground", "crank", (0, 0), (0, 0), (0, 360), 0),
        KinematicJoint("B", JointType.REVOLUTE, "crank", "rod", (1.5, 0), (0, 0), (-180, 180), 0),
        KinematicJoint("C", JointType.REVOLUTE, "rod", "piston", (3.5, 0.5), (0, 0), (-30, 30), 0),
        KinematicJoint("D", JointType.PRISMATIC, "piston", "ground", (4.0, 0), (1, 0), (-0.5, 0.5), 0),
    ]
    links = [
        KinematicLink("ground", 0, ["A", "D"], 999, is_ground=True, color="#444"),
        KinematicLink("crank", 1.5, ["A", "B"], 2.0, is_input=True, color="#ff4444"),
        KinematicLink("rod", 3.0, ["B", "C"], 1.0, color="#44aaff"),
        KinematicLink("piston", 0.5, ["C", "D"], 1.0, is_output=True, color="#44ff44"),
    ]
    return joints, links


def create_robotic_arm_2dof() -> Tuple[List[KinematicJoint], List[KinematicLink]]:
    """2-DOF planar robotic arm"""
    joints = [
        KinematicJoint("shoulder", JointType.REVOLUTE, "base", "arm1", (0, 0), (0, 0), (-170, 170), 30),
        KinematicJoint("elbow", JointType.REVOLUTE, "arm1", "arm2", (2.0, 0.5), (0, 0), (-150, 150), -45),
    ]
    links = [
        KinematicLink("base", 0, ["shoulder"], 999, is_ground=True, color="#444"),
        KinematicLink("arm1", 2.0, ["shoulder", "elbow"], 3.0, is_input=True, color="#ff4444"),
        KinematicLink("arm2", 1.5, ["elbow"], 2.0, is_output=True, color="#44ff44"),
    ]
    return joints, links


def create_scissor_lift() -> Tuple[List[KinematicJoint], List[KinematicLink]]:
    """Scissor lift mechanism"""
    joints = [
        KinematicJoint("ground_left", JointType.REVOLUTE, "base", "arm_a1", (0, 0), (0, 0), (0, 180), 30),
        KinematicJoint("ground_slide", JointType.PRISMATIC, "base", "arm_b1", (2.0, 0), (1, 0), (0, 3), 0),
        KinematicJoint("cross", JointType.REVOLUTE, "arm_a1", "arm_b1", (1.0, 1.0), (0, 0), (-90, 90), 60),
        KinematicJoint("top_left", JointType.REVOLUTE, "arm_a1", "platform", (0, 2.0), (0, 0), (-45, 45), 30),
        KinematicJoint("top_slide", JointType.PRISMATIC, "arm_b1", "platform", (2.0, 2.0), (1, 0), (0, 3), 0),
    ]
    links = [
        KinematicLink("base", 0, ["ground_left", "ground_slide"], 999, is_ground=True, color="#444"),
        KinematicLink("arm_a1", 1.5, ["ground_left", "cross", "top_left"], 2.0, is_input=True, color="#ff4444"),
        KinematicLink("arm_b1", 1.5, ["ground_slide", "cross", "top_slide"], 2.0, color="#44aaff"),
        KinematicLink("platform", 0, ["top_left", "top_slide"], 5.0, is_output=True, color="#44ff44"),
    ]
    return joints, links


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    mechanisms = [
        ("4-BAR LINKAGE", create_four_bar_linkage()),
        ("SLIDER-CRANK (Piston)", create_slider_crank()),
        ("BRAS ROBOTIQUE 2-DOF", create_robotic_arm_2dof()),
        ("SCISSOR LIFT", create_scissor_lift()),
    ]
    
    for name, (joints, links) in mechanisms:
        print(analyze_mechanism(name, joints, links))
        print()
