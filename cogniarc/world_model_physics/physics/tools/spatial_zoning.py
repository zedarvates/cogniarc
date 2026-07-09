"""
Spatial Zoning Engine — Inside/outside, near/far, directional zones.
Converts vague spatial concepts into crisp computable zones for small LLMs.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum


# ============================================================
# 1. ZONE TYPES — All spatial relationships formalized
# ============================================================

class ZoneType(Enum):
    INSIDE = "inside"           # Strictly contained within boundary
    OUTSIDE = "outside"         # Completely outside
    BOUNDARY = "boundary"       # On the edge (within epsilon)
    NEAR = "near"               # Within near-radius
    FAR = "far"                 # Beyond far-radius
    MIDFIELD = "midfield"       # Between near and far
    ABOVE = "above"             # Higher Y
    BELOW = "below"             # Lower Y
    LEFT = "left"               # Negative X
    RIGHT = "right"             # Positive X
    IN_FRONT = "in_front"       # Positive Z (3D)
    BEHIND = "behind"           # Negative Z (3D)
    BETWEEN = "between"         # Between two reference points
    ALONGSIDE = "alongside"     # Parallel, same axis
    OPPOSITE = "opposite"       # Facing each other
    SURROUNDING = "surrounding" # Encircling (all sides)
    ISOLATED = "isolated"       # No neighbors within far-radius
    CLUSTERED = "clustered"     # Multiple objects close together
    APPROACHING = "approaching" # Getting closer over time
    RECEDING = "receding"       # Getting farther over time


@dataclass
class SpatialZone:
    """A defined spatial region"""
    name: str
    zone_type: ZoneType
    center: Tuple[float, float] = (0, 0)
    radius_inner: float = 0.0    # Inside/containment radius
    radius_near: float = 1.0     # "Near" threshold
    radius_far: float = 5.0      # "Far" threshold
    boundary_thickness: float = 0.1  # Epsilon for boundary detection
    
    def classify_point(self, point: Tuple[float, float]) -> ZoneType:
        """Classify a single point relative to this zone"""
        px, py = point
        cx, cy = self.center
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        
        if dist <= self.radius_inner + self.boundary_thickness:
            if dist <= self.radius_inner - self.boundary_thickness:
                return ZoneType.INSIDE
            return ZoneType.BOUNDARY
        
        if dist <= self.radius_near:
            return ZoneType.NEAR
        elif dist <= self.radius_far:
            return ZoneType.MIDFIELD
        else:
            return ZoneType.FAR
    
    def membership_strength(self, point: Tuple[float, float]) -> Dict[ZoneType, float]:
        """Fuzzy membership: how much does this point belong to each zone? 0-1"""
        px, py = point
        cx, cy = self.center
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        
        memberships = {}
        
        # Inside (crisp at boundary)
        if dist <= self.radius_inner:
            memberships[ZoneType.INSIDE] = 1.0
        elif dist <= self.radius_inner + self.boundary_thickness:
            ratio = 1.0 - (dist - self.radius_inner) / self.boundary_thickness
            memberships[ZoneType.BOUNDARY] = 1.0 - ratio
            memberships[ZoneType.INSIDE] = ratio
        
        # Near/midfield/far (fuzzy transitions)
        if dist <= self.radius_near:
            near_strength = 1.0 - (dist / max(self.radius_near, 0.01))
            memberships[ZoneType.NEAR] = max(0, near_strength)
        elif dist <= self.radius_far:
            mid_strength = 1.0 - abs(dist - (self.radius_near + self.radius_far)/2) / \
                          max((self.radius_far - self.radius_near)/2, 0.01)
            memberships[ZoneType.MIDFIELD] = max(0, min(1, mid_strength))
        else:
            far_strength = min(1.0, (dist - self.radius_far) / self.radius_far)
            memberships[ZoneType.FAR] = max(0, far_strength)
        
        # Directional (crisp)
        dx, dy = px - cx, py - cy
        angle = math.atan2(dy, dx)
        
        # Above/below
        if abs(dy) > abs(dx):  # Vertical dominant
            if dy > 0:
                memberships[ZoneType.ABOVE] = abs(dy) / max(abs(dy) + abs(dx), 0.01)
            else:
                memberships[ZoneType.BELOW] = abs(dy) / max(abs(dy) + abs(dx), 0.01)
        else:  # Horizontal dominant
            if dx > 0:
                memberships[ZoneType.RIGHT] = abs(dx) / max(abs(dy) + abs(dx), 0.01)
            else:
                memberships[ZoneType.LEFT] = abs(dx) / max(abs(dy) + abs(dx), 0.01)
        
        return memberships


# ============================================================
# 2. SPATIAL ANALYZER — Multi-object, multi-zone
# ============================================================

class SpatialAnalyzer:
    """
    Full spatial analysis of a scene.
    Answers: What's inside? What's near? What's approaching?
    """
    
    def __init__(self):
        self.objects: Dict[str, dict] = {}     # id → {pos, vel, tags, ...}
        self.zones: List[SpatialZone] = []
        self.reference_objects: List[str] = []  # Objects used as zone centers
        self.history: List[Dict] = []           # For approach/recede detection
        self.max_history = 30                   # 0.5s at 60fps
    
    def add_object(self, obj_id: str, position: Tuple[float, float], 
                   velocity: Tuple[float, float] = (0, 0),
                   radius: float = 0.5, tags: List[str] = None):
        self.objects[obj_id] = {
            "pos": position, "vel": velocity, "radius": radius,
            "tags": tags or [], "zone": None
        }
    
    def add_zone(self, zone: SpatialZone):
        self.zones.append(zone)
    
    def auto_zone_from_objects(self, center_obj_id: str, 
                                radius_near: float = 2.0, radius_far: float = 8.0):
        """Create a zone centered on an object"""
        obj = self.objects.get(center_obj_id)
        if not obj:
            return
        zone = SpatialZone(
            f"zone_{center_obj_id}",
            ZoneType.SURROUNDING,
            obj["pos"],
            radius_inner=obj["radius"],
            radius_near=radius_near,
            radius_far=radius_far
        )
        self.zones.append(zone)
        self.reference_objects.append(center_obj_id)
    
    def classify_all(self) -> Dict[str, dict]:
        """Classify every object relative to every zone"""
        results = {}
        
        for obj_id, obj in self.objects.items():
            classifications = []
            dominant_zone = None
            dominant_strength = 0
            
            for zone in self.zones:
                # Skip self-classification
                if zone.name == f"zone_{obj_id}":
                    continue
                
                primary = zone.classify_point(obj["pos"])
                memberships = zone.membership_strength(obj["pos"])
                
                classifications.append({
                    "zone": zone.name,
                    "primary": primary.value,
                    "memberships": {k.value: round(v, 3) for k, v in memberships.items()},
                    "distance": round(math.sqrt(
                        (obj["pos"][0] - zone.center[0])**2 + 
                        (obj["pos"][1] - zone.center[1])**2
                    ), 2)
                })
                
                # Track dominant zone
                strength = memberships.get(primary, 0)
                if strength > dominant_strength:
                    dominant_strength = strength
                    dominant_zone = zone.name
            
            results[obj_id] = {
                "classifications": classifications,
                "dominant_zone": dominant_zone,
                "dominant_zone_strength": round(dominant_strength, 3)
            }
        
        return results
    
    def detect_approaches(self) -> List[dict]:
        """Detect which objects are approaching/receding from each other"""
        if len(self.history) < 2:
            return []
        
        prev = self.history[-2]
        curr = self.history[-1]
        approaches = []
        
        for oid1 in prev:
            if oid1 not in curr:
                continue
            for oid2 in curr:
                if oid1 == oid2 or oid2 not in prev:
                    continue
                
                p1_prev, p2_prev = prev[oid1], prev[oid2]
                p1_curr, p2_curr = curr[oid1], curr[oid2]
                
                dist_prev = math.sqrt((p1_prev[0] - p2_prev[0])**2 + (p1_prev[1] - p2_prev[1])**2)
                dist_curr = math.sqrt((p1_curr[0] - p2_curr[0])**2 + (p1_curr[1] - p2_curr[1])**2)
                
                delta = dist_curr - dist_prev
                if abs(delta) < 0.01:
                    continue
                
                approaches.append({
                    "objects": [oid1, oid2],
                    "type": "approaching" if delta < 0 else "receding",
                    "delta": round(delta, 4),
                    "current_distance": round(dist_curr, 2),
                    "rate": round(abs(delta) * 60, 2)  # m/s
                })
        
        return sorted(approaches, key=lambda a: -a["rate"])
    
    def record_snapshot(self):
        """Record current positions for approach detection"""
        snapshot = {oid: obj["pos"] for oid, obj in self.objects.items()}
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def between_analysis(self, obj_id: str, ref_a: str, ref_b: str) -> dict:
        """Is obj_id between ref_a and ref_b?"""
        obj = self.objects.get(obj_id)
        a = self.objects.get(ref_a)
        b = self.objects.get(ref_b)
        if not all([obj, a, b]):
            return {"between": False, "reason": "missing objects"}
        
        o = np.array(obj["pos"])
        pa = np.array(a["pos"])
        pb = np.array(b["pos"])
        
        # Project obj onto line AB
        ab = pb - pa
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-10:
            return {"between": False, "reason": "A and B at same position"}
        
        ao = o - pa
        t = np.dot(ao, ab) / (ab_len * ab_len)
        
        # Distance from line
        projection = pa + t * ab
        dist_from_line = np.linalg.norm(o - projection)
        total_len = ab_len
        betweenness = 1.0 - max(0, (dist_from_line / max(obj["radius"] * 3, 0.1)))
        betweenness = max(0, min(1, betweenness))
        
        return {
            "between": 0 < t < 1 and dist_from_line < obj["radius"] * 3,
            "projection_ratio": round(t, 3),  # 0 = at A, 1 = at B
            "distance_from_line": round(dist_from_line, 3),
            "betweenness": round(betweenness, 3),
            "closer_to": ref_a if t < 0.5 else ref_b
        }
    
    def cluster_detection(self, near_threshold: float = 2.0) -> List[Set[str]]:
        """Detect clusters of nearby objects"""
        remaining = set(self.objects.keys())
        clusters = []
        
        while remaining:
            seed = remaining.pop()
            cluster = {seed}
            stack = [seed]
            
            while stack:
                current = stack.pop()
                pos_c = np.array(self.objects[current]["pos"])
                
                for other in list(remaining):
                    pos_o = np.array(self.objects[other]["pos"])
                    dist = np.linalg.norm(pos_c - pos_o)
                    
                    r_c = self.objects[current]["radius"]
                    r_o = self.objects[other]["radius"]
                    
                    if dist < near_threshold + r_c + r_o:
                        cluster.add(other)
                        remaining.remove(other)
                        stack.append(other)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters


# ============================================================
# 3. LLM INTERFACE — Natural language spatial descriptions
# ============================================================

def analyze_spatial_scene(name: str, objects: List[dict], 
                          center_object: str = None,
                          near_radius: float = 2.0, far_radius: float = 8.0) -> str:
    """
    Full spatial analysis → LLM-readable description.
    
    objects: [{id, pos, vel?, radius?, tags?}]
    """
    analyzer = SpatialAnalyzer()
    
    for obj in objects:
        analyzer.add_object(
            obj["id"],
            tuple(obj.get("pos", [0, 0])),
            tuple(obj.get("vel", [0, 0])),
            obj.get("radius", 0.5),
            obj.get("tags", [])
        )
    
    # Auto-create zones from center object
    if center_object and center_object in [o["id"] for o in objects]:
        analyzer.auto_zone_from_objects(center_object, near_radius, far_radius)
    
    # Also create zones for objects tagged as "reference"
    for obj in objects:
        if "reference" in obj.get("tags", []):
            analyzer.auto_zone_from_objects(obj["id"], near_radius, far_radius)
    
    # Classify
    classifications = analyzer.classify_all()
    clusters = analyzer.cluster_detection()
    
    # Build output
    lines = [f"{'='*60}"]
    lines.append(f"  ANALYSE SPATIALE: {name}")
    lines.append(f"{'='*60}")
    lines.append(f"  {len(objects)} objets, {len(analyzer.zones)} zones, {len(clusters)} clusters")
    
    # Per-object spatial status
    lines.append(f"\nCLASSIFICATION PAR OBJET:")
    for obj_id, result in classifications.items():
        obj = analyzer.objects[obj_id]
        clas = result["classifications"]
        
        if not clas:
            zone_info = "isolé (aucune zone de référence)"
        else:
            primary = clas[0]
            zone_info = f"{primary['primary']} (dist={primary['distance']})"
        
        tags_str = f" [{', '.join(obj['tags'][:3])}]" if obj['tags'] else ""
        lines.append(f"  • {obj_id}{tags_str} → {zone_info}")
        
        # Detailed memberships
        if clas:
            for zone, strength in sorted(clas[0]["memberships"].items(), 
                                         key=lambda x: -x[1])[:3]:
                if strength > 0.1:
                    bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                    lines.append(f"      {zone:12s} [{bar}] {strength:.0%}")
    
    # Clusters
    if clusters:
        lines.append(f"\nGROUPES DÉTECTÉS:")
        for i, cluster in enumerate(clusters):
            lines.append(f"  Cluster {i+1}: {', '.join(sorted(cluster))}")
    
    # Between analysis
    if len(objects) >= 3:
        lines.append(f"\nRELATIONS 'ENTRE':")
        for i, o1 in enumerate(objects):
            for o2 in objects[i+1:]:
                # Check if any other object is between these two
                for o3 in objects:
                    if o3["id"] == o1["id"] or o3["id"] == o2["id"]:
                        continue
                    result = analyzer.between_analysis(o3["id"], o1["id"], o2["id"])
                    if result["between"]:
                        closer = result["closer_to"]
                        lines.append(f"  {o3['id']} est entre {o1['id']} et {o2['id']} (plus proche de {closer}, btwn={result['betweenness']:.0%})")
    
    # LLM summary
    lines.append(f"\n{'─'*60}")
    lines.append("RÉSUMÉ POUR LLM:")
    
    # Count by zone
    zone_counts = {}
    for obj_id, result in classifications.items():
        if result["dominant_zone"]:
            z = result["dominant_zone"]
            zone_counts[z] = zone_counts.get(z, 0) + 1
    
    for zone, count in sorted(zone_counts.items()):
        lines.append(f"  Dans {zone}: {count} objet(s)")
    
    # Inside/near/far summary
    inside = sum(1 for r in classifications.values() 
                 if r["classifications"] and r["classifications"][0]["primary"] == "inside")
    near = sum(1 for r in classifications.values()
               if r["classifications"] and r["classifications"][0]["primary"] == "near")
    far = sum(1 for r in classifications.values()
              if r["classifications"] and r["classifications"][0]["primary"] == "far")
    clustered = len(clusters)
    
    lines.append(f"\n  Vue d'ensemble: {inside} à l'intérieur, {near} proches, {far} loin, {clustered} groupes.")
    
    if clustered > 0:
        lines.append(f"  → Le LLM peut inférer: les objets forment {clustered} groupes distincts.")
    if far > len(objects) * 0.7:
        lines.append(f"  → Scène dispersée: majorité des objets éloignés les uns des autres.")
    if near > len(objects) * 0.5:
        lines.append(f"  → Scène dense: la plupart des objets sont proches.")
    
    return "\n".join(lines)


# ============================================================
# 4. PRE-BUILT SCENES
# ============================================================

def demo_spatial_analysis():
    """Demo with multiple scenes"""
    
    # Scene 1: Room with objects
    print("1. SCÈNE D'INTÉRIEUR — Pièce avec meubles")
    objects = [
        {"id": "table", "pos": [0, 0], "radius": 2.0, "tags": ["reference", "meuble", "grand"]},
        {"id": "chaise_1", "pos": [0, -1.5], "radius": 0.5, "tags": ["meuble", "petit"]},
        {"id": "chaise_2", "pos": [0, 1.5], "radius": 0.5, "tags": ["meuble", "petit"]},
        {"id": "personne", "pos": [0.5, -1.0], "radius": 0.4, "tags": ["humain", "assis"]},
        {"id": "lampe", "pos": [4, 3], "radius": 0.3, "tags": ["electrique", "loin"]},
        {"id": "fenetre", "pos": [4, -3], "radius": 1.0, "tags": ["ouverture", "loin"]},
    ]
    print(analyze_spatial_scene("Salle à manger", objects, "table", near_radius=1.5, far_radius=4.0))
    
    # Scene 2: Battlefield — units and zones
    print("\n\n2. CHAMP DE BATAILLE — Unités et zones")
    units = [
        {"id": "tank", "pos": [0, 0], "radius": 3.0, "vel": [2, 0], "tags": ["reference", "vehicule", "lourd"]},
        {"id": "soldat_1", "pos": [1, 0.5], "radius": 0.3, "vel": [2, 0], "tags": ["infanterie", "proche"]},
        {"id": "soldat_2", "pos": [0.5, -1], "radius": 0.3, "vel": [2, 1], "tags": ["infanterie", "proche"]},
        {"id": "bunker", "pos": [-2, 0], "radius": 1.5, "tags": ["structure", "defense"]},
        {"id": "ennemi_A", "pos": [7, 1], "radius": 0.4, "vel": [-1, 0], "tags": ["ennemi", "loin"]},
        {"id": "ennemi_B", "pos": [7.5, -0.5], "radius": 0.4, "vel": [-1, 0], "tags": ["ennemi", "loin"]},
        {"id": "ennemi_C", "pos": [-6, 5], "radius": 0.5, "tags": ["ennemi", "tres_loin"]},
    ]
    print(analyze_spatial_scene("Champ de bataille", units, "tank", near_radius=3.0, far_radius=8.0))
    
    # Scene 3: City — districts and buildings
    print("\n\n3. VILLE — Quartiers et bâtiments")
    buildings = [
        {"id": "mairie", "pos": [0, 0], "radius": 3.0, "tags": ["reference", "centre", "public"]},
        {"id": "ecole", "pos": [1, 2], "radius": 1.5, "tags": ["public", "proche"]},
        {"id": "pharmacie", "pos": [-1.5, 1], "radius": 0.8, "tags": ["commerce", "proche"]},
        {"id": "supermarché", "pos": [4, -1], "radius": 2.0, "tags": ["commerce", "moyen"]},
        {"id": "usine", "pos": [-5, -4], "radius": 4.0, "tags": ["industrie", "loin", "bruyant"]},
        {"id": "parc", "pos": [3, 3], "radius": 2.5, "tags": ["loisir", "moyen"]},
        {"id": "gare", "pos": [-4, 5], "radius": 1.5, "tags": ["transport", "loin"]},
    ]
    print(analyze_spatial_scene("Centre-ville", buildings, "mairie", near_radius=2.5, far_radius=6.0))


if __name__ == "__main__":
    demo_spatial_analysis()
