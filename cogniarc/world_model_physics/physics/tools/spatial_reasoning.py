"""
Spatial Reasoning Extensions — Perception/Occlusion, Pathfinding, Scale Laws.
Three engines for LLM spatial intelligence.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum


# ============================================================
# 1. PERCEPTION & OCCLUSION — Visual fields, ray casting
# ============================================================

class VisibilityClass(Enum):
    VISIBLE = "visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    HIDDEN = "hidden"
    OUT_OF_RANGE = "out_of_range"
    BEHIND_OBSERVER = "behind_observer"


@dataclass
class Observer:
    """An entity with visual perception capabilities"""
    id: str
    position: Tuple[float, float]
    facing_angle: float = 0.0         # radians, 0 = right
    fov_angle: float = math.radians(120)  # Field of view (radians)
    max_range: float = 10.0           # How far they can see
    min_range: float = 0.1            # Blind spot near observer
    height: float = 1.7               # Eye height (for 3D occlusion)
    
    def is_in_fov(self, point: Tuple[float, float]) -> bool:
        """Is the point within the field of view?"""
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < self.min_range or dist > self.max_range:
            return False
        
        angle_to_target = math.atan2(dy, dx)
        angle_diff = abs((angle_to_target - self.facing_angle + math.pi) % (2 * math.pi) - math.pi)
        return angle_diff <= self.fov_angle / 2
    
    def angular_position(self, point: Tuple[float, float]) -> float:
        """Angle relative to facing direction (-1 to 1, normalized to FOV)"""
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        angle_to_target = math.atan2(dy, dx)
        angle_diff = (angle_to_target - self.facing_angle + math.pi) % (2 * math.pi) - math.pi
        return angle_diff / max(self.fov_angle / 2, 1e-10)


class OcclusionEngine:
    """Ray casting + occlusion analysis for visual perception"""
    
    def __init__(self):
        self.observers: List[Observer] = []
        self.occluders: List[dict] = []      # {id, pos, radius, height}
        self.targets: List[dict] = []        # {id, pos, radius, height}
    
    def add_observer(self, obs: Observer):
        self.observers.append(obs)
    
    def add_occluder(self, occluder_id: str, position: Tuple[float, float],
                     radius: float = 0.5, height: float = 2.0):
        self.occluders.append({"id": occluder_id, "pos": position, 
                               "radius": radius, "height": height})
    
    def add_target(self, target_id: str, position: Tuple[float, float],
                   radius: float = 0.5, height: float = 1.0):
        self.targets.append({"id": target_id, "pos": position,
                            "radius": radius, "height": height})
    
    def _ray_intersects_circle(self, origin: Tuple[float, float],
                                target: Tuple[float, float],
                                circle_center: Tuple[float, float],
                                circle_radius: float) -> bool:
        """Check if ray from origin to target intersects a circle"""
        ox, oy = origin
        tx, ty = target
        cx, cy = circle_center
        
        # Vector math: closest point on segment to circle center
        dx = tx - ox
        dy = ty - oy
        seg_len_sq = dx**2 + dy**2
        
        if seg_len_sq < 1e-10:
            return False
        
        # Project circle center onto line
        t = max(0, min(1, ((cx - ox) * dx + (cy - oy) * dy) / seg_len_sq))
        closest_x = ox + t * dx
        closest_y = oy + t * dy
        
        dist_sq = (closest_x - cx)**2 + (closest_y - cy)**2
        return dist_sq <= circle_radius**2
    
    def compute_visibility(self, observer: Observer) -> Dict[str, dict]:
        """Compute visibility of all targets for an observer"""
        results = {}
        
        for target in self.targets:
            # 1. Is target in FOV?
            if not observer.is_in_fov(target["pos"]):
                dist = math.sqrt((target["pos"][0]-observer.position[0])**2 + 
                                (target["pos"][1]-observer.position[1])**2)
                results[target["id"]] = {
                    "class": (VisibilityClass.OUT_OF_RANGE.value if dist > observer.max_range
                             else VisibilityClass.BEHIND_OBSERVER.value),
                    "distance": round(dist, 2),
                    "occluded_by": [],
                    "visibility_pct": 0,
                    "can_see": False,
                    "angular_position": round(observer.angular_position(target["pos"]), 2)
                }
                continue
            
            dist = math.sqrt((target["pos"][0]-observer.position[0])**2 + 
                           (target["pos"][1]-observer.position[1])**2)
            
            # 2. Ray cast to check occlusion
            occluded_by = []
            total_occlusion = 0.0
            
            for occluder in self.occluders:
                if occluder["id"] == target["id"]:
                    continue
                
                if self._ray_intersects_circle(observer.position, target["pos"],
                                                occluder["pos"], occluder["radius"]):
                    # Check height occlusion
                    obs_h = observer.height
                    occ_h = occluder.get("height", 2.0)
                    tgt_h = target.get("height", 1.0)
                    
                    # Simplified: if occluder is tall enough to block line of sight
                    if occ_h > min(obs_h, tgt_h):
                        # Compute how much of the target is occluded
                        occ_dist = math.sqrt((occluder["pos"][0]-observer.position[0])**2 + 
                                           (occluder["pos"][1]-observer.position[1])**2)
                        tgt_dist = dist
                        
                        # Angular size at distance
                        occ_angular = 2 * math.atan(occluder["radius"] / max(occ_dist, 0.1))
                        tgt_angular = 2 * math.atan(target["radius"] / max(tgt_dist, 0.1))
                        
                        occlusion_fraction = min(1.0, occ_angular / max(tgt_angular, 1e-10))
                        total_occlusion += occlusion_fraction * (1 - total_occlusion)
                        
                        occluded_by.append({
                            "by": occluder["id"],
                            "fraction": round(occlusion_fraction, 2)
                        })
            
            # 3. Classify
            vis_pct = max(0, 100 * (1 - total_occlusion))
            
            if vis_pct < 10:
                vclass = VisibilityClass.HIDDEN
            elif vis_pct < 70:
                vclass = VisibilityClass.PARTIALLY_OCCLUDED
            else:
                vclass = VisibilityClass.VISIBLE
            
            results[target["id"]] = {
                "class": vclass.value,
                "distance": round(dist, 2),
                "angular_position": round(observer.angular_position(target["pos"]), 2),
                "visibility_pct": round(vis_pct, 1),
                "occluded_by": occluded_by,
                "can_see": vclass != VisibilityClass.HIDDEN
            }
        
        return results
    
    def find_blind_spots(self, observer: Observer, resolution: int = 36) -> List[Dict]:
        """Find angular sectors where the observer can't see"""
        spots = []
        
        for i in range(resolution):
            angle = -observer.fov_angle/2 + i * observer.fov_angle / resolution
            world_angle = observer.facing_angle + angle
            
            # Cast ray
            ray_end = (
                observer.position[0] + math.cos(world_angle) * observer.max_range,
                observer.position[1] + math.sin(world_angle) * observer.max_range
            )
            
            # Check if any occluders block this ray
            blocked = False
            blocker_id = None
            closest_dist = observer.max_range
            
            for occ in self.occluders:
                if self._ray_intersects_circle(observer.position, ray_end,
                                                occ["pos"], occ["radius"]):
                    dist = math.sqrt((occ["pos"][0]-observer.position[0])**2 + 
                                    (occ["pos"][1]-observer.position[1])**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        blocked = True
                        blocker_id = occ["id"]
            
            if blocked:
                # Merge with previous spot if adjacent
                if spots and spots[-1]["end_angle"] == round(math.degrees(angle), 0) - 360/resolution:
                    spots[-1]["end_angle"] = round(math.degrees(angle), 0)
                elif not spots or spots[-1]["blocker"] != blocker_id:
                    spots.append({
                        "start_angle": round(math.degrees(angle), 0),
                        "end_angle": round(math.degrees(angle), 0),
                        "blocker": blocker_id,
                        "distance": round(closest_dist, 2)
                    })
        
        return spots
    
    def ambient_visibility_map(self, observer: Observer, 
                                grid_resolution: int = 20,
                                bounds: Tuple[float, float, float, float] = (-10, -10, 10, 10)) -> np.ndarray:
        """2D visibility map: 1=visible, 0=occluded"""
        xmin, ymin, xmax, ymax = bounds
        grid = np.ones((grid_resolution, grid_resolution))
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                x = xmin + (i + 0.5) * (xmax - xmin) / grid_resolution
                y = ymin + (j + 0.5) * (ymax - ymin) / grid_resolution
                
                if not observer.is_in_fov((x, y)):
                    grid[i, j] = 0
                    continue
                
                for occ in self.occluders:
                    if self._ray_intersects_circle(observer.position, (x, y),
                                                    occ["pos"], occ["radius"]):
                        grid[i, j] = 0
                        break
        
        return grid


# ============================================================
# 2. PATHFINDING — A* avec contraintes physiques
# ============================================================

@dataclass
class GridNode:
    x: int
    y: int
    g: float = float('inf')    # Cost from start
    h: float = 0.0             # Heuristic to goal
    f: float = float('inf')    # g + h
    parent: Optional[Tuple[int, int]] = None
    blocked: bool = False
    clearance: float = 0.0     # Distance to nearest obstacle
    
    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


class Pathfinder:
    """A* pathfinding with physics constraints (clearance, dynamic obstacles)"""
    
    def __init__(self, width: int, height: int, cell_size: float = 1.0):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], GridNode] = {}
        self.obstacles: List[dict] = []  # {pos, radius, velocity?}
        self.dynamic_obstacles: List[dict] = []
        
        # Initialize grid
        for x in range(width):
            for y in range(height):
                self.grid[(x, y)] = GridNode(x, y)
    
    def add_obstacle(self, pos: Tuple[float, float], radius: float = 1.0,
                     velocity: Tuple[float, float] = None):
        """Mark cells blocked by an obstacle"""
        cell_x = int(pos[0] / self.cell_size)
        cell_y = int(pos[1] / self.cell_size)
        cells_radius = int(math.ceil(radius / self.cell_size)) + 1
        
        for dx in range(-cells_radius, cells_radius + 1):
            for dy in range(-cells_radius, cells_radius + 1):
                cx, cy = cell_x + dx, cell_y + dy
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    # Check actual distance from cell center to obstacle
                    cell_center = ((cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size)
                    dist = math.sqrt((cell_center[0] - pos[0])**2 + (cell_center[1] - pos[1])**2)
                    if dist <= radius:
                        self.grid[(cx, cy)].blocked = True
        
        if velocity and (abs(velocity[0]) > 0.01 or abs(velocity[1]) > 0.01):
            self.dynamic_obstacles.append({"pos": pos, "radius": radius, "velocity": velocity})
        else:
            self.obstacles.append({"pos": pos, "radius": radius})
    
    def compute_clearance(self):
        """Compute distance to nearest obstacle for each free cell (for robot size)"""
        obstacle_centers = np.array([o["pos"] for o in self.obstacles + self.dynamic_obstacles])
        obstacle_radii = np.array([o["radius"] for o in self.obstacles + self.dynamic_obstacles])
        
        for (x, y), node in self.grid.items():
            if node.blocked:
                node.clearance = 0
                continue
            
            cell_center = np.array([(x + 0.5) * self.cell_size, (y + 0.5) * self.cell_size])
            
            min_dist = float('inf')
            for i in range(len(obstacle_centers)):
                dist = np.linalg.norm(cell_center - obstacle_centers[i]) - obstacle_radii[i]
                min_dist = min(min_dist, dist)
            
            node.clearance = max(0, round(min_dist, 2))
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Octile distance (allows diagonal movement)"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
    
    def _get_neighbors(self, pos: Tuple[int, int], 
                       robot_radius_cells: int = 0) -> List[Tuple[int, int]]:
        """Get valid neighbors (8-connected), respecting robot size"""
        x, y = pos
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    node = self.grid[(nx, ny)]
                    if not node.blocked and node.clearance >= robot_radius_cells * self.cell_size:
                        cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                        neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def find_path(self, start_world: Tuple[float, float], goal_world: Tuple[float, float],
                  robot_radius: float = 0.5, avoid_dynamic: bool = True,
                  future_steps: int = 10) -> dict:
        """
        A* pathfinding from start to goal.
        Returns path with physics constraints.
        """
        start = (int(start_world[0] / self.cell_size), int(start_world[1] / self.cell_size))
        goal = (int(goal_world[0] / self.cell_size), int(goal_world[1] / self.cell_size))
        
        # Validate
        if start not in self.grid or goal not in self.grid:
            return {"found": False, "reason": "Start or goal out of bounds"}
        
        if self.grid[start].blocked:
            return {"found": False, "reason": "Start position is blocked"}
        
        if self.grid[goal].blocked:
            return {"found": False, "reason": "Goal is blocked"}
        
        # Predict dynamic obstacle positions
        robot_r_cells = int(math.ceil(robot_radius / self.cell_size))
        
        # Initialize A*
        open_set: Set[Tuple[int, int]] = {start}
        closed_set: Set[Tuple[int, int]] = set()
        
        self.grid[start].g = 0
        self.grid[start].h = self._heuristic(start, goal)
        self.grid[start].f = self.grid[start].h
        
        while open_set:
            # Find lowest f-score
            current = min(open_set, key=lambda p: self.grid[p].f)
            
            if current == goal:
                # Reconstruct path
                path = []
                p = current
                while p is not None:
                    wx = (p[0] + 0.5) * self.cell_size
                    wy = (p[1] + 0.5) * self.cell_size
                    path.append((round(wx, 2), round(wy, 2)))
                    p = self.grid[p].parent
                path.reverse()
                
                return {
                    "found": True,
                    "path": path,
                    "path_length": round(len(path) * self.cell_size, 2),
                    "cost": round(self.grid[goal].g, 2),
                    "steps": len(path),
                    "has_clearance": True
                }
            
            open_set.remove(current)
            closed_set.add(current)
            
            # Check neighbors
            for nx, ny, move_cost in self._get_neighbors(current, robot_r_cells):
                if (nx, ny) in closed_set:
                    continue
                
                # Check dynamic obstacles (where will they be?)
                if avoid_dynamic:
                    cell_center = ((nx + 0.5) * self.cell_size, (ny + 0.5) * self.cell_size)
                    for dobj in self.dynamic_obstacles:
                        # Predict position after path steps
                        future_pos = (
                            dobj["pos"][0] + dobj["velocity"][0] * len(closed_set) * 0.1,
                            dobj["pos"][1] + dobj["velocity"][1] * len(closed_set) * 0.1
                        )
                        dist = math.sqrt((cell_center[0] - future_pos[0])**2 + 
                                       (cell_center[1] - future_pos[1])**2)
                        if dist < dobj["radius"] + robot_radius:
                            move_cost += 5.0  # Penalty for dynamic obstacle proximity
                
                tentative_g = self.grid[current].g + move_cost
                
                if tentative_g < self.grid[(nx, ny)].g:
                    self.grid[(nx, ny)].parent = current
                    self.grid[(nx, ny)].g = tentative_g
                    self.grid[(nx, ny)].h = self._heuristic((nx, ny), goal)
                    self.grid[(nx, ny)].f = self.grid[(nx, ny)].g + self.grid[(nx, ny)].h
                    
                    if (nx, ny) not in open_set:
                        open_set.add((nx, ny))
        
        return {"found": False, "reason": "No path exists", "explored": len(closed_set)}


# ============================================================
# 3. SQUARE-CUBE LAW — Scaling analysis
# ============================================================

class ScalingAnalyzer:
    """
    How properties change with size.
    - Area ∝ L², Volume ∝ L³
    - Strength ∝ L², Weight ∝ L³ → strength/weight ∝ 1/L
    - Explains why ants can lift 50× body weight but elephants can't jump
    """
    
    @staticmethod
    def scale_property(original_value: float, scale_factor: float, 
                       property_type: str) -> float:
        """Scale a physical property by factor"""
        exponents = {
            "length": 1, "area": 2, "volume": 3, "mass": 3, "weight": 3,
            "strength": 2, "cross_section": 2, "moment_inertia": 4,
            "frequency": -1, "pressure": 0, "density": 0,
            "surface_tension": 1, "heat_loss_rate": 2, "metabolic_rate": 0.75,
            "terminal_velocity": 0.5, "jump_height": 0,
            "bone_stress": 1,
        }
        exp = exponents.get(property_type, 1)
        return original_value * (scale_factor ** exp)
    
    @staticmethod
    def compare_organisms(name_a: str, mass_a: float, name_b: str, mass_b: float) -> dict:
        """Compare two organisms: how do properties differ at different sizes?"""
        scale = mass_b / max(mass_a, 1e-10)
        L_ratio = scale ** (1/3)  # Linear dimension ratio
        
        comparisons = []
        
        # Strength/weight ratio (decreases with size)
        sw_a = 1.0  # Normalized
        sw_b = L_ratio**2 / L_ratio**3  # ∝ 1/L
        comparisons.append(f"Force/poids: {name_a}=1.0 → {name_b}={sw_b:.2f}×")
        
        # Jump height (independent of size in theory, limited by strength)
        comparisons.append(f"Saut: {'identique' if abs(sw_b - 1) < 0.3 else f'{name_a} saute plus haut' if sw_b < 1 else f'{name_b} saute plus haut'}")
        
        # Metabolic rate
        mr_a = mass_a**0.75
        mr_b = mass_b**0.75
        mr_per_kg_a = mr_a / mass_a
        mr_per_kg_b = mr_b / mass_b
        comparisons.append(f"Métabolisme/kg: {name_a}={mr_per_kg_a:.1f} → {name_b}={mr_per_kg_b:.1f}")
        
        # Terminal velocity
        tv_a = math.sqrt(mass_a)
        tv_b = math.sqrt(mass_b)
        comparisons.append(f"Vitesse terminale: {tv_a/max(tv_b,1e-10):.1f}× {'plus rapide' if tv_a > tv_b else 'moins rapide'} pour {name_a if tv_a > tv_b else name_b}")
        
        # Bone stress (∝ L → bigger animals need thicker bones)
        bone_stress_b = L_ratio
        comparisons.append(f"Stress osseux: ×{bone_stress_b:.1f} {'⚠️ nécessite os plus épais' if bone_stress_b > 1.5 else '✅ OK'}")
        
        return {
            "scale_factor": round(scale, 1),
            "linear_ratio": round(L_ratio, 2),
            "comparisons": comparisons,
            "verdict": f"{name_b} est {scale:.0f}× plus massif — {'la gravité domine, structure compromise' if L_ratio > 3 else 'proportions viables' if L_ratio < 3 else 'limite structurelle atteinte'}"
        }
    
    @staticmethod
    def building_limits(material: str = "beton", target_height: float = 100) -> dict:
        """How tall can a building be with a given material?"""
        mat = {
            "beton": {"strength": 30e6, "density": 2400},
            "acier": {"strength": 400e6, "density": 7800},
            "bois": {"strength": 50e6, "density": 700},
            "brique": {"strength": 10e6, "density": 1900},
        }.get(material, {"strength": 30e6, "density": 2400})
        
        g = 9.81
        
        # Maximum height before base crushes under own weight
        # σ_max = ρgh_max → h_max = σ_max/(ρg)
        max_height = mat["strength"] / (mat["density"] * g)
        
        # Safety factor
        sf = max_height / max(target_height, 1.0)
        
        return {
            "material": material,
            "max_theoretical_height_m": round(max_height, 0),
            "target_height_m": target_height,
            "safety_factor": round(sf, 1),
            "feasible": sf > 2.0,
            "verdict": f"✅ Faisable (FS={sf:.1f})" if sf > 2.0 else 
                       f"⚡ Limite (FS={sf:.1f})" if sf > 1.0 else
                       f"⚠️ IMPOSSIBLE — le matériau s'écrase sous son propre poids"
        }
    
    @staticmethod
    def bridge_span_limits(material: str, span: float, load_per_meter: float = 10000) -> dict:
        """Maximum bridge span for a given material"""
        mat = {
            "acier": {"strength": 400e6, "density": 7800},
            "beton_arme": {"strength": 40e6, "density": 2500},
            "bois": {"strength": 50e6, "density": 700},
        }.get(material, {"strength": 400e6, "density": 7800})
        
        g = 9.81
        
        # Simplified: M_max = wL²/8, σ = M/Z, Z ∝ h², self-weight ∝ h
        # Max span where self-weight + live load exceeds strength
        # Approximation: L_max ∝ √(σ/ρg)
        max_span = math.sqrt(mat["strength"] / (mat["density"] * g)) * 2
        
        sf = max_span / max(span, 1.0)
        
        return {
            "material": material,
            "max_span_m": round(max_span, 0),
            "target_span_m": span,
            "safety_factor": round(sf, 1),
            "feasible": sf > 2.0,
            "verdict": f"✅ Structure possible (FS={sf:.1f})" if sf > 2.0 else
                       f"⚠️ Proche de la limite (FS={sf:.1f})" if sf > 1.0 else
                       f"❌ IMPOSSIBLE sans appuis intermédiaires"
        }


# ============================================================
# 4. DEMO — All three engines
# ============================================================

def demo():
    sep = "=" * 60
    
    # 1. Perception & Occlusion
    print(f"{sep}\n  1. PERCEPTION & OCCLUSION — Garde et intrus\n{sep}")
    engine = OcclusionEngine()
    
    guard = Observer("garde", (0, 0), math.radians(45), math.radians(90), 15.0)
    engine.add_observer(guard)
    
    engine.add_occluder("pilier_A", (3, 2), 1.0, 3.0)
    engine.add_occluder("pilier_B", (5, -1), 0.8, 2.5)
    engine.add_occluder("mur", (8, 3), 2.0, 4.0)
    
    engine.add_target("intrus_1", (4, 3), 0.4, 1.8)
    engine.add_target("intrus_2", (2, 1), 0.4, 1.7)
    engine.add_target("intrus_3", (10, 0), 0.4, 1.8)
    engine.add_target("intrus_4", (-1, 2), 0.4, 1.7)
    
    vis = engine.compute_visibility(guard)
    for tid, v in vis.items():
        icon = "👁" if v["can_see"] else "🙈"
        print(f"  {icon} {tid}: {v['class']} ({v['visibility_pct']:.0f}% visible, dist={v['distance']}m)")
        if v["occluded_by"]:
            for occ in v["occluded_by"]:
                print(f"      bloqué par {occ['by']} ({occ['fraction']:.0%})")
    
    # Blind spots
    spots = engine.find_blind_spots(guard)
    if spots:
        print(f"  Zones aveugles: {len(spots)} secteurs")
        for s in spots[:3]:
            print(f"    {s['start_angle']}°→{s['end_angle']}° (bloqué par {s['blocker']})")
    
    # 2. Pathfinding
    print(f"\n{sep}\n  2. PATHFINDING — Robot dans entrepôt\n{sep}")
    pf = Pathfinder(30, 30, cell_size=1.0)
    
    # Obstacles
    pf.add_obstacle((10, 10), 3.0)
    pf.add_obstacle((20, 15), 2.0)
    pf.add_obstacle((10, 22), 2.5)
    pf.add_obstacle((25, 5), 1.5)
    pf.add_obstacle((5, 6), 1.0)
    pf.compute_clearance()
    
    # Dynamic obstacle
    pf.add_obstacle((15, 15), 1.0, velocity=(0.5, -0.3))
    
    result = pf.find_path((2, 2), (27, 27), robot_radius=0.5)
    print(f"  Found: {result['found']}")
    if result['found']:
        print(f"  Path: {result['steps']} steps, {result['path_length']}m")
        print(f"  Cost: {result['cost']}")
        # Show path
        for i, p in enumerate(result['path']):
            if i % 5 == 0 or i == len(result['path']) - 1:
                print(f"    [{i:2d}] ({p[0]:.0f}, {p[1]:.0f})")
    
    # 3. Square-Cube Law
    print(f"\n{sep}\n  3. LOI CARRÉ-CUBE — Fourmi vs Éléphant\n{sep}")
    result = ScalingAnalyzer.compare_organisms("fourmi", 0.000005, "éléphant", 5000)
    for k, v in result.items():
        if k != "comparisons":
            print(f"  {k}: {v}")
    for c in result["comparisons"]:
        print(f"    • {c}")
    
    print(f"\n  Limite structurale:")
    bld = ScalingAnalyzer.building_limits("beton", 200)
    bridge = ScalingAnalyzer.bridge_span_limits("acier", 500)
    print(f"    Bâtiment: {bld['verdict']} (max={bld['max_theoretical_height_m']}m)")
    print(f"    Pont: {bridge['verdict']} (max={bridge['max_span_m']}m)")


if __name__ == "__main__":
    demo()
