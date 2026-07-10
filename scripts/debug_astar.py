"""Debug A* pathfinding on real game data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import arc_agi
import numpy as np
from cogniarc.object_perception import ObjectTracker, segment_regions
from cogniarc.pathfinding import GridMap, astar, astar_path_to_actions

arc = arc_agi.Arcade()
env = arc.make("wa30")
obs = env.reset()
tracker = ObjectTracker()

import random
rng = random.Random(42)
for i in range(20):
    actions = obs.available_actions
    gb = obs.frame[0].copy()
    a = rng.choice(actions)
    obs = env.step(a)
    ga = obs.frame[0].copy()
    tracker.observe(gb, a, ga)

pc = tracker.player_color
wc = tracker.wall_colors
print(f"Player color: {pc}")
print(f"Wall colors: {wc}")

# Find player
grid = obs.frame[0]
regions = segment_regions(grid)
for r in regions:
    if r.color == pc:
        px, py = int(round(r.center[1])), int(round(r.center[0]))
        print(f"Player at: col={px}, row={py} (value at grid[{py},{px}]={grid[py,px]})")

# Build GridMap with EMPTY wall colors
gm_empty = GridMap.from_observation(obs, set())
print(f"\nGridMap with empty wall set:")
print(f"  walkable[{py}, {px}] = {gm_empty.walkable[py, px]}")
print(f"  walkable[32, 32] = {gm_empty.walkable[32, 32]}")
print(f"  is_walkable({px}, {py}): {gm_empty.is_walkable(px, py)}")
print(f"  is_walkable(32, 32): {gm_empty.is_walkable(32, 32)}")

# Try A* with empty set
path1 = astar(gm_empty, (px, py), (32, 32))
print(f"\nA* with empty wall set: {'FOUND' if path1 else 'None'}")
if path1:
    print(f"  Path: {len(path1)} steps")
    actions1 = astar_path_to_actions(path1)
    print(f"  Actions: {actions1}")
else:
    # Debug: check start and goal neighbors
    print(f"  Start ({px},{py}) neighbors: {gm_empty.neighbors(px, py)}")
    print(f"  Goal (32,32) neighbors: {gm_empty.neighbors(32, 32)}")

# Build GridMap with heuristic (None = use default)
gm_heuristic = GridMap.from_observation(obs, None)
print(f"\nGridMap with heuristic wall detection:")
path2 = astar(gm_heuristic, (px, py), (32, 32))
print(f"A* with heuristic: {'FOUND' if path2 else 'None'}")
if path2:
    print(f"  Path: {len(path2)} steps")
else:
    print(f"  Start wall: {not gm_heuristic.is_walkable(px, py)}")
    print(f"  Goal wall: {not gm_heuristic.is_walkable(32, 32)}")
    # Check which colors are walls
    unique, counts = np.unique(grid, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    for idx in sorted_idx[:5]:
        c = int(unique[idx])
        print(f"  Color {c}: {counts[idx]} cells")

# Extra debug: where is background? 
bg_color = grid[0, 0]
print(f"\nBackground color (grid[0,0]): {bg_color}")
print(f"walkable at (0,0): {gm_empty.is_walkable(0, 0)}")
print(f"grid all zeros: {np.all(grid == 0)}")
print(f"Unique values: {sorted(np.unique(grid))}")
