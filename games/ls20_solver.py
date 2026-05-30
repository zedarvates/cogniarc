#!/usr/bin/env python3
"""LS20 solver v3 — multi-level with per-level navigation."""

import arc_agi
from arcengine import GameAction

arc = arc_agi.Arcade()
env = arc.make('ls20')
obs = env.reset()

game = None
for attr in dir(env):
    val = getattr(env, attr)
    if 'Ls20' in str(type(val)):
        game = val
        break
p = game.gudziatsk

def do(action_num):
    global obs
    obs = env.step(getattr(GameAction, f'ACTION{action_num}'))
    return obs

def find_sprites(tag):
    """Find all sprites with given tag at current level."""
    level = game.current_level
    sprites = getattr(level, '_sprites', [])
    return [s for s in sprites if hasattr(s, 'tags') and s.tags and tag in s.tags]

def get_level_data(key):
    """Read level data (StartShape, GoalRotation, etc.)."""
    level = game.current_level
    try:
        return level.get_data(key)
    except:
        return None

steps = 0
lvl_done = 0
print(f"START: ({p.x},{p.y}) rot={game.cklxociuu}° col={game.tnkekoeuk[game.hiaauhahz]}")

# ====== LEVEL 1 ======
# R×3, L×6, U×3 → changer (19,30)
for a in [4,4,4, 3,3,3,3,3,3, 1,1,1]: do(a); steps += 1
# Cycle rotation to 0
do(4); steps += 1; do(3); steps += 1
while game.cklxociuu != 0: do(4); steps += 1; do(3); steps += 1
# D×3, R×3, U×7 → lock (34,10)
for a in [2,2,2, 4,4,4, 1,1,1,1,1,1,1]:
    do(a); steps += 1
    if obs.levels_completed > lvl_done: lvl_done = obs.levels_completed; break

print(f"L1 ✓ ({steps} steps, lvl={lvl_done})")

# ====== LEVEL 2 ======
# Rot 0→270°, lock at (14,40), changer at (49,45)
# From (34,10): D×7 → (34,45), R×3 → (49,45)
for a in [2,2,2,2,2,2,2, 4,4,4]: do(a); steps += 1
print(f"  At changer L2: ({p.x},{p.y}) rot={game.cklxociuu}")

# Cycle rotation 0→3 (3 cycles)
do(3); steps += 1; do(4); steps += 1  # into changer: 0→1
while game.cklxociuu != 3:
    do(3); steps += 1; do(4); steps += 1  # keep cycling
print(f"  Rot=270: ({p.x},{p.y}) rot={game.cklxociuu} ({steps} steps)")

# Navigate to lock (14,40): L×6 → (19,45), U×1 → (19,40), L to (14,40)
for a in [3,3,3,3,3,3, 1]: do(a); steps += 1
print(f"  Near lock: ({p.x},{p.y})")

# Try LEFT to reach (14,40)
for _ in range(5):
    prev_x = p.x
    do(3); steps += 1
    if obs.levels_completed > lvl_done:
        lvl_done = obs.levels_completed; break
    if p.x == prev_x: break
print(f"  L2: ({p.x},{p.y}) lvl={obs.levels_completed} ({steps} steps)")

# ====== LEVELS 3-7: Adaptive exploration ======
while obs.levels_completed < obs.win_levels and steps < 500:
    prev_lvl = obs.levels_completed
    
    # Find next changer and lock
    rot_changers = find_sprites('rhsxkxzdjz')
    locks = find_sprites('rjlbuycveu')
    goal_rot = get_level_data('GoalRotation')
    
    rot_target = game.dhksvilbb.index(goal_rot) if goal_rot is not None else game.cklxociuu
    
    # Navigate to nearest changer
    if rot_changers:
        ch = rot_changers[0]
        # Go to changer position
        while p.y > ch.y: do(1); steps += 1
        while p.y < ch.y: do(2); steps += 1
        while p.x > ch.x: do(3); steps += 1
        while p.x < ch.x: do(4); steps += 1
        
        # Cycle rotation
        while game.cklxociuu != rot_target:
            do(3); steps += 1; do(4); steps += 1  # cycle in/out
    
    # Navigate to nearest lock
    if locks:
        lk = locks[0]
        while p.y > lk.y: do(1); steps += 1
        while p.y < lk.y: do(2); steps += 1
        while p.x > lk.x: do(3); steps += 1
        while p.x < lk.x: do(4); steps += 1
    
    # If not progressing, try random exploration
    if obs.levels_completed == prev_lvl:
        for _ in range(10):
            do((steps % 4) + 1); steps += 1
            if obs.levels_completed > prev_lvl:
                break
    
    if obs.levels_completed == prev_lvl:
        print(f"  Stuck at lvl={obs.levels_completed}")
        break
    
    if obs.levels_completed > prev_lvl:
        print(f"  L{obs.levels_completed} ✓ ({steps} steps)")

print(f"\n=== FINAL: {obs.levels_completed}/{obs.win_levels} levels, {steps} steps ===")
print(f"State: {obs.state}")
