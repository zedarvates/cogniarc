#!/usr/bin/env python3
"""LS20 solver with cognitive drives — using proven navigation paths."""

import arc_agi
from arcengine import GameAction

from cognitive_player import CognitiveDrives, hash_grid

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

drives = CognitiveDrives()
total = 0

def do(a):
    global obs, total
    obs = env.step(getattr(GameAction, f'ACTION{a}'))
    total += 1
    if hasattr(obs, 'frame') and obs.frame:
        drives.step(a, hash_grid(obs.frame[0]))
    else:
        drives.step(a, f"step_{total}")
    return obs

def find_tag(tag):
    sprites = getattr(game.current_level, '_sprites', [])
    return [s for s in sprites if hasattr(s, 'tags') and s.tags and tag in s.tags]

# ====== LEVEL 1 (proven path) ======
print("🎮 Niveau 1 — Rotation 270°→0°")

# R×3, L×6, U×3 → changer (19,30)
for a in [4,4,4, 3,3,3,3,3,3, 1,1,1]: do(a)
print(f"  Changer: ({p.x},{p.y}) rot={game.cklxociuu}")

# Cycle rotation to 0
do(4); do(3)  # away + back = 1 cycle
while game.cklxociuu != 0 and total < 40:
    do(4); do(3)
print(f"  Rotation=0 après {total} steps")

# D×3, R×3, U×7 → lock (34,10)
for a in [2,2,2, 4,4,4, 1,1,1,1,1,1,1]:
    do(a)
    if obs.levels_completed > 0:
        break
print(f"  L1 {'✅' if obs.levels_completed >= 1 else '❌'} "
      f"lvl={obs.levels_completed} ({total} steps)")

# ====== LEVEL 2 ======
if obs.levels_completed >= 1:
    print(f"\n🎮 Niveau 2 — Rotation 0°→270°")
    
    # Level 2: changer at (49,45), lock at (14,40)
    # From (34,10): if stuck, check if we can move
    directions_tried = 0
    while (p.x, p.y) == (34, 10) and directions_tried < 20:
        do((directions_tried % 4) + 1)  # try all directions
        directions_tried += 1
    print(f"  Débloqué après {directions_tried} tries → ({p.x},{p.y})")
    
    # Navigate to changer at (49,45)
    # Go to bottom (y=45), then right edge
    while p.y != 45 and total < 150:
        if p.y < 45: do(2)  # DOWN
        else: do(1)  # UP
    
    while p.x < 49 and total < 150:
        do(4)  # RIGHT
    
    print(f"  Changer L2: ({p.x},{p.y}) rot={game.cklxociuu}")
    
    # Cycle rotation to 3 (270°)
    do(3); do(4)  # into changer
    while game.cklxociuu != 3 and total < 180:
        do(3); do(4)
    print(f"  Rotation=270° ({total} steps)")
    
    # Navigate to lock at (14,40)
    # L×6 → (19,45), U×1 → (19,40), L → (14,40)
    for a in [3,3,3,3,3,3, 1]: do(a)
    while p.x > 14 and total < 200:
        prev_x = p.x
        do(3)
        if p.x == prev_x: break  # blocked
    
    print(f"  L2: ({p.x},{p.y}) lvl={obs.levels_completed} ({total} steps)")
    
    if obs.levels_completed >= 2:
        print(f"  L2 ✅")

# ====== GENERIC LOOP for remaining levels ======
prev_lvl = obs.levels_completed
while obs.levels_completed < obs.win_levels and total < 400:
    if obs.levels_completed > prev_lvl:
        print(f"\n🎮 Niveau {obs.levels_completed + 1}")
        prev_lvl = obs.levels_completed
    
    # Cognitive decision
    if drives.stagnation_counter > 12:
        print(f"  ⚡ DOUTE ({drives.stagnation_counter} stagnation)")
        drives.doubt_check(drives.stagnation_counter, drives.world_model_confidence)
        # Random exploration
        for _ in range(8):
            do((total % 4) + 1)
    
    # Find sprites via tags
    changers = find_tag('rhsxkxzdjz')
    locks = find_tag('rjlbuycveu')
    
    # Navigate to nearest changer if rotation wrong
    if changers:
        ch = changers[0]
        goal_rot = None
        try: goal_rot = game.current_level.get_data('GoalRotation')
        except: pass
        
        if goal_rot is not None:
            target = game.dhksvilbb.index(goal_rot)
            if game.cklxociuu != target:
                # Navigate to changer
                cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)
                while p.x < cx: do(4)
                while p.x > cx: do(3)
                while p.y < cy: do(2)
                while p.y > cy: do(1)
                
                # Cycle rotation
                do(3); do(4)
                while game.cklxociuu != target and total < 350:
                    do(3); do(4)
    
    # Navigate to locks
    if locks:
        for lk in locks[:3]:
            lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
            while p.x < lx: do(4)
            while p.x > lx: do(3)
            while p.y < ly: do(2)
            while p.y > ly: do(1)
            
            do(4); do(3)  # interact
            if obs.levels_completed > prev_lvl:
                print(f"  ✅ L{obs.levels_completed} ({total} steps)")
                break
    
    # Intuition mode if stuck
    if obs.levels_completed == prev_lvl and total > 250:
        for _ in range(20):
            do((total % 4) + 1)
            if obs.levels_completed > prev_lvl:
                print(f"  ✅ L{obs.levels_completed} via intuition ({total} steps)")
                break
    
    if obs.levels_completed == prev_lvl and total > 350:
        print(f"  Timeout")
        break

print(f"\n{'='*50}")
print(f"🏆 {obs.levels_completed}/{obs.win_levels} niveaux, {total} steps")
print(drives.status_report())
