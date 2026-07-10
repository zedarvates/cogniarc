"""Check learned action directions on holdout games."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import arc_agi
import random
from cogniarc.object_perception import ObjectTracker

for game_id in ["wa30", "ls20", "sp80", "sc25"]:
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.reset()
    tracker = ObjectTracker()
    rng = random.Random(42)
    
    for i in range(30):
        actions = obs.available_actions
        gb = obs.frame[0].copy()
        a = rng.choice(actions)
        obs = env.step(a)
        ga = obs.frame[0].copy()
        tracker.observe(gb, a, ga)
    
    print(f"\n{game_id}:")
    print(f"  Player color: {tracker.player_color}")
    print(f"  Available actions: {obs.available_actions}")
    for a in obs.available_actions:
        d = tracker.action_direction(a)
        if d:
            dr, dc = d
            # dr = Δrow (Δy), dc = Δcol (Δx)
            dir_name = ""
            if abs(dr) > abs(dc):
                dir_name = "DOWN" if dr > 0 else "UP"
            else:
                dir_name = "RIGHT" if dc > 0 else "LEFT"
            print(f"  Action {a}: (Δrow={dr:.2f}, Δcol={dc:.2f}) → {dir_name}")
        else:
            print(f"  Action {a}: no direction learned")
