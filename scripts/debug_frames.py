"""Debug: inspect obs.frame for different games."""
import arc_agi

for game_id in ["ls20", "sp80", "sc25", "re86", "bp35"]:
    try:
        arc = arc_agi.Arcade()
        env = arc.make(game_id)
        obs = env.reset()
        print(f"\n=== {game_id} ===")
        print(f"  obs.frame type: {type(obs.frame)}")
        print(f"  obs.frame: {obs.frame}")
        if isinstance(obs.frame, list):
            print(f"  len(frame): {len(obs.frame)}")
            if len(obs.frame) > 0:
                print(f"  frame[0].shape: {obs.frame[0].shape}")
            else:
                print(f"  ❌ EMPTY FRAME LIST")
        elif hasattr(obs.frame, 'shape'):
            print(f"  frame.shape: {obs.frame.shape}")
        print(f"  available_actions: {obs.available_actions}")
        print(f"  levels_completed: {obs.levels_completed}")
    except Exception as e:
        print(f"  ❌ {e}")
