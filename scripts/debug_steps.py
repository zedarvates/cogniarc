"""Check what env.step() returns for different games."""
import arc_agi

for game_id in ["ls20", "sp80", "sc25", "bp35"]:
    try:
        arc = arc_agi.Arcade()
        env = arc.make(game_id)
        obs = env.reset()
        print(f"\n=== {game_id} ===")
        print(f"  actions: {obs.available_actions}")
        # Try stepping
        for a in obs.available_actions[:1]:
            try:
                obs2 = env.step(a)
                print(f"  step({a}) OK → frame len={len(obs2.frame)}")
            except Exception as e:
                print(f"  step({a}) ERROR: {e}")
                # Try with GameAction
                try:
                    from arc_agi import GameAction, models
                    ga = getattr(GameAction, f"ACTION{a}", None)
                    if ga:
                        obs2 = env.step(ga)
                        print(f"  step(GameAction.{ga.name}) OK → frame len={len(obs2.frame)}")
                    else:
                        print(f"  GameAction.ACTION{a} not found")
                except Exception as e2:
                    print(f"  step with GameAction also ERROR: {e2}")
    except Exception as e:
        print(f"  ❌ {e}")
