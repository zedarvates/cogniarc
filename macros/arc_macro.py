#!/usr/bin/env python3
"""
ARC-AGI-3 Macro System — record/replay/parameterize action sequences.
Usage:
  python arc_macro.py record <game> <name> <actions...>
  python arc_macro.py replay <game> <name>
  python arc_macro.py explore <game> <action> <times>
"""
import json, sys, os
from pathlib import Path

MACRO_DIR = Path(__file__).parent / "recordings"
MACRO_DIR.mkdir(exist_ok=True)

def record(game, name, actions):
    """Save a macro: list of (action_id, data) tuples."""
    # Format: actions = [(act, {data}), ...]
    macro = {"game": game, "name": name, "actions": actions}
    path = MACRO_DIR / f"{game}_{name}.json"
    path.write_text(json.dumps(macro, indent=2))
    print(f"Saved: {path} ({len(actions)} steps)")
    return str(path)

def load(game, name):
    path = MACRO_DIR / f"{game}_{name}.json"
    if not path.exists():
        print(f"Not found: {path}")
        return None
    return json.loads(path.read_text())

def list_macros(game=None):
    macros = []
    for f in sorted(MACRO_DIR.glob("*.json")):
        m = json.loads(f.read_text())
        if game is None or m["game"] == game:
            macros.append({"file": f.name, "game": m["game"], "name": m["name"], "steps": len(m["actions"])})
    return macros

def replay(env, game, name, verbose=True):
    """Replay a macro on the given environment."""
    macro = load(game, name)
    if not macro:
        return False
    for i, step in enumerate(macro["actions"]):
        act = step[0]
        data = step[1] if len(step) > 1 else {}
        try:
            env.step(act, data)
            if verbose:
                print(f"  Step {i+1}: action={act} data={data}")
        except Exception as e:
            print(f"  ERROR step {i+1}: {e}")
            return False
    return True

def explore(env, action, data, times):
    """Execute same action N times and report diffs."""
    import numpy as np
    for i in range(times):
        obs = env.reset() if i == 0 else None
        g = env._game
        # Capture before
        before = np.array(obs.frame[0].data, dtype=np.int8) if obs else None
        env.step(action, data)
        if before is not None:
            after = np.array(
                [item for item in env if hasattr(item,'frame')][0].frame[0].data,
                dtype=np.int8
            ) if hasattr(env,'__iter__') else None
            if after is not None:
                diff = (before != after).sum()
                print(f"  Iter {i+1}: {diff}px changed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Commands: record, replay, list, explore")
        sys.exit(1)
    
    cmd = sys.argv[1]
    import arc_agi
    
    if cmd == "list":
        game = sys.argv[2] if len(sys.argv) > 2 else None
        for m in list_macros(game):
            print(f"  {m['file']}: {m['game']}/{m['name']} ({m['steps']} steps)")
    
    elif cmd == "record":
        game = sys.argv[2]
        name = sys.argv[3]
        # Parse actions from args: act1 data1 act2 data2 ...
        actions = []
        i = 4
        while i < len(sys.argv):
            act = int(sys.argv[i])
            data = {}
            if i + 1 < len(sys.argv) and sys.argv[i + 1].startswith("{"):
                import ast
                data = ast.literal_eval(sys.argv[i + 1])
                i += 1
            actions.append((act, data))
            i += 1
        record(game, name, actions)
    
    elif cmd == "replay":
        game = sys.argv[2]
        name = sys.argv[3]
        arc = arc_agi.Arcade()
        env = arc.make(game)
        env.reset()
        replay(env, game, name)
    
    elif cmd == "explore":
        game = sys.argv[2]
        action = int(sys.argv[3])
        data = {}
        if len(sys.argv) > 4 and sys.argv[4].startswith("{"):
            import ast
            data = ast.literal_eval(sys.argv[4])
        times = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 10
        arc = arc_agi.Arcade()
        env = arc.make(game)
        env.reset()
        explore(env, action, data, times)
