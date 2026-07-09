"""
Phase 1 T1.1 — Mesure des gaps de perception de l'ObjectTracker.

Instrumente observe(), simule des transitions avec disparitions/changements
de couleur, et décide si un matching neuronal est justifié.

Règle : si vanish_rate < 5% → pas de neuronal. Si ≥5% → micro-NN siamois.
"""

import json
import numpy as np
from typing import Dict, List, Optional
from cogniarc.object_perception import ObjectTracker


def simulate_grid_transitions(
    n_steps: int = 50,
    vanish_rate: float = 0.10,
    color_change_rate: float = 0.05,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic grid transitions with controlled perception gaps.

    Creates a small grid (10×10) with a few colored regions. Some regions
    vanish between frames (simulating consumption/destruction) and some
    change color (simulating teleportation or state change).

    Returns list of {"before": grid, "action": int, "after": grid} dicts.
    """
    rng = np.random.RandomState(seed)
    transitions = []

    for step in range(n_steps):
        grid_before = np.zeros((10, 10), dtype=np.int32)

        # Place a few colored regions
        colors = [1, 2, 3, 4]
        regions = {}
        for color in colors:
            cx = rng.randint(1, 9)
            cy = rng.randint(1, 9)
            # 2×2 block
            for dx in range(2):
                for dy in range(2):
                    x, y = min(9, cx + dx), min(9, cy + dy)
                    grid_before[y, x] = color
            regions[color] = (cx, cy)

        # Build the "after" grid
        grid_after = grid_before.copy()

        action = rng.randint(0, 5)

        # Some regions vanish
        for color in colors:
            if rng.rand() < vanish_rate:
                # Remove this region in "after"
                mask = grid_after == color
                grid_after[mask] = 0

        # Some regions change color
        for color in colors:
            if rng.rand() < color_change_rate:
                new_color = rng.choice([c for c in colors if c != color])
                mask = grid_after == color
                grid_after[mask] = new_color

        # Player region (color 1) moves
        player_pos = regions[1]
        dx, dy = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
        new_x = max(0, min(9, player_pos[0] + dx))
        new_y = max(0, min(9, player_pos[1] + dy))

        # Clear old player
        grid_after[grid_after == 1] = 0
        # Place new player
        grid_after[new_y:new_y+2, new_x:new_x+2] = 1

        transitions.append({
            "step": step,
            "before": grid_before,
            "action": action,
            "after": grid_after,
        })

    return transitions


def run_measurement(
    vanish_rate: float = 0.10,
    color_change_rate: float = 0.05,
    n_steps: int = 100,
    output_jsonl: Optional[str] = None,
) -> Dict:
    """Run ObjectTracker on synthetic transitions and measure gap stats."""
    tracker = ObjectTracker()
    transitions = simulate_grid_transitions(
        n_steps=n_steps,
        vanish_rate=vanish_rate,
        color_change_rate=color_change_rate,
        seed=42,
    )

    for t in transitions:
        tracker.observe(
            grid_before=t["before"],
            action=t["action"],
            grid_after=t["after"],
        )

    stats = tracker.perception_gap_stats()

    # Detailed analysis
    color_vanished: Dict[int, int] = {}
    for entry in tracker.vanished_log:
        c = entry["color"]
        color_vanished[c] = color_vanished.get(c, 0) + 1

    result = {
        **stats,
        "vanish_rate_target": vanish_rate,
        "color_change_rate": color_change_rate,
        "color_vanished_breakdown": color_vanished,
        "player_color_found": tracker.player_color is not None,
        "player_color": int(tracker.player_color) if tracker.player_color is not None else None,
    }

    if output_jsonl and tracker.vanished_log:
        with open(output_jsonl, "w") as f:
            for entry in tracker.vanished_log:
                f.write(json.dumps(entry) + "\n")

    return result


def run_decision(
    games_results: List[Dict],
    threshold: float = 0.05,
) -> str:
    """Decide if neural vision is justified based on measured gaps.

    threshold = 5% (from plan spec).
    """
    rates = [r["vanish_rate"] for r in games_results]
    max_rate = max(rates)
    avg_rate = sum(rates) / len(rates)

    decision = "GO" if max_rate >= threshold else "NO-GO"

    report = f"""
╔══════════════════════════════════════════════╗
║  Phase 1 T1.1 — Perception Gap Measurement  ║
╠══════════════════════════════════════════════╣
║  Games tested     : {len(games_results):>3}                        ║
║  Max vanish rate  : {max_rate:.2%}                   ║
║  Avg vanish rate  : {avg_rate:.2%}                   ║
║  Threshold        : {threshold:.0%}                       ║
║  Decision         : {decision:<20}║
╚══════════════════════════════════════════════╝
"""
    return report.strip()


if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("📊 Phase 1 T1.1 — Mesure des gaps de perception")
    print("=" * 50)

    # Test 1: Low vanish rate (should be NO-GO)
    print("\n🧪 Test 1: Low gaps (vanish=2%, color_change=2%)")
    r1 = run_measurement(vanish_rate=0.02, color_change_rate=0.02, n_steps=100)
    print(f"   Vanish rate: {r1['vanish_rate']:.2%} ({r1['vanished_count']}/{r1['total_attempts']})")
    print(f"   Player color: {r1['player_color']}")

    # Test 2: Medium vanish rate (marginal)
    print("\n🧪 Test 2: Medium gaps (vanish=5%, color_change=5%)")
    r2 = run_measurement(vanish_rate=0.05, color_change_rate=0.05, n_steps=100)
    print(f"   Vanish rate: {r2['vanish_rate']:.2%} ({r2['vanished_count']}/{r2['total_attempts']})")
    print(f"   Player color: {r2['player_color']}")

    # Test 3: High vanish rate (should be GO)
    print("\n🧪 Test 3: High gaps (vanish=15%, color_change=10%)")
    r3 = run_measurement(vanish_rate=0.15, color_change_rate=0.10, n_steps=100)
    print(f"   Vanish rate: {r3['vanish_rate']:.2%} ({r3['vanished_count']}/{r3['total_attempts']})")
    print(f"   Player color: {r3['player_color']}")

    # Decision
    print("\n" + run_decision([r1, r2, r3], threshold=0.05))

    # Save detailed logs
    import os
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "perception_gaps.jsonl")
    run_measurement(vanish_rate=0.15, color_change_rate=0.10, n_steps=200, output_jsonl=log_path)
    print(f"\n📝 Detailed log: {log_path}")

    print("\n✅ Measurement complete.")
