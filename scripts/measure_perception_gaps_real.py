"""
Phase 1 T1.1 — Mesure RÉELLE des gaps de perception sur les jeux holdout.

Au lieu de données synthétiques, on lance l'ObjectTracker sur les vrais jeux
ARC-AGI-3 et on mesure le taux de régions qui disparaissent sans explication.
"""

import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cogniarc.object_perception import ObjectTracker


# Jeux holdout documentés dans docs/EVALUATION.md
HOLDOUT_GAMES = ["sp80", "wa30", "sc25", "re86", "bp35"]
# Jeu connu (LS20) comme baseline de comparaison
BASELINE_GAMES = ["ls20"]
# Tous les jeux dispo
ALL_GAMES = sorted([
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09",
    "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
    "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
])


def measure_game(game_id: str, max_steps: int = 200, seed: int = 42) -> dict:
    """Run ObjectTracker on a real ARC-AGI game and measure perception gaps.

    Plays random actions, tracking region matching step by step.
    Some games return multiple frames per step (animation) — we use frame[0].
    Some games have non-standard action numbering (e.g. bp35 starts at 3).
    """
    import arc_agi
    import random as py_random
    rng = py_random.Random(seed)

    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.reset()

    tracker = ObjectTracker()
    levels_before = obs.levels_completed
    steps_taken = 0
    errors = 0

    for step in range(max_steps):
        # Get available actions
        actions = obs.available_actions
        if not actions:
            break

        try:
            grid_before = obs.frame[0].copy()
        except (IndexError, TypeError, AttributeError):
            errors += 1
            continue

        # Pick a random action
        action = rng.choice(actions)
        try:
            obs = env.step(action)
        except Exception:
            errors += 1
            continue

        steps_taken += 1

        try:
            grid_after = obs.frame[0].copy()
        except (IndexError, TypeError, AttributeError):
            errors += 1
            continue

        # Feed to ObjectTracker
        try:
            tracker.observe(grid_before, action, grid_after)
        except Exception:
            errors += 1

        # Check if level completed
        if obs.levels_completed > levels_before:
            break

    # Collect stats
    stats = tracker.perception_gap_stats()
    player_color = tracker.player_color
    action_dirs = {}
    for a in set(obs.available_actions):
        d = tracker.action_direction(a)
        if d is not None:
            action_dirs[int(a)] = [round(float(d[0]), 3), round(float(d[1]), 3)]

    return {
        "game": game_id,
        "steps": steps_taken,
        "levels_completed": int(obs.levels_completed),
        "vanished_count": stats["vanished_count"],
        "total_attempts": stats["total_attempts"],
        "vanish_rate": stats["vanish_rate"],
        "n_observations": stats["n_observations"],
        "player_color_found": player_color is not None,
        "player_color": int(player_color) if player_color is not None else None,
        "action_directions": action_dirs,
        "n_actions_found": len([a for a in obs.available_actions if tracker.action_direction(a) is not None]),
    }


def run_benchmark(games: list, max_steps: int = 200) -> dict:
    """Run measurement on multiple games and produce a report."""
    results = []
    for game_id in games:
        print(f"  📊 {game_id}...", end=" ", flush=True)
        try:
            result = measure_game(game_id, max_steps=max_steps)
            results.append(result)
            status = "✅" if result["player_color_found"] else "❌"
            print(f"{status} vanish={result['vanish_rate']:.1%} "
                  f"player={result['player_color']} "
                  f"actions={result['n_actions_found']}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "game": game_id,
                "error": str(e),
                "vanish_rate": -1,
                "player_color_found": False,
            })

    return results


def print_report(results: list, threshold: float = 0.05):
    """Print a formatted report."""
    print()
    print("=" * 65)
    print("📊 RAPPORT — Perception Gap sur jeux réels")
    print("=" * 65)
    print(f"{'Jeu':<8} {'Étapes':<7} {'Niveaux':<8} {'Disparues':<10} {'Tentées':<8} {'Taux':<8} {'Joueur':<8} {'Actions':<8}")
    print("-" * 65)

    valid = [r for r in results if r.get("vanish_rate", -1) >= 0]
    vanish_rates = []

    for r in valid:
        vr = r["vanish_rate"]
        vanish_rates.append(vr)
        pc = str(r.get("player_color", "?"))
        player_icon = "✅" if r.get("player_color_found") else "❌"
        print(f"{r['game']:<8} {r.get('steps', 0):<7} {r.get('levels_completed', 0):<8} "
              f"{r.get('vanished_count', 0):<10} {r.get('total_attempts', 0):<8} "
              f"{vr:<7.1%} {player_icon}{pc:<6} {r.get('n_actions_found', 0):<8}")

    if vanish_rates:
        print("-" * 65)
        print(f"{'MOYENNE':<8} {'':<7} {'':<8} {'':<10} {'':<8} "
              f"{sum(vanish_rates)/len(vanish_rates):<7.1%} "
              f"max={max(vanish_rates):.1%} min={min(vanish_rates):.1%}")

        max_vr = max(vanish_rates)
        decision = "GO 🟢" if max_vr >= threshold else "NO-GO 🔴"
        print()
        print(f"  Seuil : {threshold:.0%}")
        print(f"  Max vanish rate : {max_vr:.1%}")
        print(f"  Décision Phase 1.2 : {decision}")
        print()
        if max_vr >= threshold:
            print("  → Un micro-NN siamois de matching est JUSTIFIÉ.")
            print("  → Les régions disparaissent assez souvent pour que")
            print("    le matching par couleur seule ne suffise pas.")
        else:
            print("  → Pas de matching neuronal nécessaire.")
            print("  → Le problème est ailleurs (navigation, planification).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mesurer les gaps de perception")
    parser.add_argument("--games", nargs="+", default=HOLDOUT_GAMES + BASELINE_GAMES,
                        help="Jeux à tester (défaut: holdout + baseline)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Pas max par jeu")
    parser.add_argument("--all", action="store_true",
                        help="Tester tous les jeux disponibles")
    parser.add_argument("--output", type=str, default="outputs/perception_gap_report.json",
                        help="Fichier de sortie JSON")
    args = parser.parse_args()

    games = ALL_GAMES if args.all else args.games
    print(f"🧪 Mesure des gaps de perception sur {len(games)} jeux...")
    print(f"   Pas max: {args.steps} par jeu\n")

    results = run_benchmark(games, max_steps=args.steps)
    print_report(results)

    # Save
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📝 Rapport sauvegardé: {output_path}")
