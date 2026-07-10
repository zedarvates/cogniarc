"""
Benchmark holdout formel — mesure la progression réelle de l'agent.

Pour chaque jeu : reset, solve_level, collect metrics.
Permet de voir l'impact des changements (GenericNavigator, Mode 10, B2 fix...).
"""

import json
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_GAMES = sorted([
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09",
    "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
    "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
])

HOLDOUT = ["sp80", "wa30", "sc25", "re86", "bp35"]


def benchmark_game(game_id: str, max_steps: int = 300) -> dict:
    """Run ScientistAgent on one game, measure results."""
    import arc_agi
    from cogniarc.scientist_agent import ScientistAgent

    start = time.time()
    result = {
        "game": game_id,
        "success": False,
        "levels_completed": 0,
        "steps_taken": 0,
        "reasoning_modes_used": [],
        "phases_used": [],
        "time_sec": 0,
        "error": None,
    }

    try:
        agent = ScientistAgent(
            game_name=game_id,
            enable_benchmark=False,
            enable_skill_tree=False,
            enable_world_model=False,
            enable_nano_llm=False,
        )
    except Exception as e:
        result["error"] = f"Init failed: {e}"
        result["time_sec"] = time.time() - start
        return result

    try:
        # Try to solve level 0
        from cogniarc.scientist_agent import ReasoningMode
        mode_history = []

        try:
            solved = agent.solve_level(level_num=None)
        except Exception as e_inner:
            result["error"] = f"solve_level error: {e_inner}"
            solved = False

        result["success"] = solved
        result["levels_completed"] = int(agent.obs.levels_completed) if hasattr(agent, 'obs') and agent.obs else 0
        result["steps_taken"] = agent.steps

        # Collect mode history
        if hasattr(agent, 'mode_manager') and agent.mode_manager:
            for entry in agent.mode_manager.mode_history:
                mode_history.append(f"{entry.get('from','?')}->{entry.get('to','?')}")
            result["reasoning_modes_used"] = mode_history[-10:]  # last 10 transitions

        # Collect phases used
        phases = []
        if hasattr(agent, '_phase'):
            phases.append(agent._phase)
        result["phases_used"] = phases[-5:]

    except Exception as e:
        result["error"] = f"Runtime error: {e}"
    finally:
        result["time_sec"] = round(time.time() - start, 2)

    return result


def run_benchmark(games: list, max_steps: int = 300) -> list:
    """Run benchmark on multiple games."""
    results = []
    total = len(games)

    for i, game_id in enumerate(games):
        print(f"  [{i+1}/{total}] {game_id}...", end=" ", flush=True)
        result = benchmark_game(game_id, max_steps=max_steps)

        status = "✅" if result["success"] else "❌"
        levels = result["levels_completed"]
        steps = result["steps_taken"]
        t = result["time_sec"]
        modes = len(result.get("reasoning_modes_used", []))

        print(f"{status} levels={levels} steps={steps} "
              f"time={t}s modes={modes}")

        if result.get("error"):
            print(f"       ⚠️ {result['error']}")

        results.append(result)

    return results


def print_report(results: list):
    """Print formatted benchmark report."""
    print()
    print("=" * 65)
    print("📊 BENCHMARK HOLDOUT — Agent Performance")
    print("=" * 65)
    print(f"{'Jeu':<8} {'Niveaux':<8} {'Steps':<8} {'Temps':<8} {'Succès':<8} {'Modes':<8}")
    print("-" * 65)

    solved = 0
    total_levels = 0
    total_steps = 0

    for r in results:
        s = "✅" if r["success"] else "❌"
        print(f"{r['game']:<8} {r.get('levels_completed', 0):<8} "
              f"{r.get('steps_taken', 0):<8} {r.get('time_sec', 0):<7.1f}s "
              f"{s:<8} {len(r.get('reasoning_modes_used', [])):<8}")
        if r["success"]:
            solved += 1
            total_levels += r.get("levels_completed", 0)
        total_steps += r.get("steps_taken", 0)

    print("-" * 65)
    wr = solved / max(1, len(results))
    print(f"{'TOTAL':<8} {'':<8} {total_steps:<8} {'':<8} "
          f"{solved}/{len(results)} ({wr:.0%})")

    # Breakdown by game category
    holdout_results = [r for r in results if r["game"] in HOLDOUT]
    known_results = [r for r in results if r["game"] not in HOLDOUT]

    if holdout_results:
        h_solved = sum(1 for r in holdout_results if r["success"])
        print(f"\n  Holdout: {h_solved}/{len(holdout_results)} solved")
    if known_results:
        k_solved = sum(1 for r in known_results if r["success"])
        print(f"  Connus:  {k_solved}/{len(known_results)} solved")

    # Error analysis
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\n  ⚠️ {len(errors)} erreurs:")
        for r in errors:
            print(f"     {r['game']}: {r['error'][:80]}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark holdout")
    parser.add_argument("--games", nargs="+", default=None,
                        help="Jeux à tester (défaut: tous)")
    parser.add_argument("--steps", type=int, default=300,
                        help="Pas max par jeu")
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUTPUT_DIR, "benchmark_report.json"),
                        help="Fichier de sortie")
    args = parser.parse_args()

    games = args.games or ALL_GAMES
    print(f"🧪 Benchmark holdout — {len(games)} jeux, max {args.steps} steps/jeu")
    print()

    results = run_benchmark(games, max_steps=args.steps)
    print_report(results)

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📝 Rapport sauvegardé: {args.output}")
