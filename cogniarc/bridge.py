#!/usr/bin/env python3
"""
CogniARC Evaluator Bridge — interface entre CogniArc et le pipeline Botte.

Expose l'agent ARC-AGI-3 v4 comme un outil CLI structuré (JSON),
consommable par le MCP Gateway de Botte Secrète.

Usage:
    python -m cogniarc.bridge run ls20                     # Jouer un jeu
    python -m cogniarc.bridge run ls20 --agent v4          # Avec l'agent v4
    python -m cogniarc.bridge benchmark ls20,re86          # Benchmark multi-jeux
    python -m cogniarc.bridge benchmark all                 # Tous les jeux disponibles
    python -m cogniarc.bridge list                          # Lister les jeux
    python -m cogniarc.bridge memory                        # Voir la mémoire symbolique
    python -m cogniarc.bridge memory --reset                # Réinitialiser la mémoire

Output : JSON structuré pour intégration pipeline.

Intégration Botte :
    .mcp.json → "cogniarc-eval": "python -m cogniarc.bridge"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Ajouter le parent au path (même pattern que arc_agent.py)
_cogniarc_path = Path(__file__).resolve().parent
if str(_cogniarc_path) not in sys.path:
    sys.path.insert(0, str(_cogniarc_path))

from arc_agent import ArcAgentV4
from symbol_memory import SymbolMemory
from benchmark_tracker import BenchmarkTracker


# ── Cache des jeux disponibles ──
_KNOWN_GAMES = ["ls20", "re86"]  # À étendre


def cmd_list() -> dict:
    """Lister les jeux disponibles."""
    return {
        "status": "ok",
        "games": _KNOWN_GAMES,
        "count": len(_KNOWN_GAMES),
        "agent_version": "v4",
        "modules": [
            "temporal_inference",
            "spatial_inference",
            "attention",
            "symbolic_inference",
            "symbol_memory",
            "vision_filters",
            "arc_agent",
        ],
    }


def cmd_run(game_name: str, agent_version: str = "v4",
            max_steps: int = 300, memory_path: str = "~/.cache/cogniarc/symbol_memory.json",
            track: bool = True) -> dict:
    """Jouer un jeu avec l'agent CogniARC.

    Returns:
        Dict with result, stats, symbols used, memory state.
    """
    t0 = time.time()

    # Initialiser le tracker de benchmark
    tracker = BenchmarkTracker() if track else None
    if tracker:
        tracker.start_session(llm_model="deterministic-bfs",
                              agent_version=f"cogniarc-{agent_version}")

    # Créer et exécuter l'agent
    agent = ArcAgentV4(game_name, max_steps=max_steps, memory_path=memory_path)
    result = agent.run()

    elapsed = time.time() - t0

    # Enregistrer dans le benchmark
    if tracker:
        tracker.record_game(
            game_id=game_name,
            level=1,
            solved=result.get("plan_found", False),
            steps=result.get("steps", 0),
            time_seconds=elapsed,
            tokens_used=0,
            strategy="v4_perception_guided",
        )
        tracker.end_session()

    return {
        "status": "ok",
        "game": game_name,
        "agent_version": agent_version,
        "solved": result.get("plan_found", False),
        "steps": result.get("steps", 0),
        "elapsed_seconds": round(elapsed, 3),
        "symbols_used": result.get("symbols_used", []),
        "memory_entries": result.get("memory_entries", 0),
        "tokens_used": 0,
    }


def cmd_benchmark(games: list[str], agent_version: str = "v4",
                  max_steps: int = 300) -> dict:
    """Benchmark multi-jeux.

    Joue chaque jeu et agrège les résultats.
    """
    results = []
    total_solved = 0
    total_time = 0.0

    for game in games:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  BENCHMARK: {game}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        try:
            r = cmd_run(game, agent_version=agent_version, max_steps=max_steps)
            results.append(r)
            if r.get("solved"):
                total_solved += 1
            total_time += r.get("elapsed_seconds", 0)
        except Exception as e:
            results.append({
                "status": "error",
                "game": game,
                "error": str(e),
            })

    return {
        "status": "ok",
        "benchmark": {
            "games_count": len(games),
            "solved": total_solved,
            "solve_rate": round(total_solved / max(len(games), 1) * 100, 1),
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_game": round(total_time / max(len(games), 1), 3),
            "tokens_used": 0,
        },
        "results": results,
    }


def cmd_memory(action: str = "show") -> dict:
    """Gérer la mémoire symbolique.

    Actions:
        show    : Afficher les stats de la mémoire
        reset   : Réinitialiser la mémoire
        decay   : Appliquer la décroissance
    """
    mem = SymbolMemory()

    if action == "reset":
        # Sauvegarder une mémoire vide
        empty_mem = SymbolMemory(storage_path=str(
            Path("~/.cache/cogniarc/symbol_memory.json").expanduser()))
        empty_mem.save()
        return {"status": "ok", "action": "reset", "message": "Symbol memory reset"}

    if action == "decay":
        mem.decay()
        mem.save()

    stats = mem.stats()
    stats["status"] = "ok"
    stats["action"] = action
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="cogniarc-bridge",
        description="CogniARC Evaluator Bridge — interface pipeline Botte",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    s = sub.add_parser("run", help="Run a game")
    s.add_argument("game", help="Game name (e.g. ls20, re86)")
    s.add_argument("--agent", default="v4", choices=["v4"],
                   help="Agent version")
    s.add_argument("--max-steps", type=int, default=300)
    s.add_argument("--no-track", action="store_true",
                   help="Don't track in benchmark")

    # benchmark
    s = sub.add_parser("benchmark", help="Benchmark multiple games")
    s.add_argument("games", nargs="+", help="Game names or 'all'")
    s.add_argument("--agent", default="v4")
    s.add_argument("--max-steps", type=int, default=300)

    # list
    sub.add_parser("list", help="List available games")

    # memory
    s = sub.add_parser("memory", help="Symbol memory management")
    s.add_argument("action", nargs="?", default="show",
                   choices=["show", "reset", "decay"])

    args = p.parse_args(argv)

    try:
        if args.cmd == "list":
            result = cmd_list()
        elif args.cmd == "run":
            result = cmd_run(args.game, agent_version=args.agent,
                             max_steps=args.max_steps,
                             track=not args.no_track)
        elif args.cmd == "benchmark":
            games = _KNOWN_GAMES if "all" in args.games else args.games
            result = cmd_benchmark(games, agent_version=args.agent,
                                   max_steps=args.max_steps)
        elif args.cmd == "memory":
            result = cmd_memory(args.action)
        else:
            result = {"status": "error", "message": "Unknown command"}

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("status") == "ok" else 1

    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "message": "User interrupt"}))
        return 130
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
