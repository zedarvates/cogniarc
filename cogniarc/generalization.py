"""Generalization (dev vs holdout) reporting.

ARC-AGI is a generalization benchmark, not a memorization one (Chollet, "On
the Measure of Intelligence", arXiv:1911.01547). A solve rate measured only on
games the code was tuned against (LS20's hardcoded sprite tags, the phase
machine, escalation thresholds, ...) says nothing about whether the agent
generalizes — it could simply have memorized one game's solution dressed in
general-sounding architecture.

This module keeps that discipline explicit and queryable:
  - `cogniarc/eval_games.json` declares which game_ids are "dev" (used to
    tune code) vs "holdout" (never touched during development).
  - `compute_generalization_report()` is a pure function over BenchmarkTracker
    session data — testable with synthetic data, no live arc_agi runtime
    needed (see tests/test_generalization.py).
  - `scripts/generalization_report.py` wires it to real benchmark data.

Rule: once a game is listed under "holdout_games", no code change may use
results from that game to tune hardcoded values, thresholds, or heuristics.
If that rule is broken, move the game back to "dev_games" honestly instead of
claiming a generalization result that no longer holds.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_GAMES_CONFIG = Path(__file__).parent / "eval_games.json"


def load_game_sets(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load the dev/holdout game-id lists from eval_games.json."""
    path = path or DEFAULT_GAMES_CONFIG
    with open(path) as f:
        data = json.load(f)
    return {
        "dev_games": data.get("dev_games", []),
        "holdout_games": data.get("holdout_games", []),
    }


def _solve_rate(games) -> Optional[float]:
    """games: iterable of GameResult-like objects with `.solved`."""
    games = list(games)
    if not games:
        return None
    return sum(1 for g in games if g.solved) / len(games)


def compute_generalization_report(
    sessions, dev_games: List[str], holdout_games: List[str]
) -> str:
    """Build a markdown report comparing dev-game vs holdout-game solve rates.

    Args:
        sessions: list of SessionResult (or duck-typed equivalents with a
            `.games` list of objects exposing `.game_id` and `.solved`)
        dev_games: game_ids used to tune code during development
        holdout_games: game_ids never used to tune code

    Returns:
        Markdown report. If holdout_games is empty, says so explicitly rather
        than silently omitting the comparison — an empty holdout set means no
        generalization claim can be made yet, which is itself the finding.
    """
    by_game = defaultdict(list)
    for s in sessions:
        for g in s.games:
            by_game[g.game_id].append(g)

    untracked = sorted(set(by_game) - set(dev_games) - set(holdout_games))

    lines = ["# Generalization Report", ""]

    if not holdout_games:
        lines += [
            "⚠️ **No holdout games configured.** Every solve-rate number below "
            "(and in README benchmark tables) reflects games used to tune the "
            "code — it measures fit to known games, not generalization. Add "
            "game_ids to `holdout_games` in `cogniarc/eval_games.json` (and "
            "never tune against them) before claiming generalization.",
            "",
        ]

    dev_results = [g for gid in dev_games for g in by_game.get(gid, [])]
    holdout_results = [g for gid in holdout_games for g in by_game.get(gid, [])]

    dev_rate = _solve_rate(dev_results)
    holdout_rate = _solve_rate(holdout_results)

    lines.append("| Set | Games | Attempts | Solve rate |")
    lines.append("|-----|-------|----------|------------|")
    lines.append(
        f"| Dev (tuned against) | {len(dev_games)} | {len(dev_results)} | "
        f"{f'{dev_rate:.1%}' if dev_rate is not None else 'no data'} |"
    )
    lines.append(
        f"| Holdout (never tuned against) | {len(holdout_games)} | {len(holdout_results)} | "
        f"{f'{holdout_rate:.1%}' if holdout_rate is not None else 'no data'} |"
    )

    if dev_rate is not None and holdout_rate is not None:
        gap = dev_rate - holdout_rate
        lines += [
            "",
            f"**Generalization gap: {gap:+.1%}** (dev rate − holdout rate). "
            "A large positive gap means the agent is overfit to the dev games' "
            "specifics (hardcoded tags, tuned thresholds) rather than solving "
            "via transferable reasoning.",
        ]

    if untracked:
        lines += [
            "",
            f"ℹ️ {len(untracked)} game(s) in benchmark data are in neither list "
            f"(unclassified): {', '.join(untracked)}. Classify them in "
            "`cogniarc/eval_games.json`.",
        ]

    return "\n".join(lines)
