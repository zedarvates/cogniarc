#!/usr/bin/env python3
"""Print the dev vs holdout generalization report from real benchmark data.

Reads ~/.cache/cogniarc/benchmarks.jsonl (written by BenchmarkTracker during
ScientistAgent runs) and cogniarc/eval_games.json (dev/holdout game-id lists),
then prints compute_generalization_report()'s markdown output.

Usage:
    python scripts/generalization_report.py
"""
import os
import sys

# The report contains non-cp1252 characters (e.g. the U+2212 minus sign in
# "dev rate − holdout rate") — force UTF-8 so a Windows console doesn't crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.benchmark_tracker import BenchmarkTracker
from cogniarc.generalization import compute_generalization_report, load_game_sets


def main():
    tracker = BenchmarkTracker()
    sessions = tracker.load_all_sessions()
    game_sets = load_game_sets()

    if not sessions:
        print("No benchmark data found at ~/.cache/cogniarc/benchmarks.jsonl yet.")
        print("Run a ScientistAgent session with enable_benchmark=True first.")
        return

    print(compute_generalization_report(
        sessions, game_sets["dev_games"], game_sets["holdout_games"]
    ))


if __name__ == "__main__":
    main()
