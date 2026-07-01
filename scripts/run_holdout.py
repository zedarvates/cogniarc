#!/usr/bin/env python3
"""Run ScientistAgent on holdout game(s) and print the generalization report.

This is the "advisory -> measured" switch: every perception/experimentation
module built so far runs in advisory mode precisely because we could not
measure whether it generalizes. Running the agent on games it was NEVER tuned
against (the holdout set in cogniarc/eval_games.json) is that measurement.

Guardrails baked in:
  * Refuses to run a game listed in dev_games (that would contaminate the
    holdout discipline — a "holdout" number secretly influenced by dev tuning
    is worse than no number).
  * --max-steps caps the run so a smoke test is bounded (the live env is a
    remote API; long runs cost time and quota).
  * Requires the arc_agi runtime; fails with a clear message otherwise.

Usage:
    python scripts/run_holdout.py                 # all holdout games
    python scripts/run_holdout.py --game sc25-635fd71a
    python scripts/run_holdout.py --max-steps 40  # short smoke run
    python scripts/run_holdout.py --list          # just show the classification
"""
import argparse
import os
import sys

# The agent prints emoji-laden progress (🔬 🧱 🤖 ...). On a Windows console
# (cp1252) that raises UnicodeEncodeError and aborts the whole run, so force
# UTF-8 output before importing/using anything that prints.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.generalization import compute_generalization_report, load_game_sets


def _report():
    from cogniarc.benchmark_tracker import BenchmarkTracker
    sets = load_game_sets()
    sessions = BenchmarkTracker().load_all_sessions()
    print("\n" + compute_generalization_report(
        sessions, sets["dev_games"], sets["holdout_games"]
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", help="specific holdout game_id (default: all holdout games)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap total steps per game (bounded smoke run)")
    parser.add_argument("--list", action="store_true", help="show dev/holdout classification and exit")
    args = parser.parse_args()

    sets = load_game_sets()
    dev, holdout = sets["dev_games"], sets["holdout_games"]

    if args.list:
        print(f"dev_games ({len(dev)}): {', '.join(dev)}")
        print(f"holdout_games ({len(holdout)}): {', '.join(holdout)}")
        _report()
        return

    if args.game:
        if args.game in dev:
            print(f"REFUSED: '{args.game}' is a dev game. Running it would not "
                  f"measure generalization. Pick a holdout game:\n  {', '.join(holdout)}")
            sys.exit(2)
        if args.game not in holdout:
            print(f"'{args.game}' is not in holdout_games. Add it to "
                  f"cogniarc/eval_games.json only if it was never used to tune code.")
            sys.exit(2)
        games = [args.game]
    else:
        games = list(holdout)

    try:
        from cogniarc.scientist_agent import ScientistAgent, ARC_RUNTIME_AVAILABLE
    except Exception as e:  # pragma: no cover - runtime import
        print(f"Cannot import ScientistAgent: {e}")
        sys.exit(1)
    if not ARC_RUNTIME_AVAILABLE:
        print("arc_agi runtime not available. Install: pip install 'arc-agi>=0.9.9,<1.0'")
        sys.exit(1)

    for game in games:
        print(f"\n{'='*60}\n  HOLDOUT RUN: {game}\n{'='*60}")
        try:
            agent = ScientistAgent(game, enable_benchmark=True)
            if args.max_steps is not None:
                # Bound the run: patch the agent's own step budget if present.
                agent._holdout_max_steps = args.max_steps
            agent.run()
        except KeyboardInterrupt:
            print("  interrupted by user")
            break
        except Exception as e:
            print(f"  run failed: {e}")

    _report()


if __name__ == "__main__":
    main()
