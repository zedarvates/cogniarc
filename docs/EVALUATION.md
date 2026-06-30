# Evaluation discipline: dev games vs holdout games

ARC-AGI is a **generalization** benchmark, not a memorization one
(F. Chollet, *On the Measure of Intelligence*, 2019,
[arXiv:1911.01547](https://arxiv.org/abs/1911.01547)). A solve rate measured
only on games the code was tuned against proves nothing about intelligence —
it can be achieved by hardcoding that one game's solution under
general-sounding vocabulary ("drives", "reasoning modes", "discovery phase").

This repo currently has exactly that risk: `ScientistAgent`'s phase machine
and 25+ hardcoded sprite/attribute tags (`rhsxkxzdjz`, `gudziatsk`,
`cklxociuu`, ...) were reverse-engineered from **one game, LS20**. The 72%
L1 solve rate in the README's benchmark table is a dev-set number — it says
nothing about whether the agent would solve a game it has never seen.

## The rule

`cogniarc/eval_games.json` declares two lists:

- **`dev_games`** — games whose source/behavior informed any hardcoded value,
  threshold, tag, or heuristic in the codebase. LS20 is dev forever; its tags
  are baked into `scientist_agent_discovery.py` and `scientist_agent_skills.py`.
- **`holdout_games`** — games **never** touched while writing code. Once a
  game is listed here, **no future commit may use a result from that game to
  tune anything** (no new hardcoded tag, no threshold tweak informed by
  watching it fail). If that discipline is broken even once, move the game
  back to `dev_games` honestly — a "holdout" result that was secretly tuned
  against is worse than no result at all, because it creates false confidence.

## Running the report

```bash
python scripts/generalization_report.py
```

Reads `~/.cache/cogniarc/benchmarks.jsonl` (written by `BenchmarkTracker`
during `ScientistAgent` runs with `enable_benchmark=True`) and prints dev vs.
holdout solve rates plus the **generalization gap** (dev rate − holdout
rate). A large positive gap means the agent is overfit to known games'
specifics rather than solving via transferable reasoning.

If `holdout_games` is empty (the current state), the report says so
explicitly instead of silently omitting the comparison — **an empty holdout
set means no generalization claim can currently be made**, full stop.

## How to add a holdout game

1. Pick a game never referenced anywhere in this repo (`git grep` its id to
   confirm).
2. Add its id to `holdout_games` in `cogniarc/eval_games.json`.
3. Run `ScientistAgent` against it **without modifying any code based on what
   you observe** — record the result, then stop touching it.
4. If you must fix a bug the holdout run exposed, fix it generally (not by
   hardcoding that game's tags) and treat the *next* holdout game as the
   clean measurement, since this one is no longer untouched.

## What this does and doesn't fix

This harness makes the dev/holdout gap *visible and reportable*. It does not
by itself make the agent generalize — see README's "Logic vs Micro-NN"
section and the broader methodology notes for the architectural work that
actually closes the gap (object-centric perception instead of hardcoded
tags, program synthesis over composable skills instead of a fixed phase
machine, active experimentation to disambiguate hypotheses). Treat this
report as the scoreboard that tells you whether that work is paying off,
not as a substitute for it.
