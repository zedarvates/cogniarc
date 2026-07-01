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

If `holdout_games` is empty, the report says so explicitly instead of
silently omitting the comparison — an empty holdout set means no
generalization claim can be made, full stop. (As of 2026-07-01 this is no
longer the case: `arc_agi`'s API exposes 25 real environments, classified by
git-grepping each game id across the repo — 15 dev / 10 pristine holdout. See
"First measurements" below.)

## First measurements (2026-07-01)

```
python scripts/run_holdout.py --game sc25-635fd71a --max-steps 30       # holdout
python scripts/run_holdout.py --game ls20-9607627b --allow-dev --max-steps 60  # dev baseline
```

| Set | Attempts | Solve rate |
|-----|----------|------------|
| Dev (LS20, 60-step cap) | 2 | 0.0% |
| Holdout (SC25, 30-step cap) | 5 | 0.0% |

**Read this carefully — the 0.0%/0.0% gap is NOT evidence the agent
generalizes.** Both numbers are floor effects of a step cap too low to solve
even LS20: the README's own benchmark table shows `ScientistAgent (v3.2)` at
0% on LS20 L1 even *without* an artificial cap (the maze-navigation phase
machine gets stuck; BFS is the only solver that reaches 72%). A meaningful
dev-vs-holdout comparison needs step budgets long enough for the dev game to
show its *actual* (non-zero, tag-assisted) behavior.

### Follow-up run at a realistic budget (200 steps, matching `solve_level()`'s
own `max_iterations`) — no longer a floor effect

```
python scripts/run_holdout.py --game ls20-9607627b --allow-dev --max-steps 200  # dev
python scripts/run_holdout.py --game sc25-635fd71a --max-steps 200             # holdout
python scripts/run_holdout.py --game wa30-ee6fef47  --max-steps 200             # holdout
```

| Set | Steps taken | Attempts | Solve rate |
|-----|-------------|----------|------------|
| Dev (LS20) | 159 (died — trapped, game over, did not hit the cap) | 0/0 levels | 0.0% |
| Holdout (SC25) | 215 | 0/6 levels | 0.0% |
| Holdout (WA30) | 213 | 0/9 levels | 0.0% |

This time the dev run *ended on its own* (game over from being trapped) well
under the 200-step cap — so 0% here is the agent's **actual** behavior on its
own tuned game, not an artifact. It matches the README's own documented
`ScientistAgent v3.2 = 0% on LS20 L1` exactly. **0% dev, 0% holdout is
therefore a real (if unflattering) data point**: the phase machine is not
merely failing to generalize past LS20 — with `enable_world_model`/
`enable_nano_llm` off (this harness's defaults) it does not solve LS20 either;
only the separate BFS solver does. There is no gap to report yet because there
is no dev success to compare a holdout failure against.

**Root cause surfaced by both holdout runs**: `discover_available_actions()`
reported `movement: []` on *both* SC25 and WA30 — no action was ever detected
as moving the player. Tracing why: `ScientistAgent.__init__` finds `self.player`
by probing a hardcoded attribute-name list —
`['gudziatsk', 'player', 'agent', '_player', '_agent']` — where `gudziatsk` is
LS20's own obfuscated attribute name (`scientist_agent.py:282`, reused at
`:539-540` for level-transition refresh). If a game's internal object doesn't
expose the player under one of those five names, `self.player` stays `None`
for the entire run, and every `moved` check in `discover_available_actions()`
silently degrades to `False` since it's gated on `if self.player`. This is a
second hardcode, deeper than the sprite tags this doc already tracked, and it
plausibly explained why both new games looked "actionless" rather than merely
"differently laid out."

### Fix verified live (2026-07-01)

`ObjectTracker` gained `last_step_player_moved` (generic movement evidence,
set by `observe()`) and `current_position()` (player `(row, col)` derived from
`player_color`, no attribute name). `discover_available_actions()` now falls
back to `object_tracker.last_step_player_moved` whenever `self.player` is
`None`, instead of silently reporting `moved=False` forever.

Before the fix, SC25 scouted as `movement: []`. After the fix, the *same*
game, same scout logic, reports `movement: [2, 3, 4]`:

```
python scripts/run_holdout.py --game sc25-635fd71a --max-steps 20
# discovery.action_types: {'movement': [2, 3, 4], 'interaction': [], 'blocked': [1]}
```

This is a real, live-verified fix, not a hypothesis: the agent can now see
that SC25 has movement actions at all, which it previously could not detect
on any game whose player object isn't LS20's `gudziatsk`. It does not, by
itself, make the phase machine solve SC25 (`navigate_to_changer` etc. are
still LS20-specific concepts SC25 has no equivalent of) — that remains open
work, tracked separately.

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

**Status of the object-centric-perception item**: `cogniarc/object_perception.py`
(`ObjectTracker`) is a first concrete step — it infers player identity,
per-action direction, and wall colors purely from grid+action correlation,
with zero tags/source-reading/hardcoded direction mapping. It currently only
*reinforces* `_detect_wall_colors()`'s tag-based result for LS20 (where
source is available, so behavior there is unchanged); it has not yet been
used as the *sole* discovery path for a game with no tags at all, because
doing so without a holdout game to validate against would just be a new
unverified guess. That validation is exactly what an added holdout game
would give you — see "How to add a holdout game" above.

**Status of the active-experimentation item**: `cogniarc/active_experiment.py`
implements the disambiguate-then-observe loop — given competing hypotheses
that predict discrete outcomes per action, it picks the action with the
highest information gain (entropy of the predicted-outcome distribution),
then refutes whichever hypothesis mispredicts the observed outcome. The
worked example (`build_wall_floor_experiment`) turns "is colour X a wall or
floor?" into "move toward a cell of colour X; blocked => wall, moved =>
floor", using `ObjectTracker`'s *learned* action directions. It is wired into
`solve_level()` in advisory mode only (`suggest_wall_experiment()` records the
recommended experiment; it does not yet force the action) — same
observe-before-override discipline: letting it actually pick the agent's next
move should be gated on a holdout game showing it helps rather than hurts.

**Status of the program-synthesis item**: `cogniarc/program_synthesis.py`
does breadth-first search over a small grid-transformation DSL (D4 symmetries
+ tiling) for the shortest composition mapping all input->output example
pairs, then verifies the found program on a held-out test pair (ARC's
train/test split in miniature — see `scripts/demo_program_synthesis.py`).
Plus a standalone `infer_color_map` for the pure-recolouring rule family.
This is a tested library primitive for the transform domain; it is *not* wired
into the interactive ScientistAgent loop, because applying it there needs the
live game (to get input->output pairs from action outcomes) and a holdout
game to confirm the searched rule transfers rather than fits one game's
quirks. The DSL is intentionally tiny; growing it (translate, crop, flood-
fill, object-move, ...) is the obvious next step once there is a holdout game
to measure whether a bigger search space helps or just overfits.
